"""Project-specific shell around the vendored SAM-Audio DiT.

Adds the input projection, channel layout and query conditioning; `vendor/`
stays a verbatim copy of upstream.

Channel layout (per latent frame):

    in  = [ noisy target (128) | noisy complement (128) | mixture latent (128) ]
    out = [ v_target (128)     | v_complement (128) ]

The complement is generated jointly with the target, not obtained by
subtraction (see the paper).
"""

import torch
from torch import nn

from .vendor import DiT, ProjectionLayer, TimestepEmbedder, TransformerConfig


class HTimestepEmbedder(TimestepEmbedder):
    """Timestep embedder extended with the MeanFlow interval width `h = t - r`.

    `time` is `[B]` (no h) or `[B, 2]` = (t, h), packed by
    `SamAudioDiT._time_arg`. The `h` branch is zero-initialised, so it is a
    no-op until trained (see the paper).
    """

    def __init__(self, dim, frequency_embedding_dim, non_linearity, dropout,
                 fc_bias, max_period: int = 10000, zero_init: bool = True):
        super().__init__(dim, frequency_embedding_dim, non_linearity, dropout,
                         fc_bias, max_period)
        self.h_projection = ProjectionLayer(
            in_dim=frequency_embedding_dim,
            out_dim=dim,
            non_linearity=non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
        )
        if zero_init:
            nn.init.zeros_(self.h_projection.w2.weight)
            if self.h_projection.w2.bias is not None:
                nn.init.zeros_(self.h_projection.w2.bias)

    def forward(self, t):
        """`t`: [B] (no h), or [B, 2] holding (t, h)."""
        h = None
        if t.ndim == 2:
            t, h = t[:, 0], t[:, 1]
        emb = self.projection(self.timestep_embedding(t, self.frequency_embedding_size))
        if h is not None:
            emb = emb + self.h_projection(
                self.timestep_embedding(h, self.frequency_embedding_size))
        return emb


class SamAudioDiT(nn.Module):
    """Query-conditioned DiT backbone; all size knobs are config fields.

    Conditioning is the query embedding and `t` only. Hierarchy-level
    conditioning is added by the `LevelConditionedSamAudioDiT` subclass below.
    """

    # Whether `forward` takes a `level` argument.
    accepts_level = False

    # Whether `forward` takes the MeanFlow interval width `h` (set by h_cond).
    accepts_h = False

    # Vendored transformer class; subclasses swap it for other conditioning
    # mechanisms (see modulation_backbone.py).
    transformer_cls = DiT

    # Extra width the transformer's context inlet gets on top of `cond_dim`.
    context_dim_extra = 0

    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 128,
        dim: int = 768,
        n_layers: int = 10,
        n_heads: int = 12,
        dropout: float = 0.0,
        qk_norm: bool = True,
        max_positions: int = 1024,
        init_from: str | None = None,
        flip_time: bool = False,
        h_cond: bool = False,
        **transformer_kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        # SAM-Audio's flow runs t in the opposite direction along the same
        # straight path (v_ours(x, t) = -v_theirs(x, 1-t)); flip_time applies
        # that change of variables so a SAM-Audio warm start is exact. Off by
        # default (no effect on existing configs/checkpoints).
        self.flip_time = flip_time

        cfg = TransformerConfig(
            dim=dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            qk_norm=qk_norm,
            max_positions=max_positions,
            context_dim=cond_dim + self.context_dim_extra,
            out_channels=2 * latent_dim,
            in_channels=None,  # we project below instead of using data_proj
            **transformer_kwargs,
        )
        # Kept so subclasses can build extra conditioning with the same
        # hyper-parameters as the DiT internally.
        self.transformer_cfg = cfg
        self.transformer = self.transformer_cls(cfg)

        # MeanFlow conditioning on h = t - r; swaps in HTimestepEmbedder
        # before _maybe_init_from so the warm-start report sees it as missing.
        self.h_cond = h_cond
        self.accepts_h = h_cond
        if h_cond:
            self.transformer.t_embedder = HTimestepEmbedder(
                cfg.dim,
                cfg.frequency_embedding_dim,
                non_linearity=cfg.timestep_non_linearity,
                dropout=cfg.dropout,
                fc_bias=cfg.fc_bias,
                max_period=10000,
            )
        # Channel mixing before the DiT's own x_embedder (expects dim-wide input).
        self.proj = nn.Linear(3 * latent_dim, dim)

        # Learned stand-in for "no mixture given", used on iterations where
        # conditioning is dropped for classifier-free guidance.
        self.null_mix = nn.Parameter(torch.zeros(1, 1, latent_dim))

        self._maybe_init_from(init_from)

    def _maybe_init_from(self, init_from: str | None):
        """Warm-start weights only (no optimizer/step) from a SAM-Audio ckpt.

        Done at build time, before the trainer builds its EMA copy, so the
        EMA does not start from the random init. `init_from` is a local
        `checkpoint.pt`, a HF repo id (converted via `sam_audio_init.py`), or
        an already converted state dict.
        """
        if init_from is None:
            return
        from query_conditioned_dit.sam_audio_init import convert, is_released_checkpoint, resolve
        path = resolve(init_from)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if is_released_checkpoint(sd):
            sd = convert(sd, self.state_dict())
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(
                f"{init_from} has keys this backbone does not: {list(unexpected)[:5]}. "
                f"It was converted for a different backbone configuration.")
        # Missing tensors are expected: our query projection (input is 128-d
        # PaSST-PCA, not T5), null_mix, level_proj. Printed for the run log.
        print(f"[warm start] {init_from}\n"
              f"[warm start] loaded {len(sd)} tensors "
              f"({sum(v.numel() for v in sd.values()) / 1e6:.1f} M params); "
              f"kept this model's own init for {len(missing)}: {sorted(missing)}")
        if any("h_projection" in m for m in missing):
            # Zero-initialised, so the model is bit-identical to the loaded
            # one until h_projection trains.
            print("[warm start] the h_projection tensors above are "
                  "zero-initialised.")

    def _time_arg(self, t: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        """(t, h) in our convention -> the `time` argument the vendored DiT gets.

        Applies `flip_time` and packs `h` into `time` as a second column (see
        `HTimestepEmbedder`). `h` defaults to 0 (instantaneous velocity) when
        `h_cond` is on and no `h` is given.
        """
        if self.flip_time:
            t = 1.0 - t
        if not self.h_cond:
            if h is not None:
                raise ValueError(
                    f"{type(self).__name__} was built with h_cond=False but was "
                    f"given an interval width h. Set `h_cond: true` in the "
                    f"backbone config to train/sample improved MeanFlow.")
            return t
        h = torch.zeros_like(t) if h is None else h.to(t.dtype)
        return torch.stack([t, h.expand_as(t)], dim=-1)

    def _apply_mix_dropout(self, mix: torch.Tensor,
                           drop_mix: torch.Tensor | None) -> torch.Tensor:
        """Substitute `null_mix` on the samples flagged for the uncond branch."""
        if drop_mix is not None and bool(drop_mix.any()):
            m = drop_mix.to(mix.device).view(-1, 1, 1)
            mix = torch.where(m, self.null_mix.to(mix.dtype).expand_as(mix), mix)
        return mix

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mix: torch.Tensor,
                q_emb: torch.Tensor,
                drop_mix: torch.Tensor | None = None,
                h: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_t:      [B, L, 2*latent_dim]  noisy (target, complement)
        t:        [B]
        mix:      [B, L, latent_dim]    clean mixture latent
        q_emb:    [B, cond_dim]         PCA-projected PaSST query embedding
        drop_mix: [B] bool, or None. True replaces that sample's mixture with
                  `null_mix` (the unconditional branch for CFG).
        h:        [B], or None -- MeanFlow interval width; only used if the
                  backbone was built with `h_cond`.
        Returns [B, L, 2*latent_dim] velocity.
        """
        mix = self._apply_mix_dropout(mix, drop_mix)
        hx = self.proj(torch.cat([x_t, mix], dim=-1))
        out = self.transformer(hx, self._time_arg(t, h), memory=q_emb.unsqueeze(1))
        return -out if self.flip_time else out


class LevelConditionedSamAudioDiT(SamAudioDiT):
    """`SamAudioDiT` plus a binary hierarchy-level input.

    0 = `LEVEL_TRACK` (target is one raw track), 1 = `LEVEL_STEM` (target is a
    whole level-1 family, e.g. all of `drums`). Disambiguates queries that
    could mean either granularity (see the paper).

    A subclass rather than a flag so pre-existing checkpoints without
    `level_embed` stay loadable as plain `SamAudioDiT`.

    The level rides cross-attention context as a second token (rather than
    the adaLN path) to avoid touching `vendor/`.
    """

    accepts_level = True

    def __init__(self, *args, init_from: str | None = None, **kwargs):
        # init_from is deferred past super().__init__ so level_proj exists
        # before load_state_dict, keeping the missing-key report accurate.
        super().__init__(*args, init_from=None, **kwargs)
        # 2-d one-hot code for the level, through the same ProjectionLayer
        # type used for the timestep.
        self.level_proj = ProjectionLayer(
            in_dim=2,
            out_dim=self.cond_dim,
            non_linearity=self.transformer_cfg.timestep_non_linearity,
            dropout=self.transformer_cfg.dropout,
            fc_bias=self.transformer_cfg.fc_bias,
        )

        self._maybe_init_from(init_from)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mix: torch.Tensor,
                q_emb: torch.Tensor,
                drop_mix: torch.Tensor | None = None,
                level: torch.Tensor | None = None,
                h: torch.Tensor | None = None) -> torch.Tensor:
        """As `SamAudioDiT.forward`, plus:

        level: [B] long in {0, 1}, or None. None means all-`LEVEL_TRACK`, which
               is what datasets that do not yield a level imply.
        """
        mix = self._apply_mix_dropout(mix, drop_mix)
        hx = self.proj(torch.cat([x_t, mix], dim=-1))

        if level is None:
            level = torch.zeros(q_emb.shape[0], dtype=torch.long, device=q_emb.device)
        onehot = torch.nn.functional.one_hot(
            level.to(q_emb.device).long(), num_classes=2).float()
        lvl = self.level_proj(onehot).to(q_emb.dtype)
        lvl = lvl.expand(q_emb.shape[0], -1) if lvl.dim() == 1 else lvl
        memory = torch.stack([q_emb, lvl], dim=1)          # [B, 2, cond_dim]
        out = self.transformer(hx, self._time_arg(t, h), memory=memory)
        return -out if self.flip_time else out             # see `flip_time`

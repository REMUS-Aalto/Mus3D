"""Query-conditioned latent flow-matching separator (a SAM-Audio analogue).

Given a mixture and an audio query, samples the target source's DAC-VAE
latent with a flow-matching DiT, conditioned on the mixture latent and the
query embedding, and decodes it. Generative (quality bounded by the VAE) and
mono (see `codec.py`).

`t=1` is noise, `t=0` is data, `x_t = t*x0 + (1-t)*x1`, net regresses
`v = x0 - x1`. Sampling integrates t from 1 down to 0.
"""

import torch
from torch import nn

from query_conditioned_dit.stochastic_sampling import edm_churn_step, prior_churn_latent


class LatentFlowSeparator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        query_model: nn.Module,
        query_adapter: nn.Module,
        codec: nn.Module,
        latent_scale: float = 1.0,
        query_scale: float = 1.0,
        t_min: float = 1e-4,
        t_max: float = 1.0,
        p_mix_drop: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.query_model = query_model
        self.query_adapter = query_adapter
        self.codec = codec
        self.latent_scale = latent_scale
        # RMS of the query embedding distribution, divided out in `forward`
        # (the backbone's query projection expects unit-scale input). 1.0 is
        # right for PaSST-PCA, whose embeddings have RMS ~2.5.
        self.query_scale = query_scale
        self.t_min = t_min
        self.t_max = t_max
        # P(mixture replaced by backbone.null_mix), per sample per step;
        # trains the unconditional branch for classifier-free guidance.
        self.p_mix_drop = p_mix_drop
        # One-shot latch for the "level given but unusable" warning below.
        self._warned_level_ignored = False

        # PaSST and the DAC-VAE are frozen feature extractors, never trained.
        for m in (self.query_model, self.codec):
            m.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep the frozen encoders in eval regardless of the trainer.

        PaSST's mel front-end applies SpecAugment in training mode, which would
        make the same query audio produce a different embedding every step.
        """
        super().train(mode)
        self.query_model.eval()
        self.codec.eval()
        return self

    # ------------------------------------------------------------------ pieces

    @torch.no_grad()
    def embed_query(self, q: torch.Tensor) -> torch.Tensor:
        """[B, C, T] query audio -> [B, D] PCA-projected PaSST embedding."""
        return self.query_adapter(self.query_model(q)).float()

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, C, T] audio -> [B, L, 128] scaled latent."""
        return self.codec.encode(wav).float() / self.latent_scale

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, L, 128] scaled latent -> [B, 1, T] audio."""
        return self.codec.decode(z.float() * self.latent_scale)

    @property
    def conditions_on_level(self) -> bool:
        """Whether the backbone consumes the hierarchy level."""
        return getattr(self.backbone, "accepts_level", False)

    @property
    def conditions_on_h(self) -> bool:
        """Whether the backbone takes the MeanFlow interval width `h`."""
        return getattr(self.backbone, "accepts_h", False)

    def forward(self, x_t, t, mix_lat, q_emb, drop_mix=None, level=None, h=None):
        if self.query_scale != 1.0:
            q_emb = q_emb / self.query_scale
        if h is not None and not self.conditions_on_h:
            raise ValueError(
                f"{type(self.backbone).__name__} was built without `h_cond`, so "
                f"it cannot represent the average velocity u(x, t, h). Use an "
                f"iMF backbone config (h_cond: true).")
        # Only passed when the backbone accepts it, so a plain SamAudioDiT is
        # called with exactly the signature it always had.
        extra = {"h": h} if self.conditions_on_h else {}
        if self.conditions_on_level:
            return self.backbone(x_t, t, mix_lat, q_emb, drop_mix, level,
                                 **extra).float()
        # A level-yielding dataset paired with a level-blind backbone drops
        # the tag rather than failing; warned once.
        if level is not None and not self._warned_level_ignored:
            self._warned_level_ignored = True
            print(f"[warn] {type(self.backbone).__name__} does not take a "
                  f"hierarchy level; the dataset's level tag is being IGNORED. "
                  f"Use LevelConditionedSamAudioDiT (LFM_DiT_500M_level.yaml) "
                  f"to condition on it.")
        return self.backbone(x_t, t, mix_lat, q_emb, drop_mix, **extra).float()

    def _encode_triplet(self, stems: torch.Tensor):
        """[B, N, C, T] stems -> (mixture, target, complement) latents.

        Stem 0 is the target, the rest sum to the complement. All three go
        through the codec in one batched call.
        """
        b = stems.shape[0]
        target = stems[:, 0]
        complement = stems[:, 1:].sum(dim=1)
        mix = target + complement
        z = self.encode(torch.cat([mix, target, complement], dim=0))
        return z[:b], z[b : 2 * b], z[2 * b :]

    # ---------------------------------------------------------------- training

    def train_step(self, stems: torch.Tensor, q: torch.Tensor, t=None,
                   debug: bool = False, level=None, q_emb=None,
                   *args, **kwargs):
        """
        stems: [B, N_src, C, T] -- stem 0 is the target
        q:     [B, C, T]        -- query audio (the trainer passes stem 0)
        level: [B] long in {0, 1}, or None -- hierarchy level (None = track)
        q_emb: [B, D] or None -- pre-computed query embedding, used instead of
               embedding `q` (lets callers reuse an embedding or inject jitter
               in embedding space). `q` is still required even when given.
        """
        b = stems.shape[0]
        mix_lat, z_tgt, z_comp = self._encode_triplet(stems)
        q_emb = self.embed_query(q) if q_emb is None else q_emb.float()

        x1 = torch.cat([z_tgt, z_comp], dim=-1)            # [B, L, 256] data
        x0 = torch.randn_like(x1)                          # noise
        if t is None:
            t = torch.rand(b, device=x1.device) * (self.t_max - self.t_min) + self.t_min
        t = t.to(x1.device)

        t3 = t.view(b, 1, 1)
        x_t = t3 * x0 + (1.0 - t3) * x1
        v = x0 - x1

        # Conditioning dropout, per sample per step (see `p_mix_drop`).
        drop_mix = (torch.rand(b, device=x1.device) < self.p_mix_drop
                    if self.p_mix_drop > 0.0 else None)

        pred = self.forward(x_t, t, mix_lat, q_emb, drop_mix, level)
        loss = (pred - v).pow(2).mean()

        d = self.backbone.latent_dim
        with torch.no_grad():
            err = (pred - v).pow(2)
            loss_unnorm = err.mean(dim=(1, 2))                             # [B]
            loss_linear = loss_unnorm / (v.pow(2).mean(dim=(1, 2)) + 1e-8)
            metrics = {
                # Separate target/complement losses, to see if either head stalls.
                "loss_target": err[..., :d].mean(),
                "loss_complement": err[..., d:].mean(),
                "loss_linear": loss_linear.mean(),
                "loss_unnorm": loss_unnorm.mean(),
                # Latent RMS sanity check: should sit near 1 if latent_scale is
                # the measured std.
                "latent_rms": x1.pow(2).mean().sqrt(),
                # Should sit at p_mix_drop; 0 means CFG's unconditional branch
                # is not being trained.
                "mix_drop_frac": (drop_mix.float().mean() if drop_mix is not None
                                  else torch.zeros((), device=x1.device)),
                # Fraction of the batch at LEVEL_STEM (0 if no level given).
                "stem_frac": (level.float().mean() if level is not None
                              else torch.zeros((), device=x1.device)),
            }

        return {
            "loss": loss,
            **metrics,
            # Consumed by Trainer._log_loss_per_t.
            "t": t,
            "loss_per_sample": 10 * torch.clamp(loss_linear, min=1e-10).log10(),
            "loss_linear_per_sample": loss_linear,
            "loss_unnorm_per_sample": loss_unnorm,
        }

    # ---------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample_latent(self, mix_lat, q_emb, n_step: int = 32, cfg_scale: float = 1.0,
                      level=None, churn=0.0, prior_churn=0.0, generator=None):
        """-> ([B, L, 128] target latent, [B, L, 128] complement latent).

        `cfg_scale` > 1 turns on classifier-free guidance: each step is
        evaluated with and without the mixture and extrapolated
        `v_uncond + w * (v_cond - v_uncond)`. Only meaningful if the model was
        trained with `p_mix_drop > 0`. 1.0 is plain sampling.

        `churn` (gamma >= 0) makes the sampler stochastic; `prior_churn`
        perturbs the starting latent (EDM-style, shared with
        `model.stochastic_sampling`). Both default to 0 (deterministic).
        """
        b, length, d = mix_lat.shape
        x = torch.randn(b, length, 2 * d, device=mix_lat.device,
                        dtype=torch.float32, generator=generator)
        x = prior_churn_latent(x, prior_churn, generator=generator)
        ts = torch.linspace(1.0, 0.0, n_step + 1, device=mix_lat.device)
        drop = (torch.ones(b, dtype=torch.bool, device=mix_lat.device)
                if cfg_scale != 1.0 else None)
        for i in range(n_step):
            tt = ts[i].expand(b)
            v = self.forward(x, tt, mix_lat, q_emb, None, level)
            if drop is not None:
                v_unc = self.forward(x, tt, mix_lat, q_emb, drop, level)
                v = v_unc + cfg_scale * (v - v_unc)
            g = float(churn[i]) if torch.is_tensor(churn) else float(churn)
            x = edm_churn_step(x, v, ts[i], ts[i + 1], g, generator=generator)
        return x[..., :d], x[..., d:]

    @torch.no_grad()
    def sample(self, queries: torch.Tensor, mix: torch.Tensor, n_step: int = 32,
               return_latent: bool = False, cfg_scale: float = 1.0,
               level=None, churn=0.0, prior_churn=0.0, *args, **kwargs):
        """
        queries: [B, C, T] query audio
        mix:     [B, C, T] mixture
        level:   [B] long in {0, 1}, or None -- requested granularity: raw
                 track (0) or whole level-1 family (1).
        Returns ([B, 1, T] target, [B, 1, T] complement) audio, mono.

        With `return_latent`, also returns `(tgt, comp, z_tgt, z_comp)`, the
        latents the audio was decoded from -- useful for scoring the DiT
        alone against `encode(ground truth)`, excluding codec error.
        """
        z_tgt, z_comp = self.sample_latent(
            self.encode(mix), self.embed_query(queries), n_step=n_step,
            cfg_scale=cfg_scale, level=level, churn=churn,
            prior_churn=prior_churn,
        )
        tgt, comp = self.decode(z_tgt), self.decode(z_comp)
        if return_latent:
            return tgt, comp, z_tgt, z_comp
        return tgt, comp

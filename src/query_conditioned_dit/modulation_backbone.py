"""Query conditioning by feature modulation (adaLN) instead of cross-attention.

`t` conditions through adaLN (per-block shift/scale/gate). This file
conditions the query the same way, via its own embedder and `q_block`, and
drops cross-attention entirely. The two adaLN vectors are summed before each
block applies them. Does not touch `vendor/`, `SamAudioDiT` or
`LevelConditionedSamAudioDiT`; these are new classes with their own,
non-interchangeable state dicts.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .backbone import LevelConditionedSamAudioDiT, SamAudioDiT
from .vendor import ContextEmbedder, DiT
from .vendor.transformer import get_nonlinearity


class QueryModulatedDiT(DiT):
    """`DiT` with the context injected as adaLN modulation, not cross-attention.

    Same `forward` signature as the parent. `memory` is `[B, K, context_dim]`
    (K=1 query, or 2 with the hierarchy level); each token is embedded and
    modulated separately and the results summed. Query blocks are
    zero-initialised (adaLN-zero), so the query's influence grows from zero.
    """

    def __init__(self, config):
        super().__init__(config)

        # Remove cross-attention entirely (not just bypass it).
        for layer in self.layers:
            layer.cross_attention = None

        # y_embedder (context -> dim) is reused from the parent; what
        # follows is new.
        self.q_block_non_linearity = get_nonlinearity(config.t_block_non_linearity)
        self.q_block = nn.Linear(config.dim, config.dim * 6, bias=config.t_block_bias)
        self.q_final_block = nn.Linear(config.dim, config.dim, bias=config.t_block_bias)
        for m in (self.q_block, self.q_final_block):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _query_modulation(self, memory, memory_padding_mask):
        """[B, K, context_dim] -> (block 6D vector, final-layer D vector)."""
        y = self.y_embedder(memory)                      # [B, K, D]
        q = self.q_block(self.q_block_non_linearity(y))  # [B, K, 6D]
        qf = self.q_final_block(self.q_block_non_linearity(y))
        if memory_padding_mask is not None:
            keep = (~memory_padding_mask).to(q.dtype).unsqueeze(-1)
            q, qf = q * keep, qf * keep
        return q.sum(dim=1), qf.sum(dim=1)               # [B, 6D], [B, D]

    def forward(self, x, time, *, padding_mask=None, memory=None,
                memory_padding_mask=None):
        x = rearrange(x, "b l c-> b c l")
        h = self.x_embedder(x)
        h = rearrange(h, "b c l -> b l c")
        original_N = h.shape[1]
        h = F.dropout(h, p=self.dropout, training=self.training)

        t = self.t_embedder(time)                        # [B, D]
        t0 = self.t_block(self.t_block_non_linearity(t))  # [B, 6D]

        if memory is not None:
            q0, qf = self._query_modulation(memory, memory_padding_mask)
            t0 = t0 + q0
            t = t + qf

        for layer in self.layers:
            h = layer(
                x=h,
                cross_x=None,       # cross-attention removed above
                t=t0,
                padding_mask=padding_mask,
                memory_padding_mask=None,
                rope=self.rope_embeddings,
            )

        shift, scale = (self.final_layer_scale_shift_table[None] + t[:, None]).chunk(
            2, dim=1
        )
        if self.norm is not None:
            h = self.norm(h)
        h = h * (1 + scale) + shift
        h = F.dropout(h, p=self.dropout, training=self.training)
        output = self.output(h)
        if original_N != output.shape[1]:
            output = output[:, -original_N:]
        return output


class QueryModulatedSamAudioDiT(SamAudioDiT):
    """`SamAudioDiT` with the query conditioning the adaLN path.

    Identical interface: same `forward` signature, same channel layout, same
    `null_mix` mixture dropout, same `flip_time`; only the internal
    conditioning mechanism differs.

    `init_from` accepts the released SAM-Audio checkpoint; its per-block
    `cross_attention.*` weights are dropped on conversion.
    """

    transformer_cls = QueryModulatedDiT


class QueryModulatedLevelConditionedSamAudioDiT(LevelConditionedSamAudioDiT):
    """As above, with the hierarchy level as a second, independent modulation.

    The level rides `memory` as before, now as adaLN context instead of
    cross-attention.
    """

    transformer_cls = QueryModulatedDiT


class QueryLevelModulatedSamAudioDiT(SamAudioDiT):
    """Query and level concatenated into one conditioning vector.

    context = [ q_emb (cond_dim) | level one-hot (2) ] -> y_embedder (SwiGLU),
    letting query and level interact rather than only summing (compare
    `QueryModulatedLevelConditionedSamAudioDiT`). Subclasses `SamAudioDiT`
    directly since it has no use for a separate `level_proj`.
    """

    accepts_level = True
    transformer_cls = QueryModulatedDiT
    context_dim_extra = 2  # level one-hot rides in the same context vector

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mix: torch.Tensor,
                q_emb: torch.Tensor,
                drop_mix: torch.Tensor | None = None,
                level: torch.Tensor | None = None,
                h: torch.Tensor | None = None) -> torch.Tensor:
        """As `SamAudioDiT.forward`, plus `level`: [B] long in {0, 1}, or None
        (defaults to all-`LEVEL_TRACK`)."""
        mix = self._apply_mix_dropout(mix, drop_mix)
        hx = self.proj(torch.cat([x_t, mix], dim=-1))

        if level is None:
            level = torch.zeros(q_emb.shape[0], dtype=torch.long, device=q_emb.device)
        onehot = F.one_hot(level.to(q_emb.device).long(), num_classes=2).to(q_emb.dtype)
        cond = torch.cat([q_emb, onehot], dim=-1).unsqueeze(1)  # [B, 1, cond_dim+2]

        out = self.transformer(hx, self._time_arg(t, h), memory=cond)
        return -out if self.flip_time else out  # see `flip_time` in `SamAudioDiT`

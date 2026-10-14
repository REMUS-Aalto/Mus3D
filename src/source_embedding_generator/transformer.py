"""Transformer blocks of the SEG DiT (`recursive_dit.RecursiveSetDiT`).

A stack of pre-norm blocks, each with
  * self-attention over the slot chunks (padded chunks are masked out),
  * cross-attention to the conditioning tokens,
  * a SwiGLU feed-forward,
where the self-attention and feed-forward branches are modulated by the global
conditioning vector (adaLN: scale, shift and a sigmoid gate per branch). Branch
output projections start at zero, so every block starts as the identity.

Parameter names are fixed by the trained checkpoints and must not change.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class FourierFeatures(nn.Module):
    """Random Fourier features of a scalar: [cos(2 pi W t), sin(2 pi W t)]."""

    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features) * std)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * t @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ScaledSinusoidalEmbedding(nn.Module):
    """Sinusoidal position embedding of the sequence axis, with a learned scale."""

    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)
        half = dim // 2
        self.register_buffer("inv_freq", theta ** -(torch.arange(half).float() / half),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device)
        ang = pos[:, None] * self.inv_freq[None, :]
        return torch.cat((ang.sin(), ang.cos()), dim=-1) * self.scale


class LayerNorm(nn.Module):
    """LayerNorm without a learned bias (`beta` is a fixed zero buffer)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        out = nn.Linear(inner, dim)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        # indices 1 and 3 are parameter-free placeholders kept for the checkpoint layout
        self.ff = nn.Sequential(SwiGLU(dim, inner), nn.Identity(), out, nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Attention(nn.Module):
    """Multi-head attention; self-attention if `dim_context` is None."""

    def __init__(self, dim: int, dim_heads: int, dim_context: int | None = None):
        super().__init__()
        assert dim % dim_heads == 0
        self.heads = dim // dim_heads
        if dim_context is None:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        else:
            assert dim_context % dim_heads == 0
            self.kv_heads = dim_context // dim_heads
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_context, dim_context * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)

    def _split(self, t: torch.Tensor, heads: int) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, heads, -1).transpose(1, 2)

    def forward(self, x, context=None, mask=None):
        """`mask` [B, N] (self-attention only): True for tokens that take part."""
        if context is None:
            q, k, v = (self._split(t, self.heads) for t in self.to_qkv(x).chunk(3, dim=-1))
        else:
            q = self._split(self.to_q(x), self.heads)
            k, v = (self._split(t, self.kv_heads) for t in self.to_kv(context).chunk(2, dim=-1))
            if self.kv_heads != self.heads:
                r = self.heads // self.kv_heads
                k, v = k.repeat_interleave(r, dim=1), v.repeat_interleave(r, dim=1)
        attn_mask = None
        if mask is not None and context is None:
            b, h, n, _ = q.shape
            attn_mask = mask[:, None, None, :].expand(b, h, n, k.shape[-2])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = self.to_out(out.transpose(1, 2).flatten(2))
        if mask is not None and context is None:
            out = out.masked_fill(~mask[..., None], 0.0)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, dim_heads: int, dim_context: int, global_cond_dim: int):
        super().__init__()
        self.pre_norm = LayerNorm(dim)
        self.self_attn = Attention(dim, dim_heads)
        self.cross_attend_norm = LayerNorm(dim)
        self.cross_attn = Attention(dim, dim_heads, dim_context=dim_context)
        self.ff_norm = LayerNorm(dim)
        self.ff = FeedForward(dim)
        self.global_cond_dim = global_cond_dim
        self.to_scale_shift_gate = nn.Parameter(torch.randn(6 * dim) / dim ** 0.5)

    def forward(self, x, context, global_cond, mask=None):
        s_sa, b_sa, g_sa, s_ff, b_ff, g_ff = (
            (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1))
        h = self.pre_norm(x) * (1 + s_sa) + b_sa
        x = x + self.self_attn(h, mask=mask) * torch.sigmoid(1 - g_sa)
        x = x + self.cross_attn(self.cross_attend_norm(x), context=context)
        h = self.ff_norm(x) * (1 + s_ff) + b_ff
        return x + self.ff(h) * torch.sigmoid(1 - g_ff)


class ContinuousTransformer(nn.Module):
    """`dim_in` -> `depth` blocks of width `dim` -> `dim_out`.

    `num_heads` is used as a divisor: each head has `dim // num_heads` channels.
    """

    def __init__(self, dim: int, depth: int, *, dim_in: int, dim_out: int, num_heads: int,
                 cross_attend: bool = True, cond_token_dim: int, global_cond_dim: int):
        super().__init__()
        assert cross_attend, "the SEG DiT always cross-attends to its conditioning"
        self.project_in = nn.Linear(dim_in, dim, bias=False)
        self.project_out = nn.Linear(dim, dim_out, bias=False)
        self.global_cond_embedder = nn.Sequential(
            nn.Linear(global_cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        self.layers = nn.ModuleList(
            TransformerBlock(dim, dim // num_heads, cond_token_dim, global_cond_dim)
            for _ in range(depth))

    def forward(self, x, mask=None, global_cond=None, context=None, context_mask=None):
        x = self.project_in(x.to(next(self.parameters()).dtype))
        cond = self.global_cond_embedder(global_cond)
        for layer in self.layers:
            if torch.is_grad_enabled():
                x = checkpoint(layer, x, context, cond, mask, use_reentrant=False)
            else:
                x = layer(x, context, cond, mask)
        return self.project_out(x)

"""The DiT for recursive extraction: one source at a time, told what is taken.

Two slots per item (next source, residual), generated as an unordered,
equivariant set: no slot index, only a fresh random per-slot tag and a
role flag marking the residual slot. Everything else already extracted is
supplied as cross-attention context (mixture latent, extracted-unit
embeddings, previous residual, mixture PaSST embedding). See the paper
for the design rationale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from source_embedding_generator.transformer import (
    ContinuousTransformer,
    FourierFeatures,
    ScaledSinusoidalEmbedding,
)


class RecursiveSetDiT(nn.Module):
    def __init__(
        self,
        io_channels: int = 64,
        embed_dim: int = 512,
        depth: int = 16,
        num_heads: int = 16,          # dim_heads = embed_dim // num_heads
        embedding_dim: int = 768,
        max_groups: int = 16,         # sizes the `n` and `n_avail` embedding tables
        cond_channels: int = 256,
        chunk_pos_dim: int = 48,      # one-hot over chunk index
        group_tag_dim: int = 16,      # anonymous random per-slot binding tag
        cfg_dropout_prob: float = 0.1,
        timestep_features_dim: int = 256,
        mixture_emb_dim: int | None = None,   # PaSST/PCA embedding of the mixture
        avail_dim: int | None = None,         # descriptor of an extracted unit
        res_emb_dim: int | None = None,       # the previous residual's embedding
        res_mix_dim: None = None,             # removed variant; kept for old configs
        **transformer_kwargs,
    ):
        """The three optional widths are the optional conditioning streams.

        `avail_dim` is the full target layout (`D_pca`): a unit is described to
        the model by the same kind of vector the model would have to generate
        for it. `res_emb_dim` is again `D_pca`. Each None leaves the model
        byte-identical to the version without that feature.
        """
        super().__init__()
        if res_mix_dim is not None:
            raise ValueError("res_mix_dim (the reduced-mixture SEG variant) is no "
                             "longer supported")
        if embedding_dim % io_channels != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by io_channels ({io_channels})"
            )
        self.n_chunks = embedding_dim // io_channels
        if chunk_pos_dim < self.n_chunks:
            raise ValueError(f"chunk_pos_dim {chunk_pos_dim} < {self.n_chunks} chunks")

        self.io_channels = io_channels
        self.embedding_dim = embedding_dim
        self.embed_dim = embed_dim
        self.max_groups = max_groups
        self.chunk_pos_dim = chunk_pos_dim
        self.group_tag_dim = group_tag_dim
        self.cfg_dropout_prob = cfg_dropout_prob

        # ---- conditioning: timestep + slot count + level, into adaLN ----
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        # Number of generated slots; always 2 in the recursion, so this path
        # carries no information, but kept for checkpoint compatibility.
        self.n_embed = nn.Embedding(max_groups + 1, embed_dim)
        # Hierarchy level (0 = one slot per raw track, 1 = one slot per
        # level-1 family). A property of the item, so it rides adaLN with `t`.
        self.level_embed = nn.Embedding(2, embed_dim)
        # How many units are already extracted, as an explicit embedding
        # (attention over the context averages it away otherwise).
        self.n_avail_embed = nn.Embedding(max_groups + 1, embed_dim)

        # ---- cross-attention context: the mixture latent ----
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_channels, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.cond_pos_emb = ScaledSinusoidalEmbedding(embed_dim)
        self.null_cond = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ---- the mixture's own PaSST embedding, as one extra context token ----
        self.mixture_emb_dim = mixture_emb_dim
        if mixture_emb_dim is not None:
            self.to_mixture_embed = nn.Sequential(
                nn.Linear(mixture_emb_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        # ---- the already-extracted units, one token each ----
        self.avail_dim = avail_dim
        if avail_dim is not None:
            self.to_avail_embed = nn.Sequential(
                nn.Linear(avail_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )
            # Marks a context token as "an extracted unit" vs. a mixture frame.
            self.avail_type = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ---- the previous residual, as the embedding the model generated ----
        self.res_emb_dim = res_emb_dim
        if res_emb_dim is not None:
            self.to_res_emb = nn.Sequential(
                nn.Linear(res_emb_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )
            self.res_emb_type = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.transformer = ContinuousTransformer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            # +1 for the residual role flag
            dim_in=io_channels + chunk_pos_dim + group_tag_dim + 1,
            dim_out=io_channels,
            cross_attend=True,
            cond_token_dim=embed_dim,
            global_cond_dim=embed_dim,
            **transformer_kwargs,
        )

        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)


    # ------------------------------------------------------------------ utils

    def chunk_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        return rearrange(e, "b n (t c) -> b n c t", c=self.io_channels)

    def unchunk_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b n c t -> b n (t c)")

    def sample_group_tags(self, b: int, n: int, device, dtype) -> torch.Tensor:
        """[B, N, tag_dim] fresh anonymous tags -- orthonormal, i.i.d. per slot.

        Redrawn every forward so no tag value can come to mean a particular
        kind of source. Orthonormalised via QR (exact, not just near-
        orthogonal) for `n <= tag_dim`; falls back to a normalised Gaussian
        draw otherwise.
        """
        tags = torch.randn(b, n, self.group_tag_dim, device=device, dtype=torch.float32)
        k = min(n, self.group_tag_dim)
        q, _ = torch.linalg.qr(tags[:, :k].transpose(1, 2))      # [b, dim, k]
        tags[:, :k] = q.transpose(1, 2)
        return F.normalize(tags, dim=-1).to(dtype)

    # -------------------------------------------------------------- the context

    def _context(self, cond, mixture_emb=None, avail=None, avail_mask=None,
                 res_emb=None):
        """The token groups, concatenated. -> [B, T_ctx, embed_dim].

        Order: mixture summary and frames, previous residual, one token per
        extracted unit. Each group carries its own type embedding.
        """
        ctx = self.to_cond_embed(cond.transpose(1, 2))
        ctx = ctx + self.cond_pos_emb(ctx)
        if self.mixture_emb_dim is not None:
            if mixture_emb is None:
                raise ValueError(
                    "backbone was built with mixture_emb_dim, so forward() needs "
                    "a mixture_emb (the PaSST/PCA embedding of the mixture)"
                )
            # Prepended after the positional embedding: it is a summary, not
            # a frame.
            tok = self.to_mixture_embed(mixture_emb.to(ctx.dtype)).unsqueeze(1)
            ctx = torch.cat([tok, ctx], dim=1)

        if self.res_emb_dim is not None and res_emb is not None:
            tok = self.to_res_emb(res_emb.to(ctx.dtype)).unsqueeze(1)
            ctx = torch.cat([ctx, tok + self.res_emb_type.to(ctx.dtype)], dim=1)

        if self.avail_dim is not None and avail is not None and avail.shape[1] > 0:
            tok = self.to_avail_embed(avail.to(ctx.dtype)) + self.avail_type.to(ctx.dtype)
            if avail_mask is not None:
                # Zero padded entries rather than masking them from attention.
                tok = tok * avail_mask.unsqueeze(-1).to(tok.dtype)
            ctx = torch.cat([ctx, tok], dim=1)
        return ctx

    # ---------------------------------------------------------------- forward

    def _forward(self, x, t, context, mask, tags, is_residual, level=None,
                 n_avail=None):
        b, n, c, tc = x.shape

        x = rearrange(x, "b n c t -> (b n) c t")
        x = self.preprocess_conv(x) + x
        x = rearrange(x, "(b n) c t -> b n t c", b=b, n=n)

        # Chunk position (meaningful, fixed layout) plus the anonymous slot
        # tag (identifying, never meaningful).
        chunk = F.one_hot(torch.arange(tc, device=x.device), self.chunk_pos_dim)
        chunk = chunk.to(x.dtype).view(1, 1, tc, -1).expand(b, n, tc, -1)
        tag = tags.to(x.dtype).unsqueeze(2).expand(b, n, tc, self.group_tag_dim)
        # Role flag: 1 on the residual slot, 0 on the source slots.
        role = is_residual.to(x.dtype).view(b, n, 1, 1).expand(b, n, tc, 1)
        x = torch.cat([x, chunk, tag, role], dim=-1)
        x = rearrange(x, "b n t c -> b (n t) c")

        seq_mask = mask.unsqueeze(-1).expand(b, n, tc).reshape(b, n * tc)

        # Item-level conditioning, summed into one adaLN vector.
        n_groups = mask.sum(dim=1).clamp(0, self.max_groups)
        global_cond = self.to_timestep_embed(self.timestep_features(t[:, None]))
        global_cond = global_cond + self.n_embed(n_groups)
        if level is None:
            level = torch.zeros(b, dtype=torch.long, device=x.device)
        global_cond = global_cond + self.level_embed(level.to(x.device).long())
        if n_avail is not None:
            idx = n_avail.to(x.device).long().clamp(0, self.max_groups)
            global_cond = global_cond + self.n_avail_embed(idx)

        out = self.transformer(
            x, mask=seq_mask, context=context, context_mask=None, global_cond=global_cond
        )
        out = rearrange(out, "b t c -> b c t")
        out = self.postprocess_conv(out) + out
        return rearrange(out, "b c (n t) -> b n c t", n=n, t=tc)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
        group_ids=None,           # accepted and ignored: there are no slot ids
        group_tags: torch.Tensor | None = None,
        is_residual: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float | None = None,
        mixture_emb: torch.Tensor | None = None,
        level: torch.Tensor | None = None,
        avail: torch.Tensor | None = None,
        avail_mask: torch.Tensor | None = None,
        n_avail: torch.Tensor | None = None,
        res_emb: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        b, n = x.shape[0], x.shape[1]
        device = x.device
        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=device)
        if is_residual is None:
            # Loader convention: the residual occupies the final slot.
            is_residual = torch.zeros(b, n, dtype=torch.bool, device=device)
            is_residual[:, -1] = True
        if t.ndim == 0:
            t = t.expand(b)
        elif t.numel() == 1 and b > 1:
            t = t.reshape(1).expand(b)

        dtype = next(self.parameters()).dtype
        x, t = x.to(dtype), t.to(dtype)
        if group_tags is None:
            group_tags = self.sample_group_tags(b, n, device, dtype)
        if force_uncond:
            # Unconditional branch without building the real context first.
            # One token suffices: cross-attention over identical keys gives
            # the same output as N copies of `null_cond`.
            context = self.null_cond.to(dtype).expand(b, 1, -1)
            return self._forward(x, t, context, mask, group_tags, is_residual,
                                 level, n_avail)

        context = self._context(cond.to(dtype), mixture_emb, avail, avail_mask,
                                res_emb)
        null = self.null_cond.to(dtype).expand_as(context)

        if cfg_scale != 1.0:
            lvl2 = None if level is None else torch.cat([level, level])
            na2 = None if n_avail is None else torch.cat([n_avail, n_avail])
            out = self._forward(
                torch.cat([x, x]), torch.cat([t, t]), torch.cat([context, null]),
                torch.cat([mask, mask]), torch.cat([group_tags, group_tags]),
                torch.cat([is_residual, is_residual]), lvl2, na2,
            )
            cond_out, uncond_out = out.chunk(2, dim=0)
            return uncond_out + (cond_out - uncond_out) * cfg_scale

        p_drop = self.cfg_dropout_prob if cfg_dropout_prob is None else cfg_dropout_prob
        if self.training and p_drop > 0.0:
            drop = torch.bernoulli(torch.full((b, 1, 1), p_drop, device=device)).bool()
            context = torch.where(drop, null, context)
        return self._forward(x, t, context, mask, group_tags, is_residual, level,
                             n_avail)

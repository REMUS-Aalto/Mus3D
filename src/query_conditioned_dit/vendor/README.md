# Vendored SAM-Audio diffusion transformer

Source: https://github.com/facebookresearch/sam-audio
Commit: `bb4c6999d2677c7402360e426afc01ddfad6dce0` (default branch, cloned 2026-08-08)
Paper:  https://arxiv.org/abs/2512.18099
License: SAM License, see `../LICENSE` (applies to all of `query_conditioned_dit/`).

| file             | upstream path                    | change |
|------------------|----------------------------------|--------|
| `transformer.py` | `sam_audio/model/transformer.py` | verbatim |
| `rope.py`        | `sam_audio/model/rope.py`        | verbatim |
| `patcher.py`     | `sam_audio/model/patcher.py`     | verbatim |
| `config.py`      | `sam_audio/model/config.py`      | only `TransformerConfig` kept — the rest configures codec / T5 / vision encoders we do not use, and importing it would drag in `transformers` and `perception_models` |

Do not edit these files. Everything project-specific (input projection, the
target/complement channel layout, conditioning) lives one level up in
`../backbone.py`, so a future re-vendor is a straight file copy.

## What the upstream code assumes, and what we feed it

`DiT.forward(x, time, memory=...)` expects `x` to be **already projected to
`config.dim`** — `x_embedder` is a `Patcher` whose `in_channels` is `config.dim`,
and `config.in_channels`/`self.data_proj` is dead code in upstream's own forward.
Upstream `SAMAudio` does the projection itself (`self.proj`); so do we, in
`backbone.py`.

`memory` is the cross-attention context and is **not optional** — every
`DiTBlock` has a cross-attention branch. Upstream feeds it T5 text features. We
feed a single token: the PCA-projected PaSST query embedding. That is the whole
of our conditioning, plus `time` via the AdaLN `t_block`.

Channel layout is upstream's, inferred from `SAMAudioConfig`
(`in_channels=768`, `transformer.out_channels=256`, DAC-VAE latent dim 128):
`out_channels = 2 x 128` is the **target and the complement, both generated**,
never a subtraction. We keep that and drop upstream's third input block (a
`zeros_like` placeholder for the masked-audio branch), so our input is
`3 x 128 = 384` (noisy target, noisy complement, clean mixture latent) rather
than 768.

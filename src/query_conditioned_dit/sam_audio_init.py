"""Warm start from a released SAM-Audio checkpoint (e.g. facebook/sam-audio-large).

`vendor/` is a verbatim copy of SAM-Audio's transformer, so most tensors
transfer 1:1. `convert` maps the released state dict onto our backbone:

  * `transformer.*` transfers, except `y_embedder.*` (context differs: 1536-d
    T5 text vs. our 128-d query + level) and, for the modulation backbone,
    `cross_attention.*` (it has none).
  * `proj`: upstream packs [noisy (2x128) | zeros (2x128) | mixture x2 (2x128)]
    (768-wide); we pack [noisy target | noisy complement | mixture] (384-wide).
  * With no video, upstream's `align_masked_video` constant bias term is
    folded into `proj.bias`.

Time runs the other way upstream; `SamAudioDiT(flip_time=True)` handles that
at call time. Upstream applies no latent scaling; our `latent_scale` is left
to fine-tuning.
"""

from __future__ import annotations

from pathlib import Path

import torch


def is_released_checkpoint(sd: dict) -> bool:
    return "align_masked_video.gate" in sd and "proj.weight" in sd


def resolve(init_from: str) -> Path:
    """A local file, or a Hugging Face repo id such as `facebook/sam-audio-large`."""
    p = Path(init_from).expanduser()
    if p.exists():
        return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(init_from, "checkpoint.pt"))


def convert(sam_sd: dict, ref_sd: dict) -> dict:
    out = {}
    for k, v in sam_sd.items():
        if not k.startswith("transformer.") or k.startswith("transformer.y_embedder."):
            continue
        if ".cross_attention." in k and k not in ref_sd:
            continue
        out[k] = v.clone()

    w, b = sam_sd["proj.weight"], sam_sd["proj.bias"]
    out["proj.weight"] = torch.cat([w[:, 0:256], w[:, 512:640] + w[:, 640:768]], dim=1).clone()

    g = sam_sd["align_masked_video.gate"].float()
    c = sam_sd["align_masked_video.conv.bias"].float()
    ln = ((c - c.mean()) / torch.sqrt(c.var(unbiased=False) + 1e-5)
          * sam_sd["align_masked_video.layer_norm.weight"] + sam_sd["align_masked_video.layer_norm.bias"])
    out["proj.bias"] = (b.float() + torch.tanh(g) * ln).to(b.dtype)

    bad = [(k, tuple(v.shape), tuple(ref_sd[k].shape))
           for k, v in out.items() if k in ref_sd and v.shape != ref_sd[k].shape]
    stray = sorted(set(out) - set(ref_sd))
    if bad or stray:
        raise RuntimeError(f"released checkpoint does not fit this backbone: "
                           f"shape mismatches {bad[:5]}, unknown keys {stray[:5]}")
    return out

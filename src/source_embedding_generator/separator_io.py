"""Loads a frozen query-conditioned separator (DiT or BS-Locoformer) or a
SEG checkpoint from their checkpoint directories.
"""

from __future__ import annotations

from pathlib import Path

import torch

from utils.torch_common import print_once

PCA_TRAIN = (Path(__file__).resolve().parents[1]
             / "query_conditioned_bslocoformer" / "pca_params_train.pkl")


def checkpoint_config(ckpt_dir):
    """The `config.yaml` of a checkpoint. The released weights come without
    one: their configs are in this repo, under `checkpoints/<name>/`."""
    from omegaconf import OmegaConf

    path = Path(ckpt_dir) / "config.yaml"
    if not path.exists():
        path = (Path(__file__).resolve().parents[2] / "checkpoints"
                / Path(ckpt_dir).name / "config.yaml")
    return OmegaConf.load(path)


def _relocate_pca(cfg, key: str) -> None:
    """Checkpoint configs record the PCA file by the path of the training
    machine; if it is not there, use the copy of the same name in this repo."""
    from omegaconf import OmegaConf

    pkl = OmegaConf.select(cfg, key)
    if isinstance(pkl, str) and not Path(pkl).exists():
        local = PCA_TRAIN.with_name(Path(pkl).name)
        if local.exists():
            OmegaConf.update(cfg, key, str(local))


def load_separator(ckpt_dir, device=None, use_ema: bool = True,
                   codec_compile: bool | None = None):
    """`ckpt_dir` -> the separator, in eval mode with grads off.

    Architecture is read from the checkpoint's own `config.yaml`.
    `codec_compile` overrides `model.codec.compile`. `use_ema` selects the
    EMA shadow (the default, and what the separator is evaluated with
    elsewhere) or the raw weights. Which set was used is always printed.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    ckpt_dir = Path(ckpt_dir)
    cfg = checkpoint_config(ckpt_dir)
    # Must never warm-start at eval time; weights come from the checkpoint.
    if OmegaConf.select(cfg, "model.backbone.init_from") is not None:
        OmegaConf.update(cfg, "model.backbone.init_from", None)

    # Repairs for checkpoints whose saved config names module paths that
    # have since moved. Only relocations are repaired, never hyperparameters.
    if OmegaConf.select(cfg, "model.loss._target_") == \
            "loss.snr.ThresSNRLossWithInactiveSource":
        OmegaConf.update(cfg, "model.loss._target_",
                         "query_conditioned_bslocoformer.loss."
                         "ThresSNRLossWithInactiveSource")
    # `codec.compile` is a training speed setting; force it off/on for a
    # reproducible eval run.
    if codec_compile is not None and OmegaConf.select(cfg, "model.codec") is not None:
        OmegaConf.update(cfg, "model.codec.compile", bool(codec_compile), force_add=True)

    # `latent_separator` was renamed `query_conditioned_dit`.
    for key in ("model._target_", "model.backbone._target_", "model.codec._target_"):
        t = OmegaConf.select(cfg, key)
        if isinstance(t, str) and t.startswith("latent_separator."):
            OmegaConf.update(cfg, key, "query_conditioned_dit." + t[len("latent_separator."):])
    # Older BS-Locoformer configs name `PcaAdapter` without a `pca_pkl`.
    if (OmegaConf.select(cfg, "model.query_adapter") is not None
            and OmegaConf.select(cfg, "model.query_adapter.pca_pkl") is None):
        OmegaConf.update(cfg, "model.query_adapter.pca_pkl", str(PCA_TRAIN),
                         force_add=True)
    _relocate_pca(cfg, "model.query_adapter.pca_pkl")

    sep = instantiate(cfg.model).float()
    # `backbone.null_mix` is a benign exception: older checkpoints lack it
    # and it inits to zeros. Both spellings appear (full model vs. EMA shadow).
    pre_cfg = ("backbone.null_mix", "null_mix")
    ema_path = ckpt_dir / "ema.pth"
    # The released DiT ships `ema.pth` alone: the EMA shadow holds every
    # trained weight, so `model.pth` is only needed for the raw weights.
    if (ckpt_dir / "model.pth").exists() or not (use_ema and ema_path.exists()):
        sd = torch.load(ckpt_dir / "model.pth", map_location="cpu",
                        weights_only=False)
        # Frozen encoders are stripped before saving and rebuilt in __init__, so
        # load non-strict: any unexpected key means the wrong architecture, any
        # missing key outside the frozen prefixes means an untrained backbone.
        frozen = ("query_model.", "query_adapter.", "codec.")
        missing, unexpected = sep.load_state_dict(sd, strict=False)
        if list(unexpected):
            raise RuntimeError(
                f"unexpected keys in separator ckpt: {list(unexpected)[:5]}")
        stray = [k for k in missing
                 if not k.startswith(frozen) and k not in pre_cfg]
        if stray:
            raise RuntimeError(f"missing non-frozen keys: {stray[:5]}")

    # EMA shadow covers `model.backbone` only (see `train_dit_separator.py`),
    # with keys prefixed `ema_model.`.
    if use_ema and ema_path.exists():
        esd = torch.load(ema_path, map_location="cpu", weights_only=False)
        esd = esd if isinstance(esd, dict) else esd.state_dict()
        shadow = {k[len("ema_model."):]: v for k, v in esd.items()
                  if k.startswith("ema_model.")}
        if shadow:
            e_missing, e_unexpected = sep.backbone.load_state_dict(
                shadow, strict=False)
            e_stray = [k for k in e_missing if k not in pre_cfg]
            if e_stray or list(e_unexpected):
                raise RuntimeError(
                    f"EMA shadow mismatch: missing {e_stray[:5]}, "
                    f"unexpected {list(e_unexpected)[:5]}")
            print_once(f"[SEG] separator using EMA weights "
                       f"(step {int(esd.get('step', -1))})")
    else:
        print_once(f"[SEG] separator using RAW weights"
                   + ("" if use_ema else " (--sep_no_ema)")
                   + ("" if ema_path.exists() else f"; no ema.pth in {ckpt_dir}"))
    if device is not None:
        sep = sep.to(device)
    return sep.eval().requires_grad_(False)


def load_seg(ckpt_dir, device=None, use_ema: bool = False):
    """`ckpt_dir` -> the stem embedding generator, in eval mode with grads off.

    Built from the checkpoint's own `config.yaml`. Frozen encoders (PaSST,
    PCA, DAC-VAE, loudness) are stripped from `model.pth` and rebuilt by the
    constructor. `use_ema` loads the backbone EMA shadow; paper SEG rows use
    the raw weights (`use_ema=False`).
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    ckpt_dir = Path(ckpt_dir)
    cfg = checkpoint_config(ckpt_dir).model
    cfg.init_from = None
    _relocate_pca(cfg, "query_adapter.pca_pkl")

    seg = instantiate(cfg).float()
    sd = torch.load(ckpt_dir / "model.pth", map_location="cpu", weights_only=False)
    missing, unexpected = seg.load_state_dict(sd, strict=False)
    frozen = ("query_model.", "query_adapter.", "mixture_encoder.", "loudness.")
    stray = [k for k in missing if not k.startswith(frozen)]
    if stray or list(unexpected):
        raise RuntimeError(f"SEG checkpoint mismatch in {ckpt_dir}: missing "
                           f"{stray[:5]}, unexpected {list(unexpected)[:5]}")
    if use_ema:
        esd = torch.load(ckpt_dir / "ema.pth", map_location="cpu", weights_only=False)
        shadow = {k[len("ema_model."):]: v for k, v in esd.items()
                  if k.startswith("ema_model.")}
        seg.backbone.load_state_dict(shadow)
    if device is not None:
        seg = seg.to(device)
    return seg.eval().requires_grad_(False)

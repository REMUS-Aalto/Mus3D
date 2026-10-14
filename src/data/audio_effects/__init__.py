"""Optional audio-effect distortion, bolted onto any dataset in this repo.

`EffectChainDataset` wraps a dataset rather than being one, so any existing
loader gains effects just by being wrapped in the config.

The effects themselves are `common_audioeffects.py`, vendored unmodified from
latent-source-restoration.
"""

from data.audio_effects.chain import (
    EffectChainDataset,
    build_augmentation_chain,
    load_impulse_responses,
)

__all__ = ["EffectChainDataset", "build_augmentation_chain",
           "load_impulse_responses"]

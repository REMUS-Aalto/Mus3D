"""Query-conditioned latent flow-matching separator (the DiT separator).

Layout:
    vendor/                verbatim SAM-Audio DiT -- do not edit, see vendor/README.md
    backbone.py            our shell around it (input projection, channel layout)
    modulation_backbone.py query + level conditioning by adaLN modulation
    codec.py               frozen DAC-VAE
    separator.py           the model: encode -> flow-match target+complement -> decode
    trainer.py             thin subclass of the query-Locoformer trainer
"""

from .backbone import SamAudioDiT
from .codec import DacVaeCodec
from .separator import LatentFlowSeparator

__all__ = ["SamAudioDiT", "DacVaeCodec", "LatentFlowSeparator"]

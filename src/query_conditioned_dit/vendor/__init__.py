"""Vendored SAM-Audio model code. See README.md -- do not edit."""

from .config import TransformerConfig
# `ProjectionLayer` is re-exported, not edited: `LevelConditionedSamAudioDiT`
# projects the hierarchy level with the SAME layer `t` is projected with, and
# reusing the class is how the two stay identical.
# `TimestepEmbedder` is re-exported for the same reason as `ProjectionLayer`:
# `backbone.HTimestepEmbedder` SUBCLASSES it to add the improved-MeanFlow
# interval width `h`, so the timestep half stays literally the vendored code
# and keeps its `t_embedder.projection.*` state-dict keys.
from .transformer import ContextEmbedder, DiT, ProjectionLayer, TimestepEmbedder

__all__ = ["TransformerConfig", "DiT", "ProjectionLayer", "ContextEmbedder",
           "TimestepEmbedder"]

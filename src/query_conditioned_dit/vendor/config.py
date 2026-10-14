# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# VENDORED from facebookresearch/sam-audio -- see vendor/README.md.
# Only `TransformerConfig` is kept; the rest of the upstream file configures
# codecs / text / vision encoders this project does not use.

from typing import Optional


class TransformerConfig:
    def __init__(
        self,
        dim: int = 2048,
        n_heads: int = 16,
        n_layers: int = 16,
        dropout: float = 0.1,
        norm_eps: float = 1.0e-05,
        qk_norm: bool = True,
        fc_bias: bool = False,
        ffn_exp: int = 4,
        ffn_dim_multiplier: int = 1,
        multiple_of: int = 64,
        non_linearity: str = "swiglu",
        use_rope: bool = True,
        max_positions: int = 10000,
        frequency_embedding_dim: int = 256,
        timestep_non_linearity: str = "swiglu",
        t_block_non_linearity: str = "silu",
        t_block_bias: bool = True,
        context_dim: int = 2048,
        context_non_linearity: str = "swiglu",
        context_embedder_dropout: float = 0.0,
        context_norm: bool = False,
        out_channels: int = 256,
        in_channels: Optional[int] = None,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.fc_bias = fc_bias
        self.ffn_exp = ffn_exp
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.non_linearity = non_linearity
        self.use_rope = use_rope
        self.max_positions = max_positions
        self.frequency_embedding_dim = frequency_embedding_dim
        self.timestep_non_linearity = timestep_non_linearity
        self.t_block_non_linearity = t_block_non_linearity
        self.t_block_bias = t_block_bias
        self.context_dim = context_dim
        self.context_non_linearity = context_non_linearity
        self.context_embedder_dropout = context_embedder_dropout
        self.context_norm = context_norm
        self.out_channels = out_channels
        self.in_channels = in_channels

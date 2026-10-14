# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0 license

from typing import Union

import math

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from packaging.version import parse as V

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class BSLocoformer(nn.Module):
    """BS-Locoformer presented in [1]. Following [2], we do not use any positional encoding.

    Reference:
    [1] Saijo, Kohei, et al., "Task-Aware Unified Source Separation," ICASSP 20255.
    [2] Saijo, Kohei, et al., "A Comparative Study on Positional Encoding for Time-frequency Domain Dual-path
        Transformer-based Source Separation Models," EUSIPCO, 2025


    Args:
        encoder: nn.Module
            The encoder module. (e.g., Conv2D, Band-split, SFC-CA, etc.)
        decoder: nn.Module
            The decoder module (e.g., Conv2D, Band-split, SFC-CA, etc.)
        n_srcs: int
            Number of output sources/speakers.
        n_chan: int
            Number of microphones channels (only fixed-array geometry supported).
        n_layers: int
            Number of TF-Locoformer blocks.
        emb_dim: int
            Number of hidden dimension in the encoding Conv2D.
        norm_type: str
            Normalization layer. Must be either of "layernorm" or "rmsgroupnorm".
        num_groups: int
            Number of groups in RMSGroupNorm layer.
        tf_order: str
            Order of frequency and temporal modeling. Must be either of "ft" or "tf".
        n_heads: int
            Number of heads in multi-head self-attention.
        flash_attention: bool
            Whether to use flash attention. Only compatible with half precision.
        ffn_type: str or list
            Giving the list (e.g., ["conv1d", "conv1d"]) makes the model Macaron-style.
        ffn_hidden_dim: int or list
            Number of hidden dimension in FFN.
            Giving the list (e.g., [256, 256]) makes the model Macaron-style.
        conv1d_kernel:
            Kernel size in Conv1d.
        conv1d_shift:
            Shift size of Conv1d kernel.
        dropout: float
            Dropout probability.
        masking: bool
            Whether to perform masking or mapping
        eps: float
            Small constant for nomalization layer.
        checkpointing: bool
            If True, gradient checkpointing is applied to the encoder and decoder to save GPU memory during training.
    """

    def __init__(
        self,
        encoder,
        decoder,
        n_src: int = 2,
        n_chan: int = 1,
        n_layers: int = 4,
        # general setup
        emb_dim: int = 96,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = True,  # available when using mixed precision
        attention_dim: int = 96,
        # ffn related
        ffn_type: Union[str, list] = "swiglu_conv1d",
        ffn_hidden_dim: Union[int, list] = 128,
        conv1d_kernel: int = 8,
        conv1d_shift: int = 1,
        dropout: float = 0.1,
        # others
        masking: bool = True,
        eps: float = 1.0e-5,
        checkpointing: bool = False,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"
        self.n_src = n_src
        self.n_chan = n_chan
        self.n_layers = n_layers
        assert attention_dim % n_heads == 0, (attention_dim, n_heads)

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    # general setup
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )

        self.encoder = encoder
        self.decoder = decoder

        self.masking = masking
        self.checkpointing = checkpointing

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward.

        Args:
            input (torch.Tensor): multi-channel audio tensor with
                    M audio channels in TF-domain [B, M, F, T]

        Returns:
            batch (torch.Tensor): separated sources with a shape of
                [B, N, M, F, T]
        """
        assert input.ndim == 4, "Input must have 4 dims."

        n_batch, n_chan, n_freq, n_frames = input.shape

        # input: (B, F, M, T)
        batch0 = input.transpose(-2, -1)  # [B, M, F, T] -> [B, M, T, F]

        # normal spectrogram -> band-splitted tensor
        if self.training and self.checkpointing:
            batch = checkpoint(lambda x: self.encoder(x), batch0, use_reentrant=False)
        else:
            batch = self.encoder(batch0)  # [B, -1, T, F]
        if isinstance(batch, tuple) and len(batch) == 2:
            batch, emb = batch
        else:
            emb = None

        # separation
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        # band-splitted tensor -> normal spectrogram
        if self.training and self.checkpointing:
            batch = checkpoint(lambda x, q: self.decoder(x, query=q), batch, emb, use_reentrant=False)
        else:
            batch = self.decoder(batch, query=emb)  # [B, 2NM, T, F]
        if isinstance(batch, tuple) and len(batch) == 2:
            batch = batch[0]

        if not batch.is_complex():
            if batch.shape != (n_batch, 2, self.n_src, n_chan, n_frames, n_freq):
                batch = batch.reshape(n_batch, 2, self.n_src, n_chan, n_frames, n_freq)
            batch = batch.to(torch.float32)
            batch = torch.complex(batch[:, 0], batch[:, 1])  # [B, N, M, T, F]
        else:
            batch = batch.reshape(n_batch, self.n_src, n_chan, n_frames, n_freq)
        # complex masking
        if self.masking:
            batch = batch0.unsqueeze(1) * batch  # [B, N, M, T, F]
        return batch.transpose(-1, -2)


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        # general setup
        emb_dim=128,
        norm_type="rmsgroupnorm",
        num_groups=4,
        tf_order="ft",
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

        self.freq_path = LocoformerBlock(
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )

    # @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, input):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        """
        output = self.freq_frame_process(input) if self.tf_order == "ft" else self.frame_freq_process(input)
        return output

    def freq_frame_process(self, input):
        output = input.permute(0, 2, 3, 1)
        output = self.freq_path(output)

        output = output.permute(0, 2, 1, 3)
        output = self.frame_path(output)
        return output.permute(0, 3, 2, 1)

    def frame_freq_process(self, input):
        output = input.permute(0, 3, 2, 1)
        output = self.frame_path(output)

        output = output.permute(0, 2, 1, 3)
        output = self.freq_path(output)
        return output.permute(0, 3, 2, 1)


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        # general setup
        emb_dim=128,
        norm_type="rmsgroupnorm",
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {"swiglu_conv1d": SwiGLUConvDeconv1d}
        Norm = {"layernorm": nn.LayerNorm, "rmsgroupnorm": RMSGroupNorm}
        assert norm_type in Norm, norm_type

        self.macaron_style = len(ffn_type) == 2
        if self.macaron_style:
            assert len(ffn_hidden_dim) == 2, "Need two FFNs when using macaron style model"

        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            assert f_type in FFN, f_type
            if norm_type == "rmsgroupnorm":
                self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
            else:
                self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )

        # initialize self-attention
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    # @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        B, T, F, C = x.shape

        if self.macaron_style:
            # FFN before self-attention
            input_ = x
            output = self.ffn_norm[-1](x)  # [B, T, F, C]
            output = self.ffn[-1](output)  # [B, T, F, C]
            output = output.add_(input_)
        else:
            output = x

        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = output.reshape(B * T, F, C)
        output = self.attn(output)
        output = output.reshape(B, T, F, C).add_(input_)

        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output.add_(input_)

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        self.sdpa_kernel = SDPBackend.FLASH_ATTENTION if flash_attention else SDPBackend.EFFICIENT_ATTENTION

    # @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        with sdpa_kernel(self.sdpa_kernel):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        # Merge heads by a single transpose + reshape, then project back
        n_batch, n_heads, seq_len, head_dim = output.shape
        output = output.transpose(1, 2).reshape(n_batch, seq_len, n_heads * head_dim)  # (B, T, H*Dh)
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        # One reshape to (B, T, 3, H, Dh), one permute to (B, H, T, 3, Dh), then unbind
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input)  # (B, T, 3*H*Dh)
        x = x.reshape(n_batch, seq_len, 3, self.n_heads, -1).permute(0, 3, 1, 2, 4)  # (B, H, T, 3, Dh)
        query, key, value = x.unbind(dim=3)  # three views, each (B, H, T, Dh)
        return query, key, value


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    # @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape

        x = x.transpose(-1, -2).reshape(b * s1, h, s2)  # (b*s1, s2, h)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        x1, x2 = x.split(self.dim_inner, dim=1)
        x = x1 * F.silu(x2)
        x = self.dropout(x)

        x = self.deconv1d(x)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2]
        x = x.transpose(-1, -2).reshape(b, s1, s2, h)  # (b, s1, s2, h)
        return self.dropout(x)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups: int, dim: int, eps: float = 1e-8, bias: bool = False):
        super().__init__()
        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // num_groups
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.register_parameter("beta", None)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype

        x32 = x.to(torch.float32).unflatten(-1, (self.num_groups, self.dim_per_group))

        rms = x32.square().mean(dim=-1, keepdim=True).sqrt()
        y32 = x32 / (rms + self.eps)
        y = y32.reshape(x.shape).to(dtype_in)

        # affine
        y = y * self.gamma.to(dtype_in)
        if self.beta is not None:
            y = y + self.beta.to(dtype_in)
        return y

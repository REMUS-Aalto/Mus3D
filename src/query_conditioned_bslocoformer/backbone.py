import torch
import torchaudio
from torch import nn

from . import BanditDecoder, BanditEncoder, BSLocoformer, LevelConditionedBSLocoformer


class BanditBSLocoformer(nn.Module):
    """Wrapper for the BS-Locoformer with the bandit encoder from [1]

    Reference:
    [1] Saijo, Kohei, et al., "Input-Adaptive Spectral Feature Compression by  Sequence Modeling for Source Separation," TASLPRO 2026

    Args:
        n_chan: int
            Number of channels per stem (i.e. one for mono, two for stereo).
        masking: bool
            Whether to apply masking or predict the sources directly.
        kwargs:
            Arguments that do not apply to this backbone.
    """

    def __init__(
        self,
        n_chan: int,
        masking: bool,
        sample_rate: int = 44100,
        n_bands: int = 64,
        emb_dim: int = 96,
        mlp_dim: int = 4 * 96,
        transformer_blocks: int = 4,
        ffn_hidden_dim: list[int] | None = None,
        attention_heads: int = 8,
        norm_groups: int = 8,
        decoder_cond_dim: int = 0,
    ):
        super().__init__()

        if ffn_hidden_dim is None:
            ffn_hidden_dim = [128, 128]

        self.n_fft = 2048

        encoder = BanditEncoder(
            "musical",
            self.n_fft,
            sample_rate,
            n_bands,
            emb_dim,
            n_chan,
        )

        # `decoder_cond_dim` sets the width `OverlappingMaskEstimationModule`
        # builds its norm_mlp at (emb_dim + cond_dim); exposed so a loader can
        # match a checkpoint's value instead of the current default.
        decoder = BanditDecoder(
            "musical",
            self.n_fft,
            sample_rate,
            n_bands,
            emb_dim,
            mlp_dim,
            n_chan,
            decoder_cond_dim,
        )


        self.net = BSLocoformer(
            encoder,
            decoder,
            n_chan,
            n_layers=transformer_blocks,
            emb_dim=emb_dim,
            attention_dim=emb_dim,
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
            ffn_hidden_dim=ffn_hidden_dim,
            n_heads=attention_heads,
            num_groups=norm_groups,
            masking=masking,
        )

        self.stft = torchaudio.transforms.Spectrogram(2048, 512, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(2048, 512)

    def forward(self, input: torch.Tensor, q_embs: torch.Tensor, **kwargs) -> torch.Tensor:
        # pad for Spectrogram
        x_len = input.shape[-1]
        n_pad = self.n_fft - x_len % self.n_fft
        x = nn.functional.pad(input, (0, n_pad))
        
        # TODO possibly need to disable autocast
        x = self.stft(x)
        x = self.net(x, q_embs, **kwargs)
        x = self.istft(x)

        # discard pad
        x = x[..., :x_len]
        return x


class LevelConditionedBanditBSLocoformer(BanditBSLocoformer):
    """Bandit BS-Locoformer wrapper with hierarchy level conditioning.
    
    This is the level-conditioned version of BanditBSLocoformer, using
    LevelConditionedBSLocoformer as the core network. It accepts a `level`
    parameter in forward() and passes it to the underlying model.
    """
    
    accepts_level = True

    def __init__(
        self,
        n_chan: int,
        masking: bool,
        sample_rate: int = 44100,
        n_bands: int = 64,
        emb_dim: int = 96,
        mlp_dim: int = 4 * 96,
        transformer_blocks: int = 4,
        ffn_hidden_dim: list[int] | None = None,
        attention_heads: int = 8,
        norm_groups: int = 8,
        decoder_cond_dim: int = 0,
    ):
        super().__init__(
            decoder_cond_dim=decoder_cond_dim,
            n_chan=n_chan,
            masking=masking,
            sample_rate=sample_rate,
            n_bands=n_bands,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            transformer_blocks=transformer_blocks,
            ffn_hidden_dim=ffn_hidden_dim,
            attention_heads=attention_heads,
            norm_groups=norm_groups,
        )
        
        # Replace the BSLocoformer with LevelConditionedBSLocoformer
        # We need to extract encoder and decoder from the parent's net
        encoder = self.net.encoder
        decoder = self.net.decoder
        
        self.net = LevelConditionedBSLocoformer(
            encoder,
            decoder,
            n_chan,
            n_layers=transformer_blocks,
            emb_dim=emb_dim,
            attention_dim=emb_dim,
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
            ffn_hidden_dim=ffn_hidden_dim,
            n_heads=attention_heads,
            num_groups=norm_groups,
            masking=masking,
        )

    def forward(
        self,
        input: torch.Tensor,
        q_embs: torch.Tensor,
        level: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with level conditioning.

        Args:
            input: [B, C, T] audio tensor
            q_embs: [B, D] query embedding tensor
            level: [B] long in {0, 1}, or None. Hierarchy level.

        Returns:
            [B, C, T] separated audio
        """
        # pad for Spectrogram
        x_len = input.shape[-1]
        n_pad = self.n_fft - x_len % self.n_fft
        x = nn.functional.pad(input, (0, n_pad))
        
        # TODO possibly need to disable autocast
        x = self.stft(x)
        x = self.net(x, q_embs, level=level, **kwargs)
        x = self.istft(x)

        # discard pad
        x = x[..., :x_len]
        return x

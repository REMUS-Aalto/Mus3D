import torch
import torchaudio
from torch import nn

from static_bslocoformer.bandit_split import BanditDecoder, BanditEncoder
from static_bslocoformer.bslocoformer import BSLocoformer


class BanditBSLocoformer(nn.Module):
    """Wrapper for the BS-Locoformer with the bandit encoder from [1]

    Reference:
    [1] Saijo, Kohei, et al., "Input-Adaptive Spectral Feature Compression by  Sequence Modeling for Source Separation," TASLPRO 2026

    Args:
        n_srcs: int
            Number of output stems.
        n_chan: int
            Number of channels per stem (i.e. one for mono, two for stereo).
        masking: bool
            Whether to apply masking or predict the sources directly.
        kwargs:
            Arguments that do not apply to this backbone.
    """

    def __init__(
        self,
        n_src: int,
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

        decoder = BanditDecoder(
            "musical",
            self.n_fft,
            sample_rate,
            n_bands,
            emb_dim,
            mlp_dim,
            n_chan,
            n_src,
        )

        self.net = BSLocoformer(
            encoder,
            decoder,
            n_src,
            n_chan,
            n_layers=transformer_blocks,
            emb_dim=emb_dim,
            attention_dim=emb_dim,
            # defaults match the small model
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
            ffn_hidden_dim=ffn_hidden_dim,
            n_heads=attention_heads,
            num_groups=norm_groups,
            masking=masking,
        )

        self.stft = torchaudio.transforms.Spectrogram(2048, 512, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(2048, 512)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, 1, C, T)

        Returns:
            Tensor (B, N_src, C, T)
        """
        # pad for Spectrogram
        x_len = x.shape[-1]
        n_pad = self.n_fft - x_len % self.n_fft
        x = nn.functional.pad(x, (0, n_pad))

        # TODO possibly need to disable autocast
        x = self.stft(x)
        x = self.net(x)
        x = self.istft(x)

        # discard pad
        x = x[..., :x_len]
        return x

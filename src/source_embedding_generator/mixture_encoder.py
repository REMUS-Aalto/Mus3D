"""Mixture conditioning encoder for SEG.

`FMGenerator` cross-attends over `encode(audio, sr) -> [B, C, L]`.
`DacVaeMixtureEncoder` is the default, using the same DAC-VAE latent
space as `query_conditioned_dit`; see the paper for the design choice.
"""

from __future__ import annotations

import torch
from torch import nn


class DacVaeMixtureEncoder(nn.Module):
    """Frozen DAC-VAE as SEG's mixture conditioner. `encode` -> [B, 128, L].

    Adapts `query_conditioned_dit.codec.DacVaeCodec` (channels-last) to the
    channels-first convention `EquivariantSetDiT._context` expects.
    """

    def __init__(
        self,
        model_id: str = "facebook/dacvae-watermarked",
        audio_sr: int = 44100,
        latent_scale: float = 0.7013,
        chunk: int = 16,
        bf16: bool = True,
    ):
        super().__init__()
        from query_conditioned_dit.codec import DacVaeCodec

        # bf16 is passed to the codec rather than wrapped around it, so its
        # internal resample stays fp32; see DacVaeCodec for details.
        self.codec = DacVaeCodec(model_id=model_id, audio_sr=audio_sr,
                                 chunk=chunk, bf16=bf16)
        self.latent_scale = latent_scale
        self.bf16 = bf16
        self.latent_dim = self.codec.latent_dim
        self.sample_rate = audio_sr

    def train(self, mode: bool = True):
        """Frozen for good."""
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sr: int | None = None) -> torch.Tensor:
        """[B, C, T] audio at `audio_sr` -> [B, 128, L] conditioning latents.

        `sr` is accepted and ignored; the codec resamples internally.
        """
        z = self.codec.encode(audio)
        return z.transpose(1, 2).float() / self.latent_scale

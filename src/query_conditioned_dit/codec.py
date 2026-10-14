"""Frozen DAC-VAE, the latent space this separator works in.

`facebook/dacvae-watermarked` -- the same codec SAM-Audio uses. 48 kHz, mono,
hop 1920 -> 25 latent frames/s, latent dim 128.

Handles resampling (dataset is 44.1 kHz) and mono folding (dataset is
stereo), so latents have no stereo image. Also not deepcopyable (weight_norm),
so it must never be handed to `ema_pytorch.EMA` -- see `ema_fn` in
`train_dit_separator.py`.

Encoding takes the posterior mean, not a sample, so the flow-matching target
is deterministic for a given audio input.
"""

import torch
import torchaudio as ta
from torch import nn

DACVAE_SR = 48000
DACVAE_HOP = 1920
DACVAE_LATENT_DIM = 128


class DacVaeCodec(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/dacvae-watermarked",
        audio_sr: int = 44100,
        chunk: int = 16,
        bf16: bool = False,
        compile: bool = False,
    ):
        """`bf16` / `compile` speed up encode, which dominates the training
        step; both default off so numerics stay unchanged unless opted in.
        `decode` always runs in fp32.
        """
        super().__init__()
        import dacvae

        self.audio_sr = audio_sr
        self.chunk = chunk
        self.bf16 = bf16
        self.latent_dim = DACVAE_LATENT_DIM
        self.hop = DACVAE_HOP
        self.sample_rate = DACVAE_SR

        self.vae = dacvae.DACVAE.load(model_id).eval().requires_grad_(False)
        if compile:
            # Encoder only; decode stays eager.
            self.vae.encoder = torch.compile(self.vae.encoder)
        self.to_codec = ta.transforms.Resample(audio_sr, DACVAE_SR)
        self.from_codec = ta.transforms.Resample(DACVAE_SR, audio_sr)

    def train(self, mode: bool = True):
        """Frozen for good: stay in eval whatever the trainer does."""
        super().train(False)
        return self

    def n_frames(self, n_samples: int) -> int:
        n = int(round(n_samples * DACVAE_SR / self.audio_sr))
        return -(-n // DACVAE_HOP)

    def _chunked(self, fn, x):
        if x.shape[0] <= self.chunk:
            return fn(x)
        return torch.cat([fn(c) for c in x.split(self.chunk)], dim=0)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, C, T] audio at `audio_sr` -> [B, L, 128] latents.

        Channels are averaged to mono; the codec has no stereo mode.
        """
        x = self.to_codec(wav.float().mean(dim=-2, keepdim=True))

        def enc(c):
            z = self.vae.encoder(self.vae._pad(c))
            mean, _ = self.vae.quantizer.in_proj(z).chunk(2, dim=1)
            return mean

        # Always returns fp32; bf16 is an internal speed choice only.
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.bf16 and x.is_cuda):
            return self._chunked(enc, x).transpose(1, 2).float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, L, 128] latents -> [B, 1, T] audio at `audio_sr`."""
        z = z.transpose(1, 2).float()
        # fp32 explicitly, even if a caller has autocast on.
        with torch.autocast("cuda", enabled=False):
            wav = self._chunked(lambda c: self.vae.decode(c), z)
        return self.from_codec(wav)


class SameCodec(nn.Module):
    """Frozen SAME-L autoencoder, `DacVaeCodec`-shaped drop-in for
    `LatentFlowSeparator`.

    Stereo (`io_channels=2`, latent dim 256), unlike `DacVaeCodec`; no
    resampling, so `audio_sr` must match the checkpoint's native rate
    (44.1 kHz for SAME-L, asserted in `__init__`).
    """

    def __init__(
        self,
        model_name: str = "same-l",
        audio_sr: int = 44100,
        bf16: bool = True,
    ):
        super().__init__()
        from source_embedding_generator.SAME_autoencoder import SAMEAutoencoder

        self.audio_sr = audio_sr
        self.ae = SAMEAutoencoder(model_name=model_name, bf16=bf16)
        assert self.ae.sample_rate == audio_sr, (
            f"SameCodec built for audio_sr={audio_sr}, but {model_name} is "
            f"natively {self.ae.sample_rate} Hz -- SAMEAutoencoder.encode does "
            "not resample."
        )
        self.latent_dim = self.ae.latent_dim
        self.hop = self.ae.downsampling_ratio
        self.sample_rate = self.ae.sample_rate

    def train(self, mode: bool = True):
        """Frozen for good: stay in eval whatever the trainer does."""
        super().train(False)
        return self

    def n_frames(self, n_samples: int) -> int:
        return -(-n_samples // self.hop)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, 2, T] stereo audio at `audio_sr` -> [B, L, latent_dim] latents."""
        z = self.ae.encode(wav, self.audio_sr)
        return z.transpose(1, 2)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, L, latent_dim] latents -> [B, 2, T] stereo audio."""
        return self.ae.decode(z.transpose(1, 2))

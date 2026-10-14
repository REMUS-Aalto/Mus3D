"""MERT-MSE: mean squared error between MERT-v1-95M layer-12 embeddings.

Recipe of Bereuter & Sontacchi (gensvs 2.0.4): mono downmix, resampling to
24 kHz with gensvs' Kaiser-sinc settings, a 16-bit PCM round trip, then the MSE
over the [frames, 768] embedding. Used by `reproduce_separation_from_audio.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MERT_ID = "m-a-p/MERT-v1-95M"
MERT_LAYER = 12
MERT_SR = 24000
RESAMPLE_KW = dict(lowpass_filter_width=64,
                   rolloff=0.9475937167399596,
                   resampling_method="sinc_interp_kaiser",
                   beta=14.769656459379492)


class Mert:
    """MERT-v1-95M, loaded once, embedding one mono excerpt at a time."""

    def __init__(self, device: str = "cuda", layer: int = MERT_LAYER):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.device = torch.device(device)
        self.layer = layer
        self.model = AutoModel.from_pretrained(
            MERT_ID, trust_remote_code=True).to(self.device).eval()
        self.proc = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_ID, trust_remote_code=True)
        self._resamplers: dict[int, torch.nn.Module] = {}

    def _resample(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == MERT_SR:
            return x
        if sr not in self._resamplers:
            import torchaudio
            self._resamplers[sr] = torchaudio.transforms.Resample(
                sr, MERT_SR, **RESAMPLE_KW)
        return self._resamplers[sr](x)

    def embed(self, path: Path) -> np.ndarray | None:
        """[n_frames, 768] float16, or None if the file is silent."""
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = torch.from_numpy(x.T).mean(0, keepdim=True)   # mono, [1, T]
        if float(x.abs().max()) < 1e-6:
            return None
        x = self._resample(x, sr)
        # The 16-bit round-trip `gensvs` performs by writing a PCM_S wav and
        # reading it back with `dtype='int16'`. Clipped, because a resampler
        # can overshoot past full scale and soundfile would wrap it.
        x = torch.round(x.clamp(-1.0, 1.0) * 32768.0).clamp(-32768, 32767) / 32768.0
        inputs = self.proc(x.squeeze(0).numpy(), sampling_rate=MERT_SR,
                           return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        h = torch.stack(out.hidden_states).squeeze(1)      # [13, frames, 768]
        return h[self.layer].cpu().numpy().astype(np.float16)


def mse_mert(a: np.ndarray, b: np.ndarray) -> float:
    """Eq. (1): mean squared error over the whole [frames, 768] embedding.

    Accumulated in float32. The arrays are float16 because that is what
    `gensvs` caches and compares, but `np.mean` over float16 saturates well
    before these sums are done.
    """
    n = min(a.shape[0], b.shape[0])
    d = a[:n].astype(np.float32) - b[:n].astype(np.float32)
    return float(np.mean(d ** 2))

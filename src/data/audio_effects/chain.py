"""Apply chains of audio effects to whatever a dataset already yields.

`EffectChainDataset` wraps any iterable dataset in this repo and returns items
of the same structure with the audio distorted; `__getattr__` delegation keeps
`collate_fn`, `song_ids`, etc. working through the wrapper. The demo path
calls the underlying dataset directly, so demo songs stay un-distorted.

Effects are applied per slot, not to the summed mixture, since every consumer
here defines the mixture as the sum of the slots; distorting the mixture
instead would break that invariant.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset

ACCEPTED_SR = (44100, 48000)


def load_impulse_responses(csv_paths, target_sr: int = 44100,
                           max_len_s: float = 0.75):
    """IR index CSV(s) -> a preloaded DataFrame for `ConvolutionalReverb`.

    Each CSV has an `impulse_response` column of paths, relative to the
    directory named by the `IR_ROOT` environment variable (or absolute), already
    T60-filtered upstream. IRs are read once in the parent process so forked
    DataLoader workers share them copy-on-write, and trimmed to `max_len_s`
    to bound convolution cost.
    """
    import os

    import pandas as pd
    import soundfile as sf
    from scipy.signal import resample_poly

    from collections.abc import Mapping

    from omegaconf import DictConfig, OmegaConf

    if isinstance(csv_paths, DictConfig):
        csv_paths = OmegaConf.to_container(csv_paths, resolve=True)
    if isinstance(csv_paths, (str, os.PathLike)):
        csv_paths = {"default": csv_paths}
    elif not isinstance(csv_paths, Mapping):
        # A bare list of CSV paths; Mapping is checked first since enumerating
        # a dict would otherwise iterate over its keys.
        csv_paths = {str(i): p for i, p in enumerate(csv_paths)}

    frames = None
    for name, path in csv_paths.items():
        if not os.path.exists(path):
            print(f"[effects] IR index not found: {path}")
            continue
        df = pd.read_csv(path)
        if "impulse_response" not in df.columns:
            print(f"[effects] IR CSV {path} lacks an 'impulse_response' column")
            continue
        frames = df if frames is None else pd.concat([frames, df], ignore_index=True)
        print(f"[effects] indexed {len(df)} IRs from {name}")

    if frames is None or len(frames) == 0:
        return None
    frames = frames.reset_index(drop=True)

    ir_root = os.environ.get("IR_ROOT")
    if ir_root is None and not frames["impulse_response"].map(os.path.isabs).all():
        raise RuntimeError("the IR index holds relative paths: set IR_ROOT to the "
                           "directory holding the impulse-response datasets")
    max_len = int(max_len_s * target_sr)
    loaded, n_fail = [], 0
    for path in frames["impulse_response"]:
        try:
            audio, sr = sf.read(os.path.join(ir_root or "", path), dtype="float32",
                                always_2d=True)
        except Exception:
            loaded.append(None)
            n_fail += 1
            continue
        if sr != target_sr:
            g = np.gcd(int(sr), target_sr)
            audio = resample_poly(audio, target_sr // g, int(sr) // g,
                                  axis=0).astype("float32")
        if audio.shape[0] > max_len:
            audio = audio[:max_len]
        loaded.append((audio.shape[0], np.ascontiguousarray(audio)))

    frames = frames.copy()
    frames["impulse_response"] = pd.Series(loaded, index=frames.index, dtype=object)
    frames = frames[frames["impulse_response"].notnull()].reset_index(drop=True)
    print(f"[effects] preloaded {len(frames)} IRs at {target_sr} Hz "
          f"(<= {max_len} samples, {n_fail} failed)")
    return frames


def build_augmentation_chain(
    audio_effects,
    count_probs=None,
    p_effects: float = 0.5,
    impulse_responses=None,
    shuffle: bool = True,
):
    """Instantiate the effect list into an `AugmentationChain`, or None.

    Two selection modes:

    *   `count_probs` set: sample a number of effects k, then pick k distinct
        effects by `weight`.
    *   `count_probs` None: each effect fires independently with its own
        `probability` (falling back to `p_effects`).

    A `ConvolutionalReverb` entry is dropped when no impulse responses are
    supplied.
    """
    import hydra.utils

    from data.audio_effects.common_audioeffects import (
        AugmentationChain,
        ConvolutionalReverb,
    )

    if not audio_effects:
        return None

    effects = []
    for cfg in audio_effects:
        cfg = dict(cfg)
        prob = cfg.pop("probability", p_effects)
        weight = cfg.pop("weight", 1.0)
        value = weight if count_probs is not None else prob

        target = str(cfg.get("_target_", ""))
        if target.endswith("ConvolutionalReverb"):
            if impulse_responses is None or len(impulse_responses) == 0:
                print("[effects] no impulse responses; skipping ConvolutionalReverb")
                continue
            cfg.pop("_target_", None)
            cfg.setdefault("impulse_responses", impulse_responses)
            effects.append((ConvolutionalReverb(**cfg), value))
            continue

        effects.append((hydra.utils.instantiate(cfg), value))

    if not effects:
        return None

    return AugmentationChain(
        effects,
        shuffle=shuffle,
        apply_to="target",
        count_probs=None if count_probs is None else list(count_probs),
    )


class EffectChainDataset(IterableDataset):
    """Wrap a dataset; distort the audio it yields.

    Args:
        dataset: the dataset to wrap. Its items may be a `[N, C, T]` tensor
            (separator / LFM style) or `(tensor, labels)` (SEG style); both are
            passed through with the same structure.
        audio_effects: list of effect configs, each with `_target_` and
            optionally `weight` / `probability` / `parameter_ranges`.
        count_probs: categorical distribution over the NUMBER of effects per
            draw. See `build_augmentation_chain`.
        p_apply: probability that an ITEM is distorted at all. `p_apply < 1`
            keeps clean audio in the distribution, which matters because the
            eventual test input is usually not distorted.
        sample_rate: must be 44.1 or 48 kHz -- several effects assume it.
        match_rms: rescale the wet signal so its RMS equals the dry signal's.
            The chain is meant to change TIMBRE, not level -- level is the
            dataset's `gain_db` augmentation, applied separately. Individual
            processors carry `normalize: rms`, but only where the config sets
            it, so without this the chain still drifts in level (reverb and the
            codec effects do not normalise) and can push peaks past unity.
    """

    def __init__(
        self,
        dataset,
        audio_effects=None,
        count_probs=None,
        p_effects: float = 0.5,
        p_apply: float = 0.5,
        impulse_responses=None,
        sample_rate: int = 44100,
        silence_rms: float = 1e-5,
        match_rms: bool = True,
        record_effects: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        # Instantiated here, not by Hydra: `_recursive_: false` keeps
        # `audio_effects` as plain dicts so `build_augmentation_chain` can see
        # them before any effect is constructed.
        import hydra.utils
        from omegaconf import DictConfig

        if isinstance(dataset, (dict, DictConfig)):
            dataset = hydra.utils.instantiate(dataset)

        if sample_rate not in ACCEPTED_SR:
            raise ValueError(
                f"audio effects expect {ACCEPTED_SR} Hz, got {sample_rate}"
            )
        self.dataset = dataset
        self.p_apply = p_apply
        self.silence_rms = silence_rms
        self.match_rms = match_rms
        self.seed = seed
        if impulse_responses is not None and not hasattr(impulse_responses, "columns"):
            impulse_responses = load_impulse_responses(impulse_responses)
        self.chain = build_augmentation_chain(
            audio_effects, count_probs, p_effects, impulse_responses
        )
        # Off during training; only read by inspection tooling.
        self.record_effects = record_effects
        self.last_effects: list[dict] = []
        self._recording: list[dict] = []
        if self.chain is None:
            print("[effects] no chain built; wrapper is a pass-through")
        elif record_effects:
            self._install_recorder()

    def _install_recorder(self):
        """Make the chain report which effects it fired, and with what settings.

        `AugmentationChain` keeps no record of a draw, so `_apply_fx` is
        wrapped in place on this instance rather than forking the vendored
        file.
        """
        chain = self.chain
        original = chain._apply_fx
        outer = self

        def recording_apply_fx(fx, input_y, target_y):
            out = original(fx, input_y, target_y)
            try:
                # ParameterList has no iteration API; read the instance dict.
                plist = getattr(fx, "parameters", None)
                params = {
                    k: getattr(v, "value", v)
                    for k, v in vars(plist).items()
                    if hasattr(v, "value")
                } if plist is not None else {}
                outer._recording.append(
                    {"effect": type(fx).__name__, "params": params}
                )
            except Exception:
                outer._recording.append({"effect": type(fx).__name__, "params": {}})
            return out

        chain._apply_fx = recording_apply_fx

    def __getattr__(self, name):
        # `dataset` itself is a miss during unpickling in a worker, which
        # would otherwise recurse forever.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Un-distorted, deliberately: this is the demo/validation path."""
        return self.dataset[idx]

    def _apply(self, wav: torch.Tensor) -> torch.Tensor:
        """[C, T] -> [C, T] with one chain draw applied.

        The chain works in numpy `[T, C]` float32 and may return a different
        length, so the result is trimmed or zero-padded back.

        Slots under `silence_rms` are skipped entirely: most effects here
        normalise by RMS and would otherwise amplify near-silence (or an
        empty residual buffer) into a loud artefact.
        """
        if self.record_effects:
            self._recording = []
            self.last_effects = []
        # Zero non-finite input rather than let it reach the model and
        # poison the gradient.
        if not torch.isfinite(wav).all():
            print("[effects] non-finite input audio; zeroing the offending samples")
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        if float(wav.pow(2).mean().sqrt()) < self.silence_rms:
            return wav
        t, c = wav.shape[-1], wav.shape[0]
        x = np.ascontiguousarray(wav.numpy().T.astype("float32"))
        try:
            _, wet = self.chain(x.copy(), x.copy(), mono=(c == 1))
        except Exception as e:  # an effect blew up on this audio
            print(f"[effects] chain failed, passing audio through: {e}")
            return wav
        wet = np.asarray(wet, dtype="float32")
        if wet.ndim == 1:
            wet = wet[:, None]
        if wet.shape[1] == 1 and c == 2:
            wet = np.repeat(wet, 2, axis=1)
        out = torch.from_numpy(np.ascontiguousarray(wet.T))
        if out.shape[-1] > t:
            out = out[..., :t]
        elif out.shape[-1] < t:
            out = torch.nn.functional.pad(out, (0, t - out.shape[-1]))
        if not torch.isfinite(out).all():
            return wav
        if self.record_effects:
            self.last_effects = list(self._recording)
        if self.match_rms:
            dry = float(wav.pow(2).mean().sqrt())
            wet_rms = float(out.pow(2).mean().sqrt())
            # A chain that nearly annihilates the signal makes this gain
            # enormous; re-check finiteness rather than assume.
            if wet_rms > 0.0:
                scaled = out * (dry / wet_rms)
                if torch.isfinite(scaled).all():
                    out = scaled
        return out

    def _distort(self, audio: torch.Tensor) -> torch.Tensor:
        """[N, C, T] slots -> distorted slots, one chain draw each."""
        return torch.stack([self._apply(a) for a in audio])

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else worker.id
        rng = np.random.default_rng([wid, self.seed])
        # The effects use numpy's global RNG internally; without this every
        # worker draws the same effect parameters in the same order.
        np.random.seed(int(rng.integers(2**31 - 1)))

        for item in self.dataset:
            if self.chain is None or rng.random() >= self.p_apply:
                yield item
                continue
            if isinstance(item, tuple):
                audio, *rest = item
                yield (self._distort(audio), *rest)
            else:
                yield self._distort(item)

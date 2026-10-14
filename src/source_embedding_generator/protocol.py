"""Sampling protocols of the training sets, and the loudness gate they use.

`sample_recursive`   one step of the recursive extraction: C units already
                     available (conditioning), one target and one residual.
                     Used by `MoisesDBRecursiveDataset` (SEG training).
`sample_targets`     K target units at once; the remaining units are the
                     residual. Used by `MoisesDBOriginalMixSeparationDataset`
                     (separator training, where the residual is not used).
`active_tracks`      which tracks are audible in a segment (RMS, relative level
                     and `TorchLoudness`, ITU-R BS.1770 integrated loudness).

Both samplers are pure and rng-driven, operating on per-track audio plus the
MoisesDB level-1 family of each track.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin, firwin2, freqz
from torch import nn

EPS = 1e-8


# --------------------------------------------------------------- the partition


LEVEL_TRACK, LEVEL_STEM = 0, 1


def sample_targets(
    n_tracks: int,
    track_groups: list[str],
    k_max: int | None = None,
    rng: np.random.Generator | None = None,
    k_bias: float = 1.0,
    level: int | None = None,
    p_stem: float = 0.5,
    p_full: float = 0.15,
) -> tuple[list[list[int]], list[int], bool, int]:
    """Partition a song segment at one hierarchy level, plus a residual.

    Args:
        n_tracks: number of raw tracks N available in this song segment.
        track_groups: length-N list of MoisesDB level-1 category names.
        k_max: optional cap on K.
        rng: numpy Generator; a fresh default one if omitted.
        k_bias: skews K towards larger values; `1.0` is uniform.
        level: `LEVEL_TRACK` (0, one unit per raw track) or `LEVEL_STEM`
            (1, one unit per level-1 family). `None` draws it with
            probability `p_stem` of landing on stem level.
        p_stem: P(level = 1) when `level` is not given.
        p_full: P(every unit becomes a target, so the residual is silent).
            Otherwise K is drawn from `[1, n_units - 1]` and the residual
            is guaranteed to carry content.

    Returns:
        targets:  list of K_active lists of raw-track indices, one per slot.
        residual: raw-track indices summed into the residual slot.
        full_decomposition: True when the residual is silent.
        level: the level actually used; the model is conditioned on it.
    """
    rng = np.random.default_rng() if rng is None else rng
    if n_tracks < 1:
        raise ValueError("need at least one track")
    if level is None:
        level = LEVEL_STEM if rng.random() < p_stem else LEVEL_TRACK

    # ---- units at this level -------------------------------------------------
    if level == LEVEL_STEM:
        by_family: dict[str, list[int]] = {}
        for i, g in enumerate(track_groups):
            by_family.setdefault(g, []).append(i)
        units = list(by_family.values())
    else:
        units = [[i] for i in range(n_tracks)]

    n_units = len(units)

    # The grounding case (full decomposition) is drawn explicitly, rather
    # than left to fall out of the K draw, so `k_bias` and `p_full` control
    # granularity and the full/residual split independently.
    if n_units == 1 or rng.random() < p_full:
        return units, [], True, level

    # Otherwise K < n_units, so the residual always carries real content.
    hi = n_units - 1 if k_max is None else min(n_units - 1, k_max)
    if k_bias == 1.0:
        k = int(rng.integers(1, hi + 1))
    else:
        u = float(rng.random()) ** (1.0 / float(k_bias))
        k = int(np.clip(1 + int(u * hi), 1, hi))

    order = rng.permutation(n_units)
    chosen, rest = order[:k], order[k:]
    targets = [units[i] for i in chosen]
    residual = [i for r in rest for i in units[r]]
    return targets, residual, len(residual) == 0, level


LUFS_FLOOR = -44.0
"""Absolute loudness gate: no track below this ever becomes a slot."""

# Gated BS.1770 loudness minus 20*log10(RMS) bound, used to skip metering
# tracks that RMS alone already places well above or below the floor.
_LUFS_MINUS_RMS_DB = (-8.0, 16.0)


def _lufs_gate(tracks: torch.Tensor, rms: torch.Tensor, floor: float,
               meter) -> torch.Tensor:
    """[N] bool: is each track's gated loudness at or above `floor` LUFS?

    Two-stage: RMS bounds loudness to a known interval, so most tracks are
    decided without metering; only the ambiguous band pays the exact,
    much slower BS.1770 meter.
    """
    db = 20.0 * torch.log10(rms.clamp(min=EPS))
    lo, hi = _LUFS_MINUS_RMS_DB
    sure_pass = db + lo >= floor
    sure_fail = db + hi < floor
    unsure = ~(sure_pass | sure_fail)
    if not bool(unsure.any()):
        return sure_pass
    keep = sure_pass.clone()
    idx = torch.nonzero(unsure, as_tuple=True)[0]
    keep[idx] = meter(tracks[idx].mean(dim=-2)) >= floor
    return keep


def active_tracks(
    tracks: torch.Tensor,
    abs_floor: float = 1e-4,
    rel_floor_db: float = -50.0,
    lufs_floor: float | None = None,
    meter=None,
) -> torch.Tensor:
    """[N, C, T] -> [N] bool, which tracks are actually playing in this crop.

    Must run before K is sampled, so a silent track is never selected as a
    target. Two thresholds: `abs_floor` catches digital silence and dither,
    `rel_floor_db` (relative to the loudest track) catches bleed and reverb
    tails while staying safe for a quietly mastered song.
    """
    rms = tracks.pow(2).mean(dim=(-2, -1)).sqrt()
    loudest = rms.max().clamp(min=EPS)
    keep = (rms > abs_floor) & (rms > loudest * 10.0 ** (rel_floor_db / 20.0))
    # Absolute loudness gate, applied after any gain augmentation.
    if lufs_floor is not None and meter is not None:
        keep &= _lufs_gate(tracks, rms, lufs_floor, meter)
    return keep


RESIDUAL_LABEL = "residual"


class TorchLoudness(nn.Module):
    """ITU-R BS.1770 gated integrated loudness, differentiable and batched.

    Ported from claritychallenge/clarity `clarity/predictor/torch_msbg.py`
    (`torchloudnorm`), trimmed to measurement only -- nothing here
    normalises audio, it only reports LUFS. Two speed optimisations over
    the straight port (single fused FIR kernel, block sums via overlap
    rather than `nn.Unfold`), numerically checked against it.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        kernel_size: int = 1025,
        block_size: float = 0.4,
        overlap: float = 0.75,
        gamma_a: float = -70.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.gamma_a = gamma_a

        # K-weighting, as a pair of FIRs matching pyloudnorm's IIR response.
        b_shelf = np.array([1.53090959, -2.65116903, 1.16916686])
        a_shelf = np.array([1.0, -1.66375011, 0.71265753])
        w, h = freqz(b_shelf, a_shelf, fs=sample_rate)
        freq = np.append(w, sample_rate / 2)
        gain = np.append(np.abs(h), np.abs(h)[-1])
        fir_shelf = firwin2(kernel_size, freq, gain, fs=sample_rate)
        fir_hp = firwin(kernel_size, 38.0, pass_zero="highpass", fs=sample_rate)

        # One kernel for both filters in series (convolution is associative).
        k = np.convolve(fir_shelf, fir_hp)
        self.register_buffer("kernel", torch.tensor(k, dtype=torch.float32))
        # Both FIRs are linear phase, so the series kernel's group delay is
        # exactly half its length.
        self.k_delay = (len(k) - 1) // 2

        self.frame_size = int(block_size * sample_rate)
        self.frame_shift = int(block_size * sample_rate * (1 - overlap))
        self._kfft = None

    def _kweight(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, T], K-weighted, via overlap-save FFT convolution.

        Convolves in short blocks instead of one full-length transform, so
        the cost stays O(T log nfft) with a small, cache-friendly nfft.
        The kernel's own spectrum is cached across calls.
        """
        t, k = x.shape[-1], self.kernel.shape[-1]
        nfft = 1 << max(15, (2 * k - 1).bit_length())
        step = nfft - k + 1
        if self._kfft is None or self._kfft.shape[-1] != nfft // 2 + 1:
            self._kfft = torch.fft.rfft(self.kernel, nfft)
        kf = self._kfft.to(x.device)

        # Pad front by k-1 for history, back to a whole number of steps.
        n_blk = -(-(t + self.k_delay) // step)
        xp = F.pad(x, (k - 1, n_blk * step - t))
        blocks = xp.unfold(-1, nfft, step)                      # [B, n_blk, nfft]
        y = torch.fft.irfft(torch.fft.rfft(blocks, nfft) * kf, nfft)
        # Overlap-save: drop each block's circular-wrap garbage, concatenate.
        y = y[..., k - 1:].reshape(*x.shape[:-1], n_blk * step)
        return y[..., self.k_delay : self.k_delay + t]

    def _block_energy(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, n_blocks] mean square over 75%-overlapping blocks.

        Sums per-hop energy then adds four consecutive hop sums, instead of
        a float64 prefix sum, since each block is exactly four hops long at
        75% overlap.
        """
        hop = self.frame_shift
        per = self.frame_size // hop                # 4, for 0.4 s at 75% overlap
        if self.frame_size == per * hop:
            n_hop = x.shape[-1] // hop
            if n_hop < per:
                return x.pow(2).mean(dim=-1, keepdim=True)
            e = x[..., : n_hop * hop].pow(2).unflatten(-1, (n_hop, hop)).sum(-1)
            e = e.unfold(-1, per, 1).sum(-1) / self.frame_size
            return e

        # Fallback for a block/overlap pair that is not an exact multiple.
        c = F.pad(x.double().pow(2).cumsum(dim=-1), (1, 0))
        n = (x.shape[-1] - self.frame_size) // self.frame_shift + 1
        start = torch.arange(n, device=x.device) * self.frame_shift
        z = (c[:, start + self.frame_size] - c[:, start]) / self.frame_size
        return z.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] mono -> [B] LUFS."""
        z = self._block_energy(self._kweight(x.float()))
        el = -0.691 + 10 * torch.log10(z + EPS)

        # Two-stage gate: absolute at gamma_a, then relative at -10 LU.
        gate_a = (el > self.gamma_a).float()
        z_a = (z * gate_a).sum(1, keepdim=True) / (gate_a.sum(1, keepdim=True) + EPS)
        gamma_r = -0.691 + 10 * torch.log10(z_a + EPS) - 10

        gate = gate_a * (el > gamma_r).float()
        z_ar = (z * gate).sum(1, keepdim=True) / (gate.sum(1, keepdim=True) + EPS)
        return (-0.691 + 10 * torch.log10(z_ar + EPS)).squeeze(1)


# ---------------------------------------------------- the RECURSIVE protocol


def sample_recursive(
    n_tracks: int,
    track_groups: list[str],
    rng: np.random.Generator | None = None,
    level: int | None = None,
    p_stem: float = 0.5,
) -> tuple[list[list[int]], list[int], list[int], int]:
    """One step of a recursive extraction: what is known, what to find, what is left.

    Unlike `sample_targets`, extraction here is sequential and the model
    is told what has already been taken:

        available  C units whose true embeddings are conditioning
        target     one unit, generated
        residual   everything still unclaimed, also generated

    `C ~ Uniform(0, n_units - 1)` covers the whole trajectory in training,
    from C = 0 (conditioned on the mixture alone) to C = n_units - 1
    (empty residual). Levels are as in `sample_targets`.

    Returns:
        available: C unit index-lists (possibly empty)
        target:    one unit, as raw-track indices
        residual:  raw-track indices of everything unclaimed (possibly empty)
        level:     the level used; the model is conditioned on it
    """
    rng = np.random.default_rng() if rng is None else rng
    if n_tracks < 1:
        raise ValueError("need at least one track")
    if level is None:
        level = LEVEL_STEM if rng.random() < p_stem else LEVEL_TRACK

    if level == LEVEL_STEM:
        by_family: dict[str, list[int]] = {}
        for i, g in enumerate(track_groups):
            by_family.setdefault(g, []).append(i)
        units = list(by_family.values())
    else:
        units = [[i] for i in range(n_tracks)]

    n_units = len(units)
    order = rng.permutation(n_units)
    c = int(rng.integers(0, n_units))
    available = [units[i] for i in order[:c]]
    target = units[order[c]]
    residual = [i for r in order[c + 1:] for i in units[r]]
    return available, target, residual, int(level)


def build_recursive_slots(tracks, available, target, residual, track_groups=None):
    """Realise one recursive step as audio: `[C + 2, C, T]`.

    Order is load-bearing: the C available units, then the target, then
    the residual. The last two are generated; the first C are conditioning.
    """
    slots = [tracks[idx].sum(dim=0) for idx in available]
    slots.append(tracks[target].sum(dim=0))
    slots.append(tracks[residual].sum(dim=0) if residual
                 else torch.zeros_like(tracks[0]))
    audio = torch.stack(slots)
    if track_groups is None:
        return audio
    labels = [track_groups[u[0]] for u in available]
    labels += [track_groups[target[0]], RESIDUAL_LABEL]
    return audio, labels


def collate_recursive(batch):
    """Collate recursive steps, which differ in how many units are available.

    Returns:
        audio  [sum_i (C_i + 2), C, T]  packed, no padding
        counts [B]                      C_i + 2 per item
        labels list[str]                packed alongside
        levels [B]                      hierarchy level per item
        n_units [B]                     units the segment decomposes into (N)
    """
    audios = [it[0] for it in batch]
    labels = [x for it in batch for x in it[1]]
    levels = torch.tensor([it[2] for it in batch], dtype=torch.long)
    counts = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    # N is logging-only, and older datasets emit 3-tuples, so it is optional.
    n_units = torch.tensor([it[3] if len(it) > 3 else 0 for it in batch],
                           dtype=torch.long)
    return torch.cat(audios, dim=0), counts, labels, levels, n_units

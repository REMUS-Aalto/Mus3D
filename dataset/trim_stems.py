"""Silence-trim a prepared corpus's stems into a parallel trimmed tree.

Why this exists: `medleydb-stems/trimmed` held byte-identical copies of
`clean/`, and `albumdb-stems/trimmed` was a SYMLINK to `clean/` -- neither was
ever trimmed; this script produced the trees that now sit there. Only MoisesDB
had genuinely trimmed audio (median 67% of the original kept, minimum 3%). The
substitution pool used by the random-mix and
controlled-cacophony loaders samples a track at a random offset with no
alignment to preserve, so an untrimmed pool means crops that land in dead air
and get thrown away -- the loader bottleneck this removes.

Method, matching how the MoisesDB copies were made (ffmpeg's RMS silence
remover, -50 dBFS, 20 ms window):

    frame RMS over 20 ms hops -> keep frames >= -50 dBFS -> pad each kept run
    by `pad_ms` -> bridge gaps shorter than `min_gap_ms` -> concatenate, with a
    short crossfade at each join.

The crossfade is the one addition. ffmpeg concatenates hard, which clicks at
every join; a click is a broadband transient that PaSST hears as percussion, so
it would be a fabricated source in the very embeddings this corpus trains.
Bridging short gaps serves the same end -- it keeps musical rests intact instead
of splicing every eighth-note.

Writes to `--out-dir` (default `trimmed`) and SKIPS files that already exist, so
re-running is cheap and never clobbers. Point it somewhere else if the target
tree is a symlink into the source audio, as AlbumDB's once was -- writing
"into" that would corrupt `clean/`.

    python dataset/trim_stems.py --root <dataset root> \
        --corpus medleydb-stems --corpus albumdb-stems --jobs 32
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

THRESH_DB = -50.0
WIN_MS = 20.0
PAD_MS = 100.0
MIN_GAP_MS = 250.0
FADE_MS = 10.0


def _keep_mask(x: np.ndarray, sr: int) -> np.ndarray:
    """[T, C] -> [n_frames] bool, which 20 ms frames are above the floor."""
    hop = max(1, int(sr * WIN_MS / 1000.0))
    n = len(x) // hop
    if n == 0:
        return np.ones(1, dtype=bool)
    frames = x[: n * hop].reshape(n, hop, -1)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=(1, 2)))
    return rms >= 10.0 ** (THRESH_DB / 20.0)


def _runs(mask: np.ndarray, hop: int, n_samples: int, sr: int):
    """Kept frames -> sample ranges, padded and with short gaps bridged."""
    pad = int(sr * PAD_MS / 1000.0)
    min_gap = int(sr * MIN_GAP_MS / 1000.0)

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    edges = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[edges + 1]])
    ends = np.concatenate([idx[edges], [idx[-1]]])

    spans = []
    for s, e in zip(starts, ends):
        a = max(0, s * hop - pad)
        b = min(n_samples, (e + 1) * hop + pad)
        if spans and a - spans[-1][1] < min_gap:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def trim_one(args) -> tuple[str, int, int]:
    src, dst = Path(args[0]), Path(args[1])
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    hop = max(1, int(sr * WIN_MS / 1000.0))
    spans = _runs(_keep_mask(x, sr), hop, len(x), sr)
    if not spans:
        # Nothing above the floor anywhere. Keep a short head rather than write
        # an empty file: an empty entry in the pool would raise on read, and a
        # track this quiet is dropped by `active_tracks` downstream anyway.
        spans = [(0, min(len(x), sr * 10))]

    fade = int(sr * FADE_MS / 1000.0)
    parts = []
    for i, (a, b) in enumerate(spans):
        seg = x[a:b].copy()
        if fade and len(seg) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
            if i > 0:
                seg[:fade] *= ramp
            if i < len(spans) - 1:
                seg[-fade:] *= ramp[::-1]
        parts.append(seg)
    y = np.concatenate(parts, axis=0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, format="FLAC")
    return str(src), len(x), len(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--out-dir", default="trimmed")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()

    jobs = []
    for corp in args.corpus:
        base = args.root / corp
        for sub in ("clean", "bleeding"):
            d = base / sub
            if not d.is_dir() or d.is_symlink():
                continue
            for leaf in sorted(d.iterdir()):
                if not leaf.is_dir():
                    continue
                for f in sorted(leaf.iterdir()):
                    if f.suffix.lower() in (".wav", ".flac"):
                        dst = base / args.out_dir / leaf.name / (f.stem + ".flac")
                        if not dst.exists():
                            jobs.append((str(f), str(dst)))
    print(f"{len(jobs)} files to trim, {args.jobs} workers")

    kept_in = kept_out = 0
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, (name, n_in, n_out) in enumerate(ex.map(trim_one, jobs, chunksize=4), 1):
            kept_in += n_in
            kept_out += n_out
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  kept {kept_out / max(kept_in, 1):.1%} so far",
                      flush=True)
    if jobs:
        print(f"done: kept {kept_out / max(kept_in, 1):.1%} of total duration")


if __name__ == "__main__":
    main()

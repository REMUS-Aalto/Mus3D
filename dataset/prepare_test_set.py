"""Build the MuS3D test set from MoisesDB: 10 s segments, written to disk as audio.

The test set is the MoisesDB test split (Banquet's fold 5, 48 songs) cut into
non-overlapping 10 s segments (the final partial segment is dropped). For every
segment it holds the mixture and one target per source, at two levels:

    <out>/manifest.json         <out>/audio/<song>/seg<NNN>/{mixture,target_<group>}.wav
    <out>/manifest_track.json   <out>/audio_track/<song>/seg<NNN>/{mixture,target_<leaf>}.wav

`audio` targets the 11 MoisesDB instrument groups, `audio_track` the individual
instruments (tracks of the same instrument are summed, and so is the whole
vocals family, which stays a single `vocals` target at both levels). The mixture
is the sum of the song's tracks, not MoisesDB's distributed mixture. Everything
is 44.1 kHz stereo float32 WAV.

Nothing here is drawn at random. The songs and their tracks come from
`dataset/splits/moisesdb-test.json`, and the two manifests are the released ones
(`results_paper/discovery/ground_truth/`): they are copied next to the audio, and
every segment written is checked against the levels they record, so a build
that disagrees with the released test set fails instead of passing silently.

    uv run python dataset/prepare_test_set.py --root <dir holding MoisesDB/> --out <test_set>

`--root` is the dataset root the training configs call `dset_dir`; it must hold
`MoisesDB/moisesdb_v0.1/<song>/`. The result is about 49 GB. Re-running skips
the songs already written. Point `MUS3D_TEST_SET` at `<out>` for the scripts in
`evaluation/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data.moisesdb import LEAF_TO_GROUP  # noqa: E402

SPLIT = REPO / "dataset/splits/moisesdb-test.json"
RESULTS = Path(os.environ.get("RESULTS_PAPER", REPO / "results_paper"))
SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 10 * SAMPLE_RATE
#: level -> (manifest, audio directory)
LEVELS = {"stem": ("manifest.json", "audio"),
          "track": ("manifest_track.json", "audio_track")}
#: largest accepted difference, in dB, from a level recorded in the manifest
TOL_DB = 1e-3


def slug(name: str) -> str:
    return name.replace(" ", "_")


def _dbrms(x: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.square(x)) + 1e-12))


def _resolve(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    parts = tuple(x for x in Path(rel).parts if x != ".")
    for cand in ("MoisesDB", "moisesdb"):
        q = root.joinpath(cand, *parts[1:])
        if q.exists():
            return q
    raise FileNotFoundError(p)


def _target(label: str, level: str) -> str:
    leaf = slug(label)
    group = LEAF_TO_GROUP[leaf]
    return "vocals" if group == "vocals" else leaf if level == "track" else group


def load_song(root: Path, sources: list) -> dict[str, tuple[np.ndarray, dict]]:
    """-> {level: (mixture [2, T], {target: [2, T]})}, T a whole number of segments.

    The order of the sums is that of the original build (tracks in split order
    into their target, targets in order of first appearance into the mixture),
    which makes every sample identical to the original test set.
    """
    acc: dict[str, dict[str, np.ndarray]] = {level: {} for level in LEVELS}
    for label, rel in sources:
        path = _resolve(root, rel)
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{path}: {sr} Hz, expected {SAMPLE_RATE}")
        audio = audio.T
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        audio = audio[:2]
        for level, stems in acc.items():
            key = _target(label, level)
            cur = stems.get(key)
            if cur is None or audio.shape[1] > cur.shape[1]:
                grown = np.zeros((2, audio.shape[1]), dtype=np.float32)
                if cur is not None:
                    grown[:, : cur.shape[1]] = cur
                cur = stems[key] = grown
            cur[:, : audio.shape[1]] += audio

    out = {}
    for level, stems in acc.items():
        total = max(x.shape[1] for x in stems.values()) // SEGMENT_SAMPLES * SEGMENT_SAMPLES
        mixture = np.zeros((2, total), dtype=np.float32)
        for key, x in stems.items():
            k = min(x.shape[1], total)
            mixture[:, :k] += x[:, :k]
            padded = np.zeros((2, total), dtype=np.float32)
            padded[:, :k] = x[:, :k]
            stems[key] = padded
        out[level] = (mixture, stems)
    return out


def _song(job) -> tuple[str, int, list[str]]:
    """Write one song at both levels -> (song, files written, mismatches)."""
    song, sources, root, out, expected = job
    todo = {level: items for level, items in expected.items()
            if not all((out / rel).exists() for rel in _files(items))}
    if not todo:
        return song, 0, []

    written, bad = 0, []
    built = load_song(root, sources)
    for level, items in todo.items():
        mixture, stems = built[level]
        want = {(it["stem"], it["segment_index"]): it for it in items}
        have = {(k, i) for k in stems for i in range(mixture.shape[1] // SEGMENT_SAMPLES)}
        if have != set(want):
            bad.append(f"{song} [{level}]: built {len(have)} targets, the manifest "
                       f"lists {len(want)}")
            continue
        for (key, i), it in sorted(want.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            sl = slice(i * SEGMENT_SAMPLES, (i + 1) * SEGMENT_SAMPLES)
            for rel, x, ref in ((it["mixture_path"], mixture[:, sl], it["mixture_dbrms"]),
                                (it["target_path"], stems[key][:, sl], it["target_dbrms"])):
                if abs(_dbrms(x) - ref) > TOL_DB:
                    bad.append(f"{rel}: {_dbrms(x):.4f} dB, the manifest records {ref:.4f}")
                dst = out / rel
                if dst.name == "mixture.wav" and dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                # written under a temporary name, so an interrupted run never
                # leaves a truncated file that the next run would skip
                tmp = dst.with_name(dst.name + ".part")
                sf.write(tmp, x.T, SAMPLE_RATE, format="WAV", subtype="FLOAT")
                tmp.replace(dst)
                written += 1
    return song, written, bad


def _files(items: list[dict]) -> set[str]:
    return {it["mixture_path"] for it in items} | {it["target_path"] for it in items}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="dataset root holding MoisesDB/moisesdb_v0.1/")
    ap.add_argument("--out", required=True, type=Path, help="test set directory to write")
    ap.add_argument("--manifests", type=Path, default=RESULTS / "discovery/ground_truth",
                    help="directory of the released manifest.json and manifest_track.json "
                         "(default: $RESULTS_PAPER/discovery/ground_truth)")
    ap.add_argument("--songs", nargs="*", default=None,
                    help="build only these song ids (default: all 48)")
    ap.add_argument("--workers", type=int, default=8,
                    help="songs built in parallel; each holds one song in memory")
    args = ap.parse_args()

    info = json.load(open(SPLIT))["song_info"]
    expected: dict[str, dict[str, list]] = {song: {} for song in info}
    for level, (manifest, _) in LEVELS.items():
        path = args.manifests / manifest
        if not path.exists():
            raise SystemExit(f"missing {path}: --manifests must point at the released "
                             "results_paper/discovery/ground_truth")
        for it in json.load(open(path))["items"]:
            expected[it["song_id"]].setdefault(level, []).append(it)
    songs = sorted(info) if args.songs is None else args.songs
    unknown = [s for s in songs if s not in info]
    if unknown:
        raise SystemExit(f"not in the test split: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, info[s]["sources"], args.root, args.out, expected[s]) for s in songs]
    written, bad = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _song_id, n, errs in tqdm(ex.map(_song, jobs), total=len(jobs), desc="songs"):
            written += n
            bad.extend(errs)
    if bad:
        print("\n".join(bad[:20]))
        raise SystemExit(f"{len(bad)} mismatches with the released manifests: the "
                         "MoisesDB copy under --root differs from the one the test "
                         "set was built from")
    for manifest, _ in LEVELS.values():
        shutil.copyfile(args.manifests / manifest, args.out / manifest)
    n_seg = len({(s, it["segment_index"]) for s in songs for it in expected[s]["stem"]})
    print(f"{len(songs)} songs, {n_seg} segments, {written} files written -> {args.out}")


if __name__ == "__main__":
    main()

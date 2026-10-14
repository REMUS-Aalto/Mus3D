"""Silence-trim the MoisesDB tracks of the substitution pool.

Writes `<root>/moisesdb-stems/trimmed/<leaf>/<track id>.flac` for the train and
validation tracks listed in `dataset/splits/moisesdb-trimmed.txt`.
`make_mss_dataset_info.py` then points the pool (`stem_paths`) at these copies
and leaves every other track on its full-length file, which gives the pool the
paper's models were trained with.

The trimming is that of `trim_stems.py` (frames below -50 dBFS removed, short
gaps kept). The copies the paper used were cut with ffmpeg's silence remover at
the same threshold, so these are equivalent, not bit-identical.

    python dataset/trim_moisesdb_pool.py --root <dataset root> --jobs 32
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trim_stems import trim_one  # noqa: E402

SPLITS = Path(__file__).resolve().parent / "splits"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()

    ids = set((SPLITS / "moisesdb-trimmed.txt").read_text().split())
    jobs = []
    for split in ("train", "val"):
        info = json.loads((SPLITS / f"moisesdb-{split}.json").read_text())["song_info"]
        for song in info.values():
            for label, rel in song["sources"]:
                src = args.root / rel
                if src.stem not in ids:
                    continue
                if not src.exists():
                    raise SystemExit(f"missing {src}")
                dst = (args.root / "moisesdb-stems/trimmed" / label.replace(" ", "_")
                       / (src.stem + ".flac"))
                if not dst.exists():
                    jobs.append((str(src), str(dst)))
    print(f"{len(jobs)} of {len(ids)} tracks to trim, {args.jobs} workers")

    kept_in = kept_out = 0
    with ProcessPoolExecutor(args.jobs) as ex:
        for _name, n_in, n_out in ex.map(trim_one, jobs, chunksize=4):
            kept_in += n_in
            kept_out += n_out
    if kept_in:
        print(f"kept {100 * kept_out / kept_in:.0f}% of the audio")


if __name__ == "__main__":
    main()

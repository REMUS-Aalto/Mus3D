"""Inventory every MedleyDB song in ``MedleyDB/all_v1v2``.

Emits ``dataset/medleydb_songs.csv`` and prints a summary. Per song: genre,
``has_bleed``, and the number of stems, all read from the song's
``*_METADATA.yaml``.

MUSDB25 draws 46 of its 149 songs straight from MedleyDB, keeping the MedleyDB
song id verbatim (``AClassicEducation_NightOwl``), so the overlap is an exact
id intersection -- no name normalisation needed.  That overlap is recorded in
the ``in_musdb25`` column and nothing is filtered on it: MUSDB25 is no longer
used, so its 46 songs are ordinary MedleyDB songs like any other.  The column
survives because it is the only record of which songs a MUSDB25-trained
baseline would have seen.

A copy of MedleyDB may still be in flight, so songs are reported
as ``complete`` only when the metadata, the stem dir, and one file per stem
named in the metadata are all present; incomplete ones are listed separately
rather than silently counted.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import yaml

#: dataset root (the configs' `dset_dir`); override with DATASET_ROOT
ROOT = Path(os.environ.get("DATASET_ROOT", "data"))
MEDLEYDB = ROOT / "MedleyDB/all_v1v2"
MUSDB25 = ROOT / "musdb25/musdb25"
OUT_CSV = Path(__file__).resolve().parent / "medleydb_songs.csv"

FIELDS = [
    "song_id", "artist", "album", "genre", "has_bleed", "n_stems",
    "instrumental", "excerpt", "origin", "in_musdb25", "complete", "n_stem_files",
]


def musdb25_ids() -> set[str]:
    if not MUSDB25.is_dir():
        return set()
    return {d.name for d in MUSDB25.iterdir() if d.is_dir() and not d.name.startswith(".")}


def medleydb_ids() -> list[str]:
    return sorted(d.name for d in MEDLEYDB.iterdir() if d.is_dir() and not d.name.startswith("."))


def describe(song_id: str, in_musdb25: bool = False) -> dict | None:
    """-> a row, or None if the metadata has not been copied across yet."""
    song = MEDLEYDB / song_id
    meta_path = song / f"{song_id}_METADATA.yaml"
    if not meta_path.exists():
        return None
    meta = yaml.safe_load(meta_path.read_text())

    stems = meta.get("stems") or {}
    stem_dir = song / (meta.get("stem_dir") or f"{song_id}_STEMS")
    present = sum(
        1 for s in stems.values()
        if s.get("filename") and (stem_dir / s["filename"]).exists()
    )

    return {
        "song_id": song_id,
        "artist": meta.get("artist", ""),
        "album": meta.get("album", ""),
        "genre": meta.get("genre", ""),
        "has_bleed": meta.get("has_bleed", ""),
        "n_stems": len(stems),
        "instrumental": meta.get("instrumental", ""),
        "excerpt": meta.get("excerpt", ""),
        "origin": meta.get("origin", ""),
        "in_musdb25": "yes" if in_musdb25 else "no",
        "complete": "yes" if stems and present == len(stems) else "no",
        "n_stem_files": present,
    }


def main() -> None:
    in_musdb = musdb25_ids()
    all_ids = medleydb_ids()

    print(f"MedleyDB dirs on disk: {len(all_ids)}")
    print(f"  also in MUSDB25:     {sum(1 for i in all_ids if i in in_musdb)} "
          f"(recorded, not filtered)")

    rows, no_meta = [], []
    for song_id in all_ids:
        row = describe(song_id, song_id in in_musdb)
        (rows.append(row) if row else no_meta.append(song_id))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    complete = [r for r in rows if r["complete"] == "yes"]
    partial = [r for r in rows if r["complete"] == "no"]

    print(f"\nwith metadata: {len(rows)}  (complete: {len(complete)}, "
          f"partial: {len(partial)}, metadata not copied yet: {len(no_meta)})")
    if complete:
        print(f"total stems in complete songs: {sum(r['n_stems'] for r in complete)}")
        bleed = sum(1 for r in complete if str(r["has_bleed"]).lower() == "yes")
        print(f"bleeding: {bleed} yes / {len(complete) - bleed} no")

        by_genre: dict[str, int] = {}
        for r in complete:
            by_genre[r["genre"]] = by_genre.get(r["genre"], 0) + 1
        print("\ngenres:")
        for g, n in sorted(by_genre.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:4d}  {g}")

        print(f"\n{'song_id':52s} {'genre':22s} {'bleed':6s} stems")
        for r in sorted(complete, key=lambda r: r["song_id"]):
            print(f"{r['song_id']:52s} {r['genre']:22s} "
                  f"{str(r['has_bleed']):6s} {r['n_stems']:5d}")

    if partial:
        print(f"\nincomplete (stem files still copying): {len(partial)}")
        for r in partial:
            print(f"  {r['song_id']} ({r['n_stem_files']}/{r['n_stems']} stems)")
    if no_meta:
        print(f"\nno metadata yet: {len(no_meta)}")
        for i in no_meta:
            print(f"  {i}")

    print("\nwrote", OUT_CSV)


if __name__ == "__main__":
    main()

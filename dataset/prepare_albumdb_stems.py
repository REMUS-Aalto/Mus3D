"""Build a MoisesDB-style multitrack dataset out of AlbumDB.

Output layout (mirrors ``moisesdb-stems`` / ``musdb25-stems`` so
``src.data.moisesdb.MoisesDBDataset`` conventions carry over)::

    albumdb-stems/
        clean/<moisesdb_leaf>/<SongId>.wav      # the tracks
        trimmed -> clean                        # symlink; no silence trimming
        mixtures/<SongId>.wav                   # == sum of the tracks
        songs.json                              # {SongId: {sources, artist, ...}}

There is NO train/test split: AlbumDB is 10 songs from a single album and is
used whole.

Processing, per song:
  * every ``*.wav`` in the song dir is one stereo track; its MoisesDB leaf comes
    from the file name via ``src.data.albumdb_taxonomy``;
  * tracks sharing a leaf are summed, so a song has at most one file per leaf.
    This is also what merges AlbumDB's channel-position and section splits --
    ``vox_back_l`` + ``vox_back_r`` + ``vox_back_c`` -> one ``background_vocals``;
  * if any sample of any track exceeds [-1, 1], ONE shared scalar gain is
    applied to every track of that song, preserving ``sum(tracks) == mixture``.

Resumable: a song whose entry is already in ``songs.json`` is skipped.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from albumdb_taxonomy import LABEL_TO_LEAF, leaf_for  # noqa: E402

#: dataset root (the configs' `dset_dir`); override with DATASET_ROOT
ROOT = Path(os.environ.get("DATASET_ROOT", "data"))
SRC = ROOT / "AlbumDB/stems_mixed"
OUT = ROOT / "albumdb-stems"
SAMPLE_RATE = 44100
SUBTYPE = "PCM_24"  # matches moisesdb-stems
ARTIST = "AlbumDB"


def song_dirs() -> list[Path]:
    return sorted(d for d in SRC.iterdir() if d.is_dir() and not d.name.startswith("."))


def song_id(dir_name: str) -> str:
    """'10 It Ain't Easy (To Mend A Heart)' -> '10_It_Aint_Easy_To_Mend_A_Heart'."""
    s = re.sub(r"[^A-Za-z0-9 ]", "", dir_name)
    return re.sub(r"\s+", "_", s.strip())


def song_title(dir_name: str) -> str:
    """Drop the leading track number: '01 Extraordinary' -> 'Extraordinary'."""
    return re.sub(r"^\d+\s+", "", dir_name).strip()


def check_coverage(songs: list[Path]) -> None:
    """Fail before any writing if a stem name is not in the table."""
    names = {p.stem.lower() for s in songs for p in s.glob("*.wav")}
    missing = sorted(names - set(LABEL_TO_LEAF))
    if missing:
        raise RuntimeError(f"stem names absent from albumdb_taxonomy: {missing}")
    unused = sorted(set(LABEL_TO_LEAF) - names)
    if unused:
        print(f"[note] {len(unused)} table entries unused: {unused}")


def convert_song(song_dir: Path) -> dict:
    sid = song_id(song_dir.name)
    tracks = sorted(song_dir.glob("*.wav"))

    # sum tracks that land on the same MoisesDB leaf
    by_leaf: dict[str, np.ndarray] = {}
    used_labels: dict[str, int] = defaultdict(int)
    for path in tracks:
        leaf = leaf_for(path.stem)
        used_labels[path.stem.lower()] += 1
        audio, sr = sf.read(path, dtype="float64", always_2d=True)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{path}: {sr} Hz, expected {SAMPLE_RATE}")
        audio = audio.T  # -> [C, T]
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        if leaf in by_leaf:
            n = min(by_leaf[leaf].shape[-1], audio.shape[-1])
            by_leaf[leaf] = by_leaf[leaf][:, :n] + audio[:, :n]
        else:
            by_leaf[leaf] = audio

    n = min(a.shape[-1] for a in by_leaf.values())
    by_leaf = {k: v[:, :n] for k, v in by_leaf.items()}
    mixture = sum(by_leaf.values())

    # One shared gain for the whole song, so sum(tracks) == mixture survives.
    # The MIXTURE is what must fit in [-1, 1]; the stems ride the same scalar.
    # An individual stem peaking above the mixture is possible in principle
    # (phase cancellation between tracks), and would clip in the PCM_24 file,
    # so it is asserted against rather than silently allowed.
    peak = float(np.abs(mixture).max())
    gain = 1.0 if peak <= 1.0 else 0.999 / peak
    track_peak = max(np.abs(a).max() for a in by_leaf.values()) * gain
    if track_peak > 1.0:
        raise RuntimeError(f"{sid}: stem peaks at {track_peak:.3f} after gain")
    if gain != 1.0:
        by_leaf = {k: v * gain for k, v in by_leaf.items()}
        mixture = mixture * gain

    sources = []
    for leaf, audio in sorted(by_leaf.items()):
        path = OUT / "clean" / leaf / f"{sid}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio.T.astype(np.float32), SAMPLE_RATE, subtype=SUBTYPE)
        sources.append(f"clean/{leaf}/{sid}")
    mix_path = OUT / "mixtures" / f"{sid}.wav"
    mix_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mix_path, mixture.T.astype(np.float32), SAMPLE_RATE, subtype=SUBTYPE)

    return {
        "song_id": sid,
        "entry": {
            "sources": sources,
            "artist": ARTIST,
            "name": song_title(song_dir.name),
            "mixture": f"mixtures/{sid}",
            "gain": gain,
            "n_frames": int(n),
            "n_raw_tracks": len(tracks),
            "labels": dict(sorted(used_labels.items())),
        },
    }


def main() -> None:
    songs = song_dirs()
    check_coverage(songs)
    print(f"{len(songs)} songs")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "songs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    todo = [s for s in songs if song_id(s.name) not in manifest]
    print(f"{len(todo)} to do")

    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, res in enumerate(pool.map(convert_song, todo), 1):
            manifest[res["song_id"]] = res["entry"]
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            print(f"  {i}/{len(todo)} {res['song_id']} "
                  f"({len(res['entry']['sources'])} tracks, gain={res['entry']['gain']:.3f})")

    # MoisesDBDataset reads its track pool from `trimmed/`; we do not trim.
    link = OUT / "trimmed"
    if not link.exists():
        link.symlink_to("clean")

    print("done ->", OUT)


if __name__ == "__main__":
    main()

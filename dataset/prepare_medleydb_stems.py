"""Build a MoisesDB-style multitrack dataset out of MedleyDB.

Output layout::

    medleydb-stems/
        clean/<moisesdb_leaf>/<SongId>.wav      # songs with has_bleed == no
        bleeding/<moisesdb_leaf>/<SongId>.wav   # songs with has_bleed == yes
        trimmed/<moisesdb_leaf>/<SongId>.wav    # symlinks to both of the above
        mixtures/<SongId>.wav                   # == sum of the tracks
        songs.json                              # {SongId: {sources, ...}}

``clean`` / ``bleeding`` mirror ``moisesdb-stems``, and ``songs.json`` sources
carry that prefix so a consumer can select on it.  ``trimmed/`` is the union
(symlinks, no copies) because ``MoisesDBDataset`` reads its track pool from
there; nothing is silence-trimmed.

Song selection (see ``list_medleydb_songs.py`` for the full inventory):
  * genre is not Classical or Musical Theatre;
  * not a ``MusicDelta_Chinese*`` excerpt;
  * at least ``MIN_TRACKS`` leaf tracks survive the taxonomy.

MUSDB25 overlap is NOT a criterion. It was, while musdb25-stems was in use and
the 46 shared songs would have been duplicated; MUSDB25 has since been dropped
entirely, so those songs are ordinary MedleyDB songs and are included. The
inventory still carries an ``in_musdb25`` column and it is copied into each
manifest entry, so the overlap remains queryable.

Processing, per song:
  * every entry in the metadata's ``stems`` is one track, labelled by
    ``dataset/medleydb_taxonomy.py``; tracks sharing a leaf are summed, so a
    song has at most one file per leaf;
  * ``main system`` tracks (the room/PA capture of the whole band) are DROPPED
    -- see the note on the mixture below;
  * one shared scalar gain per song keeps the mixture inside [-1, 1].

On "the stems add up to the mixture"
------------------------------------
The mixture written here is the SUM OF THE TRACKS, so the identity holds
exactly by construction (verified to ~1e-6 by ``--verify``).

It is *not* MedleyDB's own ``*_MIX.wav``, and no choice of gains would make it
so.  Fitting a per-stem least-squares gain vector against the official mix was
measured on all 93 songs: it leaves a median residual of only 12.6 dB SNR, and
in 26 songs the best fit assigns NEGATIVE gains to some stems -- the residual is
mastering (compression, EQ, limiting) applied to the mix after the stems were
summed, not a gain mismatch, and no linear rescaling can undo it. Forcing a fit
would corrupt the stems to chase a target that is not reachable.

Each song's entry therefore records ``medleydb_mix_snr_db``: how well our
mixture matches the official one after the best single positive gain, together
with that gain (``medleydb_mix_gain``). Songs where the official mix was heavily
mastered show up as a low SNR and can be filtered downstream.

Resumable: a song already in ``songs.json`` is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from medleydb_taxonomy import NON_SOURCE, leaf_for  # noqa: E402

#: dataset root (the configs' `dset_dir`); override with DATASET_ROOT
ROOT = Path(os.environ.get("DATASET_ROOT", "data"))
MEDLEYDB = ROOT / "MedleyDB/all_v1v2"
OUT = ROOT / "medleydb-stems"
INVENTORY = Path(__file__).resolve().parent / "medleydb_songs.csv"

SAMPLE_RATE = 44100
SUBTYPE = "PCM_24"  # matches moisesdb-stems
SKIP_GENRES = {"Classical", "Musical Theatre"}
SKIP_PREFIXES = ("MusicDelta_Chinese",)

#: A song whose ``main system`` tracks carry more than this fraction of the
#: total energy is dropped entirely. Removing the room mic from such a song
#: leaves only near-silent spot mics -- there is no usable multitrack there.
MAIN_SYSTEM_MAX_ENERGY = 0.5

#: Songs with fewer than this many LEAF tracks are discarded. The count is of
#: distinct MoisesDB leaves after same-leaf stems are summed, not of MedleyDB
#: stems -- a 5-stem song whose stems all collapse to `full_acoustic_drumkit`
#: is a one-source song as far as this dataset is concerned.
MIN_TRACKS = 3

#: Skip reasons that can only be discovered by reading the audio. Anything else
#: in skipped.json is derived from the selection rules and is recomputed.
RUNTIME_SKIP_PREFIXES = ("main system", "only ")

#: Windows used to score our mixture against MedleyDB's official mix.
PROBE_WINDOWS, PROBE_SECONDS = 6, 10


def partition_songs() -> tuple[list[dict], dict[str, str]]:
    """(rows to convert, {song_id: why it was excluded}).

    Every MedleyDB song on disk lands in exactly one of the two, so
    ``len(selected) + len(excluded)`` is the full corpus and ``skipped.json``
    accounts for everything that is not in ``songs.json`` -- including the
    songs ruled out before any audio is read (Classical, Musical Theatre,
    ``MusicDelta_Chinese*``), not just the ones that fail during conversion.
    """
    if not INVENTORY.exists():
        raise SystemExit(f"missing {INVENTORY}; run dataset/list_medleydb_songs.py first")
    on_disk = sorted(d.name for d in MEDLEYDB.iterdir() if d.is_dir() and not d.name.startswith("."))
    by_id = {r["song_id"]: r for r in csv.DictReader(INVENTORY.open())}

    selected, excluded = [], {}
    for sid in on_disk:
        row = by_id.get(sid)
        if row is None:
            excluded[sid] = "not in the inventory CSV (re-run list_medleydb_songs.py)"
        elif sid.startswith(SKIP_PREFIXES):
            excluded[sid] = "out of scope: MusicDelta_Chinese excerpt"
        elif row["genre"] in SKIP_GENRES:
            excluded[sid] = f"out of scope: genre {row['genre']}"
        else:
            selected.append(row)
    return selected, excluded


def mix_agreement(song: Path, meta: dict, mixture: np.ndarray) -> tuple[float, float]:
    """(SNR in dB, best gain) of our mixture against MedleyDB's official mix.

    Scored on `PROBE_WINDOWS` windows rather than the whole song: the number is
    a diagnostic, and reading a 14-minute mix in full for it is not worth the
    IO. The gain is the least-squares scalar, so the SNR reflects the shape of
    the disagreement rather than a level offset.
    """
    mix_path = song / meta["mix_filename"]
    if not mix_path.exists():
        return float("nan"), float("nan")
    n = min(sf.info(mix_path).frames, mixture.shape[-1])
    w = PROBE_SECONDS * SAMPLE_RATE
    ours, theirs = [], []
    for start in np.linspace(0, max(0, n - w), PROBE_WINDOWS).astype(int):
        k = int(min(w, n - start))
        theirs.append(sf.read(mix_path, frames=k, start=int(start),
                              dtype="float64", always_2d=True)[0].ravel())
        ours.append(mixture[:, start:start + k].T.ravel())
    a = np.concatenate(ours)
    b = np.concatenate(theirs)
    denom = float((a * a).sum())
    if denom <= 0:
        return float("nan"), float("nan")
    gain = float((a @ b) / denom)
    resid = float(((gain * a - b) ** 2).sum())
    snr = 10 * np.log10(float((b * b).sum()) / resid) if resid > 0 else float("inf")
    return snr, gain


def convert_song(row: dict) -> dict:
    sid = row["song_id"]
    bleed = str(row["has_bleed"]).lower() == "yes"
    group = "bleeding" if bleed else "clean"

    song = MEDLEYDB / sid
    meta = yaml.safe_load((song / f"{sid}_METADATA.yaml").read_text())
    stem_dir = song / meta["stem_dir"]

    # ---- read every stem once, at float32; 14-minute songs make float64
    # accumulators for a dozen leaves cost more RAM than this is worth.
    tracks, dropped = [], []
    for entry in meta["stems"].values():
        label = (entry.get("instrument") or "").strip().lower()
        path = stem_dir / entry["filename"]
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{path}: {sr} Hz, expected {SAMPLE_RATE}")
        audio = audio.T
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        (dropped if label in NON_SOURCE else tracks).append((label, audio))

    # ---- room mics: drop them, or drop the song if they ARE the song
    energy = lambda xs: sum(float(np.square(a, dtype=np.float64).sum()) for _, a in xs)  # noqa: E731
    total = energy(tracks) + energy(dropped)
    ms_frac = (energy(dropped) / total) if total > 0 else 0.0
    if dropped and ms_frac > MAIN_SYSTEM_MAX_ENERGY:
        return {"song_id": sid, "skipped": f"main system is {ms_frac:.0%} of energy"}
    if not tracks:
        return {"song_id": sid, "skipped": "no usable stems"}

    # ---- sum tracks that land on the same MoisesDB leaf
    by_leaf: dict[str, np.ndarray] = {}
    used_labels: dict[str, int] = defaultdict(int)
    for label, audio in tracks:
        leaf = leaf_for(label)
        used_labels[label] += 1
        if leaf in by_leaf:
            n = min(by_leaf[leaf].shape[-1], audio.shape[-1])
            by_leaf[leaf] = by_leaf[leaf][:, :n] + audio[:, :n]
        else:
            by_leaf[leaf] = audio

    if len(by_leaf) < MIN_TRACKS:
        return {"song_id": sid,
                "skipped": f"only {len(by_leaf)} leaf tracks (min {MIN_TRACKS})"}

    n = min(a.shape[-1] for a in by_leaf.values())
    by_leaf = {k: v[:, :n] for k, v in by_leaf.items()}
    mixture = sum(by_leaf.values())

    # ---- one shared gain, so sum(tracks) == mixture survives normalisation
    peak = float(np.abs(mixture).max())
    gain = 1.0 if peak <= 1.0 else 0.999 / peak
    if gain != 1.0:
        by_leaf = {k: v * gain for k, v in by_leaf.items()}
        mixture = mixture * gain
    track_peak = max(float(np.abs(a).max()) for a in by_leaf.values())
    if track_peak > 1.0:
        raise RuntimeError(f"{sid}: stem peaks at {track_peak:.3f} after gain")

    snr, mix_gain = mix_agreement(song, meta, mixture)

    sources = []
    for leaf, audio in sorted(by_leaf.items()):
        path = OUT / group / leaf / f"{sid}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio.T, SAMPLE_RATE, subtype=SUBTYPE)
        sources.append(f"{group}/{leaf}/{sid}")
    mix_path = OUT / "mixtures" / f"{sid}.wav"
    mix_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mix_path, mixture.T, SAMPLE_RATE, subtype=SUBTYPE)

    artist, _, title = sid.partition("_")
    return {
        "song_id": sid,
        "entry": {
            "sources": sources,
            "artist": row.get("artist") or artist,
            "name": title,
            "album": row.get("album", ""),
            "genre": row.get("genre", ""),
            "in_musdb25": row.get("in_musdb25", "") == "yes",
            "bleeding": bleed,
            "instrumental": row.get("instrumental", ""),
            "excerpt": row.get("excerpt", ""),
            "mixture": f"mixtures/{sid}",
            "gain": gain,
            "n_frames": int(n),
            "n_raw_tracks": len(tracks),
            "dropped_main_system": len(dropped),
            "main_system_energy_frac": round(ms_frac, 4),
            "medleydb_mix_snr_db": None if np.isnan(snr) else round(float(snr), 2),
            "medleydb_mix_gain": None if np.isnan(mix_gain) else round(float(mix_gain), 4),
            "labels": dict(sorted(used_labels.items())),
        },
    }


def prune_undersized(manifest: dict, skips: dict[str, str]) -> int:
    """Delete songs already on disk that fall below ``MIN_TRACKS``.

    The threshold is applied to what was written, not only to what is about to
    be written, so raising ``MIN_TRACKS`` later cleans up an existing dataset
    instead of leaving stale audio that nothing references. Files are removed
    rather than orphaned, and the song moves into ``skipped.json`` with the same
    reason a fresh conversion would have given it.
    """
    doomed = [sid for sid, e in manifest.items() if len(e["sources"]) < MIN_TRACKS]
    for sid in doomed:
        entry = manifest.pop(sid)
        for src in entry["sources"]:
            leaf = Path(src).parent.name
            (OUT / f"{src}.wav").unlink(missing_ok=True)
            (OUT / "trimmed" / leaf / f"{sid}.wav").unlink(missing_ok=True)
        (OUT / f"{entry['mixture']}.wav").unlink(missing_ok=True)
        skips[sid] = f"only {len(entry['sources'])} leaf tracks (min {MIN_TRACKS})"
        print(f"  pruned {sid} ({len(entry['sources'])} tracks)")

    # A leaf that only ever held pruned songs would linger as an empty dir and
    # show up in `MoisesDBDataset.stem_names` as a family with no audio.
    for d in sorted(OUT.glob("*/*")):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return len(doomed)


def write_nobleeding(manifest: dict) -> Path:
    """``songs_nobleeding.json``: the same manifest, clean songs only.

    Same schema as ``songs.json`` and same relative source paths, so it is a
    drop-in for anything that reads the manifest -- point a dataset at this file
    to train on the bleed-free subset without touching the audio on disk. Note
    that ``trimmed/`` still links BOTH groups, so a consumer that samples tracks
    from the directory tree rather than from the manifest is unaffected by it.
    """
    clean = {k: v for k, v in manifest.items() if not v["bleeding"]}
    path = OUT / "songs_nobleeding.json"
    path.write_text(json.dumps(clean, indent=1, sort_keys=True))
    print(f"songs_nobleeding.json: {len(clean)} clean songs, "
          f"{sum(len(v['sources']) for v in clean.values())} tracks")
    return path


def link_trimmed() -> None:
    """``trimmed/<leaf>/`` = the union of clean/ and bleeding/, as symlinks."""
    for group in ("clean", "bleeding"):
        for leaf_dir in sorted((OUT / group).glob("*")):
            if not leaf_dir.is_dir():
                continue
            dest = OUT / "trimmed" / leaf_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in leaf_dir.glob("*.wav"):
                link = dest / f.name
                if not link.exists():
                    link.symlink_to(Path("../..") / group / leaf_dir.name / f.name)


def verify() -> None:
    """Re-read what was written and check sum(sources) == mixture."""
    manifest = json.loads((OUT / "songs.json").read_text())
    worst = 0.0
    for sid, e in manifest.items():
        total = sum(sf.read(OUT / f"{s}.wav", dtype="float64")[0] for s in e["sources"])
        mix = sf.read(OUT / f"{e['mixture']}.wav", dtype="float64")[0]
        err = float(np.abs(total - mix).max())
        worst = max(worst, err)
        if err > 1e-4:
            print(f"  [FAIL] {sid}: max|sum-mix| = {err:.2e}")
    print(f"verify: {len(manifest)} songs, worst max|sum-mix| = {worst:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify", action="store_true",
                    help="only re-check sum(stems) == mixture on what is written")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    rows, excluded = partition_songs()
    print(f"{len(rows)} songs selected "
          f"({sum(1 for r in rows if r['has_bleed'] == 'no')} clean, "
          f"{sum(1 for r in rows if r['has_bleed'] == 'yes')} bleeding), "
          f"{len(excluded)} excluded up front")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "songs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    skips_path = OUT / "skipped.json"
    # Reasons found during conversion persist across runs (that is what keeps a
    # re-run cheap); every other reason is recomputed, so relaxing a selection
    # rule actually re-admits the songs it used to exclude instead of leaving
    # them stranded behind a stale entry.
    previous = json.loads(skips_path.read_text()) if skips_path.exists() else {}
    selected_ids = {r["song_id"] for r in rows}
    runtime = {
        k: v for k, v in previous.items()
        if k in selected_ids and v.startswith(RUNTIME_SKIP_PREFIXES)
    }
    skips = {**runtime, **excluded}

    # Backfill fields added after a song was first written, so every entry has
    # the same schema whether or not the dataset was built in one pass.
    by_id = {r["song_id"]: r for r in rows}
    for sid, entry in manifest.items():
        if "in_musdb25" not in entry and sid in by_id:
            entry["in_musdb25"] = by_id[sid].get("in_musdb25", "") == "yes"
    todo = [r for r in rows if r["song_id"] not in manifest and r["song_id"] not in skips]
    print(f"{len(todo)} to do")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(convert_song, todo), 1):
            if "skipped" in res:
                skips[res["song_id"]] = res["skipped"]
                skips_path.write_text(json.dumps(skips, indent=1, sort_keys=True))
                print(f"  {i}/{len(todo)} SKIP {res['song_id']}: {res['skipped']}")
                continue
            manifest[res["song_id"]] = res["entry"]
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            e = res["entry"]
            print(f"  {i}/{len(todo)} {res['song_id']} ({len(e['sources'])} tracks, "
                  f"gain={e['gain']:.3f}, mixSNR={e['medleydb_mix_snr_db']}dB)")

    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    if prune_undersized(manifest, skips):
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    link_trimmed()
    write_nobleeding(manifest)

    skips_path.write_text(json.dumps(skips, indent=1, sort_keys=True))
    print("\nskipped.json:")
    for reason, n in sorted(Counter(skips.values()).items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {reason}")
    total = len(manifest) + len(skips)
    print(f"\n{len(manifest)} songs written, {len(skips)} skipped, "
          f"{total} accounted for -> {OUT}")


if __name__ == "__main__":
    main()

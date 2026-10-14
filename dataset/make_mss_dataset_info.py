"""Build MSS-schema dataset-info JSONs (the format `data.moisesdb.MoisesDBDataset` reads).

Two jobs:

*   **convert** a prepared corpus (`<root>/<corpus>-stems/{songs.json,trimmed/}`,
    the layout `MoisesDBDataset` grew up with) into the schema;
*   **merge** several of those, plus manifests written elsewhere, into one file.

The schema, all paths relative to a single dataset ROOT so that one file can
span corpora::

    {"song_info":  {id: {"sources": [[leaf, relpath], ...], "artist", "name"}},
     "stem_paths": {leaf: [relpath, ...]},
     "stem_tax":   {family: [leaf, ...]}}

Song ids are prefixed with the corpus name on merge. MoisesDB ids are UUIDs and
MedleyDB ids are titles, so they would not collide -- but a merged file is a
SPLIT definition, and a split whose keys depend on two corpora never having
picked the same name is not one worth debugging later.

`stem_tax` is emitted from `STEM_TAX` restricted to leaves that actually have
audio in this file, matching what `MoisesDBDataset.available_tax` computes.

Usage
-----
    python dataset/make_mss_dataset_info.py \
        --root <dataset root> \
        --corpus medleydb:medleydb-stems --corpus albumdb:albumdb-stems \
        --info moisesdb:mss-dataset-infos/moisesdb-train.json \
        --out mss-dataset-infos/moisesdb_medleydb_albumdb_train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.moisesdb import LEAF_TO_GROUP, STEM_TAX  # noqa: E402


def from_corpus(root: Path, rel: str, songs_file: str = "songs.json") -> dict:
    """Prepared `<rel>` corpus -> `{"song_info", "stem_paths"}` under `root`.

    The two keys point at DIFFERENT copies of the same stems, and the difference
    is not cosmetic:

    *   `song_info` -> `clean/`, the FULL-LENGTH files. A song's tracks are all
        cropped at one shared offset, and that offset only denotes the same
        musical instant if no track has been shortened independently of the
        others. Silence trimming is per-track, so pointing song sources at
        `trimmed/` would de-align every multi-track item -- which is precisely
        the coherence SEG is trained on.
    *   `stem_paths` -> the SILENCE-TRIMMED copies (see `trim_pool`, which
        rewrites them after this runs). This pool is read ONLY by the loaders
        that mix unrelated tracks -- the random-mix dataset and controlled
        cacophony -- where each track carries its own random offset, so there is
        no alignment to preserve and a crop into dead air is pure waste.
    """
    base = root / rel
    with open(base / songs_file) as f:
        songs = json.load(f)

    def _ext(sub):
        return next((e for e in (".flac", ".wav")
                     if any((base / sub).glob(f"*/*{e}"))), ".flac")

    ext = _ext("clean")

    song_info, stem_paths = {}, {}
    missing = 0
    for sid, info in songs.items():
        sources = []
        for s in info["sources"]:
            parts = s.split("/")
            sub, leaf, name = parts[0], parts[1], parts[-1]
            path = base / sub / leaf / (name + ext)
            if not path.exists():
                missing += 1
                continue
            if leaf not in LEAF_TO_GROUP:
                raise ValueError(f"{rel}: leaf outside STEM_TAX: {leaf}")
            sources.append([leaf.replace("_", " "), str(path.relative_to(root))])

            stem_paths.setdefault(leaf, []).append(
                str(path.relative_to(root)))
        if len(sources) >= 2:
            song_info[sid] = {
                "sources": sources,
                "artist": info.get("artist", rel),
                "name": info.get("name", sid),
            }
    if missing:
        print(f"[{rel}] {missing} sources had no audio file, skipped")
    return {"song_info": song_info, "stem_paths": stem_paths}


def drop_bleeding(root: Path, part: dict, corpus: str = "MoisesDB",
                  release: str = "moisesdb_v0.1", whole_songs: bool = False) -> dict:
    """Remove stems flagged `has_bleed` in the raw MoisesDB metadata.

    The upstream manifests carry a `-no_bleed` suffix but identical content: the
    generating script unpacks the per-stem bleed flag and never tests it, so the
    filter exists in the filename only. The flag itself is per track in each
    song's `data.json`, so the filtering is done here from the release rather
    than trusted to the manifest.

    Why it matters for an EVAL split specifically: a bleeding stem contains
    audible traces of the other instruments, so its PaSST embedding is not an
    embedding of that source alone -- `matched_cosine` would be scored against a
    contaminated reference and would reward reproducing the contamination.

    MedleyDB records the same property structurally rather than as a flag: our
    prepare script writes bleeding stems under `bleeding/` instead of `clean/`,
    so those are dropped by path. Corpora that state neither (AlbumDB) pass
    through untouched -- absence of a flag is not evidence of bleed.

    `whole_songs` chooses between the two things "no bleed" can mean, and the
    difference matters for an EVAL split:

    *   False (stem level) -- drop the bleeding stems, keep the song. Every
        target embedding is then clean, but the MIXTURE is the sum of what
        survived, so an affected song is evaluated with one instrument missing
        from the arrangement. Fine when the targets are what is scored; wrong if
        anything is measured against the true mix.
    *   True (song level) -- drop any song with a bleeding stem. Every remaining
        mixture is a real, complete arrangement. Costs roughly half the songs
        (26 of 48 val, 26 of 47 test survive), because ~45% of songs carry at
        least one bleeding stem even though only ~7% of stems are flagged.
    """
    def _bleeds_by_path(rel: str) -> bool:
        return "bleeding" in Path(rel).parts

    bleed: dict[str, bool] = {}
    for song_dir in (root / corpus / release).iterdir():
        meta = song_dir / "data.json"
        if not meta.exists():
            continue
        with open(meta) as f:
            for stem in json.load(f)["stems"]:
                for tr in stem["tracks"]:
                    bleed[tr["id"]] = bool(tr.get("has_bleed", False))

    song_info, dropped, songs_lost = {}, 0, 0
    for sid, info in part["song_info"].items():
        keep = [s for s in info["sources"]
                if not bleed.get(Path(s[-1]).stem, False)
                and not _bleeds_by_path(s[-1])]
        if whole_songs and len(keep) != len(info["sources"]):
            dropped += len(info["sources"])
            songs_lost += 1
            continue
        dropped += len(info["sources"]) - len(keep)
        # A song whose remaining stems cannot form a mixture is not a usable
        # item; keeping it would only produce rejected draws at load time.
        if len(keep) >= 2:
            song_info[sid] = {**info, "sources": keep}
        else:
            songs_lost += 1

    kept = {s[-1] for v in song_info.values() for s in v["sources"]}
    stem_paths = {leaf: [p for p in paths if p in kept]
                  for leaf, paths in part["stem_paths"].items()}
    stem_paths = {k: v for k, v in stem_paths.items() if v}
    how = "whole songs" if whole_songs else "stems"
    print(f"[no_bleed/{how}] dropped {dropped} sources, {songs_lost} songs")
    return {"song_info": song_info, "stem_paths": stem_paths}


def trim_pool(root: Path, part: dict,
              corpora=("moisesdb-stems", "medleydb-stems", "albumdb-stems")) -> dict:
    """Point `stem_paths` at the SILENCE-TRIMMED copy of each stem where one exists.

    Only the pool moves. `song_info` keeps the full-length files, because a
    song's tracks share one crop offset and trimming is per-track -- see
    `from_corpus`.

    The trimmed trees were produced with ffmpeg's RMS silence remover at
    -50 dBFS / 20 ms, which is the same floor `active_tracks` uses relatively,
    so the two agree about what counts as playing.

    Matching is by (leaf, stem id): the trimmed copy keeps the source id as its
    filename, and the id alone is ambiguous because MedleyDB and AlbumDB name
    one file per leaf after the SONG. Stems with no trimmed copy (MoisesDB
    bleeding tracks, which were never trimmed) keep their full-length path --
    `_random_crop`'s retry is what covers those.
    """
    idx: dict[tuple[str, str], str] = {}
    for corp in corpora:
        tdir = root / corp / "trimmed"
        if not tdir.is_dir():
            continue
        for leaf_dir in tdir.iterdir():
            if leaf_dir.is_dir():
                for f in leaf_dir.iterdir():
                    idx[(leaf_dir.name, f.stem)] = str(f.relative_to(root))

    stem_paths, hit, tot = {}, 0, 0
    for leaf, paths in part["stem_paths"].items():
        out = []
        for p in paths:
            tot += 1
            t = idx.get((leaf.replace(" ", "_"), Path(p).stem))
            hit += t is not None
            out.append(t or p)
        stem_paths[leaf] = out
    print(f"[trim_pool] {hit}/{tot} pool entries point at a trimmed copy")
    return {"song_info": part["song_info"], "stem_paths": stem_paths}


def load_info(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {"song_info": d["song_info"], "stem_paths": d["stem_paths"]}


def merge(parts: dict[str, dict]) -> dict:
    # Ids are prefixed with the corpus name only when there is something to
    # disambiguate. A single-corpus file then keeps its ORIGINAL ids, so it
    # stays a drop-in replacement for the upstream manifest it was derived from
    # -- which matters for the `-no_bleed` files, whose whole purpose is to be
    # swapped in where the unfiltered ones are used.
    tag = len(parts) > 1
    song_info, stem_paths = {}, {}
    for prefix, part in parts.items():
        for sid, info in part["song_info"].items():
            song_info[f"{prefix}/{sid}" if tag else sid] = info
        for leaf, paths in part["stem_paths"].items():
            leaf = leaf.replace(" ", "_")
            stem_paths.setdefault(leaf, []).extend(paths)

    # Deduplicate while keeping order: a file can be listed once per song that
    # uses it, and `stem_paths` is a SAMPLING POOL for same-leaf substitution --
    # duplicates would weight those files up for no stated reason.
    stem_paths = {k: list(dict.fromkeys(v)) for k, v in stem_paths.items()}

    stem_tax = {g: [l for l in leaves if stem_paths.get(l)]
                for g, leaves in STEM_TAX.items()}
    stem_tax = {g: v for g, v in stem_tax.items() if v}
    return {"song_info": song_info, "stem_paths": stem_paths,
            "stem_tax": stem_tax}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--corpus", action="append", default=[],
                    metavar="NAME:REL[:SONGS_JSON]",
                    help="prepared corpus directory relative to --root")
    ap.add_argument("--info", action="append", default=[], metavar="NAME:PATH",
                    help="existing MSS-schema JSON (path relative to --root "
                         "or absolute)")
    ap.add_argument("--no-bleed", action="store_true",
                    help="drop stems flagged has_bleed in the release metadata "
                         "(the upstream -no_bleed files do not)")
    ap.add_argument("--raw-pool", action="store_true",
                    help="keep the substitution pool on full-length files "
                         "instead of the silence-trimmed copies")
    ap.add_argument("--no-bleed-songs", action="store_true",
                    help="drop the whole SONG if any of its stems bleeds, so "
                         "every remaining mixture is a complete arrangement")
    ap.add_argument("--out", required=True, type=Path,
                    help="output path, relative to --root or absolute")
    args = ap.parse_args()

    parts = {}
    for spec in args.corpus:
        name, rel, *rest = spec.split(":")
        parts[name] = from_corpus(args.root, rel,
                                  songs_file=rest[0] if rest else "songs.json")
    for spec in args.info:
        name, path = spec.split(":", 1)
        p = Path(path)
        parts[name] = load_info(p if p.is_absolute() else args.root / p)

    # Bleed filtering FIRST: it prunes the pool by matching pool paths against
    # the kept song paths, and `trim_pool` rewrites pool paths to a different
    # tree -- run in the other order, that match fails and the pool is emptied.
    if args.no_bleed or args.no_bleed_songs:
        parts = {k: drop_bleeding(args.root, v, whole_songs=args.no_bleed_songs)
                 for k, v in parts.items()}

    if not args.raw_pool:
        parts = {k: trim_pool(args.root, v) for k, v in parts.items()}

    out = merge(parts)
    dst = args.out if args.out.is_absolute() else args.root / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)

    for name, part in parts.items():
        print(f"  {name:12s} {len(part['song_info']):4d} songs, "
              f"{sum(len(v['sources']) for v in part['song_info'].values()):5d} sources")
    n_src = sum(len(v["sources"]) for v in out["song_info"].values())
    print(f"-> {dst}: {len(out['song_info'])} songs, {n_src} sources, "
          f"{len(out['stem_paths'])} leaves, {len(out['stem_tax'])} families")


if __name__ == "__main__":
    main()

"""Assemble a paper data release from regenerated model outputs.

    uv run python evaluation/build_release.py --release <out of generate_estimates.py>

Given the SEG trajectories and model audio written by `generate_estimates.py`,
this

  1. defines the scored populations exactly as the paper does:
       groups     VDBGP families active in the segment, detected by SEG AND by
                  MusicFlamingo;
       all_instr  individual tracks active in the segment, detected by SEG;
  2. writes `results_paper/populations/{groups_vdbgp,all_instr}.csv` (one row per
     item, one audio path per system) and `*_all_sources.csv` (every SEG step);
  3. links (or copies) the reference audio of every population item into
     `estimates_paper/` from the test set. The baselines' audio is not produced
     by this code: with MUS3D_BASELINES set to a tree holding it
     (`results/<run>.csv`) it is linked too, otherwise the baselines keep
     their released per-item scores;
  4. links the unchanged discovery inputs (ground truth, oracle classifier,
     Multi-label Classifier and MusicFlamingo predictions) from the released data.

Nothing is written outside `--release`. Afterwards the release is scored with
`reproduce_tables.py` and `reproduce_separation_from_audio.py` by pointing
RESULTS_PAPER / ESTIMATES_PAPER at it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: the test set built by `dataset/prepare_test_set.py`
TEST_SET = Path(os.environ.get("MUS3D_TEST_SET", REPO / "test_set"))
#: tree holding the baselines' audio, optional
BASELINE_ROOT = os.environ.get("MUS3D_BASELINES")
OLD = Path(os.environ.get(
    "RELEASED_RESULTS", os.environ.get("RESULTS_PAPER", REPO / "results_paper")))
VDBGP = {"vocals", "drums", "bass", "guitar", "piano"}

#: population column -> results run under MUS3D_BASELINES holding the audio of every item
BASELINES = {
    "groups": {
        "Ref_GroundTruth-AE": "_reference_dacvae",
        "Ref_IdealRatioMask": "_oracle_irm2",
        "Static_HTDemucs": "htdemucs_6s_fixed",
        "Static_BS-Locoformer": "static-locoformer-med_fixed",
        "Oracle_SAM-A": "sam-audio-large_text",
        "Blind_Banquet-other-song": "banquet_other_song",
        "Blind_MusicFlamingo+SAM-A": "mf-bvoices+sam-audio-large_text",
    },
    "all_instr": {
        "Ref_GroundTruth-AE": "_reference_dacvae_track",
        "Ref_IdealRatioMask": "_oracle_irm2_track",
        "Oracle_SAM-A": "sam-audio-large-track_text",
    },
}
MODELS = ["Oracle_DiT", "Oracle_BS-Locoformer", "Blind_MuS3D-DiT", "Blind_MuS3D-BS-Locoformer"]
#: block -> (manifest, audio dir, seg_embeddings level dir, item column, population file)
BLOCK = {"groups": ("manifest.json", "audio", "groups", "stem", "groups_vdbgp.csv"),
         "all_instr": ("manifest_track.json", "audio_track", "instruments", "track", "all_instr.csv")}


def slug(name: str) -> str:
    return name.replace(" ", "_")


def seg_id(i) -> str:
    return f"seg{int(i):03d}"


def truth(manifest: str) -> dict:
    t = defaultdict(set)
    for it in json.load(open(TEST_SET / manifest))["items"]:
        if it["active"]:
            t[(it["song_id"], seg_id(it["segment_index"]))].add(it["stem"])
    return t


def trajectories(release: Path, lvl_dir: str) -> dict:
    out = {}
    base = release / "results_paper/discovery/seg_embeddings" / lvl_dir / "SEG"
    for f in base.glob("*/seg*/trajectory.csv"):
        out[(f.parent.parent.name, f.parent.name)] = list(csv.DictReader(open(f)))
    return out


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if not src.exists():
        raise SystemExit(f"missing {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:                 # another filesystem
        shutil.copyfile(src, dst)


def baseline_paths(run: str) -> dict:
    """(song, segNNN, target) -> audio path, from the baseline results csv."""
    out, root = {}, Path(BASELINE_ROOT)
    for r in csv.DictReader(open(root / "results" / f"{run}.csv")):
        out[(r["song"], seg_id(r["segment_index"]), r["target"])] = root / r["est_path"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", required=True)
    ap.add_argument("--no-require-models", dest="require_models", action="store_false",
                    help="write the populations before the model audio exists, so the "
                         "oracle stages can be run over exactly these items")
    ap.add_argument("--check", type=int, default=300,
                    help="items per baseline whose linked audio is compared byte-for-byte "
                         "with the previous release")
    args = ap.parse_args()
    rel = Path(args.release)
    rp, ep = rel / "results_paper", rel / "estimates_paper"

    # unchanged discovery inputs
    for f in (OLD / "discovery/ground_truth").iterdir():
        link(f, rp / "discovery/ground_truth" / f.name)
    for lvl in ("groups", "instruments"):
        for f in (OLD / "discovery" / lvl).glob("*_discovery.csv"):
            if f.name.startswith(("OracleClassifier", "MultiLabelClassifier", "MusicFlamingo")):
                link(f, rp / "discovery" / lvl / f.name)
    link(OLD / "names.csv", rp / "names.csv")

    mf = {}
    for r in csv.DictReader(open(OLD / "discovery/groups/MusicFlamingo_discovery.csv")):
        mf[(r["song"], seg_id(r["segment_index"]))] = set(filter(None, r["predicted_families"].split("|")))

    for block, (manifest, audio, lvl_dir, col, pop_file) in BLOCK.items():
        old_rows = list(csv.DictReader(open(OLD / "populations" / pop_file)))
        prompt = {r[col]: r["prompt"] for r in old_rows}
        old_path = {(r["song"], r["segment"], r[col]): r for r in old_rows}
        tr, trajs = truth(manifest), trajectories(rel, lvl_dir)

        items = []
        for (song, sg), steps in sorted(trajs.items()):
            named = {s["named"] for s in steps}
            for x in sorted(named & tr[(song, sg)]):
                if block == "groups" and not (x in VDBGP and x in mf.get((song, sg), set())):
                    continue
                items.append((song, sg, x))
        baselines = BASELINES[block] if BASELINE_ROOT else {}
        base = {c: baseline_paths(run) for c, run in baselines.items()}
        if not baselines:
            # no baseline audio to score: their released per-item scores stand in
            for c in BASELINES[block]:
                for f in (OLD / "separation" / block).glob(f"*/{c}.csv"):
                    link(f, rp / "separation" / block / f.parent.name / f.name)

        header = ["song", "segment", col, "Ref_Mixture", "Ref_GroundTruth",
                  *baselines, *MODELS, "prompt"]
        rows, n_missing = [], 0
        for song, sg, x in items:
            r = {"song": song, "segment": sg, col: x,
                 "prompt": prompt.get(x, x.replace("_", " "))}
            r["Ref_Mixture"] = f"{block}/Ref_Mixture/{song}/{sg}/mixture.wav"
            if args.require_models and not all(
                    (ep / f"{block}/{c}/{song}/{sg}/{slug(x)}.wav").exists() for c in MODELS):
                n_missing += 1      # model outputs were generated for a subset
                continue
            link(TEST_SET / audio / song / sg / "mixture.wav", ep / r["Ref_Mixture"])
            r["Ref_GroundTruth"] = f"{block}/Ref_GroundTruth/{song}/{sg}/{slug(x)}.wav"
            link(TEST_SET / audio / song / sg / f"target_{slug(x)}.wav", ep / r["Ref_GroundTruth"])
            for c in baselines:
                r[c] = f"{block}/{c}/{song}/{sg}/{slug(x)}.wav"
                link(base[c][(song, sg, x)], ep / r[c])
            for c in MODELS:
                r[c] = f"{block}/{c}/{song}/{sg}/{slug(x)}.wav"
            rows.append(r)
        out = rp / "populations" / pop_file
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        keys = {(r["song"], r["segment"], r[col]) for r in rows}
        print(f"[{block}] {len(rows)} items with all model outputs ({n_missing} without), "
              f"(previous release {len(old_rows)}, "
              f"shared {len(keys & set(old_path))})")

        if not args.require_models:
            continue
        # the linked baselines must be the very files of the previous release
        shared = sorted(keys & set(old_path))
        old_est = OLD.parent / "estimates_paper"
        for c in baselines if old_est.is_dir() else []:
            for k in random.Random(0).sample(shared, min(args.check, len(shared))):
                new = ep / f"{block}/{c}/{k[0]}/{k[1]}/{slug(k[2])}.wav"
                if new.read_bytes() != (old_est / old_path[k][c]).read_bytes():
                    raise SystemExit(f"{c} {k}: linked audio differs from the previous release")
        if baselines and old_est.is_dir():
            print(f"[{block}] baseline audio identical to the previous release on the checked items")

        # every SEG step, with its status
        with open(rp / "populations" / pop_file.replace(".csv", "_all_sources.csv").replace("_vdbgp", ""),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["song", "segment", "step", "named", "stop_cos", "status", "embedding",
                        "Blind_MuS3D-DiT", "Blind_MuS3D-BS-Locoformer"])
            for (song, sg), steps in sorted(trajs.items()):
                seen = set()
                for s in steps:
                    status = ("duplicate" if s["named"] in seen else
                              "detected" if s["named"] in tr[(song, sg)] else "unmatched")
                    seen.add(s["named"])
                    wav = f"{song}/{sg}/step{s['step']}_{slug(s['named'])}.wav"
                    w.writerow([song, sg, s["step"], s["named"], s["stop_cos"], status, s["emb_path"],
                                f"{block}/Blind_MuS3D-DiT_all_sources/{wav}",
                                f"{block}/Blind_MuS3D-BS-Locoformer_all_sources/{wav}"])


if __name__ == "__main__":
    main()

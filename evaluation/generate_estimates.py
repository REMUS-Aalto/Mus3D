"""Regenerate the model outputs behind the paper tables, with fixed seeds.

Writes a data tree laid out like the released `results_paper/` + `estimates_paper/`
under `--out` (never inside the released data):

    <out>/results_paper/discovery/seg_embeddings/<groups|instruments>/<SEG|SEG-res-embedding>/
        <song>/segNNN/step<i>_<name>.npy + trajectory.csv
    <out>/estimates_paper/<groups|all_instr>/<Row>/<song>/segNNN/<name>.wav
    <out>/estimates_paper/<groups|all_instr>/<Row>_all_sources/<song>/segNNN/step<i>_<name>.wav

Stages (each shardable with --shard i/n, resumable: existing outputs are skipped):

    seg            SEG recursion on every benchmark segment (both variants, both levels)
    oracle_dit     DiT separator, oracle queries, on the paper population items
    oracle_loco    BS-Locoformer, oracle queries, on the paper population items
    mus3d_dit      DiT separator on every SEG (mixemb) step
    mus3d_loco     BS-Locoformer on every SEG (mixemb) step
    classifiers    the Oracle Classifier and Multi-label Classifier discovery rows,
                   written as <out>/results_paper/discovery/<level>/<Row>_discovery.csv

    CUDA_VISIBLE_DEVICES=0 uv run python evaluation/generate_estimates.py seg --out <dir> --shard 0/4

REPRODUCIBLE: every segment is seeded with `seed + crc32("<song>/segNNN")`, the
CUDA kernels are pinned to deterministic algorithms and the codec's torch.compile
is off, so a rerun of a stage reproduces its audio bit for bit (checked on A40s,
across GPUs and processes, by `tests/test_checkpoints.py::TestDeterminism`).
Sharding is over the benchmark's segment list for every stage, so `--shard i/n`
selects the same segments in each stage. The settings are the paper's:
SEG raw weights, 32 steps, churn 0.1, at most 16 extractions, mono input; DiT EMA
weights, 32 steps, churn 0.5, mono; BS-Locoformer stereo. Level flag: 1 (groups)
or 0 (individual tracks). Oracle queries are the target of the scored segment,
RMS-normalised to -20 dBFS for groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: the test set built by `dataset/prepare_test_set.py`
TEST_SET = Path(os.environ.get("MUS3D_TEST_SET", REPO / "test_set"))
RESULTS = Path(os.environ.get("RESULTS_PAPER", REPO / "results_paper"))
#: the released `checkpoints/` directory
CKPTS = Path(os.environ.get("MUS3D_CHECKPOINTS", REPO / "checkpoints"))
CKPT = {
    "SEG": CKPTS / "SEG_mixemb",
    "SEG-res-embedding": CKPTS / "SEG_resemb",
    "dit": CKPTS / "DiT",
    "loco": CKPTS / "BS-Locoformer_query",
}
CLASSIFIER = CKPTS / "classifiers/stem_classifier_pcatrain.pkl"
MULTILABEL = CKPTS / "classifiers/multilabel_classifier.pkl"
PCA = CKPTS / "classifiers/pca_params_train.pkl"
#: block -> (seg_embeddings level dir, classifier level, separator level flag, audio dir)
BLOCK = {"groups": ("groups", "stem", 1, "audio"),
         "all_instr": ("instruments", "track", 0, "audio_track")}
POP = {"groups": "groups_vdbgp.csv", "all_instr": "all_instr.csv"}


def slug(name: str) -> str:
    return name.replace(" ", "_")


def deterministic() -> None:
    """Make the CUDA kernels used here pick the same algorithm every run.

    With this and the per-segment seed, a rerun of any stage reproduces its
    audio bit for bit on the same GPU model (checked in `tests/`).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_for(seed: int, song: str, seg: str) -> None:
    s = (seed + zlib.crc32(f"{song}/{seg}".encode())) % (2 ** 31)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


def read(path: Path) -> torch.Tensor:
    x, _ = sf.read(str(path), dtype="float32", always_2d=True)
    t = torch.from_numpy(x.T.copy())
    return t if t.shape[0] == 2 else t.repeat(2, 1)


def write(path: Path, audio: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), audio.detach().float().cpu().numpy().T, 44100)
    tmp.rename(path)


def segments() -> list[tuple[str, str]]:
    man = json.load(open(TEST_SET / "manifest.json"))["items"]
    return sorted({(it["song_id"], f"seg{int(it['segment_index']):03d}") for it in man})


def shard(items: list, spec: str | None, limit: int | None = None) -> list:
    if spec:
        i, n = (int(x) for x in spec.split("/"))
        items = items[i::n]
    return items[:limit] if limit else items


def check_outside(out: Path) -> None:
    released = {RESULTS.resolve(), RESULTS.resolve().parent / "estimates_paper",
                TEST_SET.resolve(), CKPTS.resolve()}
    o = out.resolve()
    if any(o == r or r in o.parents for r in released):
        raise SystemExit(f"--out {out} is inside the released data; refusing to write there")


# ---------------------------------------------------------------- SEG ----

def run_seg(args, out: Path) -> None:
    from source_embedding_generator.separator_io import load_seg
    from classifiers.stem_classifier import StemClassifier

    clf = StemClassifier.load(str(CLASSIFIER))
    todo = shard(segments(), args.shard, args.limit)
    for row in ("SEG", "SEG-res-embedding"):
        seg = load_seg(CKPT[row], device="cuda", use_ema=False)
        for block in ("groups", "all_instr"):
            lvl_dir, clf_level, level, _ = BLOCK[block]
            base = out / "results_paper/discovery/seg_embeddings" / lvl_dir / row
            for song, sg in tqdm(todo, desc=f"{row} {block}"):
                d = base / song / sg
                if (d / "trajectory.csv").exists():
                    continue
                mix = read(TEST_SET / "audio" / song / sg / "mixture.wav")
                seed_for(args.seed, song, sg)
                with torch.no_grad():
                    r = seg.inference(mix.mean(0, keepdim=True)[None].cuda(), level=level,
                                      max_steps=16, steps=32, churn=0.1, audio=False)
                names = clf.predict(r["emb"], clf_level)
                d.mkdir(parents=True, exist_ok=True)
                rows = []
                for i, (nm, e, sc) in enumerate(zip(names, r["emb"], r["stop_cos"])):
                    rel = f"discovery/seg_embeddings/{lvl_dir}/{row}/{song}/{sg}/step{i}_{slug(nm)}.npy"
                    np.save(out / "results_paper" / rel, e)
                    rows.append({"step": i, "named": nm, "stop_cos": f"{float(sc):.4f}",
                                 "emb_path": rel})
                tmp = d / "trajectory.tmp"
                with open(tmp, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["step", "named", "stop_cos", "emb_path"])
                    w.writeheader()
                    w.writerows(rows)
                tmp.rename(d / "trajectory.csv")
        del seg
        torch.cuda.empty_cache()


# -------------------------------------------------------- classifiers ----

def run_classifiers(args, out: Path) -> None:
    """The two classifier rows of the discovery table.

    Oracle Classifier: the stem classifier names every active reference source
    from its PaSST-PCA embedding. Multi-label Classifier: the tags over their
    thresholds for the mixture's embedding, one pass per segment.
    """
    from classifiers.stem_classifier import StemClassifier
    from query_conditioned_bslocoformer.query_model import PasstWrapper, PcaAdapter

    if args.shard:
        raise SystemExit("the classifiers stage is not sharded")
    passt, pca = PasstWrapper().cuda().eval(), PcaAdapter(str(PCA)).cuda().eval()
    oracle, tagger = StemClassifier.load(str(CLASSIFIER)), StemClassifier.load(str(MULTILABEL))

    def embed(path: Path) -> np.ndarray:
        with torch.no_grad():
            return pca(passt(read(path)[None].cuda()).float()).cpu().numpy()

    todo = shard(segments(), None, args.limit)
    for block in args.blocks:
        lvl_dir, clf_level, _, audio_dir = BLOCK[block]
        dst = {row: out / "results_paper/discovery" / lvl_dir / f"{row}_discovery.csv"
               for row in ("OracleClassifier", "MultiLabelClassifier")}
        if all(p.exists() for p in dst.values()):
            continue
        manifest = "manifest.json" if block == "groups" else "manifest_track.json"
        active = defaultdict(list)
        for it in json.load(open(TEST_SET / manifest))["items"]:
            if it["active"]:
                active[(it["song_id"], f"seg{int(it['segment_index']):03d}")].append(it["stem"])
        rows = {row: [] for row in dst}
        for song, sg in tqdm(todo, desc=f"classifiers {block}"):
            d = TEST_SET / audio_dir / song / sg
            names = [oracle.predict(embed(d / f"target_{slug(x)}.wav"), clf_level)[0]
                     for x in active[(song, sg)]]
            if names:
                rows["OracleClassifier"].append([song, int(sg[3:]), "|".join(names)])
            # the tagger hears what SEG hears; the vocals family is one target
            # at both levels
            leaves = tagger.tags(embed(TEST_SET / "audio" / song / sg / "mixture.wav"))[0]
            groups = [tagger.leaf_to_group.get(leaf, leaf) for leaf in leaves]
            pred = sorted({g if block == "groups" or g == "vocals" else leaf
                           for leaf, g in zip(leaves, groups)})
            rows["MultiLabelClassifier"].append([song, int(sg[3:]), "|".join(pred)])
        for row, path in dst.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["song", "segment_index", "predicted_names"])
                w.writerows(rows[row])


# ---------------------------------------------------------- separators ----

def load_sep(kind: str):
    from source_embedding_generator.separator_io import load_separator
    # codec_compile=False: torch.compile's autotuning is non-deterministic
    # across processes, which would break the bit-for-bit rerun guarantee.
    return load_separator(CKPT[kind], device="cuda", use_ema=(kind == "dit"),
                          codec_compile=False)


def separate(kind: str, sep, mix: torch.Tensor, q: torch.Tensor, level: int,
             query_is_embedding: bool) -> torch.Tensor:
    """mix [2, T]; q [k, D] embeddings or [k, 2, T] audio -> [k, C, T]."""
    k = q.shape[0]
    lvl = torch.full((k,), level, dtype=torch.long, device="cuda")
    with torch.no_grad():
        if kind == "dit":
            emb = q.cuda() if query_is_embedding else sep.embed_query(q.cuda())
            z = sep.encode(mix[None].cuda()).repeat(k, 1, 1)
            zt, _ = sep.sample_latent(z, emb, n_step=32, level=lvl, churn=0.5)
            return sep.decode(zt).float().cpu()
        est = sep.sample(queries=q.cuda().unsqueeze(1), mix=mix[None].cuda().repeat(k, 1, 1),
                         level=lvl, query_is_embedding=query_is_embedding)[:, 0]
        return est.float().cpu()


def run_oracle(args, out: Path, kind: str) -> None:
    row = {"dit": "Oracle_DiT", "loco": "Oracle_BS-Locoformer"}[kind]
    sep = load_sep(kind)
    for block in args.blocks:
        _, _, level, audio_dir = BLOCK[block]
        # This release's own population if built, else the released one.
        pop = out / "results_paper/populations" / POP[block]
        if not pop.exists():
            pop = RESULTS / "populations" / POP[block]
        by_seg = defaultdict(list)
        for r in csv.DictReader(open(pop)):
            by_seg[(r["song"], r["segment"])].append(r.get("track") or r["stem"])
        todo = [k for k in shard(segments(), args.shard, args.limit) if k in by_seg]
        for song, sg in tqdm(todo, desc=f"{row} {block}"):
            d = out / "estimates_paper" / block / row / song / sg
            names = [n for n in by_seg[(song, sg)] if not (d / f"{slug(n)}.wav").exists()]
            if not names:
                continue
            mix = read(TEST_SET / audio_dir / song / sg / "mixture.wav")
            qs = []
            for n in names:
                q = read(TEST_SET / audio_dir / song / sg / f"target_{slug(n)}.wav")
                if block == "groups":
                    rms = q.pow(2).mean().sqrt()
                    q = q * (10 ** (-20 / 20) / rms) if rms > 1e-8 else q
                qs.append(q)
            seed_for(args.seed, song, sg)
            est = separate(kind, sep, mix, torch.stack(qs), level, query_is_embedding=False)
            for n, e in zip(names, est):
                write(d / f"{slug(n)}.wav", e)


def run_mus3d(args, out: Path, kind: str) -> None:
    row = {"dit": "Blind_MuS3D-DiT", "loco": "Blind_MuS3D-BS-Locoformer"}[kind]
    sep = load_sep(kind)
    for block in args.blocks:
        lvl_dir, _, level, _ = BLOCK[block]
        base = out / "results_paper/discovery/seg_embeddings" / lvl_dir / "SEG"
        for song, sg in tqdm(shard(segments(), args.shard, args.limit), desc=f"{row} {block}"):
            traj = base / song / sg / "trajectory.csv"
            if not traj.exists():
                raise SystemExit(f"missing {traj}: run the `seg` stage first")
            steps = list(csv.DictReader(open(traj)))
            d_all = out / "estimates_paper" / block / f"{row}_all_sources" / song / sg
            d = out / "estimates_paper" / block / row / song / sg
            if not steps or (d / "done").exists():
                continue
            mix = read(TEST_SET / "audio" / song / sg / "mixture.wav")
            q = torch.from_numpy(np.stack([np.load(out / "results_paper" / s["emb_path"])
                                           for s in steps])).float()
            seed_for(args.seed, song, sg)
            est = separate(kind, sep, mix, q, level, query_is_embedding=True)
            summed: dict[str, torch.Tensor] = {}
            for s, e in zip(steps, est):
                write(d_all / f"step{s['step']}_{slug(s['named'])}.wav", e)
                summed[s["named"]] = summed[s["named"]] + e if s["named"] in summed else e
            for n, e in summed.items():      # the same name twice is summed for scoring
                write(d / f"{slug(n)}.wav", e)
            (d / "done").touch()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["seg", "oracle_dit", "oracle_loco", "mus3d_dit", "mus3d_loco",
                                      "classifiers"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", nargs="+", default=["groups", "all_instr"], choices=list(BLOCK))
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N segments of the shard")
    args = ap.parse_args()
    out = Path(args.out)
    check_outside(out)
    deterministic()
    if args.stage == "seg":
        run_seg(args, out)
    elif args.stage == "classifiers":
        run_classifiers(args, out)
    elif args.stage.startswith("oracle"):
        run_oracle(args, out, args.stage.split("_")[1])
    else:
        run_mus3d(args, out, args.stage.split("_")[1])


if __name__ == "__main__":
    main()

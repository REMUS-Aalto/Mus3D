"""Multi-label classifier baseline (the "Multi-label Classifier" discovery row).

Given one mixture, names every source in it directly:
`PaSST(mixture) -> PCA-128 -> sigmoid(Wx + b) -> 35 leaf tags -> 11
families by grouped max`. One forward pass, no generation or separation.
Uses the same dataloader, representation, vocabulary and head-fitting
protocol as `stem_classifier`, so the two are comparable; see the paper
for why this baseline exists and how it differs from SEG's recursion.

Training:

    PYTHONPATH=src ./.venv/bin/python3 -m classifiers.multilabel_classifier \
        --n_train 20000 --n_crops_val 3 --device cuda \
        --cache output/mixture_tagger_emb.npz

Usage (the pickle is `StemClassifier`-shaped):

    from classifiers.stem_classifier import StemClassifier
    tag = StemClassifier.load("src/classifiers/multilabel_classifier.pkl")
    tag.tags(emb)                      # every leaf over its threshold
    tag.proba(emb, level="stem")       # the 11 families, by grouped max
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from data.moisesdb import MoisesDBRecursiveDataset
from classifiers.stem_classifier import (
    LAMBDAS,
    build_effects,
    macro_metrics,
    train_head,
    tune_thresholds,
)

DATA_CONFIG = "configs/data/seg.yaml"
DEFAULT_PCA = "src/query_conditioned_bslocoformer/pca_params_train.pkl"
DEFAULT_PATH = Path(__file__).with_name("multilabel_classifier.pkl")


# =====================================================================
# 1. THE DATA -- SEG's loader, yielding the MIXTURE and what is in it
# =====================================================================

class MoisesDBMixtureTaggerDataset(MoisesDBRecursiveDataset):
    """SEG's recursive dataset with one thing changed: what an item is.

    Inherits the songs, crop offset, gain, activity gating, merges, level
    draw and unit dropout unchanged; drops the recursive partition since
    this task has no steps.

    Yields `(units [U, C, T], leaf_sets, song_id)`:

        units      slots at the drawn level, distorted per slot; the
                   mixture is `units.sum(0)`, taken after effects.
        leaf_sets  `|`-joined leaf names carried by each unit.
        song_id    so `lam` and thresholds can be chosen on held-out songs.
    """

    def __iter__(self):
        import soundfile as sf

        from source_embedding_generator.protocol import LEVEL_STEM

        worker_info = torch.utils.data.get_worker_info()
        wid = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng([wid, self.root_seed])

        while True:
            try:
                loaded = None
                for _ in range(self.max_retries):
                    song_id = self._pick_song(rng)
                    loaded = self._load_song_segment(song_id, rng)
                    if loaded is not None:
                        break
                if loaded is None:
                    continue
                # Leaves of the loaded tracks, before dropout.
                all_leaves = list(getattr(self, "last_leaves", None) or ())
                level = self._choose_level(rng)
                tracks, groups = self._drop_units(*loaded, level, rng)
                kept = list(getattr(self, "last_kept_idx", None) or ())
                if (all_leaves and kept and len(kept) == len(tracks)
                        and max(kept) < len(all_leaves)):
                    leaves = [all_leaves[i] for i in kept]
                else:
                    # No recorded leaves would silently coarsen the fine
                    # target; skip the crop instead.
                    continue

                if level == LEVEL_STEM:
                    by_family: dict[str, list[int]] = {}
                    for i, g in enumerate(groups):
                        by_family.setdefault(g, []).append(i)
                    idx = list(by_family.values())
                    units = torch.stack([tracks[i].sum(0) for i in idx])
                    sets = ["|".join(sorted({leaves[i] for i in u})) for u in idx]
                else:
                    units, sets = tracks, list(leaves)

                yield units, sets, song_id

            except sf.LibsndfileError:
                print("[warn] Soundfile error.")
                continue


def collate_mixtures(batch):
    """-> (mixtures [B, C, T], leaf-set strings [B], song ids [B]).

    The sum happens here, after the effect wrapper's per-slot draw, matching
    how SEG's `train_step` builds the mixture.
    """
    mixes = torch.stack([it[0].sum(0) for it in batch])
    sets = ["|".join(sorted({l for s in it[1] for l in s.split("|") if l}))
            for it in batch]
    songs = [it[2] for it in batch]
    return mixes, sets, songs


def build_train_loader(config_path, batch_size, num_workers, seed=0):
    """The SEG training dataloader, with the item swapped and nothing else.

    Only `train.dataset._target_` is overridden; every other setting comes
    from the config SEG trains on.
    """
    import hydra.utils as hu
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    OmegaConf.update(
        cfg, "train.dataset._target_",
        "classifiers.multilabel_classifier.MoisesDBMixtureTaggerDataset")
    ds = hu.instantiate(cfg.train)
    if hasattr(ds, "seed"):
        ds.seed = seed
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_mixtures,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    ), cfg


def load_pca(pca_pkl):
    """The query-space projection as numpy: `(x - mean) @ components.T`.

    Applied after extraction, so one cache of raw PaSST embeddings serves
    both the 128-d query space and `--no_pca`, and the PCA can be changed
    without paying for PaSST again.
    """
    with open(pca_pkl, "rb") as f:
        prm = pickle.load(f)
    mean = np.asarray(prm["mean"], dtype=np.float32)
    comp = np.asarray(prm["components"], dtype=np.float32)
    return lambda X: (X.astype(np.float32) - mean) @ comp.T


@torch.no_grad()
def extract_train(loader, passt, pca, n_examples, device, tag="train"):
    """-> (X [N, D], leaf-set strings, song ids). One embedding per MIXTURE."""
    X, sets, songs = [], [], []
    for mixes, s, g in loader:
        X.append(pca(passt(mixes.to(device))).float().cpu().numpy())
        sets += s
        songs += g
        if len(songs) % 512 < len(s):
            print(f"  [{tag}] {len(songs)}/{n_examples}", flush=True)
        if len(songs) >= n_examples:
            break
    return np.concatenate(X)[:n_examples], np.array(sets[:n_examples]), \
        np.array(songs[:n_examples])


@torch.no_grad()
def extract_val(ds, passt, pca, n_crops, device, seed=1, tag="val", fx=None,
                batch_size=8):
    """Every val song, `n_crops` times: the full segment mixture and its leaves.

    Deterministic coverage rather than the infinite training iterator, so
    the split is the same test set every run. Skips the level draw and
    dropout, since a val mixture should be the whole segment.

    `fx` distorts each track independently before the sum, for the distorted
    condition; the clean condition passes None.
    """
    rng = np.random.default_rng(seed)
    mixes, sets, songs = [], [], []
    for ci in range(n_crops):
        for song_id in ds.song_ids:
            loaded = ds._load_song_segment(song_id, rng)
            if loaded is None:
                continue
            tracks, groups = loaded
            leaves = list(getattr(ds, "last_leaves", None) or ())
            if len(leaves) != len(tracks):
                continue
            if fx is not None and fx.chain is not None:
                tracks = fx._distort(tracks)
            mixes.append(tracks.sum(0))
            sets.append("|".join(sorted(set(leaves))))
            songs.append(song_id)
        print(f"  [{tag}] crop {ci + 1}/{n_crops}: {len(mixes)} mixtures",
              flush=True)
    X = []
    for i in range(0, len(mixes), batch_size):
        batch = torch.stack(mixes[i:i + batch_size]).to(device)
        X.append(pca(passt(batch)).float().cpu().numpy())
    return np.concatenate(X), np.array(sets), np.array(songs)


# =====================================================================
# 2. THE CLI -- extract, fit, report, save
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_config", default=DATA_CONFIG,
                   help="the SEG data config. Its `train` node is the loader "
                        "and its `val` node the held-out split.")
    p.add_argument("--n_train", type=int, default=20000,
                   help="mixtures to extract. One per item of the SEG loader.")
    p.add_argument("--n_crops_val", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="for PaSST during extraction")
    p.add_argument("--head_device", default="cpu",
                   help="the head is 128x35; a GPU buys nothing")
    p.add_argument("--out", default=str(DEFAULT_PATH))
    p.add_argument("--pca_pkl", default=DEFAULT_PCA,
                   help="the PCA the query space is defined by. MUST match the "
                        "one the benchmark runner embeds mixtures with.")
    p.add_argument("--no_pca", action="store_true",
                   help="fit on the raw 768-d PaSST embedding. A larger input "
                        "space, and the baseline at its strongest -- but then "
                        "the pickle is NOT readable by anything expecting the "
                        "128-d query space.")
    p.add_argument("--cache", default=None,
                   help="npz of extracted embeddings; reused if it exists")
    p.add_argument("--vocal_subcategories", action="store_true",
                   help="keep the 6 vocal leaves as separate fine tags. OFF by "
                        "default, for the reason `stem_classifier` gives: the "
                        "loaders sum the whole vocals family into one source.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Line-buffer stdout so progress prints promptly under redirection.
    sys.stdout.reconfigure(line_buffering=True)

    from data.moisesdb import LEAF_TO_GROUP

    # ---- taxonomy, verbatim from `stem_classifier` -------------------
    # Must match `stem_classifier`'s collapse or the two baselines are
    # scored on different label spaces.
    VOCALS = "vocals"
    if args.vocal_subcategories:
        leaf_map = {l: l for l in LEAF_TO_GROUP}
    else:
        leaf_map = {l: (VOCALS if g == VOCALS else l)
                    for l, g in LEAF_TO_GROUP.items()}
    LEAF_TO_GROUP_CLF = {t: LEAF_TO_GROUP[l] for l, t in leaf_map.items()}
    families = sorted(set(LEAF_TO_GROUP_CLF.values()))
    leaves = sorted(set(LEAF_TO_GROUP_CLF))
    child_index = [[i for i, l in enumerate(leaves) if LEAF_TO_GROUP_CLF[l] == f]
                   for f in families]
    print(f"[taxonomy] {len(leaves)} fine tags, {len(families)} families")

    # ---- embeddings --------------------------------------------------
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        print(f"[cache] loading {cache}")
        z = np.load(cache, allow_pickle=True)
        Xtr, qtr, gtr = z["Xtr"], z["qtr"], z["gtr"]
        Xva, qva, gva = z["Xva"], z["qva"], z["gva"]
        Xvf, qvf = z["Xvf"], z["qvf"]
    else:
        import hydra.utils as hu
        from omegaconf import OmegaConf

        from query_conditioned_bslocoformer.query_model import PasstWrapper

        device = torch.device(args.device)
        passt = PasstWrapper().eval().to(device)
        # Raw 768 into the cache; the projection happens below.
        pca = torch.nn.Identity()

        loader, cfg = build_train_loader(args.data_config, args.batch_size,
                                         args.num_workers, seed=args.seed)
        print(f"[extract] train mixtures from {args.data_config} "
              f"(p_apply={cfg.train.p_apply}, {len(loader.dataset.song_ids)} songs)")
        Xtr, qtr, gtr = extract_train(loader, passt, pca, args.n_train, device)
        del loader

        val_ds = hu.instantiate(OmegaConf.load(args.data_config).val)
        print(f"[extract] val split, clean ({len(val_ds.song_ids)} songs)")
        Xva, qva, gva = extract_val(val_ds, passt, pca, args.n_crops_val, device,
                                    seed=1, batch_size=args.batch_size)
        # A second val condition with every track distorted, matching
        # real inference audio.
        print("[extract] val split, fully distorted")
        Xvf, qvf, _ = extract_val(val_ds, passt, pca, args.n_crops_val, device,
                                  seed=1, tag="val_fx",
                                  fx=build_effects(args.data_config),
                                  batch_size=args.batch_size)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, Xtr=Xtr, qtr=qtr, gtr=gtr, Xva=Xva, qva=qva,
                     gva=gva, Xvf=Xvf, qvf=qvf)
            print(f"[cache] wrote {cache}")

    # ---- targets -----------------------------------------------------
    li = {l: i for i, l in enumerate(leaves)}
    fi = {f: i for i, f in enumerate(families)}

    def targets(q):
        """`|`-joined leaf sets -> (Y_fine [N, 35], Y_coarse [N, 11]).

        Coarse is derived from fine, so the hierarchy constraint always holds.
        """
        Yf = np.zeros((len(q), len(leaves)), dtype=np.float32)
        Yc = np.zeros((len(q), len(families)), dtype=np.float32)
        for i, s in enumerate(q):
            for leaf in str(s).split("|"):
                tag = leaf_map.get(leaf, leaf)
                if tag in li:
                    Yf[i, li[tag]] = 1.0
                    Yc[i, fi[LEAF_TO_GROUP_CLF[tag]]] = 1.0
        return Yf, Yc

    if args.no_pca:
        print(f"[repr] raw PaSST-{Xtr.shape[1]}")
    else:
        print(f"[repr] PaSST -> PCA ({args.pca_pkl})")
        proj = load_pca(args.pca_pkl)
        Xtr, Xva, Xvf = proj(Xtr), proj(Xva), proj(Xvf)

    Yf_tr, Yc_tr = targets(qtr)
    Yf_va, Yc_va = targets(qva)
    Yf_vf, Yc_vf = targets(qvf)
    print(f"[targets] {Xtr.shape[0]} train mixtures, "
          f"{Yf_tr.sum(1).mean():.2f} leaves each on average")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zva, Zvf = (Xtr - mu) / sd, (Xva - mu) / sd, (Xvf - mu) / sd

    # ---- lambda and thresholds, on songs held out of TRAIN ------------
    songs = np.array(sorted(set(gtr)))
    order = np.random.default_rng(0).permutation(len(songs))
    hold = set(songs[order[: max(1, len(songs) // 5)]])
    vm = np.array([g in hold for g in gtr])
    print(f"\n[head] {len(songs) - len(hold)} train / {len(hold)} internal-val "
          f"songs, {(~vm).sum()} / {vm.sum()} mixtures")

    # Clipped inverse-frequency weighting, as in `stem_classifier`.
    freq = Yf_tr[~vm].mean(0)
    pos_w = np.clip(1.0 / np.clip(freq, 1e-4, None), 1.0, 50.0)

    best = (float("inf"), None, None)
    for lam in LAMBDAS:
        m, vloss = train_head(
            Ztr[~vm], Yf_tr[~vm], Yc_tr[~vm], Ztr[vm], Yf_tr[vm], Yc_tr[vm],
            child_index, lam=lam, epochs=args.epochs, pos_weight=pos_w,
            device=args.head_device)
        print(f"  lambda={lam:.2f}  internal val fine BCE {vloss:.4f}")
        if vloss < best[0]:
            best = (vloss, lam, m)
    vloss, lam, m = best
    print(f"  selected lambda={lam:.2f} (fine val BCE {vloss:.4f})")

    dev = torch.device(args.head_device)
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    with torch.no_grad():
        pf_iv, pc_iv = m(as_t(Ztr[vm]))
        th_f = tune_thresholds(pf_iv.cpu().numpy(), Yf_tr[vm])
        th_c = tune_thresholds(pc_iv.cpu().numpy(), Yc_tr[vm])

        print(f"\n[val] {len(set(gva))} held-out no-bleed songs, "
              f"{len(Xva)} mixtures")
        for tag, Z, Yf, Yc in [("clean", Zva, Yf_va, Yc_va),
                               ("distorted", Zvf, Yf_vf, Yc_vf)]:
            pf, pc = m(as_t(Z))
            pf, pc = pf.cpu().numpy(), pc.cpu().numpy()
            rf, rc = macro_metrics(pf, Yf, th_f), macro_metrics(pc, Yc, th_c)
            # Micro set P/R/F1 is the headline, matching `discovery_prf.py`,
            # computed at the benchmark's own operating point: fine
            # thresholds folded up through `leaf_to_group` (only the fine
            # thresholds are saved in the blob).
            up = np.zeros(Yc.shape, dtype=bool)
            for j, l in enumerate(leaves):
                up[:, fi[LEAF_TO_GROUP_CLF[l]]] |= (pf[:, j] > th_f[j])
            tp = float((up & (Yc > 0)).sum())
            pr = tp / max(up.sum(), 1)
            rc_ = tp / max(Yc.sum(), 1)
            print(f"  {tag:10s} micro P {pr:.3f} R {rc_:.3f} "
                  f"F1 {2 * pr * rc_ / max(pr + rc_, 1e-9):.3f} "
                  f"(leaf thresholds folded up, as the benchmark runs it)  |  "
                  f"coarse ROC {rc[0]:.3f} PR {rc[1]:.3f} macroF1 {rc[2]:.3f} "
                  f"({rc[3]} tags)  |  "
                  f"fine ROC {rf[0]:.3f} PR {rf[1]:.3f} macroF1 {rf[2]:.3f}")

    # ---- save --------------------------------------------------------
    # Same blob layout as `stem_classifier`, so `StemClassifier` loads it
    # unchanged.
    blob = {
        "mean": mu, "scale": sd,
        "W": m.fine.weight.detach().cpu().numpy().T,
        "b": m.fine.bias.detach().cpu().numpy(),
        "leaves": leaves, "families": families,
        "leaf_to_group": LEAF_TO_GROUP_CLF,
        "thresholds": th_f,
        "meta": {
            "kind": "mixture-tagger",
            "method": "multi-label tagger on mixture embeddings; head and "
                      "objective as arXiv:2302.08136 (linear fine head + "
                      "grouped max pooling)",
            "lambda": lam,
            "query": ("PasstWrapper (raw 768)" if args.no_pca else
                      "PasstWrapper -> PcaAdapter (PaSST-PCA-128), unnormalised"),
            "pca_pkl": None if args.no_pca else args.pca_pkl,
            "data_config": args.data_config,
            "trained_on": "MIXTURES from the SEG train dataloader, not isolated "
                          "stems -- see the module docstring",
            "n_examples_train": int(Xtr.shape[0]),
            "n_songs_train": int(len(set(gtr))),
            "dim": int(Xtr.shape[1]),
            "vocal_subcategories": bool(args.vocal_subcategories),
            "n_fine_tags": len(leaves),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(blob, f)
    print(f"\n[saved] {args.out}  (lambda={lam})")


if __name__ == "__main__":
    main()

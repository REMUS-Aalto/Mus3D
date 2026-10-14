"""Hierarchical probe naming a source from its PaSST-PCA-128 embedding.

    from classifiers.stem_classifier import StemClassifier
    clf = StemClassifier.load()
    clf.predict(emb, level="stem")     # ['drums', 'bass', 'vocals', ...]
    clf.predict(emb, level="track")    # ['hi_hat', 'bass_guitar', ...]
    clf.tags(emb)                      # every fine tag over its threshold

Following Chen et al., "An Attention-based Approach to Hierarchical
Multi-label Music Instrument Classification" (arXiv:2302.08136); see the
paper for the model, taxonomy and deviations from that reference.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

#: The paper's lambda grid, verbatim.
LAMBDAS = (0.70, 0.75, 0.80, 0.85, 0.90)

#: dataset root, as prepared by `dataset/prepare_datasets.sh`
DATASET_ROOT = os.environ.get("DATASET_ROOT", "data")
TRAIN_INFO = "dataset/splits/moisesdb_medleydb_albumdb_train.json"
VAL_INFO = "dataset/splits/moisesdb-val-no_bleed.json"
FX_CONFIG = "configs/data/seg.yaml"
DEFAULT_PATH = Path(__file__).with_name("stem_classifier.pkl")
#: The PCA the query space is defined by. Overridable, and NOT the default the
#: newer configs carry (`pca_params_train.pkl`, fit on the train split only):
#: kept as-is so re-running this file reproduces the existing probes, while a
#: probe meant for a `pca_params_train` generator passes `--pca_pkl`.
DEFAULT_PCA = "src/query_conditioned_bslocoformer/pca_params_train.pkl"


# =====================================================================
# 1. THE MODEL
# =====================================================================

class HierarchicalHead(nn.Module):
    """A linear fine-level classifier, with coarse tags pooled from its children.

    `child_index[c]` lists the fine-tag rows belonging to coarse tag c. It is
    passed in rather than derived so the tag ORDER is fixed by the caller and
    the same order reaches the saved probe.
    """

    def __init__(self, dim: int, n_fine: int, child_index: list[list[int]]):
        super().__init__()
        self.fine = nn.Linear(dim, n_fine)
        self.child_index = child_index

    def forward(self, x):
        p_fine = torch.sigmoid(self.fine(x))
        p_coarse = torch.stack([p_fine[:, idx].max(dim=1).values
                                for idx in self.child_index], dim=1)
        return p_fine, p_coarse


# =====================================================================
# 2. FITTING IT
# =====================================================================

def train_head(Xtr, Yf_tr, Yc_tr, Xva, Yf_va, Yc_va, child_index, lam=0.85,
               epochs=800, lr=3e-3, wd=1e-3, pos_weight=None, device="cpu",
               seed=0):
    """Fit one head at one lambda. -> (model, best fine-level val loss).

    Full-batch Adam. Returned loss is the fine-level loss alone, since it
    is what selects lambda, and scoring the combined loss would bias
    toward a large (1 - lam).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    Xtr_t, Xva_t = t(Xtr), t(Xva)
    Yf_t, Yc_t, Yf_v = t(Yf_tr), t(Yc_tr), t(Yf_va)

    m = HierarchicalHead(Xtr.shape[1], Yf_tr.shape[1], child_index).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

    # Per-tag positive weighting for class balancing across a heavy-tailed
    # tag frequency distribution.
    pw = None if pos_weight is None else t(pos_weight)

    def bce(p, y, w=None):
        p = p.clamp(1e-6, 1 - 1e-6)
        return -(y * p.log() * (1.0 if w is None else w)
                 + (1 - y) * (1 - p).log()).mean()

    best_loss, best_state = float("inf"), None
    for ep in range(epochs):
        m.train()
        pf, pc = m(Xtr_t)
        loss = lam * bce(pf, Yf_t, pw) + (1 - lam) * bce(pc, Yc_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0:
            m.eval()
            with torch.no_grad():
                v = bce(m(Xva_t)[0], Yf_v).item()
            if v < best_loss:
                best_loss = v
                best_state = {k: p.detach().clone() for k, p in m.state_dict().items()}
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best_loss


def tune_thresholds(p, y, grid=np.arange(0.05, 0.96, 0.05)):
    """Per-tag thresholds maximising F1 on validation -- the paper's protocol."""
    th = np.full(p.shape[1], 0.5)
    for j in range(p.shape[1]):
        if y[:, j].sum() == 0:
            continue
        best_f1 = -1.0
        for cut in grid:
            pred = p[:, j] > cut
            tp = float((pred & (y[:, j] > 0)).sum())
            if tp == 0:
                continue
            prec, rec = tp / max(pred.sum(), 1), tp / max(y[:, j].sum(), 1)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, th[j] = f1, float(cut)
    return th


def macro_metrics(p, y, th):
    """-> (macro ROC-AUC, PR-AUC, F1, n scored) -- the paper's three metrics.

    Tags absent from the evaluation split are skipped, not scored as zero.
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    roc, pr, f1 = [], [], []
    for j in range(y.shape[1]):
        if y[:, j].sum() in (0, len(y)):
            continue
        roc.append(roc_auc_score(y[:, j], p[:, j]))
        pr.append(average_precision_score(y[:, j], p[:, j]))
        f1.append(f1_score(y[:, j], p[:, j] > th[j], zero_division=0))
    return float(np.mean(roc)), float(np.mean(pr)), float(np.mean(f1)), len(roc)


# =====================================================================
# 3. THE FITTED PROBE -- what everything else calls
# =====================================================================

class StemClassifier:
    """The fitted probe. Consumes PaSST-PCA-128 exactly as SEG produces it:
    `query_adapter(query_model(audio))`, unnormalised.
    """

    def __init__(self, blob: dict):
        self.mean = np.asarray(blob["mean"], dtype=np.float64)
        self.scale = np.asarray(blob["scale"], dtype=np.float64)
        self.W = np.asarray(blob["W"], dtype=np.float64)             # [128, 40]
        self.b = np.asarray(blob["b"], dtype=np.float64)             # [40]
        self.leaves: list[str] = list(blob["leaves"])
        self.families: list[str] = list(blob["families"])
        self.leaf_to_group: dict[str, str] = dict(blob["leaf_to_group"])
        self.thresholds = np.asarray(blob["thresholds"], dtype=np.float64)
        self.meta: dict = dict(blob.get("meta", {}))
        self._child = [
            np.array([i for i, l in enumerate(self.leaves)
                      if self.leaf_to_group[l] == f], dtype=np.int64)
            for f in self.families
        ]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "StemClassifier":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def check_basis(self, model, *, where: str = "") -> None:
        """Refuse a head fitted in a different PCA basis than `model` emits.

        A rotated basis still returns a confident name for every embedding,
        just the wrong one, so this compares the basis file the head names
        against the one the model's `PcaAdapter` loaded and raises on a
        mismatch rather than silently mis-scoring.
        """
        # A head with no recorded basis predates the basis being pinned,
        # so it is treated as the old default rather than unknown.
        LEGACY = "pca_params.pkl"
        want = (self.meta or {}).get("pca_pkl") or LEGACY
        got = getattr(getattr(model, "query_adapter", None), "pca_pkl", None)
        tag = f" [{where}]" if where else ""
        if got is None:
            print(f"[clf]{tag} head wants {want}; the model does not record "
                  f"which basis it loaded, so the pairing is UNCHECKED.")
            return
        if Path(want).name != Path(got).name:
            how = ("" if (self.meta or {}).get("pca_pkl") else
                   " (the head records no pca_pkl, so it is taken to be the "
                   "pre-pinning default)")
            raise ValueError(
                f"classifier/model PCA basis mismatch{tag}: the head was fitted "
                f"on {want}{how}, the model emits {got}. The head would name "
                f"every embedding confidently and wrongly. Pass the classifier "
                f"whose meta names {Path(got).name}.")
        print(f"[clf]{tag} basis agrees ({Path(want).name})")

    def proba(self, emb: np.ndarray, level: str = "stem"):
        """[N, 128] -> ([N, C] per-tag probabilities, tag names).

        Sigmoid, one per tag; does not sum to 1, since a family sum can
        genuinely carry several fine tags at once.
        """
        emb = np.asarray(emb, dtype=np.float64)
        if emb.ndim == 1:
            emb = emb[None]
        z = (emb - self.mean) / self.scale @ self.W + self.b
        p_fine = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        if level == "track":
            return p_fine, self.leaves
        if level != "stem":
            raise ValueError(f"level must be 'stem' or 'track', got {level!r}")
        return np.stack([p_fine[:, c].max(1) for c in self._child], 1), self.families

    def predict(self, emb: np.ndarray, level: str = "stem") -> list[str]:
        """The single most probable tag per embedding -- one source, one name."""
        p, names = self.proba(emb, level)
        return [names[i] for i in p.argmax(1)]

    def predict_family(self, emb: np.ndarray) -> list[str]:
        """The coarse label whatever the level."""
        return self.predict(emb, "stem")

    def tags(self, emb: np.ndarray) -> list[list[str]]:
        """The multi-label reading: every fine tag over its tuned threshold."""
        p, names = self.proba(emb, "track")
        return [[names[j] for j in np.nonzero(row > self.thresholds)[0]]
                for row in p]


def naming_report(pred: list[str], true: list[str]) -> dict:
    """Predicted names for n extractions vs the song's true unit labels.

        recall    fraction of the song's distinct labels claimed by at
                  least one extraction (a detection failure, not SI-SDR).
        split     fraction of extractions sharing their label with another.
        spurious  fraction of extractions whose label is not in the song.

    Needs only reference labels, not audio, so it is cheap to log every step.
    """
    n = len(pred)
    if n == 0:
        return {"recall": 0.0, "split": 0.0, "spurious": 0.0}
    truth = set(true)
    seen: dict[str, int] = {}
    for p in pred:
        seen[p] = seen.get(p, 0) + 1
    return {
        "recall": (float(np.mean([t in seen for t in truth])) if truth else float("nan")),
        "split": float(sum(c for c in seen.values() if c > 1)) / n,
        "spurious": float(sum(c for p, c in seen.items() if p not in truth)) / n,
    }


# =====================================================================
# 4. THE TRAINING SET -- embeddings of real slots, drawn like SEG's own
# =====================================================================

def build_dataset(dset_info: str, dset_dir: str, gain_db: float):
    """The SEG loader, with dropout off: every surviving track is an example.

    Same loader SEG trains on, so embeddings come from the distribution
    the generator learned to model.
    """
    from data.moisesdb import MoisesDBRecursiveDataset

    return MoisesDBRecursiveDataset(
        dset_dir=dset_dir, sample_size=441000, sample_rate=44100, min_rms=1e-4,
        rel_floor_db=-50.0, min_active_tracks=2, merge_same_leaf=True,
        p_track_dropout=0.0, gain_db=gain_db, dset_info=dset_info,
    )


def build_effects(config_path: str = FX_CONFIG):
    """The FX wrapper from the SEG data config, for its chain only.

    Fits the probe on the same distorted-audio distribution SEG trains on.
    Instantiated whole since the chain is not separable from the wrapper's
    `__init__`; only `_distort` and `p_apply` are used.
    """
    import hydra.utils as hu
    from omegaconf import OmegaConf

    return hu.instantiate(OmegaConf.load(config_path).train)


@torch.no_grad()
def extract(ds, passt, pca, n_crops: int, device, seed: int = 0, tag: str = "",
            fx=None, p_apply: float | None = None):
    """-> (X [N,128], leaves, families, is_family_sum, songs, leaf SETS).

    Each crop yields every surviving track and the sum of each level-1
    family. The last return value is the multi-label fine target as a
    `|`-joined string. `fx`, if given, applies the augmentation chain per
    slot, matching training.
    """
    rng = np.random.default_rng(seed)
    X, leaves, fams, is_sum, songs, sets = [], [], [], [], [], []
    for ci in range(n_crops):
        for song_id in ds.song_ids:
            loaded = ds._load_song_segment(song_id, rng)
            if loaded is None:
                continue
            tracks, groups = loaded
            lv = list(getattr(ds, "last_leaves", groups))
            if len(lv) != len(tracks):
                # A missing `last_leaves` would otherwise fall back to
                # coarse labels; skip the crop instead.
                continue

            by_family: dict[str, list[int]] = {}
            for i, g in enumerate(groups):
                by_family.setdefault(g, []).append(i)

            # One PaSST pass per crop: the tracks, then one summed slot per
            # family. Batched together because PaSST is the whole cost here.
            units = [tracks[i] for i in range(len(tracks))]
            unit_leaf, unit_fam = list(lv), list(groups)
            unit_sum = [False] * len(tracks)
            unit_set = list(lv)
            for fam, idx in by_family.items():
                units.append(tracks[idx].sum(0))
                unit_leaf.append("")               # a family sum has no ONE leaf
                unit_fam.append(fam)
                unit_sum.append(True)
                unit_set.append("|".join(sorted({lv[i] for i in idx})))

            audio = torch.stack(units)
            if fx is not None and fx.chain is not None:
                if rng.random() < (fx.p_apply if p_apply is None else p_apply):
                    audio = fx._distort(audio)
            X.append(pca(passt(audio.to(device))).float().cpu().numpy())
            leaves += unit_leaf
            fams += unit_fam
            is_sum += unit_sum
            songs += [song_id] * len(unit_fam)
            sets += unit_set
        print(f"  [{tag}] crop {ci + 1}/{n_crops}: "
              f"{sum(len(x) for x in X)} embeddings", flush=True)
    return (np.concatenate(X), np.array(leaves), np.array(fams),
            np.array(is_sum), np.array(songs), np.array(sets))


def shaped_noise(X, scale, rng, npz="output/query_noise.npz"):
    """Perturb like a generated query does, not like white noise.

    Uses a fitted anisotropic covariance of within-source displacement,
    applied in unit-norm space then rescaled back. Returns None if the
    npz is missing.
    """
    if not Path(npz).exists():
        return None
    q = np.load(npz)
    ev, V = q["eigvals"], q["eigvecs"]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    g = (rng.normal(size=(X.shape[0], len(ev))) * np.sqrt(np.maximum(ev, 0))) @ V.T
    p = X / np.clip(n, 1e-12, None) + scale * g
    return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None) * n


# =====================================================================
# 5. THE CLI -- extract, fit, report, save
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dset_dir", default=DATASET_ROOT)
    p.add_argument("--train_info", default=TRAIN_INFO)
    p.add_argument("--val_info", default=VAL_INFO)
    p.add_argument("--n_crops", type=int, default=6)
    p.add_argument("--n_crops_val", type=int, default=3)
    p.add_argument("--gain_db", type=float, default=6.0)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="for PaSST during extraction")
    p.add_argument("--head_device", default="cpu",
                    help="the head is 128x40; a GPU buys nothing")
    p.add_argument("--out", default=str(DEFAULT_PATH))
    p.add_argument("--pca_pkl", default=DEFAULT_PCA,
                    help="the PCA the query space is defined by. MUST be the "
                         "same one the separator and SEG are configured with "
                         "(`model.query_adapter.pca_pkl`) -- the head is fit in "
                         "these coordinates and a probe fit on one PCA reads "
                         "noise off another. Written into the pickle's meta.")
    p.add_argument("--cache", default=None,
                    help="npz of extracted embeddings; reused if it exists")
    p.add_argument("--effects_config", default=FX_CONFIG,
                    help="data config whose `train` node carries the FX chain")
    p.add_argument("--no_effects", action="store_true",
                    help="fit on clean slots. Mismatches SEG's training data.")
    p.add_argument("--vocal_subcategories", action="store_true",
                    help="keep the 6 vocal leaves as separate fine tags. OFF by "
                         "default since 2026-08-17: the loaders now sum the whole "
                         "vocals family into one source, so the sub-categories "
                         "have no distinct embeddings left to name.")
    args = p.parse_args()

    from data.moisesdb import LEAF_TO_GROUP

    # ---- taxonomy ---------------------------------------------------
    # The probe's fine label space: the dataset's leaf set, except the
    # `vocals` family collapses to a single `vocals` tag (matches
    # `MoisesDBRecursiveDataset._merge_vocals`). Only the fine level
    # changes; `vocals` remains one of the 11 families.
    VOCALS = "vocals"
    if args.vocal_subcategories:
        leaf_map = {l: l for l in LEAF_TO_GROUP}
    else:
        leaf_map = {l: (VOCALS if g == VOCALS else l)
                    for l, g in LEAF_TO_GROUP.items()}
    LEAF_TO_GROUP_CLF = {t: LEAF_TO_GROUP[l] for l, t in leaf_map.items()}
    _n_vocal = sum(1 for g in LEAF_TO_GROUP.values() if g == VOCALS)
    print(f"[taxonomy] {len(LEAF_TO_GROUP_CLF)} fine tags"
          + ("" if args.vocal_subcategories else
             f" ({_n_vocal} vocal leaves collapsed into one '{VOCALS}' tag, "
             f"{len(LEAF_TO_GROUP)} -> {len(LEAF_TO_GROUP_CLF)})"))

    # ---- embeddings -------------------------------------------------
    # The expensive half (PaSST over every track of every crop). Cached so
    # re-running the head costs nothing.
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        print(f"[cache] loading {cache}")
        z = np.load(cache, allow_pickle=True)
        Xtr, ftr, gtr, qtr, str_ = z["Xtr"], z["ftr"], z["gtr"], z["qtr"], z["str"]
        Xva, lva, fva, sva, gva, qva = (z["Xva"], z["lva"], z["fva"], z["sva"],
                                        z["gva"], z["qva"])
        Xvf, lvf, fvf, svf, qvf = z["Xvf"], z["lvf"], z["fvf"], z["svf"], z["qvf"]
    else:
        from query_conditioned_bslocoformer.query_model import PasstWrapper, PcaAdapter

        device = torch.device(args.device)
        print(f"[pca] {args.pca_pkl}")
        passt = PasstWrapper().eval().to(device)
        pca = PcaAdapter(args.pca_pkl).eval().to(device)
        fx = None if args.no_effects else build_effects(args.effects_config)
        print(f"[effects] {'OFF' if fx is None else args.effects_config}")

        print("[extract] train split")
        Xtr, ltr, ftr, str_, gtr, qtr = extract(
            build_dataset(args.train_info, args.dset_dir, args.gain_db),
            passt, pca, args.n_crops, device, seed=0, tag="train", fx=fx)
        print("[extract] val split (no-bleed), clean")
        Xva, lva, fva, sva, gva, qva = extract(
            build_dataset(args.val_info, args.dset_dir, 0.0),
            passt, pca, args.n_crops_val, device, seed=1, tag="val")
        # A second val set, every slot distorted, matching inference audio.
        print("[extract] val split (no-bleed), fully distorted")
        Xvf, lvf, fvf, svf, gvf, qvf = extract(
            build_dataset(args.val_info, args.dset_dir, 0.0),
            passt, pca, args.n_crops_val, device, seed=1, tag="val_fx",
            fx=fx, p_apply=1.0)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, Xtr=Xtr, ltr=ltr, ftr=ftr, str=str_, gtr=gtr, qtr=qtr,
                     Xva=Xva, lva=lva, fva=fva, sva=sva, gva=gva, qva=qva,
                     Xvf=Xvf, lvf=lvf, fvf=fvf, svf=svf, gvf=gvf, qvf=qvf)
            print(f"[cache] wrote {cache}")

    # ---- targets ----------------------------------------------------
    families = sorted(set(LEAF_TO_GROUP_CLF.values()))
    leaves = sorted(set(LEAF_TO_GROUP_CLF))
    child_index = [[i for i, l in enumerate(leaves) if LEAF_TO_GROUP_CLF[l] == f]
                   for f in families]
    # The recorded true leaf of each val example goes through the same collapse.
    _collapse = np.vectorize(lambda s: leaf_map.get(str(s), str(s)))
    lva, lvf = _collapse(lva), _collapse(lvf)
    li = {l: i for i, l in enumerate(leaves)}
    fi = {f: i for i, f in enumerate(families)}

    def targets(q):
        """`|`-joined leaf sets -> (Y_fine [N, 40], Y_coarse [N, 11]).

        Coarse target is derived from the fine one, so the hierarchy
        constraint always holds.
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

    Yf_tr, Yc_tr = targets(qtr)
    Yf_va, Yc_va = targets(qva)
    Yf_vf, Yc_vf = targets(qvf)

    # Standardise on train. Stored with the model: inference must not re-fit it.
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zva, Zvf = (Xtr - mu) / sd, (Xva - mu) / sd, (Xvf - mu) / sd

    # ---- lambda, on an internal split -------------------------------
    # Split by song, taken out of train, so the no-bleed val split stays
    # untouched for reporting.
    songs = np.array(sorted(set(gtr)))
    order = np.random.default_rng(0).permutation(len(songs))
    hold = set(songs[order[: max(1, len(songs) // 5)]])
    vm = np.array([g in hold for g in gtr])
    print(f"\n[head] {len(songs) - len(hold)} train / {len(hold)} internal-val "
          f"songs, {(~vm).sum()} / {vm.sum()} examples")

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

    # ---- thresholds and the report ----------------------------------
    dev = torch.device(args.head_device)
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    with torch.no_grad():
        pf_iv, pc_iv = m(as_t(Ztr[vm]))
        th_f = tune_thresholds(pf_iv.cpu().numpy(), Yf_tr[vm])
        th_c = tune_thresholds(pc_iv.cpu().numpy(), Yc_tr[vm])

        # `dist+noise` stands in for a generated query rather than a real
        # stem embedding.
        conds = [("clean", Zva, Yf_va, Yc_va, fva, sva, lva),
                 ("distorted", Zvf, Yf_vf, Yc_vf, fvf, svf, lvf)]
        noisy = shaped_noise(Xvf, 1.0, np.random.default_rng(0))
        if noisy is not None:
            conds.append(("dist+noise", (noisy - mu) / sd, Yf_vf, Yc_vf,
                          fvf, svf, lvf))

        print(f"\n[val] {len(set(gva))} held-out no-bleed songs")
        for tag, Z, Yf, Yc, fam, sm, lf in conds:
            pf, pc = m(as_t(Z))
            pf, pc = pf.cpu().numpy(), pc.cpu().numpy()
            rf, rc = macro_metrics(pf, Yf, th_f), macro_metrics(pc, Yc, th_c)
            lt = ~sm
            print(f"  {tag:10s} family {np.mean([families[i] for i in pc.argmax(1)] == fam):.3f}"
                  f"  leaf@track {np.mean([leaves[i] for i in pf[lt].argmax(1)] == lf[lt]):.3f}"
                  f"  |  coarse ROC {rc[0]:.3f} PR {rc[1]:.3f} F1 {rc[2]:.3f} ({rc[3]} tags)"
                  f"  |  fine ROC {rf[0]:.3f} PR {rf[1]:.3f} F1 {rf[2]:.3f} ({rf[3]} tags)")

    # ---- save -------------------------------------------------------
    blob = {
        "mean": mu, "scale": sd,
        "W": m.fine.weight.detach().cpu().numpy().T,   # [128, n_fine_tags]
        "b": m.fine.bias.detach().cpu().numpy(),
        "leaves": leaves, "families": families,
        # The collapsed map, matching the fine tags the head was fit with.
        "leaf_to_group": LEAF_TO_GROUP_CLF,
        "thresholds": th_f,
        "meta": {
            "method": "arXiv:2302.08136, linear fine head + grouped max pooling",
            "lambda": lam,
            "query": "PasstWrapper -> PcaAdapter (PaSST-PCA-128), unnormalised",
            # The 128-d space is only defined relative to this PCA file.
            "pca_pkl": args.pca_pkl,
            # Recorded from the data, not the CLI, since a cached run does
            # not re-extract.
            "n_examples_train": int(Xtr.shape[0]),
            "n_songs_train": int(len(set(gtr))),
            "effects": (None if args.no_effects else args.effects_config),
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

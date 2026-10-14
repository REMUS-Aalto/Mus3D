"""Metrics for generated source-embedding sets (PaSST -> PCA-128, MoisesDB).

Pure function library: everything takes `[B,N,D]` predictions/targets and
a `[B,N]` bool mask over valid slots. Similarity numbers should be read
against `baseline_cosines`, since emitting the mixture embedding in every
slot is a strong trivial baseline.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_BIG = 1e6


# --------------------------------------------------------------------------
# assignment helpers
# --------------------------------------------------------------------------
def masked_cost(cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Make padded slots map to themselves at zero cost, and nothing else.

    Lets every item in a batch solve the assignment at the same [N,N] shape
    regardless of how many valid sources it has.
    """
    b, n, _ = cost.shape
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    both_pad = (~mask).unsqueeze(2) & (~mask).unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool, device=cost.device).unsqueeze(0).expand(b, n, n)
    out = torch.where(valid_pair, cost, torch.full_like(cost, _BIG))
    return torch.where(both_pad & eye, torch.zeros_like(cost), out)


def assignment_from_cost(cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hungarian/brute-force assignment on an arbitrary [B,N,N] cost matrix."""
    cost = masked_cost(cost.float(), mask)
    b, n, _ = cost.shape
    if n <= 6:
        perms = torch.tensor(
            list(itertools.permutations(range(n))), device=cost.device, dtype=torch.long
        )
        rows = torch.arange(n, device=cost.device).view(1, n).expand(perms.shape[0], n)
        return perms[cost[:, rows, perms].sum(dim=-1).argmin(dim=1)]

    # N routinely exceeds 6 here, so brute force is out and this is the
    # normal path.
    from scipy.optimize import linear_sum_assignment

    cpu = cost.detach().cpu().numpy()
    out = torch.empty(b, n, dtype=torch.long)
    for i in range(b):
        r, c = linear_sum_assignment(cpu[i])
        out[i, torch.as_tensor(r)] = torch.as_tensor(c)
    return out.to(cost.device)


def partial_assignment(pred: np.ndarray, true: np.ndarray, floor: float = 0.5):
    """Optimal partial assignment, extraction -> true unit. -> (pairs, cosines).

    Rectangular by construction: `pred` and `true` need not be the same
    length, and unmatched leftovers on either side are left unclaimed
    rather than force-paired, so an extraction with no plausible match
    stays unmatched instead of stealing a true unit's correct pair.
    Implemented as one augmented `linear_sum_assignment`, with a
    per-extraction dummy column priced at `floor` so any pair below it
    is rejected; still globally optimal, not a post-hoc filter.
    """
    from scipy.optimize import linear_sum_assignment

    e = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    t = true / (np.linalg.norm(true, axis=-1, keepdims=True) + 1e-9)
    cos = e @ t.T
    n, U = cos.shape
    aug = np.full((n, U + n), float(floor), dtype=cos.dtype)
    aug[:, :U] = cos
    r_, c_ = linear_sum_assignment(-aug)
    return ({int(a): int(b) for a, b in zip(r_, c_) if int(b) < U}, cos)

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid slots, per item, then over the batch."""
    n_valid = mask.sum(dim=1).clamp(min=1)
    return ((values * mask).sum(dim=1) / n_valid).mean()


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------
def cosine_matrix(a: torch.Tensor, b: torch.Tensor,
                  mean: torch.Tensor | None = None) -> torch.Tensor:
    """[B,N,D], [B,M,D] -> [B,N,M] cosine similarities.

    `mean` (the dataset mean embedding from `measure_stats`) subtracts the
    common component first, correcting a residual offset in the PCA-128
    space; pass it when available.
    """
    if mean is not None:
        m = mean.view(1, 1, -1).to(a.dtype).to(a.device)
        a, b = a - m, b - m
    return torch.bmm(
        F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1).transpose(1, 2)
    )


def match_by_cosine(pred, target, mask, mean=None):
    """Optimal assignment of generated to true embeddings, by cosine.

    Returns `(perm [B,N], cos [B,N])`; `perm[b,i]` is the true index matched
    to generated slot i. Padded slots match themselves and their cosine is
    meaningless -- always read it through `mask`. The residual channel is
    ordered, so score it separately with `residual_cosine` instead.
    """
    sim = cosine_matrix(pred, target, mean)
    perm = assignment_from_cost(-sim, mask)
    return perm, torch.gather(sim, 2, perm.unsqueeze(-1)).squeeze(-1)


def matched_cosine(pred, target, mask, mean=None) -> torch.Tensor:
    _, cos = match_by_cosine(pred, target, mask, mean)
    return masked_mean(cos, mask)


def matched_mse(pred, target, mask, mean=None) -> torch.Tensor:
    """MSE at the SAME cosine-optimal assignment `matched_cosine` uses.

    Unlike cosine this is not scale-invariant: a model whose set is angularly
    correct but off in magnitude (`sigma_data` drift, guidance miscalibration)
    scores a good `matched_cosine` and a bad `matched_mse`. Read them together.
    """
    perm, _ = match_by_cosine(pred, target, mask, mean)
    matched_target = torch.gather(target, 1, perm.unsqueeze(-1).expand_as(target))
    return masked_mean((pred.float() - matched_target.float()).pow(2).mean(-1), mask)


def residual_cosine(pred, target, is_residual) -> torch.Tensor:
    """Cosine on the residual slot only, without matching.

    The residual is not permutation-equivariant, so it is scored
    positionally. `is_residual` is a [B,N] bool mask, not a column index,
    since each item's residual sits at its own last valid slot.
    """
    p, t = pred[is_residual], target[is_residual]
    return F.cosine_similarity(p.float(), t.float(), dim=-1).mean()


def norm_ratio(pred, target, mask) -> torch.Tensor:
    """Mean ||pred|| / ||target|| over valid slots.

    Cosine is blind to scale, and flow-matching samples drift in norm when
    `sigma_data` is mis-set. This should sit at 1.0; a systematic offset points
    at the noise schedule, not at the model's semantics.
    """
    pn = pred.float().norm(dim=-1)
    tn = target.float().norm(dim=-1).clamp(min=1e-8)
    return masked_mean(pn / tn, mask)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def baseline_cosines(target, mask, mix_emb, mean_emb=None) -> dict[str, torch.Tensor]:
    """The numbers a real model has to beat.

    `mix`  -- emit the mixture embedding in every slot: the lazy solution, and a
              surprisingly strong one. Below this, the model learned nothing
              about decomposition.
    `mean` -- emit the dataset-mean embedding in every slot: how much of any
              cosine is just the embedding space's anisotropy.
    """
    out = {"mix": matched_cosine(mix_emb.unsqueeze(1).expand_as(target), target, mask)}
    if mean_emb is not None:
        out["mean"] = matched_cosine(
            mean_emb.view(1, 1, -1).to(target.device).expand_as(target), target, mask
        )
    return out


# --------------------------------------------------------------------------
# identification against category centroids
# --------------------------------------------------------------------------
def classify_to_centroids(emb: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """[B,N,D] + [K,D] -> [B,N] long: nearest category centroid by cosine."""
    return (
        F.normalize(emb.float(), dim=-1)
        @ F.normalize(centroids.float().to(emb.device), dim=-1).t()
    ).argmax(dim=-1)


def set_identification_accuracy(pred, mask, centroids, true_labels) -> torch.Tensor:
    """Fraction of the true category multiset recovered, per item, averaged.

    Multiset intersection, so it needs no matching, ignores order, and
    cannot be gamed by emitting N copies of the mixture. Log alongside
    `identification_chance` and the ceiling (same metric on true
    embeddings); chance is not zero.
    """
    pred_labels = classify_to_centroids(pred, centroids)
    k = centroids.shape[0]
    scores = torch.zeros(pred.shape[0], device=pred.device)
    for i in range(pred.shape[0]):
        m = mask[i]
        p = torch.bincount(pred_labels[i][m], minlength=k)
        t = torch.bincount(true_labels[i][m], minlength=k)
        scores[i] = torch.minimum(p, t).sum().float() / m.sum().clamp(min=1)
    return scores.mean()


def identification_chance(mask: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Expected `set_identification_accuracy` for uniform random labels.

    Closed form 1-((K-1)/K)^n per item. Assumes uniform labels; the real
    category distribution is skewed, so treat a shuffled-set score as the
    better empirical floor when available.
    """
    n = mask.sum(dim=1).float()
    return (1.0 - ((n_classes - 1) / n_classes) ** n).mean()


def per_class_identification(pred, mask, centroids, true_labels, n_classes: int):
    """-> [K] per-category recall of `classify_to_centroids`, NaN where absent.

    The multiset score hides which categories the model can and cannot name;
    with 11 unbalanced MoisesDB categories that breakdown is most of the signal
    (expect vocals/drums/bass to work long before 'other plucked' does).
    """
    pred_labels = classify_to_centroids(pred, centroids)[mask]
    true_flat = true_labels[mask]
    out = torch.full((n_classes,), float("nan"), device=pred.device)
    for c in range(n_classes):
        sel = true_flat == c
        if sel.any():
            out[c] = (pred_labels[sel] == c).float().mean()
    return out


def confusion_matrix(pred, mask, centroids, true_labels, n_classes: int):
    """-> [K,K] counts of (true category, predicted category). Plot this."""
    pred_labels = classify_to_centroids(pred, centroids)[mask]
    true_flat = true_labels[mask]
    cm = torch.zeros(n_classes, n_classes, device=pred.device)
    idx = true_flat * n_classes + pred_labels
    cm.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=cm.dtype))
    return cm


# --------------------------------------------------------------------------
# manifold / distribution checks
# --------------------------------------------------------------------------
def nearest_real_distance(pred, bank, mask) -> torch.Tensor:
    """Cosine to the nearest real embedding in a reference bank.

    A set scoring well on matched cosine but far from every real embedding
    is off-manifold. `bank` comes from `measure_stats`' npz.
    """
    sim = F.normalize(pred.float(), dim=-1) @ F.normalize(bank.float().to(pred.device), dim=-1).t()
    return masked_mean(sim.max(dim=-1).values, mask)


def set_diversity(samples, mask) -> torch.Tensor:
    """Mean matched cosine between two independent samples of the same mixture.

    A collapsed model returns the same set every time and scores ~1.0.
    `samples` is [2, B, N, D]; read alongside matched cosine.
    """
    return matched_cosine(samples[0], samples[1], mask)


def intra_set_cosine(pred, mask) -> torch.Tensor:
    """Mean cosine between DISTINCT slots of the same generated set.

    A degenerate model emits N near-copies of one vector; that scores ~1.0 here
    while the true sets score much lower. Compare against the same quantity on
    the targets -- an easy, matching-free collapse detector.
    """
    sim = cosine_matrix(pred, pred)
    b, n, _ = sim.shape
    pair = (mask.unsqueeze(2) & mask.unsqueeze(1)) & ~torch.eye(
        n, dtype=torch.bool, device=pred.device
    ).unsqueeze(0)
    return (sim * pair).sum() / pair.sum().clamp(min=1)


def rms(emb, mask) -> torch.Tensor:
    """RMS per element over valid slots -- compare with the `sigma_data` used.

    If generated RMS drifts from the measured data RMS, the sampler's noise
    schedule is miscalibrated regardless of how good the cosines look.
    """
    return masked_mean(emb.float().pow(2).mean(-1), mask).sqrt()


def evaluate(pred, target, mask, stats: dict | None = None, mix_emb=None,
             true_labels=None, residual_index: int | None = -1) -> dict[str, float]:
    """Convenience roll-up: every applicable metric above, as plain floats.

    `stats` is the dict/npz from `measure_stats` (`mean`, `centroids`, `bank`,
    `rms`, `stem_names`). Everything optional is skipped when not supplied, so
    this is safe to call from a trainer with whatever is on hand.
    """
    def _t(x):
        return torch.as_tensor(x, device=pred.device, dtype=torch.float32)

    mean = _t(stats["mean"]) if stats is not None and "mean" in stats else None
    out = {
        "matched_cosine": matched_cosine(pred, target, mask),
        "matched_cosine_centred": matched_cosine(pred, target, mask, mean),
        "matched_mse": matched_mse(pred, target, mask),
        "norm_ratio": norm_ratio(pred, target, mask),
        "intra_set_cosine": intra_set_cosine(pred, mask),
        "intra_set_cosine_target": intra_set_cosine(target, mask),
        "rms": rms(pred, mask),
        "rms_target": rms(target, mask),
    }
    if residual_index is not None:
        out["residual_cosine"] = residual_cosine(pred, target, residual_index)
    if mix_emb is not None:
        for k, v in baseline_cosines(target, mask, mix_emb, mean).items():
            out[f"baseline_{k}"] = v
    if stats is not None and true_labels is not None and "centroids" in stats:
        cent = _t(stats["centroids"])
        out["identification"] = set_identification_accuracy(pred, mask, cent, true_labels)
        out["identification_ceiling"] = set_identification_accuracy(
            target, mask, cent, true_labels
        )
        out["identification_chance"] = identification_chance(mask, cent.shape[0])
    if stats is not None and "bank" in stats:
        out["nearest_real_cosine"] = nearest_real_distance(pred, _t(stats["bank"]), mask)
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# self-check: run this library against real measured embeddings.
#   python src/source_embedding_generator/metrics.py <stats.npz>
# --------------------------------------------------------------------------
def _self_check(stats_path: str, b: int = 64, n: int = 8, seed: int = 0):
    import numpy as np

    z = np.load(stats_path, allow_pickle=True)
    stats = {k: z[k] for k in z.files}
    rng = np.random.default_rng(seed)
    bank = torch.as_tensor(stats["bank"], dtype=torch.float32)
    cent = torch.as_tensor(stats["centroids"], dtype=torch.float32)
    names = [str(s) for s in stats["stem_names"]]

    # Build synthetic "true" sets from real embeddings drawn per category.
    bank_lab = classify_to_centroids(bank.unsqueeze(0), cent)[0]
    target, labels = [], []
    for _ in range(b):
        cats = torch.as_tensor(rng.choice(len(names), size=n, replace=True))
        rows = []
        for c in cats:
            pool = (bank_lab == c).nonzero().flatten()
            pool = pool if len(pool) else torch.arange(len(bank))
            rows.append(bank[pool[rng.integers(len(pool))]])
        target.append(torch.stack(rows))
        labels.append(cats)
    target = torch.stack(target)
    true_labels = torch.stack(labels)
    mask = torch.ones(b, n, dtype=torch.bool)
    mix_emb = target.mean(dim=1)  # stand-in for the mixture embedding

    mean_t = torch.as_tensor(stats["mean"], dtype=torch.float32)
    cases = {
        "oracle (pred = target)": target,
        "shuffled targets": target[torch.randperm(b)],
        "collapsed (mixture in every slot)": mix_emb.unsqueeze(1).expand_as(target),
        "collapsed (global mean)": mean_t.view(1, 1, -1).expand_as(target),
        "gaussian noise at data RMS": torch.randn_like(target) * float(stats["rms"]),
    }
    keys = None
    for label, pred in cases.items():
        m = evaluate(pred.contiguous(), target, mask, stats, mix_emb, true_labels)
        if keys is None:
            keys = list(m)
            print("case".ljust(36) + "".join(k[:13].rjust(15) for k in keys))
        print(label.ljust(36) + "".join(f"{m[k]:15.3f}" for k in keys))


if __name__ == "__main__":
    import sys

    _self_check(sys.argv[1] if len(sys.argv) > 1
                else str(Path(__file__).with_name("passt_pca128train_moisesdb_stats.npz")))

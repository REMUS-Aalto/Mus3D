"""Trainer for the recursive SEG, and the evaluation that makes it readable.

The training loop, checkpointing, signal handling and the loss-vs-t figure
are generic; the demo logs the recursion as a trajectory (extract, look at
what is left, extract again, stop):

    recursive/stop_cos_step{i}     cosine of step i's residual to the stop
                                   token; should rise toward 1 at completion.
    recursive/n_steps_pred/true    predicted vs. true stopping step.
    recursive/stop_error           pred - true, signed.
    recursive/cum_matched_cosine   optimal-assignment cosine of everything
                                   extracted so far against the true units.
    recursive/max_dup_cosine       cosine of the newest embedding to the
                                   closest already-extracted one.
    <prefix>_tables/<level>/songN  one table per song: matched unit,
                                   classifier name, stop_cos, cosine, and,
                                   for songs with audio, the separator's
                                   estimate and its SI-SDR. Rows marked
                                   MISSED are true units nothing matched.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import wandb
from utils.logging import MetricsLogger
from source_embedding_generator.separator_io import load_separator
from utils.torch_common import exists, print_once, set_seed, sort_dict
from source_embedding_generator import protocol
from classifiers.stem_classifier import naming_report


#: The partial assignment every set comparison in this repo uses.
from source_embedding_generator.metrics import partial_assignment as _assign


def _vocal_leaf(leaf: str) -> str:
    """Any vocals-family leaf -> the single name `vocals`.

    The dataset merges the vocals family into one unit but keeps the first
    leaf's name; the stem classifier's vocabulary only has `vocals`, so
    renaming here makes ground truth and classifier agree on naming.
    """
    from data.moisesdb import LEAF_TO_GROUP

    return "vocals" if LEAF_TO_GROUP.get(str(leaf)) == "vocals" else str(leaf)


def _matched_cosine(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Optimal-assignment cosine between two sets -> (mean, coverage-penalised).

    `mean` is over matched pairs only. `penalised` divides by
    `max(len(a), len(b))` instead, so under- or over-extraction is charged
    rather than ignored (Hungarian on a rectangular matrix silently drops
    the unmatched side).
    """
    from scipy.optimize import linear_sum_assignment

    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    c = a @ b.T
    r, col = linear_sum_assignment(-c)
    total = float(c[r, col].sum())
    return total / len(r), total / max(len(a), len(b))


def _si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    """[N, C, T] estimate/reference -> [N] scale-invariant SDR in dB.

    Scale-invariant because the separator has no reason to reproduce a slot's
    absolute gain: SEG tells it WHAT to extract, and nothing in the query
    embedding says how loud the result should be.
    """
    est, ref = est.flatten(1).float(), ref.flatten(1).float()
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps) * ref
    return 10 * torch.log10(proj.pow(2).sum(-1) / ((est - proj).pow(2).sum(-1) + eps) + eps)


class RecursiveFMTrainer:
    def __init__(
        self,
        model,
        ema,
        optimizer,
        scheduler,
        dataloader,
        accel,
        cfg,
        ckpt_dir=None,
        stats_path: str | None = None,
        val_dataset=None,
    ):
        self.model = accel.unwrap_model(model)
        self.ema = ema
        self.opt = optimizer
        self.sche = scheduler
        self.dataloader = dataloader
        self.accel = accel
        self.cfg = cfg
        self.cfg_t=cfg.trainer
        self.EPS=1e-8

        self.wandb_logger = MetricsLogger()
        self.cli_logger = MetricsLogger()
        self.test_logger = MetricsLogger()
        self._t_loss_buf: list = []
        self._k_buf: list = []

        self.states = {
            "global_step": 0,
            "best_metrics": float("inf"),
            "prev_metrics": float("inf"),
        }

        if torch.cuda.is_available():
            self.s_event = torch.cuda.Event(enable_timing=True)
            self.e_event = torch.cuda.Event(enable_timing=True)
        else:
            self.s_event = None
            self.e_event = None

        self.terminate_requested = False
        self._setup_signal_handlers()

        if ckpt_dir is not None:
            self._load_ckpt(ckpt_dir)

        self._load_stats(stats_path)
        self._separator = None
        self._val_dset = None

    def _val_dataset(self):
        """The held-out dataset, or None. Instantiated once, failures non-fatal.

        A missing or unreadable eval set costs only the `val/*` curves, not
        the training run.
        """
        if self._val_dset is not None:
            return self._val_dset or None
        val_cfg = self.cfg.data.get("val") if hasattr(self.cfg, "data") else None
        if val_cfg is None:
            self._val_dset = False
            return None
        try:
            self._val_dset = hydra.utils.instantiate(val_cfg)
            if type(self._val_dset).__name__ == "EffectChainDataset":
                raise ValueError(
                    "data.val is wrapped in EffectChainDataset; validation must "
                    "never be distorted. Point data.val at the bare dataset."
                )
            print_once(f"[SEG] val demo on {val_cfg.dset_dir} "
                       f"({len(self._val_dset)} songs)")
        except Exception as e:
            print_once(f"[warn] could not build data.val, val metrics off: {e}")
            self._val_dset = False
        return self._val_dset or None

    def _load_separator(self):
        """The trained query-conditioned separator, or None. Loaded once.

        Running the frozen separator on generated embeddings turns a cosine
        into audio and an SI-SDR against the true slot. The model owns it
        (`RecursiveFMGenerator(separator=...)`); this only bridges the
        config, loading `trainer.separator_ckpt_dir` if the model has none.
        Opt-in and non-fatal: a missing checkpoint degrades to no audio demos.
        """
        if self._separator is not None:
            return self._separator or None

        try:
            sep = self.model.separator
        except Exception as e:
            print_once(f"[warn] model separator unusable: {e}")
            sep = None
        if sep is not None:
            self._separator = sep
            return sep

        ckpt_dir = getattr(self.cfg_t, "separator_ckpt_dir", None)
        if not ckpt_dir:
            self._separator = False
            return None
        try:
            self._separator = load_separator(
                ckpt_dir, device=self.accel.device,
                use_ema=bool(getattr(self.cfg_t, "separator_use_ema", True)))
            self.model._sep[0] = self._separator
            print_once(f"[SEG] separator demo enabled from {ckpt_dir}")
        except Exception as e:
            print_once(f"[warn] could not load separator from {ckpt_dir}: {e}")
            self._separator = False
        return self._separator or None

    def _load_stats(self, stats_path):
        """Reference statistics from `measure_stats.py`, or None.

        Only the demo metrics use these (centroids, mean, baselines).
        Training does not need them; a missing file just disables them.
        """
        self.stats = None
        self.stem_names = []
        if stats_path is None:
            stats_path = getattr(self.cfg_t, "stats_path", None)
        if stats_path is None:
            print_once("[warn] No stats_path: demo identification metrics disabled. "
                       "Run measure_stats.py to enable them.")
            return
        if not os.path.exists(stats_path):
            print_once(f"[warn] stats_path does not exist: {stats_path} -- demo "
                       "identification metrics disabled. Run measure_stats.py.")
            return
        d = np.load(stats_path, allow_pickle=True)
        self.stats = {k: torch.as_tensor(d[k]) for k in ("mean", "centroids")
                      if k in d}
        self.stem_names = [str(x) for x in d["stem_names"]] if "stem_names" in d else []

        measured = float(d["rms"]) if "rms" in d else None
        sigma = getattr(self.model, "sigma_data", None)
        if measured and sigma and abs(measured - sigma) / measured > 0.25:
            print_once(f"[warn] sigma_data={sigma} is far from the measured "
                       f"embedding RMS {measured:.4f}; re-run measure_stats.py.")

    def _setup_signal_handlers(self):
        """Set up SIGTERM/SIGINT handlers for graceful termination.

        Main process only, to avoid race conditions.
        """
        save = getattr(self.cfg_t, "save_ckpt_on_exit", True)
        if not save:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            return

        if not self.accel.is_main_process:
            return

        def signal_handler(signum, frame):
            if not self.terminate_requested:
                print(
                    f"\n[Signal {signal.Signals(signum).name} received] Saving checkpoint and waiting for workers to finish"
                )
                self.terminate_requested = True

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _load_ckpt(self, dir: str):
        import json

        print_once(f"\n[Resuming training from the checkpoint directory] -> {dir}")
        ckpts = {"model": self.model, "optimizer": self.opt, "scheduler": self.sche}

        for k, v in ckpts.items():
            sd = torch.load(f"{dir}/{k}.pth", weights_only=False)
            if k == "model":
                # Strict, except for what `_strip_frozen` deliberately left
                # out of the file; anything else missing is a real mismatch.
                r = v.load_state_dict(sd, strict=False)
                unloaded = [k for k in r.missing_keys
                            if not k.startswith(self._FROZEN_PREFIXES)]
                if unloaded or r.unexpected_keys:
                    raise RuntimeError(
                        f"checkpoint does not match the model: "
                        f"{len(unloaded)} missing {unloaded[:10]}, "
                        f"{len(r.unexpected_keys)} unexpected "
                        f"{r.unexpected_keys[:10]}. An architecture change is a "
                        f"warm start (INIT=), not a resume.")
            elif k == "optimizer":
                n_new = sum(len(g["params"]) for g in v.state_dict()["param_groups"])
                n_old = sum(len(g["params"]) for g in sd["param_groups"])
                if n_new != n_old:
                    print_once(
                        f"[optimizer] parameter count changed ({n_old} -> {n_new}); "
                        f"resuming WITHOUT the saved Adam moments. Expect a brief "
                        f"loss bump while they re-estimate.")
                else:
                    v.load_state_dict(sd)
            else:
                v.load_state_dict(sd)

        ema_path = f"{dir}/ema.pth"
        if self.ema is not None and os.path.exists(ema_path):
            saved = torch.load(ema_path, weights_only=False)
            cur = self.ema.state_dict()
            merged = {k: (saved[k] if k in saved else v) for k, v in cur.items()}
            added = [k for k in cur if k not in saved]
            dropped = [k for k in saved if k not in cur]
            if added or dropped:
                print_once(
                    f"[ema] resuming across an architecture change: "
                    f"{len(added)} new key(s) seeded from init, "
                    f"{len(dropped)} stale key(s) dropped. "
                    f"new: {added[:5]}{' ...' if len(added) > 5 else ''}"
                )
            self.ema.load_state_dict(merged)

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    def _log_loss_per_t(self, n_bins: int = 20):
        """Flow-matching loss binned by t -- the main diagnostic for this model.

        A healthy rectified-flow run is roughly flat with a rise toward t=1
        (pure noise, where the velocity is hardest to predict). A spike at low t
        usually means sigma_data is mis-scaled; a flat-and-high curve everywhere
        means the conditioning is not being used.
        """
        if not self._t_loss_buf:
            return
        t_all = torch.cat([x[0] for x in self._t_loss_buf]).numpy()
        loss_all = torch.cat([x[1] for x in self._t_loss_buf]).numpy()
        self._t_loss_buf.clear()

        bins = np.linspace(0, 1, n_bins + 1)
        bidx = np.digitize(t_all, bins, right=True).clip(1, n_bins) - 1
        centers = 0.5 * (bins[:-1] + bins[1:])

        means = np.array([loss_all[bidx == i].mean() if (bidx == i).any() else np.nan
                          for i in range(n_bins)])
        stds = np.array([loss_all[bidx == i].std() if (bidx == i).any() else 0.0
                         for i in range(n_bins)])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(centers, means, width=0.8 / n_bins, yerr=stds, capsize=3)
        axes[0].set_xlabel("t"); axes[0].set_ylabel("FM loss")
        axes[0].set_title("Loss vs t")
        axes[1].plot(centers, 10 * np.log10(np.maximum(means, 1e-12)), marker="o")
        axes[1].set_xlabel("t"); axes[1].set_ylabel("FM loss (dB)")
        axes[1].set_title("Loss vs t (dB)")
        fig.tight_layout()

        tmp_path = "/tmp/seg_loss_per_t.png"
        fig.savefig(tmp_path, dpi=60)
        plt.close(fig)
        try:
            self.accel.log({"loss_per_t": wandb.Image(tmp_path)},
                           step=self.states["global_step"])
        except Exception as e:
            print(f"[warn] loss_per_t wandb log failed: {e}")

        try:
            self._log_k_hist()
        except Exception as e:
            print(f"[warn] K histogram failed: {e}")

    def _log_tsne(self, songs, background=None, prefix="demo", tag=""):
        """One t-SNE figure per song, all songs fitted together. -> {key: Image}

        Shows why a generated set is wrong, not just how far (see the paper
        for the failure modes this makes visible). All songs, the residual
        trajectories, and an optional background cloud are fit in a single
        t-SNE, then each figure draws one song against the shared layout.

        Each `songs` entry is one song:

            si          song index, lines up figure and table.
            pred        [n, D] generated sources, in extraction order.
            true        [U, D] the song's true units, `labels` their names.
            gt          {pred_i: true_j}, the assignment `_assign` made.
            residuals   [n, D] or None; `mixture` [D] or None.

        `background=(emb, labels)` is an extra cloud of true embeddings from
        songs the model was never run on, giving the t-SNE map enough points
        to form family clusters. Usual t-SNE caveat: read grouping, not
        inter-cluster distance; trajectories as order, not length.
        """
        from sklearn.manifold import TSNE

        songs = [s for s in songs
                 if len(s.get("pred", ())) and len(s.get("true", ()))]
        if not songs:
            return {}
        d = songs[0]["true"].shape[-1]

        def _arr(x):
            return (np.zeros((0, d), dtype=np.float32) if x is None
                    else np.asarray(x, dtype=np.float32).reshape(-1, d))

        bg, bg_lab = _arr(None), []
        if background is not None:
            bg_, bg_lab = background
            bg = _arr(bg_)
            bg_lab = [str(x) for x in bg_lab][: len(bg)]
            bg = bg[: len(bg_lab)]

        # One fit; each block's offset is recorded to slice the layout apart
        # again per song.
        blocks, spans = [], []
        for s_ in songs:
            span = {}
            for k in ("true", "pred", "residuals", "mixture"):
                v = _arr(s_.get(k))
                span[k] = (sum(len(x) for x in blocks), len(v))
                blocks.append(v)
            spans.append(span)
        bg_at = sum(len(x) for x in blocks)
        blocks.append(bg)
        x = np.concatenate(blocks, axis=0)
        if len(x) < 6:
            return {}
        xy = TSNE(n_components=2,
                  perplexity=max(5.0, min(30.0, (len(x) - 1) / 3)),
                  init="pca", random_state=0).fit_transform(x)
        bg2 = xy[bg_at: bg_at + len(bg)]

        fams = sorted({str(l) for s_ in songs for l in s_["labels"]}
                      | set(bg_lab))
        cmap = plt.get_cmap("tab20")
        colour = {f: cmap(i % 20) for i, f in enumerate(fams)}

        images = {}
        for s_, span in zip(songs, spans):
            def block(k, span=span):
                o, n_ = span[k]
                return xy[o: o + n_]

            true2, pred2 = block("true"), block("pred")
            res2, mix2 = block("residuals"), block("mixture")
            lab = [str(l) for l in s_["labels"]]
            gt = dict(s_["gt"])
            matched_true = set(gt.values())

            fig, ax = plt.subplots(figsize=(10.5, 6) if len(bg) else (7, 6))
            # THE MAP: held-out true embeddings, faded. The other demo songs are
            # in the same fit but are not drawn -- this figure is about one song.
            for f in fams:
                sel = np.array([l == f for l in bg_lab], dtype=bool)
                if sel.any():
                    ax.scatter(*bg2[sel].T, color=colour[f], marker=".", s=26,
                               alpha=0.5, edgecolors="none", zorder=1,
                               label=None if f in lab else f)
            # True units: filled where matched, open where missed.
            for j, f in enumerate(lab):
                if j in matched_true:
                    ax.scatter(*true2[j], color=colour[f], marker="o", s=52,
                               edgecolors="none", zorder=3, label=f)
                else:
                    ax.scatter(*true2[j], facecolors="none",
                               edgecolors=colour[f], marker="o", s=78,
                               linewidths=1.7, zorder=3, label=f"{f} (MISSED)")
            # Generated sources: colour of their assigned unit, black if unmatched.
            for i in range(len(pred2)):
                c = colour[lab[gt[i]]] if i in gt else "0.1"
                ax.scatter(*pred2[i], color=c, marker="x", s=52, linewidths=1.7,
                           zorder=4)
                ax.annotate(str(i), pred2[i], textcoords="offset points",
                            xytext=(4, 4), fontsize=6.5, color="0.25")
                if i in gt:
                    ax.plot([true2[gt[i], 0], pred2[i, 0]],
                            [true2[gt[i], 1], pred2[i, 1]],
                            color="0.7", lw=0.6, zorder=0)
            # Both trajectories dotted, told apart by weight and colour.
            if len(pred2) > 1:
                ax.plot(*pred2.T, ls=":", lw=1.0, color="0.5", zorder=2,
                        label="source trajectory")
            walk = np.concatenate([mix2, res2], axis=0)
            if len(walk) > 1:
                ax.plot(*walk.T, ls=":", lw=1.6, color="k", zorder=2,
                        label="residual trajectory")
            if len(mix2):
                ax.scatter(*mix2[0], color="k", marker="*", s=220, zorder=5,
                           label="mixture")
            if len(res2):
                ax.scatter(*res2.T, color="k", marker=r"$\ast$", s=110,
                           zorder=5, label="residual (per step)")
                for i, p in enumerate(res2):
                    ax.annotate(str(i), p, textcoords="offset points",
                                xytext=(4, -9), fontsize=6.5, color="0.25")

            ax.set_title(
                f"t-SNE of SEG embeddings{tag} -- song {s_['si']}\n"
                "o true   x generated (numbered by step)   "
                "* mixture   asterisk residual   dotted = trajectory")
            # One legend entry per name, in the order first drawn.
            h, l_ = ax.get_legend_handles_labels()
            seen, hh, ll = set(), [], []
            for h_, name in zip(h, l_):
                if name and name not in seen:
                    seen.add(name)
                    hh.append(h_)
                    ll.append(name)
            if len(bg):
                ax.legend(hh, ll, fontsize=6.5, loc="center left",
                          bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
            else:
                ax.legend(hh, ll, fontsize=7, loc="best", ncol=2)
            fig.tight_layout()

            tmp = (f"/tmp/seg_tsne_{prefix.replace('/', '_')}"
                   f"_song{s_['si']}.png")
            fig.savefig(tmp, dpi=80)
            plt.close(fig)
            images[f"{prefix}/tsne/song{s_['si']}"] = wandb.Image(tmp)
        return images

    CL_COLS = ["status", "step", "matched GT", "named", "stop_cos",
               "cos_to_matched_true", "SI-SDR", "mixture", "stems sum",
               "ground truth", "separated", "complement"]

    @torch.no_grad()
    def _audio_rows(self, sep_songs, prefix: str, log: dict):
        """The audio the generator produced -> table cells + SI-SDR.

        Consumes `est`/`comp`/`leftover` from `inference(audio=True)` and
        scores them against the true unit each extraction was assigned to.

            separated    what this query pulled out.
            complement   everything else for it.

        `leftover` present means the decomposition partitions the mixture,
        so `mix_recon_si_sdr` is a real conservation check; absent, only the
        stems are scored. Unmatched extractions keep their row and audio but
        are excluded from the SI-SDR mean.
        """
        sr = int(self.model.loudness.sample_rate)
        rows, si_all, mix_recon, n_unmatched = {}, [], [], 0

        for s_ in sep_songs:
            est, comp = s_.get("est"), s_.get("comp")
            if est is None:
                continue
            k = int(s_["n"])
            est, comp = est[:k], comp[:k]
            ref = s_["ref"][:k]
            if est.shape[-2] != ref.shape[-2]:
                ref = ref.mean(dim=-2, keepdim=True)
            n_t = min(est.shape[-1], ref.shape[-1])
            est, ref, comp = est[..., :n_t], ref[..., :n_t], comp[..., :n_t]
            si = _si_sdr(est, ref)
            ok = s_["matched"][:k]
            mix = s_["mix"]

            mix_a = wandb.Audio(mix[0].mean(0).float().cpu().numpy(), sample_rate=sr)
            sum_a = wandb.Audio(est.sum(0).mean(0).float().cpu().numpy(),
                                sample_rate=sr)
            rows[s_["si"]] = [{
                "slot": i,
                "mixture": mix_a if i == 0 else None,
                "stems sum": sum_a if i == 0 else None,
                "ground truth": (wandb.Audio(
                    ref[i].mean(0).float().cpu().numpy(), sample_rate=sr)
                    if bool(ok[i]) else None),
                "separated": wandb.Audio(
                    est[i].mean(0).float().cpu().numpy(), sample_rate=sr),
                "complement": wandb.Audio(
                    comp[i].mean(0).float().cpu().numpy(), sample_rate=sr),
                "SI-SDR": round(float(si[i]), 2) if bool(ok[i]) else None,
            } for i in range(k)]

            n_unmatched += int((~ok).sum())
            si_all += si[ok].tolist()
            mref = mix[0].mean(0, keepdim=True) if est.shape[-2] == 1 else mix[0]
            left = s_.get("leftover")
            total = est.sum(0) if left is None else est.sum(0) + left[..., :n_t]
            mix_recon.append(float(_si_sdr(
                total[None], mref[..., :n_t][None])[0]))

        if si_all:
            log[f"{prefix}/sep/si_sdr"] = float(np.mean(si_all))
        if mix_recon:
            log[f"{prefix}/sep/mix_recon_si_sdr"] = float(np.mean(mix_recon))
        log[f"{prefix}/sep/n_unmatched"] = float(n_unmatched)
        if si_all:
            print(f"\t->->-> {prefix}/sep: SI-SDR "
                  f"{np.mean(si_all):.2f} dB over {len(si_all)} stems "
                  f"({n_unmatched} unmatched excluded)")
        return rows

    def start_training(self):
        self.model.train()
        if self.s_event is not None:
            self.s_event.record()

        print_once("\n[ Started training ]\n")

        if self.accel.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(
                f"=== Model params: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable ==="
            )
            self.accel.log(
                {"params/total": total_params, "params/trainable": trainable_params},
                step=0,
            )

        dataloader_iter = iter(self.dataloader)
        while not self.terminate_requested:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                continue

            metrics = self.run_step(batch)

            if self.accel.is_main_process:
                self.wandb_logger.add(metrics)
                self.cli_logger.add(metrics)

                if self._its_time(self.cfg_t.logging.n_step_log):
                    self._log_metrics()

                if self._its_time(self.cfg_t.logging.n_step_print):
                    self._print_metrics()

                if self._its_time(self.cfg_t.logging.n_step_ckpt):
                    self._save_ckpt()

                if self._its_time(self.cfg_t.logging.n_step_sample):
                    try:
                        self._sample_demo()
                    except Exception as e:
                        print(f"[warn] _sample_demo failed: {e}")
                    val = self._val_dataset()
                    if val is not None:
                        try:
                            self._sample_demo(dset=val, prefix="val")
                        except Exception as e:
                            print(f"[warn] _sample_demo(val) failed: {e}")

                if self._its_time(self.cfg_t.logging.n_step_figure):
                    self._log_loss_per_t()

            self.states["global_step"] += 1

        del dataloader_iter

        if self.accel.is_main_process and self.terminate_requested:
            print("\n[Termination requested]")
            if getattr(self.cfg_t, "save_ckpt_on_exit", True):
                print("Saving final checkpoint...")
                self._save_ckpt()
            wandb.finish()
        sys.exit(0)

    def _train_kwargs(self) -> dict:
        """Extras passed to `train_step` on every step. Empty here."""
        return {}

    def _demo_step_count(self) -> int:
        """Sampler steps for one demo pass.

        `max(steps)`: flow matching wants the finest discretisation it is
        configured for.
        """
        return max(self.cfg_t.logging.steps)

    def run_step(self, batch, train: bool = True):
        """`batch` is (slots [M,C,T], counts [B], ...) -- see `collate_recursive`.

        The protocol (which units are already available, which one is the
        target, level-1 merging, residual aggregation) all happens in the
        dataset worker, so there is nothing to prepare here.
        """
        slots, counts = batch[0], batch[1]
        batch = (slots.to(self.accel.device), counts.to(self.accel.device),
                 *batch[2:])

        if train:
            self.opt.zero_grad()

        output = self.model.train_step(batch, debug=self.cfg_t.debug,
                                       **self._train_kwargs())

        if train:
            self.accel.backward(output["loss"])
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg_t.max_grad_norm,
                )
            self.opt.step()
            self.sche.step()

            if exists(self.ema):
                self.ema.update()

        stats = getattr(self.model, "last_stats", None)
        if stats is not None:
            self._t_loss_buf.append(
                (stats["t"].float().cpu(), stats["loss"].float().cpu())
            )
        if stats is not None and "step_idx" in stats:
            lvl, nu = stats["level"], stats.get("n_units")
            self._k_buf.append((
                stats["step_idx"].float().cpu(),
                (stats["step_idx"] + stats["has_residual"]).float().cpu()
                if nu is None else nu.float().cpu(),
                (torch.full_like(stats["step_idx"], -1) if lvl is None else lvl).cpu(),
            ))

        return {k: v.detach() for k, v in output.items()}

    def _its_time(self, itv: int):
        return (self.states["global_step"] - 1) % itv == 0

    def _best_is_eval_metric(self) -> bool:
        keys = self.cfg_t.logging.metrics_for_best_ckpt
        return any("/" in k for k in keys)

    def _best_mode(self) -> str:
        return str(getattr(self.cfg_t.logging, "best_metric_mode", "min"))

    def _is_better(self, new: float, old: float) -> bool:
        return new > old if self._best_mode() == "max" else new < old

    def _record_eval_metrics(self, log: dict):
        """Stash the scalars a demo just computed, then re-evaluate `best`.

        Called by the demos. Only plain numbers are kept -- wandb Tables and
        Images are not comparable and would break the float() below.
        """
        if not hasattr(self, "eval_metrics"):
            self.eval_metrics = {}
        for k, v in log.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.eval_metrics[k] = float(v)
        if self._best_is_eval_metric():
            self._update_best(self.eval_metrics, save_on_improve=True)

    def _update_best(self, source: dict, save_on_improve: bool = False):
        """Recompute latest/best from `source`; optionally checkpoint on a new best.

        `save_on_improve` is needed because eval metrics update on a
        different cadence than checkpointing, so a new best score must be
        saved immediately or the weights that produced it are gone.
        """
        keys = self.cfg_t.logging.metrics_for_best_ckpt
        missing = [k for k in keys if k not in source]
        if missing:
            if self._best_is_eval_metric():
                print_once(f"[best-ckpt] {missing} not available yet; "
                           f"`best` not updated. It appears after the first "
                           f"demo -- if it never does, check the metric name "
                           f"and that `data.val` exists.")
            return

        if self.states.get("best_metrics") in (None, float("inf")) \
                and self._best_mode() == "max":
            self.states["best_metrics"] = float("-inf")

        m_latest = float(sum(float(source[k]) for k in keys))
        self.states["latest_metrics"] = m_latest
        if self._is_better(m_latest, self.states["best_metrics"]):
            self.states["best_metrics"] = m_latest
            print(f"\t->->-> new best {'+'.join(keys)} = {m_latest:.4f}")
            if save_on_improve:
                self._save_ckpt()

    def _log_metrics(self, sort_by_key: bool = True):
        metrics = self.wandb_logger.pop()

        metrics["lr"] = self.sche.get_last_lr()[0]
        if sort_by_key:
            metrics = sort_dict(metrics)

        self.accel.log(metrics, step=self.states["global_step"])

        if not self._best_is_eval_metric():
            self._update_best({k: float(v) for k, v in metrics.items()})

    def _print_metrics(self, sort_by_key: bool = True):
        if self.e_event is not None and self.s_event is not None:
            self.e_event.record()
            torch.cuda.synchronize()
            p_time = self.s_event.elapsed_time(self.e_event) / 1000.0
        else:
            p_time = 0.0

        metrics = self.cli_logger.pop()
        metrics = {k: v.item() for k, v in metrics.items()}
        if sort_by_key:
            metrics = sort_dict(metrics)

        step = self.states["global_step"]
        s = f"Step {step} ({p_time:.1e} [sec]): " + " /".join(
            [f"[{k}] - {v:.3e}" for k, v in metrics.items()]
        )
        print(s)

        if self.s_event is not None:
            self.s_event.record()

    _FROZEN_PREFIXES = ("query_model.", "mixture_encoder.", "loudness.")

    def _strip_frozen(self, state_dict):
        return {
            k: v for k, v in state_dict.items()
            if not k.startswith(self._FROZEN_PREFIXES)
        }

    def _save_ckpt(self):
        out_dir = self.cfg_t.output_dir + "/ckpt"

        latest_dir = out_dir + "/latest"
        os.makedirs(latest_dir, exist_ok=True)
        ckpts = {
            "model": self.model,
            "optimizer": self.opt,
            "scheduler": self.sche,
        }
        for name, m in ckpts.items():
            sd = m.state_dict()
            if name == "model":
                sd = self._strip_frozen(sd)
            torch.save(sd, f"{latest_dir}/{name}.pth")

        if self.ema is not None:
            torch.save(self.ema.state_dict(), f"{latest_dir}/ema.pth")

        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        if "latest_metrics" in self.states \
                and self.states["latest_metrics"] == self.states["best_metrics"]:
            shutil.copytree(latest_dir, out_dir + "/best", dirs_exist_ok=True)

        print("\t->->-> Saved checkpoints.")

    def _log_k_hist(self):
        """Distribution of the extraction step and the decomposition size K.

        `step` is which extraction the item is in its trajectory. `N` is
        how many units the item's segment decomposes into, carried from the
        dataset rather than derived, so the joint plot shows the real
        spread. Split by level, since track vs. stem counts differ widely.
        """
        if not self._k_buf:
            return
        step = torch.cat([x[0] for x in self._k_buf]).numpy()
        k = torch.cat([x[1] for x in self._k_buf]).numpy()
        lvl = torch.cat([x[2] for x in self._k_buf]).numpy()
        self._k_buf.clear()

        rows = [(protocol.LEVEL_TRACK, "track"), (protocol.LEVEL_STEM, "stem")]
        if (lvl < 0).all():
            rows = [(-1, "all")]

        fig, axes = plt.subplots(len(rows), 3, figsize=(16, 3.4 * len(rows)),
                                 squeeze=False)
        logs = {}
        hi = int(max(k.max(), 1))
        edges = np.arange(0.5, hi + 1.5)
        for r, (lv, lname) in enumerate(rows):
            m = lvl == lv
            for c, (v, qname) in enumerate(((step, "step"), (k, "N"))):
                ax = axes[r][c]
                if m.any():
                    ax.hist(v[m], bins=edges, edgecolor="black")
                    logs[f"recursion/{lname}/{qname}_mean"] = float(v[m].mean())
                    logs[f"recursion/{lname}/{qname}_max"] = float(v[m].max())
                ax.set_xlabel(qname)
                ax.set_ylabel("items")
                ax.set_title(f"{lname}: {qname}  (n={int(m.sum())})")
                ax.set_xticks(np.arange(1, hi + 1))

            ax = axes[r][2]
            if m.any():
                h, _, _ = np.histogram2d(step[m], k[m], bins=[edges, edges])
                ax.imshow(h.T, origin="lower", aspect="auto", cmap="magma",
                          extent=(0.5, hi + 0.5, 0.5, hi + 0.5))
                ax.plot([0.5, hi + 0.5], [0.5, hi + 0.5], "w--", lw=0.8)
                on_diag = float((step[m] == k[m]).mean())
                logs[f"recursion/{lname}/frac_final_step"] = on_diag
                ax.set_title(f"{lname}: step vs N  (final {on_diag:.0%})")
            ax.set_xlabel("step")
            ax.set_ylabel("N")
        fig.tight_layout()

        tmp_path = "/tmp/seg_recursion_depth.png"
        fig.savefig(tmp_path, dpi=60)
        plt.close(fig)
        logs["recursion/depth_hist"] = wandb.Image(tmp_path)
        try:
            self.accel.log(logs, step=self.states["global_step"])
        except Exception as e:
            print(f"[warn] recursion depth wandb log failed: {e}")

    def _stem_classifier(self):
        """The fitted naming probe, or None. Loaded once, never re-read.

        Absence is not an error: the probe is a separate artifact
        (`stem_classifier.pkl`), and its absence just disables naming
        metrics rather than being retried every demo.
        """
        if not hasattr(self, "_stem_clf"):
            from classifiers.stem_classifier import StemClassifier

            try:
                clf_path = os.environ.get("SEG_STEM_CLF")
                self._stem_clf = (StemClassifier.load(clf_path) if clf_path
                                  else StemClassifier.load())
                print(f"[demo] stem classifier loaded: {self._stem_clf.meta.get('method')}")
            except Exception as e:
                print(f"[demo] no stem classifier ({e}); naming metrics skipped")
                self._stem_clf = None
        return self._stem_clf

    @staticmethod
    @contextmanager
    def _frozen_rng(seed: int):
        """Run a block off a fixed RNG stream, then restore the old one.

        Makes the demo reproducible without disturbing training's own RNG
        stream when this runs mid-training.
        """
        st, cst = torch.get_rng_state(), None
        if torch.cuda.is_available():
            cst = torch.cuda.get_rng_state_all()
        pst, nst = random.getstate(), np.random.get_state()
        try:
            set_seed(int(seed))
            yield
        finally:
            torch.set_rng_state(st)
            if cst is not None:
                torch.cuda.set_rng_state_all(cst)
            random.setstate(pst)
            np.random.set_state(nst)

    @torch.no_grad()
    def _sample_demo(self, dset=None, prefix: str = "demo"):
        """The recursion, decomposed song by song, scored and logged.

        One call, `model.inference(...)`, and everything after it is scoring:
        the assignment, the trajectory, the naming, the tables and the t-SNE.
        """
        self.model.eval()
        self._load_separator()
        dset = self.dataloader.dataset if dset is None else dset
        base = getattr(dset, "dataset", dset)
        device = self.accel.device
        n_sample = int(getattr(self.cfg_t.logging, 'n_recursive_songs',
                               self.cfg_t.logging.n_samples_per_step))
        max_steps = int(getattr(self.cfg_t.logging, "recursive_max_steps", 12))
        inf_kw = getattr(self.cfg_t.logging, "inference", None)
        inf_kw = {} if inf_kw is None else dict(inf_kw)
        if inf_kw:
            print_once(f"[demo] inference knobs: {inf_kw}")
        # THE SEPARATOR'S OWN KNOBS, which are not `extract` arguments: they
        # belong to the separation, so they live as attributes on the generator
        # and every call site that turns an embedding into audio -- the peel
        # inside the recursion, the parallel pass after it -- reads the same
        # two. Set here rather than forwarded, for exactly that reason.
        sep_kw = getattr(self.cfg_t.logging, "separator", None)
        sep_kw = {} if sep_kw is None else dict(sep_kw)
        allowed = {"sep_steps", "sep_churn"}
        if set(sep_kw) - allowed:
            raise ValueError(
                f"trainer.logging.separator accepts {sorted(allowed)}, got "
                f"{sorted(set(sep_kw) - allowed)}")
        for k, v in sep_kw.items():
            setattr(self.model, k, type(getattr(self.model, k))(v))
        if sep_kw:
            print_once(f"[demo] separator knobs: {sep_kw}")

        # THE ONE SEED THE WHOLE DEMO IS DRAWN FROM: the song crops, the
        # sampler noise, the separator. Fixed by default so two runs of the
        # same checkpoint are comparable song for song.
        demo_seed = int(getattr(self.cfg_t.logging, "demo_seed", 0))
        rng = np.random.default_rng(demo_seed)
        songs = []
        for idx in range(min(n_sample, len(base))):
            loaded = base._load_song_segment(base.song_ids[idx], rng)
            if loaded is None:
                continue
            tracks, groups = loaded
            songs.append((tracks, groups, list(getattr(base, "last_leaves", groups))))
        if not songs:
            print(f"[warn] _sample_demo({prefix}): no usable songs.")
            self.model.train()
            return

        clf = self._stem_classifier()

        n_bg = int(getattr(self.cfg_t.logging, "n_tsne_bg_songs", 0))
        bg_base, start = base, len(songs)
        bg_cfg = self.cfg.data.get("tsne_bg") if hasattr(self.cfg, "data") else None
        if n_bg and bg_cfg is not None:
            try:
                d_ = hydra.utils.instantiate(bg_cfg)
                bg_base, start = getattr(d_, "dataset", d_), 0
            except Exception as e:
                print(f"[warn] data.tsne_bg unusable, falling back to the "
                      f"eval split: {e}")
        bg_songs = []
        for idx in range(start, min(start + n_bg, len(bg_base))):
            loaded = bg_base._load_song_segment(bg_base.song_ids[idx], rng)
            if loaded is None:
                continue
            tracks, groups = loaded
            bg_songs.append((tracks, groups,
                             list(getattr(bg_base, "last_leaves", groups))))
        if n_bg:
            print(f"[demo] t-SNE background: {len(bg_songs)} extra songs")

        modes = [(protocol.LEVEL_TRACK, "track"),
                 (protocol.LEVEL_STEM, "stem")]
        for level, lname in modes:
            stop_err, cum_cos, dup_cos, n_pred, n_true = [], [], [], [], []
            traj_stop = []
            name_agree, name_stats = [], []
            pen_cos = []
            traj_rows, missed_rows, cl_rows = {}, {}, {}
            cl_stats = {"n_pred": [], "n_true": [], "cos": [], "si": [],
                        "recon": []}

            # One entry per song for the t-SNE figures.
            tsne_songs = []
            n_sep = int(getattr(self.cfg_t.logging, "n_separator_demos", 2))
            match_floor = float(getattr(self.cfg_t.logging,
                                        "match_min_cosine", 0.5))
            sep_songs = []
            for si, (tracks, groups, leaves) in enumerate(songs):
                if level == protocol.LEVEL_STEM:
                    by = {}
                    for i, g in enumerate(groups):
                        by.setdefault(g, []).append(i)
                    units, ulab = list(by.values()), list(by.keys())
                else:
                    units = [[i] for i in range(len(tracks))]
                    ulab = [_vocal_leaf(leaves[i] if i < len(leaves)
                                        else groups[i])
                            for i in range(len(tracks))]

                audio = torch.stack([tracks[u].sum(0) for u in units]).to(device)
                true_emb = self.model.query_adapter(
                    self.model._chunked(self.model.query_model, audio)).cpu().numpy()
                mix = audio.sum(0, keepdim=True)

                # Reseeded per song, so one song's step count does not
                # perturb another's noise draw.
                with self._frozen_rng(demo_seed * 100003 + level * 1009 + si):
                    out = self.model.inference(
                        mix, level=level, max_steps=max_steps,
                        steps=self._demo_step_count(),
                        audio=si < n_sep, **inf_kw,
                    )
                emb, scos, n = out["emb"], out["stop_cos"], out["n"]

                n_pred.append(n)
                n_true.append(len(units))
                stop_err.append(n - len(units))
                traj_stop.append(scos)
                mc, mcp = _matched_cosine(emb, true_emb)
                cum_cos.append(mc)
                pen_cos.append(mcp)

                e = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-9)
                dup_cos.append([float((e[:i] @ e[i]).max()) if i else 0.0
                                for i in range(n)])

                gt_idx, cos_mat = _assign(emb, true_emb, floor=match_floor)
                gt = {i: str(ulab[j]) for i, j in gt_idx.items()}

                if si < n_sep and n:
                    ref = torch.stack([
                        audio[gt_idx[i]] if i in gt_idx
                        else torch.zeros_like(audio[0]) for i in range(n)
                    ])
                    sep_songs.append(dict(
                        si=si, mix=mix, ref=ref, n=n,
                        est=out["est"], comp=out["comp"],
                        leftover=out["leftover"],
                        matched=torch.tensor([i in gt_idx for i in range(n)],
                                             device=device),
                        perm=torch.arange(n, device=device),
                        labels=[gt.get(i, "(unmatched)") for i in range(n)],
                    ))

                clf_level = "track" if level == protocol.LEVEL_TRACK else "stem"
                named = (clf.predict(emb, clf_level) if clf is not None
                         else ["?"] * n)
                if clf is not None:
                    name_agree += [float(named[i] == gt[i])
                                   for i in range(n) if i in gt]
                    name_stats.append(
                        naming_report(named, [str(u) for u in ulab]))

                tsne_songs.append(dict(
                    si=si, pred=emb, true=true_emb,
                    labels=[str(x) for x in ulab], gt=gt_idx,
                    residuals=out["residuals"],
                    mixture=(out["mixture_emb"] if out["mixture_emb"] is not None
                             else self.model.query_adapter(
                                 self.model._chunked(self.model.query_model, mix)
                             )[0].float().cpu().numpy()),
                ))

                if si < max(3, n_sep):
                    traj_rows[si] = [{
                        "status": "extracted" if i in gt_idx else "UNMATCHED",
                        "step": i,
                        "matched GT": gt.get(i, "-"),
                        "named": named[i],
                        "stop_cos": float(scos[i]),
                        "cos_to_matched_true": (float(cos_mat[i, gt_idx[i]])
                                                if i in gt_idx else None),
                    } for i in range(n)]

                    taken = set(gt_idx.values())
                    for j in range(len(units)):
                        if j in taken:
                            continue
                        row = {"status": "MISSED", "step": None,
                               "matched GT": str(ulab[j]), "named": "-"}
                        if si < n_sep:
                            sr_ = int(self.model.loudness.sample_rate)
                            row["ground truth"] = wandb.Audio(
                                audio[j].mean(0).float().cpu().numpy(),
                                sample_rate=sr_)
                        missed_rows.setdefault(si, []).append(row)

            log = {
                f"{prefix}/{lname}/matched_cosine": float(np.nanmean(cum_cos)),
                f"{prefix}/{lname}/matched_cosine_penalised": float(np.nanmean(pen_cos)),
                f"{prefix}/{lname}/n_steps_pred": float(np.mean(n_pred)),
                f"{prefix}/{lname}/n_steps_true": float(np.mean(n_true)),
                f"{prefix}/{lname}/stop_error": float(np.mean(stop_err)),
                f"{prefix}/{lname}/stop_abs_error": float(np.mean(np.abs(stop_err))),
            }
            if name_stats:
                for k in ("recall", "split", "spurious"):
                    v = [d[k] for d in name_stats if d[k] == d[k]]
                    if v:
                        log[f"{prefix}/{lname}/name_{k}"] = float(np.mean(v))
            if name_agree:
                log[f"{prefix}/{lname}/name_agree"] = float(np.mean(name_agree))
            for i in range(max_steps):
                v = [t[i] for t in traj_stop if len(t) > i]
                if v:
                    log[f"{prefix}/{lname}/stop_cos_step{i}"] = float(np.mean(v))
                v = [t[i] for t in dup_cos if len(t) > i and i > 0]
                if v:
                    log[f"{prefix}/{lname}/max_dup_cosine_step{i}"] = float(np.mean(v))
            flat = [x for t in dup_cos for x in t[1:]]
            if flat:
                log[f"{prefix}/{lname}/max_dup_cosine"] = float(np.mean(flat))
                log[f"{prefix}/{lname}/dup_rate_over_0.9"] = float(np.mean(np.array(flat) > 0.9))

            sep_rows = {}
            try:
                if sep_songs:
                    sep_rows = self._audio_rows(
                        sep_songs, prefix=f"{prefix}/{lname}", log=log)
            except Exception as e:
                import traceback
                print(f"[warn] separator audio log failed: {e}")
                traceback.print_exc()

            cols = ["status", "step", "matched GT", "named",
                    "stop_cos", "cos_to_matched_true", "SI-SDR",
                    "mixture", "stems sum", "ground truth",
                    "separated", "complement"]
            for si_ in sorted(set(traj_rows) | set(missed_rows)):
                table = wandb.Table(columns=cols)
                got = sep_rows.get(si_, [])
                for r in traj_rows.get(si_, []):
                    step = r["step"]
                    cell = got[step] if step < len(got) else {}
                    table.add_data(*[r.get(c, cell.get(c)) for c in cols])
                for r in missed_rows.get(si_, []):
                    table.add_data(*[r.get(c) for c in cols])
                log[f"{prefix}_tables/{lname}/song{si_}"] = table

            for si_, rows in sorted(cl_rows.items()):
                t_ = wandb.Table(columns=self.CL_COLS)
                for r in rows:
                    t_.add_data(*[r.get(c) for c in self.CL_COLS])
                log[f"{prefix}_closedloop/{lname}/song{si_}"] = t_
            if cl_stats["n_pred"]:
                cl = f"{prefix}/{lname}/closedloop"
                log[f"{cl}/n_steps_pred"] = float(np.mean(cl_stats["n_pred"]))
                log[f"{cl}/stop_error"] = float(np.mean(
                    np.array(cl_stats["n_pred"]) - np.array(cl_stats["n_true"])))
                if cl_stats["cos"]:
                    log[f"{cl}/matched_cosine"] = float(np.mean(cl_stats["cos"]))
                if cl_stats["si"]:
                    log[f"{cl}/si_sdr"] = float(np.mean(cl_stats["si"]))
                if cl_stats["recon"]:
                    log[f"{cl}/mix_recon_si_sdr"] = float(np.mean(cl_stats["recon"]))
                print(f"\t->->-> {cl}: steps {np.mean(cl_stats['n_pred']):.1f} "
                      f"vs true {np.mean(cl_stats['n_true']):.1f}, SI-SDR "
                      + (f"{np.mean(cl_stats['si']):.2f} dB"
                         if cl_stats["si"] else "n/a"))

            bg_emb, bg_lab = [], []
            try:
                for tracks, groups, leaves in bg_songs:
                    if level == protocol.LEVEL_STEM:
                        by = {}
                        for i, g in enumerate(groups):
                            by.setdefault(g, []).append(i)
                        u_, l_ = list(by.values()), list(by.keys())
                    else:
                        u_ = [[i] for i in range(len(tracks))]
                        l_ = [_vocal_leaf(leaves[i] if i < len(leaves)
                                          else groups[i])
                              for i in range(len(tracks))]
                    a_ = torch.stack([tracks[u].sum(0) for u in u_]).to(device)
                    bg_emb.append(self.model.query_adapter(
                        self.model._chunked(self.model.query_model, a_)).cpu().numpy())
                    bg_lab += [str(x) for x in l_]
            except Exception as e:
                print(f"[warn] t-SNE background embedding failed: {e}")
                bg_emb, bg_lab = [], []

            try:
                bg = ((np.concatenate(bg_emb, 0), bg_lab) if bg_emb else None)
                log.update(self._log_tsne(
                    tsne_songs, background=bg, prefix=f"{prefix}/{lname}",
                    tag=f" (recursive, {lname})"))
            except Exception as e:
                import traceback
                print(f"[warn] recursive t-SNE failed: {e}")
                traceback.print_exc()

            try:
                self._record_eval_metrics(log)
            except Exception as e:
                print(f"[warn] recording eval metrics failed: {e}")

            try:
                self.accel.log(log, step=self.states["global_step"])
            except Exception as e:
                print(f"[warn] recursive demo log failed: {e}")
            print(f"\t->->-> {prefix}/{lname}: steps {np.mean(n_pred):.1f} vs "
                  f"true {np.mean(n_true):.1f}, |err| {np.mean(np.abs(stop_err)):.2f}")

        self.model.train()

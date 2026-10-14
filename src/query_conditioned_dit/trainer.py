"""Trainer for the latent flow-matching separator.

Thin subclass of the query-conditioned Locoformer's `Trainer`, which already
provides the loop, checkpointing and signal handling. Only the demo differs.

The demo logs, per stem, gt / vae(gt) / separated audio and their SI-SDR.
`vae(gt)` (ground truth through encode/decode, no separation) is the codec
ceiling: the gap between gt and vae(gt) is the codec's, not the separator's.
"""

import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from data.moisesdb import LEAF_TO_GROUP

from query_conditioned_bslocoformer.trainer import Trainer as _QueryTrainer


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB for 1-D (or [1, T]) mono tensors."""
    est, ref = est.flatten().float(), ref.flatten().float()
    n = min(est.numel(), ref.numel())
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    proj = (est @ ref) / (ref @ ref + eps) * ref
    noise = est - proj
    return float(10 * torch.log10((proj.pow(2).sum() + eps) / (noise.pow(2).sum() + eps)))


def _family_of(stem_name: str) -> str:
    """Demo stem label -> MoisesDB level-1 family, or 'other'."""
    return LEAF_TO_GROUP.get(stem_name.replace(" ", "_"), "other")


def _si_sdr_chart(per_family: dict[str, list[float]], step: int) -> str:
    """Bar chart of per-family SI-SDR at one step. Returns the PNG path."""
    fams = sorted(per_family)
    means = [float(np.mean(per_family[f])) for f in fams]
    counts = [len(per_family[f]) for f in fams]

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(fams)), 4))
    bars = ax.bar(range(len(fams)), means, color="tab:blue")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f"{f}\n(n={c})" for f, c in zip(fams, counts)],
                       fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title(f"Per-family SI-SDR, separated vs ground truth (step {step})")
    ax.axhline(0.0, color="0.4", lw=0.8)
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.1f}", (b.get_x() + b.get_width() / 2, m),
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=8)
    fig.tight_layout()
    path = "/tmp/lfm_si_sdr_per_family.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


class LatentFlowTrainer(_QueryTrainer):

    # PaSST and the DAC-VAE are frozen, rebuilt from published weights in
    # __init__, so they are excluded from checkpoints; only `backbone.` is
    # trained. The base Trainer's _save_ckpt saves the whole model, hence
    # this override.
    _FROZEN_PREFIXES = ("query_model.", "query_adapter.", "codec.")

    def _extra_sample_kwargs(self, step: int) -> dict:
        """Extra kwargs for `model.sample()` in the demo."""
        return {}

    def _extra_train_kwargs(self, batch) -> dict:
        """Forward the dataset's hierarchy level to `train_step`, if present.

        Datasets that yield `(stems, level)` get it passed through; plain
        stem-only loaders leave it absent (SamAudioDiT.forward then defaults
        to all-LEVEL_TRACK).
        """
        if isinstance(batch, (tuple, list)) and len(batch) > 1:
            return {"level": batch[1].to(self.accel.device)}
        return {}

    @classmethod
    def _strip_frozen(cls, state_dict):
        out = {}
        for k, v in state_dict.items():
            # Tolerate a DDP "module." prefix when comparing.
            probe = k[len("module."):] if k.startswith("module.") else k
            if not probe.startswith(cls._FROZEN_PREFIXES):
                out[k] = v
        return out

    def _save_ckpt(self):
        out_dir = self.cfg_t.output_dir + "/ckpt"
        latest_dir = out_dir + "/latest"
        os.makedirs(latest_dir, exist_ok=True)

        for name, m in (("model", self.model), ("optimizer", self.opt),
                        ("scheduler", self.sche)):
            sd = m.state_dict()
            if name == "model":
                sd = self._strip_frozen(sd)
            self._atomic_save(sd, f"{latest_dir}/{name}.pth")

        # EMA is over the backbone only (see train_dit_separator.py); the
        # base trainer does not save it, so this does it separately.
        if getattr(self, "ema", None) is not None:
            self._atomic_save(self.ema.state_dict(), f"{latest_dir}/ema.pth")

        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        if self.states["latest_metrics"] == self.states["best_metrics"]:
            shutil.copytree(latest_dir, out_dir + "/best", dirs_exist_ok=True)

        print("\t->->-> Saved checkpoints.")

    @staticmethod
    def _atomic_save(obj, path: str):
        """`torch.save` via a temp file and `os.replace`.

        Avoids leaving a truncated checkpoint if the save is interrupted:
        `os.replace` is atomic, so a killed save leaves a junk `.tmp` and the
        previous checkpoint intact.
        """
        tmp = f"{path}.tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _migrate_opt_state(self, sd: dict) -> dict:
        """Adam state saved before the level rewrite -> today's parameter list.

        AdamW state is keyed by position in the parameter list; replacing
        `level_embed.weight` with `level_proj.w1/w2/w3` changes that length.
        Safe only because the level parameters are registered last in both
        the old and new module, so earlier indices still line up -- checked
        by shape before applying, rather than assumed.
        """
        if not getattr(self, "_level_migration", False):
            return sd
        groups = sd.get("param_groups", [])
        n_new = sum(len(g["params"]) for g in self.opt.param_groups)
        n_old = sum(len(g["params"]) for g in groups)
        if len(groups) != 1 or n_new - n_old != 2:
            raise RuntimeError(
                f"cannot migrate optimizer state: {n_old} saved params vs "
                f"{n_new} live, in {len(groups)} group(s)")
        last = groups[0]["params"][-1]
        tail = sd["state"].get(last, {}).get("exp_avg")
        if tail is None or tuple(tail.shape) != (2, self.model.backbone.cond_dim):
            raise RuntimeError(
                "cannot migrate optimizer state: the last saved parameter is "
                f"{None if tail is None else tuple(tail.shape)}, not the "
                "(2, cond_dim) level_embed this migration expects")
        kept = {i: s for i, s in sd["state"].items() if i != last}
        print(f"[resume] migrating optimizer state: keeping Adam moments for "
              f"{len(kept)} parameters, dropping the dead level_embed entry, "
              f"starting {n_new - len(kept)} level_proj entries fresh.")
        out = dict(sd)
        out["state"] = kept
        out["param_groups"] = [dict(groups[0], params=list(range(n_new)))]
        return out

    def _load_ckpt(self, dir: str):
        print(f"\n[Resuming training from the checkpoint directory] -> {dir}")
        # `trainer.warm_start=true`: take the weights only -- no Adam moments,
        # scheduler or EMA; step counter restarts at 0. For checkpoints too
        # old for `_migrate_opt_state` to handle.
        warm = bool(getattr(self.cfg_t, "warm_start", False))
        if warm:
            print("[resume] warm_start: loading model weights only "
                  "(no optimizer/scheduler/EMA state, step counter restarts).")
        parts = (("model", self.model),) if warm else (
            ("model", self.model), ("optimizer", self.opt), ("scheduler", self.sche))
        # `trainer.allow_fresh_optimizer=true`: keep going with fresh Adam
        # moments if optimizer.pth is unreadable, keeping weights/EMA/
        # scheduler/step count. Opt-in, never automatic.
        allow_fresh_opt = bool(getattr(self.cfg_t, "allow_fresh_optimizer", False))
        for k, v in parts:
            try:
                sd = torch.load(f"{dir}/{k}.pth", weights_only=False)
            except Exception as e:
                if k == "optimizer" and allow_fresh_opt:
                    print(f"[resume] optimizer.pth is unreadable ({type(e).__name__}: "
                          f"{e}); allow_fresh_optimizer=true, so Adam moments start "
                          f"fresh. Weights, EMA, scheduler and step count are kept.")
                    continue
                raise
            if k == "model":
                # Partial by construction: the frozen encoders were stripped on
                # save and are rebuilt from their own weights in __init__.
                missing, unexpected = v.load_state_dict(sd, strict=False)
                unexpected = list(unexpected)
                # `level_embed.weight`: dead, zero-initialised pre-rewrite
                # level embedding, superseded by `level_proj`; safe to drop.
                dead = [u for u in unexpected if "level_embed" in u]
                if dead:
                    print(f"[resume] dropping dead pre-rewrite {dead} "
                          f"(never trained; superseded by `level_proj`).")
                    unexpected = [u for u in unexpected if "level_embed" not in u]
                if unexpected:
                    raise RuntimeError(f"unexpected keys in model ckpt: {unexpected[:5]}")
                # Anything missing that is NOT a frozen encoder is a real
                # problem -- a silently untrained DiT, for instance.
                stray = list(self._strip_frozen({m: None for m in missing}))
                # Warm-starting a level-conditioned backbone from a checkpoint
                # without `level_proj`: allowed, starts from random init, not
                # a bit-identical resume.
                lvl = [s for s in stray if "level_proj" in s]
                if lvl:
                    print(f"[resume] no level_proj in this checkpoint "
                          f"({lvl}); starting it from its random init.")
                    stray = [s for s in stray if "level_proj" not in s]
                    self._level_migration = True
                if stray and warm:
                    print(f"[resume] warm_start: {len(stray)} parameter(s) not in "
                          f"this checkpoint ({stray[:5]}); using their init.")
                    stray = []
                if stray:
                    raise RuntimeError(f"missing non-frozen keys: {stray[:5]}")
            elif k == "optimizer":
                v.load_state_dict(self._migrate_opt_state(sd))
            else:
                v.load_state_dict(sd)

        if warm:
            return

        ema_path = f"{dir}/ema.pth"
        if getattr(self, "ema", None) is not None and os.path.exists(ema_path):
            esd = torch.load(ema_path, weights_only=False)
            if getattr(self, "_level_migration", False):
                # EMA shadow has the same dead level_embed and no level_proj;
                # non-strict load keeps its averaged weights and seeds
                # level_proj from the live (freshly initialised) model.
                esd = {k: v for k, v in esd.items() if "level_embed" not in k}
                miss, unexp = self.ema.load_state_dict(esd, strict=False)
                stray = [m for m in miss if "level_proj" not in m]
                if stray or list(unexp):
                    raise RuntimeError(
                        f"EMA migration mismatch: missing {stray[:5]}, "
                        f"unexpected {list(unexp)[:5]}")
                print(f"[resume] EMA shadow migrated: {len(miss)} level_proj "
                      f"tensors seeded from the live model.")
            else:
                self.ema.load_state_dict(esd)

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    @staticmethod
    def _demo_units(stems, stem_names, mode):
        """-> [(target [1, C, T], label)] for one evaluation granularity.

        `track`: one unit per active raw track.
        `stem` : one unit per MoisesDB level-1 family, that family's active
                 tracks summed (kick + hi-hat + snare -> one `drums` unit).
        """
        if mode == "track":
            return [(stems[i:i + 1], stem_names[i]) for i in range(stems.shape[0])]
        by_family: dict[str, list[int]] = {}
        for i, name_i in enumerate(stem_names):
            by_family.setdefault(_family_of(name_i), []).append(i)
        return [(stems[idxs].sum(dim=0, keepdim=True), fam)
                for fam, idxs in by_family.items()]

    @property
    def _val_selects_best(self) -> bool:
        """Is `ckpt/best` chosen by held-out SI-SDR rather than training loss?"""
        return (self.val_dataset is not None
                and bool(getattr(self.cfg_t.logging, "best_by_val_si_sdr", True)))

    def _log_metrics(self, sort_by_key: bool = True):
        """As the base, but do not let training loss claim `best_metrics`
        when held-out SI-SDR is the selection criterion (see `_val_selects_best`).
        """
        if not self._val_selects_best:
            return super()._log_metrics(sort_by_key)
        keep = {k: self.states[k] for k in ("latest_metrics", "best_metrics")
                if k in self.states}
        super()._log_metrics(sort_by_key)
        self.states.update(keep)

    def _maybe_select_best(self, prefix, mode, step, si_sdr_latent_target):
        """`ckpt/best` = the highest held-out latent-domain SI-SDR seen.

        Latent-domain rather than audio-domain because it scores what the DiT
        actually predicts, without the codec's own error on top. Only track
        mode, and only the first `logging.steps` entry, so the criterion is
        one fixed quantity. `prefix` guards against selecting on training
        songs when no `data.val` is configured.
        """
        if not self._val_selects_best or prefix != "val" or mode != "track":
            return
        if step != self.cfg_t.logging.steps[0]:
            return
        # A resumed run may carry `best_metrics` from a different criterion
        # (training loss); reset once, and record which criterion is in use.
        if self.states.get("best_metric_kind") != "val_si_sdr_latent_target":
            self.states["best_metric_kind"] = "val_si_sdr_latent_target"
            self.states["best_metrics"] = float("inf")
            print("\t->->-> best-checkpoint criterion is now val "
                  "si_sdr_latent_target; previous best_metrics discarded "
                  "(it was a training loss).")
        # Negated: `best_metrics` is a minimum everywhere else.
        score = -si_sdr_latent_target
        self.states["latest_metrics"] = score
        if score < self.states["best_metrics"]:
            self.states["best_metrics"] = score
            print(f"\t->->-> new BEST by val si_sdr_latent_target: "
                  f"{si_sdr_latent_target:.2f} dB")
            self._save_ckpt()

    @torch.no_grad()
    def _validate(self):
        """Flow-matching loss over the whole held-out val set, deterministically.

        Every song in `data.val`, every source in it as the target in turn.
        `t` is a fixed grid over [t_min, t_max] and the RNG is forked and
        reseeded, so the result is a deterministic function of the weights.
        Raw weights, not the EMA shadow, to stay comparable with `loss`.
        """
        from source_embedding_generator.protocol import LEVEL_TRACK, active_tracks

        dset = self.val_dataset
        if dset is None:
            return
        was_training = self.model.training
        self.model.eval()

        n_t = int(getattr(self.cfg_t.logging, "n_val_t", 4))
        m = self.accel.unwrap_model(self.model)
        # Midpoints of n_t equal bins: covers the schedule without ever hitting
        # the endpoints, where the loss is degenerate rather than informative.
        grid = [m.t_min + (m.t_max - m.t_min) * (i + 0.5) / n_t for i in range(n_t)]
        uses_level = getattr(m, "conditions_on_level", False)

        tot = {"loss": 0.0, "loss_linear": 0.0}
        n_items = 0
        # `train_step` draws its own x0, so the pass is only reproducible if the
        # global RNG is. Forked, not reseeded, so training's stream is untouched.
        # `p_mix_drop` is off meanwhile: scoring 10% of the val set on the
        # unconditional branch would just add variance to the number.
        p_drop, m.p_mix_drop = m.p_mix_drop, 0.0
        fork = torch.random.fork_rng(devices=[self.accel.device]
                                     if self.accel.device.type == "cuda" else [])
        fork.__enter__()
        torch.manual_seed(20260818)
        for i in range(len(dset)):
            stems, _ = dset[i]
            stems = stems.to(self.accel.device)
            keep = active_tracks(stems)
            stems = stems[keep]
            if stems.shape[0] < 2:
                continue                      # nothing to separate from
            for j in range(stems.shape[0]):
                # Target first, the contract `train_step` expects; the rest sum
                # to the complement, so every source is scored once per song.
                order = [j] + [k for k in range(stems.shape[0]) if k != j]
                x = stems[order].unsqueeze(0)
                for tv in grid:
                    out = m.train_step(
                        x, x[:, 0],
                        t=torch.full((1,), tv, device=self.accel.device),
                        level=(torch.tensor([LEVEL_TRACK], device=self.accel.device)
                               if uses_level else None),
                    )
                    tot["loss"] += float(out["loss"])
                    tot["loss_linear"] += float(out["loss_linear"])
                    n_items += 1

        fork.__exit__(None, None, None)
        m.p_mix_drop = p_drop
        if was_training:
            self.model.train()
        if n_items == 0:
            return
        metrics = {f"val/{k}": v / n_items for k, v in tot.items()}
        metrics["val/n"] = n_items
        wandb.log(metrics, step=self.states["global_step"])
        print("\t->->-> val: " + "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if k != "val/n")
            + f"  ({n_items} scored)")
        # Deliberately NOT wired into `best_metrics`: that still tracks training
        # loss, as it always has. `_log_metrics` rewrites `latest_metrics` every
        # `n_step_log`, so the two would fight over the same slot on different
        # scales. Watch the `val/loss` curve for overfitting; changing which
        # checkpoint is called "best" is a separate decision.

    @torch.no_grad()
    def _sample_demo(self):
        from source_embedding_generator.protocol import (
            LEVEL_STEM,
            LEVEL_TRACK,
            active_tracks,
        )

        self.model.eval()
        # Falls back to the training dataset when no `data.val` is configured;
        # `prefix` then marks these numbers as fit, not generalisation.
        dset = self.val_dataset if self.val_dataset is not None else self.dataloader.dataset
        prefix = "val" if self.val_dataset is not None else "demo"
        uses_level = getattr(self.model, "conditions_on_level", False)
        sr = dset.sample_rate
        n_sample: int = self.cfg_t.logging.n_samples_per_step

        # Fixed songs, evenly spaced through the val set, so the demo table is
        # a before/after of the same songs rather than a random redraw.
        idxs = [(k * len(dset)) // n_sample for k in range(n_sample)]
        items = [dset[int(i)] for i in idxs]

        for step in self.cfg_t.logging.steps:
            # Two granularities, scored and tabled separately; each wrapped
            # so a failure in one does not cost the other.
            _levels = {"track": LEVEL_TRACK, "stem": LEVEL_STEM}
            _modes = self.cfg_t.logging.get("demo_modes", ["track", "stem"])
            for mode, level_id in ((m, _levels[m]) for m in _modes):
                try:
                    level = (torch.tensor([level_id], device=self.accel.device)
                             if uses_level else None)
                    # Accumulated across every demo song, so these can be
                    # plotted as scalars over training, not just table cells.
                    acc = {"target": [], "complement": [], "ceiling": [],
                           "latent_target": [], "latent_complement": []}
                    per_family: dict[str, list[float]] = {}
                    per_family_latent: dict[str, list[float]] = {}

                    for idx, (stems, info) in enumerate(items):
                        stems = stems.to(self.accel.device)
                        mix = stems.sum(dim=0, keepdim=True)          # [1, C, T]
                        peak = mix.abs().max()
                        if peak > 1:
                            mix, stems = mix / peak, stems / peak

                        song_id, artist, name, offset, stem_names = info
                        # Only separate tracks that are actually audible in
                        # this crop.
                        keep = active_tracks(stems)
                        stems = stems[keep]
                        stem_names = [s for s, k in zip(stem_names, keep.tolist()) if k]

                        units = self._demo_units(stems, stem_names, mode)
                        n = len(units)

                        cols = ["stem", "gt", "vae(gt) [ceiling]", "separated",
                                "gt complement", "separated complement",
                                "SI-SDR vae(gt)", "SI-SDR separated",
                                "SI-SDR separated complement", "SI-SDR latent"]
                        table = wandb.Table(columns=cols)

                        for tgt, name_i in units:
                            comp = mix - tgt

                            sep_t, sep_c, z_t, z_c = self.model.sample(
                                queries=tgt, mix=mix, n_step=step,
                                return_latent=True, level=level,
                                **self._extra_sample_kwargs(step),
                            )
                            # Mono reference: the codec is mono, so comparing
                            # against the stereo stem would charge the model for
                            # downmixing.
                            ref_t = tgt.mean(dim=-2, keepdim=True)
                            ref_c = comp.mean(dim=-2, keepdim=True)
                            vae_t = self.model.decode(self.model.encode(tgt))

                            def aud(x):
                                return wandb.Audio(
                                    x[0].clamp(-1, 1).cpu().numpy().T, sample_rate=sr
                                )

                            si_t = si_sdr(sep_t, ref_t)
                            si_c = si_sdr(sep_c, ref_c)
                            si_v = si_sdr(vae_t, ref_t)

                            # Latent-space SI-SDR: prediction vs encoded ground
                            # truth, excluding the codec's own error.
                            si_lat_t = si_sdr(z_t, self.model.encode(tgt))
                            si_lat_c = si_sdr(z_c, self.model.encode(comp))
                            acc["latent_target"].append(si_lat_t)
                            acc["latent_complement"].append(si_lat_c)
                            per_family_latent.setdefault(
                                _family_of(name_i), []).append(si_lat_t)

                            table.add_data(
                                name_i,
                                aud(ref_t), aud(vae_t), aud(sep_t),
                                aud(ref_c), aud(sep_c),
                                si_v, si_t, si_c, si_lat_t,
                            )

                            acc["target"].append(si_t)
                            acc["complement"].append(si_c)
                            acc["ceiling"].append(si_v)
                            per_family.setdefault(_family_of(name_i), []).append(si_t)

                        self.accel.log(
                            {f"{step}-step {mode} sample {idx}": table,
                             f"mix {idx}": wandb.Audio(
                                 mix[0].clamp(-1, 1).cpu().numpy().T, sample_rate=sr)},
                            step=self.states["global_step"],
                        )
                        print(f"\t->->-> demo {idx} [{mode}]: {artist} - {name} "
                              f"({song_id}) @ {offset / sr:.1f}s, {n} units")

                    if acc["target"]:
                        gstep = self.states["global_step"]
                        log = {
                            f"{prefix}/{mode}/{step}step/si_sdr_target":
                                float(np.mean(acc["target"])),
                            f"{prefix}/{mode}/{step}step/si_sdr_complement":
                                float(np.mean(acc["complement"])),
                            # Codec ceiling and the gap to it (only the gap is
                            # the separator's to close).
                            f"{prefix}/{mode}/{step}step/si_sdr_vae_ceiling":
                                float(np.mean(acc["ceiling"])),
                            f"{prefix}/{mode}/{step}step/si_sdr_gap_to_ceiling":
                                float(np.mean(acc["ceiling"]) - np.mean(acc["target"])),
                            f"{prefix}/{mode}/{step}step/n_tracks_scored":
                                len(acc["target"]),
                            f"{prefix}/{mode}/{step}step/si_sdr_latent_target":
                                float(np.mean(acc["latent_target"])),
                            f"{prefix}/{mode}/{step}step/si_sdr_latent_complement":
                                float(np.mean(acc["latent_complement"])),
                        }
                        # One curve per level-1 family: a mean over everything
                        # hides a model that nails drums and cannot touch vocals.
                        for fam, vals in per_family.items():
                            log[f"{prefix}/{mode}/{step}step/si_sdr/{fam}"] = \
                                float(np.mean(vals))
                        for fam, vals in per_family_latent.items():
                            log[f"{prefix}/{mode}/{step}step/si_sdr_latent/{fam}"] = \
                                float(np.mean(vals))

                        try:
                            log[f"{prefix}/{mode}/{step}step/si_sdr_per_family_chart"] = \
                                wandb.Image(_si_sdr_chart(per_family, gstep))
                        except Exception as e:
                            print(f"[warn] si_sdr chart failed: {e}")

                        self.accel.log(log, step=gstep)
                        self._maybe_select_best(
                            prefix, mode, step,
                            float(np.mean(acc["latent_target"])))
                        print(f"\t->->-> [{mode}] SI-SDR target "
                              f"{np.mean(acc['target']):.2f} dB "
                              f"| ceiling {np.mean(acc['ceiling']):.2f} dB "
                              f"| latent {np.mean(acc['latent_target']):.2f} dB "
                              f"| {len(acc['target'])} units")


                except Exception as e:
                    print(f"[warn] demo mode '{mode}' failed, continuing: {e}")
        self.model.train()
        print("\t->->-> Sampled.")

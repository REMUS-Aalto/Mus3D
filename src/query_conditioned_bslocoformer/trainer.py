import json
import math
import os
import shutil
import signal
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import wandb
from utils.logging import MetricsLogger
from utils.torch_common import exists, print_once, sort_dict


class Trainer:
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
        val_dataset=None,
    ):
        self.model = accel.unwrap_model(model)
        self.ema = ema
        self.opt = optimizer
        self.sche = scheduler
        self.dataloader = dataloader
        # Held-out demo/eval source, from `data.val` if configured; None
        # falls back to the training dataset in `_sample_demo`.
        self.val_dataset = val_dataset
        self.accel = accel
        self.cfg = cfg
        self.cfg_t = cfg.trainer
        self.EPS = 1e-8

        # Timestep sampler for flow matching training
        self.ts_sampler = (
            hydra.utils.instantiate(self.cfg_t.timestep_sampler)
            if "timestep_sampler" in self.cfg_t
            else None
        )

        # Non-finite-step bookkeeping (see `_is_nonfinite_step`).
        self._n_nonfinite = 0
        self._n_dumped = 0

        self.wandb_logger = MetricsLogger()
        self.cli_logger = MetricsLogger()
        self.test_logger = MetricsLogger()
        self._t_loss_buf: list = []

        self.states = {
            "global_step": 0,
            "best_metrics": float("inf"),
            "prev_metrics": float("inf"),
        }

        # Time measurement
        if torch.cuda.is_available():
            self.s_event = torch.cuda.Event(enable_timing=True)
            self.e_event = torch.cuda.Event(enable_timing=True)
        else:
            self.s_event = None
            self.e_event = None

        # Signal handling for graceful termination
        self.terminate_requested = False
        self._setup_signal_handlers()

        # Resume training
        if ckpt_dir is not None:
            self._load_ckpt(ckpt_dir)

    def _setup_signal_handlers(self):
        """
        Set up signal handlers for graceful termination.
        Handles SIGTERM (SLURM job termination) and SIGINT (Ctrl+C).
        Only the main process should handle signals to avoid race conditions.
        """
        save = getattr(self.cfg_t, "save_ckpt_on_exit", True)
        if not save:
            # Let Ctrl+C kill immediately with no output across all processes
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
            v.load_state_dict(torch.load(f"{dir}/{k}.pth", weights_only=False))

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    def _log_loss_per_t(self, n_bins: int = 20):
        if not self._t_loss_buf:
            return
        t_all = torch.cat([x[0] for x in self._t_loss_buf]).numpy()
        db_all = torch.cat([x[1] for x in self._t_loss_buf]).numpy()
        lin_all = torch.cat([x[2] for x in self._t_loss_buf]).numpy()
        raw_all = torch.cat([x[3] for x in self._t_loss_buf]).numpy()
        self._t_loss_buf.clear()

        bins = np.linspace(0, 1, n_bins + 1)
        bidx = np.digitize(t_all, bins, right=True).clip(1, n_bins) - 1
        centers = 0.5 * (bins[:-1] + bins[1:])
        w = 1 / n_bins * 0.8

        def bin_stats(vals):
            means = np.array(
                [
                    vals[bidx == i].mean() if (bidx == i).any() else np.nan
                    for i in range(n_bins)
                ]
            )
            stds = np.array(
                [
                    vals[bidx == i].std() if (bidx == i).any() else 0.0
                    for i in range(n_bins)
                ]
            )
            return means, stds

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for ax, vals, ylabel, title in zip(
            axes,
            [db_all, lin_all, raw_all],
            ["loss (dB)", "loss (linear, normalized)", "loss (unnormalized MSE)"],
            ["Loss dB vs t", "Loss linear vs t", "Loss unnorm vs t"],
        ):
            m, s = bin_stats(vals)
            ax.bar(centers, m, width=w, yerr=s, capsize=3)
            ax.set_xlabel("t")
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        fig.tight_layout()
        tmp_path = "/tmp/loss_per_t.png"
        fig.savefig(tmp_path, dpi=50)
        plt.close(fig)
        try:
            self.accel.log(
                {"loss_per_t": wandb.Image(tmp_path)}, step=self.states["global_step"]
            )
        except Exception as e:
            print(f"[warn] loss_per_t wandb log failed: {e}")

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

            # Update
            metrics = self.run_step(batch)

            if self.accel.is_main_process:
                self.wandb_logger.add(metrics)
                self.cli_logger.add(metrics)

                # Log
                if self._its_time(self.cfg_t.logging.n_step_log):
                    self._log_metrics()

                # Print
                if self._its_time(self.cfg_t.logging.n_step_print):
                    self._print_metrics()

                # Save Checkpoint
                if self._its_time(self.cfg_t.logging.n_step_ckpt):
                    self._save_ckpt()

                # Validate: fixed, exhaustive held-out pass (see
                # LatentFlowTrainer._validate); no-op without a val set.
                n_step_val = getattr(self.cfg_t.logging, "n_step_val",
                                     self.cfg_t.logging.n_step_sample)
                if hasattr(self, "_validate") and self._its_time(n_step_val):
                    try:
                        self._validate()
                    except Exception as e:
                        print(f"[warn] _validate failed: {e}")

                # Sample
                if self._its_time(self.cfg_t.logging.n_step_sample):
                    try:
                        self._sample_demo()
                    except Exception as e:
                        print(f"[warn] _sample_demo failed: {e}")

                # Per t log
                if self._its_time(self.cfg_t.logging.n_step_figure):
                    self._log_loss_per_t()

            self.states["global_step"] += 1

        # Workers are shut down once the end of the iteration is reached, or when the iterator becomes garbage collected.
        del dataloader_iter

        # Save checkpoint before exiting
        if self.accel.is_main_process and self.terminate_requested:
            print("\n[Termination requested]")
            if getattr(self.cfg_t, "save_ckpt_on_exit", True):
                print("Saving final checkpoint...")
                self._save_ckpt()
            wandb.finish()
        sys.exit(0)

    def _extra_train_kwargs(self, batch) -> dict:
        """Per-model extras pulled out of the batch and passed to `train_step`.

        Empty here: the Locoformer conditions on the query alone. Subclasses
        whose model consumes more of the item override this -- see
        `LatentFlowTrainer`, which forwards the hierarchy level.
        """
        return {}

    def _prepare_batch(self, batch):
        """Collated batch -> `(stems, queries, extras)` for `train_step`.

        Some datasets yield `(stems, level, ...)`; extras are left to
        `_extra_train_kwargs`. Split out of `run_step` so a subclass can
        reshape the item before the model sees it.
        """
        stems = batch[0] if isinstance(batch, (tuple, list)) else batch
        return stems, stems[:, 0, :], self._extra_train_kwargs(batch)

    def run_step(self, batch, train: bool = True):
        stems, queries, extras = self._prepare_batch(batch)
        bs = stems.shape[0]

        # Sample timestep
        t = (
            self.ts_sampler.sample(bs, device=self.accel.device)
            if self.ts_sampler
            else None
        )

        # Update
        if train:
            self.opt.zero_grad()

        # Uses the first stem of each batch as the query for training
        # stems [B, N, C, T] and queries [B, C, T]
        output = self.model.train_step(stems, queries, t=t,
                                        debug=self.cfg_t.debug,
                                        **extras)

        skipped = False
        if train:
            self.accel.backward(output["loss"])
            grad_norm = None
            if self.accel.sync_gradients:
                grad_norm = self.accel.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg_t.max_grad_norm,
                )

            # One non-finite gradient is enough to end a run: clip_grad_norm_
            # divides every gradient by the global norm, so a single NaN
            # anywhere makes the whole step NaN. Drop the step instead.
            if self._is_nonfinite_step(output["loss"], grad_norm):
                skipped = True
                self._report_nonfinite(batch, stems, queries, output, grad_norm)
                self.opt.zero_grad(set_to_none=True)
                # The LR schedule stays on the global-step clock: a skipped
                # step is a dropped batch, not a rewind.
                self.sche.step()
            else:
                self.opt.step()
                self.sche.step()

                # EMA
                if exists(self.ema):
                    self.ema.update()

        # Detach tensors
        result = {k: v.detach() for k, v in output.items()}

        if train:
            # Report the skip rate, and keep the non-finite values themselves
            # out of the running means so the CLI and wandb keep showing the
            # loss of the steps that actually trained.
            zero = torch.zeros((), device=self.accel.device)
            if skipped:
                result = {"nonfinite_skips": zero + 1.0}
            else:
                result["nonfinite_skips"] = zero

        # Store per-sample loss for t/loss plots
        if "t" in result and "loss_per_sample" in result:
            self._t_loss_buf.append(
                (
                    result.pop("t").float().cpu(),
                    result.pop("loss_per_sample").float().cpu(),
                    result.pop("loss_linear_per_sample").float().cpu(),
                    result.pop("loss_unnorm_per_sample").float().cpu(),
                )
            )

        return result

    def _is_nonfinite_step(self, loss, grad_norm) -> bool:
        """True when this step must not be applied to the weights.

        Off via `trainer.skip_nonfinite_steps=false` for anyone who would
        rather the run die loudly than train through a bad batch.
        """
        if not self.cfg_t.get("skip_nonfinite_steps", True):
            return False
        bad = not bool(torch.isfinite(loss.detach()))
        if grad_norm is not None:
            bad = bad or not math.isfinite(float(grad_norm))
        return bad

    def _report_nonfinite(self, batch, stems, queries, output, grad_norm):
        """Say what went non-finite, and keep the batch that did it.

        The batch is the only thing that can identify the trigger after the
        fact -- the RNG state that drew it is gone by the next step -- so the
        first few are written to `<output_dir>/nan_dumps/`. Cheap: this runs
        only on a step that was already thrown away.
        """
        self._n_nonfinite += 1
        step = self.states["global_step"]

        finite_in = bool(torch.isfinite(stems).all())
        finite_q = bool(torch.isfinite(queries).all())
        which = [k for k, v in output.items()
                 if torch.is_tensor(v) and v.numel() == 1
                 and not bool(torch.isfinite(v))]
        print(f"\n[nonfinite] step {step}: SKIPPED "
              f"(loss keys {which}, grad_norm={grad_norm}, "
              f"batch finite={finite_in}, query finite={finite_q}) "
              f"-- {self._n_nonfinite} skipped so far")
        if not finite_in:
            per_slot = [
                (int(i), int(j))
                for i in range(stems.shape[0])
                for j in range(stems.shape[1])
                if not bool(torch.isfinite(stems[i, j]).all())
            ]
            print(f"[nonfinite] non-finite (item, slot) pairs: {per_slot}")

        max_dumps = int(self.cfg_t.get("max_nonfinite_dumps", 3))
        if self._n_dumped < max_dumps:
            d = self.cfg_t.output_dir + "/nan_dumps"
            os.makedirs(d, exist_ok=True)
            path = f"{d}/step_{step}.pt"
            try:
                torch.save(
                    {"batch": [b.detach().cpu() if torch.is_tensor(b) else b
                               for b in (batch if isinstance(batch, (tuple, list))
                                         else [batch])],
                     "grad_norm": float(grad_norm) if grad_norm is not None else None,
                     "global_step": step},
                    path,
                )
                self._n_dumped += 1
                print(f"[nonfinite] batch written to {path}")
            except Exception as e:
                print(f"[nonfinite] could not dump the batch: {e}")

        try:
            self.accel.log({"nonfinite_step": 1.0}, step=step)
        except Exception:
            pass

    def _its_time(self, itv: int):
        return (self.states["global_step"] - 1) % itv == 0

    def _log_metrics(self, sort_by_key: bool = True):
        metrics = self.wandb_logger.pop()

        # Learning rate
        metrics["lr"] = self.sche.get_last_lr()[0]
        if sort_by_key:
            metrics = sort_dict(metrics)

        self.accel.log(metrics, step=self.states["global_step"])

        # Update states
        m_for_ckpt = self.cfg_t.logging.metrics_for_best_ckpt
        # A window in which every step was skipped as non-finite carries none
        # of these keys; leave the previous values standing rather than die.
        if all(k in metrics for k in m_for_ckpt):
            m_latest = float(sum([metrics[k].detach() for k in m_for_ckpt]))
            self.states["latest_metrics"] = m_latest
            # NaN loses every comparison, so a poisoned window can never
            # become the best -- which is what we want.
            if m_latest < self.states["best_metrics"]:
                self.states["best_metrics"] = m_latest

    def _print_metrics(self, sort_by_key: bool = True):
        if self.e_event is not None and self.s_event is not None:
            self.e_event.record()
            torch.cuda.synchronize()
            p_time = self.s_event.elapsed_time(self.e_event) / 1000.0  # [sec]
        else:
            p_time = 0.0

        metrics = self.cli_logger.pop()
        # tensor to scalar
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

    def _save_ckpt(self):
        out_dir = self.cfg_t.output_dir + "/ckpt"

        # Save latest checkpoint
        latest_dir = out_dir + "/latest"
        os.makedirs(latest_dir, exist_ok=True)
        ckpts = {
            "model": self.model,
            "optimizer": self.opt,
            "scheduler": self.sche,
        }
        for name, m in ckpts.items():
            torch.save(m.state_dict(), f"{latest_dir}/{name}.pth")

        # Save states and configuration
        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        # Save best checkpoint
        if self.states["latest_metrics"] == self.states["best_metrics"]:
            shutil.copytree(latest_dir, out_dir + "/best", dirs_exist_ok=True)

        print("\t->->-> Saved checkpoints.")

    @torch.no_grad()
    def _sample_demo(self):
        self.model.eval()

        steps: list = self.cfg_t.logging.steps
        n_sample: int = self.cfg_t.logging.n_samples_per_step

        # Load a random sample
        dset = self.val_dataset if self.val_dataset is not None else self.dataloader.dataset
        idxs = torch.randint(len(dset), size=(n_sample,))
        stems_list, info = zip(*[dset[idx] for idx in idxs])

        for step in steps:
            for idx in range(n_sample):
                # WANDB logging table columns
                columns = ["info", "gt", "sep"]
                wandb_table = wandb.Table(columns)

                stems = stems_list[idx].to(self.accel.device).unsqueeze(0)
                mix = stems.sum(dim=1)

                # For sample demo adjust the mix and stem volume so the mix does not clip
                max_peak = mix.abs().max()
                if max_peak > 1:
                    mix /= max_peak
                    stems /= max_peak

                sep_stems = self.model.sample(
                    queries=stems,
                    sources=stems,
                    n_step=step,
                )

                sep_mix = sep_stems.sum(dim=1)

                id, artist, name, offset, stem_names = info[idx]
                offset_sec = offset / dset.sample_rate
                minutes = int(offset_sec // 60)
                seconds = int(offset_sec % 60)
                offset_formatted = f"{minutes}:{seconds:02d}"

                info_data = wandb.Html(
                    "<br>".join([id, artist, name, offset_formatted])
                )

                def norm(x):
                    return wandb.Audio(
                        x.clamp(-1, 1).cpu().numpy().T, sample_rate=dset.sample_rate
                    )

                table_rows = [[info_data, norm(mix[0]), norm(sep_mix[0])]]
                for i in range(stems.shape[1]):
                    table_rows += [
                        [wandb.Html(stem_names[i]), norm(stems[0, i]), norm(sep_stems[0, i])]
                    ]

                for row in table_rows:
                    wandb_table.add_data(*row)

                self.accel.log(
                    {f"{step}-step sample {idx}": wandb_table},
                    step=self.states["global_step"],
                )

        self.model.train()

        print("\t->->-> Sampled.")

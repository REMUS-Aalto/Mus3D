"""Flow-matching generator for recursive source extraction.

One step of the recursion: given the mixture latent, the units already
extracted, and a hierarchy level, generate the next source embedding and
the new residual.

    train_step(batch)      one step, teacher-forced on true available units
    extract(mixture, ...)  the full loop, one source at a time, until it stops

Target is the PaSST embedding PCA-projected to 128 dimensions. `t=1` is
noise, `t=0` is data, `x_t = t*x0 + (1-t)*x1`, net regresses `v = x0 - x1`.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from torch import nn

from source_embedding_generator.stochastic_sampling import edm_churn_step
from source_embedding_generator.protocol import TorchLoudness


NO_AUDIO = {"est": None, "comp": None, "leftover": None}


class RecursiveFMGenerator(nn.Module):
    """See module docstring. Batch is `(slots, counts, labels, levels)`."""

    def __init__(
        self,
        backbone: nn.Module,
        query_model: nn.Module,
        query_adapter: nn.Module,
        mixture_encoder: nn.Module,
        diffusion_params: object,
        sample_rate: int = 44100,
        embed_batch: int = 8,
        use_mixture_passt: bool = False,
        use_residual_embedding: bool = False,
        stop_cos: float = 0.9,
        separator=None,
        sep_use_ema: bool = True,
        sep_steps: int = 32,
        sep_churn: float = 0.0,
        init_from=None,
        init_use_ema: bool = False,
    ):
        super().__init__()

        self.backbone = backbone
        self.query_model = query_model
        self.query_adapter = query_adapter
        self.mixture_encoder = mixture_encoder
        self.loudness = TorchLoudness(sample_rate=sample_rate)

        self.sigma_data = diffusion_params.sigma_data
        self.t_min = diffusion_params.t_min
        self.t_max = diffusion_params.t_max
        self.target_scale = float(self.sigma_data)

        self.embed_batch = embed_batch

        self.use_mixture_passt = use_mixture_passt
        backbone_dim = getattr(backbone, "mixture_emb_dim", None)
        if use_mixture_passt != (backbone_dim is not None):
            raise ValueError(
                f"model.use_mixture_passt={use_mixture_passt} but "
                f"backbone.mixture_emb_dim={backbone_dim}; set both or neither."
            )

        self.use_residual_embedding = use_residual_embedding
        backbone_remb = getattr(backbone, "res_emb_dim", None)
        if use_residual_embedding != (backbone_remb is not None):
            raise ValueError(
                f"model.use_residual_embedding={use_residual_embedding} but "
                f"backbone.res_emb_dim={backbone_remb}; set both or neither."
            )

        for m in (self.query_model, self.mixture_encoder):
            m.eval().requires_grad_(False)

        if stop_cos is None:
            raise ValueError(
                "model.stop_cos must be a cosine threshold. It was briefly "
                "nullable, to select a loudness-share rule that no longer exists."
            )
        self.stop_cos = float(stop_cos)
        self._stop_token = None
        self._rng = np.random.default_rng()

        self._sep = [separator]
        # Separator ODE budget and churn; ignored when `separator` is
        # already a loaded module.
        self.sep_use_ema = bool(sep_use_ema)
        self.sep_steps = int(sep_steps)
        self.sep_churn = float(sep_churn)

        # Warm start, applied last so it overrides each module's own init.
        # See `_warm_start`.
        if init_from is not None:
            self._warm_start(init_from, use_ema=init_use_ema)

    @staticmethod
    def _normalise_donor(sd: dict, own_keys: set) -> dict:
        """Donor state dict, in whatever form it was saved, -> our key layout.

        `ema.pth` keys are `ema_model.<...>` with no `backbone.` prefix
        (the EMA is kept over the backbone only, see `train_seg.ema_fn`),
        so that prefix is restored when needed. `online_model.*` and the
        wrapper's own scalars are dropped.
        """
        out = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            key = k[len("module."):] if k.startswith("module.") else k
            if key.startswith("online_model.") or key in ("initted", "step", "decay"):
                continue
            if key.startswith("ema_model."):
                key = key[len("ema_model."):]
            out[key] = v
        if out and not (set(out) & own_keys):
            prefixed = {f"backbone.{k}": v for k, v in out.items()}
            if set(prefixed) & own_keys:
                return prefixed
        return out

    def _warm_start(self, init_from, use_ema: bool = False, min_frac: float = 0.5):
        """Load another SEG run's weights into this one. Non-strict, and loud.

        `init_from` is a checkpoint directory (`model.pth`/`ema.pth`) or a
        `.pth` file. Non-strict so a warm start can add modules the donor
        lacks; guarded by `min_frac` (raises if too little of this model
        was filled) and by printing every missing/unexpected key. Frozen
        encoders (`query_model.`, `mixture_encoder.`) are excluded from the
        count since the trainer strips and rebuilds them separately.
        """
        from pathlib import Path

        src = Path(init_from)
        if src.is_dir():
            name = "ema.pth" if use_ema else "model.pth"
            if not (src / name).exists():
                raise FileNotFoundError(f"{src} has no {name}")
            src = src / name
        sd = torch.load(src, map_location="cpu", weights_only=False)
        # Take the first mapping that looks like a model state dict.
        if not any(isinstance(v, torch.Tensor) for v in sd.values()):
            for k in ("module", "model", "shadow_params", "state_dict"):
                if isinstance(sd.get(k), dict):
                    sd = sd[k]
                    break

        frozen = ("query_model.", "mixture_encoder.", "query_adapter.")
        own = {k: v for k, v in self.state_dict().items()
               if not k.startswith(frozen)}
        sd = self._normalise_donor(sd, set(own))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith(frozen)]
        filled = len(own) - len(missing)
        frac = filled / max(len(own), 1)
        print(f"[warm start] {src}\n"
              f"             {filled}/{len(own)} tensors filled ({frac:.1%})")
        if missing:
            print(f"             MISSING (kept as initialised, {len(missing)}): "
                  f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"             UNUSED from donor ({len(unexpected)}): "
                  f"{unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
        if frac < min_frac:
            raise ValueError(
                f"warm start filled only {frac:.1%} of this model from {src}; "
                f"that is a mismatched donor, not a warm start.")

    @property
    def separator(self):
        """The frozen separator, or None. A path is loaded on first access."""
        sep = self._sep[0]
        if sep is None or isinstance(sep, bool):
            return None
        if isinstance(sep, (str, os.PathLike)):
            from source_embedding_generator.separator_io import load_separator
            sep = load_separator(sep, device=next(self.parameters()).device,
                                 use_ema=self.sep_use_ema)
            self._sep[0] = sep
        return sep

    def __setattr__(self, name, value):
        """Route `model.separator = ...` to the property setter above.

        Needed because `nn.Module.__setattr__` otherwise intercepts an
        `nn.Module` value and files it in `_modules` without consulting the
        property setter.
        """
        if name == "separator":
            RecursiveFMGenerator.separator.fset(self, value)
            return
        super().__setattr__(name, value)

    @separator.setter
    def separator(self, sep):
        """Attach a separator after construction (e.g. chosen at eval time)."""
        self._sep[0] = sep

    @torch.no_grad()
    def separate(self, mixture, run, n_step=None):
        """Parallel separation: one separator ODE over all queries at once.

        `run` is the dict `inference` has built so far; embeddings are read
        from it. -> `{"est", "comp", "leftover"}`, or None with no separator.

            est        [n, C, T] what each query pulled out.
            comp       [n, C, T] everything else, for that query.
            leftover   always None: every query is separated independently
                       from the same mixture, so there is no partition.

        Generated query vectors go in with `query_is_embedding=True`, since
        SEG and the separator share the PaSST -> PCA-128 query space.
        """
        sep = self.separator
        if sep is None:
            return None
        n_step = self.sep_steps if n_step is None else int(n_step)
        q = self._as_queries(run["emb"], mixture.device)
        k = q.shape[0]
        lvl = self._level_for(run, k, mixture.device)
        if hasattr(sep, "sample_latent"):
            lat = sep.encode(mixture)
            z_t, z_c = sep.sample_latent(lat.expand(k, -1, -1), q,
                                         n_step=n_step, level=lvl,
                                         churn=self.sep_churn)
            out = sep.decode(torch.cat([z_t, z_c], 0))
            est, comp = out[:k], out[k:]
        else:
            # Waveform separator (query BS-Locoformer): one forward call,
            # [1, k, C, T]; the complement is what the estimate leaves.
            est = sep.sample(q.unsqueeze(0), mix=mixture, query_is_embedding=True,
                             level=lvl)[0]
            comp = mixture - est
        return {"est": est, "comp": comp, "leftover": None}

    def _level_for(self, run, k, device):
        """The run's level as [k] long, one entry per query, or None.

        `inference` records whatever it was given -- an int or a [B] tensor --
        and every query of one run shares it, since a run decomposes one song at
        one granularity.
        """
        lvl = run.get("level")
        if lvl is None:
            return None
        if torch.is_tensor(lvl):
            return lvl.reshape(-1)[:1].to(device).expand(k)
        return torch.full((k,), int(lvl), dtype=torch.long, device=device)

    def _as_queries(self, emb, device):
        """[n, D] embeddings, numpy or tensor -> a float tensor on `device`."""
        if not torch.is_tensor(emb):
            emb = torch.from_numpy(np.ascontiguousarray(emb))
        return emb.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def inference(self, mixture, level=0, max_steps=12, steps=32,
                  audio: bool = True, **extract_kw) -> dict:
        """One song, decomposed -> everything a caller needs to log it.

            emb        [n, D] extracted embeddings, in extraction order,
                       truncated to where this song stopped.
            stop_cos   [n] cosine of each step's residual to the stop token.
            n          where it stopped; `n_ran` how many steps the batch ran.
            residuals  [n, D] what the model said was left after each step;
                       `mixture_emb` the embedding that trajectory starts from.
            est/comp/leftover
                       audio from running the separator (see `separate`),
                       or None with no separator or `audio=False`.

        `mixture` is [1, C, T] (or [B, C, T], reading only the first item).
        """
        emb, scos, n_ran = self.extract(
            mixture, level=level, max_steps=max_steps, steps=steps,
            **extract_kw)
        out = self._pack(emb, scos, n_ran, level=level,
                         residuals=getattr(self, "last_residuals", None),
                         mixture_emb=getattr(self, "last_mixture_emb", None))
        out.update(NO_AUDIO)
        if audio:
            out.update(self.separate(mixture, out) or NO_AUDIO)
        return out

    def _pack(self, emb, scos, n_ran, level=0, residuals=None,
              mixture_emb=None) -> dict:
        """`extract`'s output -> the `inference` dict.

        Where this song stopped is read off the trajectory (first step whose
        residual is the stop token), not off the loop length, since `extract`
        runs until every item in the batch is done.
        """
        emb = emb[0].cpu().numpy()
        scos = scos[0].cpu().numpy()
        below = np.nonzero(scos > self.stop_cos)[0]
        n = int(below[0]) + 1 if len(below) else n_ran
        return {
            "emb": emb[:n],
            "stop_cos": scos[:n],
            "n": n,
            "n_ran": n_ran,
            "level": level,
            "residuals": (None if residuals is None
                          else residuals[0].float().cpu().numpy()[:n]),
            "mixture_emb": (None if mixture_emb is None
                            else mixture_emb[0].float().cpu().numpy()),
        }

    def train(self, mode: bool = True):
        """Keep the frozen encoders in eval regardless of the trainer.

        PaSST's mel front-end applies SpecAugment in training mode, which would
        make every training TARGET a different vector for the same audio.
        """
        super().train(mode)
        self.query_model.eval()
        self.mixture_encoder.eval()
        return self

    @torch.no_grad()
    def embed_mixture(self, mix):
        """[B,C,T] mixture -> [B, D_pca] PaSST/PCA embedding, or None.

        Same frozen encoder and same PCA the targets are built with, so this
        vector is directly comparable to what the model is being asked to
        generate. Returns None when the feature is off, which is what the
        backbone expects to receive in that case.
        """
        if not self.use_mixture_passt:
            return None
        return self.query_adapter(self._chunked(self.query_model, mix))

    def _passt_kwargs(self, mix) -> dict:
        """`{"mixture_passt": ...}` when enabled, `{}` otherwise."""
        if not self.use_mixture_passt:
            return {}
        return {"mixture_passt": self.embed_mixture(mix)}

    def _residual_kwargs(self, remaining, res_emb=None) -> dict:
        """Conditioning derived from what is still unextracted -> forward kwargs.

        `remaining` is that audio (training: target + residual; inference
        step 0: the mixture). `res_emb`, if passed, supplies its embedding
        directly instead of recomputing it.
        """
        out = {}
        if self.use_residual_embedding:
            if res_emb is None:
                res_emb = self.query_adapter(
                    self._chunked(self.query_model, remaining))
            out["res_emb"] = res_emb
        return out

    def _chunked(self, fn, x):
        """Apply a frozen encoder over dim 0 in `embed_batch`-sized pieces."""
        if x.shape[0] <= self.embed_batch:
            return fn(x)
        return torch.cat([fn(c) for c in x.split(self.embed_batch)], dim=0)

    @torch.no_grad()
    def stop_token(self, device) -> torch.Tensor:
        """[1, D_pca] unit vector: the symbol that means "nothing is left".

        The last coordinate axis of the PCA output space, e_{D-1} -- the
        least-variance direction, so it stays far from real embeddings; see
        the paper for why this stopping symbol was chosen. Cached per device.
        """
        if self._stop_token is None or self._stop_token.device != device:
            v = torch.zeros(1, self.backbone.embedding_dim, device=device)
            v[0, -1] = 1.0
            self._stop_token = torch.nn.functional.normalize(v, dim=-1)
        return self._stop_token

    @torch.no_grad()
    def stop_target(self, device) -> torch.Tensor:
        """[1, D_pca] the stop symbol at the scale of a real target.

        `stop_token` is normalised for detection; this is what an empty
        residual is trained to produce, scaled by `sqrt(D) * sigma_data`,
        the expected norm of a real embedding.
        """
        d = self.backbone.embedding_dim
        return self.stop_token(device) * (d ** 0.5) * float(self.sigma_data)

    def sample_time_training(self, n: int) -> torch.Tensor:
        return torch.rand(n) * (self.t_max - self.t_min) + self.t_min

    def _split(self, counts):
        """counts (= C_i + 2) -> (index of each item's target, of its residual)."""
        ends = torch.cumsum(counts, 0)
        return ends - 2, ends - 1

    @torch.no_grad()
    def _describe(self, slots, item_idx, n_items, counts):
        """Packed slots -> (targets [B,2,D], available [B,C_max,D], mask).

        All slots are embedded in one PaSST pass; the leading C are
        conditioning, the last two are the generation targets. Descriptor
        is the 128-d PCA embedding.
        """
        desc = self.query_adapter(self._chunked(self.query_model, slots))

        tgt_i, res_i = self._split(counts)

        empty = slots[res_i].pow(2).mean(dim=(-2, -1)).sqrt() < 1e-6
        if bool(empty.any()):
            idx = res_i[empty]
            desc[idx] = self.stop_target(desc.device).to(desc.dtype)

        targets = torch.stack([desc[tgt_i], desc[res_i]], dim=1)

        n_avail = (counts - 2).clamp(min=0)
        c_max = int(n_avail.max())
        avail = desc.new_zeros(n_items, max(c_max, 1), desc.shape[-1])
        amask = torch.zeros(n_items, max(c_max, 1), dtype=torch.bool,
                            device=desc.device)
        starts = torch.cumsum(counts, 0) - counts
        for b in range(n_items):
            c = int(n_avail[b])
            if c:
                avail[b, :c] = desc[starts[b] : starts[b] + c]
                amask[b, :c] = True
        return targets, avail, amask

    def flow_objective(self, x1, cond, mask, is_residual, **fwd):
        """The regression this model is trained on. -> per_item [B], t [B], metrics.

        `x1` is [B, N, D] in target space (already divided by `target_scale`);
        `cond` is the mixture latent. `t=1` is noise, `t=0` is data,
        `x_t = t*x0 + (1-t)*x1`, net regresses `v = x0 - x1`.
        """
        b = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = self.sample_time_training(b).to(x1.device)
        x_t = t.view(b, 1, 1) * x0 + (1.0 - t.view(b, 1, 1)) * x1
        v = x0 - x1
        pred = self.forward(x_t, t, cond, mask, is_residual, **fwd)
        per_item = (pred.float() - v).pow(2).mean(dim=-1).mean(dim=1)
        return {"per_item": per_item, "t": t, "metrics": {}}

    def train_step(self, batch, debug: bool = False, *args, **kwargs):
        slots, counts = batch[0], batch[1]
        counts = counts.to(slots.device)
        level = batch[3].to(slots.device) if len(batch) > 3 else None
        n_units = batch[4].to(slots.device) if len(batch) > 4 else None
        b = counts.shape[0]

        item_idx = torch.repeat_interleave(
            torch.arange(b, device=slots.device), counts
        )
        mix = torch.zeros(
            b, *slots.shape[1:], device=slots.device, dtype=slots.dtype
        ).index_add_(0, item_idx, slots)

        x1, avail, amask = self._describe(slots, item_idx, b, counts)
        x1 = x1.float() / self.target_scale
        mixture_emb = self.mixture_encoder.encode(mix, self.loudness.sample_rate).float()

        # What is still unextracted, as conditioning: target + residual slot.
        res_kwargs = {}
        if self.use_residual_embedding:
            tgt_i, res_i = self._split(counts)
            remaining = slots[tgt_i] + slots[res_i]
            res_kwargs = self._residual_kwargs(remaining)

        mask = torch.ones(b, 2, dtype=torch.bool, device=slots.device)
        is_residual = torch.zeros(b, 2, dtype=torch.bool, device=slots.device)
        is_residual[:, 1] = True

        obj = self.flow_objective(
            x1, mixture_emb, mask, is_residual,
            level=level, avail=avail, avail_mask=amask,
            n_avail=(counts - 2).clamp(min=0),
            **self._passt_kwargs(mix), **res_kwargs)
        per_item, t = obj["per_item"], obj["t"]
        loss = per_item.mean()
        _, res_i = self._split(counts)
        has_res = slots[res_i].abs().amax(dim=(-2, -1)) > 0

        self.last_stats = {
            "t": t.detach(),
            "loss": per_item.detach(),
            "step_idx": (counts - 1).detach(),
            "has_residual": has_res.detach(),
            "level": level.detach() if level is not None else None,
            "n_units": None if n_units is None else n_units.detach(),
        }

        out = {
            "loss": loss,
            "n_available": (counts - 2).float().mean(),
            "frac_final": (~has_res).float().mean(),
            **obj["metrics"],
        }
        return out

    def forward(self, x_t, t, mixture_emb, mask, is_residual, cfg_scale: float = 1.0,
                mixture_passt=None, level=None, avail=None, avail_mask=None,
                n_avail=None, res_emb=None,
                force_uncond: bool = False, group_tags=None, **cond):
        out = self.backbone(
            self.backbone.chunk_embeddings(x_t), t, mixture_emb,
            mask=mask, is_residual=is_residual, cfg_scale=cfg_scale,
            group_tags=group_tags,
            mixture_emb=mixture_passt, level=level,
            avail=avail, avail_mask=avail_mask, n_avail=n_avail,
            res_emb=res_emb, force_uncond=force_uncond,
            **cond,
        )
        return self.backbone.unchunk_embeddings(out)

    @torch.no_grad()
    def step(self, mixture_emb, avail, amask, level, steps=32, cfg_scale=1.0,
             generator=None, mixture_passt=None, device=None, n_avail=None,
             res_emb=None, group_tags=None, churn=0.0):
        """One recursion step -> (source [B,D_pca], residual [B,D_pca], full).

        `res_emb` is the previous residual's embedding, used by a model built
        with `use_residual_embedding`.
        """
        b = mixture_emb.shape[0]
        device = mixture_emb.device if device is None else device
        mask = torch.ones(b, 2, dtype=torch.bool, device=device)
        is_residual = torch.zeros(b, 2, dtype=torch.bool, device=device)
        is_residual[:, 1] = True

        x = torch.randn(b, 2, self.backbone.embedding_dim, device=device,
                        generator=generator, dtype=torch.float32)

        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            extra = {}
            if res_emb is not None:
                extra["res_emb"] = res_emb

            t_cur, t_next = ts[i], ts[i + 1]
            g = float(churn[i]) if torch.is_tensor(churn) else float(churn)

            v = self.forward(x, t_cur.expand(b), mixture_emb, mask, is_residual,
                             cfg_scale=cfg_scale, mixture_passt=mixture_passt,
                             level=level, avail=avail, avail_mask=amask,
                             n_avail=n_avail, group_tags=group_tags, **extra)
            x = edm_churn_step(x, v.float(), t_cur, t_next, g,
                               generator=generator)

        x = x * self.target_scale
        return x[:, 0], x[:, 1], x

    @torch.no_grad()
    def extract(self, mixture, level=0, max_steps=12, steps=32,
                cfg_scale=1.0, generator=None, churn=0.0, fix_tags=True):
        """The full recursion: pull sources out until the residual is spent.

        Returns `(embeddings [B,n,D], stop_cos [B,n], n_used)`. `stop_cos[:, i]`
        is the cosine of step i's residual to the stop token -- the stopping
        criterion; see the paper.

        `fix_tags` (default True) draws the DiT's anonymous slot tags once for
        the whole run instead of redrawing them every forward, making the
        recursion a function of its seed.

        `churn` (gamma >= 0) makes the backward sampler stochastic, EDM-style;
        see `step` and `model.stochastic_sampling`.
        """
        b, device = mixture.shape[0], mixture.device
        cond = self.mixture_encoder.encode(mixture, self.loudness.sample_rate).float()
        passt = self.embed_mixture(mixture)
        lvl = (level if torch.is_tensor(level)
               else torch.full((b,), int(level), dtype=torch.long, device=device))
        # One tag draw for the whole run; see `fix_tags` above.
        run_tags = None
        if fix_tags and hasattr(self.backbone, "sample_group_tags"):
            run_tags = self.backbone.sample_group_tags(
                b, 2, device, next(self.backbone.parameters()).dtype)
        res_kw = {}
        if self.use_residual_embedding:
            res_kw["res_emb"] = self.query_adapter(
                self._chunked(self.query_model, mixture))
        # Where the residual trajectory starts: the whole mixture, if already
        # computed above; None otherwise.
        prev_res_0 = res_kw.get("res_emb")
        token = self.stop_token(device)

        avail = torch.zeros(b, 1, self.backbone.embedding_dim, device=device)
        amask = torch.zeros(b, 1, dtype=torch.bool, device=device)

        # The residual generated at each step, kept for the caller.
        embs, scos, resids = [], [], []
        for i in range(max_steps):
            src, res, full = self.step(
                cond, avail, amask, lvl, steps=steps, cfg_scale=cfg_scale,
                generator=generator, mixture_passt=passt, device=device,
                n_avail=amask.sum(1), group_tags=run_tags, churn=churn,
                **res_kw,
            )
            embs.append(src)
            resids.append(res)
            scos.append((torch.nn.functional.normalize(res, dim=-1) * token).sum(-1))

            if self.use_residual_embedding:
                res_kw["res_emb"] = res
            avail = torch.cat([avail if amask.any() else avail[:, :0],
                               full[:, :1]], dim=1)
            amask = torch.cat([amask if amask.any() else amask[:, :0],
                               torch.ones(b, 1, dtype=torch.bool, device=device)],
                              dim=1)
            last = (float(scos[-1].min()) > self.stop_cos
                    or i == max_steps - 1)
            if last:
                break

        self.last_residuals = torch.stack(resids, 1)
        self.last_mixture_emb = prev_res_0

        return torch.stack(embs, 1), torch.stack(scos, 1), len(embs)

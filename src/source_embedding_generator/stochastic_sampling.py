"""EDM-style STOCHASTIC SAMPLING for the flow-matching models in this repo.

Shared by SEG (`source_embedding_generator.recursive_fm`) and the latent
separator (`query_conditioned_dit.separator`), which sample the same interpolant --
`x_t = t*eps + (1-t)*x_data`, `t=1` noise, `t=0` data, Euler on a linspace grid
-- so they take the same stochastic step. It lives in `model` rather than in
either package because the two are deliberately independent: the separator
reaches SEG only as a callback, and an import the other way would tie them.

WHY STOCHASTIC AT ALL. The deterministic ODE is a bijection, so whatever is
wrong with the starting point is transported faithfully all the way to the
answer. Injecting noise breaks that: the trajectory is repeatedly re-drawn from
the model's own marginal, so an imperfect start is FORGOTTEN rather than
carried. That is what makes it the natural partner for a seeded start
(`extract(invert_init=)`), whose seeds sit measurably inside the sqrt(D) shell.

THE CONSTRUCTION follows Karras et al. 2022 as adapted to flow matching in
NVlabs/finite-difference-flow-optimization (`fdfo/sampling.py::edm_step`), and
is verified symbolically identical to it. Each step OVERSHOOTS the Euler step to
a lower noise level `r` than the schedule asks for, then adds exactly enough
fresh noise to land back on `t_next`.

THE SAME CONSTRUCTION DRIVES THE FLOW-MAP (iMF) SAMPLERS, which is why the two
halves of the step are also exposed separately as `edm_churn_overshoot` and
`edm_churn_renoise`. An exact jump has to be TOLD the interval it is crossing
(`h`), so it needs `r` before it steps rather than after; everything else --
what `r` is, how much noise comes back -- is shared, and shared as code rather
than as a second derivation.

TWO DETAILS THAT ARE EASY TO GET WRONG, both of which this module gets right and
a naive implementation does not:

GAMMA IS DEFINED ON THE NOISE LEVEL, not on t. In EDM `sigma(t) = t` so the two
coincide; for flow matching `sigma(t) = t/(1-t)` and they do not. Scaling t
directly makes one gamma mean a 1.13x noise bump at t=0.2 and a 1.83x bump at
t=0.8 -- exactly the non-uniformity across steps that the EDM formulation exists
to remove.

ORDER MATTERS AT THE BOUNDARY. This is ODE-then-noise, not the noise-then-ODE of
Karras' original. With noise first, step 0 would churn at t=1, where sigma is
infinite and the update degenerates -- so a prior-only schedule would be a
silent no-op. Overshooting first puts the injection after the first Euler step,
where it is well defined, and makes the last step's injection vanish on its own
(r = 0 there). The prior perturbation then gets its own explicit treatment,
`prior_churn_latent`, applied before the loop.
"""

from __future__ import annotations

import numpy as np
import torch


def edm_churn_overshoot(t_next, gamma):
    """The noise level a churned step OVERSHOOTS to before re-noising.

    `sigma(r) = sigma(t_next)/(1+g)`, inverting `t(sigma) = sigma/(sigma+1)`.
    Split out of `edm_churn_step` because the iMF samplers need `r` BEFORE they
    can take their step: an exact flow-map jump is asked for the interval it is
    crossing (`h = t_cur - r`), where an Euler step only needs the velocity at
    `t_cur` and can be told the interval afterwards. One formula, two callers,
    so the overshoot cannot drift between them.
    """
    g, t_next = float(gamma), float(t_next)
    return t_next if g <= 0.0 else t_next / (1.0 + g - g * t_next)


def edm_churn_renoise(x, r, t_next, gamma, *, generator=None):
    """Add the noise that carries a sample at level `r` back up to `t_next`.

    The second half of `edm_churn_step`, split out for the same reason as
    `edm_churn_overshoot`. At `t_next = 0` (or `r = 0`) the injection vanishes
    on its own, which is what makes the last step of a schedule land on the
    data manifold rather than beside it.
    """
    g = float(gamma)
    if g <= 0.0:
        return x
    n = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (x + r * (g * g + 2.0 * g) ** 0.5 * n) / (g * r + 1.0)


def edm_churn_step(x, v, t_cur, t_next, gamma, *, generator=None):
    """One Euler step, stochastic when `gamma` > 0. Shape-agnostic.

    `v` must already be in the dtype the caller wants to accumulate in.

    At gamma = 0 this is EXACTLY `x + (t_next - t_cur) * v` -- the overshoot
    collapses (`r = t_next`) and the noise term is skipped -- so a deterministic
    run is bit-identical to one that never heard of churn. That is a property
    worth keeping: it means enabling this feature cannot silently perturb any
    existing baseline.
    """
    g = float(gamma)
    t_cur, t_next = float(t_cur), float(t_next)
    if g <= 0.0:
        return x + (t_next - t_cur) * v
    r = edm_churn_overshoot(t_next, g)
    x = x + (r - t_cur) * v
    return edm_churn_renoise(x, r, t_next, g, generator=generator)


def prior_churn_latent(x, gamma, *, generator=None):
    """Perturb the STARTING latent. gamma = 0 returns `x` untouched.

    At the top of the schedule the sample IS pure noise, so "raise the noise
    level" degenerates; what survives is a variance-preserving blend with a
    fresh draw, `(x + sqrt((1+g)^2 - 1) * n) / (1 + g)`, whose two squared
    coefficients sum to exactly 1. Same object as SEG's `x_src_mix` in a
    different parameterisation: a = sqrt(g^2+2g)/(1+g), or g = 1/sqrt(1-a^2) - 1.
    """
    g = float(gamma)
    if g <= 0.0:
        return x
    n = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (x + (g * g + 2.0 * g) ** 0.5 * n) / (1.0 + g)

def churn_schedule(kind, ts, *, gamma=0.0025, mu_center=1.3, sigma_center=1.5,
                   sigma_int=0.25, w_int=3.0, mu_logw=None, sigma_logw=1.0,
                   generator=None, device=None):
    """Stochasticity schedule -> (prior_gamma, per_step_gamma [T]).

    The three families from the FDFO appendix. `ts` is the vector of step START
    times (`ts[:-1]` of the sampler grid), so the returned vector lines up with
    the loop in `RecursiveFMGenerator.step`.

    uniform   one fixed gamma at every step. The paper's own ablation makes this
              the best of the three and the one least sensitive to how gradients
              are weighted, so it is the default here too.
    interval  a random smooth window in t, per rollout: draw a centre in LOGIT
              space, evaluate the logit-normal density on the grid, normalise it
              to sum to 1, then `exp(w*g)-1`. The normalisation is what makes
              the strength independent of how the steps happen to be spaced
              inside the window; the exp/-1 equalises the total effect when it
              is spread over several steps rather than concentrated in one.
    prior     zero everywhere except a single perturbation of the starting
              latent, with a log-normal magnitude so it is always positive.
              Returned as `prior_gamma`, NOT as element 0 of the vector -- the
              in-loop formula degenerates at t=1 and the reference likewise
              handles the prior outside its loop.

    Defaults are the paper's: gamma=0.0025 (uniform), mu_center=1.3 (a logit, so
    t ~ 0.79), sigma_center=1.5, sigma_int=0.25, w_int=3, mu_logw=log(0.1),
    sigma_logw=1. Treat gamma=0.0025 with care -- it was tuned to make LIMITED
    perturbations for RL rollout diversity, which is not the job here.
    """
    ts = torch.as_tensor(ts, dtype=torch.float32, device=device)
    T = ts.numel()
    zeros = torch.zeros(T, dtype=torch.float32, device=device)
    rnd = lambda: torch.randn((), generator=generator, device=device).item()

    if kind == "uniform":
        return 0.0, torch.full_like(zeros, float(gamma))
    if kind == "prior":
        mu = np.log(0.1) if mu_logw is None else mu_logw
        w = float(np.exp(mu + sigma_logw * rnd()))
        return float(np.exp(w) - 1.0), zeros
    if kind == "interval":
        c = mu_center + sigma_center * rnd()
        # Clamped off both ends: logit(0) and logit(1) are infinite, and ts[0]
        # is exactly 1.0 on this grid.
        x = ts.clamp(1e-4, 1 - 1e-4)
        lg = torch.log(x / (1 - x))
        dens = torch.exp(-((lg - c) ** 2) / (2 * sigma_int ** 2)) / (
            (2 * np.pi) ** 0.5 * sigma_int * x * (1 - x))
        tot = float(dens.sum())
        if not np.isfinite(tot) or tot <= 0:
            # The window fell entirely off the grid; no stochasticity is the
            # honest answer, not a renormalised numerical artefact.
            return 0.0, zeros
        return 0.0, torch.exp(w_int * dens / tot) - 1.0
    raise ValueError(f"kind must be uniform|interval|prior, got {kind!r}")

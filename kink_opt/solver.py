"""Block-coordinate alternation driver: run(), multistart(), _alternate()."""

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .geometry import MARGIN
from .lp import lp_weights_f, lp_weights_g
from .objective import total_J, optimize_positions
from .verify import _ub


def _alternate(A, XI, B, ETA, t, alive_f, alive_g, outer=6, pos_iters=40,
               optimize_pos=True, verbose=True, patience=3):
    """Block-coordinate alternation (weight-LPs <-> position NLP) shared by
    run() and grow_topology().  Lifetime masks pin dead kinks to zero weight
    via the LP upper bounds and ride along the position sort.  Keeps the best
    feasible state seen and reverts on regression (the position step is a
    nonconvex NLP and is NOT guaranteed to improve J monotonically)."""
    hist = []
    best = (total_J(A, XI, B, ETA), A, XI, B, ETA, alive_f, alive_g)
    stall = 0
    for it in range(outer):
        A = lp_weights_f(XI, B, ETA, ub=_ub(alive_f))
        B = lp_weights_g(A, XI, ETA, ub=_ub(alive_g))
        Jw = total_J(A, XI, B, ETA)
        if optimize_pos:
            XI, ETA, alive_f, alive_g = optimize_positions(
                A, XI, B, ETA, maxiter=pos_iters,
                alive_f=alive_f, alive_g=alive_g)
            A = lp_weights_f(XI, B, ETA, ub=_ub(alive_f))   # restore feasib.
            B = lp_weights_g(A, XI, ETA, ub=_ub(alive_g))
        Jp = total_J(A, XI, B, ETA)
        hist.append((Jw, Jp))
        if verbose:
            print(f"  outer {it}:  J after weight-LPs = {Jw:.5f}   "
                  f"after position step = {Jp:.5f}")
        if Jp > best[0]:
            best = (Jp, A.copy(), XI.copy(), B.copy(), ETA.copy(),
                    alive_f.copy(), alive_g.copy())
            stall = 0
        else:
            stall += 1
            A, XI, B, ETA, alive_f, alive_g = (
                best[1].copy(), best[2].copy(), best[3].copy(),
                best[4].copy(), best[5].copy(), best[6].copy())
            if stall >= patience:
                break
        if it > 1 and abs(hist[-1][1] - hist[-2][1]) < 1e-6:
            break
    Jbest, A, XI, B, ETA, alive_f, alive_g = best
    return dict(A=A, XI=XI, B=B, ETA=ETA, t=t, J=Jbest, hist=hist,
                alive_f=alive_f, alive_g=alive_g)


def run(N=24, Kf=3, Kg=2, outer=6, seed="travel", pos_iters=40,
        optimize_pos=True, verbose=True, patience=3, rng_seed=0, t=None):
    # `t` overrides the uniform grid with an arbitrary (e.g. graded, Task C)
    # non-uniform node set; N is then inferred.  total_J / the weight-LPs /
    # monotonicity checks never read node SPACING (dt cancels in the harvest
    # sum), so a non-uniform grid is transparent to them -- only seeding and
    # certification (refine_time) consult t, and both handle non-uniform t.
    t = np.linspace(0.0, 1.0, N + 1) if t is None else np.asarray(t, float)
    N = len(t) - 1
    rng = np.random.default_rng(rng_seed)

    if seed == "static":
        # interior linspace => Kf=1 sits at x=0; harvest needs co-location
        XI = np.tile(np.linspace(-0.4, 0.4, Kf + 2)[1:-1], (N + 1, 1))
        ETA = np.tile(np.linspace(-0.4, 0.4, Kg + 2)[1:-1], (N + 1, 1))
    else:  # traveling seed: g-kinks sweep, f-kinks ride the same path
        path = -0.5 + 1.0 * t                       # -0.5 -> +0.5
        ETA = path[:, None] + np.linspace(-0.02, 0.02, Kg)[None, :]
        XI = path[:, None] + np.linspace(-0.10, 0.04, Kf)[None, :]
    XI = np.clip(XI + 0.01 * rng.standard_normal(XI.shape), -1 + MARGIN, 1 - MARGIN)
    ETA = np.clip(ETA + 0.01 * rng.standard_normal(ETA.shape), -1 + MARGIN, 1 - MARGIN)
    XI.sort(axis=1); ETA.sort(axis=1)

    B = np.outer(t, np.ones(Kg)) * 0.4 / Kg         # feasible ramp to bootstrap
    A = lp_weights_f(XI, B, ETA)

    alive_f = np.ones((N + 1, Kf), dtype=bool)       # every kink alive always
    alive_g = np.ones((N + 1, Kg), dtype=bool)
    return _alternate(A, XI, B, ETA, t, alive_f, alive_g, outer=outer,
                      pos_iters=pos_iters, optimize_pos=optimize_pos,
                      verbose=verbose, patience=patience)


def _multistart_worker(args):
    s, kwargs = args
    return s, run(rng_seed=s, verbose=False, **kwargs)


def multistart(seeds=range(6), workers=None, **kwargs):
    """optimize_positions is only a local search (nonconvex NLP block), so the
    outcome depends on the initial kink jitter. Run the full pipeline from
    several rng_seed jitters and keep the globally-best feasible result.
    kwargs are forwarded to run() (N, Kf, Kg, outer, seed, pos_iters, ...).
    Each seed's run() is fully independent, so seeds are farmed out across
    processes (ProcessPoolExecutor); pass workers=1 to force serial (e.g. for
    debugging). Results are still reduced in seed order so ties break the
    same way as the serial version."""
    seeds = list(seeds)
    if workers == 1:
        results = {s: run(rng_seed=s, verbose=False, **kwargs) for s in seeds}
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = dict(ex.map(_multistart_worker,
                                   [(s, kwargs) for s in seeds]))
    best = None
    for s in seeds:
        r = results[s]
        if best is None or r["J"] > best["J"]:
            best = r
            best["rng_seed"] = s
    return best

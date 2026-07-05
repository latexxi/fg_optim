"""Run 11 (plans/run11-constructive-hierarchy.md) -- the constructive
self-similar hierarchy. Positions are set ANALYTICALLY (an affine-contracted
copy of the travelling gen-0 carrier, per generation); the only numerical
solve is the convex weight LP (`lp_weights_f`/`lp_weights_g`, exact global
optimum) -- there is no position NLP anywhere in this file. This sidesteps
the entire budget-artifact failure class that made Runs 9-10 (`topology.py`'s
`generation_ladder`/`scale_sweep`) inconclusive: a local-search optimizer can
only ever FAIL to find gain, never certify it absent; a construction with no
optimizer in the loop can.
"""

import numpy as np

from .geometry import MARGIN, conv_eval
from .lp import lp_weights_f, lp_weights_g
from .objective import total_J
from .solver import _alternate
from .verify import certify, graded_grid, _ub
from .topology import _insert_column, fit_geometric

# Gen-0 carrier: run()'s seed="travel" path, offset exactly as run() does for
# Kf=Kg=1 (np.linspace(-0.10, 0.04, 1) == [-0.10], np.linspace(-0.02, 0.02, 1)
# == [-0.02]) -- Run 8's lesson is that a co-located static base makes every
# contracted copy redundant with its parent, so gen 0 must actually travel.
XI_OFFSET, ETA_OFFSET = -0.10, -0.02


def _travel_path(t):
    return -0.5 + 1.0 * t


def build_hierarchy(n_gen, scale_t=0.5, scale_x=0.5, t0=0.0, t1=1.0,
                    coarse_N=8, base_fine_sub=4, outer=20, path=None,
                    dead_gens=None):
    """Build generations 0..n_gen of the self-similar hierarchy as a single
    solution dict (schema: A, XI, B, ETA, t, alive_f, alive_g, J, hist -- no
    rng_seed, matching `run()`'s output minus the seed).

    Column k (0-indexed) is generation k: column 0 is the travelling gen-0
    carrier (one f-kink + one g-kink, full lifetime); column k>=1 is an
    affine-contracted copy of column 0, contracted about the carrier's
    travel-end position `p_end` (its value at t1):
        col_k = p_end + scale_x**k * (col_0 - p_end)          (spatial)
        window_k = (t1 - scale_t**k * (t1 - t0), t1)           (temporal)
    all anchored at the shared right endpoint t1, mirroring
    `topology.spawn_generation`'s per-generation contraction but STACKED
    (n_gen nested copies at once, jitter-free) rather than one warm-start
    insertion. Inserted via the same `_insert_column` Task B/D machinery, so
    each new column starts at effectively zero contribution to the coarse
    LP solve below (an all-alive-elsewhere but freshly-added column has no
    prior weight to inherit; the LP is solved fresh, not warm-started).

    Grid: `t` is a graded grid (Task C, `graded_grid`) built from the n_gen
    windows so the narrowest window is resolved -- `fine_sub` per window
    scaled as `base_fine_sub * w_1/w_k`, the same scaling
    `generation_ladder`/`scale_sweep` use to keep local resolution roughly
    flat as windows shrink (see plans/run10-scale-sweep-discriminators.md
    "resolution floor"). n_gen=0 (windows=[]) reproduces a plain uniform grid.

    Weights: seeded at zero and solved by REPEATED weight-LP alternation
    only (`_alternate(..., optimize_pos=False)`) -- each step is an exact LP
    global optimum for its block, so this is a monotonically non-decreasing,
    fully convex fixed-point iteration, not a search. No position NLP is
    invoked anywhere in this construction.

    `dead_gens` (optional set of generation indices in 1..n_gen) forces
    those generations' columns permanently dead (alive_f/alive_g all False)
    before the weight solve -- used by `check_insertion_neutral` to verify a
    freshly-inserted generation is J-neutral until the LP is allowed to use
    it."""
    path = path or _travel_path
    dead_gens = set(dead_gens or ())

    ws = [scale_t ** k * (t1 - t0) for k in range(1, n_gen + 1)]
    windows = [(t1 - w, t1) for w in ws]
    fine_subs = [base_fine_sub * ws[0] / w for w in ws] if ws else []
    t = (graded_grid(windows, coarse_N=coarse_N, fine_sub=fine_subs,
                     t0=t0, t1=t1) if windows
         else np.linspace(t0, t1, coarse_N + 1))

    p = path(t)
    col_f0 = np.clip(p + XI_OFFSET, -1 + MARGIN, 1 - MARGIN)
    col_g0 = np.clip(p + ETA_OFFSET, -1 + MARGIN, 1 - MARGIN)
    p_end_f, p_end_g = col_f0[-1], col_g0[-1]

    XI, ETA = col_f0[:, None], col_g0[:, None]
    alive_f = np.ones_like(XI, dtype=bool)
    alive_g = np.ones_like(ETA, dtype=bool)

    for k in range(1, n_gen + 1):
        tb, td = windows[k - 1]
        win = (t >= tb - 1e-12) & (t <= td + 1e-12)
        col_f = np.clip(p_end_f + scale_x ** k * (col_f0 - p_end_f),
                        -1 + MARGIN, 1 - MARGIN)
        col_g = np.clip(p_end_g + scale_x ** k * (col_g0 - p_end_g),
                        -1 + MARGIN, 1 - MARGIN)
        XI, ETA, alive_f, alive_g = _insert_column(
            "f", XI, ETA, alive_f, alive_g, col_f, win)
        XI, ETA, alive_f, alive_g = _insert_column(
            "g", XI, ETA, alive_f, alive_g, col_g, win)
        if k in dead_gens:
            alive_f[:, k] = False
            alive_g[:, k] = False

    B0 = np.zeros((t.size, ETA.shape[1]))
    A0 = lp_weights_f(XI, B0, ETA, ub=_ub(alive_f))
    B0 = lp_weights_g(A0, XI, ETA, ub=_ub(alive_g))
    return _alternate(A0, XI, B0, ETA, t, alive_f, alive_g, outer=outer,
                      optimize_pos=False, verbose=False, patience=outer)


def constructive_ladder(n_gen, scale_t=0.5, scale_x=0.5, sub=8, **kwargs):
    """Run 11 driver: for k = 0..n_gen, `build_hierarchy(k, ...)` then
    `certify()` -- deterministic, no seeds, no null arm, no paired statistic
    (none needed: unlike `generation_ladder`/`scale_sweep`, nothing here is a
    local search, so there is no search noise to correct for). Records
    dJk = Jc_k - Jc_{k-1} and the ratio dJk / dJ_{k-1} (the constructive
    analogue of `topology.decay_ratios`).

    Returns dict(generations=[dict(k, Jc, constraints_ok, dJk, ratio,
    sol), ...], scale_t, scale_x) -- generations[0]'s dJk/ratio are None (no
    predecessor)."""
    generations = []
    Jprev, dJprev = None, None
    for k in range(0, n_gen + 1):
        sol = build_hierarchy(k, scale_t=scale_t, scale_x=scale_x, **kwargs)
        c = certify(sol, sub=sub)
        Jc = c["Jc"]
        ok = bool(c["rep"]["ALL CONSTRAINTS OK"])
        dJk = None if Jprev is None else Jc - Jprev
        ratio = (dJk / dJprev if dJk is not None and dJprev not in (None, 0)
                 else None)
        generations.append(dict(k=k, Jc=Jc, constraints_ok=ok, dJk=dJk,
                                ratio=ratio, sol=sol))
        Jprev = Jc
        if dJk is not None:
            dJprev = dJk
    return dict(generations=generations, scale_t=scale_t, scale_x=scale_x)


def sweep_ratios(n_gen, scale_ts, scale_xs, sub=8, **kwargs):
    """Sweep `constructive_ladder` over a (scale_t, scale_x) grid -- the
    single most informative Run 11 output (plan's "sweep the self-similarity
    ratios"). If dJk's decay ratio tracks scale_x (or scale_t), that is the
    clean scale-discriminator signal `generation_ladder`'s window_ratio
    experiment could never get cleanly from the optimizer route (paired
    statistic collapsed to n=0-1 seeds at Run 10's seed count).

    Returns list of dict(scale_t, scale_x, dJk=[...], ratios=[...],
    ladder=<constructive_ladder result>)."""
    out = []
    for st in scale_ts:
        for sx in scale_xs:
            lad = constructive_ladder(n_gen, scale_t=st, scale_x=sx, sub=sub,
                                      **kwargs)
            dJk = [g["dJk"] for g in lad["generations"] if g["dJk"] is not None]
            ratios = [g["ratio"] for g in lad["generations"]
                     if g["ratio"] is not None]
            out.append(dict(scale_t=st, scale_x=sx, dJk=dJk, ratios=ratios,
                            ladder=lad))
    return out


def check_insertion_neutral(n_gen, scale_t=0.5, scale_x=0.5, sub=8,
                            tol_rel=0.01, **kwargs):
    """Validation #1 (plan "Validation before trusting any number"):
    inserting a generation at zero-forced weight must not raise Jc above the
    previous generation's -- the same J-neutral-insertion invariant
    `add_kink`/`spawn_generation` rely on elsewhere. Builds generation n_gen
    with its own columns forced dead (`dead_gens={n_gen}`) and compares
    against `certify(build_hierarchy(n_gen - 1))`.

    A `tol_rel` (not exact equality) is used because the two builds use
    different graded grids (n_gen's grid additionally resolves the newest,
    narrowest window) -- same reasoning as `_regrid_onto_windows`'s 1% gate:
    a grid change alone can move Jc slightly even with no real J change.

    Returns dict(ok, Jc_with_dead, Jc_without, diff, diff_rel). Requires
    n_gen >= 1."""
    if n_gen < 1:
        raise ValueError("check_insertion_neutral needs n_gen >= 1")
    with_dead = build_hierarchy(n_gen, scale_t=scale_t, scale_x=scale_x,
                                dead_gens={n_gen}, **kwargs)
    without = build_hierarchy(n_gen - 1, scale_t=scale_t, scale_x=scale_x,
                              **kwargs)
    Jc_with = certify(with_dead, sub=sub)["Jc"]
    Jc_without = certify(without, sub=sub)["Jc"]
    diff = Jc_with - Jc_without
    diff_rel = abs(diff) / max(abs(Jc_without), 1e-12)
    return dict(ok=diff_rel < tol_rel, Jc_with_dead=Jc_with,
               Jc_without=Jc_without, diff=diff, diff_rel=diff_rel)


def grid_convergence_check(sol, sub_lo=8, sub_hi=16, tol_rel=0.01):
    """Validation #2: re-certify at double `certify`'s own sub-refinement and
    require Jc to move less than `tol_rel` -- if it moves more, the graded
    grid built into `sol["t"]` under-resolves the narrowest generation's
    window (mirrors `generation_ladder`'s regrid-gap guard).

    Returns dict(ok, Jc_lo, Jc_hi, diff_rel)."""
    Jc_lo = certify(sol, sub=sub_lo)["Jc"]
    Jc_hi = certify(sol, sub=sub_hi)["Jc"]
    diff_rel = abs(Jc_hi - Jc_lo) / max(abs(Jc_lo), 1e-12)
    return dict(ok=diff_rel < tol_rel, Jc_lo=Jc_lo, Jc_hi=Jc_hi,
               diff_rel=diff_rel)


def travel_sanity(sol):
    """Validation #3: confirm the gen-0 carrier actually travels (p_end !=
    p[0]) -- else every contracted copy is co-located with its parent and the
    whole ladder reproduces the Run 8 null for a trivial reason.

    Returns dict(ok, p_start, p_end) for the f-kink (column 0)."""
    p_start, p_end = float(sol["XI"][0, 0]), float(sol["XI"][-1, 0])
    return dict(ok=abs(p_end - p_start) > 1e-6, p_start=p_start, p_end=p_end)


def saturation_diagnostics(sol, k):
    """Saturation-mechanism instrumentation (plan "boundedness-proof
    pointer"), read cheaply off the already-certified/coarse fields for
    generation k's columns (index k, since gen 0 is column 0):

    - rise_budget_used = -f(x_birth, 0), the per-x Lipschitz rise budget
      `int_0^1 f_t(x,t) dt` consumed at the f-kink's BIRTH position (its
      first alive node -- not its resting position at t1, which by
      construction is the same anchor `p_end` for every generation and so
      would trivially read identically every time) -- if this saturates
      near the Lipschitz cap as k grows, later generations have no rise
      left to harvest there.
    - g_mass = sum_t b_m(t) over the g-kink's alive window, the total
      curvature (g_xx mass) the generation contributes.

    Only meaningful to call once dJk is observed to decay (plan: "record
    WHY, so the result points at a proof rather than just asserting a
    number"). Returns dict(k, x_birth, rise_budget_used, g_mass)."""
    XI, A = sol["XI"], sol["A"]
    ETA, B, alive_g = sol["ETA"], sol["B"], sol["alive_g"]
    birth = np.flatnonzero(sol["alive_f"][:, k])[0]
    x_birth = float(XI[birth, k])
    f_x0 = float(conv_eval(np.array([x_birth]), A[0], XI[0])[0])
    win = alive_g[:, k]
    g_mass = float(B[win, k].sum())
    return dict(k=k, x_birth=x_birth, rise_budget_used=-f_x0, g_mass=g_mass)


__all__ = [
    "build_hierarchy", "constructive_ladder", "sweep_ratios",
    "check_insertion_neutral", "grid_convergence_check", "travel_sanity",
    "saturation_diagnostics", "fit_geometric",
]

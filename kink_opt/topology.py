"""Topology moves (Task B): birth/prune kinks between alternations, the
renormalization warm start (Task D), and the Run 9 generation-gain ladder."""

import numpy as np

from .geometry import MARGIN
from .lp import lp_weights_f, lp_weights_g
from .objective import total_J
from .solver import _alternate
from .verify import _ub, certify, _interp_to_grid, graded_grid


def add_kink(family, XI, ETA, alive_f, alive_g, parent, t,
             t_birth, t_death, dx=0.05, rng=None):
    """Insert a new kink trajectory as a small perturbation of an existing
    `parent` column of the chosen family ("f" or "g"), alive only on
    [t_birth, t_death].  The new column carries NO weight yet (the caller
    re-solves the weight LPs), so J and feasibility are unchanged at insertion.
    Returns (XI, ETA, alive_f, alive_g) with one family's matrices grown by a
    column.  A dead node (weight pinned to 0 outside the window) contributes
    nothing to f/g, so its position is a free, harmless extra checkpoint until
    the window opens."""
    jit = dx * (rng.standard_normal() if rng is not None else 1.0)
    win = (t >= t_birth - 1e-12) & (t <= t_death + 1e-12)
    src = (XI[:, parent] if family == "f" else ETA[:, parent]) + jit
    return _insert_column(family, XI, ETA, alive_f, alive_g, src, win)


def _insert_column(family, XI, ETA, alive_f, alive_g, col, win):
    """Append one kink trajectory `col` (Np1,) to `family` with lifetime mask
    `win` (Np1, bool); positions clipped into bounds.  Shared by add_kink
    (perturbed copy, Task B) and spawn_generation (contracted copy, Task D)."""
    col = np.clip(np.asarray(col, float), -1 + MARGIN, 1 - MARGIN)
    if family == "f":
        return (np.column_stack([XI, col]), ETA,
                np.column_stack([alive_f, win]), alive_g)
    if family == "g":
        return (XI, np.column_stack([ETA, col]),
                alive_f, np.column_stack([alive_g, win]))
    raise ValueError("family must be 'f' or 'g'")


def _seed_grown(base, XI2, ETA2, af2, ag2):
    """Bootstrap feasible weights for a family grown by one column (via
    add_kink / _insert_column / spawn_generation), from a converged solution
    `base` whose XI/ETA the grown XI2/ETA2 extend.  The new column carries
    ZERO weight at insertion (by construction of the callers above), so this
    is J-neutral -- it only restores exact LP feasibility on the grown shape,
    it does not change the objective.  `base["B"]` is padded with a zero
    column for any new g-kink so the first LP sees matching (Kg+1) shapes
    (no such padding is needed for A: it is solved fresh by the LP, not
    warm-started, since HiGHS finds the exact optimum in one call)."""
    B0 = np.zeros((base["t"].size, ETA2.shape[1]))
    B0[:, :base["B"].shape[1]] = base["B"]
    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
    B0 = lp_weights_g(A0, XI2, ETA2, ub=_ub(ag2))
    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
    return A0, B0


def _lifetime_window(mask_col, t):
    """[t_birth, t_death] of a boolean lifetime column: the times of its first
    and last alive node.  All-dead (shouldn't happen) falls back to [t0, t1]."""
    idx = np.nonzero(mask_col)[0]
    if idx.size == 0:
        return t[0], t[-1]
    return t[idx[0]], t[idx[-1]]


def _kink_diagnostics(r, family, col_idx, parent_idx, tol=1e-8):
    """Post-optimization measurements for one kink column, used by the Run 9
    generation ladder to test STRATEGY.md's self-similarity hypotheses
    directly rather than by eyeballing printouts.  `family` is "f" or "g";
    `col_idx` the new kink's column, `parent_idx` the kink it was seeded from
    (perturbed copy, add_kink) or contracted from (spawn_generation).

    Returns dict with:
      lifetime  -- (t_on, t_off): first/last time node where |weight| > tol.
                   This is the EFFECTIVE active window, which may be narrower
                   than any imposed lifetime mask -- the self-similarity test
                   is whether the optimizer contracts it further on its own.
      extent    -- (min_x, max_x) of the position trajectory over that window.
      jump_mean -- mean over the active window of 2*w/(1-x**2), the "jump"
                   formula from the module docstring (same formula for f and
                   g: both are hat-sum representations with identical
                   curvature/jump algebra, only g's jump enters the harvest
                   sum as g_xx).
      offset_from_parent -- mean |x_new - x_parent| over the active window
                   (tests "riding on the parent path").
    A column that is entirely zero-weight (fully pruned away, i.e. the LP
    gave the new kink nothing at every node) falls back to reporting over its
    imposed lifetime mask instead of an empty active window, so the numbers
    stay meaningful ("it used none of its allotted window") rather than NaN.
    """
    W = r["A"] if family == "f" else r["B"]
    P = r["XI"] if family == "f" else r["ETA"]
    mask = r.get("alive_f") if family == "f" else r.get("alive_g")
    t = r["t"]
    w, p, p_parent = W[:, col_idx], P[:, col_idx], P[:, parent_idx]
    active = np.abs(w) > tol
    if not active.any():
        active = mask[:, col_idx] if mask is not None else np.ones_like(w, dtype=bool)
    lifetime = _lifetime_window(active, t)
    extent = (float(p[active].min()), float(p[active].max())) if active.any() else (t[0], t[-1])
    jump = 2.0 * np.abs(w) / (1.0 - p**2)
    jump_mean = float(jump[active].mean()) if active.any() else 0.0
    offset_mean = float(np.abs(p - p_parent)[active].mean()) if active.any() else 0.0
    return dict(lifetime=lifetime, extent=extent, jump_mean=jump_mean,
                offset_from_parent=offset_mean)


def spawn_generation(sol, scale_t=0.5, scale_x=0.5, families=("f", "g"),
                     rng=None):
    """Task D -- the renormalization warm start.  If the hierarchy is
    self-similar, generation k+1 is an affinely-rescaled copy of generation k:
    shorter lifetime, narrower spatial extent, riding on top of generation k's
    path.  For each requested family, take the current finest carrier (its most
    active kink), contract that trajectory SPATIALLY by `scale_x` about the end
    of its travel path (p_end), and TEMPORALLY to a window of length
    `scale_t` x its lifetime placed at that end.  The contracted copy is
    inserted at ZERO weight via the Task B `_insert_column` machinery -- so J
    and feasibility are unchanged at insertion -- and the caller re-optimizes.
    Unlike a randomly-perturbed insertion (add_kink), this seeds the new kink
    already riding the parent path at a fraction of its extent, which is a much
    better basin for the position NLP (see Run 8).
    Returns (XI, ETA, alive_f, alive_g) with one new column per family."""
    XI, ETA = sol["XI"].copy(), sol["ETA"].copy()
    af, ag = sol["alive_f"].copy(), sol["alive_g"].copy()
    t = sol["t"]
    for family in families:
        W = sol["A"] if family == "f" else sol["B"]
        cols = sol["XI"] if family == "f" else sol["ETA"]
        mask = sol["alive_f"] if family == "f" else sol["alive_g"]
        parent = int(np.abs(W).max(axis=0).argmax())    # finest = most active
        p = cols[:, parent]
        alive_idx = np.nonzero(mask[:, parent])[0]
        tb, td = _lifetime_window(mask[:, parent], t)
        span = scale_t * (td - tb)
        wb, wd = td - span, td                           # window at travel end
        win = (t >= wb - 1e-12) & (t <= wd + 1e-12)
        p_end = p[alive_idx[-1]] if alive_idx.size else p[-1]
        col = p_end + scale_x * (p - p_end)              # contract about p_end
        if rng is not None:
            col = col + 0.01 * scale_x * rng.standard_normal(col.shape)
        XI, ETA, af, ag = _insert_column(family, XI, ETA, af, ag, col, win)
    return XI, ETA, af, ag


def prune(r, tol=1e-8):
    """Drop kinks whose |weight| stays below `tol` at every time node (dead
    trajectories the LP never used).  Keeps at least one kink per family."""
    A, XI, B, ETA = r["A"], r["XI"], r["B"], r["ETA"]
    af, ag = r["alive_f"], r["alive_g"]
    kf = np.abs(A).max(axis=0) > tol
    kg = np.abs(B).max(axis=0) > tol
    if not kf.any():
        kf[np.abs(A).max(axis=0).argmax()] = True
    if not kg.any():
        kg[np.abs(B).max(axis=0).argmax()] = True
    out = dict(r)
    out["A"], out["XI"], out["alive_f"] = A[:, kf], XI[:, kf], af[:, kf]
    out["B"], out["ETA"], out["alive_g"] = B[:, kg], ETA[:, kg], ag[:, kg]
    return out


def generation_step(cur, window, seeds=range(3), dx=0.05, outer=12,
                    pos_iters=40, patience=None, sub=8, force_dead=False):
    """One rung of the Run 9 generation-gain ladder (STRATEGY.md Section 5):
    insert ONE new f-kink and ONE new g-kink, both alive only on `window` =
    (t_birth, t_death), as perturbed copies of the current most-active kink
    of each family (`add_kink`, Task B machinery -- zero-weight at insertion,
    so J is unchanged before re-optimization). Tried over several `seeds`
    (multistart on the insertion jitter, since `_alternate`'s position block
    is only a local search); the best FEASIBLE candidate by J_certified is
    kept (falls back to the best candidate overall, flagged `feasible=False`,
    if none certify -- an optimizer failure must be visible, not silently
    dropped).

    Passing `window = (t[0], t[-1])` (full lifetime) turns this into the
    "guard arm" (a free, unrestricted insertion) -- same code path, since
    windowed-vs-free is just this one argument.

    `force_dead=True` turns this into the Run 10 NULL-CONTROL arm: both new
    columns' alive masks are forced all-False right after insertion, so the
    LP can NEVER give them weight, no matter what `window`/seed says. This
    isolates a real contamination discovered while running Run 10's
    scale_sweep: even a new kink that ends up with jump_mean=0 (dead at
    convergence) can still leave `Jc` measurably higher than the pre-insertion
    base, because the insertion jitter changes the L-BFGS-B starting vector
    (and interacts with the GAP spacing penalty against same-family kinks
    while alive during early outer iterations), which perturbs the OLD
    kinks onto a different -- sometimes better -- local optimum via the
    multi-seed search, independent of any real value the new kink provides.
    `force_dead=True` reruns the identical seeded search with that channel
    open but the new-kink-value channel closed, so `Jc_dead - Jc_before`
    measures pure search-noise; subtracting it from a real run's `dJ` is the
    control-corrected signal. See plans/run10-scale-sweep-discriminators.md.

    Column identity: the new kink is appended as the LAST column of its
    family (`cur["XI"].shape[1]` / `cur["ETA"].shape[1]` before insertion).
    This index is tracked into the post-alternation candidate BEFORE pruning
    (pruning can shift indices, and if the new kink itself dies, the
    diagnostics still need its column present with all-zero weight -- see
    `_kink_diagnostics`'s fallback). This relies on the same no-crossing
    column-identity assumption `optimize_positions`'s mask-permutation already
    depends on elsewhere in this file.

    Returns dict(sol, Jc, feasible, diagnostics=dict(f=..., g=...),
    spread=[(seed, Jc, feasible), ...]) -- `spread` is the per-seed honesty
    record (STRATEGY.md: "report the spread, not just the max")."""
    t = cur["t"]
    tb, td = window
    pf = int(np.abs(cur["A"]).max(axis=0).argmax())    # most active f-kink
    pg = int(np.abs(cur["B"]).max(axis=0).argmax())    # most active g-kink
    new_f_idx = cur["XI"].shape[1]                     # append position
    new_g_idx = cur["ETA"].shape[1]

    results = []
    for s in seeds:
        rng = np.random.default_rng(s)
        XI2, ETA2, af2, ag2 = add_kink(
            "f", cur["XI"], cur["ETA"], cur["alive_f"], cur["alive_g"],
            pf, t, tb, td, dx=dx, rng=rng)
        XI2, ETA2, af2, ag2 = add_kink(
            "g", XI2, ETA2, af2, ag2, pg, t, tb, td, dx=dx, rng=rng)
        if force_dead:
            af2[:, new_f_idx] = False
            ag2[:, new_g_idx] = False
        A0, B0 = _seed_grown(cur, XI2, ETA2, af2, ag2)
        cand = _alternate(A0, XI2, B0, ETA2, t, af2, ag2, outer=outer,
                          pos_iters=pos_iters, optimize_pos=True,
                          verbose=False, patience=patience or outer)
        diag_f = _kink_diagnostics(cand, "f", new_f_idx, pf)
        diag_g = _kink_diagnostics(cand, "g", new_g_idx, pg)
        pruned = prune(cand, tol=1e-8)
        c = certify(pruned, sub=sub)
        feasible = bool(c["rep"]["ALL CONSTRAINTS OK"])
        results.append(dict(seed=s, sol=pruned, Jc=c["Jc"], feasible=feasible,
                            diagnostics=dict(f=diag_f, g=diag_g)))

    feasible_results = [r for r in results if r["feasible"]]
    pool = feasible_results if feasible_results else results
    best = max(pool, key=lambda r: r["Jc"])
    return dict(sol=best["sol"], Jc=best["Jc"], feasible=best["feasible"],
                diagnostics=best["diagnostics"],
                spread=[(r["seed"], r["Jc"], r["feasible"]) for r in results])


def _regrid_onto_windows(cur, windows, fine_subs, coarse_N=8, sub=8,
                         tol_rel=0.01, verbose=True):
    """Migrate a converged solution `cur` onto a graded grid built around
    `windows` (Task C), and verify the migration is J-neutral. Shared by
    `generation_ladder` (one regrid up front for the whole ladder) and
    `scale_sweep` (one fresh regrid per sweep point, Run 10 Experiment 1).

    The regrid is a strict REFINEMENT of `cur`'s own nodes, never a
    coarsening: `graded_grid`'s own uniform background (`coarse_N`) can be
    coarser than `cur["t"]` outside the windows, which would silently lose
    resolution there (a real bug caught once in Run 9 -- see
    plans/run9-code-plan.md). Union with `cur["t"]` guarantees every
    original node survives.

    Weights are re-solved (exact LP repair) on the new grid to restore
    feasibility; positions/weights are otherwise linearly interpolated.
    Checked against `certify(cur)` before returning -- raises past
    `tol_rel` relative gap, since a silent large mismatch would corrupt
    every downstream dJ.

    Returns (cur_regridded, regrid_Jc)."""
    base_Jc = certify(cur, sub=sub)["Jc"]
    t0, t1 = cur["t"][0], cur["t"][-1]
    grid = graded_grid(windows, coarse_N=coarse_N, fine_sub=fine_subs,
                       t0=t0, t1=t1)
    t_new = np.union1d(grid, cur["t"])

    A2 = _interp_to_grid(cur["A"], cur["t"], t_new)
    XI2 = _interp_to_grid(cur["XI"], cur["t"], t_new)
    B2 = _interp_to_grid(cur["B"], cur["t"], t_new)
    ETA2 = _interp_to_grid(cur["ETA"], cur["t"], t_new)
    alive_f2 = np.ones((len(t_new), XI2.shape[1]), dtype=bool)
    alive_g2 = np.ones((len(t_new), ETA2.shape[1]), dtype=bool)
    A2 = lp_weights_f(XI2, B2, ETA2, ub=_ub(alive_f2))   # restore exact
    B2 = lp_weights_g(A2, XI2, ETA2, ub=_ub(alive_g2))   # feasibility
    out = dict(A=A2, XI=XI2, B=B2, ETA=ETA2, t=t_new, J=total_J(A2, XI2, B2, ETA2),
              hist=[], alive_f=alive_f2, alive_g=alive_g2)
    regrid_Jc = certify(out, sub=sub)["Jc"]
    if verbose:
        print(f"  base J_certified = {base_Jc:.5f}  ->  regridded onto "
              f"{len(t_new)} nodes, J_certified = {regrid_Jc:.5f}")
    if abs(regrid_Jc - base_Jc) > tol_rel * abs(base_Jc):
        raise RuntimeError(
            f"regrid is not within {tol_rel:.0%} of base: base {base_Jc:.5f} "
            f"vs regridded {regrid_Jc:.5f} -- fix the migration before "
            f"trusting any dJ")
    return out, regrid_Jc


def generation_ladder(base, n_gen=4, window0=0.5, window_ratio=0.5,
                      seeds=range(3), dx=0.05, base_fine_sub=4, coarse_N=8,
                      outer=12, pos_iters=40, sub=8, budget_fn=None,
                      verbose=True):
    """Run 9 driver -- the STRATEGY.md Section 5 generation-gain measurement.
    Starting from a converged `base` solution (Run 3's G0 in the driver
    below), births one windowed generation at a time via `generation_step`
    and tracks dJk = Jk - J_{k-1}. Roughly constant dJk over several
    generations supports the log-growth conjecture (sup J = +infinity);
    decaying dJk means J is bounded and the mesh's ln(Nx) growth was a
    discretization artifact.

    Window schedule: w_k = window0 * window_ratio**(k-1), anchored at the
    shared right endpoint (the travel-path end all generations ride toward),
    so window_k = (t1 - w_k, t1). This is precomputed for ALL n_gen
    generations up front (not regridded per generation) so a single graded
    time grid (Task C) serves the whole ladder: `graded_grid` gets a
    per-window `fine_sub` scaled as `base_fine_sub * window0/w_k`, which
    counteracts the shrinking window width so each generation's local node
    count stays roughly flat instead of collapsing below the "finest
    lifetime spans >= 8 fine steps" honesty floor (STRATEGY.md Section 5) --
    certify()'s own further sub-refinement (`sub`) multiplies on top of that.

    `base` is migrated onto the precomputed grid ONCE (linear interpolation
    of weights/positions via `_interp_to_grid`, all-alive masks since it
    carries no windows yet, then an exact weight-LP re-solve to restore
    feasibility -- the same repair `certify` already relies on). This
    migration should be near J-neutral (some drift is expected -- a different
    node count samples the harvest sum differently, same reason Run 7 uses a
    1% bar rather than exact equality); it is checked against `certify(base)`
    before generation 1 runs and raises past a 1% relative gap, since a
    silent large mismatch there would corrupt every downstream dJk.

    Each generation additionally runs a "guard arm" -- the same
    `generation_step` call with a full-lifetime window instead of the
    imposed one -- so a constant windowed dJk can be checked against free
    (non-hierarchical) insertion, per STRATEGY.md's non-negotiable guard.
    The ladder always advances on the WINDOWED arm; the guard is comparison
    only and is never adopted, so this can't quietly degrade into the
    unrestricted growth of Runs 4-5.

    Each generation ALSO runs a NULL-CONTROL arm (`force_dead=True`, the same
    Run 10 fix already applied to `scale_sweep` -- see that function's
    docstring for the full contamination mechanism: insertion jitter alone
    can move the OLD kinks to a better local optimum via the multi-seed
    search, independent of any value the new kink provides, so raw `dJk` is
    not trustworthy on its own). `corrected_dJk` is `_paired_dJ(windowed
    spread, null spread)` -- per-seed differencing, not a plain `Jk -
    null_Jk` best-of-max subtraction (same reasoning as `scale_sweep`).
    Before this arm existed, this docstring described `dJk` as the
    generation gain; that was only ever the UNCORRECTED number -- read
    `corrected_dJk`/`corrected_dJk_se` instead, and treat plain `dJk` as
    contaminated by search noise the same way `scale_sweep`'s raw `dJ` was.

    `budget_fn(k, n_live)`, if given, overrides the flat `outer`/`pos_iters`
    per generation (`n_live = cur["XI"].shape[1] + cur["ETA"].shape[1]`,
    the accumulated kink count going INTO generation k) -> `(outer_k,
    pos_iters_k)`. Default `None` reproduces the flat-budget behaviour
    exactly (Run 9's n_gen=5 escalation-needed case -- see
    plans/run9-generation-gain-ladder.md -- is what this exists for: the
    position-NLP dimensionality grows with accumulated kink count across the
    ladder, and a flat per-ladder budget silently starves later generations).

    Returns dict(generations=[dict(k, w_k, window, Jc, dJk, feasible,
    guard_Jc, guard_dJk, guard_feasible, null_Jc, null_dJk, null_feasible,
    corrected_dJk, corrected_dJk_se, corrected_dJk_n, diagnostics, spread,
    sol), ...], base_Jc)."""
    cur = prune(base, tol=1e-8)
    t0, t1 = cur["t"][0], cur["t"][-1]
    ws = [window0 * window_ratio**(k - 1) for k in range(1, n_gen + 1)]
    windows = [(t1 - w, t1) for w in ws]
    fine_subs = [base_fine_sub * window0 / w for w in ws]
    cur, regrid_Jc = _regrid_onto_windows(cur, windows, fine_subs,
                                          coarse_N=coarse_N, sub=sub,
                                          verbose=verbose)

    generations = []
    J = regrid_Jc
    for k in range(1, n_gen + 1):
        w_k, window_k = ws[k - 1], windows[k - 1]
        if budget_fn is not None:
            n_live = cur["XI"].shape[1] + cur["ETA"].shape[1]
            outer_k, pos_iters_k = budget_fn(k, n_live)
        else:
            outer_k, pos_iters_k = outer, pos_iters
        windowed = generation_step(cur, window_k, seeds=seeds, dx=dx,
                                   outer=outer_k, pos_iters=pos_iters_k, sub=sub)
        guard = generation_step(cur, (t0, t1), seeds=seeds, dx=dx,
                                outer=outer_k, pos_iters=pos_iters_k, sub=sub)
        null = generation_step(cur, window_k, seeds=seeds, dx=dx,
                               outer=outer_k, pos_iters=pos_iters_k, sub=sub,
                               force_dead=True)
        dJk = windowed["Jc"] - J
        guard_dJk = guard["Jc"] - J
        null_dJk = null["Jc"] - J
        paired = _paired_dJ(windowed["spread"], null["spread"])
        corrected_dJk = paired["mean"] if paired is not None else float("nan")
        corrected_dJk_se = paired["se"] if paired is not None else float("nan")
        corrected_dJk_n = paired["n"] if paired is not None else 0
        if verbose:
            print(f"  gen {k}: w_k={w_k:.4f}  Jc {J:.5f} -> "
                  f"{windowed['Jc']:.5f}  dJk={dJk:+.5f}  "
                  f"(feasible={windowed['feasible']})   guard: "
                  f"{guard['Jc']:.5f}  guard_dJk={guard_dJk:+.5f}  "
                  f"(feasible={guard['feasible']})   null_dJk={null_dJk:+.5f}  "
                  f"(feasible={null['feasible']})   corrected_dJk="
                  f"{corrected_dJk:+.5f} +/- {corrected_dJk_se:.5f} "
                  f"(n={corrected_dJk_n})")
        generations.append(dict(
            k=k, w_k=w_k, window=window_k, Jc=windowed["Jc"], dJk=dJk,
            feasible=windowed["feasible"], guard_Jc=guard["Jc"],
            guard_dJk=guard_dJk, guard_feasible=guard["feasible"],
            null_Jc=null["Jc"], null_dJk=null_dJk, null_feasible=null["feasible"],
            corrected_dJk=corrected_dJk, corrected_dJk_se=corrected_dJk_se,
            corrected_dJk_n=corrected_dJk_n,
            diagnostics=windowed["diagnostics"], spread=windowed["spread"],
            sol=windowed["sol"]))
        cur, J = windowed["sol"], windowed["Jc"]
    return dict(generations=generations, base_Jc=regrid_Jc)


def _paired_dJ(spread_a, spread_b):
    """Paired per-seed difference of two `generation_step` `spread` lists
    (list of (seed, Jc, feasible)), matched by seed value, restricted to
    seeds feasible in BOTH arms. This is the statistically correct way to
    compare two noisy arms sharing the same seed set -- much of the
    search-path randomness up to the point the two arms diverge (e.g. the
    windowed vs null-control alive mask) is common to both, so it cancels in
    the per-seed subtraction, unlike subtracting two independently-computed
    best-of-max values (each a high-variance order statistic; see Run 10's
    "corrected_dJ came back inconclusive" finding in
    plans/run10-scale-sweep-discriminators.md).

    Returns None if no seed is feasible in both arms, else
    dict(mean, se, n, per_seed=[(seed, diff), ...]) where `diff = a - b` for
    each common feasible seed, `se` is the standard error of the mean
    (NaN if n==1)."""
    a = {s: jc for s, jc, ok in spread_a if ok}
    b = {s: jc for s, jc, ok in spread_b if ok}
    common = sorted(set(a) & set(b))
    if not common:
        return None
    diffs = np.array([a[s] - b[s] for s in common])
    se = float(diffs.std(ddof=1) / np.sqrt(diffs.size)) if diffs.size > 1 else float("nan")
    return dict(mean=float(diffs.mean()), se=se, n=int(diffs.size),
               per_seed=list(zip(common, diffs.tolist())))


def scale_sweep(base, ws, seeds=range(5), dx=0.05, base_fine_sub=4,
                coarse_N=8, outer=40, pos_iters=100, sub=8, verbose=True):
    """Run 10 Experiment 1 -- single-generation gain vs window scale, decoupled
    from the ladder's accumulation confounds (kink count, shared grid size --
    see plans/run9-generation-gain-ladder.md's n_gen=5 budget-escalation
    trail and plans/run10-scale-sweep-discriminators.md). For each `w` in
    `ws`, independently: regrid `base` fresh around window = (t1-w, t1)
    (`_regrid_onto_windows`, one window only, so problem size stays in the
    same class at every point -- one verified budget covers the whole
    sweep), then run ONE `generation_step` (windowed), ONE guard-arm
    `generation_step` (full lifetime), and ONE NULL-CONTROL arm
    (`force_dead=True` -- see `generation_step`'s docstring) from that
    regridded base. No state carries between points -- each `w` measures the
    gain of adding a single generation directly to `base`, not to the
    previous point's result.

    `fine_sub` is scaled per point as `base_fine_sub * max(ws) / w` (same
    formula `generation_ladder` uses, anchored here at the sweep's own widest
    window rather than a separate `window0`), NOT held at a flat
    `base_fine_sub` for every `w`. An earlier version used a flat value,
    which silently hit `graded_grid`'s "at least 2 local sub-intervals" floor
    for every `w <= 0.0625` in the default sweep (3 nodes spanning the whole
    window) -- a real, independent under-resolution confound on top of the
    search-noise one below, caught only by checking `graded_grid`'s node-count
    formula directly against the sweep's `ws`, not by any run-time symptom.
    Points at or below that floor cannot show real kink development
    regardless of the true window-scale answer; scaling `fine_sub` keeps
    local resolution (subintervals per window) roughly constant across `ws`,
    same as the ladder already does.

    The null-control arm exists because the windowed/guard `dJ` alone is
    contaminated: a new kink that ends up with `jump_mean=0` (dead at
    convergence) can still leave `Jc` measurably higher than the regridded
    base, purely because the insertion jitter perturbs the position NLP's
    starting point / GAP-penalty landscape enough to move the OLD kinks to a
    different local optimum via the multi-seed search -- independent of any
    real value the new kink provides.

    The comparison uses `_paired_dJ` (per-seed differencing, matched by seed
    value between the windowed and null-control `spread`s), NOT
    `windowed["Jc"] - null["Jc"]` (subtracting two independent best-of-max
    values). A first version of this function did the latter and produced
    numbers with no discernible trend and 3/6 points negative -- diagnosed
    as best-of-max being a high-variance order statistic unsuited to
    comparing two noisy arms sharing the same seed set; the shared search
    randomness up to the point the two arms' alive masks diverge mostly
    cancels in a per-seed subtraction, which best-of-max, throws away. See
    plans/run10-scale-sweep-discriminators.md's "Corrected Experiment 1
    results ... INCONCLUSIVE" section for the full diagnosis.

    `corrected_dJ_mean(w)` (and its standard error `corrected_dJ_se`) roughly
    constant, and clearly nonzero relative to its SE, as w -> 0 supports the
    log-growth conjecture; trending to (and consistent with) 0 supports J
    bounded. Report the SE alongside the mean -- a mean near zero with a
    large SE is "inconclusive," not "no gain."

    Returns dict(points=[dict(w, n_nodes, dJ, feasible, guard_dJ,
    guard_feasible, null_dJ, null_feasible, corrected_dJ_mean,
    corrected_dJ_se, corrected_dJ_n, diagnostics, spread, sol), ...],
    base_Jc)."""
    base = prune(base, tol=1e-8)
    base_Jc = certify(base, sub=sub)["Jc"]
    t0, t1 = base["t"][0], base["t"][-1]
    w_ref = max(ws)

    points = []
    for w in ws:
        window = (t1 - w, t1)
        fine_sub_w = base_fine_sub * w_ref / w
        cur, regrid_Jc = _regrid_onto_windows(
            base, [window], [fine_sub_w], coarse_N=coarse_N, sub=sub,
            verbose=False)
        windowed = generation_step(cur, window, seeds=seeds, dx=dx,
                                   outer=outer, pos_iters=pos_iters, sub=sub)
        guard = generation_step(cur, (t0, t1), seeds=seeds, dx=dx,
                                outer=outer, pos_iters=pos_iters, sub=sub)
        null = generation_step(cur, window, seeds=seeds, dx=dx,
                               outer=outer, pos_iters=pos_iters, sub=sub,
                               force_dead=True)
        dJ = windowed["Jc"] - regrid_Jc
        guard_dJ = guard["Jc"] - regrid_Jc
        null_dJ = null["Jc"] - regrid_Jc
        paired = _paired_dJ(windowed["spread"], null["spread"])
        corrected_mean = paired["mean"] if paired is not None else float("nan")
        corrected_se = paired["se"] if paired is not None else float("nan")
        corrected_n = paired["n"] if paired is not None else 0
        if verbose:
            print(f"  w={w:.5f}  ({len(cur['t'])} nodes)  regrid_Jc="
                  f"{regrid_Jc:.5f}  dJ={dJ:+.5f} (feasible={windowed['feasible']})"
                  f"   guard_dJ={guard_dJ:+.5f} (feasible={guard['feasible']})"
                  f"   null_dJ={null_dJ:+.5f} (feasible={null['feasible']})"
                  f"   corrected_dJ={corrected_mean:+.5f} +/- {corrected_se:.5f} "
                  f"(n={corrected_n})")
        points.append(dict(
            w=w, n_nodes=len(cur["t"]), dJ=dJ, feasible=windowed["feasible"],
            guard_dJ=guard_dJ, guard_feasible=guard["feasible"],
            null_dJ=null_dJ, null_feasible=null["feasible"],
            corrected_dJ_mean=corrected_mean, corrected_dJ_se=corrected_se,
            corrected_dJ_n=corrected_n,
            diagnostics=windowed["diagnostics"], spread=windowed["spread"],
            sol=windowed["sol"]))
    return dict(points=points, base_Jc=base_Jc)


def _budget_stable(step_fn, *args, outer, pos_iters, factor=2.0,
                   tol_rel=0.10, tol_abs=0.005, **kwargs):
    """Run 10 Experiment 3 -- budget-adequacy gate. Calls `step_fn(*args,
    outer=outer, pos_iters=pos_iters, **kwargs)`, then again at
    `outer*factor`/`pos_iters*factor`; accepts the point only if Jc barely
    moves between the two, i.e. the lower budget was already adequate. This
    is the automated version of the by-hand check that caught three separate
    under-convergence artifacts in Run 9 (see
    plans/run9-generation-gain-ladder.md) -- a point that fails this should
    not be read as "gain is truly absent", only as "optimizer didn't
    converge here".

    `step_fn` is `generation_step` (or anything with the same `outer=`/
    `pos_iters=` kwargs and a `Jc` key in its return dict).

    Returns dict(accepted, delta, result) -- `result` is always the
    HIGHER-budget run (so a rejected point can still be inspected/reused),
    `delta = Jc_hi - Jc_lo`."""
    lo = step_fn(*args, outer=outer, pos_iters=pos_iters, **kwargs)
    hi = step_fn(*args, outer=int(round(outer * factor)),
                pos_iters=int(round(pos_iters * factor)), **kwargs)
    delta = hi["Jc"] - lo["Jc"]
    accepted = abs(delta) < max(tol_abs, tol_rel * abs(lo["Jc"]))
    return dict(accepted=accepted, delta=delta, result=hi)


def _gate_report(point):
    """Cheap (no rerun) budget-adequacy signal for one `generation_step` /
    `scale_sweep` point result, from fields it already returns: majority of
    seeds feasible, and both new kinks show real activity (`jump_mean > 0`,
    not starved to zero weight -- the specific failure signature Run 9 hit
    at n_gen=5). Does not replace `_budget_stable`, only screens which points
    are worth the expensive doubled rerun.

    `point` needs `spread` (list of (seed, Jc, feasible)) and `diagnostics`
    (dict(f=..., g=...) from `_kink_diagnostics`) -- both present on
    `generation_step`'s return dict and on each `scale_sweep`/
    `generation_ladder` point/generation.

    Returns dict(feasible_ok, jump_ok, ready)."""
    feasible_frac = np.mean([ok for _, _, ok in point["spread"]])
    feasible_ok = feasible_frac >= 0.5
    jump_ok = (point["diagnostics"]["f"]["jump_mean"] > 0 and
              point["diagnostics"]["g"]["jump_mean"] > 0)
    return dict(feasible_ok=bool(feasible_ok), jump_ok=bool(jump_ok),
               ready=bool(feasible_ok and jump_ok))


def decay_ratios(generations):
    """dJk[i+1] / dJk[i] for consecutive generations of a `generation_ladder`
    result -- Run 10 Experiment 2 (window_ratio discriminator). Under the
    bounded-J hypothesis (dJ roughly proportional to window scale w), this
    sequence should cluster near the ladder's own `window_ratio`; under the
    log-growth hypothesis it should not track `window_ratio` at all."""
    dJks = [g["dJk"] for g in generations]
    return [dJks[i + 1] / dJks[i] for i in range(len(dJks) - 1)]


def corrected_decay_ratios(generations):
    """Same as `decay_ratios` but on `corrected_dJk` (the null-control-
    subtracted, paired-per-seed gain -- see `generation_ladder`'s
    docstring), not raw `dJk`. This is the number Experiment 2 should
    actually be read from: raw `dJk` is exposed to the same search-noise
    contamination `scale_sweep` found and fixed in Experiment 1 (a
    zero-value insertion can still move `Jc` via the multi-seed search
    finding a better basin for the OLD kinks), so a raw decay-ratio
    tracking `window_ratio` is not by itself evidence of real scale-
    dependent decay. `corrected_dJk` is a plain Python float (from
    `_paired_dJ`, unlike `dJk`'s numpy float64), so a 0/0 (both arms exactly
    flat -- e.g. a stalled generation where neither the windowed nor the
    null-control arm moved `Jc` at all) raises `ZeroDivisionError` in pure
    Python instead of numpy's silent `nan`; computed via numpy here so that
    case reports `nan` (genuinely undefined ratio) instead of crashing."""
    dJks = np.asarray([g["corrected_dJk"] for g in generations], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return [float(dJks[i + 1] / dJks[i]) for i in range(len(dJks) - 1)]


def fit_geometric(dJk):
    """Least-squares geometric fit dJk[k] ~= dJ1_fit * r**(k-1) (Run 10
    Experiment 4, only needed if Experiments 1-2 come back ambiguous).
    Fits in log-space (ordinary least squares on log(dJk) vs k). Raises
    ValueError on any non-positive entry -- a sign flip or zero already
    means the geometric-decay hypothesis this fit assumes doesn't apply, so
    silently fitting through it would misreport the extrapolation.

    Returns (r, dJ1_fit)."""
    dJk = np.asarray(dJk, float)
    if dJk.size < 2:
        raise ValueError("need at least 2 points to fit a ratio")
    if np.any(dJk <= 0):
        raise ValueError("fit_geometric requires all dJk > 0")
    k = np.arange(1, dJk.size + 1)
    slope, intercept = np.polyfit(k, np.log(dJk), 1)
    return float(np.exp(slope)), float(np.exp(intercept + slope))


def grow_topology(base, n_gen=2, cand_seeds=range(2), dx=0.05, windows=None,
                  tol=1e-4, outer=20, pos_iters=40, patience=5, sub=8,
                  verbose=True):
    """Task B driver: from a converged solution, greedily birth one kink at a
    time.  Each generation tries inserting into each family, over a few
    lifetime windows and jitter seeds; every candidate is re-optimized with
    the block alternation, pruned, and certified.  The best candidate is kept
    only if J_certified strictly improves by more than `tol`; otherwise the
    search stops.  `windows` is a list of (t_birth, t_death); default probes a
    full lifetime plus two shorter co-moving windows.  Returns the grown
    solution dict (with lifetime masks)."""
    cur = prune(base, tol=1e-8)
    curJ = certify(cur, sub=sub)["Jc"]
    if verbose:
        print(f"  base J_certified = {curJ:.5f}  "
              f"(Kf={cur['XI'].shape[1]}, Kg={cur['ETA'].shape[1]})")

    for gen in range(1, n_gen + 1):
        t = cur["t"]
        win_list = windows if windows is not None else [
            (t[0], t[-1]), (0.5, 1.0), (0.6, 0.9)]
        best_cand, best_candJ, best_desc = None, curJ + tol, None

        for family in ("f", "g"):
            W = cur["A"] if family == "f" else cur["B"]
            parent = int(np.abs(W).max(axis=0).argmax())   # most active kink
            for (tb, td) in win_list:
                for cs in cand_seeds:
                    rng = np.random.default_rng(1000 * gen + cs)
                    XI2, ETA2, af2, ag2 = add_kink(
                        family, cur["XI"], cur["ETA"], cur["alive_f"],
                        cur["alive_g"], parent, t, tb, td, dx=dx, rng=rng)
                    # bootstrap weights for the grown dimensions (zero-weight
                    # insertion => this re-solve starts from the current soln).
                    # The unchanged family's current weights seed the first LP.
                    if family == "f":
                        A0 = lp_weights_f(XI2, cur["B"], ETA2, ub=_ub(af2))
                    else:
                        A0 = cur["A"]
                    B0 = lp_weights_g(A0, XI2, ETA2, ub=_ub(ag2))
                    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
                    cand = _alternate(A0, XI2, B0, ETA2, t, af2, ag2,
                                      outer=outer, pos_iters=pos_iters,
                                      optimize_pos=True, verbose=False,
                                      patience=patience)
                    cand = prune(cand, tol=1e-8)
                    c = certify(cand, sub=sub)
                    if not c["rep"]["ALL CONSTRAINTS OK"]:
                        continue
                    if c["Jc"] > best_candJ:
                        best_cand, best_candJ = cand, c["Jc"]
                        best_desc = (family, tb, td, cs)

        if best_cand is None:
            if verbose:
                print(f"  gen {gen}: no accepted insertion "
                      f"(no candidate beat {curJ:.5f} + {tol}); stopping.")
            break
        fam, tb, td, cs = best_desc
        if verbose:
            print(f"  gen {gen}: ACCEPT  +{fam}-kink on "
                  f"[{tb:.2f},{td:.2f}] (seed {cs})   "
                  f"J_certified {curJ:.5f} -> {best_candJ:.5f}   "
                  f"(Kf={best_cand['XI'].shape[1]}, "
                  f"Kg={best_cand['ETA'].shape[1]})")
        cur, curJ = best_cand, best_candJ
    return cur

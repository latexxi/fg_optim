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
                    pos_iters=40, patience=None, sub=8):
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


def generation_ladder(base, n_gen=4, window0=0.5, window_ratio=0.5,
                      seeds=range(3), dx=0.05, base_fine_sub=4, coarse_N=8,
                      outer=12, pos_iters=40, sub=8, verbose=True):
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

    Returns dict(generations=[dict(k, w_k, window, Jc, dJk, feasible,
    guard_Jc, guard_dJk, guard_feasible, diagnostics, spread), ...],
    base_Jc)."""
    cur = prune(base, tol=1e-8)
    base_Jc = certify(cur, sub=sub)["Jc"]

    t0, t1 = cur["t"][0], cur["t"][-1]
    ws = [window0 * window_ratio**(k - 1) for k in range(1, n_gen + 1)]
    windows = [(t1 - w, t1) for w in ws]
    fine_subs = [base_fine_sub * window0 / w for w in ws]
    grid = graded_grid(windows, coarse_N=coarse_N, fine_sub=fine_subs,
                       t0=t0, t1=t1)
    # A regridding must be a strict REFINEMENT of `base`'s own nodes, never a
    # coarsening -- graded_grid's own uniform background (`coarse_N`) may be
    # coarser than base's original grid outside the windows, which would
    # silently lose resolution there and corrupt every downstream dJk. Union
    # with cur["t"] guarantees every original node survives.
    t_new = np.union1d(grid, cur["t"])

    A2 = _interp_to_grid(cur["A"], cur["t"], t_new)
    XI2 = _interp_to_grid(cur["XI"], cur["t"], t_new)
    B2 = _interp_to_grid(cur["B"], cur["t"], t_new)
    ETA2 = _interp_to_grid(cur["ETA"], cur["t"], t_new)
    alive_f2 = np.ones((len(t_new), XI2.shape[1]), dtype=bool)
    alive_g2 = np.ones((len(t_new), ETA2.shape[1]), dtype=bool)
    A2 = lp_weights_f(XI2, B2, ETA2, ub=_ub(alive_f2))   # restore exact
    B2 = lp_weights_g(A2, XI2, ETA2, ub=_ub(alive_g2))   # feasibility
    cur = dict(A=A2, XI=XI2, B=B2, ETA=ETA2, t=t_new, J=total_J(A2, XI2, B2, ETA2),
              hist=[], alive_f=alive_f2, alive_g=alive_g2)
    regrid_Jc = certify(cur, sub=sub)["Jc"]
    if verbose:
        print(f"  base J_certified = {base_Jc:.5f}  ->  regridded onto "
              f"{len(t_new)} nodes, J_certified = {regrid_Jc:.5f}")
    if abs(regrid_Jc - base_Jc) > 0.01 * abs(base_Jc):
        raise RuntimeError(
            f"regrid is not within 1% of base: base {base_Jc:.5f} vs "
            f"regridded {regrid_Jc:.5f} -- fix the migration before "
            f"trusting any dJk")

    generations = []
    J = regrid_Jc
    for k in range(1, n_gen + 1):
        w_k, window_k = ws[k - 1], windows[k - 1]
        windowed = generation_step(cur, window_k, seeds=seeds, dx=dx,
                                   outer=outer, pos_iters=pos_iters, sub=sub)
        guard = generation_step(cur, (t0, t1), seeds=seeds, dx=dx,
                                outer=outer, pos_iters=pos_iters, sub=sub)
        dJk = windowed["Jc"] - J
        guard_dJk = guard["Jc"] - J
        if verbose:
            print(f"  gen {k}: w_k={w_k:.4f}  Jc {J:.5f} -> "
                  f"{windowed['Jc']:.5f}  dJk={dJk:+.5f}  "
                  f"(feasible={windowed['feasible']})   guard: "
                  f"{guard['Jc']:.5f}  guard_dJk={guard_dJk:+.5f}  "
                  f"(feasible={guard['feasible']})")
        generations.append(dict(
            k=k, w_k=w_k, window=window_k, Jc=windowed["Jc"], dJk=dJk,
            feasible=windowed["feasible"], guard_Jc=guard["Jc"],
            guard_dJk=guard_dJk, guard_feasible=guard["feasible"],
            diagnostics=windowed["diagnostics"], spread=windowed["spread"]))
        cur, J = windowed["sol"], windowed["Jc"]
    return dict(generations=generations, base_Jc=regrid_Jc)


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

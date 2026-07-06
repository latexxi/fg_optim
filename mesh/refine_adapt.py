"""Adaptive (harvest-gauge, band-refined) refinement driver — REVISED two-phase.

Per `plans/mesh/03-driver.md` §3.0: a diagnosis run falsified the original
single-phase design. Cold-starting `alternating_maximization` (coordinate ascent
on a bilinear objective, many fixed points) at a deep grid lands in a random
stuck basin (`Jc` non-monotone, e.g. 2.288, 2.574, 2.551, ...). Only a
disciplined uniform climb from k=1 reliably threads the good basin. So this
driver is split:

  Phase A — BASIN DISCIPLINE (mandatory): uniform dyadic climb `k0 -> k_seed`,
            x doubles globally each step, exactly `refine_baseline.
            dyadic_refinement`'s loop (reusing its `interpolate_to_next_level`
            helper and the same g-init ramp). This is not a step to skip or
            shortcut -- it's the only path that has been shown to reproduce
            the trustworthy uniform-ladder numbers.

  Phase B — DEPTH (the efficiency lever): from the Phase-A seed, `n_band`
            generations of band-only x refinement (`adapt.band_refine`,
            `|x|<BAND` only) + `prolong.adaptive_warm_start` with `t`
            UNCHANGED (primer §0.4 fact 1: the mesh cannot see time-node
            *position*, only its count `M` -- so `tau_regrid`/`regauge` would
            be an inert no-op here and is not called), then re-solve.

  Phase C — M-CLIMB (the bounded/unbounded discriminator): from the Phase-B
            seed, `n_mclimb` doublings of `M` (x grid FROZEN), warm-started
            across M by `prolong.adaptive_warm_start` (t_new has more nodes ->
            `regauge_time` stretches the field over index space; `prolong_x` is
            identity since x is unchanged), then re-solve. This is the ONLY
            axis that is basin-disciplined across M: cold-solving at a deep M
            scatters (03 §3.0 finding 1, e.g. uniform-k7 Jc(M) non-monotone
            2.449/2.584/2.409/2.720), but warm-starting from the converged
            lower-M state threads the basin and gives a clean monotone ladder.
            This is the un-confounded J-vs-resolution read the whole track was
            built for -- run it and read `dJk` across Phase C.

Phases A+B run at the starting `M`; Phase C then climbs M with x frozen. Fact 3
(03 §3.0): at fixed M, x-refinement saturates -- Phase B's `dJk` decays toward 0.
That saturation is *not* by itself evidence for bounded J; whether the saturation
*ceiling* rises with M is what Phase C measures (and 04's `m_sweep` cross-checks).
"""
import time
import numpy as np

from .grid import make_grids
from .constraints import build_constraints, check_feasible
from .alternating import alternating_maximization
from .refine_baseline import interpolate_to_next_level
from .adapt import band_refine, BAND
from .prolong import adaptive_warm_start


def adaptive_refinement(k_seed=4, n_band=5, k0=1, M=32, n_mclimb=0,
                        max_iter=80, tol=1e-8, verbose=True):
    """Two-phase harvest-gauge refinement at FIXED M. Returns list of result dicts.

    Phase A (mandatory basin discipline): uniform dyadic climb k0..k_seed,
    exactly `dyadic_refinement`'s loop (x doubles globally each step, warm
    start by `interpolate_to_next_level`, re-solve, feasibility-assert the
    interpolated warm start before moving on).

    Phase B (depth): n_band generations of band_refine(x) [|x|<BAND only] +
    adaptive_warm_start(f, g, x, t, x_new, t) [t passed through UNCHANGED --
    M is fixed the whole run] + re-solve.

    Each generation is recorded as a dict:
        {gen, phase ('A'|'B'), N, M, n_nodes=N*(M+1), Jc, dJk=Jc-Jc_prev,
         x_grid, t_grid, f, g, elapsed}

    Jc is asserted non-decreasing across ALL generations (both phases,
    continuous numbering) -- warm starts are J-neutral by construction (primer
    §0.4 facts 1 & 3) and each per-grid LP solve is exact for its own basin,
    so the disciplined sequence must be monotone up to ~1e-6. A drop raises.
    """
    results = []
    gen = 0
    Jc_prev = None

    # ---------------- Phase A: uniform climb, basin discipline ----------------
    x, t = make_grids(k0, M)
    N = len(x)
    g_init = np.array([[0.5 * t[j] * (x[i] ** 2 - 1.0) for j in range(M + 1)]
                       for i in range(N)])
    f, g = None, g_init

    for k in range(k0, k_seed + 1):
        x, t = make_grids(k, M)
        N = len(x)
        if verbose:
            print(f"\nGen {gen} [Phase A] uniform k={k} (N={N}, M={M})...")
        f_init = None if f is None else f
        t0 = time.time()
        f, g, J_hist = alternating_maximization(x, t, f_init=f_init, g_init=g,
                                                max_iter=max_iter, tol=tol, verbose=False)
        elapsed = time.time() - t0
        Jc = J_hist[-1]
        dJk = 0.0 if Jc_prev is None else Jc - Jc_prev
        if verbose:
            print(f"  Gen {gen}: J = {Jc:.8f}, dJk = {dJk:+.6f}, time = {elapsed:.1f}s")
        if Jc_prev is not None:
            assert dJk > -1e-6, (
                f"Jc dropped at gen {gen} (Phase A, k={k}): "
                f"{Jc_prev:.8f} -> {Jc:.8f} (dJk={dJk:.2e}) "
                "-- warm start or feasibility bug upstream"
            )

        results.append({
            'gen': gen, 'phase': 'A', 'N': N, 'M': M, 'n_nodes': N * (M + 1),
            'Jc': Jc, 'dJk': dJk,
            'x_grid': x.copy(), 't_grid': t.copy(),
            'f': f.copy(), 'g': g.copy(), 'elapsed': elapsed,
        })
        Jc_prev = Jc
        gen += 1

        if k < k_seed:
            f, x_new = interpolate_to_next_level(f, x, t)
            g, _ = interpolate_to_next_level(g, x, t)
            A_eq_f, b_eq_f, A_ub_f, b_ub_f = build_constraints(x_new, t, True)
            A_eq_g, b_eq_g, A_ub_g, b_ub_g = build_constraints(x_new, t, False)
            eq_f, ub_f = check_feasible(f.flatten(), A_eq_f, b_eq_f, A_ub_f, b_ub_f)
            eq_g, ub_g = check_feasible(g.flatten(), A_eq_g, b_eq_g, A_ub_g, b_ub_g)
            assert eq_f and ub_f, f"Interpolated f infeasible at k={k + 1}"
            assert eq_g and ub_g, f"Interpolated g infeasible at k={k + 1}"
            x = x_new

    # ---------------- Phase B: band-only x depth, t (M) fixed ----------------
    for b in range(n_band):
        Jc_prev = results[-1]['Jc']

        x_new = band_refine(x, band=BAND)
        f0, g0 = adaptive_warm_start(f, g, x, t, x_new, t)   # t_new == t: M unchanged

        N = len(x_new)
        if verbose:
            print(f"\nGen {gen} [Phase B] band-refine (N={N}, M={M})...")
        t0 = time.time()
        f, g, J_hist = alternating_maximization(x_new, t, f_init=f0, g_init=g0,
                                                max_iter=max_iter, tol=tol, verbose=False)
        elapsed = time.time() - t0
        Jc = J_hist[-1]
        dJk = Jc - Jc_prev
        if verbose:
            print(f"  Gen {gen}: J = {Jc:.8f}, dJk = {dJk:+.6f}, time = {elapsed:.1f}s")

        assert dJk > -1e-6, (
            f"Jc dropped at gen {gen} (Phase B, band gen {b}): "
            f"{Jc_prev:.8f} -> {Jc:.8f} (dJk={dJk:.2e}) "
            "-- warm start or feasibility bug upstream"
        )

        results.append({
            'gen': gen, 'phase': 'B', 'N': N, 'M': M, 'n_nodes': N * (M + 1),
            'Jc': Jc, 'dJk': dJk,
            'x_grid': x_new.copy(), 't_grid': t.copy(),
            'f': f.copy(), 'g': g.copy(), 'elapsed': elapsed,
        })

        x = x_new
        gen += 1

    # ---------------- Phase C: M-climb, x frozen, warm-start across M ----------------
    M_cur = M
    for c in range(n_mclimb):
        Jc_prev = results[-1]['Jc']

        M_new = 2 * M_cur
        _, t_new = make_grids(k0, M_new)     # only t used; its length (M_new+1) is all that matters
        f0, g0 = adaptive_warm_start(f, g, x, t, x, t_new)   # x frozen; t grows -> regauge stretch

        N = len(x)
        if verbose:
            print(f"\nGen {gen} [Phase C] M-climb (N={N}, M={M_new})...")
        t0 = time.time()
        f, g, J_hist = alternating_maximization(x, t_new, f_init=f0, g_init=g0,
                                                max_iter=max_iter, tol=tol, verbose=False)
        elapsed = time.time() - t0
        Jc = J_hist[-1]
        dJk = Jc - Jc_prev
        if verbose:
            print(f"  Gen {gen}: J = {Jc:.8f}, dJk = {dJk:+.6f}, time = {elapsed:.1f}s")

        assert dJk > -1e-6, (
            f"Jc dropped at gen {gen} (Phase C, M-climb {c}): "
            f"{Jc_prev:.8f} -> {Jc:.8f} (dJk={dJk:.2e}) "
            "-- warm start or feasibility bug upstream"
        )

        results.append({
            'gen': gen, 'phase': 'C', 'N': N, 'M': M_new, 'n_nodes': N * (M_new + 1),
            'Jc': Jc, 'dJk': dJk,
            'x_grid': x.copy(), 't_grid': t_new.copy(),
            'f': f.copy(), 'g': g.copy(), 'elapsed': elapsed,
        })

        t = t_new
        M_cur = M_new
        gen += 1

    return results


def two_d_climb(k_seed=3, n_band_seed=1, k0=1, M0=16, n_steps=4,
                max_iter=80, tol=1e-8, verbose=True):
    """Genuinely-2-D basin-disciplined climb: grow x-band AND M *together* each step.

    Phase C (`adaptive_refinement(n_mclimb=)`) freezes x and climbs only M -- one
    x-slice. This is the un-slice'd version: after a Phase-A/B seed at (k_seed, M0),
    each of `n_steps` iterations does BOTH `band_refine(x)` and `M -> 2*M` in a
    single warm start (`adaptive_warm_start` -> `regauge_time` stretches over the
    finer time-index grid, `prolong_x` inserts the new band nodes; both J-neutral,
    feasibility asserted), then re-alternates. So x-band-resolution and M climb in
    lockstep -- the joint (x, M) resolution limit the whole track was built to read,
    with both axes basin-disciplined (no cold scatter on either).

    Returns the seed's result dicts followed by the interleaved steps (phase 'D').
    """
    seed = adaptive_refinement(k_seed=k_seed, n_band=n_band_seed, k0=k0, M=M0,
                               n_mclimb=0, max_iter=max_iter, tol=tol, verbose=verbose)
    results = list(seed)
    last = results[-1]
    f, g = last['f'], last['g']
    x, t = last['x_grid'], last['t_grid']
    M_cur = M0
    gen = last['gen'] + 1

    for s in range(n_steps):
        Jc_prev = results[-1]['Jc']

        x_new = band_refine(x, band=BAND)
        M_new = 2 * M_cur
        _, t_new = make_grids(k0, M_new)
        f0, g0 = adaptive_warm_start(f, g, x, t, x_new, t_new)   # x AND t both grow

        N = len(x_new)
        if verbose:
            print(f"\nGen {gen} [Phase D] 2-D step (N={N}, M={M_new})...")
        t0 = time.time()
        f, g, J_hist = alternating_maximization(x_new, t_new, f_init=f0, g_init=g0,
                                                max_iter=max_iter, tol=tol, verbose=False)
        elapsed = time.time() - t0
        Jc = J_hist[-1]
        dJk = Jc - Jc_prev
        if verbose:
            print(f"  Gen {gen}: J = {Jc:.8f}, dJk = {dJk:+.6f}, time = {elapsed:.1f}s")

        assert dJk > -1e-6, (
            f"Jc dropped at gen {gen} (Phase D, 2-D step {s}): "
            f"{Jc_prev:.8f} -> {Jc:.8f} (dJk={dJk:.2e}) -- warm start/feasibility bug"
        )

        results.append({
            'gen': gen, 'phase': 'D', 'N': N, 'M': M_new, 'n_nodes': N * (M_new + 1),
            'Jc': Jc, 'dJk': dJk,
            'x_grid': x_new.copy(), 't_grid': t_new.copy(),
            'f': f.copy(), 'g': g.copy(), 'elapsed': elapsed,
        })
        x, t, M_cur = x_new, t_new, M_new
        gen += 1

    return results


if __name__ == "__main__":
    # Full 2-D basin-disciplined climb: Phase A (k-basin) + B (band-x depth) + C (M-climb).
    res = adaptive_refinement(k_seed=4, n_band=3, k0=1, M=16, n_mclimb=4, verbose=True)
    print("\n gen | ph |   N  | M  | nodes  |   Jc      |  dJk")
    for r in res:
        print(f" {r['gen']:3d} |  {r['phase']} | {r['N']:4d} | {r['M']:2d} | "
              f"{r['n_nodes']:6d} | {r['Jc']:.6f} | {r['dJk']:+.6f}")
    Js = [r['Jc'] for r in res]
    print("dJk:", np.round(np.diff(Js), 5))

    # -------------------- Acceptance checks (plans/mesh/03-driver.md §3.4) --------------------
    from .refine_baseline import dyadic_refinement

    # 1. Phase A must reproduce the disciplined uniform ladder exactly (it IS that ladder)
    res_a = adaptive_refinement(k_seed=5, n_band=0, k0=1, M=32, verbose=False)
    base = dyadic_refinement(k_start=1, k_max=5, M=32, verbose=False)
    for ra, rb in zip(res_a, base):
        assert abs(ra['Jc'] - rb['J']) < 1e-6, (ra['N'], ra['Jc'], rb['J'])
    print("\nPhase A reproduces disciplined uniform ladder: OK")

    # 2. Full run is monotone non-decreasing and Phase B stays feasible (asserted inside)
    res_full = adaptive_refinement(k_seed=4, n_band=4, k0=1, M=32, verbose=False)
    Js_full = [r['Jc'] for r in res_full]
    assert all(Js_full[i] >= Js_full[i - 1] - 1e-6 for i in range(1, len(Js_full))), Js_full
    print("full climb+band run monotone: OK   final Jc=%.4f at N=%d, %d nodes"
          % (res_full[-1]['Jc'], res_full[-1]['N'], res_full[-1]['n_nodes']))

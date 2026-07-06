"""Validation + the M-sweep bounded/unbounded discriminator.

Per `plans/mesh/04-validate.md` (REVISED): at fixed M, x-refinement saturates (see
`refine_adapt.py` / `plans/mesh/03-driver.md` §3.0 finding 3), so a fixed-M `dJk->0`
is NOT by itself evidence of bounded J -- it could be an M-ceiling artifact. The
real bounded-vs-unbounded discriminator is whether that saturation ceiling itself
rises as M grows. `m_sweep` is that experiment; everything else here is a gate or
a supporting figure/table, not the verdict.

Four gates (§4.2), the M-sweep (§4.1, the money experiment), an efficiency table
(§4.3), and four figures (§4.4). `main()` runs all of it. Nothing here modifies the
solver (`grid.py`, `constraints.py`, `objective.py`, `lp_subproblem.py`,
`alternating.py`, `refine_baseline.py`) or the adaptive machinery (`adapt.py`,
`prolong.py`, `refine_adapt.py`) -- pure read-and-report.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .grid import make_grids
from .constraints import build_constraints, check_feasible
from .refine_baseline import dyadic_refinement
from .refine_adapt import adaptive_refinement
from .adapt import BAND

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# External sanity-anchor points (plans/gen1-2-3-inspection.md), from the sibling
# fg_opt3 npz solutions. These come from a JOINT (x, M) refinement, a different
# construction than our fixed-M-then-sweep-M experiment below -- plotted only as
# an outside reference on the money plot, never averaged into our own numbers.
REF_NPZ = [
    # (label, N, M, J)
    ("k04/gen1", 17, 128, 2.6253),
    ("k05/gen2", 33, 128, 2.8361),
    ("k06/gen3", 65, 256, 3.0553),
]


def _per_x_harvest(f, g, x_grid, t_grid):
    """Harvest attributed to each interior x-node i: sum_j f_diff[i,j]*kappa_g[i,j].

    Same discrete kappa_g / f_diff pieces `objective.compute_J` and
    `harvest_per_interval` use, just summed over the other axis (j, not i) --
    gate 3's / mesh_harvest.png's per-x mass profile. Recomputed locally (not
    imported from objective.py's private `_kappa_g`) to keep this a read-only
    consumer of the public solver API.
    """
    N = len(x_grid)
    M = len(t_grid) - 1
    h_left = (x_grid[1:N - 1] - x_grid[0:N - 2])[:, None]
    h_right = (x_grid[2:N] - x_grid[1:N - 1])[:, None]
    kappa_g = ((g[2:N, :] - g[1:N - 1, :]) / h_right
               - (g[1:N - 1, :] - g[0:N - 2, :]) / h_left)
    f_diff = f[1:N - 1, 1:] - f[1:N - 1, :-1]
    return np.sum(f_diff * kappa_g[:, :M], axis=1)   # shape (N-2,), aligned to x_grid[1:-1]


# ----------------------------------------------------------------------------
# 4.2 Gates
# ----------------------------------------------------------------------------

def gate1_phase_a_fidelity(M=32, k_seed=5, verbose=True):
    """Phase-A basin fidelity: adaptive_refinement(n_band=0) must equal
    dyadic_refinement to <1e-6 per level. If this fails, Phase A isn't threading
    the disciplined basin and every downstream number is void (04-validate §4.2 #1).
    """
    res = adaptive_refinement(k_seed=k_seed, n_band=0, k0=1, M=M, verbose=False)
    base = dyadic_refinement(k_start=1, k_max=k_seed, M=M, verbose=False)
    diffs = [abs(ra['Jc'] - rb['J']) for ra, rb in zip(res, base)]
    max_diff = max(diffs) if diffs else 0.0
    passed = max_diff < 1e-6
    if verbose:
        print(f"[GATE 1] Phase-A basin fidelity (M={M}, k=1..{k_seed}): "
              f"max|Jc_adapt - J_dyadic| = {max_diff:.3e}  -> {'PASS' if passed else 'FAIL'}")
    return {'name': 'gate1_phase_a_fidelity', 'passed': passed, 'max_diff': max_diff,
            'adapt': res, 'dyadic': base}


def gate2_monotonicity_feasibility(M=32, k_seed=4, n_band=4, verbose=True):
    """Full climb+band run: Jc non-decreasing across ALL generations, and every
    generation's (f, g) independently passes check_feasible on its own grid
    (defense-in-depth on top of the internal assertions in refine_adapt.py /
    prolong.py). Reports the worst (most negative) dJk and any feasibility break.
    """
    res = adaptive_refinement(k_seed=k_seed, n_band=n_band, k0=1, M=M, verbose=False)
    Js = [r['Jc'] for r in res]
    diffs = np.diff(Js)
    worst_drift = float(np.min(diffs)) if len(diffs) else 0.0
    monotone_ok = worst_drift > -1e-6

    feasible = True
    first_bad_gen = None
    for r in res:
        A_eq_f, b_eq_f, A_ub_f, b_ub_f = build_constraints(r['x_grid'], r['t_grid'], True)
        A_eq_g, b_eq_g, A_ub_g, b_ub_g = build_constraints(r['x_grid'], r['t_grid'], False)
        eq_f, ub_f = check_feasible(r['f'].ravel(), A_eq_f, b_eq_f, A_ub_f, b_ub_f)
        eq_g, ub_g = check_feasible(r['g'].ravel(), A_eq_g, b_eq_g, A_ub_g, b_ub_g)
        gen_ok = eq_f and ub_f and eq_g and ub_g
        feasible = feasible and gen_ok
        if not gen_ok and first_bad_gen is None:
            first_bad_gen = r['gen']

    passed = monotone_ok and feasible
    if verbose:
        print(f"[GATE 2] Monotonicity + feasibility (M={M}, k_seed={k_seed}, n_band={n_band}): "
              f"worst dJk = {worst_drift:+.3e}, all-gen feasible = {feasible} "
              f"-> {'PASS' if passed else 'FAIL'}"
              + ("" if feasible else f"  (first infeasible gen: {first_bad_gen})"))
    return {'name': 'gate2_monotonicity_feasibility', 'passed': passed,
            'worst_drift': worst_drift, 'feasible': feasible, 'run': res}


def gate3_band_mass(run, band=BAND, verbose=True):
    """Band-mass premise (I2): >=95% of harvest mass must sit in |x|<BAND on every
    solved generation, else band-refine is starving real structure leaking into
    the arms (04-validate §4.2 #3).
    """
    fracs = []
    for r in run:
        x, t = r['x_grid'], r['t_grid']
        h = _per_x_harvest(r['f'], r['g'], x, t)
        xi = x[1:-1]
        total = h.sum()
        band_mass = h[np.abs(xi) < band].sum()
        frac = float(band_mass / total) if abs(total) > 1e-12 else 1.0
        fracs.append(frac)
    min_frac = min(fracs) if fracs else 1.0
    passed = min_frac >= 0.95
    if verbose:
        print(f"[GATE 3] Band-mass premise (|x|<{band}): min frac over {len(fracs)} generations "
              f"= {min_frac:.4f}  -> {'PASS' if passed else 'FAIL'}")
        print("          per-gen fracs: " + ", ".join(f"{fr:.3f}" for fr in fracs))
    return {'name': 'gate3_band_mass', 'passed': passed, 'min_frac': min_frac, 'fracs': fracs}


def gate4_band_vs_uniform(run, M=32, k_uniform_max=7, verbose=True):
    """Band-vs-uniform gap at matched resolution (§4.2 #4). Diagnosis (03 §3.0
    finding 4) found band captures ~87% of the x-gain at ~half the nodes -- this
    gate quantifies the Jc ratio at matched N honestly (informative threshold
    0.80, not a strict correctness bar the way gates 1-2 are).
    """
    band_gens = [r for r in run if r['phase'] == 'B']
    final = band_gens[-1] if band_gens else run[-1]
    N_band = final['N']

    base = dyadic_refinement(k_start=1, k_max=k_uniform_max, M=M, verbose=False)
    match = next((r for r in base if len(r['x_grid']) >= N_band), base[-1])
    N_uniform = len(match['x_grid'])
    ratio = final['Jc'] / match['J'] if match['J'] else float('nan')
    passed = ratio >= 0.80
    if verbose:
        print(f"[GATE 4] Band-vs-uniform at matched N (band N={N_band} vs uniform N={N_uniform}, "
              f"M={M}): Jc_band={final['Jc']:.6f}  Jc_uniform={match['J']:.6f}  "
              f"ratio={ratio:.4f}  -> {'PASS' if passed else 'FAIL'} (informative, threshold 0.80)")
    return {'name': 'gate4_band_vs_uniform', 'passed': passed, 'ratio': ratio,
            'N_band': N_band, 'N_uniform': N_uniform,
            'Jc_band': final['Jc'], 'Jc_uniform': match['J']}


# ----------------------------------------------------------------------------
# 4.1 The M-sweep -- the central experiment
# ----------------------------------------------------------------------------

def m_sweep(Ms=(16, 32, 64, 128), k_seed=4, n_band=4, verbose=True):
    """For each M, run adaptive_refinement (climb+band, fixed M) and record the
    saturated ceiling Jc_sat(M) = the best Jc reached at that M.

    Returns a list of dicts, one per M, each `{'M', 'status', ...}`. On success
    (`status='ok'`) also carries `Jc_sat`, `N_at_sat`, `nodes_at_sat`, `elapsed`,
    and the full generation list `results` (both phases). On failure
    (`status in ('memory_error','error')`) carries `error` and stops there --
    every point up to the failure is still returned and usable.

    Read-off (see module docstring / 04-validate.md §4.1):
      * Jc_sat(M) rising ~linearly in log(M), no leveling  -> UNBOUNDED lean.
      * Jc_sat(M) leveling to a horizontal asymptote        -> BOUNDED lean.
      * noisy / non-monotone in M                           -> basin trouble;
        do NOT interpret (flagged below via `monotone_ok`).
    """
    out = []
    if verbose:
        print(f"m_sweep: Ms={Ms}, k_seed={k_seed}, n_band={n_band} "
              "(every point via the disciplined climb-from-k0=1 path)")
    for M in Ms:
        entry = {'M': M}
        try:
            t0 = time.time()
            res = adaptive_refinement(k_seed=k_seed, n_band=n_band, k0=1, M=M, verbose=False)
            elapsed = time.time() - t0
            best = max(res, key=lambda r: r['Jc'])
            entry.update({
                'status': 'ok',
                'Jc_sat': best['Jc'], 'N_at_sat': best['N'], 'nodes_at_sat': best['n_nodes'],
                'elapsed': elapsed, 'results': res,
            })
            if verbose:
                print(f"  M={M:4d}: Jc_sat={best['Jc']:.6f}  N={best['N']:4d}  "
                      f"nodes={best['n_nodes']:6d}  ({elapsed:.1f}s)")
        except MemoryError as e:
            entry.update({'status': 'memory_error', 'error': str(e)})
            if verbose:
                print(f"  M={M:4d}: MemoryError -- sweep stops here ({e})")
        except Exception as e:  # pragma: no cover -- defensive, per task spec
            entry.update({'status': 'error', 'error': repr(e)})
            if verbose:
                print(f"  M={M:4d}: FAILED -- {repr(e)}")
        out.append(entry)

    ok = [e for e in out if e['status'] == 'ok']
    if len(ok) >= 2:
        js = [e['Jc_sat'] for e in ok]
        monotone_ok = all(js[i] >= js[i - 1] - 1e-6 for i in range(1, len(js)))
        for e in ok:
            e['monotone_ok'] = monotone_ok
        if verbose:
            tag = ("monotone in M -- OK to interpret" if monotone_ok else
                   "NON-MONOTONE in M -- basin trouble (finding 2): "
                   "do NOT interpret bounded/unbounded from this run")
            print(f"  Jc_sat(M) = {[round(j, 4) for j in js]}  -> {tag}")
    elif ok:
        ok[0]['monotone_ok'] = True  # trivially, with one point
    return out


# ----------------------------------------------------------------------------
# 4.3 Efficiency table
# ----------------------------------------------------------------------------

def efficiency_table(sweep, k_uniform_max=7, verbose=True):
    """For a target Jc* (the M=32 band ceiling from `sweep`, or the first
    successful M if 32 itself failed), how many nodes uniform vs band routes
    need to reach it, at each M in the sweep. Band's frozen arms should give a
    node-count reduction -- that ratio is how much deeper the same RAM budget
    reaches (§4.3).
    """
    ok = [e for e in sweep if e['status'] == 'ok']
    if not ok:
        if verbose:
            print("[TABLE 4.3] no successful M-sweep points -- skipped")
        return []
    ref = next((e for e in ok if e['M'] == 32), ok[0])
    Jc_star = ref['Jc_sat']

    rows = []
    if verbose:
        print(f"\n[TABLE 4.3] nodes to reach Jc* = {Jc_star:.6f} "
              f"(M={ref['M']} band ceiling, this run)")
        print("   M | uniform: (N, nodes)              | band: (N, nodes)                 | node ratio")
    for e in ok:
        M = e['M']
        base = dyadic_refinement(k_start=1, k_max=k_uniform_max, M=M, verbose=False)
        u_match = next((r for r in base if r['J'] >= Jc_star - 1e-3), None)
        if u_match is not None:
            Nu = len(u_match['x_grid'])
            nodes_u = Nu * (M + 1)
            u_str = f"({Nu:4d}, {nodes_u:6d})"
        else:
            Nu = nodes_u = None
            u_str = f"not reached by k={k_uniform_max} (best {base[-1]['J']:.4f})"

        b_match = next((r for r in e['results'] if r['Jc'] >= Jc_star - 1e-3), None)
        if b_match is not None:
            Nb, nodes_b = b_match['N'], b_match['n_nodes']
            b_str = f"({Nb:4d}, {nodes_b:6d})"
        else:
            Nb = nodes_b = None
            b_str = f"not reached (best {e['Jc_sat']:.4f})"

        ratio = (nodes_u / nodes_b) if (nodes_u and nodes_b) else float('nan')
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "n/a"
        if verbose:
            print(f" {M:3d} | {u_str:32s} | {b_str:32s} | {ratio_str}")
        rows.append({'M': M, 'N_uniform': Nu, 'nodes_uniform': nodes_u,
                     'N_band': Nb, 'nodes_band': nodes_b, 'ratio': ratio,
                     'Jc_star': Jc_star})
    return rows


# ----------------------------------------------------------------------------
# 4.4 Figures
# ----------------------------------------------------------------------------

def fig_msweep(sweep, path):
    """mesh_msweep.png -- the money plot: Jc_sat(M) vs log2(M)."""
    ok = [e for e in sweep if e['status'] == 'ok']
    Ms = [e['M'] for e in ok]
    Js = [e['Jc_sat'] for e in ok]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(np.log2(Ms), Js, 'o-', color='#a3123a', lw=2.2, ms=8,
             label='Jc_sat(M), this sweep (band route)', zorder=4)
    for m, j in zip(Ms, Js):
        ax.annotate(f"{j:.3f}", (np.log2(m), j), textcoords="offset points",
                     xytext=(0, 9), fontsize=8, ha='center', color='#a3123a')

    for label, N, M, J in REF_NPZ:
        ax.scatter([np.log2(M)], [J], marker='x', s=70, color='#2f6f4f',
                    linewidths=2, zorder=5)
        ax.annotate(f"{label}\n(N={N})", (np.log2(M), J), textcoords="offset points",
                     xytext=(8, -14), fontsize=7, color='#2f6f4f')

    ax.set_xlabel("log2(M)")
    ax.set_ylabel("Jc_sat(M)  (best certified J reached at that M)")
    ax.set_title("The money plot: does the fixed-M x-saturation ceiling rise with M?\n"
                  "rising-linear -> unbounded lean; leveling -> bounded lean")
    ax.legend(loc='lower right', fontsize=8,
               title="x: fg_opt3 npz anchors\n(different, joint x,M refinement)",
               title_fontsize=6.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def fig_ladder(sweep, path):
    """mesh_ladder.png -- Jc(N) for the climb+band run at each M, log-x."""
    ok = [e for e in sweep if e['status'] == 'ok']
    cmap = plt.get_cmap('viridis')
    n = max(1, len(ok) - 1)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for i, e in enumerate(ok):
        color = cmap(i / n)
        nodes = [r['n_nodes'] for r in e['results']]
        Js = [r['Jc'] for r in e['results']]
        phases = [r['phase'] for r in e['results']]
        ax.plot(nodes, Js, '-', color=color, lw=1.6, label=f"M={e['M']}", zorder=2)
        a_pts = [(nd, j) for nd, j, p in zip(nodes, Js, phases) if p == 'A']
        b_pts = [(nd, j) for nd, j, p in zip(nodes, Js, phases) if p == 'B']
        if a_pts:
            ax.scatter(*zip(*a_pts), color=color, marker='o', s=26, zorder=3)
        if b_pts:
            ax.scatter(*zip(*b_pts), color=color, marker='^', s=26, zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel("nodes = N*(M+1)  (log scale)")
    ax.set_ylabel("Jc")
    ax.set_title("Jc(N) climb+band ladder, one line per M\n"
                  "(circle = Phase A uniform climb, triangle = Phase B band depth)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def fig_grid(gate2_run, path, M=32, k_uniform=4):
    """mesh_grid.png -- adaptive (x,t) nodes for the deepest generation next to
    the uniform grid at the Phase-A seed level. Visual of the band x (uniform t)
    node allocation.
    """
    final = gate2_run[-1]
    x_adapt, t_adapt = final['x_grid'], final['t_grid']
    x_uniform, t_uniform = make_grids(k_uniform, M)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)

    Xu, Tu = np.meshgrid(x_uniform, t_uniform, indexing='ij')
    axes[0].scatter(Tu, Xu, s=5, color='#3b4cc0')
    axes[0].axhspan(-BAND, BAND, color='#3b4cc0', alpha=0.10)
    axes[0].set_title(f"uniform grid (N={len(x_uniform)}, M={M})")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")

    Xa, Ta = np.meshgrid(x_adapt, t_adapt, indexing='ij')
    axes[1].scatter(Ta, Xa, s=2.5, color='#b5171e')
    axes[1].axhspan(-BAND, BAND, color='#b5171e', alpha=0.10)
    axes[1].set_title(f"band-refined grid (N={len(x_adapt)}, M={final['M']})")
    axes[1].set_xlabel("t")

    fig.suptitle("Node allocation: uniform vs band x-refine  (|x|<0.4 band shaded)")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def fig_harvest(gate2_run, path, band=BAND):
    """mesh_harvest.png -- per-x harvest profile, |x|<0.4 band shaded (gate 3
    visual)."""
    final = gate2_run[-1]
    x, t = final['x_grid'], final['t_grid']
    h = _per_x_harvest(final['f'], final['g'], x, t)
    xi = x[1:-1]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.axvspan(-band, band, color='#f0a500', alpha=0.15, label=f"|x|<{band} (I2 band)")
    ax.plot(xi, h, color='#4b1d76', lw=1.7)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("per-node harvest  sum_j f_diff[i,j]*kappa_g[i,j]")
    ax.set_title(f"Harvest profile, deepest band generation (N={len(x)}, M={final['M']})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(" mesh/validate_adapt.py -- gates + M-sweep + figures  (plans/mesh/04-validate.md)")
    print("=" * 78)

    print("\n-- Gates (4.2) --")
    g1 = gate1_phase_a_fidelity(M=32, k_seed=5)
    g2 = gate2_monotonicity_feasibility(M=32, k_seed=4, n_band=4)
    g3 = gate3_band_mass(g2['run'])
    g4 = gate4_band_vs_uniform(g2['run'], M=32)

    reportable = g1['passed'] and g2['passed']
    if not reportable:
        print("\n*** Gate 1 or 2 FAILED -- everything below is NOT reportable per "
              "04-validate.md §4.2 ***")

    print("\n-- M-sweep (4.1, the central experiment) --")
    sweep = m_sweep()

    print("\n-- Per-M generation ladders --")
    for e in sweep:
        if e['status'] != 'ok':
            print(f"\n  M={e['M']}: {e['status']} -- {e.get('error')}")
            continue
        print(f"\n  M={e['M']}:")
        print("   gen | ph |   N  | nodes  |   Jc      |  dJk")
        for r in e['results']:
            print(f"   {r['gen']:3d} |  {r['phase']} | {r['N']:4d} | {r['n_nodes']:6d} | "
                  f"{r['Jc']:.6f} | {r['dJk']:+.6f}")

    print("\n-- Efficiency table (4.3) --")
    table = efficiency_table(sweep)

    print("\n-- Figures (4.4) --")
    fig_msweep(sweep, os.path.join(REPO_ROOT, "mesh_msweep.png"))
    fig_ladder(sweep, os.path.join(REPO_ROOT, "mesh_ladder.png"))
    fig_grid(g2['run'], os.path.join(REPO_ROOT, "mesh_grid.png"))
    fig_harvest(g2['run'], os.path.join(REPO_ROOT, "mesh_harvest.png"))
    print("  wrote mesh_msweep.png, mesh_ladder.png, mesh_grid.png, mesh_harvest.png to "
          + REPO_ROOT)

    print("\n" + "=" * 78)
    print(" DONE." + ("" if reportable else "  (gate 1/2 FAILED -- see above, not reportable)"))
    print("=" * 78)

    return {'gate1': g1, 'gate2': g2, 'gate3': g3, 'gate4': g4,
            'reportable': reportable, 'sweep': sweep, 'table': table}


if __name__ == "__main__":
    main()

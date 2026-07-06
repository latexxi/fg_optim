"""BASELINE (uniform) dyadic refinement, copied from fg_opt3's refine.py.

Kept unchanged as the comparison point for the adaptive version (plans/mesh/).
The baseline: doubles x uniformly EVERYWHERE each level, keeps M fixed and t
uniform, warm-starts by linear interpolation of both f and g onto the new grid.

Its known waste (what the adaptive version fixes):
  * ~half the x-nodes fall in the dead arms |x|>0.4 where no harvest lives
  * time nodes are uniform, but harvest concentrates near tau*~0.38
The adaptive replacement lives in adapt.py / refine_adapt.py (to be implemented).
"""
import time
import numpy as np
from .grid import make_grids
from .constraints import build_constraints, check_feasible
from .alternating import alternating_maximization


def interpolate_to_next_level(f, x_old, t_grid):
    """Prolong a level-k solution to k+1: old x-nodes copy, midpoints linear-interp.

    Returns (f_new, x_new) with N_new = 2*N_old - 1.
    """
    N_old = len(x_old)
    N_new = 2 * N_old - 1
    x_new = np.empty(N_new)
    f_new = np.empty((N_new, f.shape[1]))
    for i in range(N_old - 1):
        x_new[2 * i] = x_old[i]
        f_new[2 * i, :] = f[i, :]
        x_new[2 * i + 1] = 0.5 * (x_old[i] + x_old[i + 1])
        f_new[2 * i + 1, :] = 0.5 * (f[i, :] + f[i + 1, :])
    x_new[-1] = x_old[-1]
    f_new[-1, :] = f[-1, :]
    return f_new, x_new


def dyadic_refinement(k_start=1, k_max=4, M=8, max_iter=50, tol=1e-8, verbose=True):
    """Solve at k_start, then uniformly refine to k_max. Returns list of result dicts."""
    results = []
    x, t = make_grids(k_start, M)
    N = len(x)
    g_init = np.array([[0.5 * t[j] * (x[i] ** 2 - 1.0) for j in range(M + 1)]
                       for i in range(N)])
    f, g = None, g_init

    for k in range(k_start, k_max + 1):
        x, t = make_grids(k, M)
        N = len(x)
        if verbose:
            print(f"\nLevel k={k} (N={N}, M={M})...")
        f_init = None if f is None else f
        t0 = time.time()
        f, g, J_hist = alternating_maximization(x, t, f_init=f_init, g_init=g,
                                                max_iter=max_iter, tol=tol, verbose=verbose)
        elapsed = time.time() - t0
        if verbose:
            print(f"  Level k={k}: J = {J_hist[-1]:.8f}, time = {elapsed:.1f}s")
        results.append({'k': k, 'J': J_hist[-1], 'f': f.copy(), 'g': g.copy(),
                        'x_grid': x.copy(), 't_grid': t.copy(), 'elapsed': elapsed})
        if k < k_max:
            f, x_new = interpolate_to_next_level(f, x, t)
            g, _ = interpolate_to_next_level(g, x, t)
            A_eq_f, b_eq_f, A_ub_f, b_ub_f = build_constraints(x_new, t, True)
            A_eq_g, b_eq_g, A_ub_g, b_ub_g = build_constraints(x_new, t, False)
            eq_f, ub_f = check_feasible(f.flatten(), A_eq_f, b_eq_f, A_ub_f, b_ub_f)
            eq_g, ub_g = check_feasible(g.flatten(), A_eq_g, b_eq_g, A_ub_g, b_ub_g)
            assert eq_f and ub_f, f"Interpolated f infeasible at k={k+1}"
            assert eq_g and ub_g, f"Interpolated g infeasible at k={k+1}"
            x = x_new
    return results


if __name__ == "__main__":
    results = dyadic_refinement(k_start=1, k_max=6, M=32, verbose=True)
    print("\nSummary:")
    for r in results:
        print(f"  k={r['k']}: J = {r['J']:.8f}")

"""Alternating-LP maximization of J. Copied from fg_opt3 (test/plot block dropped).

Each iteration: LP over f given g, then LP over g given f. Both are exact global
LPs, so J is monotonically non-decreasing across half-steps. Grid (uniform or
adaptive/non-uniform) is passed in; this driver never reads spacing itself.
"""
import numpy as np
from .constraints import build_constraints
from .objective import compute_J
from .lp_subproblem import solve_f_given_g, solve_g_given_f
from .highs_warm import HiGHSWarmLP


def alternating_maximization(x_grid, t_grid, f_init=None, g_init=None,
                             max_iter=50, tol=1e-8, verbose=False):
    """Returns (f, g, J_history). J_history has two entries per iteration."""
    N = len(x_grid)
    M = len(t_grid) - 1

    A_eq_f, b_eq_f, A_ub_f, b_ub_f = build_constraints(x_grid, t_grid, monotone_increasing=True)
    A_eq_g, b_eq_g, A_ub_g, b_ub_g = build_constraints(x_grid, t_grid, monotone_increasing=False)

    n_vars = N * (M + 1)
    warm_f = HiGHSWarmLP(A_ub_f, b_ub_f, A_eq_f, b_eq_f, n_vars)
    warm_g = HiGHSWarmLP(A_ub_g, b_ub_g, A_eq_g, b_eq_g, n_vars)

    f = f_init if f_init is not None else np.zeros((N, M + 1))
    g = g_init if g_init is not None else np.zeros((N, M + 1))

    J_history = []
    for iteration in range(1, max_iter + 1):
        f = solve_f_given_g(g, x_grid, t_grid, A_eq_f, b_eq_f, A_ub_f, b_ub_f, warm_f=warm_f)
        J_history.append(compute_J(f, g, x_grid, t_grid))

        g = solve_g_given_f(f, x_grid, t_grid, A_eq_g, b_eq_g, A_ub_g, b_ub_g, warm_g=warm_g)
        J_history.append(compute_J(f, g, x_grid, t_grid))

        if verbose:
            print(f"  iter {iteration:3d}: J = {J_history[-1]:.8f}")
        if len(J_history) >= 4 and abs(J_history[-1] - J_history[-3]) < tol:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
    return f, g, J_history

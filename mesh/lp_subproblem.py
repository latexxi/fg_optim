"""LP objective vectors + the two convex sub-solves. Copied from fg_opt3 (tests dropped).

Both sub-solves are exact global LPs (HiGHS). Non-uniform x is handled (h_left/h_right
read per node), so the adaptive band-refined grid needs no change here.
"""
import numpy as np
from scipy.optimize import linprog


def build_c_f(g, x_grid, t_grid):
    """Objective coeffs for max J over f, given fixed g.  c_f @ f_vec == J."""
    N = len(x_grid)
    M = len(t_grid) - 1
    n_vars = N * (M + 1)
    h_left = (x_grid[1:N - 1] - x_grid[0:N - 2])[:, None]
    h_right = (x_grid[2:N] - x_grid[1:N - 1])[:, None]
    kappa_g = ((g[2:N, :] - g[1:N - 1, :]) / h_right
               - (g[1:N - 1, :] - g[0:N - 2, :]) / h_left)   # (N-2, M+1)
    ii = np.arange(1, N - 1)
    jj = np.arange(M)
    flat_jp1 = (ii[:, None] * (M + 1) + jj[None, :] + 1).ravel()
    flat_j = (ii[:, None] * (M + 1) + jj[None, :]).ravel()
    kg = kappa_g[:, :M].ravel()
    c_f = np.zeros(n_vars)
    c_f[flat_jp1] += kg
    c_f[flat_j] -= kg
    return c_f


def build_c_g(f, x_grid, t_grid):
    """Objective coeffs for max J over g, given fixed f.  kappa_g is linear in g."""
    N = len(x_grid)
    M = len(t_grid) - 1
    n_vars = N * (M + 1)
    f_diff = f[1:N - 1, 1:] - f[1:N - 1, :-1]              # (N-2, M)
    h_left = (x_grid[1:N - 1] - x_grid[0:N - 2])[:, None]
    h_right = (x_grid[2:N] - x_grid[1:N - 1])[:, None]
    w_im1 = f_diff / h_left
    w_i = -f_diff * (1.0 / h_left + 1.0 / h_right)
    w_ip1 = f_diff / h_right
    ii = np.arange(1, N - 1)
    jj = np.arange(M)
    flat_im1 = ((ii - 1)[:, None] * (M + 1) + jj[None, :]).ravel()
    flat_i = (ii[:, None] * (M + 1) + jj[None, :]).ravel()
    flat_ip1 = ((ii + 1)[:, None] * (M + 1) + jj[None, :]).ravel()
    c_g = np.zeros(n_vars)
    np.add.at(c_g, flat_im1, w_im1.ravel())
    np.add.at(c_g, flat_i, w_i.ravel())
    np.add.at(c_g, flat_ip1, w_ip1.ravel())
    return c_g


def solve_f_given_g(g, x_grid, t_grid, A_eq_f, b_eq_f, A_ub_f, b_ub_f, warm_f=None):
    """Maximize J over f given fixed g. Returns f (N, M+1). warm_f=HiGHSWarmLP or None."""
    N = len(x_grid)
    M = len(t_grid) - 1
    c_f = build_c_f(g, x_grid, t_grid)
    n_vars = N * (M + 1)
    if warm_f is not None:
        x = warm_f.solve(c_f)
    else:
        res = linprog(-c_f, A_ub=A_ub_f, b_ub=b_ub_f, A_eq=A_eq_f, b_eq=b_eq_f,
                      method='highs', bounds=[(None, None)] * n_vars)
        if res.status != 0:
            raise RuntimeError(f"LP (f) failed: {res.message}")
        x = res.x
    return x.reshape(N, M + 1)


def solve_g_given_f(f, x_grid, t_grid, A_eq_g, b_eq_g, A_ub_g, b_ub_g, warm_g=None):
    """Maximize J over g given fixed f. Returns g (N, M+1). warm_g=HiGHSWarmLP or None."""
    N = len(x_grid)
    M = len(t_grid) - 1
    c_g = build_c_g(f, x_grid, t_grid)
    n_vars = N * (M + 1)
    if warm_g is not None:
        x = warm_g.solve(c_g)
    else:
        res = linprog(-c_g, A_ub=A_ub_g, b_ub=b_ub_g, A_eq=A_eq_g, b_eq=b_eq_g,
                      method='highs', bounds=[(None, None)] * n_vars)
        if res.status != 0:
            raise RuntimeError(f"LP (g) failed: {res.message}")
        x = res.x
    return x.reshape(N, M + 1)

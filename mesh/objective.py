"""The objective J and its per-time-interval decomposition.

`compute_J` is copied verbatim from fg_opt3. `harvest_per_interval` is the same
integrand kept per time-interval instead of summed — it is the tau-gauge CDF's
raw material (see plans/mesh/01-grids.md) and is new here, not in fg_opt3.
"""
import numpy as np


def _kappa_g(g, x_grid):
    """Discrete g_xx (kink) at interior nodes i=1..N-2, all times. Shape (N-2, M+1).

    Non-uniform x is handled: h_left/h_right read per node.
    """
    N = len(x_grid)
    h_left = (x_grid[1:N - 1] - x_grid[0:N - 2])[:, None]   # (N-2, 1)
    h_right = (x_grid[2:N] - x_grid[1:N - 1])[:, None]      # (N-2, 1)
    return ((g[2:N, :] - g[1:N - 1, :]) / h_right
            - (g[1:N - 1, :] - g[0:N - 2, :]) / h_left)     # (N-2, M+1)


def compute_J(f, g, x_grid, t_grid):
    """J(f, g) = sum_{i,j} (f[i,j+1]-f[i,j]) * kappa_g[i,j].

    dt cancels analytically: J is a pure sum over consecutive time-node pairs, so
    it is INVARIANT under any monotone reparametrization of the time axis. This is
    the gauge fact the adaptive strategy exploits — see plans/mesh/00-primer.md.
    """
    N = len(x_grid)
    M = len(t_grid) - 1
    kappa_g = _kappa_g(g, x_grid)                           # (N-2, M+1)
    f_diff = f[1:N - 1, 1:] - f[1:N - 1, :-1]              # (N-2, M)
    return float(np.sum(f_diff * kappa_g[:, :M]))


def harvest_per_interval(f, g, x_grid, t_grid):
    """Harvest attributed to each time-interval j: dJ_t[j] = sum_i f_diff[i,j]*kappa_g[i,j].

    sum(harvest_per_interval) == compute_J exactly. Its running cumulative sum,
    normalized to [0,1], is the harvest-gauge CDF tau(t). Shape (M,).
    """
    N = len(x_grid)
    M = len(t_grid) - 1
    kappa_g = _kappa_g(g, x_grid)                           # (N-2, M+1)
    f_diff = f[1:N - 1, 1:] - f[1:N - 1, :-1]              # (N-2, M)
    return np.sum(f_diff * kappa_g[:, :M], axis=0)          # (M,)

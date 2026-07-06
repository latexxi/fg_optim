"""Grid construction for the mesh optimizer.

`make_grids` is the baseline **uniform** dyadic grid copied from fg_opt3. The
adaptive (tau-gauge, band-refined) grid builders live in `adapt.py` (to be
implemented per plans/mesh/01-grids.md) — this module stays the uniform baseline
so the two are directly comparable.
"""
import numpy as np


def make_grids(k: int, M: int):
    """Uniform dyadic grid.

    Parameters
    ----------
    k : int
        Dyadic level. Spatial grid has N = 2^k + 1 points on [-1, 1].
    M : int
        Number of time intervals. Time grid has M+1 points on [0, 1].

    Returns
    -------
    x_grid : ndarray, shape (N,)
    t_grid : ndarray, shape (M+1,)
    """
    N = 2 ** k + 1
    x_grid = np.linspace(-1.0, 1.0, N)
    t_grid = np.linspace(0.0, 1.0, M + 1)
    return x_grid, t_grid

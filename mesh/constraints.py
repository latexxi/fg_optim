"""LP constraint matrices for f or g. Copied verbatim from fg_opt3 (test block dropped).

Key facts the adaptive strategy relies on (all verifiable below):
  * slope rows read h = dx  -> non-uniform x fully supported already
  * convexity rows read dx  -> non-uniform x fully supported already
  * monotonicity-in-t rows have RHS 0 and coeffs +-1, NO dt -> time-node PLACEMENT
    is completely transparent to the LP. Move time nodes anywhere: constraints and
    J are blind to spacing. This is what makes the tau-gauge regrid free.
"""
import numpy as np
from scipy.sparse import csr_matrix


def idx(i, j, M):
    """Flat index for node (i, j) in row-major order: i*(M+1) + j."""
    return i * (M + 1) + j


def build_constraints(x_grid, t_grid, monotone_increasing: bool):
    """Equality + inequality constraint matrices for f (monotone_increasing=True)
    or g (False).

    Returns
    -------
    A_eq, b_eq, A_ub, b_ub  (A_eq/A_ub are csr_matrix)
    """
    N = len(x_grid)
    M = len(t_grid) - 1
    n_vars = N * (M + 1)

    # ------- Equality: boundary columns x=+-1 for all t, + terminal/initial slice
    eq_rows_idx, eq_cols_idx, eq_data = [], [], []
    row_eq = 0
    for j in range(M + 1):
        eq_rows_idx.append(row_eq); eq_cols_idx.append(idx(0, j, M)); eq_data.append(1.0); row_eq += 1
        eq_rows_idx.append(row_eq); eq_cols_idx.append(idx(N - 1, j, M)); eq_data.append(1.0); row_eq += 1
    j_bc = M if monotone_increasing else 0
    for i in range(N):
        eq_rows_idx.append(row_eq); eq_cols_idx.append(idx(i, j_bc, M)); eq_data.append(1.0); row_eq += 1
    n_eq = row_eq
    A_eq = csr_matrix((eq_data, (eq_rows_idx, eq_cols_idx)), shape=(n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    # ------- Inequality: slope |.|<=1, convexity in x, monotonicity in t
    ub_rows_idx, ub_cols_idx, ub_data, ub_rhs = [], [], [], []
    row_ub = 0

    # slope: |f[i+1,j]-f[i,j]| <= h  (RHS = dx)
    for i in range(N - 1):
        h = x_grid[i + 1] - x_grid[i]
        c_lo = idx(i, 0, M)
        c_hi = idx(i + 1, 0, M)
        for j in range(M + 1):
            ub_rows_idx += [row_ub, row_ub]; ub_cols_idx += [c_hi + j, c_lo + j]; ub_data += [1.0, -1.0]; ub_rhs.append(h); row_ub += 1
            ub_rows_idx += [row_ub, row_ub]; ub_cols_idx += [c_lo + j, c_hi + j]; ub_data += [1.0, -1.0]; ub_rhs.append(h); row_ub += 1

    # convexity in x: (f[i]-f[i-1])/hl - (f[i+1]-f[i])/hr <= 0
    for i in range(1, N - 1):
        h_left = x_grid[i] - x_grid[i - 1]
        h_right = x_grid[i + 1] - x_grid[i]
        inv_l, inv_r = 1.0 / h_left, 1.0 / h_right
        c_im1, c_i, c_ip1 = idx(i - 1, 0, M), idx(i, 0, M), idx(i + 1, 0, M)
        for j in range(M + 1):
            ub_rows_idx += [row_ub] * 3
            ub_cols_idx += [c_i + j, c_im1 + j, c_ip1 + j]
            ub_data += [inv_l + inv_r, -inv_l, -inv_r]
            ub_rhs.append(0.0); row_ub += 1

    # monotonicity in t: RHS 0, coeffs +-1, NO dt
    for i in range(N):
        c_i = idx(i, 0, M)
        for j in range(M):
            if monotone_increasing:
                ub_rows_idx += [row_ub, row_ub]; ub_cols_idx += [c_i + j, c_i + j + 1]; ub_data += [1.0, -1.0]
            else:
                ub_rows_idx += [row_ub, row_ub]; ub_cols_idx += [c_i + j + 1, c_i + j]; ub_data += [1.0, -1.0]
            ub_rhs.append(0.0); row_ub += 1

    n_ub = row_ub
    A_ub = csr_matrix((ub_data, (ub_rows_idx, ub_cols_idx)), shape=(n_ub, n_vars))
    b_ub = np.array(ub_rhs)
    return A_eq, b_eq, A_ub, b_ub


def check_feasible(v, A_eq, b_eq, A_ub, b_ub, tol=1e-9):
    eq_ok = np.allclose(np.asarray(A_eq @ v).ravel(), b_eq, atol=tol)
    ub_ok = np.all(np.asarray(A_ub @ v).ravel() <= b_ub + tol)
    return eq_ok, ub_ok

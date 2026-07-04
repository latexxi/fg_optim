"""Weight LPs: J is linear in each family's weights with positions frozen."""

import numpy as np
from scipy.optimize import linprog

from .geometry import hat_matrix, conv_eval, _wbounds


def lp_weights_f(XI, B, ETA, ub=None, lip_rhs=None, rise_cap=None):
    """Globally optimal f-weights A given all positions and g-weights.
    `ub` (Np1, Kf), if given, upper-bounds each weight per time node -- a 0
    entry pins a kink dead there (lifetime windows, Task B); default None is
    unbounded (original behaviour).

    Run-13 cell (§5A) boundary-data channels, both default-None (bit-for-bit
    unchanged when omitted):
      `lip_rhs` (Np1, 2): per-node (left, right) arm-slope caps, replacing the
        constant 1.0 -- the residual Lipschitz slack the parent leaves the child
        (channel 1). Entries must be <= 1.
      `rise_cap` = (xs, rho): xs (P,) sample x-positions, rho (P,) caps; adds
        the explicit rise-budget rows -f(xs, t_k) = sum_i A[k,i] hat(xs; XI[k,i])
        <= rho, at EVERY node k (channel 2 -- the residual rise budget; the
        `1-|x|` cap is otherwise only implicit, so a partially-spent parent has
        no other way to be expressed). rho is already in the child's frame
        (rho/r rescaling, A2, is the caller's responsibility)."""
    Np1, Kf = XI.shape
    N = Np1 - 1
    nv = Np1 * Kf

    c = np.zeros((Np1, Kf))
    for k in range(N):
        em = 0.5 * (ETA[k] + ETA[k + 1])
        bm = 0.5 * (B[k] + B[k + 1])
        jm = 2.0 * bm / (1.0 - em ** 2)
        Hk, Hk1 = hat_matrix(em, XI[k]), hat_matrix(em, XI[k + 1])
        # J step = jm . [ (A[k] . h_k) - (A[k+1] . h_{k+1}) ]   (signs from f=-sum a h)
        c[k] += jm @ Hk
        c[k + 1] -= jm @ Hk1
    c = c.reshape(nv)

    blocks, rhs = [], []
    # Lipschitz (exact): boundary slopes of the convex PL function
    for k in range(Np1):
        r = np.zeros((2, Np1, Kf))
        r[0, k] = 1.0 / (1.0 + XI[k])
        r[1, k] = 1.0 / (1.0 - XI[k])
        blocks.append(r.reshape(2, nv))
        if lip_rhs is None:
            rhs.extend([1.0, 1.0])
        else:
            rhs.extend([float(lip_rhs[k, 0]), float(lip_rhs[k, 1])])
    # monotone rise f_t>=0 (exact at union of kinks)
    for k in range(N):
        xchk = np.concatenate([XI[k], XI[k + 1]])
        Hk, Hk1 = hat_matrix(xchk, XI[k]), hat_matrix(xchk, XI[k + 1])  # (2Kf,Kf)
        block = np.zeros((2 * Kf, Np1, Kf))
        block[:, k + 1, :] = Hk1     # f^{k+1}-f^k>=0  <=>  A.h terms
        block[:, k, :] = -Hk
        blocks.append(block.reshape(2 * Kf, nv))
        rhs.extend([0.0] * (2 * Kf))
    # rise budget (§5A channel 2): -f(xs, t_k) <= rho(xs) at every node
    if rise_cap is not None:
        xs, rho = np.asarray(rise_cap[0], float), np.asarray(rise_cap[1], float)
        P = xs.shape[0]
        for k in range(Np1):
            block = np.zeros((P, Np1, Kf))
            block[:, k, :] = hat_matrix(xs, XI[k])   # (P, Kf); sum_i A[k,i] hat
            blocks.append(block.reshape(P, nv))
            rhs.extend(rho.tolist())

    bounds = _wbounds((Np1, Kf), dead_node=N, ub=ub)
    res = linprog(-c, A_ub=np.vstack(blocks), b_ub=np.array(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("f-LP failed: " + res.message)
    return res.x.reshape(Np1, Kf)


def lp_weights_g(A, XI, ETA, ub=None, lip_rhs=None):
    """Globally optimal g-weights B given all positions and f-weights.
    `ub` (Np1, Kg), if given, upper-bounds each weight per time node (0 = dead
    there), imposing lifetime windows (Task B); default None is unbounded.

    `lip_rhs` (Np1, 2): Run-13 cell (§5A) per-node (left, right) arm-slope caps
    replacing the constant 1.0 (default None = unchanged, bit-for-bit)."""
    Np1, Kg = ETA.shape
    N = Np1 - 1
    nv = Np1 * Kg

    c = np.zeros((Np1, Kg))
    for k in range(N):
        em = 0.5 * (ETA[k] + ETA[k + 1])
        df = conv_eval(em, A[k + 1], XI[k + 1]) - conv_eval(em, A[k], XI[k])
        coef = df / (1.0 - em ** 2)           # j = (B_k+B_{k+1})/(1-em^2)
        c[k] += coef
        c[k + 1] += coef
    c = c.reshape(nv)

    blocks, rhs = [], []
    for k in range(Np1):
        r = np.zeros((2, Np1, Kg))
        r[0, k] = 1.0 / (1.0 + ETA[k])
        r[1, k] = 1.0 / (1.0 - ETA[k])
        blocks.append(r.reshape(2, nv))
        if lip_rhs is None:
            rhs.extend([1.0, 1.0])
        else:
            rhs.extend([float(lip_rhs[k, 0]), float(lip_rhs[k, 1])])
    # g_t <= 0  (g deepens): sum B^k h^k - sum B^{k+1} h^{k+1} <= 0
    for k in range(N):
        xchk = np.concatenate([ETA[k], ETA[k + 1]])
        Hk, Hk1 = hat_matrix(xchk, ETA[k]), hat_matrix(xchk, ETA[k + 1])
        block = np.zeros((2 * Kg, Np1, Kg))
        block[:, k, :] = Hk
        block[:, k + 1, :] = -Hk1
        blocks.append(block.reshape(2 * Kg, nv))
        rhs.extend([0.0] * (2 * Kg))

    bounds = _wbounds((Np1, Kg), dead_node=0, ub=ub)
    res = linprog(-c, A_ub=np.vstack(blocks), b_ub=np.array(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("g-LP failed: " + res.message)
    return res.x.reshape(Np1, Kg)

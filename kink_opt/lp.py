"""Weight LPs: J is linear in each family's weights with positions frozen."""

import numpy as np
from scipy.optimize import linprog

from .geometry import hat_matrix, conv_eval, _wbounds


def lp_weights_f(XI, B, ETA, ub=None):
    """Globally optimal f-weights A given all positions and g-weights.
    `ub` (Np1, Kf), if given, upper-bounds each weight per time node -- a 0
    entry pins a kink dead there (lifetime windows, Task B); default None is
    unbounded (original behaviour)."""
    Np1, Kf = XI.shape
    N = Np1 - 1
    nv = Np1 * Kf
    ix = lambda k, i: k * Kf + i

    c = np.zeros(nv)
    for k in range(N):
        em = 0.5 * (ETA[k] + ETA[k + 1])
        bm = 0.5 * (B[k] + B[k + 1])
        jm = 2.0 * bm / (1.0 - em ** 2)
        Hk, Hk1 = hat_matrix(em, XI[k]), hat_matrix(em, XI[k + 1])
        # J step = jm . [ (A[k] . h_k) - (A[k+1] . h_{k+1}) ]   (signs from f=-sum a h)
        for i in range(Kf):
            c[ix(k, i)] += float((jm * Hk[:, i]).sum())
            c[ix(k + 1, i)] -= float((jm * Hk1[:, i]).sum())

    rows, rhs = [], []
    # Lipschitz (exact): boundary slopes of the convex PL function
    for k in range(Np1):
        for denom in (1.0 + XI[k], 1.0 - XI[k]):
            r = np.zeros(nv)
            r[[ix(k, i) for i in range(Kf)]] = 1.0 / denom
            rows.append(r); rhs.append(1.0)
    # monotone rise f_t>=0 (exact at union of kinks)
    for k in range(N):
        xchk = np.concatenate([XI[k], XI[k + 1]])
        Hk, Hk1 = hat_matrix(xchk, XI[k]), hat_matrix(xchk, XI[k + 1])
        for p in range(len(xchk)):
            r = np.zeros(nv)
            for i in range(Kf):
                r[ix(k + 1, i)] += Hk1[p, i]   # f^{k+1}-f^k>=0  <=>  A.h terms
                r[ix(k, i)] -= Hk[p, i]
            rows.append(r); rhs.append(0.0)

    bounds = _wbounds((Np1, Kf), dead_node=N, ub=ub)
    res = linprog(-c, A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("f-LP failed: " + res.message)
    return res.x.reshape(Np1, Kf)


def lp_weights_g(A, XI, ETA, ub=None):
    """Globally optimal g-weights B given all positions and f-weights.
    `ub` (Np1, Kg), if given, upper-bounds each weight per time node (0 = dead
    there), imposing lifetime windows (Task B); default None is unbounded."""
    Np1, Kg = ETA.shape
    N = Np1 - 1
    nv = Np1 * Kg
    ix = lambda k, m: k * Kg + m

    c = np.zeros(nv)
    for k in range(N):
        em = 0.5 * (ETA[k] + ETA[k + 1])
        df = conv_eval(em, A[k + 1], XI[k + 1]) - conv_eval(em, A[k], XI[k])
        coef = df / (1.0 - em ** 2)           # j = (B_k+B_{k+1})/(1-em^2)
        for m in range(Kg):
            c[ix(k, m)] += coef[m]
            c[ix(k + 1, m)] += coef[m]

    rows, rhs = [], []
    for k in range(Np1):
        for denom in (1.0 + ETA[k], 1.0 - ETA[k]):
            r = np.zeros(nv)
            r[[ix(k, m) for m in range(Kg)]] = 1.0 / denom
            rows.append(r); rhs.append(1.0)
    # g_t <= 0  (g deepens): sum B^k h^k - sum B^{k+1} h^{k+1} <= 0
    for k in range(N):
        xchk = np.concatenate([ETA[k], ETA[k + 1]])
        Hk, Hk1 = hat_matrix(xchk, ETA[k]), hat_matrix(xchk, ETA[k + 1])
        for p in range(len(xchk)):
            r = np.zeros(nv)
            for m in range(Kg):
                r[ix(k, m)] += Hk[p, m]
                r[ix(k + 1, m)] -= Hk1[p, m]
            rows.append(r); rhs.append(0.0)

    bounds = _wbounds((Np1, Kg), dead_node=0, ub=ub)
    res = linprog(-c, A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("g-LP failed: " + res.message)
    return res.x.reshape(Np1, Kg)

"""Basis geometry: the hat function, convex-PL evaluation, weight-LP bounds."""

import numpy as np

MARGIN = 0.03      # keep kinks inside (-1+MARGIN, 1-MARGIN)
GAP = 0.02         # minimal spacing between same-family kinks
PEN_W = 200.0      # penalty weight in the position NLP


def hat_matrix(x, xi):
    """H[p,i] = hat(x[p]; xi[i]),  hat(+-1)=0, hat(node)=1."""
    x = np.atleast_1d(np.asarray(x, float))
    xi = np.atleast_1d(np.asarray(xi, float))
    L = (1.0 + x[:, None]) / (1.0 + xi[None, :])
    R = (1.0 - x[:, None]) / (1.0 - xi[None, :])
    return np.clip(np.where(x[:, None] <= xi[None, :], L, R), 0.0, None)


def conv_eval(x, w, xi):
    """F(x) = - sum_i w_i hat(x; xi_i): convex PL, F<=0, F(+-1)=0."""
    w = np.atleast_1d(w)
    if w.size == 0:
        return np.zeros(np.atleast_1d(x).shape)
    return -(hat_matrix(x, xi) * w[None, :]).sum(axis=1)


def _wbounds(shape, dead_node, ub=None):
    """Box bounds for a weight LP over a (Np1, K) weight matrix, flattened in
    the same (k outer, kink inner) order the LPs index with.  `dead_node` is
    the time index whose weights are pinned to 0 by a boundary condition
    (k=N for f's terminal f(x,1)=0, k=0 for g's initial g(x,0)=0).  `ub` is an
    optional (Np1, K) per-node upper bound (np.inf = unbounded); nodes with
    ub=0 are 'dead', which is how a kink's lifetime window is imposed.  With
    ub=None every non-dead node is unbounded, reproducing the original LPs
    exactly (so all-alive topologies regress to the previous behaviour)."""
    Np1, K = shape
    if ub is None:
        ub = np.full(shape, np.inf)
    bounds = []
    for k in range(Np1):
        for i in range(K):
            hi = 0.0 if k == dead_node else float(ub[k, i])
            bounds.append((0.0, hi if np.isfinite(hi) else None))
    return bounds

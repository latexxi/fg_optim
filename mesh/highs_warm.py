"""Persistent HiGHS LP instance with basis warm-starting. Copied verbatim from fg_opt3.

When only the objective vector changes between solves (constraints fixed), passing
the previous basis to HiGHS lets simplex start primal-feasible and only need
dual-simplex pivots to restore dual feasibility — typically O(1) pivots.

    warm = HiGHSWarmLP(A_ub, b_ub, A_eq, b_eq, n_vars)
    x = warm.solve(c)      # cold solve, saves basis
    x = warm.solve(c_new)  # warm solve from saved basis
"""
import numpy as np
import highspy
from scipy.sparse import vstack


class HiGHSWarmLP:
    """Persistent LP:  max c @ x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  x unbounded."""

    def __init__(self, A_ub, b_ub, A_eq, b_eq, n_vars):
        self._n_vars = n_vars
        self._col_idx = np.arange(n_vars, dtype=np.int32)
        self._basis = None

        h = highspy.Highs()
        h.silent()
        h.setOptionValue("presolve", "on")
        h.setOptionValue("solver", "simplex")   # simplex supports basis warm-start
        self._h = h

        inf = h.getInfinity()
        h.addVars(n_vars,
                  np.full(n_vars, -inf, dtype=np.float64),
                  np.full(n_vars, inf, dtype=np.float64))

        A_all = vstack([A_ub, A_eq], format='csr').astype(np.float64)
        n_ub = A_ub.shape[0]
        lb_all = np.concatenate([np.full(n_ub, -inf, dtype=np.float64),
                                 b_eq.astype(np.float64)])
        ub_all = np.concatenate([b_ub.astype(np.float64),
                                 b_eq.astype(np.float64)])
        h.addRows(A_all.shape[0], lb_all, ub_all,
                  A_all.nnz,
                  A_all.indptr.astype(np.int32),
                  A_all.indices.astype(np.int32),
                  A_all.data)

    def solve(self, c):
        """Maximize c @ x subject to the fixed constraints. Returns x (n_vars,)."""
        h = self._h
        h.changeColsCost(self._n_vars, self._col_idx, -c.astype(np.float64))
        if self._basis is not None and self._basis.valid:
            h.setBasis(self._basis)
        h.run()
        status = h.getModelStatus()
        sol = h.getSolution()
        if not sol.value_valid:
            raise RuntimeError(f"HiGHS LP failed: {h.modelStatusToString(status)}")
        self._basis = h.getBasis()
        return np.array(sol.col_value, dtype=np.float64)

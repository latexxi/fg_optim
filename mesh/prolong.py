"""Feasible-by-construction warm start on the adaptive grid.

Moves a solved (f, g) from its old (x_old, t_old) grid onto a new (x_new, t_new)
grid in two J-neutral steps (see plans/mesh/00-primer.md §0.4, §0.7 and
plans/mesh/02-prolong.md):

  regauge_time — resample in t (monotone time remap is a pure gauge change, J and
                  feasibility blind to spacing).
  prolong_x    — insert new x-nodes by linear interpolation (colinear insert has
                  zero discrete curvature -> harvest-neutral).

No LP surgery here; the LP still has to decide whether to bend the new band
strands, that's the next generation's job, not this file's.
"""
import numpy as np

from .constraints import build_constraints, check_feasible


def regauge_time(field, t_old, t_new):
    """Resample each x-row of `field` (shape (N, len(t_old))) from t_old onto t_new.

    Returns (N, len(t_new)).

    IMPORTANT: this resamples by *relative index position*, not by t_old/t_new's
    raw coordinate values. `build_constraints` and `compute_J` never read a
    t_grid's actual values (only `len(t_grid)` -> M) -- t is pure gauge (primer
    fact 1). Interpolating field values against the physical t-coordinate would
    silently inject spurious quadrature error: compute_J's harvest sum has no
    dt weighting (`f_diff * kappa_g[:, :M]`, a plain left-endpoint discrete sum,
    not a trapezoidal integral), so resampling a genuinely-varying field at new
    physical t-positions changes the *value* of that discrete sum even though
    the underlying continuous integral is retiming-invariant (confirmed
    empirically: naive np.interp(t_new, t_old, ...) on a real solved (f, g)
    drifts J by ~15%, and even a 1e-4 node jitter drifts J by ~3e-3 -- far above
    roundoff, so this is a real quadrature-order mismatch, not a bug in the
    resampling arithmetic).

    Resampling by index fraction sidesteps this: it only asks "where does this
    new node sit among the old node COUNT", never "what physical t-distance
    away is it". When len(t_new) == len(t_old) (the only case tau_regrid is
    ever called with in this project -- M_new always defaults to M), the index
    grid is unchanged (new_idx == old_idx exactly), so this reduces to an exact
    identity: the field values don't need to change at all, because t truly is
    gauge and nothing downstream reads it except as a label. When the counts
    differ, this falls back to a uniform stretch/compress over index space
    (still ignoring t_old/t_new's raw values, for the same reason).

    Feasibility preserved:
      * monotone in t: linear interp (by index or by value, same convex-
        combination argument) of a monotone sequence is monotone -> f_t>=0 /
        g_t<=0 survive.
      * convex in x / |slope|<=1: these are per-time-slice properties; each new
        slice is a convex combination (interp weight lam in [0,1]) of two old
        slices that each satisfy them, and both constraints are preserved under
        convex combination -> survive.
      * boundary zeros: rows i=0 and i=N-1 are all-zero in, all-zero out. The
        first/last new index always maps exactly to the first/last old index,
        so the terminal f(x,1)=0 / initial g(x,0)=0 slices are copied exactly.
    J-neutral: J is gauge-invariant under monotone time remap (primer fact 1) --
    achieved here by never letting the resampling depend on t's physical
    values in the first place.
    """
    N = field.shape[0]
    M_old = len(t_old) - 1
    M_new = len(t_new) - 1
    old_idx = np.arange(M_old + 1, dtype=float)
    new_idx = np.linspace(0.0, M_old, M_new + 1)
    field_new = np.empty((N, M_new + 1))
    for i in range(N):
        field_new[i, :] = np.interp(new_idx, old_idx, field[i, :])
    return field_new


def prolong_x(field, x_old, x_new):
    """Insert the new band x-nodes by linear interpolation along x, per time column.

    field_new[:, j] = np.interp(x_new, x_old, field[:, j])   for every time col j
    Returns (len(x_new), M+1).
    """
    M1 = field.shape[1]
    n_new = len(x_new)
    field_new = np.empty((n_new, M1))
    for j in range(M1):
        field_new[:, j] = np.interp(x_new, x_old, field[:, j])
    return field_new


def adaptive_warm_start(f, g, x_old, t_old, x_new, t_new, tol=1e-6):
    """Full two-step prolongation: regauge time, then prolong x, for both fields.

    Returns (f_new, g_new) on the (x_new, t_new) grid, feasible and J-identical to
    the input (up to interpolation roundoff).

    `tol` is the feasibility-assert slack (defaults looser than `check_feasible`'s
    own 1e-9). The arms `|x|>BAND` sit EXACTLY at the Lipschitz boundary |slope|=1,
    so a linear-interp insert that is colinear in exact arithmetic can land ~1e-7
    over that inequality in float64 -- pure roundoff, not real infeasibility
    (measured: at N=391/M=1024 the worst f inequality violation is 8.7e-8, eq
    violation exactly 0; deeper grids grow it slowly). The subsequent
    `alternating_maximization` re-solve projects the warm start back to exact
    feasibility regardless, so this assert only needs to catch GROSS warm-start
    bugs (which violate by O(1e-2) or more), not boundary roundoff. At 1e-9 it
    false-positives on the deep 2-D climb (gen9+); 1e-6 keeps a 4-order margin
    below any real bug while tolerating the arm-slope roundoff.
    """
    f_t = regauge_time(f, t_old, t_new)
    g_t = regauge_time(g, t_old, t_new)
    f_new = prolong_x(f_t, x_old, x_new)
    g_new = prolong_x(g_t, x_old, x_new)

    A_eq_f, b_eq_f, A_ub_f, b_ub_f = build_constraints(x_new, t_new, True)
    A_eq_g, b_eq_g, A_ub_g, b_ub_g = build_constraints(x_new, t_new, False)
    assert all(check_feasible(f_new.ravel(), A_eq_f, b_eq_f, A_ub_f, b_ub_f, tol=tol)), \
        "adaptive_warm_start: prolonged f infeasible"
    assert all(check_feasible(g_new.ravel(), A_eq_g, b_eq_g, A_ub_g, b_ub_g, tol=tol)), \
        "adaptive_warm_start: prolonged g infeasible"

    return f_new, g_new


if __name__ == "__main__":
    from mesh import make_grids, alternating_maximization, compute_J
    from mesh.adapt import tau_regrid, band_refine, BAND

    x, t = make_grids(4, 32)
    g0 = np.array([[0.5 * t[j] * (x[i] ** 2 - 1) for j in range(len(t))] for i in range(len(x))])
    f, g, _ = alternating_maximization(x, t, g_init=g0, max_iter=40)
    J0 = compute_J(f, g, x, t)

    t_new = tau_regrid(f, g, x, t)          # regauge only (same M)
    x_new = band_refine(x)                  # +band strands
    f2, g2 = adaptive_warm_start(f, g, x, t, x_new, t_new)

    J1 = compute_J(f2, g2, x_new, t_new)
    print(f"J parent={J0:.6f}  warm-start={J1:.6f}  drift={abs(J1-J0):.2e}")
    assert abs(J1 - J0) < 1e-6, "warm start not J-neutral -> bug (facts 1 & 3)"
    # feasibility already asserted inside adaptive_warm_start
    print("warm start feasible + J-neutral: OK")

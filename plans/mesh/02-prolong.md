# 02 — Warm start on the new grid (`mesh/prolong.py`)

Read `00-primer.md` (esp. §0.4 fact 3, §0.7 caveat) and finish `01-grids.md` first.

Build the feasible-by-construction initial guess when moving a solved `(f,g)` from
its old `(x_old, t_old)` grid onto the new adaptive `(x_new, t_new)`. Two steps:
**regauge time**, then **prolong x**. New file `mesh/prolong.py`.

The invariant to preserve: the warm start must be J-identical to the parent and
feasible before any LP runs (§0.7 — the LP supplies the gain, not the warm start).

## 2.1 `regauge_time`

```python
def regauge_time(field, t_old, t_new):
    """Resample each x-row of `field` (shape (N, len(t_old))) from t_old onto t_new.

    field_new[i, :] = np.interp(t_new, t_old, field[i, :])   for every row i
    Returns (N, len(t_new)).

    Feasibility preserved:
      * monotone in t: linear interp of a monotone sequence is monotone -> f_t>=0 /
        g_t<=0 survive.
      * convex in x / |slope|<=1: these are per-time-slice properties; each new
        slice is a convex combination (interp weight lam in [0,1]) of two old slices
        that each satisfy them, and both constraints are preserved under convex
        combination -> survive.
      * boundary zeros: rows i=0 and i=N-1 are all-zero in, all-zero out. Terminal
        f(x,1)=0 / initial g(x,0)=0 survive because t_new keeps the endpoints
        (t_new[0]=0, t_new[-1]=1 from task 01), so those exact slices are copied.
    J-neutral: J is gauge-invariant under monotone time remap (primer fact 1).
    """
```

Order of operations: regauge time FIRST (on the old x-grid, cheap), THEN prolong x.
Doing time first keeps the arrays small during the interp.

## 2.2 `prolong_x`

```python
def prolong_x(field, x_old, x_new):
    """Insert the new band x-nodes by linear interpolation along x, per time column.

    field_new[:, j] = np.interp(x_new, x_old, field[:, j])   for every time col j
    Returns (len(x_new), M+1).

    Because x_new ⊇ x_old (task 01 union), old nodes copy EXACTLY (np.interp returns
    the exact sample at coincident points) and only the inserted band midpoints are
    interpolated. A linear-interp midpoint is colinear with its neighbors -> its
    discrete g_xx (kink) is exactly 0 -> harvest-neutral insert (primer fact 3).
    Feasibility: linear interp along x preserves convexity, |slope|<=1, and the
    x=±1 boundary zeros (endpoints are in x_old, copied exactly).
    """
```

## 2.3 `adaptive_warm_start`

```python
def adaptive_warm_start(f, g, x_old, t_old, x_new, t_new):
    """Full two-step prolongation: regauge time, then prolong x, for both fields.

    Returns (f_new, g_new) on the (x_new, t_new) grid, feasible and J-identical to
    the input (up to interpolation roundoff).

    f_t = regauge_time(f, t_old, t_new);  g_t = regauge_time(g, t_old, t_new)
    f_new = prolong_x(f_t, x_old, x_new); g_new = prolong_x(g_t, x_old, x_new)

    MUST assert feasibility before returning (primer §0.8):
      A_eq_f,b_eq_f,A_ub_f,b_ub_f = build_constraints(x_new, t_new, True)
      A_eq_g,b_eq_g,A_ub_g,b_ub_g = build_constraints(x_new, t_new, False)
      assert all(check_feasible(f_new.ravel(), A_eq_f,b_eq_f,A_ub_f,b_ub_f))
      assert all(check_feasible(g_new.ravel(), A_eq_g,b_eq_g,A_ub_g,b_ub_g))
    """
```

## 2.4 Acceptance check (`if __name__ == "__main__":`)

```python
from mesh import make_grids, alternating_maximization, compute_J, build_constraints, check_feasible
from mesh.adapt import tau_regrid, band_refine, BAND

x, t = make_grids(4, 32)
g0 = np.array([[0.5*t[j]*(x[i]**2-1) for j in range(len(t))] for i in range(len(x))])
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
```

The `drift < 1e-6` assert is the load-bearing check: if it fails, either the time
regauge isn't monotone-preserving or a band midpoint picked up spurious curvature —
both violate the primer's gauge facts and must be fixed here, not masked in the driver.

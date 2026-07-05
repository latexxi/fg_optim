# 02 — D1 verification: is the arm-only slope channel sufficient?

**Prereq:** read `00-primer.md`. **Depends on:** nothing but 00 (independent of 01).
**File to edit:** `kink_opt/cell.py`. **Deliverable:** `check_interior_slope`.

## Why this exists (design fork D1)

Channel 1 injects the slope budget `β` only at the **arms** `x=±1` (the LP's
Lipschitz rows are arm-slope rows; `env_to_lp` uses a scalar cap `min(β)`). But `β`
is a profile over *interior* x. For a convex piecewise-linear `f`, the maximum
`|f_x|` occurs at an arm, so **arm-only may suffice** — but only if the parent's
residue never has interior slope exceeding its arm slope. If it can, the child could
locally violate the inherited Lipschitz budget and the whole coupling is unsound.

This task builds a cheap assertion that the arm-only assumption holds on the actual
environments the loop visits. It is a **trust gate**, not new physics — but if it
fires, tasks 03/04's verdict is untrustworthy until channel 1 is upgraded to carry
interior slope rows.

## `check_interior_slope(sol, family="f", tol=1e-9) -> dict`

For the solved cell's f-slice at the midpoint read node, check that the maximum
interior `|f_x|` does not exceed the arm `|f_x|` (beyond `tol`).

```python
def check_interior_slope(sol, family="f", tol=1e-9):
    """D1 gate: verify max interior |f_x| <= max arm |f_x| on a solved cell's
    read slice, so the arm-only slope channel (env_to_lp) faithfully caps the
    residue. Returns dict(ok, max_interior, arm_left, arm_right, margin)."""
    from .melt import _slope_f
    t = sol["t"]
    idx = int(np.argmin(np.abs(t - 0.5)))
    W = sol["A"] if family == "f" else sol["B"]
    P = sol["XI"] if family == "f" else sol["ETA"]
    a, xi = W[idx], P[idx]
    xs = np.linspace(-1.0 + MARGIN, 1.0 - MARGIN, 401)
    interior = np.abs(_slope_f(xs, a, xi)).max()
    arm_left = np.abs(_slope_f(np.array([-1.0 + MARGIN]), a, xi)[0])
    arm_right = np.abs(_slope_f(np.array([1.0 - MARGIN]), a, xi)[0])
    arm = max(arm_left, arm_right)
    return dict(ok=bool(interior <= arm + tol), max_interior=float(interior),
                arm_left=float(arm_left), arm_right=float(arm_right),
                margin=float(arm - interior))
```

## Acceptance gate

Run `check_interior_slope` on a solved cell for a few environments the loop will
actually see, and report:

1. On the flat seed: `sol = cell_solve(flat_env())["sol"]`; print the dict. Expect
   `ok=True` (a single convex tent's slope is maximal at an arm).
2. On a spent environment: build one via task 01's read-off fed back in — e.g.
   `E1 = cell_read_env(cell_solve(flat_env())["sol"], r=0.5)`;
   `sol2 = cell_solve(E1)["sol"]`; print `check_interior_slope(sol2)`. Expect
   `ok=True`.

If both pass, record in the parent plan (`plans/run13-selfreproducing-cell.md` §5A,
fork D1) that arm-only is verified sufficient for these environments and no interior
slope rows are needed. **If either fails**, do NOT proceed to trusting task 04's
verdict — instead note it and escalate: channel 1 needs per-interior-x slope rows
(analogous to the `rise_cap` block, but on slope), which is a larger change spec'd as
future work in the parent plan's §5A D1.

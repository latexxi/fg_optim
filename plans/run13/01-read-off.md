# 01 — E′ read-off + environment distance

**Prereq:** read `00-primer.md` (esp. §0.6). **Depends on:** nothing but 00.
**File to edit:** `kink_opt/cell.py`. **Deliverables:** `cell_read_env`, `cell_env_distance`.

## Goal

Given a solved cell (`sol` from `cell_solve(...)["sol"]`), read the **outgoing
environment** `E′ = dict(x_hat, beta, rho, r)` that the next octave's cell inherits,
and a distance function that says how close two environments are (the
self-reproduction metric for the fixed-point loop).

## 1.1 `cell_read_env(sol, r, n_sample=41, family="f") -> dict`

Read at the temporal midpoint, in the next octave's co-moving spatial frame. Exact
slope, no finite differencing.

```python
def cell_read_env(sol, r, n_sample=41, family="f"):
    """Outgoing environment E' of a solved cell (the input to the next octave).
    Reads the residual slope-slack beta(x_hat) and remaining-rise rho(x_hat) at
    the cell's temporal MIDPOINT (t_hat = 1/2; NOT t_hat=1, which is terminal-
    pinned -- see 00-primer.md §0.6), sampled over the next octave's support
    (half-width w_next = r about the carrier's midpoint position). rho is stored
    PHYSICAL (env_to_lp divides by r at the next injection). Returns a dict with
    the same schema as flat_env: dict(x_hat, beta, rho, r)."""
    from .melt import _slope_f           # exact PL slope; already in the repo
    t = sol["t"]
    idx = int(np.argmin(np.abs(t - 0.5)))          # t_hat = 1/2 read node
    W = sol["A"] if family == "f" else sol["B"]
    P = sol["XI"] if family == "f" else sol["ETA"]
    a, xi = W[idx], P[idx]
    c = float(xi[np.argmax(a)]) if a.size else 0.0  # carrier position (heaviest kink)
    x_hat = np.linspace(-1.0 + MARGIN, 1.0 - MARGIN, n_sample)
    x_abs = np.clip(c + r * x_hat, -1.0 + MARGIN, 1.0 - MARGIN)
    slope = _slope_f(x_abs, a, xi)
    beta = 1.0 - np.abs(slope)
    rho = (1.0 - np.abs(x_abs)) + conv_eval(x_abs, a, xi)
    return dict(x_hat=x_hat, beta=beta, rho=rho, r=float(r))
```

Notes:
- `conv_eval` and `MARGIN` are already imported at the top of `cell.py`. Add the
  `_slope_f` import (local import inside the function is fine, as shown, to avoid any
  import-cycle worry).
- The carrier is a single kink (Kf=1), so `c` is just that kink's position; the
  `argmax(a)` generalizes harmlessly if you ever raise Kf. If `a` is all-zero
  (degenerate), `c=0.0` is a safe fallback.
- `beta` may exceed 1 or go slightly negative only through numerical noise — do **not**
  clip it here; the injection path (`env_to_lp`) already takes `min(β, 1.0)`. Clipping
  here would hide a real problem.
- Keep `n_sample` equal to the incoming env's `n_sample` (default 41 matches
  `flat_env`) so `cell_env_distance` sees a common `x_hat` grid.

## 1.2 `cell_env_distance(Ea, Eb) -> float`

Max-norm over the concatenated `(β, ρ/r)` profiles. The **`ρ/r` normalization is
load-bearing** (assumption A2, §0.6): comparing raw `ρ` lets a pure change of frame
fabricate a fixed point. `β` is compared directly (frame-invariant).

```python
def cell_env_distance(Ea, Eb):
    """Self-reproduction metric between two environments on the SAME x_hat grid.
    Max-norm of concatenated (beta, rho/r) differences -- rho is normalized by
    the frame contraction r (A2) so a change of frame alone cannot fake a match.
    Raises if the x_hat grids differ."""
    if not np.allclose(Ea["x_hat"], Eb["x_hat"]):
        raise ValueError("cell_env_distance requires a common x_hat grid")
    da = np.concatenate([Ea["beta"], Ea["rho"] / max(Ea["r"], 1e-12)])
    db = np.concatenate([Eb["beta"], Eb["rho"] / max(Eb["r"], 1e-12)])
    return float(np.max(np.abs(da - db)))
```

(This mirrors `melt.env_distance` but adds the `/r` normalization and drops the
band-spec coupling. Do not modify `melt.env_distance` — Run 12 depends on it.)

## Acceptance gate

Add a small `__main__`-style check (either extend `cell.py`'s `__main__` or write
`scratchpad`-local script) and show its output:

1. **Schema match.** `E1 = cell_read_env(cell_solve(flat_env())["sol"], r=0.5)`.
   Assert `E1` has keys `{x_hat, beta, rho, r}`, all arrays length `n_sample`,
   `E1["r"] == 0.5`, `x_hat` matches `flat_env(n_sample=41)["x_hat"]` shape.
2. **Finite & sane.** `np.isfinite(E1["beta"]).all()`, `np.isfinite(E1["rho"]).all()`,
   and `β ∈ [-0.05, 1.05]` (small slack for numerical noise) — print min/max of each.
3. **Distance zero to self.** `cell_env_distance(E1, E1) == 0.0`.
4. **Distance nonzero flat vs read.** `cell_env_distance(flat_env(), E1) > 0` (the
   solved cell has spent *some* budget, so its residue differs from the flat seed) —
   print the value. If it is 0.0, the read-off is not reading a spent state (likely you
   read the wrong node — re-check the `t_hat=1/2` index).

Report the four numbers. No verdict is produced here; this task only builds the
read-off and its metric.

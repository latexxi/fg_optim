# 01 — Grid construction (`mesh/adapt.py`)

Read `00-primer.md` first (esp. §0.4 gauge facts, §0.5 tau, §0.6 band).

Build the adaptive grids: a **tau-gauge** time grid and a **band-refined** x grid.
Pure grid arrays — no LP here. New file `mesh/adapt.py`.

## Constants

```python
BAND = 0.4   # harvest-band half-width (I2)
```

## 1.1 `tau_regrid`

```python
def tau_regrid(f, g, x_grid, t_grid, M_new=None, tau_boost=0.0):
    """Re-place time nodes to equidistribute HARVEST instead of time.

    Returns t_new (shape (M_new+1,)), a monotone grid on [0,1] with t_new[0]=0,
    t_new[-1]=1, clustered where harvest is collected.

    Method:
      dJ_t = harvest_per_interval(f, g, x_grid, t_grid)      # (M,)
      tau  = concatenate([[0], cumsum(dJ_t)]); tau /= tau[-1]  # (M+1,) monotone 0..1
      # invert the CDF: equal-harvest targets -> their t positions
      targets = linspace(0, 1, (M_new or M) + 1)
      t_new   = interp(targets, tau, t_grid)

    tau_boost > 0 optionally over-concentrates near the melt peak by blending the
    equal-harvest grid with extra density around tau*~0.38 (I3); leave 0.0 for the
    plain equal-harvest grid and only add if 03/04 show the melt event under-
    resolved.

    Guarantees: t_new[0]==0, t_new[-1]==1, strictly increasing. If dJ_t has zeros
    (dead intervals) tau is flat there and interp maps many targets into the live
    region automatically — that is the desired starving of dead time.
    """
```

Edge cases to handle:
- `tau` must be **strictly** increasing for `np.interp` to invert cleanly. If any
  `dJ_t[j] <= 0` (possible: a dead interval, or tiny negative roundoff), the CDF is
  flat/duplicated there. Fix by `dJ_t = np.clip(dJ_t, 0, None)` then add a tiny
  uniform floor `eps * (t range)` so `tau` is strictly monotone:
  `tau = cumsum(clip(dJ_t,0,None) + 1e-12); tau -= tau[0]; tau /= tau[-1]` after
  prepending 0 appropriately. Keep endpoints exact (`t_new[0]=0`, `t_new[-1]=1`).
- If the incoming solution is trivial (J≈0), fall back to a uniform grid (return
  `linspace(0,1,(M_new or M)+1)`) and don't divide by zero.

## 1.2 `band_refine`

```python
def band_refine(x_grid, band=BAND):
    """One octave of x-refinement, INSIDE THE BAND ONLY.

    Bisect every x-interval whose midpoint lies in |x| < band; leave arm intervals
    (|midpoint| >= band) alone. Returns x_new (sorted, includes all old nodes).

    mids = 0.5*(x_grid[:-1] + x_grid[1:])
    add  = mids[np.abs(mids) < band]
    x_new = np.union1d(x_grid, add)          # sorted, dedup, old nodes preserved

    Result: band strand count roughly doubles per call (D1/D2); arms untouched.
    Endpoints ±1 preserved (they're in x_grid and never added twice).
    """
```

Notes:
- `np.union1d` sorts and dedups; old nodes are exactly preserved (needed so the
  prolongation in task 02 can copy them). Endpoints stay ±1.
- Do NOT force a power-of-two N. The whole point is to break from uniform doubling;
  N grows by ~(band fraction)·(old interval count), not ×2.
- A pure-arm interval that straddles the band edge (one midpoint just outside): the
  `< band` test keeps it coarse. That's fine — the band edge is where harvest is
  already ~0. If 04's band-mass check shows leakage, widen the test to `< band*1.1`.

## 1.3 Acceptance check (put in `if __name__ == "__main__":`)

```python
# 1. tau_regrid on a real solved solution is monotone, hits endpoints, same length
from mesh import make_grids, alternating_maximization, compute_J, harvest_per_interval
x, t = make_grids(4, 32)
g0 = np.array([[0.5*t[j]*(x[i]**2-1) for j in range(len(t))] for i in range(len(x))])
f, g, _ = alternating_maximization(x, t, g_init=g0, max_iter=30)
t2 = tau_regrid(f, g, x, t)
assert t2[0] == 0 and abs(t2[-1]-1) < 1e-12 and np.all(np.diff(t2) > 0)
# tau grid should put >50% of its nodes in the live half of [0,1] near the melt
# (sanity, not exact): median node time shifts toward tau*~0.38 region
print("median t_new =", np.median(t2), " (uniform would be 0.5)")

# 2. band_refine doubles band nodes, leaves arms, preserves endpoints
xb = band_refine(x)
assert xb[0] == -1 and xb[-1] == 1
assert np.all(np.isin(x, xb))                       # old nodes preserved
n_band_old = np.sum(np.abs(x) < BAND)
n_band_new = np.sum(np.abs(xb) < BAND)
n_arm_old  = np.sum(np.abs(x) >= BAND)
n_arm_new  = np.sum(np.abs(xb) >= BAND)
assert n_arm_new == n_arm_old                       # arms untouched
print(f"band nodes {n_band_old}->{n_band_new}, arm nodes {n_arm_old} (unchanged)")
```

Both asserts pass → task done. Re-export nothing yet (adapt is used by 02/03).

# 03 — Adaptive refinement driver (`mesh/refine_adapt.py`) — REVISED

Read `00-primer.md`, finish `01-grids.md` and `02-prolong.md` first.

**This spec was rewritten after a diagnosis run falsified the first design.** Read
§3.0 before implementing — the schedule is the whole point now.

## 3.0 What the diagnosis found (why the schedule matters)

The mesh's `alternating_maximization` is coordinate ascent (max_f, then max_g) on a
bilinear objective with MANY fixed points. Consequences, all measured:

1. **Cold-starting at a deep grid lands in a random stuck basin.** Fixed M=32 cold
   x-sweep gave non-monotone `Jc`: `2.288, 2.574, 2.551, 2.479, 2.463`. Fixed-N cold
   M-sweep was garbage (`... 2.000, 2.730`). Cold solves are NOT reliable optima.
2. **Only a disciplined climb from k=1 is trustworthy.** Uniform doubling k=1→6 at
   M=32 gives a clean monotone ladder `2.0, 2.0, 2.16, 2.173, 2.284, 2.584`. This is
   the ONLY path that reproduces the reference behaviour. **Never cold-start deep.**
3. **At fixed M, x-refinement saturates.** Seeded from a healthy basin, both uniform
   and band-refine jump once (~+0.26 to ~2.58) then go flat. The `ln(res)` growth
   needs BOTH x and M refined — x alone hits an M-ceiling (~2.59 at M=32).
4. **Band-refinement is a real efficiency lever.** From a healthy basin it captured
   ~87% of the x-gain at ~half the nodes (117 vs 257 for matched Jc). This survives.

So the driver is **two-phase**: climb uniform from k=1 (basin discipline), THEN
band-refine for cheap depth — all at a FIXED M per run. Whether the fixed-M x-ladder
saturates (bounded-looking) is only meaningful once task 04 checks it's M-stable;
the M axis is 04's discriminator, not this driver's job.

The `tau_regrid`/`regauge` lever is INERT (primer §0.4 fact 1) — dropped from the
schedule entirely. Do not call it.

## 3.1 `adaptive_refinement`

```python
def adaptive_refinement(k_seed=4, n_band=5, k0=1, M=32,
                        max_iter=80, tol=1e-8, verbose=True):
    """Two-phase harvest-gauge refinement at FIXED M. Returns list of result dicts.

    Phase A — BASIN DISCIPLINE (uniform climb, x doubles each step):
        Start make_grids(k0, M) from the standard g-init ramp
        (g_init[i,j] = 0.5*t[j]*(x[i]**2 - 1)), solve.
        Then climb k0+1 .. k_seed by interpolate_to_next_level (uniform double) +
        re-solve, EXACTLY like refine_baseline.dyadic_refinement. This threads the
        good coordinate-ascent basin. (Diagnosis finding 2 — mandatory.)

    Phase B — DEPTH (band-refine x only, the efficiency lever):
        For n_band generations, from the healthy seed:
            x_new = band_refine(x)                         # |x|<BAND only (§01)
            f0,g0 = adaptive_warm_start(f, g, x, t, x_new, t)   # t UNCHANGED (§02)
            f, g, J_hist = alternating_maximization(x_new, t, f_init=f0, g_init=g0,
                                                    max_iter, tol)
            x = x_new
    M is FIXED for the whole run (fact 1: time-position inert; fact 3: x saturates at
    fixed M — that saturation-vs-not is what 04 probes by re-running at 2*M).

    Record per generation (both phases), dict:
        {gen, phase ('A'|'B'), N, M, n_nodes=N*(M+1), Jc, dJk=Jc-Jc_prev,
         x_grid, t_grid, f, g, elapsed}

    Assert Jc non-decreasing across ALL generations (warm start is J-neutral +
    per-grid LP is exact for that basin, so the disciplined sequence must be
    monotone up to ~1e-6). A drop signals a warm-start/feasibility bug — raise.
    """
```

Notes for the implementer:
- Phase A is literally `dyadic_refinement`'s loop inlined (or call its helper
  `interpolate_to_next_level`); reuse it, don't reinvent. Solve with the same
  g-init ramp at `k0`.
- Phase B keeps `t` (hence M) fixed — pass the same `t` into `adaptive_warm_start`
  as `t_new`. `regauge_time` will be the identity (M unchanged), so the warm start
  is exactly `prolong_x`.
- `n_nodes = N*(M+1)` — the efficiency metric. Band gens grow N slowly (arms frozen),
  so Phase B reaches deep x-resolution at far fewer nodes than uniform would.

## 3.2 `__main__` — run it

```python
if __name__ == "__main__":
    res = adaptive_refinement(k_seed=4, n_band=5, k0=1, M=32, verbose=True)
    print("\n gen | ph |   N  | M  | nodes  |   Jc      |  dJk")
    for r in res:
        print(f" {r['gen']:3d} |  {r['phase']} | {r['N']:4d} | {r['M']:2d} | "
              f"{r['n_nodes']:6d} | {r['Jc']:.6f} | {r['dJk']:+.6f}")
    Js = [r['Jc'] for r in res]
    print("dJk:", np.round(np.diff(Js), 5))
```

## 3.3 What to look for (the science)

Read the `dJk` column ACROSS BOTH PHASES, but interpret Phase B (fixed-M band
depth) knowing it can only ever saturate at fixed M (finding 3). The real
bounded-vs-unbounded call is NOT made here — it needs 04's M-sweep. This driver's
deliverable is: a trustworthy, monotone, cheap-to-extend `Jc(N)` curve at fixed M,
reaching deeper x-resolution than uniform can afford at equal nodes.

- If Phase B `dJk` decays to ~0 → x-saturation at this M (expected). 04 then asks:
  does the saturation *ceiling* rise with M (→ unbounded joint limit) or not (→
  bounded)?
- Report the node-count at which band-depth matches the best uniform Jc — that ratio
  is the efficiency win.

**Do not call bounded/unbounded from this run alone.** (Same discipline the
`kink_opt` Run 10 write-up used: one axis at a time, gates before verdicts.)

## 3.4 Acceptance checks

```python
import numpy as np
from mesh.refine_baseline import dyadic_refinement

# 1. Phase A must reproduce the disciplined uniform ladder exactly (it IS that ladder)
res = adaptive_refinement(k_seed=5, n_band=0, k0=1, M=32, verbose=False)
base = dyadic_refinement(k_start=1, k_max=5, M=32, verbose=False)
for ra, rb in zip(res, base):
    assert abs(ra['Jc'] - rb['J']) < 1e-6, (ra['N'], ra['Jc'], rb['J'])
print("Phase A reproduces disciplined uniform ladder: OK")

# 2. Full run is monotone non-decreasing and Phase B stays feasible (asserted inside)
res = adaptive_refinement(k_seed=4, n_band=4, k0=1, M=32, verbose=False)
Js = [r['Jc'] for r in res]
assert all(Js[i] >= Js[i-1] - 1e-6 for i in range(1, len(Js))), Js
print("full climb+band run monotone: OK   final Jc=%.4f at N=%d, %d nodes"
      % (res[-1]['Jc'], res[-1]['N'], res[-1]['n_nodes']))

# 3. Band depth reaches a given Jc at fewer nodes than uniform (the efficiency claim)
# (compare res[-1] band gen against the uniform level of equal-or-greater Jc)
```

Check 1 is load-bearing: it proves Phase A is the trustworthy basin path. Check 2
proves the band phase doesn't break monotonicity/feasibility. If check 1 fails,
Phase A isn't threading the basin and every Phase B number is untrustworthy.

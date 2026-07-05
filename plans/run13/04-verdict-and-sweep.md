# 04 — The verdict: sweep `r`, apply the decision rule

**Prereq:** read `00-primer.md` (esp. §0.6). **Depends on:** 03 (`fixed_point`,
`tiling_gain`). Ideally also 02 passing (D1 trust gate). **File to edit:**
`kink_opt/cell.py` (+ a narrated demo, see below). **Deliverable:** `cell_sweep`, and
a printed bounded / unbounded / inconclusive verdict.

## Goal

Run the fixed-point loop across a range of frame contractions `r`, read the converged
harvest `δ̂*`, the env-reproduction distance, and the per-octave ratio, then apply the
**corrected decision rule** (parent plan §6). This is the deliverable Run 13 exists to
produce.

## 4.1 `cell_sweep(rs, n_iter=12, tol=1e-4, **kw) -> list[dict]`

```python
def cell_sweep(rs, n_iter=12, tol=1e-4, **kw):
    """Run fixed_point + tiling_gain for each frame contraction r in `rs`.
    Returns a list of dict(r, converged, dist_final, delta_star, gamma_geom,
    gamma_emp_last, verdict)."""
    out = []
    for r in rs:
        fp = fixed_point(r, n_iter=n_iter, tol=tol, **kw)
        tg = tiling_gain(fp["deltas"], r)
        dist_final = fp["dists"][-1] if fp["dists"] else float("nan")
        gamma_emp_last = tg["gamma_emp"][-1] if tg["gamma_emp"] else float("nan")
        out.append(dict(r=float(r), converged=fp["converged"],
                        dist_final=float(dist_final),
                        delta_star=tg["delta_star"],
                        gamma_geom=tg["gamma_geom"],
                        gamma_emp_last=float(gamma_emp_last),
                        verdict=_verdict(fp, tg)))
    return out
```

## 4.2 The decision rule (`_verdict`) — parent plan §6, applied

The corrected rule: `γ` near 1 is **inconclusive both ways** (a geometric sum
converges up to γ=1; a `c/k` tail keeps γ<1 yet diverges — harmonic). So:

- **BOUNDED** requires: the env loop converged (`converged=True`, a real fixed point,
  so `δ̂*` is a genuine stationary harvest), AND the per-octave ratio is bounded away
  from 1 below — `γ ≤ ~0.9` — AND the gains are actually shrinking (`k·Δ_k → 0`, i.e.
  no harmonic tail; check `octave_gains` decay faster than `1/k`).
- **UNBOUNDED** requires: `δ̂*` bounded below (does not collapse to 0) AND `γ ≥ 1`
  (per-octave gain not shrinking) at a genuine fixed point.
- **INCONCLUSIVE** otherwise: `γ ∈ (0.9, 1.0)`, or the env loop did not converge (no
  fixed point ⟹ no honest `γ`), or `δ̂` is budget-sensitive (task 03's stability
  caveat fired).

```python
def _verdict(fp, tg):
    if not fp["converged"]:
        return "INCONCLUSIVE (no env fixed point)"
    g = tg["gamma_geom"]
    ds = float(tg["delta_star"])
    if not np.isfinite(ds) or ds <= 1e-6:
        return "INCONCLUSIVE (delta_star collapsed)"
    if g <= 0.9:
        return "BOUNDED"
    if g >= 1.0:
        return "UNBOUNDED"
    return "INCONCLUSIVE (gamma in knife-edge band)"
```

Note `gamma_geom = 2r` is pinned by the tiling choice, so the sweep over `r` is
effectively a sweep over `γ`; the empirical `gamma_emp_last` is the honesty
cross-check that `δ̂` is not drifting in a way that breaks the geometric assumption. If
`gamma_emp_last` and `gamma_geom` disagree by more than ~10%, the `δ̂`-constant premise
is violated — downgrade to INCONCLUSIVE and report the discrepancy rather than
trusting `gamma_geom`.

## 4.3 The narrated demo (make it reproducible)

Add a "Run 13" entry to `kink_opt/demos.py` following that file's existing numbered,
narrated pattern (each run prints a header explaining what it tests and why), OR — if
that is too heavy — add a `cell_sweep` block under `cell.py`'s `__main__`. Minimum:

```python
rs = [0.35, 0.45, 0.5, 0.55, 0.65]
for row in cell_sweep(rs):
    print(f"  r={row['r']:.2f}  conv={row['converged']}  "
          f"dist={row['dist_final']:.2e}  d*={row['delta_star']:.4f}  "
          f"g_geom={row['gamma_geom']:.2f}  g_emp={row['gamma_emp_last']:.2f}  "
          f"-> {row['verdict']}")
```

## Acceptance gate & how to read the result

1. The sweep runs to completion for every `r` (no exceptions), printing one line each.
2. **Consistency check (this is the real validation):** because `γ = 2r` is pinned, the
   verdict must move monotonically with `r` — small `r` → BOUNDED, large `r` →
   UNBOUNDED, with an INCONCLUSIVE band straddling `r = 0.5`. If instead every `r`
   collapses `δ̂*→0` or never converges, the loop is starved/broken, not decisive —
   escalate to the task 03 stability discipline (raise `outer`/`sub`) before reporting.
3. **Honest reporting.** Whatever the sweep shows, write it into the parent plan
   (`plans/run13-selfreproducing-cell.md`): add a "Run 13 result" subsection stating,
   for each `r`, `converged / δ̂* / γ / verdict`, and the one-line reading. Follow the
   file's established honesty conventions (see how Runs 9–12 report INCONCLUSIVE /
   BOUNDED with caveats — do not overclaim).

### The load-bearing caveat to state explicitly

The verdict is only as trustworthy as assumptions **A1 (E-sufficiency)** and **A2
(ρ/r rescaling)** — see task 05. If task 05 has not been run, say so: the `r`-sweep
gives a verdict *conditional on* E being a sufficient statistic and on the `ρ/r`
normalization being right. A clean monotone `r`→verdict pattern is *supporting*
evidence; it becomes *proof-grade* only with task 05's probes passing and, ultimately,
the exact-arithmetic pass (parent plan §7, Stage F — not in this task set).

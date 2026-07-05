# 03 — The fixed-point loop and the tiling multiply

**Prereq:** read `00-primer.md` (esp. §0.6). **Depends on:** 01 (`cell_read_env`,
`cell_env_distance`). **File to edit:** `kink_opt/cell.py`. **Deliverables:**
`cell_step`, `fixed_point`, `tiling_gain`.

## Goal

Close the map `CELL: E ↦ (δ̂, E′)` and iterate it: `E_{n+1} = CELL(E_n)` from the
flat seed to a fixed point `E*`. Record the sequence of per-cell harvests `δ̂_n` and
env-reproduction distances, then convert `δ̂` into per-octave gain via the tiling
multiply. This is where the boundedness signal is produced (task 04 reads the verdict
off it).

## 3.1 `cell_step(env, r, coarse_N=8, outer=40, sub=8) -> dict`

One full CELL evaluation: the existing `↦ δ̂` half plus the new `↦ E′` half.

```python
def cell_step(env, r, coarse_N=8, outer=40, sub=8):
    """One octave of CELL: E -> (delta_hat, E'). Solves the injected cell
    (existing half) and reads its outgoing environment (task 01). Returns
    dict(delta_hat, env_out, constraints_ok, sol)."""
    res = cell_solve(env, coarse_N=coarse_N, outer=outer, sub=sub)
    env_out = cell_read_env(res["sol"], r=r)
    return dict(delta_hat=res["Jc"], env_out=env_out,
                constraints_ok=res["constraints_ok"], sol=res["sol"])
```

`delta_hat` is the **certified** harvest (`Jc`), which is convention-safe (see the
parent plan §7). Always carry `constraints_ok` forward — a step whose cell fails
certification is not a usable data point.

## 3.2 `fixed_point(r, n_iter=12, tol=1e-4, coarse_N=8, outer=40, sub=8) -> dict`

Iterate from the flat seed. `env_to_lp` inside `cell_solve` divides `ρ` by `env["r"]`,
so the incoming env must carry the frame's `r`. **Seed:** `flat_env()` carries `r=1`
(boundary data). After the first step, every env carries `r` = the frame contraction
you are testing; overwrite the seed's `r` with the test `r` before the first step so
the loop is stationary. (Rationale: the seed's `r=1` is only for the flat no-op gate;
the *iteration* is at fixed frame contraction `r`.)

```python
def fixed_point(r, n_iter=12, tol=1e-4, coarse_N=8, outer=40, sub=8):
    """Iterate E_{n+1} = CELL(E_n) at fixed frame contraction r, from the flat
    seed, to a fixed point. Returns dict(deltas, dists, envs, converged, r).
      deltas[n]  = delta_hat of step n (certified per-cell harvest)
      dists[n]   = cell_env_distance(env_in, env_out) at step n
      converged  = True if dists[-1] < tol
    Stops early once dists < tol."""
    env = dict(flat_env())            # copy; do not mutate the module seed
    env["r"] = float(r)               # iterate at the tested frame contraction
    deltas, dists, envs = [], [], [env]
    for _ in range(n_iter):
        step = cell_step(env, r=r, coarse_N=coarse_N, outer=outer, sub=sub)
        if not step["constraints_ok"]:
            # a failed cell is not a data point; record NaN and stop
            deltas.append(float("nan")); dists.append(float("nan"))
            break
        d = cell_env_distance(env, step["env_out"])
        deltas.append(step["delta_hat"]); dists.append(d)
        envs.append(step["env_out"])
        env = step["env_out"]
        if d < tol:
            break
    converged = bool(dists and np.isfinite(dists[-1]) and dists[-1] < tol)
    return dict(deltas=deltas, dists=dists, envs=envs,
                converged=converged, r=float(r))
```

**Watch for (expected failure modes, report them honestly):**
- **No convergence** (`dists` plateaus above `tol`): the environment does not
  self-reproduce at this `r` — a real result (composition fails), not a bug. Report it.
- **`δ̂` collapses to ~0** across iterations: the injected budget starved the cell.
  Check it is not a resolution/`outer` artifact by re-running one point at
  `outer=80, sub=12` and confirming `δ̂` is stable (same discipline Runs 9–12 needed;
  see parent plan's "budget-artifact history"). If `δ̂` moves a lot, the budget is too
  low — raise `outer`/`sub` until it is stable, then use that as the floor.
- **Certification fails at high `sub`**: the LP's `A_ub` memory is O((Np1·Kf)²); the
  single-kink cell is tiny so this should not bite, but if it does, keep `sub ≤ 12`.

## 3.3 `tiling_gain(deltas, r, tiles_per_octave=2) -> dict`

Convert the per-cell harvest sequence into per-octave gains and the ratio `γ`. Per
§0.6: octave-k gain `Δ_k = (tiles · r)^k · δ̂_k`, and with `tiles=2` the ratio is
`γ = 2r` when `δ̂` is constant; if `δ̂` drifts, report the empirical ratio too.

```python
def tiling_gain(deltas, r, tiles_per_octave=2):
    """Per-octave gains from the per-cell harvest sequence (§0.6, §5 tiling).
      Delta_k = (tiles_per_octave * r)**k * delta_hat_k
      gamma_geom  = tiles_per_octave * r          (the pinned-tiling ratio)
      gamma_emp[k]= Delta_{k+1} / Delta_k         (empirical, catches delta drift)
    Returns dict(octave_gains, gamma_geom, gamma_emp, delta_star)."""
    d = np.asarray(deltas, float)
    k = np.arange(d.size)
    octave = (tiles_per_octave * r) ** k * d
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_emp = octave[1:] / octave[:-1]
    delta_star = float(d[np.isfinite(d)][-1]) if np.isfinite(d).any() else float("nan")
    return dict(octave_gains=octave.tolist(),
                gamma_geom=float(tiles_per_octave * r),
                gamma_emp=gamma_emp.tolist(), delta_star=delta_star)
```

`tiles_per_octave=2` is the pinned tiling choice (parent plan §5). Keep it a named
argument, not a hardcoded literal, so a future single-child geometry can override it
(and must re-derive the count — the two are not interchangeable).

## Acceptance gate

Run and report for `r = 0.5` (the knife-edge) and one value each side, e.g.
`r ∈ {0.4, 0.5, 0.6}`:

1. `fp = fixed_point(r)`; print `fp["deltas"]`, `fp["dists"]`, `fp["converged"]`.
2. `tg = tiling_gain(fp["deltas"], r)`; print `tg["gamma_geom"]`, `tg["gamma_emp"]`,
   `tg["delta_star"]`, `tg["octave_gains"]`.
3. Sanity: `gamma_geom == 2*r` exactly (0.8, 1.0, 1.2). `delta_star` finite and > 0 on
   at least the converged points. If `dists` is monotone-decreasing, note it (evidence
   the map is a contraction on environments — necessary for a meaningful fixed point).

Do **not** declare a bounded/unbounded verdict here — that is task 04, which needs the
sweep and the decision rule. This task's job is a working, honest loop plus its
diagnostics.

# 05 — Sufficiency probes A1 / A2 (deferred, non-blocking)

**Prereq:** read `00-primer.md` (esp. §0.4, §0.6). **Depends on:** 03 (`cell_step`,
`fixed_point`). **Status:** DEFERRED — do this only after task 04 produces a verdict.
**File to edit:** `kink_opt/cell.py`. **Deliverables:** `probe_e_sufficiency`,
`probe_rho_rescaling_sensitivity`.

These turn task 04's verdict from *conditional* into *trustworthy*. Both are cheap.
Both can silently manufacture or destroy the fixed point, so a verdict without them is
"supporting evidence," not proof.

## A1 — Is `E` a sufficient statistic? (`probe_e_sufficiency`)

The whole `map-on-environments` framing assumes `(β, ρ)` at one read time **screens
off** the rest of the accumulated structure — that the child's `δ̂` depends on the
parent only through `E`. But cell weights are LP-solved against everything present. If
two different underlying structures with the *same* `E` give *different* `δ̂`, then `E`
is missing a coordinate and no fixed point of `CELL` means anything.

**Test:** produce two solved cells whose read-off environments `E` agree to a tight
tolerance but whose internal structure differs (e.g. reached via different `outer`, a
slightly perturbed carrier offset, or a different but E-matched incoming env), then
compare their next-step `δ̂`.

```python
def probe_e_sufficiency(r=0.5, tol_env=1e-3, **kw):
    """A1: perturb the deep structure while holding E fixed to tolerance;
    require delta_hat invariant. Returns dict(env_gap, delta_a, delta_b,
    delta_gap, ok). ok=False => E needs another coordinate (first candidate:
    the parent's local drift velocity at handoff)."""
    # Build two cells that AGREE in E but DIFFER internally. One concrete recipe:
    #   A: standard flat-seed step.
    #   B: same, but reached with a different `outer` (or a small carrier jitter),
    #      then verify their env read-offs are within tol_env before comparing delta.
    a = cell_step(dict(flat_env(), r=r), r=r, **kw)
    b = cell_step(dict(flat_env(), r=r), r=r, outer=kw.get("outer", 40) * 2)
    env_gap = cell_env_distance(a["env_out"], b["env_out"])
    # step once more from each and compare the harvest
    da = cell_step(a["env_out"], r=r, **kw)["delta_hat"]
    db = cell_step(b["env_out"], r=r, **kw)["delta_hat"]
    delta_gap = abs(da - db)
    return dict(env_gap=float(env_gap), delta_a=float(da), delta_b=float(db),
                delta_gap=float(delta_gap),
                ok=bool(env_gap < tol_env and delta_gap < 5e-3))
```

Refine the "two structures, same E" recipe as needed — the essential requirement is
`env_gap < tol_env` (the environments really do match) while the *internal* states
differ. If `ok=False` with `env_gap` genuinely small but `delta_gap` large, that is
the important negative result: record it in the parent plan (§4 A1) and note the first
extra coordinate to try is the parent's local drift velocity at handoff (this also
ties to fork D2 — signed slope).

## A2 — Is the verdict robust to the `ρ/r` normalization? (`probe_rho_rescaling_sensitivity`)

`r` is a chosen frame parameter; the concern is that the *verdict* is an artifact of
the specific `ρ/r` rule rather than the physics. Task 04 already sweeps `r`; this probe
checks the orthogonal question — that perturbing the normalization *convention* (not
`r` itself) does not flip the verdict.

```python
def probe_rho_rescaling_sensitivity(r=0.5, factors=(0.8, 1.0, 1.25), **kw):
    """A2: re-run the loop with rho scaled by an extra convention factor and
    check the converged delta_star / convergence are insensitive. A verdict
    that flips under a small convention change is units-driven, not physical.
    Returns list of dict(factor, converged, delta_star)."""
    out = []
    for fac in factors:
        # inject an extra convention factor by scaling the seed's rho
        seed = dict(flat_env()); seed["r"] = r; seed["rho"] = seed["rho"] * fac
        # run the loop from this perturbed seed (inline a fixed_point variant
        # that accepts a seed, or temporarily monkeypatch -- keep it local)
        ...
    return out
```

(Implementation: either add a `seed=` argument to `fixed_point` — cleaner — or write a
small local loop mirroring it. Prefer adding `fixed_point(..., seed=None)` defaulting
to `flat_env()`, which also makes A1's recipe easier.)

## Acceptance gate

1. `probe_e_sufficiency()` runs; print `env_gap`, `delta_gap`, `ok`. Interpret: `ok=True`
   supports A1 (E is sufficient at this point); `ok=False` with small `env_gap` is a
   real finding — E needs another coordinate.
2. `probe_rho_rescaling_sensitivity()` runs; print the table. Interpret: converged
   `delta_star` roughly constant across `factors`, and verdict unchanged ⟹ A2 robust.
3. Fold both results into the parent plan's §4 (A1/A2) and §15 honesty ledger. Only
   with both probes passing should task 04's verdict be described as more than
   conditional/supporting.

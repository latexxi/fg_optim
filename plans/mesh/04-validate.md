# 04 — Validation, the M-sweep discriminator, figures — REVISED

Read `00-primer.md` and finish 01–03 (revised) first. New file
`mesh/validate_adapt.py` (checks + `main()` writing figures + the result note).

**Reframed after diagnosis (see 03 §3.0):** at fixed M, x-refinement *saturates*
(both uniform and band). So a fixed-M `dJk→0` is NOT evidence of bounded J — it may
be an M-ceiling artifact. The real bounded-vs-unbounded discriminator is: **does the
x-saturation ceiling rise as M grows?** That M-sweep is the central deliverable
here, not a side gate.

## 4.1 The M-sweep (central experiment)

```python
def m_sweep(Ms=(16, 32, 64, 128), k_seed=4, n_band=4):
    """For each M, run adaptive_refinement (climb+band, fixed M) and record the
    saturated ceiling Jc_sat(M) = the best Jc reached at that M.

    Return list of (M, Jc_sat, N_at_sat, nodes_at_sat).

    Read-off:
      * Jc_sat(M) rising ~linearly in log(M), no sign of leveling  -> UNBOUNDED lean
        (the joint x,M limit keeps paying — consistent with J ~ ln(res)).
      * Jc_sat(M) leveling to a horizontal asymptote                -> BOUNDED lean.
      * noisy / non-monotone in M                                   -> basin trouble
        (finding 2); tighten max_iter or the climb discipline and rerun before
        reading anything.
    """
```

Guardrails (diagnosis finding 2 — cold solves scatter):
- Every point must come from the **climb-from-k0=1** path (adaptive_refinement does
  this). If any `Jc_sat(M)` is non-monotone in M, do NOT interpret — it means the
  coordinate ascent slipped a basin at that M. Bump `max_iter`, or climb M gradually
  too (warm-start across M via `adaptive_warm_start` with growing M) and rerun.
- Watch the RAM wall: constraint matrices are `O((N·(M+1))^2)`. M=128 with a deep
  band N is the practical ceiling on a workstation — note where it stops.

## 4.2 Gates (each prints PASS/FAIL + the number)

1. **Phase-A basin fidelity.** `adaptive_refinement(k_seed=5, n_band=0)` Phase A must
   equal `dyadic_refinement(k_start=1, k_max=5)` to `<1e-6` per level (03 check 1).
   If FAIL, the basin path is broken and every number below is void.

2. **Monotonicity + feasibility.** Full climb+band run has non-decreasing `Jc` and
   every Phase B generation passes `check_feasible` on its warm start (asserted in
   `adaptive_warm_start`). Report max J-drift and any feasibility violation.

3. **Band-mass premise.** On each solved generation, ≥95% of harvest mass sits in
   `|x|<BAND` (invariant I2 — the premise that lets band-refine skip the arms).
   Per-x harvest = `sum_j f_diff[i,j]*kappa_g[i,j]`; check the band fraction. If it
   falls below 0.95 at deep gens, harvest is leaking into the arms and band-refine is
   starving real structure — report it (would explain any band-vs-uniform Jc gap).

4. **Band-vs-uniform gap at matched resolution.** Diagnosis showed band captured
   ~87% of the uniform x-gain, not 100%. Quantify it: at matched N (or matched
   band-spacing level), `Jc_band / Jc_uniform`. Report the ratio honestly — if band
   plateaus well below uniform, the efficiency win is partial and must be stated as
   such, not hidden.

A verdict from a run where gate 1 or 2 FAILs is not reportable. Say so.

## 4.3 Efficiency table

```
   M | uniform: (N, nodes) to reach Jc*  | band: (N, nodes) to reach ~Jc* | node ratio
```
For a target `Jc*` (e.g. the M=32 ceiling), how many nodes each route needs. Band's
frozen arms should give a node-count reduction — that ratio is how much deeper the
same RAM budget reaches.

## 4.4 Figures (`main()` writes PNGs to repo root, dpi≈115, magma/viridis)

- `mesh_msweep.png`: `Jc_sat(M)` vs `log(M)` — **the money plot**. Rising-linear vs
  leveling is the bounded/unbounded read, straight off it. Overlay the reference
  npz points (k04/05/06 = 2.625/2.836/3.055) if their M is known, as a sanity anchor.
- `mesh_ladder.png`: `Jc(N)` for the climb+band run at each M, one line per M, log-x.
  Shows Phase A climb, Phase B band-depth, and the per-M saturation plateau.
- `mesh_grid.png`: scatter the adaptive `(x,t)` nodes for one deep generation next to
  the uniform grid — the band × (uniform-in-t) node allocation, visual.
- `mesh_harvest.png`: per-x harvest profile, `|x|<0.4` band shaded (gate 3 visual).

## 4.5 Acceptance + write-up

Run `python3 -m mesh.validate_adapt`. Gates 1–2 PASS, gates 3–4 report their
numbers, the M-sweep table + 4 PNGs are produced.

Append a result section to `GEN_INSPECT.md` (or `plans/mesh/RESULT.md`) stating
HONESTLY, in the repo's established tone (cf. `kink_opt` Run 10/11 — report the
number that came out, gates and all, "inconclusive" if that's what it is):

- the four diagnosis findings that reshaped this (t-position inert; coordinate-ascent
  basin-dependence; fixed-M x-saturation; band-refine partial efficiency win),
- the `Jc_sat(M)` sequence and whether it reads rising (unbounded), leveling
  (bounded), or inconclusive — with gate 1/2 status,
- the band-vs-uniform node-efficiency ratio (gate 4 / §4.3),
- the caveat: this measures the SAME hierarchy the uniform mesh does; band-refine
  only relocates x-nodes and the disciplined climb only picks the basin — no new
  physics, just deeper reach.

Do not round a leveling-but-noisy `Jc_sat(M)` down to "bounded" without gate-1 basin
fidelity and a monotone-in-M sweep backing it.

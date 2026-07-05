# Run 12 — Renormalization cell: one-generation discretization + scaling law

## Status: PLAN. Successor to Run 11 (`kink_opt/construct.py`, bounded-for-that-
anchoring) and to the mesh finding (`plans/MELTING_KINKS.md`, J=3.055,
+0.215/octave over two octaves). Hypothesis under test:
`plans/BLOWUP_HYPOTHESIS.md`. Goal: replace "extrapolate a 2-3 point dJ trend"
with a k-independent statement — a per-generation cell problem whose fixed
point either proves blow-up (constant per-generation gain, by induction) or
proves boundedness (strict environment degradation, with the re-arm cost as
the identified mechanism).

## 1. Why a new discretization at all

Every prior instrument fails for a structural reason, not a tuning reason:

- **Global mesh (fg_opt3):** resolving generation k costs `Nx ~ 2^k` AND
  `Mt ~ 2^k` (t must refine with x or J regresses — k07/k08). Cost `O(4^n)`
  for n generations; capped at 2 clean octaves forever. Can only suggest.
- **Local search in this repo (Runs 5-10):** an optimizer that fails to find
  gain proves nothing (Run 10's honest inconclusive), and its per-generation
  measurement is budget-sensitive in ways that mimic the signal (Run 9's
  three artifacts, Run 10's four).
- **Run 11's construction:** artifact-free (LP-only) but the wrong ansatz —
  strict self-similar contraction to a shared point. Section 2 shows that
  ansatz *cannot* give blow-up even in principle, so its bounded verdict,
  while real, was decided at design time.

The cell discretization below is built to be simultaneously (a) artifact-free
like Run 11 (weights by convex LP only, positions analytic, deterministic)
and (b) shaped like the mesh optimum (drifting finite-width band), and
(c) O(1) cost per generation, so n generations cost O(n), not O(4^n).

## 2. Structural fact: strict self-similarity cannot blow up

Scale a feasible local structure by `x -> λx`, `t -> τt`; the Lipschitz cap
forces amplitude `-> λ`. Then

    f_t ~ λ/τ,   g_xx ~ 1/λ,   dx dt ~ λτ   =>   local J ~ λ.

A strictly contracted copy of the whole structure contributes `dJ_k ~ λ^k`:
geometric decay, bounded sum — *independent of any discretization or budget
question*. Run 11 built exactly this and measured exactly this. Conclusion:
any construction with a chance at blow-up must scale **anisotropically**.

## 3. The anisotropic scaling that gives constant dJ

Per-generation bookkeeping against the three global budgets (per-x rise
`∫f_t dt <= 1-|x|`; per-slice curvature mass <= 2; total time = 1):

| quantity                    | scaling      | budget it must respect        |
|-----------------------------|--------------|-------------------------------|
| rise share per swept x      | `r_k ~ 2^-k` | per-x rise: `Σ r_k <= 1-|x|` ✓|
| band width                  | `w_k ~ 2^-k` | (resolution, not budget)      |
| lifetime window             | `s_k ~ 2^-k` | total time: `Σ s_k <= 1` ✓    |
| curvature mass of band      | `m_k ~ O(1)` | per-slice cap renews across   |
|                             |              | (near-)disjoint windows ✓     |
| **travel length**           | `L_k ~ O(1)` | **does NOT shrink**           |

f_t spike height `h_k ~ r_k · v_k / w_k` (deposits rise-share `r_k` per swept
x at sweep speed `v_k = L_k/s_k`). Then per generation:

    dH_k ~ h_k · s_k = r_k · L_k / w_k ~ 2^-k · O(1) / 2^-k = const
    dJ_k ~ m_k · (rise harvested along path) ~ L_k · r_k / w_k ~ const

This is the `L/w` amplification (STRATEGY.md Section 3) made exact: melting =
**w, s, r contract geometrically; L stays O(1)**. Constant `dJ_k` summed over
`k <= log2(1/w_min)` generations = the mesh's observed `J ~ 0.215·log2(Nx)`.

**Where the re-arm cost lives, concretely:** the fine band needs slope
excursion `~ d_k/w_k` on top of the coarse profile. On the coarse arms the
Lipschitz budget is saturated (slope ±1, zero slack); slack exists only near
the coarse basin bottom where the slope passes through 0. So generation k+1
is confined to ride the *bottom* of generation k's basin, and the width of
that rideable region is what the coarse band "pays". Whether that squeeze
forces `r_{k+1}/w_{k+1}` (the gain rate) below `r_k/w_k` is the entire open
question, now localized to one adjacent-generation interaction.

## 4. The cell problem

### 4.1 Coordinates: per-generation co-moving frames

Generation k is discretized in its own rescaled frame, not on a global grid:

    x̂ = (x - c_k(t)) / w_k        (co-moving with the band-center path c_k)
    t̂ = (t - t_k) / s_k           (its own lifetime window)

Fixed node count per generation (K kinks per family, M time nodes — order
8 × 16), so n generations cost O(n). The repo's representation is already
half-suited: kink positions `ξ_i(t)` are Lagrangian co-moving markers, and
the graded grid (Task C) gives per-window t-refinement. What's new: the
band-center drift path `c_k(t)` and all kink offsets within the band are
**analytic** (constructed, never searched) — Run 11's discipline, band-shaped
ansatz instead of point-shaped.

### 4.2 Cell inputs/outputs (nondimensional)

A cell = one generation inserted into an "environment":

- **Inputs:** slope slack profile `β(x̂)` (Lipschitz budget left by coarser
  generations near the basin bottom), remaining rise profile `ρ(x̂)`,
  contraction factors `(λ_w, λ_s, λ_r)`, drift length `L̂`.
- **Solve:** weights by LP-only alternation (`_alternate(...,
  optimize_pos=False)`) — monotone, convex, deterministic; no position NLP
  anywhere.
- **Outputs:** certified gain `dĴ(β, ρ, λ_w, λ_s, λ_r, L̂)` and the outgoing
  environment `(β', ρ')` the band leaves for the next generation.

Gain is defined telescopically, `dJ_k = J(gen<=k) - J(gen<=k-1)`, the same
convention as `generation_ladder` — this absorbs the f-fine-against-g-coarse
cross-terms into the generation that created them.

### 4.3 Two-generation cell (probably the right minimal unit)

The re-arm interaction is adjacent-generation (fine band consumes the coarse
basin-bottom slack), so the minimal self-consistent unit is generations k and
k+1 together with a matching condition: generation k+1's incoming environment
must equal generation k's incoming environment after rescale. A one-generation
cell with prescribed `(β, ρ)` is the first implementation step; promote to
two-generation only if the outgoing-environment measurement proves too
sensitive to how `(β', ρ')` is read off.

### 4.4 Fixed point and the dichotomy

Iterate the environment map `E -> E'` (nondimensionalized). At a fixed point
`E* = E'*` the per-generation gain ratio `γ = dĴ_{k+1}/dĴ_k` is a single
well-defined number:

- **γ = 1** (equivalently `E*` exists with gain bounded below): every
  generation gains `>= c > 0`. Induction: `J >= J_0 + c·n` for all n —
  **sup J = +infinity and H[f] -> infinity** (PDF open question 2 answered).
- **γ < 1** (environment strictly degrades, `β' < β` compounding): geometric
  sum — **J bounded**, and the proof names the mechanism (re-arm cost eats
  the gain at rate γ).

Either branch is proof-shaped, unlike every prior run.

## 5. Proof scheme (why few generations suffice)

Blow-up needs only a lower bound plus induction — never an optimizer:

1. **Gadget lemma.** Exhibit one feasible generation-insertion (explicit
   piecewise-linear f/g increments) with `dJ >= c > 0` after paying its
   re-arm cost, inside environment E.
2. **Composition lemma.** The post-insertion state contains a rescaled copy
   of E with the same nondimensional parameters.
3. **Induction.** `J* >= Σ_k c = +infinity`.

Division of labor: the numerics *find* the gadget (solve the cell LP, locate
the fixed point, read γ); the proof *verifies* lemmas 1-2 exactly — the LP
solution is piecewise linear with finitely many kinks, so feasibility and the
value of dJ are finite lists of rational inequalities, checkable in exact
arithmetic. The mesh can never do step 2 (no structure to compose); Run 11
had step 2 but a gadget class that Section 2 dooms. This plan is the first
with both.

Fallback value even if the proof step stalls: O(n) cost means a 10+ generation
ladder is affordable, turning the mesh's 2-octave trend into a 10-octave
measurement of γ — decisive numerically even when not yet a theorem.

## 6. Implementation sketch (extends `kink_opt/construct.py`)

1. **Band ansatz.** `build_band(gen, w_k, s_k, L_k, r_k, K)` — K hats spread
   across the band width in the co-moving frame, kink trajectories
   `ξ_i(t) = c_k(t) + w_k · x̂_i`, drift path `c_k(t)` analytic (start with
   linear drift of length `L_k`; the mesh data suggests a there-and-back arc
   — try second).
2. **Boundary stock.** Pin `f(.,0)`, `g(.,1)` to full hats (gen 0 = the
   budget stock + the coarse carrier, as in MELTING_KINKS "boundary slices").
3. **Stacking.** `build_melt_hierarchy(n_gen, λ_w, λ_s, λ_r, L, K)` — like
   `build_hierarchy` but band-shaped and NOT endpoint-anchored: generation
   k+1 rides generation k's basin bottom along its own O(1)-length drift.
4. **Grid.** `graded_grid` sized to the narrowest window, same convention as
   Run 11; per-generation kinks alive only on their window (Task B masks).
5. **Solve + certify.** LP-only alternation, then `certify()`; read
   `dJ_k = Jc_k - Jc_{k-1}`.
6. **Environment read-off.** After solving gen <= k, measure the slope-slack
   and rise-remaining profiles in gen k's co-moving frame; compare to gen
   k-1's at the same nondimensional points — this is the map `E -> E'`.
7. **Fixed-point search.** Sweep `(λ_w, λ_s, λ_r, L̂)` for self-reproducing
   E; read γ there. Deterministic, seed-free throughout.

### Validation gates (all three Run 11 checks carry over, plus one new)

- `check_insertion_neutral` — forced-dead band must not move Jc.
- `grid_convergence_check` — Jc stable sub=8 vs sub=16.
- `travel_sanity` — generalized: the *band* must actually drift `~L_k`
  (else it degenerates to Run 11's co-located point and dJ collapse is
  meaningless).
- **New — mesh cross-check:** at n_gen=2-3, dJ_k should land in the vicinity
  of the mesh's +0.215/octave. Order-of-magnitude agreement gates trust; the
  Run 11 toy's 0.31-total-J was the tell that its ansatz missed the mesh
  structure entirely.

## 7. Risks and honest failure modes

- **No fixed point with γ=1.** Environment degrades strictly and compounds.
  Then boundedness is the theorem — acceptable (proof-quality either way),
  but state it as such; don't tune `(λ_w, λ_s, λ_r)` until γ looks like 1.
  The dichotomy is only honest if the fixed-point search is exhaustive over
  the ansatz class, and any γ reported comes with the ansatz named.
- **Ansatz too narrow.** Linear drift + uniform band may miss the mesh's
  there-and-back arc / sub-filament branching. Mitigation: mesh cross-check
  gate above; enrich drift-path family only when the gate fails.
- **Environment read-off is the new budget-sensitivity.** `(β', ρ')` measured
  on a discrete grid can hide degradation below grid resolution — the Run
  9/10 trap in new clothes. Mitigation: read-off must converge under
  `grid_convergence_check`, and the two-generation cell (4.3) exists
  precisely to double-check the one-generation map.
- **Cross-terms not telescoping cleanly.** If gen-k insertion retroactively
  changes what gen-(k-1) harvests by O(dJ), per-generation attribution blurs.
  The telescoping definition keeps the SUM honest regardless; only the
  γ-interpretation needs care.
- **K per band too small.** A band is "many hats smeared" — K=8 may
  under-resolve it and understate dJ (a one-sided error: fine for the
  blow-up branch, dangerous for concluding boundedness). Check dJ_k vs K
  saturation before trusting a γ<1 verdict.

## 8. Relation to prior runs (one line each)

- Run 8: contraction-to-endpoint warm start is redundant-by-co-location —
  first sighting of the mechanism Section 2 formalizes.
- Run 9/10: optimizer-based per-generation gains are budget-artifact-prone
  and can only ever *fail to find* gain — motivates constructive-only.
- Run 11: LP-only discipline proven out; ansatz (isotropic, point-anchored)
  now understood as structurally unable to blow up (Section 2).
- MELTING_KINKS: the target shape (drifting band, full-hat boundary stock,
  anisotropic scaling) and the calibration number (+0.215/octave).

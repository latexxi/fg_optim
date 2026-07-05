# Run 13 — The self-reproducing cell (the number that repeats)

**Supersedes the *aim* of `plans/archive/GEN0.md`** (its math is reused; its goal — a
generation-0 *optimizer* — is the wrong target, §2). **Conceptual companion:**
`plans/archive/GENERATIONS_PLAN.md` (the same induction, informal). **This revision folds in
`plans/archive/CRITIQUE.md`** (all three now archived — folded into this plan)**:**
the central construction is now the **unit-frame cell solver**
(§4), not a global-hierarchy simulation; the decision band is corrected (γ near 1 is
*inconclusive both ways*, §6); an **exact-arithmetic stage** funds the word "proof"
(§7); **E-sufficiency and ρ-rescaling** are named as load-bearing assumptions to
test (§4); the **geometry is pinned to tiling** (§5); and the **free theorem
`J* ≥ 3.05`** is pocketed immediately, independent of the γ story (§9).

Self-contained: no prior plan required.

---

## 0. The question

On `Q = [-1,1] × [0,1]`, maximize

    J[f,g] = ∫_0^1 ∫_{-1}^1 f_t(x,t) · g_xx(x,t) dx dt

over `f,g` convex in `x`, Lipschitz (`|f_x|,|g_x| ≤ 1`), zero on the spatial
boundary (`f(±1,t)=g(±1,t)=0`), monotone in time (`f_t ≥ 0`, `g_t ≤ 0`), with
`f(x,1)=0`, `g(x,0)=0`. **Open question:** `sup J` finite or `+∞`? A reference mesh
(grids to 65×257) reaches `J ~ 2.4–3.05` and grows `+0.215` per octave (per halving
of the finest time-scale) with no sign of stopping.

**The whole document turns on one fact stated once, sharply:** the mesh's growth
rate puts the measured octave ratio `γ` right at `≈ 1`, the exact boundary between
"bounded" and "`+∞`". At a knife-edge, any artifact worth a few percent flips the
verdict. Every design choice below is dictated by that: fairness is load-bearing,
two data points cannot settle it, and a verdict must be *robust*, not a single lucky
reading. (This is the last time the knife-edge is belabored; take it as the axiom
behind every "must".)

---

## 1. The coordinate (from GEN0.md §2, reused verbatim)

Fix `t`. A convex `f(·,t)` with `f(±1,t)=0` **is** its own curvature measure:

    f(x,t) = ∫ G(x,y) μ^f_t(dy),   μ^f_t = f_xx(·,t) ≥ 0,
    G(x,y) = ½(min(x,y)+1)(max(x,y)−1),   G(±1,·)=0,  G_xx=δ.

Boundary automatic; Lipschitz ⇔ `|⟨y⟩ ± mass| ≤ 2`. Codebase realizes `μ` as atoms
(kinks): `f = −Σ a_i hat(x;ξ_i)`, `a_i ≥ 0`; an atom of `μ` is a kink of `f`. Two
hard budgets: **rise per column** `∫_0^1 f_t(x,·)dt = −f(x,0) ≤ 1−|x|`; **curvature
per slice** `∫μ^g_t ≤ 2`. The objective is a pairing:
`J = ∫_0^1 (∫ f_t(y,t) μ^g_t(dy)) dt`.

---

## 2. Wrong target: V₀. Right target: the repeating increment.

GEN0.md would optimize generation 0 (one sweep, `J ≈ 2.6`) to a scalar `V₀`.
**`sup J` finite-vs-`+∞` does not depend on `V₀`** — only on whether the *per-octave
increment* `δ_k` repeats or decays: `J = V₀ + Σ_{k≥1} δ_k`. The mesh's `+0.215/octave`
is a claim `δ ≈ 0.215`, `γ ≈ 1`; and `0.215 ≪ 2.6`, so subtracting two heavy solves
to read a signal 10× below their noise floor is doomed (the Run 9–10 confound). We
measure `δ`/`γ` directly and never optimize a generation.

---

## 3. Why the budget doesn't trivially close it

A **stationary** `μ^g` (fixed tent at `x₀`) gives `J = mass·(−f(x₀,0)) ≤ 2(1−|x₀|) ≤ 2`
(the tent cap; correct). **Any `J > 2` forces `μ^g_t` to move.** A **moving** kink
`β(t)` dodges the per-column cap: its contribution `∫_0^1 f_t(β(t),t) dt` is a line
integral along a moving point, *not* `−f(β,0)`, so it sips *fresh* stock from a new
column each instant and can chase `∫ max_x f_t dt`. GEN0.md names both the alignment
principle and the depth cap (a slice bottomed at `c` is `≤ 1−|c|` deep) but never
collides them. That collision is the open question; no local bound resolves it — only
a self-consistency (fixed-point) argument.

---

## 4. The object: the **unit-frame cell**, iterated (not the cascade, simulated)

**The central correction from `CRITIQUE.md`.** The advertised advantage — decide an
*infinite* cascade at *fixed* cost — is only real if one octave is solved **in
rescaled coordinates with the environment as boundary data**, and *that map is
iterated*. Building the global hierarchy (`build_melt_hierarchy`) and reading
increments off a dense global certify is a **simulation of the cascade**: certifying
generation `k` still resolves width-`2^{-k}` features, cost grows with depth, you get
`n_gen ≈ 4`, and `γ` comes from 2–3 ratios — the very "two data points" this plan
mocks the mesh for, one level up. So the load-bearing new machinery is:

**`CELL: E ↦ (δ̂, E′)`** — a solver on a **fixed unit frame** (O(1) kinks over a unit
time window `[0,1]` in rescaled coordinates), taking the incoming environment `E` as
background/boundary data (the residue the cell harvests), returning the rescaled
increment `δ̂` and the outgoing environment `E′`. Iterate `E_{n+1} = CELL(E_n)` to a
fixed point `E*`. Cost is O(1) per iteration and **independent of depth** — that is
the entire point, and it does not exist in the current inventory. (The global-
hierarchy build is retained only as a cheap cross-check of the first 2–3 octaves,
§12 Stage B, not as the measurement.)

The induction factors (from `GENERATIONS_PLAN.md`) into two claims to check
*separately*:

- **Gadget (step works):** the cell nets `δ̂ > 0` after its re-arm cost, for *every*
  `E` in a neighborhood of `E*` — not one point (the iteration visits a family).
- **Composition (step repeats):** `E′ = E` at `E*` (`env_distance → 0`), licensing
  infinite iteration.

Two assumptions in this framing are doing real work and are **not** free — both are
cheap to test and both can silently manufacture or destroy a fixed point:

- **(A1) E is a sufficient statistic.** The map-on-environments assumes `(β, ρ)` at
  one read time *screens off* the coarser structure — that the child's `δ̂` depends on
  the parent only through `E`. But cell weights are LP-solved against the *whole*
  accumulated structure, including the parent's motion during/after the child's
  window. **Test (Stage F):** perturb the deep structure while holding `(β,ρ)` fixed
  to tolerance; require `δ̂` invariant. If `δ̂` moves, `E` needs another coordinate
  (first candidate: the parent's local drift velocity at handoff) before any fixed
  point means anything.
- **(A2) how E rescales.** `β` is dimensionless slope-slack. `ρ` is rise budget,
  which under the cell's anisotropic scaling shrinks like the per-octave rise share
  `r`. `env_distance` **must** compare `ρ/r` (normalized), not raw `ρ`, or units alone
  fabricate a fixed point. **State the normalization explicitly in code**, and verify
  the verdict is insensitive to reasonable alternatives (Stage C).

**Environment definition** (already in `melt.py`, extend for A2):
`β(x) = 1 − |f_x(x,t_read)|`, `ρ(x) = (1−|x|) + f(x,t_read)`, read in the child's
co-moving frame; store the rise share `r` alongside so `env_distance` normalizes `ρ`.

---

## 5. Geometry: **tiling** (pinned, not left ambiguous)

`CRITIQUE.md` flags a real ambiguity: "child is a *sub-window* of the parent"
(nested, co-harvests against the parent's live `f_t`) vs "anchored at the parent's
*window-end*" (sequential, inherits only residue) vs the source docs' **tiling**
(generation `k` is `2^k` finer sweeps filling the parent's window). These are
different physics. **Pin it to tiling**, because it is what GEN0.md §5 ("`2^k` passes
at scale `2^{-k}`") and `GENERATIONS_PLAN.md` ("two fit back-to-back… generation 2 is
four sweeps") actually describe, and it is the geometry whose arithmetic matches the
mesh's *constant-per-octave* story:

    per-octave gain  =  (tiling count 2^k)  ×  (per-cell increment δ̂ · scale^k).

The unit-frame cell (§4) makes this natural: all `2^k` children of an octave are the
**same** cell in rescaled coordinates seeing the **same** `E*`, so we solve one cell
and multiply. This is a *different, explicit* claim from "one child per octave earning
`O(1)` via `r·L/w`", and the plan commits to the former. (If a future run wants the
single-child geometry, it must re-derive the count factor — it is not interchangeable.)

---

## 5A. Stage-C interface spec (the CELL solver — build against this)

Concrete design for `CELL: E ↦ (δ̂, E′)`, grounded in the existing LP
(`kink_opt/lp.py`). Read before coding; the **design forks** at the end are the parts
that need a decision, not just typing.

**What the cell solves.** The child adds an *increment* `Δf, Δg` on top of a **frozen
parent residue** at the handoff. In the child's own rescaled unit frame (`x̂ ∈ [-1,1]`,
`t̂ ∈ [0,1]`), the child is a fresh copy of the original problem **except** the two
±1/`(1-|x|)` budgets are replaced by what the parent left. O(1) kinks per family, weights
LP-only (`_alternate(optimize_pos=False)`), zero position NLP — same engine as
`build_melt_hierarchy`, new boundary data.

**E has exactly two injection channels** (this is the whole coupling):

1. **Slope budget `β(x)` → the Lipschitz RHS.** `lp_weights_f` enforces Lipschitz as the
   two arm-slope rows `Σ_i A[k,i]/(1±XI[k,i]) ≤ 1.0`. The child inherits only the
   *remaining* slack, so **the RHS `1.0` becomes the parent's local slack** at the
   relevant boundary. Concretely: `lp_weights_f(..., lip_rhs=(bL, bR))` with
   `bL, bR ≤ 1` read from `E` (default `(1,1)` reproduces the current LP bit-for-bit).
2. **Rise budget `ρ(x)` → an added row.** The `1-|x|` rise cap is currently *implicit*
   (it follows from Lipschitz + `f(±1)=0`), so there is **no existing row to scale** — a
   parent that has already spent rise cannot be expressed by shrinking the arms alone.
   The child needs an **explicit rise-cap row per column** `-f_child(x) ≤ ρ(x)` at the
   sampled `x̂`, i.e. `Σ_i A[k,i]·hat(x̂_p; XI[k,i]) ≤ ρ(x̂_p)` for each sample `p`. This
   is new LP machinery (a `hat_matrix`-built block, same vectorized style as the
   monotone-rise rows). It is the single most important new constraint and the current
   inventory has nothing equivalent.

**Reading the outputs.**
- `δ̂` = the cell's own certified harvest (`certify` on the unit-frame solve, refined —
  convention-safe per §7's harvest note).
- `E′` = `read_environment` on the cell's *output* slice at its handoff time (already
  implemented; extend to also return the rise share `r`, §4 A2).

**The iteration.** `E_{n+1} = CELL(E_n)`; seed `E_0` = the flat/full budget
(`β≡1, ρ≡1-|x|`, the coarsest carrier's residue). Converge on `env_distance(E_n, E_{n+1})`
(normalized `ρ/r`, §4 A2). O(1) cost per step — run many steps (the point vs Stage B's
depth-4).

**Reuse map:** `lp_weights_f/g` (+ `lip_rhs=` and the new rise-cap block), `_alternate(
optimize_pos=False)`, `certify`, `read_environment` (+ `r`), `env_distance` (+ `ρ/r`
norm). New: the rise-cap LP block; the `CELL` driver; the fixed-point loop; the tiling
multiply (§5).

**Design forks to resolve before coding (each can silently break the fixed point):**
- **(D1) arm-only vs interior Lipschitz.** `β` is a profile over interior `x`, but the LP
  only constrains slope at the arms `±1`. For a convex PL `f` the max `|f_x|` is at an
  arm, so arm-only *may* suffice — **verify** that the parent's interior slope never
  exceeds its arm slope in the residue frame; if it can, add interior slope rows. Default:
  start arm-only, assert the interior bound holds, escalate if it fires.
- **(D2) signed slope vs magnitude (ties to §4 A1).** Available child slope at a column is
  direction-dependent: if the parent has slope `s` there, `Δf_x ∈ [-1-s, 1-s]`, so the
  child needs the **signed** `f_parent_x`, not just `β = 1-|s|`. If the E-sufficiency
  probe (Stage E) fails, this is the first suspect — carry signed slope in `E`.
- **(D3) how `ρ` rescales (A2).** Under the anisotropic child scaling, the rise budget
  shrinks by the rise share `r`; the rise-cap row RHS must be `ρ/r` in the unit frame, or
  a fixed point is fabricated by units. Fix `r` explicitly, verify verdict-insensitivity
  to reasonable choices.

---

## 6. Decision rule — corrected (γ near 1 is inconclusive *both* ways)

`CRITIQUE.md` is right that the old rule was wrong in two directions. Fix:

- **A convergent geometric sum stays convergent right up to γ = 1.** `γ = 0.95`
  *stably* ⇒ `Σδ_k = δ_1/(1−0.95)` finite ⇒ **bounded** (large `J`, not infinite).
  So "`γ ≥ 0.9` ⇒ evidence for `+∞`" is a motivated misreading. The **only** clean
  unbounded verdict is `δ_k` **bounded below** by a fixed `c > 0` (the gadget lemma,
  i.e. effectively `γ ≥ 1` in exact arithmetic).
- **`γ < 1` at each `k` does not imply bounded.** `δ_k ∼ c/k` gives
  `γ_k = k/(k+1) < 1 ∀k` yet `Σ = ∞` (harmonic). Over 2–3 octaves this looks like
  "`γ` slightly below 1, roughly flat" — indistinguishable from true decay unless you
  discriminate the *functional form*. So **fit `δ_k` against both** `c·γ^k`
  (geometric → bounded) **and** `a + b/k`-type power laws (→ `Σ` can diverge), and
  **track `k·δ_k`**: bounded if it →0, divergent if it →const. Two ratios cannot do
  this; this is another reason the O(1)-cost unit-frame solver (many octaves) is
  required, not the depth-4 simulation.

**Verdict bands:**
- **Bounded** — `δ_k` fits `c·γ^k` with `γ` **bounded away from 1** (no upward
  drift), `k·δ_k → 0`, at a genuine fixed point (A1/A2 clean).
- **Unbounded** — `δ_k` bounded below by `c > 0` across octaves (gadget lemma),
  ideally confirmed in exact arithmetic (§7).
- **Inconclusive** — `γ ∈ [0.9, 1.0]` without functional-form discrimination; or
  power-law and geometric fits both plausible; or A1/A2 fail. The band near 1 is
  **inconclusive in both directions**, never annexed by either.

`mesh_cross_check`'s factor-of-3 bar (0.07–0.65 around 0.215) is decoration at a
knife-edge — keep it as a sanity print, **strike it from the decision inputs**. Note
if `λ ≠ 0.5`, "per-octave" needs a rate conversion (γ per e-folding) before any
comparison to `0.215` — see §12.

---

## 7. Exact arithmetic — what makes it a proof, not a plot

The cell is piecewise-linear with LP-solved (hence rational-attainable) weights, so
both lemmas are **finite lists of rational inequalities, checkable exactly**. The
current stages read `γ` in floating point — that is a *measurement*, fairer than the
mesh but still a suggestive float. To earn the word "proof": **at the stabilized
point, freeze the construction, rationalize all breakpoints and weights, and verify
in exact arithmetic** (a) feasibility (every constraint as a rational inequality),
(b) the gadget bound `δ̂ ≥ c > 0`, and (c) `E′`-containment (the outgoing environment
contains a rescaled copy of the incoming, so composition applies). **Until that stage
runs, the deliverable is labeled "measurement," not "proof."**

**Harvest-convention note (γ is convention-safe).** The objective is distributional in
`x` (harvest form, `g_xx` = deltas) but its coarse-grid value depends on the `t`-quadrature
convention (`total_J`'s cell-midpoint sampling vs the mesh's step-in-`t`/left-node). These
disagree by O(coarse Δt) and **converge to the same continuous `J` under time refinement**
(measured: spread 0.24 → 0.001 from `sub=1` to `sub=64` on Run 3/5). Two consequences:
(i) raw coarse `total_J` is *optimistic* — it over-counts by letting kinks whipsaw inside
one step — so only refined `Jc` (or the step-in-`t` witness, which is a valid lower bound
at coarse resolution) is trustworthy; (ii) because every `γ`/`δ_k` here is read from
**certified (refined)** `Jc`, the verdict is **independent of the convention** — this
subtlety does not threaten the boundedness analysis. (Corrects an earlier note that
mis-signed this as an *under*-count: whichever convention a solution was optimized against
scores highest on it; re-scoring is never a free lift.)

---

## 8. Do **not** pin arms (GEN0.md caution, kept)

GEN0.md boxes `mass=2, moment=0`; but §3 admits `mass<2` where arms droop, and the
generation interface is exactly where mass ramps. The cell lives at the interface (it
inherits a *partially spent* environment, not a full tent), so forcing `mass=2` there
destroys the physics. The existing LP-only solve keeps Lipschitz as **box
inequalities** (`_wbounds`) — correct, drooping arms allowed. **Add no pinned-arm
equality LP** (the over-literal reading of GEN0.md §6 F1).

---

## 9. Free theorem — **DONE: exact `sup J ≥ 3.0552`** (independent of the γ story)

`CRITIQUE.md`'s best standalone catch, now **proved and banked**. Script + certificate:
`paper/exact_lower_bound.py` → `paper/exact_lower_bound.txt`. Witness: the k=6 dyadic-mesh
optimum (`../fg_opt3/data/level_k06.npz`, 65×257), repaired to **exact feasibility**
(blend at `eps=1e-6` with strictly-interior parabola profiles). **0 exact-arithmetic
feasibility violations**; exact objective

    J = 495748403191937564416657549941713018177701221 / 162259276829213363391578010288128000000000000
      = 3.055285422686430…  >  3.05.

Unconditional, does not depend on how boundedness resolves. Publishable improvement over
the trivial `2`.

**Subtlety resolved while doing it (matters for anyone extending this).** The mesh's
`compute_J` samples `g`'s curvature at the *left* time node — that is the exact
continuous `J` of a witness with **`g` piecewise-*constant* (step) in `t`**, `f`
piecewise-linear-in-`t`. The *other* natural convention (`g` piecewise-*linear*-in-`t`,
trapezoid rule) gives the same nodal data only `J = 2.587`. So the mesh's `3.055` is a
true continuous `J` **only under the step-in-`t` reading** — which is admissible because
the problem's class is monotone/**BV**-in-`t` (`conjecture.txt`'s "bang" mechanism *is* a
near-jump in `t`), and because smoothing the jumps over a vanishing width keeps `g`
Lipschitz-in-`t` and preserves `J` in the limit. Pick the step convention; it is both
admissible and the larger value.

**Caution from the cache itself:** `fg_opt3`'s cached `k=7,8` show J *dropping* (2.979,
2.845) with a collapsed `t`-grid (257→129→65 nodes) — the resolution-starvation artifact
of §12 Stage B, live in the mesh's own numbers. The trustworthy mesh maximum is **k=6**
(t well-resolved at 257 nodes); rationalize *that*, not the degraded deeper levels. This
also means the mesh's log-growth fit (`1.51+0.37·ln N`) is depth-limited by a resolution
wall — the empirical case *for* the O(1)-cost cell, not against it.

---

## 10. Symmetry to exploit, not guess around

The problem is symmetric under `(f,g,t) ↦ (g, f, 1−t)` (maps feasible→feasible:
`f_t≥0 ↔ g_t≤0`, terminal↔initial), and computed optimizers nearly respect it. **Test
whether imposing `g(x,t) = f(x,1−t)` costs anything at level 5–6.** If free, the cell
carries **one** function instead of two — half the LP, and the exact-arithmetic
lemmas (§7) get materially shorter. Cheap, and pure upside if it holds.

---

## 11. Inventory: what exists, what's missing

Exists in `melt.py` (reuse): `read_environment` (→ `β,ρ`; extend to store `r` for A2),
`env_distance` (→ normalize by `r`), `build_melt_hierarchy`/`melt_ladder` (global —
demoted to cross-check only), the four gates (`check_band_neutral`,
`grid_convergence_check`, `band_travel_sanity`, `mesh_cross_check`), `fixed_point_sweep`
(never run). **Missing — the real work:** the unit-frame `CELL: E ↦ (δ̂, E′)` solver
(§4), the tiling count factor (§5), the `δ_k` functional-form discriminator (§6), the
exact-arithmetic verifier (§7), the E-sufficiency probe (§4 A1), and a per-generation
**resolution gate inside `certify`** (§12 Stage B).

---

## 12. Staged plan with gates

Deterministic throughout — no seeds, no multistart (a `~0.2` signal cannot survive
best-of-N order-statistic noise; Run 9–10's lesson). Every `J` is `J_certified`.

- **Stage 0 — DONE: the free theorem.** `paper/exact_lower_bound.py` rationalizes the
  k=6 mesh point, repairs to exact feasibility, verifies all constraints + computes `J`
  in exact arithmetic ⇒ **`sup J ≥ 3.0552`** (certificate `paper/exact_lower_bound.txt`).
  Independent of §4–§8.

- **Stage A (regression, not a stage): reproduce Run 12.** `melt_ladder(n_gen=3,
  anchor="fixed")` must reproduce `dJk = +0.1866, +0.0098, +0.0053`, `env_distance ≈
  0.19`. ~1 min; keep as a fixture, not a milestone.

- **Stage B — global cross-check + resolution gate.** Build the tiling geometry (§5)
  globally for the first 2–3 octaves as a *cheap sanity check only*. Add a
  **resolution gate inside `certify`**: finest kink spacing vs dense-grid spacing at
  `sub=12` — at `λ_w=0.4`, octave 4 lives at width `≈0.026`; if the certification grid
  under-resolves it, the starvation artifact re-enters *hidden inside the "honest"
  number*. Gate must FAIL loudly, not silently under-resolve. (Note `sub=16` at
  `K=8,n_gen=3` needs >13 GB — cap at `sub=12`.) Gates: `check_band_neutral`,
  `grid_convergence_check`, `band_travel_sanity` PASS at octaves 1–2.

- **Stage C — build & iterate the unit-frame cell (the core).** Implement
  `CELL: E ↦ (δ̂, E′)` **per the §5A interface spec** (two injection channels, the new
  rise-cap LP row, design forks D1–D3): O(1) kinks, unit window, `E` as boundary data,
  weights LP-only. Fix the rescaling `ρ/r` (A2) and verify the fixed point is **insensitive**
  to reasonable normalization choices. Iterate `E_{n+1}=CELL(E_n)` for **many** octaves
  (the O(1) cost is what buys this over Stage B's depth-4). Acceptance: `env_distance`
  bottoms out and stays flat over *many* iterations (not 2 points — flatness over two
  is not flatness).

- **Stage D — read the verdict per §6.** At `E*`: fit `δ_k` to `c·γ^k` **and** a
  power law, track `k·δ_k`, read `γ` across a **neighborhood** of `E*` (gadget: sign
  and value robust, §4). Apply the corrected bands (§6): near-1 is inconclusive both
  ways; unbounded needs `δ_k` bounded below; bounded needs `γ` away from 1 with
  `k·δ_k→0`. Exclude `mesh_cross_check` from the decision.

- **Stage E — E-sufficiency probe (A1).** Perturb deep structure holding `(β,ρ)`
  fixed; require `δ̂` invariant. If not, add the drift-velocity coordinate to `E` and
  re-run C–D before trusting any fixed point.

- **Stage F — exact-arithmetic proof (§7).** Only if C–E give a clean verdict:
  rationalize, verify feasibility + gadget bound + `E′`-containment exactly. This is
  what upgrades "measurement" to "proof."

- **Stage G — Run 13 in `demos.py`.** Narrated, following Run 11/12 pattern: the
  cascade-vs-cell distinction, the unit-frame iteration, the corrected verdict bands,
  and whichever of Stage F ran. Do not replace Run 12.

**λ policy.** If the octave framing is real, `λ=0.5` is distinguished and a 27-point
`(λ_w,λ_s,L)` sweep mostly tests the *parameterization*, not the physics. **Fix
`λ=0.5` (dyadic)** and spend the budget on **depth and neighborhood robustness** at
that point. Only sweep `λ` to check the *dichotomy's* λ-independence, and if so, apply
the e-folding rate conversion before any `0.215` comparison.

---

## 13. Falsification (pre-registered, both-direction honesty)

- **Falsifies unbounded:** at a genuine fixed point (A1/A2 clean), `δ_k` fits
  `c·γ^k` with `γ` bounded away from 1, `k·δ_k→0`. Then even the fair cell decays.
- **Falsifies bounded:** `δ_k` bounded below by `c>0` across many octaves (gadget),
  confirmed in exact arithmetic (§7). One witness suffices (§4 asymmetry: lower bound
  needs one cell; upper bound needs universality).
- **Inconclusive (the expected outcome if the ansatz is still wrong):** `env_distance`
  never stabilizes; **or** `γ ∈ [0.9,1.0]` with geometric and power-law fits both
  viable; **or** A1/A2 fail. Report as such — do **not** dress a knife-edge or a
  collapsing/under-resolved construction as "bounded" (the Run 11–12 failure) or annex
  the near-1 band to "unbounded" (§6).

---

## 14. Non-goals

- No GEN0.md §6 optimal-control solver; `V₀` answers nothing (§2).
- No pinned-arm equality LP (§8).
- No seeds/multistart anywhere in the cell (§12).
- No verdict from the depth-4 global simulation alone (§4), from a 2-point "flat"
  (§6/§12C), from `mesh_cross_check` (§6), or from a fixed point whose `ρ` rescaling
  or E-sufficiency was never checked (§4).

---

## 15. Exact vs guessing (honesty ledger, from `CRITIQUE.md`)

**Exact/established:** tent cap `≤2`; both budgets; the curvature-measure coordinate;
weight-LP convexity; `δ = ΔJc` already nets the re-arm cost (LP re-solves against the
whole structure — **do not add a separate cost term**, it double-counts); "any `J>2`
forces moving curvature"; and (Stage 0) `J* ≥ 3.05` once rationalized.

**Assumed/guessed (flagged, to be tested, not trusted):** self-similarity of the
*true* optimizer (pattern-matched from ~2 resolved levels); existence/accessibility of
`E*`; **E-sufficiency (A1)**; the anisotropic scaling `(w,s,r ∼ 2^{-k}, L ∼ O(1))` as
*the* family vs *a* family; identifying "one dyadic octave" with "one generation" (the
λ-conversion caveat, §12).

---

## 15A. Stage-C build progress (this branch: `run13-cell`)

Done and gated:
- **`lp.py` — both injection channels** (§5A). `lp_weights_f(..., lip_rhs=, rise_cap=)`
  and `lp_weights_g(..., lip_rhs=)`. Verified: defaults **bit-for-bit** unchanged;
  explicit `ρ=1-|x|` is a **no-op** vs the implicit budget (8e-16), i.e. the new
  rise-cap row faithfully generalizes it; a spent `ρ=0.7(1-|x|)` **binds** (J 2.36→1.65).
- **`cell.py` — `flat_env`, `env_to_lp`, `_alternate_injected`, `cell_solve`, `_flat_gate`.**
  `python3 -m kink_opt.cell`: flat-E injection reproduces the plain carrier (**diff 0.0**).
  Coupling is monotone on coarse J (`ρ×0.7→0.123`, `ρ×0.4→0.070`, `β0.7→0.086` vs flat
  `0.176`).

- **`certify` injection passthrough** (`verify.py`). Found while building: `certify`'s
  fine-grid repair re-solved the weight LPs *without* the injection, washing it out (`Jc`
  recovered the full flat 0.176 for every non-flat E). Fixed — `certify(..., lip_rhs=,
  rise_cap=)` interpolates `lip_rhs` onto the fine grid and re-applies both. Now `Jc`
  **tracks** coarse J (`ρ×0.7→0.123`, `ρ×0.4→0.070`, `β0.7→0.086`), all `constraints_ok`.
  Default None = unchanged (regression: run3 `Jc=2.3089`, run5 `2.5153`).

Remaining for Stage C is now broken into ordered, self-contained implementation
plans under **`plans/run13/`** (readable cold by a fresh agent — `00-primer.md`
carries the shared context, then tasks `01`–`05`):
- `01-read-off.md` — `cell_read_env` (E′ read-off) + `cell_env_distance` (ρ/r-normalized).
- `02-d1-gate.md` — `check_interior_slope` (D1 arm-only-sufficiency verification).
- `03-fixed-point-loop.md` — `cell_step`, `fixed_point`, `tiling_gain` (the `E→CELL(E)` loop + `2^k` tiling multiply, `γ=2r`).
- `04-verdict-and-sweep.md` — `cell_sweep` + the §6 decision rule → the bounded/unbounded verdict.
- `05-sufficiency-probes.md` — A1 (E-sufficiency) / A2 (ρ/r sensitivity), deferred.

**Fork D3 (ρ/r rescaling) and the read-frame are RESOLVED** — derived in
`plans/run13/00-primer.md §0.6`: `r` = the frame's per-octave width(=amplitude)
contraction (unit-frame Lipschitz forces f-amplitude ∝ half-width, so rise budget
`ρ` carries amplitude units and injects as `ρ/r`); `β` is frame-invariant (no
rescaling); the verdict ratio is `γ = 2r` (tiling count 2 × per-cell J-scale `r`),
with `r=1/2` the log-growth knife-edge; E′ is read at `t̂=1/2` (midpoint — `t̂=1` is
terminal-pinned, `t̂=0` is the injection seam). `r` is a *chosen* frame parameter,
swept like `λ_w`, not a solved unknown. D1 stays arm-only (task 02 verifies);
revisit D2 (signed slope) only if task 05's E-sufficiency probe fires.

### Run 13 result (Stage C)

Tasks 01–04 (`cell_read_env`, `cell_env_distance`, `check_interior_slope`,
`cell_step`, `fixed_point`, `tiling_gain`, `cell_sweep`, `_verdict`) are implemented
in `kink_opt/cell.py` and wired into `python3 -m kink_opt.cell`. Sweep over the
frame contraction `r ∈ {0.35, 0.45, 0.5, 0.55, 0.65}` (12 fixed-point iterations,
`tol=1e-4`, `coarse_N=8`, `outer=40`, `sub=8`):

| r    | converged | dist_final | δ̂*    | γ_geom (=2r) | γ_emp (last) | verdict   |
|------|-----------|------------|--------|--------------|--------------|-----------|
| 0.35 | True      | 5.97e-05   | 0.1031 | 0.70         | 0.70         | BOUNDED   |
| 0.45 | True      | 4.64e-05   | 0.1031 | 0.90         | 0.90         | BOUNDED   |
| 0.50 | True      | 4.18e-05   | 0.1031 | 1.00         | 1.00         | UNBOUNDED |
| 0.55 | True      | 3.80e-05   | 0.1031 | 1.10         | 1.10         | UNBOUNDED |
| 0.65 | True      | 3.21e-05   | 0.1031 | 1.30         | 1.30         | UNBOUNDED |

The flat-E no-op gate still prints `ok=True` (`diff=0.00e+00`) before the sweep —
existing tasks 01–03 are untouched.

Every `r` converges (env-reproduction distance ~1e-4-1e-5, well under `tol`), and
`γ_emp` tracks `γ_geom` exactly at every `r` (no drift), so the sweep's own internal
consistency check passes: the verdict moves monotonically with `r`, small `r`
BOUNDED, large `r` UNBOUNDED, flipping exactly at `r=0.5` (the pinned knife-edge —
note neither `r=0.45` nor `r=0.5` land inside the nominal `(0.9,1.0)` INCONCLUSIVE
band; `γ=2r` is deterministic in `r`, so this specific 5-point grid happens to hit
the boundary values `γ=0.9` and `γ=1.0` exactly rather than straddling them).

**Load-bearing caveat (carried over from task 03, restated here because it changes
what this table means):** `δ̂*` is bit-identical (≈0.10307) across every `r` tested.
This is not a coincidence of a well-behaved fixed point — it's because `env_to_lp`'s
channel-2 rise cap is `ρ/r`, and for every `r` swept here (`r<1`) dividing by `r`
only *loosens* that cap, so channel 2 (the rise budget) never binds; the entire
harvest is set by channel 1 (arm-only slope) alone, which converges to its own
`r`-independent fixed point (`β*≈0.7`ish giving `δ̂*≈0.103`). Consequently:

**The r-sweep verdict above is a GEOMETRIC verdict, not a physical one.** It
correctly restates the pinned tiling identity `γ=2r` and shows the fixed-point
machinery (env read-off, distance, iteration, tiling multiply) all work and are
internally consistent — that part is a real, exercised result. But because `δ̂*`
does not vary with `r` in this regime, the table cannot by itself tell you which
`r` is the *actual* physical per-octave contraction of the optimizer's own
generation sequence, nor whether that physical `r` falls above or below 0.5. In
other words: **this sweep proves the knife-edge exists and the machinery finds it
correctly, but does not yet locate which side of the knife-edge the real problem
sits on.** Pinning that needs either (a) the physical `r` read off an actual
optimizer-driven contraction (e.g. from `construct.py`/`melt.py`'s hierarchy, not
chosen by hand), or (b) a regime where channel 2 genuinely binds (a child that is
rise-limited, not just slope-limited) so `δ̂*` can respond to `r` and the sweep
tests something beyond geometry alone.

**Task 05 (A1/A2 sufficiency probes) — run, results below.**
`probe_e_sufficiency`/`probe_rho_rescaling_sensitivity` in `kink_opt/cell.py`.
- **A1 (E sufficient?):** the task file's literal recipe (same seed, `outer` doubled)
  is **degenerate** here — `_alternate_injected` converges in ~1-2 iterations, so
  `outer=40` vs `80` gives bit-identical states (`env_gap=0`, a vacuous pass). Replaced
  with a genuine internal perturbation (a small constant carrier-offset jitter,
  `xi_jit=2e-4`, giving a different converged `Jc` 0.1759 vs 0.1757). Real result:
  `env_gap=6.06e-4` (< `tol_env=1e-3`, environments genuinely match) yet
  `delta_gap=1.65e-5` (≪ `5e-3`), **`ok=True`**. Supports E-sufficiency at this
  operating point — but only against *one* perturbation direction; does not exclude a
  missing coordinate (drift velocity, D2 signed slope) that this jitter doesn't excite.
- **A2 (ρ/r-convention robust?):** `δ_star` identical across convention factors
  `{0.8, 1.0, 1.25}` at both `r=0.5` and `r=0.65`. **This is the trivial/inert outcome,
  not evidence the convention is right:** per the channel-2-inert finding above, `ρ` is
  never binding for `r<1`, so nothing scaled on `ρ` can move the answer. A2 is
  uninformative until a rise-binding regime exists. A1 is the meaningful probe of the two.

Combined honest summary: **the sweep is internally consistent and monotone as
designed, A1 supports E-sufficiency at this point, but the verdict is still geometric
(δ_star r-invariant, channel 2 inert) — a validated instrument, not yet a pinned
answer for the real problem.** The unblocking next step is not in this task set: pin
the *physical* `r` from an optimizer-driven contraction (`construct.py`/`melt.py`), or
construct a rise-limited regime where channel 2 binds so `δ_star` can respond to `r`.

### Rise-binding probe (Stage C follow-up) — both unblocking levers, run

Directly chases the two "unblocking next steps" named just above. New in `cell.py`:
`rise_binding_report(sol, rise_cap)` (channel-2 activity diagnostic — per rise-cap row
slack `rho - sum_i A[k,i]*hat(xs;XI[k,i])`, reports `frac_binding`/`min_slack`) and a
`rho_scale` knob on `flat_env` (shrinks the seed rise budget to force channel 2 to
bind). Four findings, in order:

1. **Baseline / detector calibration.** At the flat seed `frac_binding≈0.0217`
   (≈1 row per time node) with `min_slack=0` — the kink apex touching the `1-|x|`
   envelope. Structural corner, benign; channel 2 inert beyond it, as §15A already said.
   *Detector caveat:* because the seed `rho` shares the envelope's `1-|x|` **shape**,
   `frac_binding` pegs at ~0.0217 even deep in binding (only apex rows touch, but that
   one contact scales global f-amplitude) — so `frac_binding` **understates** activity;
   the honest signals are `min_slack→0` **and** `δ̂(r)` moving.

2. **Shrinking seed `rho` unlocks r-dependence.** Sweeping
   `rho_scale ∈ {1,0.5,0.25,0.1} × r ∈ {0.35,0.5,0.65}` (one `cell_solve` each): at
   `rho_scale=1` δ̂ is flat 0.1757 across r (inert, cap `=(1-|x|)/r` loosens with r); as
   `rho` shrinks the cap `= rho_scale·(1-|x|)/r` drops below the envelope, binds, and
   **δ̂ becomes r-dependent, δ̂ ∝ rho_scale/r** (e.g. at `rho_scale=0.25`: δ̂ =
   0.1255/0.0878/0.0676 for r=0.35/0.5/0.65; ratio 0.0878/0.0351 = 2.50 = the
   `rho_scale/r` prediction). So δ̂ is **not intrinsically** r-invariant — only inert at
   the flat operating point.

3. **But the rise-limited regime is transient — RELAX-BACK, universal.** Running the
   full `fixed_point` loop from a shrunk seed (`rho_scale∈{0.1,0.25}`, r∈{0.5,0.65}):
   step 0 binds (imposed), **step 1 binding vanishes** (`frac_b→0`, `min_slack` 0→0.6-0.75,
   `rho_mid` rebounds 0.10→0.85), steps 2-9 converge to the **same** slope-limited fixed
   point `δ̂*=0.10307, rho_mid=0.6894` — bit-identical to the flat seed, every r.
   Mechanism confirmed exactly: rise cap → shallow f → residual `ρ'=(1-|x|)+f≈1-|x|`
   (big, capped f consumed little budget) → next cap loose → slope retakes control. **A
   single-kink cell cannot sustain rise-limitation — one capped kink refills its own
   budget.** So the geometric verdict γ=2r is the cell's *honest* answer (channel 2
   inert at the attractor, not just at the seed), and sustaining rise-binding needs an
   ansatz that keeps *consuming* the budget — a **band** of kinks (melt.py's `K`-band).

4. **Physical `r` off `construct.py` — no clean sustained rate.** Achieved per-octave
   amplitude contraction (peak LP f-weight/gen, `n_gen=4`, `scale_t=scale_x=0.5`):
   `0.400,0.2125,0.1607,0.1534,0.1510`, ratios `0.53,0.76,0.95,0.98`. Only the **first**
   octave obeys `scale_x` (ratio 0.53 ≈ 0.5, i.e. it sits **at the knife-edge**); beyond
   gen 1 amplitude **plateaus at ~0.15** (ratio→1, a collapse floor not a contraction
   rate) while harvest dJk = `0.1065,0.0203,0.0062,~0` collapses (ratio ~0.19-0.30). The
   split is the Run 11 co-location collapse: deep gens keep weight but harvest nothing
   (redundant hats at the shared anchor). So construct.py furnishes **no single physical
   `r`** to plug into γ=2r — first octave ≈0.5, but not self-similar past it, and the
   harvest-bounded reading is *confounded by the same anchor collapse* (Run 11's caveat,
   not independent evidence).

**Combined reading (unchanged verdict, sharpened wall).** All three levers — cell fixed
point, rise probe, construct.py — **agree the knife-edge is r=0.5 and the first octave
sits on it**, and all three hit the **same wall**: every construction here collapses the
anchor (co-locates deep generations), so none measures a *sustained, non-collapsing*
per-octave rate. That is the genuine open frontier, unchanged from Runs 11-12: **a
hierarchy whose generations keep moving to genuinely new locations instead of
co-locating.** The one structural lead is finding 3's — a band ansatz (melt.py's
`K`-kinks) is the candidate that could *both* sustain rise-binding *and* avoid
co-location; building a band-cell is the next real experiment, not a quick follow-on.

---

## 16. One-paragraph summary

Boundedness of `sup J` is decided by the repeating per-octave increment `δ_k`
(ratio `γ`), not the generation-0 optimum `V₀ ≈ 2.6`. `γ` is only meaningful at a
fixed point of the transfer operator `CELL: E ↦ (δ̂, E′)` — and the decisive
methodological upgrade over Runs 9–12 is to **iterate that unit-frame cell in
rescaled coordinates at O(1) cost per octave**, not to simulate the growing global
cascade (whose depth-4 ceiling reproduces the very two-data-point weakness this plan
faults the mesh for). The verdict band is corrected: `γ` near 1 is inconclusive
*both* directions (a geometric sum converges up to `γ=1`; a `c/k` tail keeps `γ<1`
yet diverges), so bounded requires `γ` bounded away from 1 with `k·δ_k→0`, and
unbounded requires `δ_k` bounded below — ideally in **exact arithmetic**, which the
piecewise-linear cell permits and which alone earns the word "proof." Two assumptions
(E is a sufficient statistic; ρ rescales by the rise share) can fabricate a fixed
point and must be tested. Independently of all of it, rationalizing the existing mesh
solution banks an unconditional `J* ≥ 3.05` today. Optimizing generation 0 is on no
critical path.

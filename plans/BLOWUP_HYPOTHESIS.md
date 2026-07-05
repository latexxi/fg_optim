# Hypothesis: J -> infinity via "melting" (sweeping curvature bands)

## Status: HYPOTHESIS, distilled from the empirical mesh finding in
`plans/MELTING_KINKS.md` (data: `~/fg_opt3/data/level_k0*.npz`, best point
`level_k06.npz` with certified J = 3.055). Not proved either way. This file
states the proposed blow-up mechanism precisely so the constructive arm can
target it.

## The mechanism, in four steps

1. **Curvature mass is fixed.** Every x-slice of a feasible f or g carries at
   most 2 units of total slope change (the Lipschitz cap `|f_x| <= 1` on both
   arms). A tent concentrates all of it as a single Dirac delta at one point,
   so `g_xx` only ever harvests `f_t` at that one x — the accounting
   telescopes to exactly J = 2 (the tent cap).

2. **Melting redistributes that fixed mass — it adds nothing.** The high-J
   mesh optima keep the boundary slices as exact full-depth centered hats
   (`f(.,0)`, `g(.,1)`: depth 1.0000, apex at 0 — these stock the rise
   budget), but the *interior* slices spread the curvature delta into a
   finite-width band (width ~0.47 at k06; largest single cell holds only 25%
   of the mass) that **drifts in x** as t advances. Arms stay straight at
   slope ±1; only the sharp V-bottom becomes a wide, rounded, moving basin.

3. **Sweeping is why J grows.** `J = ∫∫ f_t · g_xx`. A drifting g-curvature
   band overlaps `f_t` at every x it visits, and each newly visited x still
   has its untapped per-x rise budget `∫ f_t dt = -f(x,0) = 1 - |x|` waiting.
   Same <= 2 curvature mass, far more overlap with `f_t`. Dually, f melts so
   `f_t` is a tall, narrow, *moving* spike co-located with g's band. This is
   the L/w amplification of STRATEGY.md Section 3 stated geometrically:
   spread the curvature thin (small w), drag it far (large L).

4. **Blow-up = the melt cascades self-similarly.** The coarse drifting basin
   carries a finer, faster sub-melt riding inside it, which carries a finer
   one still — one new generation per dyadic octave of x-resolution. If each
   octave contributes a roughly constant dJ, then J ~ c · log2(Nx) -> infinity
   as the mesh refines, and H[f] = ∫ sup_x f_t dt -> infinity with it (the
   PDF's open question 2), because the melt's f_t spike keeps sharpening.

## The measured signature (and its strength)

Refining Nx and Mt **together** (the only fair comparison — see confound
below):

- k04 -> k05 (Nx 17->33):              dJ = +0.211
- k05 -> k06 (Nx 33->65, Mt 129->257): dJ = +0.219

Two nearly identical increments, ~ **+0.215 per dyadic level**, i.e.
`J ≈ 2 + 0.215·(level-3)`. H[f] grows alongside: 1.0 (tents) -> 2.53 -> 2.68
-> 3.26. Constant per-octave increment is exactly the log-law signature.

## The precise blow-up condition

> H[f] -> infinity IFF the melt can keep sharpening (w -> 0) without the
> **re-arm cost** — maintaining the finer band consumes the coarser band's
> convex budget — eating the per-octave gain. Constant dJ per octave means it
> does not get eaten; geometrically decaying dJ means it does, and J is
> bounded.

The mesh says "not eaten" over two clean octaves. Two points cannot
distinguish constant from slowly decaying — same trap as Run 9's three-point
ladder, one level up in absolute J.

## Known confound (why the mesh cannot settle this)

The measurement is budget-sensitive in exactly the Run 9/10 pattern:
**t-resolution must scale with x-resolution or J goes backward.**

- k07 (Nx=129, Mt=129): J = 2.979 < k06's 3.055 — refined x, starved t, went DOWN.
- k08 (Nx=257, Mt=65):  J = 2.845 — badly t-starved.

The fine travelling melt-fronts need fine time to be resolved as they drift.
Cost of a clean octave doubles both axes, so the mesh affords only two of
them. The trend is suggestive, not proof.

## Why this does not contradict Run 11's "bounded" result

Run 11's construction (`kink_opt/construct.py`) anchored every generation at
ONE shared collapsing point with ONE kink per generation — a rigid spike
cascade. That *bans the sweep*: co-located hats are redundant in a convex-hat
sum, so of course it saturated. Run 11 proves bounded J **for that anchoring
only**. The melt hypothesis is precisely the anchoring Run 11 did not test:
generations that keep moving to genuinely new x, reaping each location's
independent rise budget.

## Falsifiable prediction / decisive experiment

Rebuild the constructive arm (LP-only weights, no position NLP — Run 11's
artifact-free virtue) so each generation is a **drifting curvature band of
finite width**, not a point:

- generation k = band of width `w_k ~ scale_x^k` drifting along an x-path of
  length `L_k` over a lifetime window `~ scale_t^k`, riding ON generation
  k-1's band (not collapsing to its endpoint);
- boundary slices pinned to full hats (the budget stock);
- read per-octave dJ from the certified LP fixed point.

Prediction if the hypothesis is TRUE: dJ per octave tends to a positive
constant (=> sup J = +infinity, H[f] -> infinity).
Prediction if FALSE: dJ decays geometrically, and the decay is attributable
to the re-arm cost (finer band eating coarser budget).

Either outcome is a proof-quality answer; the mesh (cost-capped) and Run 11
(wrong shape) can each only suggest.

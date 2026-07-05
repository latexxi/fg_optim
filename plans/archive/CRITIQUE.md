> **ARCHIVED — FOLDED INTO `plans/run13-selfreproducing-cell.md`.** This critique's
> substance is now in the active plan (see its §15 honesty ledger, §6 corrected
> decision band, §2 wrong-target argument). Kept for provenance only.

# Whacking Run 13

## What's genuinely strong

**The reframing in §2 is the best thing in the document.** Identifying that the verdict lives in the repeating increment δ (≈0.2) and not in V₀ (≈2.6), and that subtracting two heavy solves to read a signal 10× smaller than the solves' noise floor is doomed, is exactly right. Same for the knife-edge argument: at γ ≈ 1 a few-percent artifact flips the verdict, so fairness is load-bearing.

**The collapsing-anchor diagnosis is correct and important.** A γ < 1 forced by co-locating every generation on one point is not evidence — it's the construction's own contraction being measured. Naming this and refusing to relitigate Runs 11–12's "bounded" as a result is good hygiene.

**The accounting note in §7** (δ = ΔJc already nets the re-arm cost because the LP re-solves against the whole structure; don't add a cost term) prevents a subtle double-count that would otherwise be very tempting.

**§8 (don't pin arms)** is a sharp catch — the handoff regime is exactly where mass ramps, and an over-literal reading of GEN0 would have destroyed the physics at the one place the cell lives.

**Pre-registered falsification criteria (§10)** and the asymmetry of §7 (lower bound needs one witness; upper bound needs universality) are both correct and correctly deployed.

## The biggest whack: the plan never builds the cell it advertises

The paper promises `CELL: E ↦ (ΔĴ, E′)` — a *unit-frame* problem, O(1) kinks over O(1) time, one map iterated. The plan implements something else: build the **global** hierarchy with `build_melt_hierarchy`, certify the global J on a dense grid, and read increments and environments off it. That's a *simulation of the cascade*, not an iteration of the transfer operator. Two consequences:

1. **Cost doesn't actually beat the mesh by much.** Certifying generation k on a dense global grid still has to resolve width-2⁻ᵏ features. You've traded O(4ⁿ) mesh cost for something that still grows with depth, so n_gen = 4 is what you get — and then γ is read from *two, maybe three ratios after discarding the birth transient*. That is the same "two data points" the plan (rightly) mocks the mesh for, one level up. The whole advertised advantage — arbitrary depth at fixed cost because every generation is the same unit-frame object — is only real if you solve the octave *in rescaled coordinates with E as boundary data* and iterate that. That solver doesn't exist in the inventory and isn't in the stages. This is, in my view, the actual missing construction, more than the nested anchor.

2. **The exactness claim silently evaporates.** The paper's "why it is a proof, not a plot" rests on the cell being piecewise linear ⇒ both lemmas are finite lists of rational inequalities, checkable exactly. No stage does this. Stage D reads γ numerically; Stage E is a demo. As written, the deliverable is a third suggestive floating-point number — fairer than the mesh, but the "decidable question, provable in exact arithmetic" headline is unfunded. Add a stage: at the stabilized point, freeze the construction, rationalize all breakpoints/weights, and verify feasibility + δ ≥ c + E′-containment in exact arithmetic. Until then, say "measurement," not "proof."

## The dichotomy has a hole as an *experiment*

At an exact fixed point, γ constant < 1 ⇒ geometric ⇒ bounded, γ ≥ 1 ⇒ divergent. Fine as algebra. But the experiment observes 2–3 ratios and eyeballs "flat," and there's a divergent scenario the decision rule misclassifies: δ_k ~ c/k gives γ_k = k/(k+1) < 1 at every k, drifting slowly toward 1, and Σδ_k = ∞. Over two or three octaves, γ ≈ 0.75–0.9 and "roughly stable" is exactly what that looks like. So "γ < 1 stably ⇒ bounded" is only valid if γ is *bounded away from 1 with no upward drift*, which two ratios cannot establish. Fit δ_k against both c·γᵏ and a + b/k, or track k·δ_k; if you can't distinguish them, the honest verdict is inconclusive.

Symmetrically, Stage D's "γ ≥ 0.9 ⇒ evidence for +∞" is the motivated reading §10 warns against. γ = 0.95 *stably* is a convergent sum, full stop — nearness to 1 makes J large, not infinite. The only clean unbounded verdict is δ_k bounded below (gadget lemma with fixed c > 0), i.e. effectively γ ≥ 1 in exact arithmetic. The band 0.9–1.0 should be labeled "inconclusive, knife-edge" in both directions, not annexed by the unbounded side.

## Untested assumption doing all the work: E is a sufficient statistic

The map-on-environments framing assumes the pair (β, ρ) at one read time *screens off* everything else — that the child's achievable δ depends on the coarser structure only through E. But the child's weights are LP-solved against the **entire accumulated structure**, including the parent's motion during (or after) the child's window. If deep structure leaks past E, then "E′ = E" doesn't license iteration: the composition lemma quantifies over environments, but the dynamics aren't a function of the environment. This is testable and cheap: perturb the deep generations while holding (β, ρ) at the read time fixed (within tolerance), and check δ invariance. If δ moves, E needs more coordinates (e.g. the parent's local drift velocity at handoff — plausibly the first missing one) before any fixed point means anything.

Related unspecified detail with veto power: **how E rescales**. β is dimensionless slope-slack, fine; ρ is rise budget, which under the anisotropic scaling shrinks like the rise share r_k. If `env_distance` compares ρ profiles without normalizing by r_k (or with the wrong normalization), you can manufacture or destroy a fixed point purely by the choice of units. The plan never states the normalization. State it, and check the verdict is insensitive to reasonable alternatives.

## Geometry ambiguity: nested vs sequential vs tiling

§6 says the child is a *sub-window of the parent* ("nested in time") and, in the same paragraph, is anchored at the parent's *window-end* — which makes windows sequential/disjoint, not nested. These are different physical mechanisms: a child inside the parent's window co-harvests against the parent's live f_t; a child after it inherits only residue. Meanwhile the paper's Fig. 3 shows a third geometry: each generation **tiles** its parent's window with ~2ᵏ copies, and the per-octave gain is (count) × (per-child δ) — which is actually the arithmetic that matches the mesh's constant-per-octave story most naturally. One child per octave earning O(1) via r·L/w is a *different* claim from 2ᵏ children each earning O(2⁻ᵏ). The plan should pick one geometry explicitly, and if it's single-child, say why the count factor isn't needed. As written, `anchor="nested"` is underdetermined enough that two implementers would build different experiments.

## Exact vs guessing, sorted

Exact: the tent cap ≤ 2; both budgets; the pairing/curvature-measure coordinate; convexity of the weight LP; δ = ΔJc netting; "any J > 2 forces moving curvature." Also — and the plan misses this as a free deliverable — the level-6 solution is piecewise linear with LP weights, so **rationalizing it gives an unconditional theorem J* ≥ 3.05** (feasibility checkable exactly, J computable exactly). That's a publishable improvement over 2 that costs an afternoon and doesn't depend on how the γ story ends. Do it regardless.

Guessing: self-similarity of the true optimizer (pattern-matched from ~2 resolved levels); existence and accessibility of E*; E-sufficiency (above); the specific anisotropic scaling (w, s, r ~ 2⁻ᵏ, L ~ O(1)) as *the* right family rather than one family; and the identification of "one dyadic octave" with "one generation" — note that if λ_w, λ_s are swept away from 0.5, "per-octave" comparisons to the mesh's 0.215 need a rate conversion (γ per e-folding), which the plan and `mesh_cross_check` never do. The dichotomy itself is λ-independent, but the mesh cross-check isn't.

One more structural observation worth exploiting rather than guessing around: the problem is symmetric under (f, g, t) ↦ (g, f, 1−t), and the computed optimizers visibly nearly respect it. Test whether imposing g(x,t) = f(x, 1−t) costs anything at level 5–6. If it's free, the cell carries one function instead of two — half the LP, and the exact-arithmetic lemmas get materially shorter.

## Smaller weaknesses

Stage C's acceptance is "env_distance clearly below its 1→2 value and flat across the last 2 generations" — flatness over two points is not flatness; a slowly contracting or slowly escaping env looks identical. With n_gen = 4 you structurally cannot do better, which is another argument for the unit-frame solver. Also add a per-generation **resolution gate inside certify**: finest kink spacing vs dense-grid spacing at sub = 12 — at λ_w = 0.4, generation 4 lives at width ≈ 0.026, and if the certification grid under-resolves it, the starvation artifact the plan was written to kill re-enters through the back door, now hidden inside the "honest" number.

The `mesh_cross_check` factor-of-3 bar spans 0.07–0.65 around 0.215 — at a knife-edge where 5% decides, a factor-3 gate is decoration. Keep it as a sanity check, but strike it from the Stage D decision inputs.

## Unnecessary

Stage A is ritual — the Run 12 numbers are already recorded — but at ~1 minute it's harmless; keep it as a regression test, not a stage. §2/§11's relitigating of GEN0 is useful documentation but could be one paragraph. The knife-edge sermon appears four times (§0, §2, §9, §10); once, stated sharply, would land harder. And the 27-point (λ_w, λ_s, L) sweep deserves a second look: if the octave framing is real, λ = 0.5 is distinguished and the sweep is mostly testing the parameterization, not the physics — either fix λ dyadic and spend the budget on depth and neighborhood robustness at that point, or accept that γ from λ = 0.4 generations answers a slightly different question than the one the mesh asked.

## Bottom line

The plan's diagnosis (measure δ/γ at a fair fixed point, not V₀; the old anchor was rigged) is right and well-argued. Its execution has three gaps that each independently threaten the verdict: it simulates the cascade instead of iterating the unit-frame cell (so depth and cost don't improve on the mesh as claimed), it has no exact-arithmetic stage (so "proof, not plot" is aspiration), and it rests on an untested sufficiency assumption about E with an unspecified rescaling convention. Fix those, tighten the decision band (0.9–1.0 is inconclusive, both directions), resolve the nested/sequential/tiling ambiguity, and pocket the free theorem J* ≥ 3.05 now.

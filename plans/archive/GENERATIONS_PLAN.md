> **ARCHIVED — FOLDED INTO `plans/run13-selfreproducing-cell.md`** as its conceptual
> companion (the same induction, informal). Kept for provenance only.

Here is the whole cascade as one inductive idea. It's self-contained — you only need the setup from the note (maximize `∫∫ fₜ g_xx`; `f` rises and `g` falls in time; both convex in `x` with slopes capped at ±1; the two end-tents fixed).

## The one mechanism (what every generation does)

Two budgets are fixed and cannot be created, only relocated. **Curvature:** at any instant `g`'s total kink strength is at most 2 (its slope can only travel from −1 to +1 across the strip). **Rise:** each column `x` has only `1−|x|` of upward travel to give, ever. The objective pays you exactly when `g`'s curvature sits on top of `f`'s rise. Parking the curvature still (a fixed tent) spends one column's rise and caps you at `J = 2`. The only way to earn more is to **drag the fixed curvature across fresh columns**, collecting each column's untapped `1−|x|` as you pass. Call one such pass a **sweep**. Every generation is a sweep; they differ only in scale.

## Base case — generation 0

The coarsest sweep. Take the two end-tents and let the interior vertex melt: `g`'s curvature, instead of standing at the center, drifts slowly across the whole interval over the whole time `[0,1]`, while `f`'s rise tracks the same moving spot. One curvature band, one traversal, spread over the entire budget. This already beats the stationary tent (it reaches ~2.6), because it harvests several columns instead of one. But it is a *single* pass: by the time it finishes it has skimmed each visited column once and stopped. That's the whole of generation 0 — one melt, one drift, done.

## Inductive step — generation *k+1* from generation *k*

Here is the key observation that makes it a recursion rather than a one-off. When generation `k` finishes its sweep, it does **not** leave the strip exhausted. It leaves behind a *smaller, self-similar copy of the original problem*: down at the floor of the basin it just carved, there is still slope slack to bend, and there is still unspent rise budget in the columns it only skimmed. That leftover — the note calls it the **environment** — has the same shape as the strip generation 0 started in, only rescaled: narrower in `x`, shorter in time.

So you do the *same move again, one size down*. Generation `k+1` is a copy of the generation-`k` sweep with half the width and half the lifetime, riding along the drifting floor its parent laid down, harvesting the budget the parent left behind. Because the lifetime halves, **two** of these fit back-to-back inside the parent's window; because widths halve too, each tiles the parent's basin. Repeat: generation 2 is four still-finer sweeps, generation 3 is eight, and so on — one new sweep-generation per dyadic octave, each a rescaled clone of the last, each feeding on the residue of the one above it.

That "each generation hands its child an environment of the same shape" is exactly what turns a pile of increasingly fine constructions into a single repeated map. You are not solving infinitely many different problems; you are iterating **one** problem — the *cell* — on an environment that reproduces itself.

## What the induction decides

Every generation collects some gain `ΔJ_k` and pays some cost to set itself up in its parent's residue (the **re-arm cost**). Two facts make the induction rigorous rather than a hopeful extrapolation:

- **Gadget (the step works):** one explicit generation, dropped into any such environment, nets a definite positive gain after paying its cost.
- **Composition (the step repeats):** the state it leaves behind contains a rescaled copy of the same environment — so the gadget applies again, forever.

Chain them and the total is `J = ΣΔJ_k`. Now everything hinges on a single number, the ratio of consecutive gains, `γ = ΔJ_{k+1}/ΔJ_k`, read at the self-reproducing environment:

- if `γ ≥ 1`, each octave earns as much as the last, the sum diverges, and **`J = +∞`**;
- if `γ < 1`, the gains decay geometrically, the sum converges, and **`J` is bounded**.

That is the payoff of casting it inductively: the infinite question "does stacking ever-finer sweeps blow up?" collapses to one finite, decidable comparison inside a single generation — does the child out-earn its re-arm cost, or not. (The measured value sits right at `γ ≈ 1`, which is why the base-case-plus-induction framing matters instead of trusting two data points.)

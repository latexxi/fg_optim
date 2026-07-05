# Ideas

> **REFERENCE — foundational idea scratchpad, still current.** The framing here (harvest
> accounting, travel-beats-static, the log-growth / renormalization question) is the seed
> of the whole project. Developed forms: `STRATEGY.md` (briefing) and
> `plans/run13-selfreproducing-cell.md` (the active RG-cell plan).

## Core problem framing

Maximize a coupled functional J[f,g] over two constrained space-time
functions (f convex-in-x, rising in t; g convex-in-x, deepening in t),
each pinned to zero on boundary/terminal/initial conditions and Lipschitz
bounded. Static ("non-traveling") solutions cap out at a fixed constant.
Mesh-based numerics show J growing like ln(mesh resolution) as resolution
increases, instead of saturating — that empirical growth law is the thing
worth explaining.

## Objective as "harvest"

Reading g's curvature as point masses (kinks) and interpreting J as: total
rise of f, harvested exactly at the locations g's kinks pass through,
weighted by kink jump size. This reframing turns an integral identity into
an intuitive accounting problem (where/when does f "pay off").

## Why traveling beats static

A stationary kink can only harvest each unit of f-rise once — accounting
telescopes, bounding J. A *traveling* kink sweeps a path and re-harvests
fresh f-rise at every point along the way, provided f maintains a
co-moving narrow rising front at the kink's location. The gain scales with
path length over front width — an amplification mechanism absent from
static schedules.

## The central open question: why log growth, not linear?

Naively, narrower fronts (more "passes") should scale J linearly with
mesh resolution. That's not observed — growth is logarithmic. Proposed
explanation: a **hierarchical maintenance cost**. A narrow traveling front
at scale w is itself sustained by a coarser structure at scale 2w, which
in turn taxes scale 4w, and so on — a dyadic recursion. If each scale
("generation") contributes only a roughly constant increment to J, and
there are ~log2(1/w_min) generations, total J grows like the log of
resolution. This is framed explicitly as a renormalization-group style
picture: solve one generation, measure what it hands to the next, iterate,
rather than resolving all scales simultaneously on one large mesh.

Supporting anecdotal signal: mesh solutions show self-similar
"snaking inside snaking" — finer-generation kink paths appear to ride on
top of coarser-generation paths.

## Why a mesh-free representation matters here

A fixed mesh cannot distinguish "J keeps growing forever, arbitrarily
slowly" from "J saturates" — mesh spacing itself floors how narrow a front
can get. A representation where front width can shrink continuously toward
zero (rather than being bounded below by grid spacing) makes the log-growth
question directly testable rather than inferred from extrapolation.

## The decisive experiment: generation-gain measurement

Add one new generation of short-lived, spatially narrow kinks at a time
(mimicking the next dyadic scale down), re-optimize, and measure the
*increment* to J contributed by that generation alone. Plot increment vs.
generation index:
- Increment roughly constant across generations => J is unbounded, growing
  like (constant) x (number of generations) x log(1/finest width) —
  the kink-coordinate analogue of the mesh's log-growth, but without a
  mesh ceiling; supports "supremum is infinite, only approached."
- Increment decays geometrically => J is actually bounded; the mesh's
  apparent log growth was a finite-resolution transient.

Secondary things worth tracking per generation, as tests of the
self-similarity hypothesis: whether each new generation's lifetime and
spatial extent settle at a fixed contraction ratio relative to the
previous generation, whether jump sizes stay O(1)/Lipschitz-saturated
(scale-invariant), and whether new kinks physically sit "on top of" the
parent generation's path.

## Supporting conceptual extensions (not yet built)

- **Topology change during optimization**: the ability to introduce a
  brand-new kink mid-optimization (born with negligible influence so
  nothing is disrupted), and to retire kinks that decay to irrelevance —
  needed because a fixed-count parameterization can't grow the hierarchy
  on its own.
- **Non-uniform time resolution per kink**: fast, short-lived generations
  only need dense time sampling during their brief lifetime; a globally
  uniform time grid wastes effort everywhere else. Idea: let resolution
  follow each kink's own lifetime rather than being shared globally.
- **Warm-starting new generations from rescaled old ones**: if the
  hierarchy really is self-similar, the next generation can be initialized
  as a shrunk copy of the current finest generation (contracted in time
  and space, anchored near the end of its parent's path) rather than from
  a generic/random guess — should converge faster if the self-similarity
  hypothesis holds, and the convergence-speed comparison itself is a test
  of that hypothesis.

## Methodological principles worth keeping

- Never trust the optimizer's own (coarse-grid) objective value —
  always re-check on a refined time grid with constraints re-verified
  densely before treating a number as real. This matters more, not less,
  as generations get faster: the certification resolution has to keep
  pace with the shrinking timescale of the finest generation.
- Because the position search is non-convex (local search only), any
  single run's result could be a mediocre local optimum — repeat from
  multiple distinct starting points and report the spread of outcomes,
  not just the best one.

# 00 — Primer (read this first)

Shared context for all `plans/mesh/` tasks. Fresh agent: read this top to bottom,
then open your numbered task file.

## 0.1 The problem

`J[f,g] = ∫₀¹∫₋₁¹ f_t(x,t)·g_xx(x,t) dx dt`, maximized over f(x,t), g(x,t) on
`x∈[-1,1]`, `t∈[0,1]`, both **convex in x**, **Lipschitz** (|f_x|,|g_x|≤1), with
`f(±1,t)=g(±1,t)=0`, `f(x,1)=0`, `g(x,0)=0`, `f_t≥0`, `g_t≤0`.

Open question: is `sup J` **bounded** or does it grow without limit? The full-mesh
solver shows `J` rising ~`+0.21` per dyadic octave (k04→k05→k06 give J ≈ 2.625,
2.836, 3.055) — consistent with unbounded `J ~ ln(resolution)`, but the sibling
`kink_opt` renormalization cells argue it saturates (strands co-locate, go
redundant). Settling this needs **deeper generations** than the current uniform
mesh reaches before RAM/time blow up. This task track makes each generation
cheaper so we get deeper.

## 0.2 The full-mesh representation

Unlike `kink_opt` (hat basis / kink coordinates), the `mesh/` package is the
**dense** solver: `f`, `g` are `(N, M+1)` arrays of nodal values on a grid
`x_grid` (N points) × `t_grid` (M+1 points). Every node is a free LP variable.
Constraints (convexity in x, |slope|≤1, monotone in t, boundary zeros) are linear,
so each half-problem is an exact LP. `alternating_maximization` alternates
`max_f J | g` and `max_g J | f` — both exact global LPs, J monotone non-decreasing.

Solution is just `(f, g, x_grid, t_grid)`. `compute_J(f,g,x,t)` scores it.

## 0.3 What's already copied (the `mesh/` package)

Copied verbatim from the sibling repo `../fg_opt3/optimize_fg/` (tests + obsolete
analysis/plotting dropped):

| file | contents |
|------|----------|
| `mesh/grid.py` | `make_grids(k,M)` — uniform dyadic grid (N=2^k+1, uniform t) |
| `mesh/constraints.py` | `build_constraints`, `check_feasible`, `idx` |
| `mesh/objective.py` | `compute_J`, **`harvest_per_interval`** (new: per-interval J) |
| `mesh/lp_subproblem.py` | `build_c_f/g`, `solve_f_given_g`, `solve_g_given_f` |
| `mesh/highs_warm.py` | `HiGHSWarmLP` — warm-started persistent LP |
| `mesh/alternating.py` | `alternating_maximization` |
| `mesh/refine_baseline.py` | `interpolate_to_next_level`, `dyadic_refinement` — the UNIFORM baseline, comparison point, **do not modify** |

You will ADD three modules: `mesh/adapt.py`, `mesh/prolong.py`,
`mesh/refine_adapt.py`, and re-export them from `mesh/__init__.py`.

## 0.4 The three gauge facts that make this cheap (verify in the copied code)

These are why the adaptive scheme is grid-construction + interpolation only — **no
LP surgery**. All three are checkable in `mesh/constraints.py` / `mesh/objective.py`:

1. **Time-node POSITION is invisible to everything — only the COUNT matters.**
   `compute_J`: `dt` cancels (sum over node *pairs*, not weighted by `dt`).
   Monotonicity-in-t rows are `f[j+1]≥f[j]` — RHS 0, coeffs ±1, no `dt`. Nothing in
   `build_constraints` or `compute_J` ever reads a `t_grid` *value*; both read
   `len(t_grid)` only. **Consequence (stronger than "gauge"):** the mesh optimum at
   fixed `(x_grid, M)` is IDENTICAL no matter where the M+1 time slices sit in
   [0,1]. Time is purely combinatorial (count + order), carries no geometry.
   **This makes the tau-gauge time regrid a NO-OP for this solver** — clustering
   time nodes at the melt buys nothing the LP can see. `tau_regrid`/`regauge_time`
   are therefore inert here (kept as identity/analysis tools, not solver levers).
   The one real time knob is `M` itself (more slices = finer f_t evolution), and
   even then *where* the extra slices go is irrelevant. tau remains valuable for
   READING solutions (the harvest-gauge inspection), not for driving this mesh.

2. **Non-uniform x is already supported.** Slope rows use `h = dx`; convexity rows
   use `1/h_left`, `1/h_right`; `compute_J`'s `kappa_g` uses per-node `h_left`,
   `h_right`. So a graded (band-refined) x-grid needs zero constraint changes.

3. **Linear-interp inserts are curvature-free = harvest-neutral.** A new x-node
   placed by linear interpolation between two neighbors is *colinear* with them, so
   its discrete `g_xx` (kink) is exactly 0. Inserting it changes neither J nor
   feasibility (same trick `kink_opt` uses in Runs 8/11). The LP then decides
   whether to bend it — and THAT bend is the generation's gain.

## 0.5 The harvest gauge (tau)

Define the harvest CDF from a solved solution:

    dJ_t = harvest_per_interval(f, g, x, t)      # shape (M,), sums to J
    tau  = concatenate([[0], cumsum(dJ_t)]);  tau /= tau[-1]   # (M+1,), monotone 0..1

`tau[j]` = fraction of total harvest collected by time node `j`. Because time is
gauge (fact 1), `tau` is the *natural* time coordinate: equal `tau`-spacing = equal
harvest per node. Empirically (see `plans/gen1-2-3-inspection.md`, invariant I3),
harvest concentrates near `tau*≈0.38` — a universal "melt" event frozen across
generations. Uniform-`t` grids waste nodes on the dead early/late stretches.

## 0.6 The invariant ledger this rests on (from the harvest-gauge inspection)

Established gen1→gen3 (see `plans/gen1-2-3-inspection.md` / `GEN_INSPECT.md`):

- **I2 band frozen:** ≥95% of harvest mass sits in `|x| < 0.4`. Arms `|x|>0.4` are
  straight slope=±1 lines carrying ~no curvature. → refine x only in the band.
- **I3 melt schedule frozen:** harvest-spread bump peaks at `tau*≈0.38`, same每
  generation. → cluster time nodes there.
- **D1/D2 strands double:** curvature concentrations ("strands") number `2^gen`,
  spacing halves — all inside the frozen band. → each octave = twice the band
  x-resolution, arms untouched.
- **I8 f invariant in gauge:** f in the harvest gauge is generation-independent.

Use `BAND = 0.4` as the band half-width constant.

## 0.7 The honest caveat (do not oversell)

The warm start CANNOT manufacture the generation gain. By fact 3 the prolonged
solution is J-identical to the parent; the increment is entirely the next LP
choosing to bend the new zero-curvature band strands. What the adaptive scheme buys
is **efficiency**: every node the LP spends lands in the band × melt-window where
harvest actually is, instead of ~half wasted on dead arms + dead time. Same J and
same J-ceiling as the uniform mesh at equal *resolution* — but far fewer nodes per
generation, so deeper reachable before the wall. The scientific deliverable is:
does the per-octave increment stay ~constant (unbounded) or decay (bounded), read
now at generations the uniform mesh can't afford.

## 0.8 Ground rules

- Deps numpy/scipy/matplotlib/highspy assumed installed; plain imports, no build.
- Every new grid must pass `check_feasible` on its warm-started `(f,g)` before any
  LP runs — a warm start that starts infeasible is a bug (fact 3 guarantees it
  shouldn't). Assert it.
- Compare against `dyadic_refinement` at matched resolution; report node counts.
- Keep the uniform baseline untouched. Add, don't edit.

# Run 12 implementation details — the melt-band cell construction

## Status: PLAN (implementation detail layer). Expands Section 6 of
`plans/run12-renormalization-cell.md` (the top-level plan — read that first;
this file does not re-derive the motivation, the structural argument for
anisotropic scaling, or the fixed-point dichotomy, only how to build it).
Motivated by `plans/BLOWUP_HYPOTHESIS.md` and `plans/MELTING_KINKS.md`.
Predecessor code: `kink_opt/construct.py` (Run 11). No code has been written
for this plan; two small numerical experiments were run against the existing
Run 11 code to ground the runtime-cost estimates below (Section on runtime),
nothing else.

Everywhere this plan deviates from the top-level plan's Section 6 sketch, the
deviation is marked **DEVIATION** with its reason, per the task instructions.

## 0. Module layout decision

**New file: `kink_opt/melt.py`**, not an extension of `construct.py`.

Reasons:
- `construct.py`'s own docstring and `plans/run11-constructive-hierarchy.md`
  both say Run 11 is IMPLEMENTED and RUN, with numbers already cited in
  `STRATEGY.md` and `CLAUDE.md`. Editing it in place to carry a second,
  structurally different ansatz (band vs. point, non-shared-endpoint windows
  vs. shared-endpoint) risks silently perturbing a frozen, cited result via
  shared helper edits.
- The package layout convention (`kink_opt/__init__.py`'s docstring, and
  CLAUDE.md's file list) already keys modules to run boundaries: `topology`
  = Task B/D + Runs 9-10, `construct` = Run 11. `melt` = Run 12 continues
  that pattern directly.
- `construct.py`'s module-level constants (`XI_OFFSET`, `ETA_OFFSET`,
  `_travel_path`) encode a *single*-kink-per-generation carrier. A band is
  `K` kinks per family per generation sharing one center path — forcing that
  through `build_hierarchy`'s parameter list would require either a `K=1`
  special case threaded through every function or a parallel set of
  band-shaped functions living alongside the point-shaped ones in the same
  file. Cleaner as a separate module that imports what it needs.
- Nothing about the shared low-level machinery needs to move: `melt.py`
  imports from `geometry.py`, `lp.py`, `solver.py`, `verify.py`,
  `topology.py` exactly as `construct.py` does — no duplication is forced by
  the split, and `construct.py`'s `fit_geometric` is reused as-is (imported,
  not copied) since it is a generic log-space geometric fit with no
  Run-11-specific assumption baked in.

`melt.py` imports: `from .geometry import MARGIN, conv_eval`;
`from .lp import lp_weights_f, lp_weights_g`; `from .objective import total_J`;
`from .solver import _alternate`; `from .verify import certify, graded_grid,
_ub, refine_time`; `from .topology import _insert_column`;
`from .construct import fit_geometric, grid_convergence_check` (both are
already fully generic over the `sol` dict schema — no reason to re-implement
either).

## 1. The biggest design decision: what `r_k` means, and why it is not a
`build_band` input

The top-level plan's Section 6 sketch writes
`build_band(gen, w_k, s_k, L_k, r_k, K)`, listing `r_k` (rise share) as a
construction input alongside the geometric parameters.

**DEVIATION:** this plan does NOT pass `r_k` into `build_band`. Reason: the
repo's hard constraint is that weights are *never guessed* — they are always
the exact LP optimum for the frozen positions and lifetime masks currently in
force (`certify()`'s repair step, `_alternate(..., optimize_pos=False)`'s
fixed point). `r_k` (rise share deposited by generation `k`) is a property of
the *solved weights*, not of the geometry — there is no LP-compatible way to
hand it a target value without either (a) hand-setting weights (banned) or
(b) adding a new hard linear constraint capping generation `k`'s total
curvature mass, which is implementable (see below) but is a materially
different, stronger claim than "spread K hats across a band" and should be
opt-in, not the default.

Concretely: `r_k` is treated as a **measured output** of
`read_environment`/`saturation_diagnostics`-style instrumentation (Section 6
below), reported alongside `dJ_k`, exactly like Run 11's `g_mass` is a
measured output, not an input. `build_band`'s parameters are purely
geometric/topological: center path, width, window, hat count, drift shape.

If Stage 4 (the fixed-point sweep) finds that geometry-only control isn't
expressive enough to explore the `(λ_w, λ_s, λ_r, L̂)` space the top-level
plan wants (i.e. `r_k` refuses to track anything you'd call a free
parameter), the fallback is: give the generation's own columns a **finite**
per-node weight upper bound (not `0`/`inf` — `_wbounds`/`lp_weights_f`/
`lp_weights_g` already accept arbitrary finite `ub` values per node, this is
not new code in `lp.py`, just a different `ub` array than the boolean-mask
`_ub()` helper produces) sized to target a particular curvature-mass ceiling.
This still solves an exact LP (global optimum subject to a tighter box), so
it does not violate "weights never guessed" — it is flagged as Option B in
the open questions at the end, not built by default.

**Consequence for the fixed-point sweep:** the swept ansatz parameters in
this plan are `(λ_w, λ_s, L̂)`, three not four — `λ_r` is read off, not swept.
`K` (hats per band) is held at one fixed global constant across all
generations (Section 3 of the top-level plan calls for curvature mass
`m_k ~ O(1)`, i.e. roughly constant per generation, which a constant `K`
delivers structurally without needing to be dialed).

## 2. Data structures

### 2.1 `BandSpec` (one generation's band, before insertion)

A plain `dict` (not a class — matches the rest of the codebase's convention
of plain dicts for `sol`, no classes anywhere in `kink_opt/`):

```
BandSpec = dict(
    k       = int,                  # generation index, 0 = boundary stock + coarse carrier
    c       = <callable t -> float>,  # band-center path in absolute x, co-moving origin
    w       = float,                # band half-width in absolute x (hat offsets scaled by this)
    window  = (t_birth, t_death),   # float, float -- absolute t
    K       = int,                  # hats per family in this band (global constant across k>=1)
    x_hat   = np.ndarray, shape (K,),  # fixed offsets in [-1,1], shared by f and g (Section 4)
    L       = float,                # realized drift length: max(c(t))-min(c(t)) over window
    drift   = "linear" | "arc",
)
```

`k=0` is special-cased (Section 3): it is not a `BandSpec` at all, it is two
static full-lifetime single hats (the boundary stock) plus one traveling
single-hat "coarse carrier" pair, built exactly as Run 11's gen-0
(`construct._travel_path`, `XI_OFFSET`/`ETA_OFFSET`) — reused via import, not
reimplemented (**DEVIATION note**: this makes `melt.py` depend on
`construct.py` for gen-0 seeding; this is intentional reuse, not a layering
violation, since `construct.py` exports it as ordinary functions).

### 2.2 Mapping onto the solution dict (`XI`/`ETA`/`alive_f`/`alive_g`)

Unchanged schema from `run()`/`build_hierarchy()`: `XI` is `(Np1, Kf_total)`,
`ETA` is `(Np1, Kg_total)`, `alive_f`/`alive_g` boolean masks of the same
shape. A `BandSpec` with `K` hats contributes `K` columns to `XI` and `K`
columns to `ETA` (not 1, as in Run 11) — `Kf_total = 1 (boundary) + 1 (coarse
carrier) + K*n_gen`, same for `Kg_total`. Column identity is preserved by
appending in a fixed order (boundary, carrier, band 1's K columns, band 2's K
columns, ...) — same no-crossing assumption `topology.py` already documents
and relies on (`generation_step`'s "Column identity" paragraph).

Each column `i` of band `k`'s block is the trajectory

    xi_i(t) = clip(c_k(t) + w_k * x_hat_i,  -1+MARGIN, 1-MARGIN)     for ALL t

(computed at every grid node, not just inside the window — matching
`add_kink`/`_insert_column`'s existing convention that a dead node's position
is a "free, harmless extra checkpoint" outside its lifetime), with liveness
`alive_*[:, col] = (t >= t_birth - 1e-12) & (t <= t_death + 1e-12)`.

## 3. Function signatures

All in `kink_opt/melt.py` unless noted. Mirrors `construct.py`'s naming
(`build_hierarchy` -> `build_melt_hierarchy`, `constructive_ladder` ->
`melt_ladder`, `sweep_ratios` -> `melt_sweep`).

```python
def build_band(t, c, w, window, K=8, x_hat=None, drift="linear",
               dead=False):
    """One generation's band ansatz: K hat offsets `x_hat` (shared by f and
    g) spread across a band of half-width `w`, centered on path `c(t)`,
    alive only on `window`. Returns (col_f, col_g, win_mask), each
    (Np1, K)/(Np1,) -- NOT yet inserted into any XI/ETA (the caller does
    that via `topology._insert_column`, once per family, matching Run 11's
    convention).

    `c` is a callable t (array) -> x (array), already resolved to absolute
    coordinates (the caller is responsible for anchoring it -- see
    `_drift_path`). `x_hat` defaults to `np.linspace(-1, 1, K)`, i.e. hats
    evenly spread across the band's own [-w, w] extent in the co-moving
    frame; f and g use the SAME x_hat grid by default (Section 4:
    the LP, not the ansatz, is what should decide whether f's realized
    curvature ends up narrower/taller than g's -- this is the
    Stage-3 "K too small to resolve independent f/g shapes" enrichment,
    not the Stage-1 default).

    `dead=True` forces `win_mask` all-False regardless of `window` -- used
    by `check_band_neutral` (Section 6) exactly as `build_hierarchy`'s
    `dead_gens` forces a generation's columns dead for the insertion-neutral
    gate.
    """


def _drift_path(t, c_anchor, L, window, arc="linear"):
    """Band-center path for ONE generation, in absolute x: a path of length
    `L` (order O(1), the SAME absolute number at every generation -- Section
    3's "L_k ~ O(1): does NOT shrink" is implemented literally as one global
    constant `L`, not a per-k schedule) over `window`, anchored so the path
    passes through `c_anchor` at the window's temporal midpoint.

    arc="linear": c(t) = c_anchor + L * (that/s - 0.5), that = t - t_birth,
      s = window width -- straight sweep through c_anchor.
    arc="arc": c(t) = c_anchor + L * (4*u*(1-u) - 0.5), u = that/s --
      raised-parabola there-and-back, peaking at u=0.5 (a coarse proxy for
      the mesh's observed there-and-back band drift; NOT the default -- see
      open questions).
    Outside `window`, holds at the nearer endpoint value (the position array
    must still be defined at every t; only `win_mask` controls liveness).
    """


def build_melt_hierarchy(n_gen, lambda_w=0.5, lambda_s=0.5, L=0.3, K=8,
                          x_hat=None, drift="linear", t0=0.0, t1=1.0,
                          w0=0.5, s0=0.5, coarse_N=8, base_fine_sub=4,
                          outer=20, dead_gens=None):
    """Build generations 0..n_gen as a single solution dict (schema: A, XI,
    B, ETA, t, alive_f, alive_g, J, hist -- identical to build_hierarchy's).

    Generation 0: boundary stock (one static full-lifetime f-hat at x=0, one
    static full-lifetime g-hat at x=0 -- NOT pinned by an equality
    constraint; the LP already drives these to the full-depth centered hat
    on its own, exactly as Run 1/3's static seeds show) PLUS the coarse
    carrier (one traveling f/g kink pair, `construct._travel_path`, full
    lifetime) -- 4 columns total (2 boundary + 1 f-carrier + 1 g-carrier;
    note: 2 boundary hats, not 2 per family -- one f-hat, one g-hat).

    Generation k (1 <= k <= n_gen): a BandSpec with
      w_k = w0 * lambda_w**(k-1)
      s_k = s0 * lambda_s**(k-1)          (window width, absolute time)
      window_k anchored at the MIDPOINT of generation (k-1)'s own window
        (generation 0's "window" for this purpose is (t0, t1))
      c_k = _drift_path(t, c_anchor=coarse_carrier's position at that
        midpoint, L=L, window=window_k, arc=drift)
    inserted via `build_band` + `topology._insert_column`, K columns per
    family, matching `build_hierarchy`'s "insert at zero weight, J-neutral"
    invariant (Section 5).

    Grid: `graded_grid` over the n_gen windows, `fine_sub_k = base_fine_sub *
    s_1 / s_k` -- the exact `generation_ladder`/`scale_sweep`/
    `build_hierarchy` convention (topology.py's `window0/w_k` /
    construct.py's `ws[0]/w`), restated here as `s_1/s_k` since `s_1` (not
    s_0, which doesn't exist as a window) is this construction's widest
    per-generation window.

    Weights: `_alternate(..., optimize_pos=False)` only -- monotone convex
    fixed point, deterministic, no position NLP, identical discipline to
    `build_hierarchy`.

    `dead_gens` (set of generation indices in 1..n_gen): force those bands'
    columns all-dead before the weight solve -- same purpose as
    `build_hierarchy`'s `dead_gens`, used by `check_band_neutral`.
    """


def melt_ladder(n_gen, lambda_w=0.5, lambda_s=0.5, L=0.3, K=8, sub=8,
                 **kwargs):
    """Run 12 driver, direct analogue of `constructive_ladder`: for
    k=0..n_gen, `build_melt_hierarchy(k, ...)` then `certify()`. Records
    dJk = Jc_k - Jc_{k-1} and ratio = dJk/dJ_{k-1}.

    Returns dict(generations=[dict(k, Jc, constraints_ok, dJk, ratio,
    env, sol), ...], lambda_w, lambda_s, L, K) -- `env` is
    `read_environment`'s output for that generation (None for k=0), attached
    here (not computed separately) so the ladder is the one place both `dJk`
    and the environment profile are read off the SAME certified sol.
    """


def melt_sweep(n_gen, lambda_ws, lambda_ss, L=0.3, K=8, sub=8, **kwargs):
    """Sweep `melt_ladder` over a (lambda_w, lambda_s) grid -- direct
    analogue of `sweep_ratios`. Returns list of dict(lambda_w, lambda_s,
    dJk=[...], ratios=[...], ladder=<melt_ladder result>)."""


def read_environment(sol, k, band_spec, n_sample=17, family="f"):
    """Read the slope-slack profile beta(x_hat) and remaining-rise profile
    rho(x_hat) that generation k's OWN band would see, sampled at `n_sample`
    points spanning band_spec['x_hat'] (i.e. in k's own co-moving frame),
    evaluated at t_read = temporal midpoint of band_spec['window'] (nearest
    existing grid node -- no time-interpolation, see Section 7), from the
    ALREADY-SOLVED (A, XI) or (B, ETA) of `sol` (a hierarchy certified
    through some generation <= k; typically generation k-1's solve for the
    INCOMING environment, or generation k's own solve for the OUTGOING one --
    the caller decides which `sol` to pass, this function only reads).

    beta(x) = 1 - |slope of f(., t_read) at x|   (exact piecewise-constant
      formula, Section 7 -- no finite differencing)
    rho(x)  = (1 - |x|) + f(x, t_read)            (remaining Lipschitz rise
      budget at x, same formula `saturation_diagnostics` already uses for
      rise_budget_used = -f(x_birth, 0), generalized to any t)

    `family` selects which function's profile to read (f for beta/rho as
    defined; g's dual profile, if ever needed, is a symmetric call with
    family="g" and the sign/role of monotonicity flipped -- not built by
    default, Stage 2 only needs f's).

    Returns dict(x=<absolute x sample points>, x_hat=<co-moving samples>,
    t_read=float, beta=np.ndarray, rho=np.ndarray).
    """


def env_distance(env_a, env_b):
    """Distance between two `read_environment` outputs sampled at the SAME
    x_hat grid (raises if grids differ) -- max-norm of the concatenated
    (beta, rho) differences. This is the map E -> E' self-reproduction
    metric (Section 4.4 of the top-level plan): env_distance(E'_k, E_{k+1})
    small means generation k+1's incoming environment matches generation
    k's outgoing one, i.e. near a fixed point."""


def check_band_neutral(n_gen, lambda_w=0.5, lambda_s=0.5, L=0.3, K=8,
                         sub=8, tol_rel=0.01, **kwargs):
    """Validation gate #1, direct analogue of `check_insertion_neutral`:
    build generation n_gen with ITS OWN band forced dead (`dead_gens=
    {n_gen}`), compare Jc against `certify(build_melt_hierarchy(n_gen-1))`.
    Returns dict(ok, Jc_with_dead, Jc_without, diff, diff_rel)."""


def band_travel_sanity(sol, k, band_specs):
    """Validation gate #3, GENERALIZED from `travel_sanity`: confirm
    generation k's band actually drifts ~L (not co-located with its own
    center at window start) -- checks the REALIZED spread of the band's
    OWN columns' positions over its window (max-min over the live window,
    across all K columns of one family), not a single kink's start/end like
    `travel_sanity` (a band's "does it travel" signal is column-spread
    magnitude, not one trajectory's endpoints, since K hats might already be
    spread across width w_k even with zero center drift -- the sanity check
    needs to distinguish WIDTH from DRIFT: compares the observed center
    displacement (mean or median position at window start vs window end)
    against `L`, not against `w_k`).
    Returns dict(ok, k, L_expected, L_observed)."""


def mesh_cross_check(dJk_sequence, target_per_octave=0.215, tol_factor=3.0):
    """Validation gate #4 (NEW relative to Run 11 -- top-level plan Section
    6): at n_gen=2-3, dJk should land within `tol_factor`x of the mesh's
    measured +0.215/octave (order-of-magnitude, not a tight match -- Run
    11's own toy construction (0.31 total J after 4 generations) is the
    cautionary example of an ansatz passing gates 1-3 while being off by
    ~1 order of magnitude from the mesh's structure). Returns dict(ok,
    dJk_sequence, target, ratio_to_target=[dJk_i / target for each i])."""
```

`grid_convergence_check` (generation-agnostic, generic over any `sol`) is
imported from `construct.py` unchanged, not reimplemented — it only needs
`sol["A"], sol["XI"], ...` and `certify`, neither of which differ for a band
hierarchy.

### Fixed-point sweep driver

```python
def fixed_point_sweep(n_gen, lambda_ws, lambda_ss, Ls, K=8, sub=8,
                        env_tol=0.05, **kwargs):
    """Stage 4 driver. For every (lambda_w, lambda_s, L) combination, run
    `melt_ladder`, then compute env_distance(E'_k, E_{k+1,in}) for every
    adjacent generation pair k=1..n_gen-1. A combination is "near a fixed
    point" if env_distance stays below `env_tol` (and roughly flat, not
    still trending down) for the last 2-3 generations.

    At a near-fixed-point combination, read gamma = dJk / dJ_{k-1} averaged
    over the last few generations (Section 4.4 of the top-level plan: gamma
    close to 1 => every generation gains >= c > 0 => log-growth; gamma < 1,
    stable => geometric decay => bounded, with 1-gamma the per-generation
    "re-arm tax").

    Returns list of dict(lambda_w, lambda_s, L, env_distances=[...],
    gamma_est, gamma_stable, ladder=<melt_ladder result>)."""
```

## 4. Kink trajectory assembly (the exact insertion path)

Per generation `k>=1`:

1. `col_f, col_g, win = build_band(t, c_k, w_k, window_k, K=K, x_hat=x_hat,
   drift=drift, dead=(k in dead_gens))`.
2. For `i in range(K)`: `XI, ETA, alive_f, alive_g = _insert_column("f", XI,
   ETA, alive_f, alive_g, col_f[:, i], win)`, then the analogous call for
   `"g"` with `col_g[:, i]`. (`_insert_column` inserts one column at a time;
   `build_band` returning a `(Np1, K)` block means the caller loops `K`
   times per family — no new insertion primitive needed, `topology.py`'s
   existing one-column-at-a-time API is reused unchanged.)
3. Every inserted column carries **zero weight** at insertion (same
   mechanism as `add_kink`/`build_hierarchy`: a freshly-appended column has
   no prior LP solution to inherit, and the immediately-following LP solve
   is fresh, not warm-started from a nonzero guess for the new columns) —
   so `total_J` is unchanged at insertion, and `check_band_neutral` verifies
   this holds even after the weight LP is allowed to run (i.e. that the LP
   doesn't spontaneously discover it should move the OLD generations
   somewhere else purely because of the new dead columns' presence — same
   invariant `check_insertion_neutral` checks).

This exactly matches the requirement `xi_i(t) = c_k(t) + w_k * x_hat_i`,
alive only on the window, inserted through the existing zero-weight
insertion invariant.

## 5. Grid construction

Reuses `graded_grid(windows, coarse_N, fine_sub, t0, t1)` unmodified.
`windows = [window_1, ..., window_n_gen]` (generation 0 has no window — full
lifetime, already covered by the coarse background). `fine_sub` is a list,
`fine_sub_k = base_fine_sub * s_1 / s_k` — this is the exact convention cited
in `topology.generation_ladder` (`fine_subs = [base_fine_sub * window0 / w
for w in ws]`) and `construct.build_hierarchy` (`fine_subs = [base_fine_sub *
ws[0] / w for w in ws]`), restated with this plan's own variable names (`s_1`
= this construction's widest per-generation window, since there is no
"window 0" the way `build_hierarchy` has no shared endpoint either — Run 12's
windows are NOT shared-endpoint, see Section 6 below, but the `fine_sub`
formula does not depend on shared-endpointness, only on each window's own
width relative to the widest one, so it ports unchanged).

**DEVIATION note on non-shared-endpoint windows:** `graded_grid` was written
assuming (and both existing callers pass) `windows` anchored at the same
`t1` — a list of nested, overlapping intervals with a common right edge. This
plan's `windows` are NOT shared-endpoint (each is anchored at its own
parent's window midpoint, Section 3) — but nothing in `graded_grid`'s
implementation (`kink_opt/verify.py`, read start-to-finish above) actually
depends on shared endpoints; it just unions each window's own local
`linspace` with the coarse background and de-duplicates. This should be a
correctness non-issue, but is flagged because it is untested territory for
that function (every existing call site happens to pass shared-endpoint
windows) — **Stage 1/2's grid_convergence_check gate is the guardrail**: if
non-shared-endpoint windows interact with `graded_grid` in some
unanticipated way (e.g. two windows landing so close together that the
coarse-background merge-tolerance in `graded_grid`'s last few lines
collapses distinct fine regions), it will show up as a grid-convergence
failure, not a silent wrong number.

## 6. Validation gates

All four gates must PASS before any `dJk`/`gamma` number is read, matching
the repo's "any measured number must pass grid-convergence before being
trusted" rule.

| gate | Run 11 origin | Run 12 adaptation |
|---|---|---|
| insertion-neutral | `check_insertion_neutral` (dead_gens forces the NEW generation's single column pair dead) | `check_band_neutral` — forces the new generation's ENTIRE K-column band dead; same tol_rel=0.01 comparison logic, reused pattern not reused code (different column count to zero) |
| grid-convergence | `grid_convergence_check(sol, sub_lo=8, sub_hi=16)` | **imported unchanged** — fully generic over `sol` |
| travel-sanity | `travel_sanity` (single kink's XI[0,0] vs XI[-1,0]) | `band_travel_sanity` — checks the band's REALIZED center displacement over its own window against `L`, not a single trajectory's global start/end (a band could have zero center drift but nonzero column SPREAD, or vice versa — these are different failure modes and must not be conflated) |
| mesh cross-check | none (new in the top-level plan) | `mesh_cross_check` — order-of-magnitude (within a factor of ~3, not 1%) comparison of `dJk` (k=2,3) against the mesh's measured +0.215/octave; this is a coarser bar than the other three because the mesh's own number carries the k07/k08 budget-artifact caveat (`BLOWUP_HYPOTHESIS.md`) — treat this gate as "not obviously the wrong order of magnitude," not as ground truth |

## 7. Environment read-off mechanics

**Where in time:** `t_read` = the grid node nearest the temporal midpoint of
the generation's own window — an actual existing node on the graded grid
(searched via `np.searchsorted`/nearest-index, not interpolated: Task C's
"dt cancels in the harvest sum" property is about the time-INTEGRATED
objective, not about a single-slice read, so interpolating a slice would be
an avoidable approximation when an exact grid node is available nearby by
construction, since `graded_grid` densifies exactly inside each window).

**Where in x:** `n_sample` points spanning the generation's own `x_hat` range
(default matches `x_hat`'s own K points, extended to `n_sample=17` for a
smoother profile plot if `K` is small — sampling is free, it's just
`conv_eval` calls, not a re-solve).

**Slope slack `beta(x)` — exact, no finite differencing.** At a fixed time
slice, `f(x,t) = -sum_i a_i hat(x; xi_i)` is piecewise-linear with breakpoints
exactly at the (sorted) kink positions `xi_i`. On the interval between the
j-th and (j+1)-th sorted kink, the slope is the closed form already implicit
in `lp_weights_f`'s own Lipschitz rows (`geometry.py`'s hat slopes,
`1/(1+xi)` on the left branch, `-1/(1-xi)` on the right branch):

    slope(x in (xi_j, xi_{j+1})) = sum_{i<=j} a_i/(1-xi_i) - sum_{i>j} a_i/(1+xi_i)

computed directly from the sorted `(a_i, xi_i)` at `t_read` — no dense-grid
finite differencing needed (this is exactly the boundary-slope formula
`lp_weights_f` already evaluates at `k=0..N`, just evaluated at interior
breakpoints instead of at `x=+-1`). `beta(x) = 1 - |slope(x)|`: `0` on a
Lipschitz-saturated arm (matches MELTING_KINKS' measured `+-0.957` to
`+-1.000` arm slopes), positive only near a basin bottom.

**Remaining rise `rho(x)` — from `conv_eval`, same pattern as
`saturation_diagnostics`.** `rho(x) = (1 - |x|) + f(x, t_read)`
(`conv_eval(x, A[t_read_idx], XI[t_read_idx])`, the identical call
`saturation_diagnostics` already makes for `rise_budget_used`), i.e. the
per-x Lipschitz rise budget `1-|x|` minus what has already been spent
(`-f(x,t_read)`).

Both are O(n_sample) `conv_eval`/closed-form evaluations against an
already-solved slice — no new LP solve, no dense `verify_dense`-style
`nx=1601` grid.

## 8. Fixed-point sweep: parameters, ranges, cost estimate

**Swept:** `(lambda_w, lambda_s, L)` — three, not four (Section 1's
deviation: `lambda_r` is measured, not swept). `K` held fixed (a resolution
choice, checked by a K-saturation side-sweep per the top-level plan's Risk
list, not part of the fixed-point search itself).

**Ranges:** `lambda_w, lambda_s in {0.3, 0.5, 0.7}` (3x3 = 9 combinations,
matching Run 11's own sweep grid size in `sweep_ratios`, a precedent for
"affordable and already interpretable at this granularity"). `L in {0.15,
0.3, 0.5}` (absolute drift length; upper end constrained by `MARGIN` and by
not wanting the band to sweep past the coarse carrier's own extent within one
window — no hard rule here, these are starting values to be widened if
`env_distance` never gets close at any of them). Total: up to `9 * 3 = 27`
`(lambda_w, lambda_s, L)` combinations, each running a full `melt_ladder`.

**Cost per point, estimated from measurement against Run 11's actual code**
(this repo, run live against `kink_opt.construct`, not assumed): a Run
11-style ladder with `n_gen` point-generations (K=1 per generation, so
`Kf_total = n_gen+1`) measured:

| n_gen | nodes | build time | certify(sub=8) time |
|---|---|---|---|
| 4 | 45 | 0.07s | 0.56s |
| 6 | 61 | 0.11s | 1.51s |
| 8 | 77 | 0.21s | 3.26s |

`certify`'s cost (dominated by `verify_dense`'s `nx=1601` dense evaluation
and the fine-grid weight-LP repair) grows worse than linearly in node count
+ kink count here (roughly doubling every +2 generations at K=1/gen). Run
12's bands carry `K=8` (not 1) live columns per generation, i.e. `Kf_total =
2 + 8*n_gen` instead of `n_gen+1` — an ~8x larger column count at the same
`n_gen`, and the LP block-matrix construction in `lp.py` scales with
`Kf^2`-ish terms (the monotonicity block is `(2*Kf, Np1*Kf)`). **Estimate:
a single `n_gen=4, K=8` ladder likely costs low tens of seconds, not
sub-second** — this is an extrapolation, not a fresh measurement (measuring
it directly requires `build_melt_hierarchy` to exist first; flagged
explicitly as an assumption, not a citation). A `27`-point fixed-point sweep
at `n_gen=4` is therefore a "leave it running, don't iterate on it
interactively" job (tens of minutes), not a "run and read in the same
sitting" job like Run 11's sweep was. Recommendation: get Stages 1-3 working
and timed for real before committing to the full 27-point Stage 4 grid;
narrow the ranges above using Stage 3's single-point timing first.

**Outer iterations:** `_alternate(..., optimize_pos=False)` is a monotone
convex fixed point (no NLP), so it should need very few outer iterations to
converge — reuse `build_hierarchy`'s default `outer=20` unchanged; there is
no reason to expect this needs raising the way the nonconvex position-NLP
budgets in Runs 9-10 did (that entire failure class does not apply here —
Section titled "risks" below still checks it empirically once, cheaply,
rather than assuming it).

**"Self-reproducing environment" metric:** `env_distance(E'_k, E_{k+1,in})`
(Section 3's `env_distance` function) — max-norm over the sampled
`(beta, rho)` grid. A combination is read as "near a fixed point" if this
distance is small AND not still visibly shrinking generation-over-generation
(a distance that is merely still converging hasn't found the fixed point
yet, it's still approaching it — needs at least 3-4 generations to tell
those apart, hence `n_gen>=4` for Stage 4).

## 9. Staged milestones

**Stage 1 — single band on the gen-0 carrier.**
Build: gen 0 (boundary stock + coarse carrier) + ONE band (generation 1),
`build_melt_hierarchy(1, ...)`. Runnable checkpoint: `certify()` succeeds,
`check_band_neutral(1, ...)` passes, `grid_convergence_check` passes,
`band_travel_sanity(sol, 1, ...)` passes. Produces: one number (`Jc_1` vs
`Jc_0`, i.e. `dJ_1`) and one heatmap PNG (via existing `persist.save_run` +
`viz.plot_heatmaps`, no new plotting code needed — the fields are `f(x,t)`/
`g(x,t)` just like every other `sol`). **Stop and revisit the design if:**
`dJ_1 <= 0` (the band contributes nothing — check `K`, `w0`, whether the
band is actually landing where the coarse carrier's basin bottom is, i.e.
check the anchor point `c_anchor` computation before concluding anything
about the hypothesis) or if `band_travel_sanity` fails (band isn't actually
drifting — likely an anchoring/window bug, not a hypothesis result).

**Stage 2 — two generations, environment read-off.**
Build: `build_melt_hierarchy(2, ...)`. Add `read_environment` calls for
generation 1 (incoming, from the `n_gen=0` solve) and generation 1 outgoing
(from the `n_gen=1` solve) and generation 2 incoming (from the `n_gen=1`
solve again — same profile, sampled at generation 2's own, finer `x_hat`
grid). Runnable checkpoint: plot `beta(x_hat)`/`rho(x_hat)` before/after
generation 1's insertion (a simple 1D line plot — matplotlib, no new
infra); `env_distance` between generation-1-outgoing and generation-2-incoming
is a single printed number. **Stop and revisit if:** `beta`/`rho` come back
NaN or flat-zero everywhere (t_read likely landed on a dead node, or the
`x_hat` sample points fell outside the region where any kink of the k-1
generation is actually live — check `t_read`'s nearest-node search and the
window arithmetic before touching the read-off formulas themselves, since
the formulas are closed-form and unlikely to be the bug).

**Stage 3 — n_gen ladder.**
`melt_ladder(n_gen=4, ...)` (or higher if Stage 1-2 timing allows) at ONE
`(lambda_w, lambda_s, L)` point. Runnable checkpoint: the `dJk` table
(mirroring Run 11's demos.py printout: k, Jc, dJk, ratio, constraints_ok),
plus `mesh_cross_check` on `dJk[2], dJk[3]`. This is the first point where
the actual runtime cost (Section 8) is measured for real rather than
estimated — record it and use it to size Stage 4's grid before running all
27 points. **Stop and revisit if:** `mesh_cross_check` fails by more than an
order of magnitude in EITHER direction (too small: the band ansatz isn't
capturing the mesh's sweep mechanism at all, revisit `K`/`w0`/`drift` shape;
too large: suspect a grid-convergence gate that's passing for the wrong
reason, e.g. an under-resolved `verify_dense` — re-run `grid_convergence_check`
at a even wider `sub_hi` before trusting an anomalously large number).

**Stage 4 — fixed-point sweep.**
`fixed_point_sweep` over the ranges in Section 8. Runnable checkpoint: a
table of `(lambda_w, lambda_s, L, env_distance trend, gamma_est)`, and (the
single most important output) a verdict on whether any combination shows
`env_distance` stabilizing near a nonzero-`gamma`-consistent fixed point vs.
every combination showing `env_distance` still shrinking generation-over-
generation with `gamma` trending toward 0 (bounded J, re-arm cost mechanism
identified, same shape of conclusion Run 11 reached but for a genuinely
different, sweep-capable ansatz this time). **Stop and revisit the whole
plan if:** no combination in the swept ranges gets `env_distance` within an
order of magnitude of stabilizing — that means either the ranges are wrong
(widen them) or (the top-level plan's own honesty condition, Section 7's
"No fixed point with γ=1") boundedness is simply the answer for this ansatz
family too, which is a legitimate, reportable outcome, not a failure — the
plan says explicitly not to keep tuning `(λ_w, λ_s, λ_r)` chasing `γ=1` once
degradation looks structural.

## 10. `demos.py` and `persist.py` additions

**`demos.py`:** append a "Run 12" block after Run 11, following the existing
narrated pattern (a `print("=" * 70)` header explaining what Run 11 left
open — the ansatz shape, not the optimizer route, was the ceiling — and why
this run's ansatz is different). Structure to mirror Run 11's block:
1. Validation gates first (`check_band_neutral`, `grid_convergence_check`,
   `band_travel_sanity`, `mesh_cross_check`), printed PASS/FAIL exactly like
   Run 11's three-gate preamble.
2. The primary `melt_ladder` table (k, Jc, dJk, ratio, constraints_ok) at
   one `(lambda_w, lambda_s, L)` point, printed the same columnar way as
   Run 11's primary-ladder table.
3. `read_environment` before/after printout for at least one generation
   transition (the `beta`/`rho` numbers, not just a plot reference).
4. The `melt_sweep`/`fixed_point_sweep` table, IF Stage 4 has been run by
   the time this is written (gate this behind whether Stage 4's runtime is
   acceptable in the full `python3 -m kink_opt` run — if the fixed-point
   sweep takes tens of minutes per Section 8's estimate, it should NOT run
   inside the default `python3 -m kink_opt` invocation; put it behind a
   flag or a separate script the way `--profile` is already a separate mode
   of `kink_opt/__main__.py`, rather than making every future `python3 -m
   kink_opt` run take that long). This is a new consideration Run 11 didn't
   have (its whole sweep was sub-2-seconds).

**`persist.py`:** new `save_melt`/`load_melt`, mirroring `save_construct`/
`load_construct` exactly (same `gen{k}/sol.npz` + `fields.png` layout, a
`melt.json` instead of `construct.json` carrying `lambda_w, lambda_s, L, K`
instead of `scale_t, scale_x`, plus the `env` dict — `beta`/`rho` arrays —
per generation, JSON-encodable via the existing `_jsonable` helper since they
are plain numpy arrays already handled by that function elementwise). If
Stage 4 is reached, a `save_fixed_point_sweep`/`load_fixed_point_sweep` pair
analogous to `save_sweep`/`load_sweep`, keyed by sweep-point index the same
way `w{i}/` is keyed in `save_sweep`.

## 11. Where the three known trap patterns (Runs 9, 10, mesh k07/k08) could
recur here, and the guard at each spot

- **Budget-sensitivity mimicking the signal (Run 9's pattern).** Does not
  apply to the LP-only weight solve (`_alternate(outer=20)` is a monotone
  convex fixed point, not a starved nonconvex search) — but DOES apply to
  `graded_grid`'s node density: an under-resolved window can make a real
  band's `dJk` look like it collapsed, exactly like the mesh's k07/k08
  starved-`t` artifact. Guard: `grid_convergence_check` (gate #2) at every
  generation depth that gets a number reported, not just the deepest one —
  same as Run 11's practice.
- **Resolution floor silently capping local node count (Run 10 fix #3).**
  The `fine_sub_k = base_fine_sub * s_1/s_k` formula inherits `graded_grid`'s
  own `max(2, ...)` floor on local sub-intervals per window — at deep `k`
  (`s_k` very small) this floor can bind well before the formula's intended
  density does. Guard: same as `scale_sweep`'s fix — check the realized
  local node count (`n_live_nodes`-style count restricted to one window)
  against the floor explicitly at the narrowest generation tested, not just
  trust the formula.
- **Wrong statistic / search noise (Run 9's contamination, Run 10 fixes #1-2).**
  Does not apply here at all — there is no seed, no multistart, no
  nonconvex search anywhere in this construction (same as Run 11). This is
  the one trap class this whole approach is structurally immune to; worth
  stating explicitly in the demos.py narration since it's the main selling
  point carried over from Run 11.
- **New trap specific to Run 12: the environment read-off itself, sampled on
  a discrete grid, hiding degradation below its own resolution** — this is
  explicitly called out in the top-level plan's Risks section ("Environment
  read-off is the new budget-sensitivity"). Guard: `read_environment`'s
  `n_sample` should itself be checked for convergence (double it, confirm
  `env_distance` barely moves) before trusting any `env_distance` number —
  this is a FOURTH grid-convergence-shaped check this plan adds beyond gate
  #2 (which only covers the *solve's* grid, not the *read-off's* sampling
  grid) — call it out explicitly in Stage 2's checkpoint rather than
  assuming gate #2 already covers it, since it doesn't.

## 12. Open implementation questions (short — everything else above is a
decision, not a question)

1. **Drift shape: `linear` vs `arc` as Stage-1 default.** This plan defaults
   to `linear` (simplest, matches `build_hierarchy`'s precedent) and treats
   `arc` (there-and-back) as a Stage-3 enrichment if `mesh_cross_check`
   fails under `linear`. The top-level plan's own Section 6 step 1 hints the
   mesh data favors `arc` ("try second" — implying try first, then this).
   Researcher call: commit to testing `linear` first regardless, or skip
   straight to `arc` given the mesh already suggests it?
2. **`r_k` enforcement (Section 1): measured-only (this plan's default) vs.
   the finite-`ub`-cap Option B.** Only matters once Stage 4 either (a)
   can't find any fixed point in the swept `(lambda_w, lambda_s, L)` ranges,
   or (b) finds one but wants a fourth free parameter to explore
   sensitivity. Deferred; flagged so it isn't forgotten.
3. **Window-nesting anchor point.** This plan anchors each generation's
   window at its parent's window MIDPOINT (a simplification — Section 3's
   "basin bottom crossing" concept more precisely means wherever the
   parent's OWN drift path has zero velocity, which for a `linear`-drift
   parent doesn't exist (constant velocity) and for an `arc`-drift parent is
   exactly its midpoint anyway). If `arc` becomes the default (question 1),
   this anchor choice is then exactly correct and this question dissolves;
   if `linear` stays the default at the point where nesting depth matters,
   the midpoint choice is an arbitrary stand-in that should be revisited.
4. **`K` (hats per band): global constant value.** This plan assumes `K=8`
   throughout (matching the top-level plan's own "order 8x16" cost
   estimate) but does not derive it — Section 7's Risk #5 ("K per band too
   small... a one-sided error") means a too-small `K` can only ever
   understate `dJk`, so `K=8` is a reasonable starting guess but the
   K-saturation check (does `dJk` stabilize as `K` grows?) the top-level
   plan calls for is not scheduled into any specific stage above. Suggest
   running it once at Stage 3's single ladder point before committing to
   Stage 4's larger sweep at a possibly-too-small `K`.

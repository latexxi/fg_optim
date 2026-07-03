# Code plan: Run 9 generation-gain ladder

Companion to `plans/run9-generation-gain-ladder.md` (strategy). This one is
implementation-level: exact functions, signatures, file locations, data flow.

## Status: IMPLEMENTED, verified, committed (`d91fc20`)

All 6 pieces below were shipped in `kink_opt.py` and Run 9 ran cleanly as
part of `python3 kink_opt.py` (full driver ~57s including Run 9). The file
was later split into the `kink_opt/` package (`b7eedf7`); line locations
below are updated to the post-split files, verified byte-identical output
across the whole demo suite before/after the split. Two real issues
surfaced during verification and are recorded here since they change how
much to trust the numbers, not just how the code is organized:

1. **Regrid resolution-loss bug (found and fixed).** The original plan had
   `generation_ladder` build the ladder's time grid directly from
   `graded_grid(...)`. But `graded_grid`'s own uniform background
   (`coarse_N`) can be *coarser* than the base solution's original grid
   outside the lifetime windows -- migrating onto it silently lost
   resolution and made the regrid measurably non-J-neutral (2.309 -> 2.265,
   a real defect caught by the "regrid must reproduce base J" guard, not
   discretization noise). Fixed by unioning the graded grid with the base
   grid's own `t` (`np.union1d`) so the migration is a strict superset,
   never a coarsening. See step 5 below.
2. **Budget was too low for feasibility (found and fixed).** The first full
   run had every generation land `feasible=False` by a tiny margin (~3e-8
   against the `1e-9` `verify_dense` floor). Diagnosed by rerunning the same
   candidate at higher `outer`/`pos_iters` and watching the violation
   collapse to float noise (~1e-15) while `Jc` also improved -- confirming
   under-convergence, not a real defect. The Run 9 driver call was bumped
   from `outer=10, pos_iters=30` to `outer=25, pos_iters=60` accordingly.

Two small deviations from the original plan text, both harmless:
- `_kink_diagnostics`'s output is used per-family as `dict(f=..., g=...)`
  rather than a plain list -- clearer at the call site, same content.
- `_seed_grown`'s signature takes `base` as an explicit first argument
  (`_seed_grown(base, XI2, ETA2, af2, ag2)`) rather than closing over a
  module-level `G0`, since the ladder calls it once per generation with a
  different `cur` each time (Run 8's original closure only ever needed one
  fixed `G0`).

## New code, in order (final line numbers, post kink_opt/ package split)

### 1. `graded_grid` — extend `fine_sub` to accept per-window density

**Location:** `kink_opt/verify.py:54`.

`fine_sub` may be a scalar (old behavior, bit-for-bit unchanged) or a list
the same length as `windows`, giving each window its own density. The
ladder passes `fine_sub_k = base_fine_sub * (window0 / w_k)` per window so
the local node count stays roughly flat across generations instead of
shrinking toward the "finest lifetime spans >= 8 fine steps" floor.
Verified: scalar path bit-identical to a matching-value list; mismatched
list length raises `ValueError`; varying the list produces genuinely
different per-window density; empty-windows path unaffected.

### 2. `_seed_grown` — module function (was a Run 8 closure)

**Location:** `kink_opt/topology.py:43`, near `_insert_column`.

`_seed_grown(base, XI2, ETA2, af2, ag2)`: pads `base["B"]` with a zero
column for any new g-kink, then runs the weight-LP pair
(`lp_weights_f` -> `lp_weights_g` -> `lp_weights_f`) to bootstrap a
feasible, J-unchanged starting point after a family grows by one column.
Run 8's `__main__` block was updated to call this instead of its old local
closure (mechanical dedupe, confirmed byte-identical Run 8 output).

### 3. `_kink_diagnostics` — post-optimization measurement helper

**Location:** `kink_opt/topology.py:70`, near `_lifetime_window`.

`_kink_diagnostics(r, family, col_idx, parent_idx, tol=1e-8)` returns
`dict(lifetime=(t_on, t_off), extent=(min_x, max_x), jump_mean,
offset_from_parent)`, computed over the kink's *effective* active window
(`|weight| > tol`), not its imposed mask -- this is the actual
self-similarity test (does the optimizer keep the full imposed window or
contract further on its own). If the LP starves the new kink to zero
weight everywhere, falls back to reporting over the imposed lifetime mask
instead of returning NaN, so "the optimizer used none of its allotted
window" is a visible, meaningful number rather than a crash or a blank.
Verified standalone against a hand-built toy solution for both the
active-weight case and the fully-pruned fallback case, and observed firing
correctly in a real run (a toy 2+2 base where the new kink got starved).

### 4. `generation_step` — one rung of the ladder

**Location:** `kink_opt/topology.py:166`.

```
generation_step(cur, window, seeds=range(3), dx=0.05, outer=12,
                pos_iters=40, patience=None, sub=8)
```
Inserts one new f-kink and one new g-kink, both alive only on
`window = (t_birth, t_death)`, as perturbed copies (`add_kink`) of each
family's current most-active column. For each seed: insert, `_seed_grown`,
`_alternate`, compute `_kink_diagnostics` on the *pre-prune* candidate
(tracking the new column by its append-time index, since pruning can shift
indices and the fully-pruned fallback needs the column still physically
present), then `prune` + `certify`. Keeps the best *feasible* candidate by
`Jc`, falling back to the best candidate overall (flagged
`feasible=False`) if none certify -- an optimizer failure must be visible,
not silently dropped. Passing `window = (t[0], t[-1])` (full lifetime)
turns the identical call into the guard arm.

Returns `dict(sol, Jc, feasible, diagnostics=dict(f=..., g=...),
spread=[(seed, Jc, feasible), ...])`.

### 5. `generation_ladder` — the driver

**Location:** `kink_opt/topology.py:230`, right after `generation_step`.

```
generation_ladder(base, n_gen=4, window0=0.5, window_ratio=0.5,
                   seeds=range(3), dx=0.05, base_fine_sub=4, coarse_N=8,
                   outer=12, pos_iters=40, sub=8, verbose=True)
```

1. `cur = prune(base, 1e-8)`; `base_Jc = certify(cur)["Jc"]`.
2. Precompute the full window schedule: `w_k = window0 * window_ratio**(k-1)`
   for `k in 1..n_gen`, `windows_k = (t1 - w_k, t1)` (all anchored at the
   shared right endpoint, so later windows nest inside earlier ones).
3. `grid = graded_grid(windows, coarse_N=coarse_N, fine_sub=[base_fine_sub *
   window0/w_k for each k], t0=t0, t1=t1)`, then `t_new =
   np.union1d(grid, cur["t"])` -- the fix from the Status section above:
   this guarantees the migration is a strict refinement, never a
   coarsening, of `base`'s own grid.
4. Migrate `cur` onto `t_new` via `_interp_to_grid` (lifted out of
   `refine_time` as a shared helper, `kink_opt/verify.py:30`) for `A, XI, B, ETA`;
   all-alive masks (no windows yet); re-solve the weight LPs once to
   restore exact feasibility. Check `certify(cur)["Jc"]` is within 1% of
   `base_Jc` (a relative bar, not exact equality -- Run 7 already
   established that changing node count changes the harvest sum's discrete
   sampling slightly, so an absolute-zero bar would be too strict); raises
   `RuntimeError` past that, since a silent large mismatch would corrupt
   every downstream `dJk`.
5. Loop `k in 1..n_gen`: `windowed = generation_step(cur, windows_k, ...)`,
   `guard = generation_step(cur, (t0, t1), ...)`, record `dJk =
   windowed["Jc"] - J`, `guard_dJk = guard["Jc"] - J`, advance
   `cur = windowed["sol"]` (the guard is comparison-only, never adopted --
   it can't quietly degrade into Runs 4-5's unrestricted growth).

Returns `dict(generations=[dict(k, w_k, window, Jc, dJk, feasible,
guard_Jc, guard_dJk, guard_feasible, diagnostics, spread), ...], base_Jc)`.

### 6. `main()` Run 9 block

**Location:** `kink_opt/demos.py:243` onward, after Run 8.

Narrated header (Run 8's null doesn't block the measurement; imposed
windows are the point, not the warm start), `ladder = generation_ladder(G0,
n_gen=3, window0=0.5, window_ratio=0.5, seeds=range(3), outer=25,
pos_iters=60, coarse_N=8, base_fine_sub=4, sub=8)`, a per-generation table
(`k, w_k, Jc, dJk, ok, guard_Jc, guard_dJk, ok`), the diagnostics line for
each new kink, the per-seed spread, and a closing `dJk` / `guard_dJk`
sequence print pointing at STRATEGY.md Section 5 for the interpretation
rule.

## Verification performed

- `graded_grid` list-`fine_sub`: scalar-vs-matching-list bit-identical,
  mismatched length raises, varying values change density, empty-windows
  path unaffected.
- `_interp_to_grid` extraction: `refine_time` regression-checked (coarse
  node values preserved exactly on the fine grid) before and after the
  refactor.
- `_kink_diagnostics`: toy-solution unit test for both branches (active
  weight, fully-pruned fallback).
- `generation_step`: smoke-tested standalone on a toy 2+2 run (windowed and
  guard-arm code paths both exercised); the fully-pruned diagnostics
  fallback fired correctly in a real (non-contrived) case.
- `generation_ladder`: regrid-neutrality bug found and fixed (see Status);
  full driver reproduces Run 1-8 baselines byte-for-byte unchanged
  (1.916 / 2.220 / 2.309 / 2.420 / 2.544 / 2.411 / 2.2925 / 2.3065 / Run 8
  null) after every step.
- No changes were made to `total_J`, `grad_total_J`, `penalty`,
  `grad_penalty`, `_step_diff_grad`, or `optimize_positions` -- confirmed
  out of scope for this feature, as planned.

## Result (see run9-generation-gain-ladder.md for interpretation)

On Run 3's G0 (`J_certified` 2.3089), 3 generations at
`window0=0.5, window_ratio=0.5`: `dJk = [+0.0546, +0.1048, +0.0045]` against
`guard_dJk = [+0.0926, +0.1229, +0.0309]` -- guard ahead at every
generation, `dJk` non-monotone over only 3 points. Not a verdict; see the
strategy plan for what to run next.

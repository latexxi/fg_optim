# Code plan: Run 9 generation-gain ladder

Companion to `plans/run9-generation-gain-ladder.md` (strategy). This one is
implementation-level: exact functions, signatures, file locations, data flow.

## New code, in order

### 1. `graded_grid` — extend `fine_sub` to accept per-window density

**Location:** `kink_opt.py`, existing function (~line 439).

Problem: as w_k halves each generation, a single scalar `fine_sub` makes
`n_loc = max(2, round(fine_sub * coarse_N * span / (t1-t0)))` shrink toward
the floor of 2 — violates the "finest lifetime spans >= 8 fine steps" rule
from STRATEGY.md Section 5.

Change: `fine_sub` may be a scalar (old behavior, unchanged) or a list same
length as `windows`, giving each window its own density. Ladder code passes
`fine_sub_k = base_fine_sub * (window0 / w_k)` per window so `n_loc` stays
roughly flat across generations instead of shrinking. Backward compatible —
scalar path is bit-for-bit identical, so Runs 1-7 stay unchanged (verify with
existing invariant check before moving on).

### 2. `_seed_grown` — promote from Run 8's local closure to a module function

**Location:** new, near `_insert_column` (~line 686) — it's Task B/D
machinery, not driver code.

Signature: `_seed_grown(base, XI2, ETA2, af2, ag2)`. Currently this logic
lives inline in the `__main__` Run 8 block (lines ~1004-1013): pad `base["A"]`
/ `base["B"]` with a zero column for each grown family, then run the
weight-LP pair (`lp_weights_f` -> `lp_weights_g` -> `lp_weights_f`) to
bootstrap a feasible, J-unchanged starting point after `_insert_column` /
`add_kink` grows a family by one column. The ladder calls this every
generation (Run 8 only needed it once), so it earns a real name instead of a
closure. Run 8's `__main__` block gets updated to call the promoted function
(no behavior change, just deduplication).

### 3. `_kink_diagnostics` — post-optimization measurement helper

**Location:** new, near `_lifetime_window` (~line 700).

Signature: `_kink_diagnostics(r, family, col_idx, parent_idx, tol=1e-8)`.
Returns `dict(lifetime=(t_on, t_off), extent=(min_x, max_x), jump_mean,
offset_from_parent)`:

- `t_on, t_off`: first/last time node where `|weight[:, col_idx]| > tol`
  (the *effective* active window, which may be narrower than the imposed
  mask — this is exactly the self-similarity test: does the optimizer keep
  the full imposed window or contract it further on its own).
- `extent`: min/max of the position trajectory over the active window.
- `jump_mean`: mean over the active window of `2*w/(1-x**2)` (the "jump"
  formula from the module docstring), using the family's own weight column.
- `offset_from_parent`: mean `|position[:, col_idx] - position[:, parent_idx]|`
  over the active window — tests "riding on the parent path."

This is pure measurement, no solver interaction — keeps STRATEGY.md's
secondary-measurement list (lifetime/extent, jump size, offset) machine-
computed rather than eyeballed from printouts.

### 4. `generation_step` — one rung of the ladder

**Location:** new, near `grow_topology` (~line 763) — same family of
"insert, re-optimize, certify" drivers.

Signature:
```
generation_step(cur, window, seeds=range(3), dx=0.05,
                outer=12, pos_iters=40, patience=None, sub=8)
```
`window = (t_birth, t_death)`, applied identically to both a new f-kink and a
new g-kink (STRATEGY: "1 for f, 1 for g"). For each `rng_seed` in `seeds`:

1. Pick each family's current most-active column as parent (`argmax` of
   `max(|weight|, axis=0)`, same idiom as `grow_topology`/`spawn_generation`).
2. `add_kink("f", ..., parent_f, t, *window, dx=dx, rng=rng)`, then
   `add_kink("g", ..., parent_g, t, *window, dx=dx, rng=rng)` chained on the
   growing XI/ETA/alive_f/alive_g.
3. `_seed_grown` to bootstrap feasible weights for the two new (zero-weight)
   columns.
4. `_alternate(..., outer=outer, pos_iters=pos_iters, patience=patience or
   outer)` to re-optimize.
5. `prune` (drops the new kink if the LP starved it — a real possible
   outcome, not an error).
6. `certify` for `Jc`.

Keep the best-`Jc` seed; return
`dict(sol=best_r, Jc=best_Jc, diagnostics=[_kink_diagnostics(...) for each
new column], spread=[Jc per seed])` — the spread is the "report the spread,
not just the max" honesty requirement.

Also used for the **guard arm**: called with `window = (t[0], t[-1])`
(full lifetime) instead of the imposed shrinking window — same function, no
separate code path, since "windowed vs free" is just a `window` argument.

### 5. `generation_ladder` — the driver

**Location:** new, near `grow_topology`.

Signature:
```
generation_ladder(base, n_gen=4, window0=0.5, window_ratio=0.5,
                   seeds=range(3), dx=0.05, base_fine_sub=4, coarse_N=8,
                   outer=12, pos_iters=40, sub=8, verbose=True)
```

Steps:
1. `cur = prune(base, 1e-8)`; `J = [certify(cur, sub=sub)["Jc"]]`.
2. Precompute the full window schedule up front:
   `w_k = window0 * window_ratio**(k-1)` for `k in 1..n_gen`,
   `windows_k = (t1 - w_k, t1)` (anchored at the shared travel-path end,
   matching Run 8's window convention).
3. Build one graded grid for the *whole* ladder via the extended
   `graded_grid(windows=[windows_k for all k], coarse_N=coarse_N,
   fine_sub=[base_fine_sub * window0 / w_k for each k])` — computed once,
   not regridded per generation, so every generation's certification sees
   the same node set plus its own refine_time multiplier.
4. Migrate `cur` onto that grid once: linear-interpolate `A, XI, B, ETA`
   from `cur["t"]` onto the new `t` (same interpolation `refine_time`
   already does internally — reuse its `interp` closure or lift it to a
   shared helper), set `alive_f`/`alive_g` all-True on the new grid (G0 has
   no windows yet), re-solve the weight LPs once to restore exact
   feasibility on the new grid, then confirm `certify` reproduces `J[0]`
   within tolerance (regression guard — regridding must be J-neutral).
5. For `k in 1..n_gen`:
   - `windowed = generation_step(cur, windows_k, seeds=seeds, dx=dx,
     outer=outer, pos_iters=pos_iters, sub=sub)`
   - `guard = generation_step(cur, (t[0], t[-1]), seeds=seeds, dx=dx,
     outer=outer, pos_iters=pos_iters, sub=sub)`
   - `cur = windowed["sol"]` (the ladder always advances on the windowed
     arm — that's the hypothesis under test; the guard is comparison only,
     never adopted, so it can't quietly turn into Runs 4-5's unrestricted
     growth).
   - append `dict(k=k, w_k=w_k, Jc=windowed["Jc"],
     dJk=windowed["Jc"] - J[-1], guard_Jc=guard["Jc"],
     guard_dJk=guard["Jc"] - J[-1], diagnostics=windowed["diagnostics"],
     spread=windowed["spread"])` to the results list.
   - `J.append(windowed["Jc"])`.
6. Return `dict(generations=results, base_Jc=J[0])`.

### 6. `__main__` Run 9 block

**Location:** end of file, after Run 8 (~line 1050+), following the existing
narrated-header pattern (see Run 8's header at line 975 for house style).

- Header explains: Run 8 showed the warm start doesn't beat random at k=0;
  Run 9 doesn't need the warm start — it runs the actual generation-gain
  measurement STRATEGY.md Section 5 asks for, using `add_kink` multistart as
  the (already-working) insertion mechanism, with imposed shrinking windows
  as the experimental variable.
- `G0 = prune(r2, 1e-8)` (reuse Run 3's solution, same as Run 8).
- `res = generation_ladder(G0, n_gen=4, seeds=range(3))`.
- Print a table: `k, w_k, Jc, dJk, guard_Jc, guard_dJk, spread` — one row per
  generation — plus the per-generation diagnostics line (lifetime/extent/
  jump/offset for the two new kinks).
- Closing interpretation print: is `dJk` roughly flat or decaying, and does
  it survive comparison to `guard_dJk`.

## Verification before trusting any of this

- `graded_grid` per-window `fine_sub` list: unit-check against the existing
  scalar path (`fine_sub=[x]*len(windows)` must equal `fine_sub=x` exactly)
  before using it in the ladder.
- Regridding `cur` onto the precomputed ladder grid must reproduce
  `certify(base)["Jc"]` (within ~1e-6) before generation 1 runs — if it
  doesn't, the bug is in the migration step, not the ladder logic, and must
  be fixed first.
- `generation_step`'s zero-weight insertion must satisfy `total_J` unchanged
  immediately after `_seed_grown`, same check Run 8 already does for
  `spawn_generation` — reuse that assertion pattern here.
- No changes to `total_J`, `grad_total_J`, `penalty`, `grad_penalty`,
  `_step_diff_grad`, or `optimize_positions` are needed for this feature —
  if implementation pressure tempts a shortcut there, stop; that's the
  finite-difference-verified core and out of scope for Run 9.

## Suggested build order

1. `graded_grid` list-`fine_sub` extension + regression check against scalar
   behavior.
2. `_seed_grown` promotion (mechanical, dedupe Run 8).
3. `_kink_diagnostics` (pure function, easy to test standalone against a
   hand-built toy `r`).
4. `generation_step` (reuses 2 and existing `add_kink`/`_alternate`/`prune`/
   `certify`) — test in isolation with `n_gen=1` equivalent before wiring
   into the ladder.
5. `generation_ladder` (wires 1-4 together).
6. Run 9 `__main__` block.

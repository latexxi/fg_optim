# plans/mesh — adaptive harvest-gauge refinement for the full-mesh optimizer

Goal: reach deeper dyadic generations of the full-mesh solver (`mesh/` package)
at fixed node budget, by spending nodes only where harvest lives — a **tau-gauge**
time grid and **band-only** x refinement — so the bounded-vs-unbounded J question
can be pushed past the current RAM/time wall.

Read in order:

- `00-primer.md`  — context, the gauge facts that make this cheap, the copied-code
  map, the invariant ledger this rests on, and the honest caveat. **Read first.**
- `01-grids.md`   — `adapt.py`: `tau_regrid`, `band_refine` (grid construction).
- `02-prolong.md` — `prolong.py`: `regauge_time`, `prolong_x`, `adaptive_warm_start`.
- `03-driver.md`  — `refine_adapt.py`: `adaptive_refinement` driver.
- `04-validate.md`— validation gates, baseline comparison, figures.

Each task is self-contained: signatures, math, feasibility argument, and an
acceptance check. Implement + run the acceptance check before moving on.

The uniform baseline (`mesh/refine_baseline.py`, `dyadic_refinement`) is already
copied and is the comparison point — do not modify it.

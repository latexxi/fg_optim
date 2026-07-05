# Run 13 Stage-C implementation plans (the `↦ E′` half of the CELL)

These files break the remaining Run 13 work into ordered, self-contained tasks so
an agent starting from **empty context** can implement them. Read `00-primer.md`
first (it carries all the shared context), then do the numbered tasks in order.

The parent plan is `plans/run13-selfreproducing-cell.md`. The `E ↦ δ̂` half of the
cell is already built and passing (`kink_opt/cell.py`); these tasks build the
`↦ E′` half and drive it to the boundedness verdict.

## Order and dependencies

| File | Task | Depends on | Deliverable |
|------|------|-----------|-------------|
| `00-primer.md` | Shared context (read first) | — | — |
| `01-read-off.md` | `cell_read_env` + `cell_env_distance` | 00 | E′ read-off + its metric |
| `02-d1-gate.md` | `check_interior_slope` (D1 verification) | 00 | trust gate for channel 1 |
| `03-fixed-point-loop.md` | `cell_step`, `tiling_gain`, `fixed_point` | 01 | the `E → CELL(E)` loop + per-octave gains |
| `04-verdict-and-sweep.md` | `cell_sweep`, apply decision rule | 03 | the bounded/unbounded γ verdict |
| `05-sufficiency-probes.md` | A1 (E-sufficiency), A2 sensitivity | 03 | trust probes (deferred, non-blocking) |

01 and 02 are independent of each other; both only need 00. 03 needs 01. 04 needs
03. 05 needs 03 and is optional/deferred — do it only after 04 gives a verdict.

## Definition of done for the whole stage

`04` prints, for a sweep of frame-contraction values `r`, the converged cell
harvest `δ̂*`, the env-reproduction distance at convergence, the per-octave ratio
`γ = 2r`, and a bounded / unbounded / inconclusive verdict per the decision rule.
That verdict is the deliverable Run 13 exists to produce.

# 00 — Primer (read this first)

Shared context for all Run 13 Stage-C tasks. If you are a fresh agent, read this
top to bottom, then open the numbered task file you were assigned.

## 0.1 The problem

The repo is a numerical-optimization prototype (Python package `kink_opt/`, no build
system — plain imports; deps numpy/scipy/matplotlib assumed installed). It studies

    J[f, g] = ∫₀¹ ∫₋₁¹ f_t(x,t) · g_xx(x,t) dx dt

maximized over pairs f(x,t), g(x,t) on x∈[-1,1], t∈[0,1], both convex in x, Lipschitz
(|f_x|,|g_x| ≤ 1), with f(±1,t)=g(±1,t)=0, f(x,1)=0, g(x,0)=0, f_t ≥ 0, g_t ≤ 0.

The open question: is `sup J` bounded, or does it grow without limit (mesh data shows
`J ∼ ln(resolution)`)? **Run 13's thesis:** the answer is decided by a per-octave
increment read at a *fixed point* of a renormalization map, not by any single
optimum. You do not need to understand the whole theory — just the cell interface
below.

## 0.2 Representation (how f and g are stored)

Each time slice is a negative sum of "hat" (tent) basis functions:

    f(x,t) = -Σ_i A[k,i] · hat(x; XI[k,i]),   A ≥ 0
    g(x,t) = -Σ_m B[k,m] · hat(x; ETA[k,m]),  B ≥ 0

`hat(x;c)` is the piecewise-linear tent with `hat(±1)=0`, `hat(c)=1`. Convexity in x
is automatic (weights ≥ 0); `f(±1)=0` is built in. `XI`/`ETA` are kink positions,
`A`/`B` are weights, indexed `[time_node k, kink i]`. Arrays have shape `(Np1, K)`
where `Np1` = number of time nodes, `K` = number of kinks in that family.

A **solution dict** `sol` has keys: `A, XI, B, ETA` (each `(Np1, K)`), `t` (the time
grid, shape `(Np1,)`), and optionally `alive_f, alive_g` (bool masks, lifetime
windows). `total_J(A, XI, B, ETA)` (in `kink_opt/objective.py`) computes J.

## 0.3 The cell — what already exists

`kink_opt/cell.py` implements the **`E ↦ δ̂` half** of the renormalization map
`CELL: E ↦ (δ̂, E′)`. It solves **one octave** on a fixed unit frame (x̂∈[-1,1],
t̂∈[0,1], a single kink per family, Kf=Kg=1), taking an incoming **environment** `E`
as boundary data, at O(1) cost independent of octave depth.

`E` is a dict (see `flat_env`): `dict(x_hat, beta, rho, r)`:
- `x_hat` — dimensionless sample points in the child frame, shape `(n_sample,)`, spanning ≈[-1,1].
- `beta` — slope-slack profile `β(x̂) = 1 - |f_x|` (residual Lipschitz slack the parent left). Shape `(n_sample,)`, entries in [0,1].
- `rho` — remaining-rise profile `ρ(x̂) = (1-|x|) + f` (residual rise budget), **physical amplitude units**. Shape `(n_sample,)`.
- `r` — the per-octave amplitude/width contraction of the frame (a scalar; see §0.6). Seed uses `r=1.0`.

`E` couples into the LP through exactly two channels (both already wired):
1. **Slope budget** `β → lip_rhs`: `lp_weights_f(..., lip_rhs=)` replaces the constant Lipschitz RHS `1.0` with the residual slack. `env_to_lp` maps this (currently the D1 arm-only default: a scalar cap `min(β)`).
2. **Rise budget** `ρ → rise_cap`: `lp_weights_f(..., rise_cap=(xs, caps))` adds explicit rows `Σ_i A[k,i]·hat(xs; XI[k,i]) ≤ caps` at every node. `env_to_lp` passes `caps = ρ / r` (the A2 rescaling, §0.6).

Key functions in `cell.py` (all present, tested):
- `flat_env(n_sample=41) -> E` — the seed environment E₀ (β≡1, ρ=1-|x̂|, r=1). Injecting it is a no-op.
- `env_to_lp(env, Np1) -> (lip_rhs, rise_cap)` — maps E to the two LP kwargs.
- `_unit_carrier(coarse_N=8) -> (t, XI, ETA)` — the gen-0 travelling carrier, single kink per family, on t∈[0,1]. Positions from `construct._travel_path` + `XI_OFFSET=-0.10` / `ETA_OFFSET=-0.02`.
- `_alternate_injected(XI, ETA, lip_rhs, rise_cap, outer=40, tol=1e-9) -> (A, B)` — LP-only alternation (positions frozen) with E injected into every LP solve. Monotone in J (each block is an exact LP optimum).
- `cell_solve(env, coarse_N=8, outer=40, sub=8) -> dict(J, Jc, constraints_ok, sol)` — one full CELL evaluation of the `↦ δ̂` half: seed carrier, inject E, alternate, `certify`. `Jc` is the certified harvest (call it `δ̂`). `sol` is the solution dict.
- `_flat_gate()` — asserts injecting `flat_env()` reproduces the injection-free carrier (`diff < 1e-9`). Run `python3 -m kink_opt.cell` to see it pass.

## 0.4 Certification (trust the certified J, not the raw J)

`certify(sol, sub=8, lip_rhs=None, rise_cap=None)` (in `kink_opt/verify.py`) is the
honest J: it refines the time grid, repairs feasibility by re-solving the weight LPs
there, and checks every constraint on a dense x-grid. **For a cell you MUST pass the
same `lip_rhs`/`rise_cap`** or the repair washes the injected environment out. Read
`Jc` and `rep["ALL CONSTRAINTS OK"]` from its return dict. `cell_solve` already does
this correctly — reuse it; do not hand-roll certification.

## 0.5 What is still missing (your job, across tasks 01–05)

The `↦ E′` half and the driver that turns it into a verdict:
- **Read E′** off a solved cell (`cell_read_env`, task 01).
- **A distance** between environments, normalized so units can't fake a fixed point (`cell_env_distance`, task 01).
- **A D1 verification** that the arm-only slope channel is sufficient (task 02).
- **The loop** `E_{n+1} = CELL(E_n)` to a fixed point, plus the tiling multiply that turns per-cell `δ̂` into per-octave gain (task 03).
- **The verdict**: sweep the frame contraction `r`, apply the decision rule (task 04).
- **Sufficiency probes** A1/A2 (task 05, deferred).

## 0.6 The `r` derivation and read-frame (design fork D3 — RESOLVED, use as given)

These two decisions are settled; implement exactly as stated. Do not re-derive.

**Why `r` exists.** Work the child in its unit frame x̂=(x−c)/w. Unit-frame Lipschitz
`|f̂_x̂|≤1` with physical `|f_x|≤1` forces the f-amplitude `A_f = w` (the spatial
half-width). So **rise budget scales like width**. `ρ` has amplitude units, so to use
the parent's residual `ρ` as a cap in the child's unit-frame LP you must divide by the
child amplitude:

    ρ̂ = ρ / w_child,   and per octave w_child = r · w_parent,   so   **r = the per-octave width (= amplitude) contraction of the frame.**

`env_to_lp` already applies `ρ / r`. `r` is a **chosen parameter of the coordinate
frame** (like `λ_w` in `melt.py`), *not* a solved unknown — you sweep it (task 04).
Slope-slack `β` is frame-invariant (dimensionless), so it is **not** divided by r.

**The verdict ratio.** One cell's J-contribution scales like its width (`[J]=[f][g]/[x]
= w·w/w = w`), so octave-k cells each contribute `∝ r^k`. Tiling puts `2` cells per
parent window per octave (§5 of the parent plan — pinned). Hence per-octave gain
`Δ_k = 2^k · r^k · δ̂ = (2r)^k · δ̂`, and total `J = δ̂ · Σ_k (2r)^k`. **So the
per-octave ratio is `γ = 2r`** (given `δ̂` converges to a constant). `r = 1/2 ⟹ γ = 1`
is the log-growth knife-edge: `2r < 1` bounded, `2r > 1` unbounded — a cross-check on
whatever the env-reproduction + `δ̂` sequence shows.

**Read-frame — where to read E′.**
- **Spatially:** the child co-moving frame — center `c` = the carrier kink position at the read node; sample `x_abs = clip(c + w_next · x̂, -1+MARGIN, 1-MARGIN)` with `w_next = r` (the next octave's half-width in this unit frame) and x̂ over the same grid as the incoming E. The env is expressed in dimensionless x̂∈[-1,1]; the absolute mapping differs octave to octave but the profile lives on the fixed x̂ grid, so `cell_env_distance` is well-posed.
- **Temporally: read at t̂ = 1/2 (window midpoint), NOT t̂ = 1.** Reason: the unit carrier pins its terminal node dead (`f̂(x̂,1)=0`, a leak of the global `f(x,1)=0` BC), so ρ at t̂=1 reads as full/unspent — an artifact. t̂=0 is the injection seam (reads back the incoming E — circular). t̂=1/2 is the sustained interior, the representative residue a tiling's interleaved next-octave cells actually inherit.
- **`ρ` is stored physical** in E′ (dividing by r happens at the *next* injection via `env_to_lp`), matching how `flat_env` stores it. Seed `flat_env` keeps `r=1` (boundary data, not a cell output); every `cell_read_env` output carries `r` = the frame contraction.

## 0.7 Conventions and how to run

- Constants: `MARGIN` (keeps kinks inside the domain) is in `kink_opt/geometry.py`, imported in `cell.py`. Use it when clipping x-positions.
- Reuse, don't reinvent: `_slope_f(x_query, a, xi)` (exact PL slope, in `kink_opt/melt.py`), `conv_eval(x, a, xi)` (evaluates `-Σ a·hat`, in `kink_opt/geometry.py`), `total_J` (in `kink_opt/objective.py`), `certify` (in `kink_opt/verify.py`).
- Run the cell's self-check: `python3 -m kink_opt.cell` (currently prints the flat-E no-op gate). Add your own `__main__` demos there or in a task-specific script; keep the existing gate passing.
- There is **no test suite**. Each task file specifies its own acceptance gate — implement and run it, and show the numbers.
- Do not touch the LP defaults or `certify`'s default (None) path — Runs 1–12 depend on them being bit-for-bit unchanged. Everything you add is opt-in via new functions or new default-None kwargs.

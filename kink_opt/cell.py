"""Run 13 (plans/run13-selfreproducing-cell.md, esp. §4/§5A) -- the unit-frame
renormalization cell  CELL: E -> (delta_hat, E').

The boundedness of sup J is decided by the per-octave increment read at a fixed
point of this map, NOT by any generation optimum. The cell solves one octave in a
fixed unit frame with the incoming environment E (the residue the parent leaves) as
boundary data, at O(1) cost independent of depth -- unlike the global-hierarchy
builds (construct.py/melt.py) whose cost grows with generation count.

E couples through exactly two LP channels (§5A), both now supported by lp.py:
  channel 1  beta(x) = 1-|f_x|      -> lp_weights_f(..., lip_rhs=)   residual slope
  channel 2  rho(x)  = (1-|x|)+f    -> lp_weights_f(..., rise_cap=)  residual rise

STATUS: LP injection, `certify` passthrough (so certified J reflects the injected
environment), and the flat-E no-op gate are implemented and pass. The rest of the
map (E' read-off, rho/r rescaling, tiling multiply, the fixed-point iteration) is
gated on design forks D1-D3 in the plan's §5A -- see the TODO in `cell_solve`.
"""

import numpy as np

from .geometry import MARGIN, conv_eval
from .lp import lp_weights_f, lp_weights_g
from .objective import total_J
from .verify import certify
from .construct import _travel_path, XI_OFFSET, ETA_OFFSET


def flat_env(n_sample=41):
    """E_0: the coarsest carrier's residue -- full slope slack (beta==1) and full
    rise budget (rho == 1-|x|), rise share r=1. Injecting this must be a no-op."""
    xs = np.linspace(-1 + MARGIN, 1 - MARGIN, n_sample)
    return dict(x_hat=xs, beta=np.ones_like(xs), rho=1.0 - np.abs(xs), r=1.0)


def env_to_lp(env, Np1):
    """Map E to the f-LP boundary-data kwargs.
      channel 1 (D1 default: arm-only, conservative scalar cap = min slack;
        flat beta==1 => cap 1, a no-op),
      channel 2 (A2 rescaling: rho/r in the child frame)."""
    cap = min(float(np.min(env["beta"])), 1.0)
    lip_rhs = np.full((Np1, 2), cap)
    rise_cap = (np.asarray(env["x_hat"], float),
                np.asarray(env["rho"], float) / max(float(env["r"]), 1e-12))
    return lip_rhs, rise_cap


def _unit_carrier(coarse_N=8):
    """The gen-0 travelling carrier on the unit frame (Kf=Kg=1), same seed as
    construct.build_hierarchy's column 0."""
    t = np.linspace(0.0, 1.0, coarse_N + 1)
    p = _travel_path(t)
    XI = np.clip(p + XI_OFFSET, -1 + MARGIN, 1 - MARGIN)[:, None]
    ETA = np.clip(p + ETA_OFFSET, -1 + MARGIN, 1 - MARGIN)[:, None]
    return t, XI, ETA


def _alternate_injected(XI, ETA, lip_rhs, rise_cap, outer=40, tol=1e-9):
    """LP-only alternation (positions frozen) WITH E injected into every LP
    re-solve -- the injected analogue of _alternate(optimize_pos=False). Each
    block is an exact LP global optimum, so J is monotone non-decreasing."""
    Np1 = XI.shape[0]
    B = np.zeros((Np1, ETA.shape[1]))
    A = lp_weights_f(XI, B, ETA, lip_rhs=lip_rhs, rise_cap=rise_cap)
    Jprev = -np.inf
    for _ in range(outer):
        A = lp_weights_f(XI, B, ETA, lip_rhs=lip_rhs, rise_cap=rise_cap)
        B = lp_weights_g(A, XI, ETA, lip_rhs=lip_rhs)
        J = total_J(A, XI, B, ETA)
        if J - Jprev < tol:
            break
        Jprev = J
    return A, B


def cell_solve(env, coarse_N=8, outer=40, sub=8):
    """One CELL evaluation: seed the unit carrier, inject E, alternate LP-only to
    the convex fixed point, certify. Returns dict(J, Jc, sol).

    TODO (Stage C, forks D1-D3): read E' via read_environment on the output, the
    rho/r rescaling normalization, the 2^k tiling multiply, and the E->CELL(E)
    fixed-point loop."""
    t, XI, ETA = _unit_carrier(coarse_N)
    lip_rhs, rise_cap = env_to_lp(env, t.size)
    A, B = _alternate_injected(XI, ETA, lip_rhs, rise_cap, outer=outer)
    sol = dict(A=A, XI=XI, B=B, ETA=ETA, t=t,
               alive_f=np.ones_like(XI, bool), alive_g=np.ones_like(ETA, bool))
    c = certify(sol, sub=sub, lip_rhs=lip_rhs, rise_cap=rise_cap)
    return dict(J=total_J(A, XI, B, ETA), Jc=c["Jc"],
                constraints_ok=bool(c["rep"]["ALL CONSTRAINTS OK"]), sol=sol)


def _flat_gate(coarse_N=8):
    """Gate: injecting flat E_0 must reproduce the plain (injection-free) carrier
    -- validates the whole channel-1+channel-2 plumbing end to end."""
    t, XI, ETA = _unit_carrier(coarse_N)
    # plain: no injection
    Np1 = t.size
    B = np.zeros((Np1, ETA.shape[1]))
    A = lp_weights_f(XI, B, ETA)
    Jprev = -np.inf
    for _ in range(40):
        A = lp_weights_f(XI, B, ETA)
        B = lp_weights_g(A, XI, ETA)
        J = total_J(A, XI, B, ETA)
        if J - Jprev < 1e-9:
            break
        Jprev = J
    J_plain = total_J(A, XI, B, ETA)
    # injected flat
    r = cell_solve(flat_env(), coarse_N=coarse_N)
    dJ = abs(r["J"] - J_plain)
    return dict(J_plain=J_plain, J_flat=r["J"], Jc_flat=r["Jc"],
                diff=dJ, ok=dJ < 1e-9)


def cell_read_env(sol, r, n_sample=41, family="f"):
    """Outgoing environment E' of a solved cell (the input to the next octave).
    Reads the residual slope-slack beta(x_hat) and remaining-rise rho(x_hat) at
    the cell's temporal MIDPOINT (t_hat = 1/2; NOT t_hat=1, which is terminal-
    pinned -- see 00-primer.md §0.6), sampled over the next octave's support
    (half-width w_next = r about the carrier's midpoint position). rho is stored
    PHYSICAL (env_to_lp divides by r at the next injection). Returns a dict with
    the same schema as flat_env: dict(x_hat, beta, rho, r)."""
    from .melt import _slope_f           # exact PL slope; already in the repo
    t = sol["t"]
    idx = int(np.argmin(np.abs(t - 0.5)))          # t_hat = 1/2 read node
    W = sol["A"] if family == "f" else sol["B"]
    P = sol["XI"] if family == "f" else sol["ETA"]
    a, xi = W[idx], P[idx]
    c = float(xi[np.argmax(a)]) if a.size else 0.0  # carrier position (heaviest kink)
    x_hat = np.linspace(-1.0 + MARGIN, 1.0 - MARGIN, n_sample)
    x_abs = np.clip(c + r * x_hat, -1.0 + MARGIN, 1.0 - MARGIN)
    slope = _slope_f(x_abs, a, xi)
    beta = 1.0 - np.abs(slope)
    rho = (1.0 - np.abs(x_abs)) + conv_eval(x_abs, a, xi)
    return dict(x_hat=x_hat, beta=beta, rho=rho, r=float(r))


def cell_env_distance(Ea, Eb):
    """Self-reproduction metric between two environments on the SAME x_hat grid.
    Max-norm of concatenated (beta, rho/r) differences -- rho is normalized by
    the frame contraction r (A2) so a change of frame alone cannot fake a match.
    Raises if the x_hat grids differ."""
    if not np.allclose(Ea["x_hat"], Eb["x_hat"]):
        raise ValueError("cell_env_distance requires a common x_hat grid")
    da = np.concatenate([Ea["beta"], Ea["rho"] / max(Ea["r"], 1e-12)])
    db = np.concatenate([Eb["beta"], Eb["rho"] / max(Eb["r"], 1e-12)])
    return float(np.max(np.abs(da - db)))


def check_interior_slope(sol, family="f", tol=1e-9):
    """D1 gate: verify max interior |f_x| <= max arm |f_x| on a solved cell's
    read slice, so the arm-only slope channel (env_to_lp) faithfully caps the
    residue. Returns dict(ok, max_interior, arm_left, arm_right, margin)."""
    from .melt import _slope_f
    t = sol["t"]
    idx = int(np.argmin(np.abs(t - 0.5)))
    W = sol["A"] if family == "f" else sol["B"]
    P = sol["XI"] if family == "f" else sol["ETA"]
    a, xi = W[idx], P[idx]
    xs = np.linspace(-1.0 + MARGIN, 1.0 - MARGIN, 401)
    interior = np.abs(_slope_f(xs, a, xi)).max()
    arm_left = np.abs(_slope_f(np.array([-1.0 + MARGIN]), a, xi)[0])
    arm_right = np.abs(_slope_f(np.array([1.0 - MARGIN]), a, xi)[0])
    arm = max(arm_left, arm_right)
    return dict(ok=bool(interior <= arm + tol), max_interior=float(interior),
                arm_left=float(arm_left), arm_right=float(arm_right),
                margin=float(arm - interior))


def cell_step(env, r, coarse_N=8, outer=40, sub=8):
    """One octave of CELL: E -> (delta_hat, E'). Solves the injected cell
    (existing half) and reads its outgoing environment (task 01). Returns
    dict(delta_hat, env_out, constraints_ok, sol)."""
    res = cell_solve(env, coarse_N=coarse_N, outer=outer, sub=sub)
    env_out = cell_read_env(res["sol"], r=r)
    return dict(delta_hat=res["Jc"], env_out=env_out,
                constraints_ok=res["constraints_ok"], sol=res["sol"])


def fixed_point(r, n_iter=12, tol=1e-4, coarse_N=8, outer=40, sub=8, seed=None):
    """Iterate E_{n+1} = CELL(E_n) at fixed frame contraction r, from `seed`
    (default: flat_env()), to a fixed point. Returns dict(deltas, dists, envs,
    converged, r).
      deltas[n]  = delta_hat of step n (certified per-cell harvest)
      dists[n]   = cell_env_distance(env_in, env_out) at step n
      converged  = True if dists[-1] < tol
    Stops early once dists < tol.

    `seed=None` reproduces the original flat_env()-seeded behavior exactly
    (tasks 03/04 depend on this). `seed=` lets task 05's probes inject a
    custom starting environment (a perturbed carrier / rescaling convention)
    without duplicating the iteration loop."""
    env = dict(flat_env()) if seed is None else dict(seed)  # copy; never mutate the input
    env["r"] = float(r)               # iterate at the tested frame contraction
    deltas, dists, envs = [], [], [env]
    for _ in range(n_iter):
        step = cell_step(env, r=r, coarse_N=coarse_N, outer=outer, sub=sub)
        if not step["constraints_ok"]:
            # a failed cell is not a data point; record NaN and stop
            deltas.append(float("nan")); dists.append(float("nan"))
            break
        d = cell_env_distance(env, step["env_out"])
        deltas.append(step["delta_hat"]); dists.append(d)
        envs.append(step["env_out"])
        env = step["env_out"]
        if d < tol:
            break
    converged = bool(dists and np.isfinite(dists[-1]) and dists[-1] < tol)
    return dict(deltas=deltas, dists=dists, envs=envs,
                converged=converged, r=float(r))


def tiling_gain(deltas, r, tiles_per_octave=2):
    """Per-octave gains from the per-cell harvest sequence (§0.6, §5 tiling).
      Delta_k = (tiles_per_octave * r)**k * delta_hat_k
      gamma_geom  = tiles_per_octave * r          (the pinned-tiling ratio)
      gamma_emp[k]= Delta_{k+1} / Delta_k         (empirical, catches delta drift)
    Returns dict(octave_gains, gamma_geom, gamma_emp, delta_star)."""
    d = np.asarray(deltas, float)
    k = np.arange(d.size)
    octave = (tiles_per_octave * r) ** k * d
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_emp = octave[1:] / octave[:-1]
    delta_star = float(d[np.isfinite(d)][-1]) if np.isfinite(d).any() else float("nan")
    return dict(octave_gains=octave.tolist(),
                gamma_geom=float(tiles_per_octave * r),
                gamma_emp=gamma_emp.tolist(), delta_star=delta_star)


def _verdict(fp, tg):
    """Task 04 decision rule (parent plan §6, corrected): gamma near 1 is
    inconclusive both ways (a geometric sum converges up to gamma=1; a c/k tail
    keeps gamma<1 yet still diverges -- harmonic), so BOUNDED/UNBOUNDED both
    require a genuine env fixed point plus delta_star bounded away from 0, and
    only then does gamma's side of the gamma=1 knife-edge decide."""
    if not fp["converged"]:
        return "INCONCLUSIVE (no env fixed point)"
    g = tg["gamma_geom"]
    ds = float(tg["delta_star"])
    if not np.isfinite(ds) or ds <= 1e-6:
        return "INCONCLUSIVE (delta_star collapsed)"
    if g <= 0.9:
        return "BOUNDED"
    if g >= 1.0:
        return "UNBOUNDED"
    return "INCONCLUSIVE (gamma in knife-edge band)"


def cell_sweep(rs, n_iter=12, tol=1e-4, **kw):
    """Run fixed_point + tiling_gain for each frame contraction r in `rs`.
    Returns a list of dict(r, converged, dist_final, delta_star, gamma_geom,
    gamma_emp_last, verdict) -- the Run 13 verdict deliverable (task 04)."""
    out = []
    for r in rs:
        fp = fixed_point(r, n_iter=n_iter, tol=tol, **kw)
        tg = tiling_gain(fp["deltas"], r)
        dist_final = fp["dists"][-1] if fp["dists"] else float("nan")
        gamma_emp_last = tg["gamma_emp"][-1] if tg["gamma_emp"] else float("nan")
        out.append(dict(r=float(r), converged=fp["converged"],
                        dist_final=float(dist_final),
                        delta_star=tg["delta_star"],
                        gamma_geom=tg["gamma_geom"],
                        gamma_emp_last=float(gamma_emp_last),
                        verdict=_verdict(fp, tg)))
    return out


def _unit_carrier_jittered(coarse_N=8, xi_jit=0.0, eta_jit=0.0):
    """Like _unit_carrier, but with an extra constant offset added to each
    family's trajectory -- the "differ internally" knob for probe_e_sufficiency.
    Does not modify _unit_carrier (task 01-04 code); a small local variant."""
    t = np.linspace(0.0, 1.0, coarse_N + 1)
    p = _travel_path(t)
    XI = np.clip(p + XI_OFFSET + xi_jit, -1 + MARGIN, 1 - MARGIN)[:, None]
    ETA = np.clip(p + ETA_OFFSET + eta_jit, -1 + MARGIN, 1 - MARGIN)[:, None]
    return t, XI, ETA


def _cell_solve_jittered(env, xi_jit=0.0, eta_jit=0.0, coarse_N=8, outer=40, sub=8):
    """Like cell_solve, but seeded from _unit_carrier_jittered instead of
    _unit_carrier -- gives probe_e_sufficiency a second cell that is genuinely
    internally different (different kink trajectory) while still able to read
    off a matching E. Mirrors cell_solve's body exactly aside from the seed."""
    t, XI, ETA = _unit_carrier_jittered(coarse_N, xi_jit, eta_jit)
    lip_rhs, rise_cap = env_to_lp(env, t.size)
    A, B = _alternate_injected(XI, ETA, lip_rhs, rise_cap, outer=outer)
    sol = dict(A=A, XI=XI, B=B, ETA=ETA, t=t,
               alive_f=np.ones_like(XI, bool), alive_g=np.ones_like(ETA, bool))
    c = certify(sol, sub=sub, lip_rhs=lip_rhs, rise_cap=rise_cap)
    return dict(J=total_J(A, XI, B, ETA), Jc=c["Jc"],
                constraints_ok=bool(c["rep"]["ALL CONSTRAINTS OK"]), sol=sol)


def probe_e_sufficiency(r=0.5, tol_env=1e-3, xi_jit=2e-4, eta_jit=-1.5e-4, **kw):
    """A1 (plans/run13/05-sufficiency-probes.md): is E a sufficient statistic?
    Build two solved cells that agree in their read-off E to tight tolerance
    but differ internally, then check whether their *next-step* delta_hat
    also agrees. If it doesn't, E is missing a coordinate (first candidate:
    parent's local drift velocity at handoff) and no fixed point of CELL
    means anything on its own.

    NOTE on the "differ internally" recipe: the task doc's suggested variant
    (same flat seed, reached with 2x `outer`) was tried first and found
    degenerate here -- _alternate_injected's convex block-alternation already
    converges (J stops improving past 1e-9) in ~1-2 outer iterations for this
    carrier, so outer=40 vs outer=80 produces BIT-IDENTICAL internal states,
    not just matching E (env_gap=0, delta_gap=0 -- a vacuous pass). Instead
    this perturbs the carrier's spatial offset by a small constant jitter
    (xi_jit/eta_jit), which does converge to a genuinely different internal
    solution (different Jc, not bit-identical) while still reading off an E
    within tol_env of the unperturbed cell's -- a real test of the recipe's
    intent.

    Returns dict(env_gap, delta_a, delta_b, delta_gap, ok)."""
    a = cell_step(dict(flat_env(), r=r), r=r, **kw)
    b_solved = _cell_solve_jittered(dict(flat_env(), r=r), xi_jit=xi_jit,
                                     eta_jit=eta_jit,
                                     coarse_N=kw.get("coarse_N", 8),
                                     outer=kw.get("outer", 40),
                                     sub=kw.get("sub", 8))
    b = dict(delta_hat=b_solved["Jc"], env_out=cell_read_env(b_solved["sol"], r=r),
              constraints_ok=b_solved["constraints_ok"], sol=b_solved["sol"])
    env_gap = cell_env_distance(a["env_out"], b["env_out"])
    # step once more from each and compare the harvest
    da = cell_step(a["env_out"], r=r, **kw)["delta_hat"]
    db = cell_step(b["env_out"], r=r, **kw)["delta_hat"]
    delta_gap = abs(da - db)
    return dict(env_gap=float(env_gap), delta_a=float(da), delta_b=float(db),
                delta_gap=float(delta_gap),
                ok=bool(env_gap < tol_env and delta_gap < 5e-3))


def probe_rho_rescaling_sensitivity(r=0.5, factors=(0.8, 1.0, 1.25), **kw):
    """A2 (plans/run13/05-sufficiency-probes.md): is the verdict robust to the
    rho/r normalization *convention*, as opposed to r itself (already swept by
    task 04)? Re-runs fixed_point from a seed whose rho is pre-scaled by an
    extra convention factor and reports the converged delta_star per factor.

    CAVEAT (carried over from task 03's finding): the converged delta_star is
    bit-identical across all r<1 because env_to_lp's channel-2 rise cap
    rho/r only LOOSENS for r<1 -- channel 2 never binds, channel 1 (the
    arm-only slope cap) drives everything. So a flat table here (converged
    True/True/True, delta_star constant across factors) is the EXPECTED,
    TRIVIAL outcome in this regime, not evidence the ρ/r convention is
    robustly correct -- it is evidence channel 2 is inert. This probe would
    only be informative at an r (or a construction) where the rise-budget
    channel actually binds; as run here it is a much weaker check than A1.

    Returns list of dict(factor, converged, delta_star)."""
    out = []
    for fac in factors:
        seed = dict(flat_env())
        seed["r"] = r
        seed["rho"] = seed["rho"] * fac
        fp = fixed_point(r, seed=seed, **kw)
        tg = tiling_gain(fp["deltas"], r)
        out.append(dict(factor=float(fac), converged=fp["converged"],
                         delta_star=tg["delta_star"]))
    return out


if __name__ == "__main__":
    g = _flat_gate()
    print(f"flat-E no-op gate: J_plain={g['J_plain']:.6f}  J_flat={g['J_flat']:.6f}"
          f"  diff={g['diff']:.2e}  Jc={g['Jc_flat']:.6f}  ok={g['ok']}")

    print("\nRun 13 -- cell_sweep: fixed-point delta_hat/gamma vs frame contraction r")
    print("(gamma_geom = 2r is pinned by tiling; r=0.5 is the log-growth knife-edge)")
    rs = [0.35, 0.45, 0.5, 0.55, 0.65]
    for row in cell_sweep(rs):
        print(f"  r={row['r']:.2f}  conv={row['converged']}  "
              f"dist={row['dist_final']:.2e}  d*={row['delta_star']:.4f}  "
              f"g_geom={row['gamma_geom']:.2f}  g_emp={row['gamma_emp_last']:.2f}  "
              f"-> {row['verdict']}")

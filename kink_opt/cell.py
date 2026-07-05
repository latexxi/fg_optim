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

from .geometry import MARGIN
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


if __name__ == "__main__":
    g = _flat_gate()
    print(f"flat-E no-op gate: J_plain={g['J_plain']:.6f}  J_flat={g['J_flat']:.6f}"
          f"  diff={g['diff']:.2e}  Jc={g['Jc_flat']:.6f}  ok={g['ok']}")

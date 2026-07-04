"""Time-grid refinement, lifetime-mask mapping, graded grids, and the
certify()/report() verification pipeline."""

import numpy as np

from .geometry import conv_eval
from .lp import lp_weights_f, lp_weights_g
from .objective import total_J


def refine_time(A, XI, B, ETA, t, sub=8):
    """Linear-in-time interpolation of weights AND positions onto a finer
    time grid. This is the continuous-time meaning of the discrete solution;
    evaluating J on the fine grid exposes any exploitation of the coarse
    midpoint quadrature (kinks whipsawing inside one step).

    The fine grid subdivides EACH coarse interval into `sub` pieces, so a
    non-uniform (graded, Task C) grid keeps its grading -- a dense lifetime
    window stays dense after refinement instead of being flattened to a
    global uniform spacing (which would under-resolve short windows and so
    misreport J).  On a uniform grid this is identical to the old global
    linspace (bit-for-bit, so Runs 1-6 certify unchanged)."""
    t = np.asarray(t, float)
    segs = [np.linspace(t[i], t[i + 1], sub + 1)[:-1] for i in range(len(t) - 1)]
    tf = np.concatenate(segs + [t[-1:]])
    return (_interp_to_grid(A, t, tf), _interp_to_grid(XI, t, tf),
            _interp_to_grid(B, t, tf), _interp_to_grid(ETA, t, tf), tf)


def _interp_to_grid(M, t, tnew):
    """Linear-in-time interpolation of each column of M (Np1, K) from grid `t`
    onto `tnew`. Shared by `refine_time` (uniform sub-refinement) and the
    Run 9 `generation_ladder` (one-off migration onto a precomputed graded
    grid)."""
    return np.column_stack([np.interp(tnew, t, M[:, j])
                            for j in range(M.shape[1])])


def _refine_mask(alive, t, tf):
    """Map a coarse-grid lifetime mask (Np1, K) to a fine grid `tf`.  Each
    column's window is contiguous, so it is fully described by its birth/death
    times (first/last coarse node where it is alive); a fine node is alive iff
    it lies in [birth, death].  All-alive columns map to all-alive (birth=t0,
    death=t1), so certification of windowless solutions is unchanged."""
    out = np.zeros((len(tf), alive.shape[1]), dtype=bool)
    for j in range(alive.shape[1]):
        idx = np.flatnonzero(alive[:, j])
        if idx.size == 0:
            continue
        out[:, j] = (tf >= t[idx[0]] - 1e-12) & (tf <= t[idx[-1]] + 1e-12)
    return out


def graded_grid(windows, coarse_N=8, fine_sub=4, t0=0.0, t1=1.0):
    """Build a global NON-uniform time grid (Task C): a coarse background of
    `coarse_N` uniform intervals over [t0, t1], with each lifetime window
    [tb, td] additionally subdivided `fine_sub` times denser.  Fine-generation
    kinks live fast and short, so uniform global grids would have to be dense
    EVERYWHERE to resolve the shortest window (~1/w_min nodes across the whole
    span); grading puts the nodes only where a kink is actually alive, so the
    total node count grows with the number of generations, not with 1/w_min.

    `windows` is a list of (t_birth, t_death). `fine_sub` is either a scalar
    (applied to every window, the original behaviour) or a sequence of the
    same length as `windows` giving each window its own density -- needed by
    the Run 9 generation ladder, where later (narrower) windows need a larger
    fine_sub just to keep their local node count from shrinking below the
    "finest lifetime spans >= 8 fine steps" floor. A scalar reproduces the
    exact old behaviour (`fine_sub=[x]*len(windows)` == `fine_sub=x`).

    The returned grid is sorted and de-duplicated, always contains the coarse
    nodes (so coarse-scale structure is representable) and both endpoints.
    Passing windows=[] (or windows that already span [t0,t1]) reproduces a
    plain uniform grid, so this is a strict superset of run()'s default grid."""
    fine_subs = (list(fine_sub) if hasattr(fine_sub, "__len__")
                 else [fine_sub] * len(windows))
    if len(fine_subs) != len(windows):
        raise ValueError("fine_sub list must match len(windows)")
    nodes = [np.linspace(t0, t1, coarse_N + 1)]
    for (tb, td), fs in zip(windows, fine_subs):
        tb, td = max(tb, t0), min(td, t1)
        if td <= tb:
            continue
        # local resolution: match the coarse step inside the window, then
        # refine it fine_sub x denser (at least 2 sub-intervals per window).
        span = td - tb
        n_loc = max(2, int(round(fs * coarse_N * span / (t1 - t0))))
        nodes.append(np.linspace(tb, td, n_loc + 1))
    t = np.unique(np.concatenate(nodes))
    # merge nodes closer than a tiny fraction of the coarse step (numerical
    # duplicates from window edges landing near coarse nodes)
    tol = 1e-9 * (t1 - t0)
    keep = np.concatenate([[True], np.diff(t) > tol])
    return t[keep]


def n_live_nodes(r):
    """Total count of LIVE (kink, time-node) decision variables in a solution
    -- the Task C cost metric.  A windowed kink only counts the nodes inside
    its lifetime; an all-alive kink counts every node.  For a uniform all-alive
    solution this is just (Kf+Kg)*Np1."""
    af, ag = r.get("alive_f"), r.get("alive_g")
    nf = int(af.sum()) if af is not None else r["XI"].shape[0] * r["XI"].shape[1]
    ng = int(ag.sum()) if ag is not None else r["ETA"].shape[0] * r["ETA"].shape[1]
    return nf + ng


def verify_dense(A, XI, B, ETA, t, nx=1601, tol=2e-2):
    """Rebuild f,g on a dense grid; check constraints; cross-check J via
       the mesh-friendly identity  J = -int int f_{xt} g_x dx dt."""
    x = np.linspace(-1.0, 1.0, nx)
    Np1 = len(t)
    F = np.array([conv_eval(x, A[k], XI[k]) for k in range(Np1)])
    G = np.array([conv_eval(x, B[k], ETA[k]) for k in range(Np1)])

    rep = {}
    rep["f terminal max|f(x,1)|"] = float(np.abs(F[-1]).max())
    rep["g initial  max|g(x,0)|"] = float(np.abs(G[0]).max())
    rep["min f_t (want >=0)"] = float(np.diff(F, axis=0).min())
    rep["max g_t (want <=0)"] = float(np.diff(G, axis=0).max())
    rep["max |f_x|"] = float(np.abs(np.diff(F, axis=1) / (x[1] - x[0])).max())
    rep["max |g_x|"] = float(np.abs(np.diff(G, axis=1) / (x[1] - x[0])).max())
    rep["min f 2nd-diff (want >=0)"] = float(np.diff(F, 2, axis=1).min())
    rep["min g 2nd-diff (want >=0)"] = float(np.diff(G, 2, axis=1).min())

    fx = np.gradient(F, x, axis=1)
    gx = np.gradient(G, x, axis=1)
    fxt = np.diff(fx, axis=0)                       # per-step change of slope
    gx_mid = 0.5 * (gx[1:] + gx[:-1])
    J_dense = -np.trapezoid(fxt * gx_mid, x, axis=1).sum()
    rep["J (dense cross-check)"] = float(J_dense)

    ok = (rep["f terminal max|f(x,1)|"] < 1e-9 and
          rep["g initial  max|g(x,0)|"] < 1e-9 and
          rep["min f_t (want >=0)"] > -1e-9 and
          rep["max g_t (want <=0)"] < 1e-9 and
          rep["max |f_x|"] < 1.0 + tol and
          rep["max |g_x|"] < 1.0 + tol and
          rep["min f 2nd-diff (want >=0)"] > -1e-9 and
          rep["min g 2nd-diff (want >=0)"] > -1e-9)
    rep["ALL CONSTRAINTS OK"] = ok
    return rep


def _ub(alive):
    """Lifetime mask -> per-node weight upper bounds (inf where alive)."""
    return np.where(alive, np.inf, 0.0)


def certify(r, sub=8, lip_rhs=None, rise_cap=None):
    """Honest J without printing: interpolate to a sub x finer time grid,
    REPAIR feasibility by re-solving the (convex) weight LPs there with
    positions frozen, then verify every constraint on a dense grid.  If the
    solution carries lifetime masks (alive_f/alive_g), the fine-grid repair
    respects those windows so measured J reflects the imposed lifetimes;
    without masks it re-solves the weights freely (original behaviour).

    Run-13 cell (§5A): `lip_rhs` (Np1, 2) and/or `rise_cap` (xs, rho), if given,
    are re-applied on the fine grid so the repaired J reflects the injected
    environment (without them the repair re-solves the LPs freely and washes the
    injection out -- meaningless certified J for a non-flat cell). `lip_rhs` is
    interpolated onto the fine grid like the weights; `rise_cap` is x-space and
    passes through unchanged. Default None = original behaviour, unchanged.
    Returns dict(J_interp, Jc, rep)."""
    Af, XIf, Bf, ETAf, tf = refine_time(r["A"], r["XI"], r["B"], r["ETA"],
                                        r["t"], sub=sub)
    J_interp = total_J(Af, XIf, Bf, ETAf)
    af, ag = r.get("alive_f"), r.get("alive_g")
    ub_f = _ub(_refine_mask(af, r["t"], tf)) if af is not None else None
    ub_g = _ub(_refine_mask(ag, r["t"], tf)) if ag is not None else None
    lipf = _interp_to_grid(lip_rhs, r["t"], tf) if lip_rhs is not None else None
    # repair: weights re-optimized on the fine grid (positions fixed) --
    # restores exact feasibility; only possible because the weight blocks
    # are LPs.  Injected environment (lipf/rise_cap) re-applied so J reflects it.
    Af = lp_weights_f(XIf, Bf, ETAf, ub=ub_f, lip_rhs=lipf, rise_cap=rise_cap)
    Bf = lp_weights_g(Af, XIf, ETAf, ub=ub_g, lip_rhs=lipf)
    Jc = total_J(Af, XIf, Bf, ETAf)
    rep = verify_dense(Af, XIf, Bf, ETAf, tf)
    return dict(J_interp=J_interp, Jc=Jc, rep=rep)


def report(tag, r, sub=8):
    """Print certify()'s honest J. The number printed as J_certified is
    achieved by a fully feasible pair (window-aware when masks are present)."""
    c = certify(r, sub=sub)
    seed_tag = f"   rng_seed = {r['rng_seed']}" if "rng_seed" in r else ""
    print(f"  [{tag}]  J_coarse = {r['J']:.5f}   J_interp(x{sub}) = "
          f"{c['J_interp']:.5f}   J_certified = {c['Jc']:.5f}"
          f"   dense_check = {c['rep']['J (dense cross-check)']:.5f}"
          f"   constraints_ok = {c['rep']['ALL CONSTRAINTS OK']}{seed_tag}")
    return c["Jc"]

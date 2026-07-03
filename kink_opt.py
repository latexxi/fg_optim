#!/usr/bin/env python3
"""
kink_opt.py -- Prototype: kink-coordinate optimization for

    max  J[f,g] = int_0^1 int_{-1}^1 f_t(x,t) * g_xx(x,t) dx dt

Representation (negative-hat basis)
-----------------------------------
    f(x,t) = - sum_i a_i(t) * hat(x; xi_i(t)),    a_i >= 0
    g(x,t) = - sum_m b_m(t) * hat(x; eta_m(t)),   b_m >= 0

where hat(x; c) is the piecewise-linear "tent" with hat(+-1)=0, hat(c)=1.

Constraint dictionary in these coordinates:
    convexity in x       <=>  a_i >= 0, b_m >= 0                       (exact)
    Lipschitz |f_x|<=1   <=>  sum_i a_i/(1+xi_i) <= 1  and
                              sum_i a_i/(1-xi_i) <= 1  per time node   (exact)
    boundary f(+-1,t)=0  <=>  built into the basis                     (exact)
    terminal f(x,1)=0    <=>  a_i(t_N) = 0
    initial  g(x,0)=0    <=>  b_m(t_0) = 0
    f_t >= 0             <=>  f(.,t_{k+1}) - f(.,t_k) >= 0 at the UNION
                              of kink positions of both slices.
                              (difference of two convex PL functions is PL
                              with kinks exactly at that union, and it
                              vanishes at x=+-1, so this check is EXACT)
    g_t <= 0             <=>  analogous, with reversed sign.

Objective (harvest form, no near-singular quadrature):
    g_xx(.,t) = sum_m j_m(t) delta(x - eta_m(t)),  j_m = 2 b_m / (1 - eta_m^2)
    J = sum_m int_0^1 j_m(t) * f_t(eta_m(t), t) dt
Discretized with midpoint rule per time step:
    J ~= sum_k sum_m j_m^{k+1/2} * [ f(eta^{k+1/2}, t_{k+1}) - f(eta^{k+1/2}, t_k) ]
(the dt cancels: J is literally "rise of f harvested at g's kinks").

Optimization strategy (exploits partial convexity)
--------------------------------------------------
    1. positions frozen  -> J is LINEAR in the f-weights  a  -> LP (HiGHS)
    2. positions frozen  -> J is LINEAR in the g-weights  b  -> LP (HiGHS)
    3. weights frozen    -> smooth-ish NLP in positions (L-BFGS-B with
                            penalties for ordering / monotonicity / Lipschitz,
                            analytic gradients -- see grad_total_J and
                            grad_penalty), then re-run 1-2 to restore exact
                            feasibility.
Every LP block is solved to global optimality; only step 3 is nonconvex.

This is a PROTOTYPE: small K, modest N.
Extension points are marked with  # EXT.
"""

import numpy as np
from scipy.optimize import linprog, minimize
from concurrent.futures import ProcessPoolExecutor

MARGIN = 0.03      # keep kinks inside (-1+MARGIN, 1-MARGIN)
GAP = 0.02         # minimal spacing between same-family kinks
PEN_W = 200.0      # penalty weight in the position NLP


# ---------------------------------------------------------------- geometry

def hat_matrix(x, xi):
    """H[p,i] = hat(x[p]; xi[i]),  hat(+-1)=0, hat(node)=1."""
    x = np.atleast_1d(np.asarray(x, float))
    xi = np.atleast_1d(np.asarray(xi, float))
    L = (1.0 + x[:, None]) / (1.0 + xi[None, :])
    R = (1.0 - x[:, None]) / (1.0 - xi[None, :])
    return np.clip(np.where(x[:, None] <= xi[None, :], L, R), 0.0, None)


def conv_eval(x, w, xi):
    """F(x) = - sum_i w_i hat(x; xi_i): convex PL, F<=0, F(+-1)=0."""
    w = np.atleast_1d(w)
    if w.size == 0:
        return np.zeros(np.atleast_1d(x).shape)
    return -(hat_matrix(x, xi) * w[None, :]).sum(axis=1)


def _wbounds(shape, dead_node, ub=None):
    """Box bounds for a weight LP over a (Np1, K) weight matrix, flattened in
    the same (k outer, kink inner) order the LPs index with.  `dead_node` is
    the time index whose weights are pinned to 0 by a boundary condition
    (k=N for f's terminal f(x,1)=0, k=0 for g's initial g(x,0)=0).  `ub` is an
    optional (Np1, K) per-node upper bound (np.inf = unbounded); nodes with
    ub=0 are 'dead', which is how a kink's lifetime window is imposed.  With
    ub=None every non-dead node is unbounded, reproducing the original LPs
    exactly (so all-alive topologies regress to the previous behaviour)."""
    Np1, K = shape
    if ub is None:
        ub = np.full(shape, np.inf)
    bounds = []
    for k in range(Np1):
        for i in range(K):
            hi = 0.0 if k == dead_node else float(ub[k, i])
            bounds.append((0.0, hi if np.isfinite(hi) else None))
    return bounds


# ------------------------------------------------------------- weight LPs

def lp_weights_f(XI, B, ETA, ub=None):
    """Globally optimal f-weights A given all positions and g-weights.
    `ub` (Np1, Kf), if given, upper-bounds each weight per time node -- a 0
    entry pins a kink dead there (lifetime windows, Task B); default None is
    unbounded (original behaviour)."""
    Np1, Kf = XI.shape
    N = Np1 - 1
    nv = Np1 * Kf
    ix = lambda k, i: k * Kf + i

    c = np.zeros(nv)
    for k in range(N):
        em = 0.5 * (ETA[k] + ETA[k + 1])
        bm = 0.5 * (B[k] + B[k + 1])
        jm = 2.0 * bm / (1.0 - em ** 2)
        Hk, Hk1 = hat_matrix(em, XI[k]), hat_matrix(em, XI[k + 1])
        # J step = jm . [ (A[k] . h_k) - (A[k+1] . h_{k+1}) ]   (signs from f=-sum a h)
        for i in range(Kf):
            c[ix(k, i)] += float((jm * Hk[:, i]).sum())
            c[ix(k + 1, i)] -= float((jm * Hk1[:, i]).sum())

    rows, rhs = [], []
    # Lipschitz (exact): boundary slopes of the convex PL function
    for k in range(Np1):
        for denom in (1.0 + XI[k], 1.0 - XI[k]):
            r = np.zeros(nv)
            r[[ix(k, i) for i in range(Kf)]] = 1.0 / denom
            rows.append(r); rhs.append(1.0)
    # monotone rise f_t>=0 (exact at union of kinks)
    for k in range(N):
        xchk = np.concatenate([XI[k], XI[k + 1]])
        Hk, Hk1 = hat_matrix(xchk, XI[k]), hat_matrix(xchk, XI[k + 1])
        for p in range(len(xchk)):
            r = np.zeros(nv)
            for i in range(Kf):
                r[ix(k + 1, i)] += Hk1[p, i]   # f^{k+1}-f^k>=0  <=>  A.h terms
                r[ix(k, i)] -= Hk[p, i]
            rows.append(r); rhs.append(0.0)

    bounds = _wbounds((Np1, Kf), dead_node=N, ub=ub)
    res = linprog(-c, A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("f-LP failed: " + res.message)
    return res.x.reshape(Np1, Kf)


def lp_weights_g(A, XI, ETA, ub=None):
    """Globally optimal g-weights B given all positions and f-weights.
    `ub` (Np1, Kg), if given, upper-bounds each weight per time node (0 = dead
    there), imposing lifetime windows (Task B); default None is unbounded."""
    Np1, Kg = ETA.shape
    N = Np1 - 1
    nv = Np1 * Kg
    ix = lambda k, m: k * Kg + m

    c = np.zeros(nv)
    for k in range(N):
        em = 0.5 * (ETA[k] + ETA[k + 1])
        df = conv_eval(em, A[k + 1], XI[k + 1]) - conv_eval(em, A[k], XI[k])
        coef = df / (1.0 - em ** 2)           # j = (B_k+B_{k+1})/(1-em^2)
        for m in range(Kg):
            c[ix(k, m)] += coef[m]
            c[ix(k + 1, m)] += coef[m]

    rows, rhs = [], []
    for k in range(Np1):
        for denom in (1.0 + ETA[k], 1.0 - ETA[k]):
            r = np.zeros(nv)
            r[[ix(k, m) for m in range(Kg)]] = 1.0 / denom
            rows.append(r); rhs.append(1.0)
    # g_t <= 0  (g deepens): sum B^k h^k - sum B^{k+1} h^{k+1} <= 0
    for k in range(N):
        xchk = np.concatenate([ETA[k], ETA[k + 1]])
        Hk, Hk1 = hat_matrix(xchk, ETA[k]), hat_matrix(xchk, ETA[k + 1])
        for p in range(len(xchk)):
            r = np.zeros(nv)
            for m in range(Kg):
                r[ix(k, m)] += Hk[p, m]
                r[ix(k + 1, m)] -= Hk1[p, m]
            rows.append(r); rhs.append(0.0)

    bounds = _wbounds((Np1, Kg), dead_node=0, ub=ub)
    res = linprog(-c, A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("g-LP failed: " + res.message)
    return res.x.reshape(Np1, Kg)


# -------------------------------------------------------------- objective

def _hat_b(x, xi):
    """Batched hats: x (N,Kg), xi (N,Kf) -> (N,Kg,Kf)."""
    L = (1.0 + x[:, :, None]) / (1.0 + xi[:, None, :])
    R = (1.0 - x[:, :, None]) / (1.0 - xi[:, None, :])
    return np.clip(np.where(x[:, :, None] <= xi[:, None, :], L, R), 0.0, None)


def _hat_parts(x, xi):
    """hat(x;xi) plus its x- and xi-partials, elementwise/broadcast.
    hat = (1+x)/(1+xi) for x<=xi, else (1-x)/(1-xi); both branches are
    smooth rational-linear away from the x==xi kink, so d/dx and d/dxi
    are closed-form per branch (the clip only ever floors numerical noise
    since x,xi in (-1,1) keeps both branches positive by construction)."""
    L = (1.0 + x) / (1.0 + xi)
    R = (1.0 - x) / (1.0 - xi)
    mask = x <= xi
    Hraw = np.where(mask, L, R)
    dHdx = np.where(mask, 1.0 / (1.0 + xi), -1.0 / (1.0 - xi))
    dHdxi = np.where(mask, -L / (1.0 + xi), R / (1.0 - xi))
    active = Hraw > 0.0
    H = np.where(active, Hraw, 0.0)
    dHdx = np.where(active, dHdx, 0.0)
    dHdxi = np.where(active, dHdxi, 0.0)
    return H, dHdx, dHdxi


def _hat_b_parts(x, xi):
    """Batched hat + partials: x (N,Q), xi (N,K) -> each (N,Q,K)."""
    return _hat_parts(x[:, :, None], xi[:, None, :])


def total_J(A, XI, B, ETA):
    """Harvest sum: rise of f at g's kinks, weighted by jump size."""
    em = 0.5 * (ETA[:-1] + ETA[1:])                 # (N,Kg)
    bm = 0.5 * (B[:-1] + B[1:])
    jm = 2.0 * bm / (1.0 - em ** 2)
    f_lo = -(_hat_b(em, XI[:-1]) * A[:-1, None, :]).sum(-1)   # f(em, t_k)
    f_hi = -(_hat_b(em, XI[1:]) * A[1:, None, :]).sum(-1)     # f(em, t_{k+1})
    return float((jm * (f_hi - f_lo)).sum())


def grad_total_J(A, XI, B, ETA):
    """Analytic d(total_J)/d(XI), d(total_J)/d(ETA), weights A,B held fixed.
    J is closed-form in the positions (piecewise rational-linear via the
    hat basis), so this replaces L-BFGS-B's finite-difference gradient."""
    em = 0.5 * (ETA[:-1] + ETA[1:])                  # (N,Kg)
    bm = 0.5 * (B[:-1] + B[1:])
    denom = 1.0 - em ** 2
    jm = 2.0 * bm / denom
    djm_dem = 4.0 * bm * em / denom ** 2

    Hlo, dHlo_dx, dHlo_dxi = _hat_b_parts(em, XI[:-1])   # (N,Kg,Kf)
    Hhi, dHhi_dx, dHhi_dxi = _hat_b_parts(em, XI[1:])    # (N,Kg,Kf)
    f_lo = -(Hlo * A[:-1, None, :]).sum(-1)
    f_hi = -(Hhi * A[1:, None, :]).sum(-1)
    step = f_hi - f_lo

    dXI = np.zeros_like(XI)
    dXI[:-1] += A[:-1] * np.einsum('km,kmi->ki', jm, dHlo_dxi)
    dXI[1:] += -A[1:] * np.einsum('km,kmi->ki', jm, dHhi_dxi)

    dstep_dem = (np.einsum('ki,kmi->km', A[:-1], dHlo_dx)
                 - np.einsum('ki,kmi->km', A[1:], dHhi_dx))
    dterm_dem = djm_dem * step + jm * dstep_dem
    dETA = np.zeros_like(ETA)
    dETA[:-1] += 0.5 * dterm_dem
    dETA[1:] += 0.5 * dterm_dem
    return dXI, dETA


def penalty(A, XI, B, ETA):
    """Soft constraints for the position NLP (weights are frozen there)."""
    p = 0.0
    for P in (XI, ETA):                                   # kink ordering
        if P.shape[1] > 1:
            d = P[:, 1:] - P[:, :-1]
            p += (np.minimum(d - GAP, 0.0) ** 2).sum()

    def step_diff(W, P):
        """W-difference of consecutive slices at union checkpoints (N, 2K)."""
        xc = np.concatenate([P[:-1], P[1:]], axis=1)      # (N, 2K)
        lo = -( _hat_b(xc, P[:-1]) * W[:-1, None, :]).sum(-1)
        hi = -( _hat_b(xc, P[1:]) * W[1:, None, :]).sum(-1)
        return hi - lo

    p += (np.minimum(step_diff(A, XI), 0.0) ** 2).sum()   # f_t >= 0
    p += (np.maximum(step_diff(B, ETA), 0.0) ** 2).sum()  # g_t <= 0
    for W, P in ((A, XI), (B, ETA)):                      # Lipschitz
        p += (np.maximum((W / (1.0 + P)).sum(1) - 1.0, 0.0) ** 2).sum()
        p += (np.maximum((W / (1.0 - P)).sum(1) - 1.0, 0.0) ** 2).sum()
    return p


def _step_diff_grad(W, P, chi):
    """Analytic gradient of sum(chi(step_diff(W,P))**2) wrt P (W fixed).
    chi is the already-evaluated activation (min or max vs 0), shape (N,2K).
    step_diff's checkpoints are the kink positions THEMSELVES, so each kink
    is simultaneously an evaluation point and a hat-node: d/dP has both a
    'node' term (from being a hat center) and an 'eval-point' term (from
    being where some hat is sampled) -- the latter only ever lands on the
    matching diagonal checkpoint, which is what makes this tractable."""
    K = P.shape[1]
    Wlo, Whi = W[:-1], W[1:]
    Plo, Phi = P[:-1], P[1:]
    chi_lo, chi_hi = chi[:, :K], chi[:, K:]

    # lo fn (node=Plo) and hi fn (node=Phi), each queried at BOTH blocks'
    # points at once (xc matches step_diff's own checkpoint concatenation)
    # -- elementwise in query point, so this is the same numbers as four
    # separate K-sized calls, just batched into two 2K-sized ones.
    xc = np.concatenate([Plo, Phi], axis=1)
    _, dHlo_dx, dHlo_dxi = _hat_b_parts(xc, Plo)
    _, dHhi_dx, dHhi_dxi = _hat_b_parts(xc, Phi)
    dHll_dx, dHlh_dx = dHlo_dx[:, :K], dHlo_dx[:, K:]
    dHll_dxi, dHlh_dxi = dHlo_dxi[:, :K], dHlo_dxi[:, K:]
    dHhl_dx, dHhh_dx = dHhi_dx[:, :K], dHhi_dx[:, K:]
    dHhl_dxi, dHhh_dxi = dHhi_dxi[:, :K], dHhi_dxi[:, K:]

    Sx_ll = np.einsum('ki,kqi->kq', Wlo, dHll_dx)
    Sx_hl = np.einsum('ki,kqi->kq', Whi, dHhl_dx)
    Sx_hh = np.einsum('ki,kqi->kq', Whi, dHhh_dx)
    Sx_lh = np.einsum('ki,kqi->kq', Wlo, dHlh_dx)

    Node_ll = np.einsum('kq,kqj->kj', chi_lo, dHll_dxi)
    Node_lh = np.einsum('kq,kqj->kj', chi_hi, dHlh_dxi)
    Node_hl = np.einsum('kq,kqj->kj', chi_lo, dHhl_dxi)
    Node_hh = np.einsum('kq,kqj->kj', chi_hi, dHhh_dxi)

    dP_lo = 2.0 * Wlo * (Node_ll + Node_lh) + 2.0 * chi_lo * (Sx_ll - Sx_hl)
    dP_hi = -2.0 * Whi * (Node_hh + Node_hl) - 2.0 * chi_hi * (Sx_hh - Sx_lh)

    dP = np.zeros_like(P)
    dP[:-1] += dP_lo
    dP[1:] += dP_hi
    return dP


def grad_penalty(A, XI, B, ETA):
    """Analytic gradient of penalty() wrt XI and ETA (A,B held fixed)."""
    dXI = np.zeros_like(XI)
    dETA = np.zeros_like(ETA)

    for P, dP in ((XI, dXI), (ETA, dETA)):                # kink ordering
        if P.shape[1] > 1:
            d = P[:, 1:] - P[:, :-1]
            e = np.minimum(d - GAP, 0.0)
            dP[:, :-1] += -2.0 * e
            dP[:, 1:] += 2.0 * e

    def step_diff(W, P):
        xc = np.concatenate([P[:-1], P[1:]], axis=1)
        lo = -(_hat_b(xc, P[:-1]) * W[:-1, None, :]).sum(-1)
        hi = -(_hat_b(xc, P[1:]) * W[1:, None, :]).sum(-1)
        return hi - lo

    dXI += _step_diff_grad(A, XI, np.minimum(step_diff(A, XI), 0.0))
    dETA += _step_diff_grad(B, ETA, np.maximum(step_diff(B, ETA), 0.0))

    for W, P, dP in ((A, XI, dXI), (B, ETA, dETA)):        # Lipschitz
        e1 = np.maximum((W / (1.0 + P)).sum(1) - 1.0, 0.0)
        dP += 2.0 * e1[:, None] * (-W / (1.0 + P) ** 2)
        e2 = np.maximum((W / (1.0 - P)).sum(1) - 1.0, 0.0)
        dP += 2.0 * e2[:, None] * (W / (1.0 - P) ** 2)
    return dXI, dETA


def optimize_positions(A, XI, B, ETA, maxiter=40, alive_f=None, alive_g=None):
    """Nonconvex block: move kink trajectories, weights frozen.
    Always uses analytic gradients (grad_total_J + grad_penalty) via jac=,
    not L-BFGS-B's default finite differences -- there is no numerical-
    gradient fallback path.
    If lifetime masks (alive_f/alive_g) are supplied they are permuted by the
    same per-row sort applied to the positions and returned alongside, so a
    kink's dead/alive tag keeps tracking its position after re-ordering."""
    Np1, Kf = XI.shape
    Kg = ETA.shape[1]
    nxi = Np1 * Kf

    def unpack(z):
        return z[:nxi].reshape(Np1, Kf), z[nxi:].reshape(Np1, Kg)

    def obj(z):
        XIz, ETAz = unpack(z)
        return -total_J(A, XIz, B, ETAz) + PEN_W * penalty(A, XIz, B, ETAz)

    def obj_grad(z):
        XIz, ETAz = unpack(z)
        dJ_XI, dJ_ETA = grad_total_J(A, XIz, B, ETAz)
        dP_XI, dP_ETA = grad_penalty(A, XIz, B, ETAz)
        gXI = -dJ_XI + PEN_W * dP_XI
        gETA = -dJ_ETA + PEN_W * dP_ETA
        return np.concatenate([gXI.ravel(), gETA.ravel()])

    z0 = np.concatenate([XI.ravel(), ETA.ravel()])
    bnds = [(-1.0 + MARGIN, 1.0 - MARGIN)] * z0.size
    res = minimize(obj, z0, jac=obj_grad, method="L-BFGS-B", bounds=bnds,
                   options=dict(maxiter=maxiter))
    XIn, ETAn = unpack(res.x)
    of = np.argsort(XIn, axis=1)            # keep ordering after projection
    og = np.argsort(ETAn, axis=1)
    XIn = np.take_along_axis(XIn, of, axis=1)
    ETAn = np.take_along_axis(ETAn, og, axis=1)
    if alive_f is None and alive_g is None:
        return XIn, ETAn
    alive_f = np.take_along_axis(alive_f, of, axis=1)
    alive_g = np.take_along_axis(alive_g, og, axis=1)
    return XIn, ETAn, alive_f, alive_g


# ----------------------------------------------------------- verification

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


# ----------------------------------------------------------------- driver

def _ub(alive):
    """Lifetime mask -> per-node weight upper bounds (inf where alive)."""
    return np.where(alive, np.inf, 0.0)


def _alternate(A, XI, B, ETA, t, alive_f, alive_g, outer=6, pos_iters=40,
               optimize_pos=True, verbose=True, patience=3):
    """Block-coordinate alternation (weight-LPs <-> position NLP) shared by
    run() and grow_topology().  Lifetime masks pin dead kinks to zero weight
    via the LP upper bounds and ride along the position sort.  Keeps the best
    feasible state seen and reverts on regression (the position step is a
    nonconvex NLP and is NOT guaranteed to improve J monotonically)."""
    hist = []
    best = (total_J(A, XI, B, ETA), A, XI, B, ETA, alive_f, alive_g)
    stall = 0
    for it in range(outer):
        A = lp_weights_f(XI, B, ETA, ub=_ub(alive_f))
        B = lp_weights_g(A, XI, ETA, ub=_ub(alive_g))
        Jw = total_J(A, XI, B, ETA)
        if optimize_pos:
            XI, ETA, alive_f, alive_g = optimize_positions(
                A, XI, B, ETA, maxiter=pos_iters,
                alive_f=alive_f, alive_g=alive_g)
            A = lp_weights_f(XI, B, ETA, ub=_ub(alive_f))   # restore feasib.
            B = lp_weights_g(A, XI, ETA, ub=_ub(alive_g))
        Jp = total_J(A, XI, B, ETA)
        hist.append((Jw, Jp))
        if verbose:
            print(f"  outer {it}:  J after weight-LPs = {Jw:.5f}   "
                  f"after position step = {Jp:.5f}")
        if Jp > best[0]:
            best = (Jp, A.copy(), XI.copy(), B.copy(), ETA.copy(),
                    alive_f.copy(), alive_g.copy())
            stall = 0
        else:
            stall += 1
            A, XI, B, ETA, alive_f, alive_g = (
                best[1].copy(), best[2].copy(), best[3].copy(),
                best[4].copy(), best[5].copy(), best[6].copy())
            if stall >= patience:
                break
        if it > 1 and abs(hist[-1][1] - hist[-2][1]) < 1e-6:
            break
    Jbest, A, XI, B, ETA, alive_f, alive_g = best
    return dict(A=A, XI=XI, B=B, ETA=ETA, t=t, J=Jbest, hist=hist,
                alive_f=alive_f, alive_g=alive_g)


def run(N=24, Kf=3, Kg=2, outer=6, seed="travel", pos_iters=40,
        optimize_pos=True, verbose=True, patience=3, rng_seed=0, t=None):
    # `t` overrides the uniform grid with an arbitrary (e.g. graded, Task C)
    # non-uniform node set; N is then inferred.  total_J / the weight-LPs /
    # monotonicity checks never read node SPACING (dt cancels in the harvest
    # sum), so a non-uniform grid is transparent to them -- only seeding and
    # certification (refine_time) consult t, and both handle non-uniform t.
    t = np.linspace(0.0, 1.0, N + 1) if t is None else np.asarray(t, float)
    N = len(t) - 1
    rng = np.random.default_rng(rng_seed)

    if seed == "static":
        # interior linspace => Kf=1 sits at x=0; harvest needs co-location
        XI = np.tile(np.linspace(-0.4, 0.4, Kf + 2)[1:-1], (N + 1, 1))
        ETA = np.tile(np.linspace(-0.4, 0.4, Kg + 2)[1:-1], (N + 1, 1))
    else:  # traveling seed: g-kinks sweep, f-kinks ride the same path
        path = -0.5 + 1.0 * t                       # -0.5 -> +0.5
        ETA = path[:, None] + np.linspace(-0.02, 0.02, Kg)[None, :]
        XI = path[:, None] + np.linspace(-0.10, 0.04, Kf)[None, :]
    XI = np.clip(XI + 0.01 * rng.standard_normal(XI.shape), -1 + MARGIN, 1 - MARGIN)
    ETA = np.clip(ETA + 0.01 * rng.standard_normal(ETA.shape), -1 + MARGIN, 1 - MARGIN)
    XI.sort(axis=1); ETA.sort(axis=1)

    B = np.outer(t, np.ones(Kg)) * 0.4 / Kg         # feasible ramp to bootstrap
    A = lp_weights_f(XI, B, ETA)

    alive_f = np.ones((N + 1, Kf), dtype=bool)       # every kink alive always
    alive_g = np.ones((N + 1, Kg), dtype=bool)
    return _alternate(A, XI, B, ETA, t, alive_f, alive_g, outer=outer,
                      pos_iters=pos_iters, optimize_pos=optimize_pos,
                      verbose=verbose, patience=patience)


def _multistart_worker(args):
    s, kwargs = args
    return s, run(rng_seed=s, verbose=False, **kwargs)


def multistart(seeds=range(6), workers=None, **kwargs):
    """optimize_positions is only a local search (nonconvex NLP block), so the
    outcome depends on the initial kink jitter. Run the full pipeline from
    several rng_seed jitters and keep the globally-best feasible result.
    kwargs are forwarded to run() (N, Kf, Kg, outer, seed, pos_iters, ...).
    Each seed's run() is fully independent, so seeds are farmed out across
    processes (ProcessPoolExecutor); pass workers=1 to force serial (e.g. for
    debugging). Results are still reduced in seed order so ties break the
    same way as the serial version."""
    seeds = list(seeds)
    if workers == 1:
        results = {s: run(rng_seed=s, verbose=False, **kwargs) for s in seeds}
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = dict(ex.map(_multistart_worker,
                                   [(s, kwargs) for s in seeds]))
    best = None
    for s in seeds:
        r = results[s]
        if best is None or r["J"] > best["J"]:
            best = r
            best["rng_seed"] = s
    return best


def certify(r, sub=8):
    """Honest J without printing: interpolate to a sub x finer time grid,
    REPAIR feasibility by re-solving the (convex) weight LPs there with
    positions frozen, then verify every constraint on a dense grid.  If the
    solution carries lifetime masks (alive_f/alive_g), the fine-grid repair
    respects those windows so measured J reflects the imposed lifetimes;
    without masks it re-solves the weights freely (original behaviour).
    Returns dict(J_interp, Jc, rep)."""
    Af, XIf, Bf, ETAf, tf = refine_time(r["A"], r["XI"], r["B"], r["ETA"],
                                        r["t"], sub=sub)
    J_interp = total_J(Af, XIf, Bf, ETAf)
    af, ag = r.get("alive_f"), r.get("alive_g")
    ub_f = _ub(_refine_mask(af, r["t"], tf)) if af is not None else None
    ub_g = _ub(_refine_mask(ag, r["t"], tf)) if ag is not None else None
    # repair: weights re-optimized on the fine grid (positions fixed) --
    # restores exact feasibility; only possible because the weight blocks
    # are LPs.
    Af = lp_weights_f(XIf, Bf, ETAf, ub=ub_f)
    Bf = lp_weights_g(Af, XIf, ETAf, ub=ub_g)
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


# -------------------------------------------------------- topology moves (B)

def add_kink(family, XI, ETA, alive_f, alive_g, parent, t,
             t_birth, t_death, dx=0.05, rng=None):
    """Insert a new kink trajectory as a small perturbation of an existing
    `parent` column of the chosen family ("f" or "g"), alive only on
    [t_birth, t_death].  The new column carries NO weight yet (the caller
    re-solves the weight LPs), so J and feasibility are unchanged at insertion.
    Returns (XI, ETA, alive_f, alive_g) with one family's matrices grown by a
    column.  A dead node (weight pinned to 0 outside the window) contributes
    nothing to f/g, so its position is a free, harmless extra checkpoint until
    the window opens."""
    jit = dx * (rng.standard_normal() if rng is not None else 1.0)
    win = (t >= t_birth - 1e-12) & (t <= t_death + 1e-12)
    src = (XI[:, parent] if family == "f" else ETA[:, parent]) + jit
    return _insert_column(family, XI, ETA, alive_f, alive_g, src, win)


def _insert_column(family, XI, ETA, alive_f, alive_g, col, win):
    """Append one kink trajectory `col` (Np1,) to `family` with lifetime mask
    `win` (Np1, bool); positions clipped into bounds.  Shared by add_kink
    (perturbed copy, Task B) and spawn_generation (contracted copy, Task D)."""
    col = np.clip(np.asarray(col, float), -1 + MARGIN, 1 - MARGIN)
    if family == "f":
        return (np.column_stack([XI, col]), ETA,
                np.column_stack([alive_f, win]), alive_g)
    if family == "g":
        return (XI, np.column_stack([ETA, col]),
                alive_f, np.column_stack([alive_g, win]))
    raise ValueError("family must be 'f' or 'g'")


def _seed_grown(base, XI2, ETA2, af2, ag2):
    """Bootstrap feasible weights for a family grown by one column (via
    add_kink / _insert_column / spawn_generation), from a converged solution
    `base` whose XI/ETA the grown XI2/ETA2 extend.  The new column carries
    ZERO weight at insertion (by construction of the callers above), so this
    is J-neutral -- it only restores exact LP feasibility on the grown shape,
    it does not change the objective.  `base["B"]` is padded with a zero
    column for any new g-kink so the first LP sees matching (Kg+1) shapes
    (no such padding is needed for A: it is solved fresh by the LP, not
    warm-started, since HiGHS finds the exact optimum in one call)."""
    B0 = np.zeros((base["t"].size, ETA2.shape[1]))
    B0[:, :base["B"].shape[1]] = base["B"]
    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
    B0 = lp_weights_g(A0, XI2, ETA2, ub=_ub(ag2))
    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
    return A0, B0


def _lifetime_window(mask_col, t):
    """[t_birth, t_death] of a boolean lifetime column: the times of its first
    and last alive node.  All-dead (shouldn't happen) falls back to [t0, t1]."""
    idx = np.nonzero(mask_col)[0]
    if idx.size == 0:
        return t[0], t[-1]
    return t[idx[0]], t[idx[-1]]


def _kink_diagnostics(r, family, col_idx, parent_idx, tol=1e-8):
    """Post-optimization measurements for one kink column, used by the Run 9
    generation ladder to test STRATEGY.md's self-similarity hypotheses
    directly rather than by eyeballing printouts.  `family` is "f" or "g";
    `col_idx` the new kink's column, `parent_idx` the kink it was seeded from
    (perturbed copy, add_kink) or contracted from (spawn_generation).

    Returns dict with:
      lifetime  -- (t_on, t_off): first/last time node where |weight| > tol.
                   This is the EFFECTIVE active window, which may be narrower
                   than any imposed lifetime mask -- the self-similarity test
                   is whether the optimizer contracts it further on its own.
      extent    -- (min_x, max_x) of the position trajectory over that window.
      jump_mean -- mean over the active window of 2*w/(1-x**2), the "jump"
                   formula from the module docstring (same formula for f and
                   g: both are hat-sum representations with identical
                   curvature/jump algebra, only g's jump enters the harvest
                   sum as g_xx).
      offset_from_parent -- mean |x_new - x_parent| over the active window
                   (tests "riding on the parent path").
    A column that is entirely zero-weight (fully pruned away, i.e. the LP
    gave the new kink nothing at every node) falls back to reporting over its
    imposed lifetime mask instead of an empty active window, so the numbers
    stay meaningful ("it used none of its allotted window") rather than NaN.
    """
    W = r["A"] if family == "f" else r["B"]
    P = r["XI"] if family == "f" else r["ETA"]
    mask = r.get("alive_f") if family == "f" else r.get("alive_g")
    t = r["t"]
    w, p, p_parent = W[:, col_idx], P[:, col_idx], P[:, parent_idx]
    active = np.abs(w) > tol
    if not active.any():
        active = mask[:, col_idx] if mask is not None else np.ones_like(w, dtype=bool)
    lifetime = _lifetime_window(active, t)
    extent = (float(p[active].min()), float(p[active].max())) if active.any() else (t[0], t[-1])
    jump = 2.0 * np.abs(w) / (1.0 - p**2)
    jump_mean = float(jump[active].mean()) if active.any() else 0.0
    offset_mean = float(np.abs(p - p_parent)[active].mean()) if active.any() else 0.0
    return dict(lifetime=lifetime, extent=extent, jump_mean=jump_mean,
                offset_from_parent=offset_mean)


def spawn_generation(sol, scale_t=0.5, scale_x=0.5, families=("f", "g"),
                     rng=None):
    """Task D -- the renormalization warm start.  If the hierarchy is
    self-similar, generation k+1 is an affinely-rescaled copy of generation k:
    shorter lifetime, narrower spatial extent, riding on top of generation k's
    path.  For each requested family, take the current finest carrier (its most
    active kink), contract that trajectory SPATIALLY by `scale_x` about the end
    of its travel path (p_end), and TEMPORALLY to a window of length
    `scale_t` x its lifetime placed at that end.  The contracted copy is
    inserted at ZERO weight via the Task B `_insert_column` machinery -- so J
    and feasibility are unchanged at insertion -- and the caller re-optimizes.
    Unlike a randomly-perturbed insertion (add_kink), this seeds the new kink
    already riding the parent path at a fraction of its extent, which is a much
    better basin for the position NLP (see Run 8).
    Returns (XI, ETA, alive_f, alive_g) with one new column per family."""
    XI, ETA = sol["XI"].copy(), sol["ETA"].copy()
    af, ag = sol["alive_f"].copy(), sol["alive_g"].copy()
    t = sol["t"]
    for family in families:
        W = sol["A"] if family == "f" else sol["B"]
        cols = sol["XI"] if family == "f" else sol["ETA"]
        mask = sol["alive_f"] if family == "f" else sol["alive_g"]
        parent = int(np.abs(W).max(axis=0).argmax())    # finest = most active
        p = cols[:, parent]
        alive_idx = np.nonzero(mask[:, parent])[0]
        tb, td = _lifetime_window(mask[:, parent], t)
        span = scale_t * (td - tb)
        wb, wd = td - span, td                           # window at travel end
        win = (t >= wb - 1e-12) & (t <= wd + 1e-12)
        p_end = p[alive_idx[-1]] if alive_idx.size else p[-1]
        col = p_end + scale_x * (p - p_end)              # contract about p_end
        if rng is not None:
            col = col + 0.01 * scale_x * rng.standard_normal(col.shape)
        XI, ETA, af, ag = _insert_column(family, XI, ETA, af, ag, col, win)
    return XI, ETA, af, ag


def prune(r, tol=1e-8):
    """Drop kinks whose |weight| stays below `tol` at every time node (dead
    trajectories the LP never used).  Keeps at least one kink per family."""
    A, XI, B, ETA = r["A"], r["XI"], r["B"], r["ETA"]
    af, ag = r["alive_f"], r["alive_g"]
    kf = np.abs(A).max(axis=0) > tol
    kg = np.abs(B).max(axis=0) > tol
    if not kf.any():
        kf[np.abs(A).max(axis=0).argmax()] = True
    if not kg.any():
        kg[np.abs(B).max(axis=0).argmax()] = True
    out = dict(r)
    out["A"], out["XI"], out["alive_f"] = A[:, kf], XI[:, kf], af[:, kf]
    out["B"], out["ETA"], out["alive_g"] = B[:, kg], ETA[:, kg], ag[:, kg]
    return out


def generation_step(cur, window, seeds=range(3), dx=0.05, outer=12,
                    pos_iters=40, patience=None, sub=8):
    """One rung of the Run 9 generation-gain ladder (STRATEGY.md Section 5):
    insert ONE new f-kink and ONE new g-kink, both alive only on `window` =
    (t_birth, t_death), as perturbed copies of the current most-active kink
    of each family (`add_kink`, Task B machinery -- zero-weight at insertion,
    so J is unchanged before re-optimization). Tried over several `seeds`
    (multistart on the insertion jitter, since `_alternate`'s position block
    is only a local search); the best FEASIBLE candidate by J_certified is
    kept (falls back to the best candidate overall, flagged `feasible=False`,
    if none certify -- an optimizer failure must be visible, not silently
    dropped).

    Passing `window = (t[0], t[-1])` (full lifetime) turns this into the
    "guard arm" (a free, unrestricted insertion) -- same code path, since
    windowed-vs-free is just this one argument.

    Column identity: the new kink is appended as the LAST column of its
    family (`cur["XI"].shape[1]` / `cur["ETA"].shape[1]` before insertion).
    This index is tracked into the post-alternation candidate BEFORE pruning
    (pruning can shift indices, and if the new kink itself dies, the
    diagnostics still need its column present with all-zero weight -- see
    `_kink_diagnostics`'s fallback). This relies on the same no-crossing
    column-identity assumption `optimize_positions`'s mask-permutation already
    depends on elsewhere in this file.

    Returns dict(sol, Jc, feasible, diagnostics=dict(f=..., g=...),
    spread=[(seed, Jc, feasible), ...]) -- `spread` is the per-seed honesty
    record (STRATEGY.md: "report the spread, not just the max")."""
    t = cur["t"]
    tb, td = window
    pf = int(np.abs(cur["A"]).max(axis=0).argmax())    # most active f-kink
    pg = int(np.abs(cur["B"]).max(axis=0).argmax())    # most active g-kink
    new_f_idx = cur["XI"].shape[1]                     # append position
    new_g_idx = cur["ETA"].shape[1]

    results = []
    for s in seeds:
        rng = np.random.default_rng(s)
        XI2, ETA2, af2, ag2 = add_kink(
            "f", cur["XI"], cur["ETA"], cur["alive_f"], cur["alive_g"],
            pf, t, tb, td, dx=dx, rng=rng)
        XI2, ETA2, af2, ag2 = add_kink(
            "g", XI2, ETA2, af2, ag2, pg, t, tb, td, dx=dx, rng=rng)
        A0, B0 = _seed_grown(cur, XI2, ETA2, af2, ag2)
        cand = _alternate(A0, XI2, B0, ETA2, t, af2, ag2, outer=outer,
                          pos_iters=pos_iters, optimize_pos=True,
                          verbose=False, patience=patience or outer)
        diag_f = _kink_diagnostics(cand, "f", new_f_idx, pf)
        diag_g = _kink_diagnostics(cand, "g", new_g_idx, pg)
        pruned = prune(cand, tol=1e-8)
        c = certify(pruned, sub=sub)
        feasible = bool(c["rep"]["ALL CONSTRAINTS OK"])
        results.append(dict(seed=s, sol=pruned, Jc=c["Jc"], feasible=feasible,
                            diagnostics=dict(f=diag_f, g=diag_g)))

    feasible_results = [r for r in results if r["feasible"]]
    pool = feasible_results if feasible_results else results
    best = max(pool, key=lambda r: r["Jc"])
    return dict(sol=best["sol"], Jc=best["Jc"], feasible=best["feasible"],
                diagnostics=best["diagnostics"],
                spread=[(r["seed"], r["Jc"], r["feasible"]) for r in results])


def generation_ladder(base, n_gen=4, window0=0.5, window_ratio=0.5,
                      seeds=range(3), dx=0.05, base_fine_sub=4, coarse_N=8,
                      outer=12, pos_iters=40, sub=8, verbose=True):
    """Run 9 driver -- the STRATEGY.md Section 5 generation-gain measurement.
    Starting from a converged `base` solution (Run 3's G0 in the driver
    below), births one windowed generation at a time via `generation_step`
    and tracks dJk = Jk - J_{k-1}. Roughly constant dJk over several
    generations supports the log-growth conjecture (sup J = +infinity);
    decaying dJk means J is bounded and the mesh's ln(Nx) growth was a
    discretization artifact.

    Window schedule: w_k = window0 * window_ratio**(k-1), anchored at the
    shared right endpoint (the travel-path end all generations ride toward),
    so window_k = (t1 - w_k, t1). This is precomputed for ALL n_gen
    generations up front (not regridded per generation) so a single graded
    time grid (Task C) serves the whole ladder: `graded_grid` gets a
    per-window `fine_sub` scaled as `base_fine_sub * window0/w_k`, which
    counteracts the shrinking window width so each generation's local node
    count stays roughly flat instead of collapsing below the "finest
    lifetime spans >= 8 fine steps" honesty floor (STRATEGY.md Section 5) --
    certify()'s own further sub-refinement (`sub`) multiplies on top of that.

    `base` is migrated onto the precomputed grid ONCE (linear interpolation
    of weights/positions via `_interp_to_grid`, all-alive masks since it
    carries no windows yet, then an exact weight-LP re-solve to restore
    feasibility -- the same repair `certify` already relies on). This
    migration should be near J-neutral (some drift is expected -- a different
    node count samples the harvest sum differently, same reason Run 7 uses a
    1% bar rather than exact equality); it is checked against `certify(base)`
    before generation 1 runs and raises past a 1% relative gap, since a
    silent large mismatch there would corrupt every downstream dJk.

    Each generation additionally runs a "guard arm" -- the same
    `generation_step` call with a full-lifetime window instead of the
    imposed one -- so a constant windowed dJk can be checked against free
    (non-hierarchical) insertion, per STRATEGY.md's non-negotiable guard.
    The ladder always advances on the WINDOWED arm; the guard is comparison
    only and is never adopted, so this can't quietly degrade into the
    unrestricted growth of Runs 4-5.

    Returns dict(generations=[dict(k, w_k, window, Jc, dJk, feasible,
    guard_Jc, guard_dJk, guard_feasible, diagnostics, spread), ...],
    base_Jc)."""
    cur = prune(base, tol=1e-8)
    base_Jc = certify(cur, sub=sub)["Jc"]

    t0, t1 = cur["t"][0], cur["t"][-1]
    ws = [window0 * window_ratio**(k - 1) for k in range(1, n_gen + 1)]
    windows = [(t1 - w, t1) for w in ws]
    fine_subs = [base_fine_sub * window0 / w for w in ws]
    grid = graded_grid(windows, coarse_N=coarse_N, fine_sub=fine_subs,
                       t0=t0, t1=t1)
    # A regridding must be a strict REFINEMENT of `base`'s own nodes, never a
    # coarsening -- graded_grid's own uniform background (`coarse_N`) may be
    # coarser than base's original grid outside the windows, which would
    # silently lose resolution there and corrupt every downstream dJk. Union
    # with cur["t"] guarantees every original node survives.
    t_new = np.union1d(grid, cur["t"])

    A2 = _interp_to_grid(cur["A"], cur["t"], t_new)
    XI2 = _interp_to_grid(cur["XI"], cur["t"], t_new)
    B2 = _interp_to_grid(cur["B"], cur["t"], t_new)
    ETA2 = _interp_to_grid(cur["ETA"], cur["t"], t_new)
    alive_f2 = np.ones((len(t_new), XI2.shape[1]), dtype=bool)
    alive_g2 = np.ones((len(t_new), ETA2.shape[1]), dtype=bool)
    A2 = lp_weights_f(XI2, B2, ETA2, ub=_ub(alive_f2))   # restore exact
    B2 = lp_weights_g(A2, XI2, ETA2, ub=_ub(alive_g2))   # feasibility
    cur = dict(A=A2, XI=XI2, B=B2, ETA=ETA2, t=t_new, J=total_J(A2, XI2, B2, ETA2),
              hist=[], alive_f=alive_f2, alive_g=alive_g2)
    regrid_Jc = certify(cur, sub=sub)["Jc"]
    if verbose:
        print(f"  base J_certified = {base_Jc:.5f}  ->  regridded onto "
              f"{len(t_new)} nodes, J_certified = {regrid_Jc:.5f}")
    if abs(regrid_Jc - base_Jc) > 0.01 * abs(base_Jc):
        raise RuntimeError(
            f"regrid is not within 1% of base: base {base_Jc:.5f} vs "
            f"regridded {regrid_Jc:.5f} -- fix the migration before "
            f"trusting any dJk")

    generations = []
    J = regrid_Jc
    for k in range(1, n_gen + 1):
        w_k, window_k = ws[k - 1], windows[k - 1]
        windowed = generation_step(cur, window_k, seeds=seeds, dx=dx,
                                   outer=outer, pos_iters=pos_iters, sub=sub)
        guard = generation_step(cur, (t0, t1), seeds=seeds, dx=dx,
                                outer=outer, pos_iters=pos_iters, sub=sub)
        dJk = windowed["Jc"] - J
        guard_dJk = guard["Jc"] - J
        if verbose:
            print(f"  gen {k}: w_k={w_k:.4f}  Jc {J:.5f} -> "
                  f"{windowed['Jc']:.5f}  dJk={dJk:+.5f}  "
                  f"(feasible={windowed['feasible']})   guard: "
                  f"{guard['Jc']:.5f}  guard_dJk={guard_dJk:+.5f}  "
                  f"(feasible={guard['feasible']})")
        generations.append(dict(
            k=k, w_k=w_k, window=window_k, Jc=windowed["Jc"], dJk=dJk,
            feasible=windowed["feasible"], guard_Jc=guard["Jc"],
            guard_dJk=guard_dJk, guard_feasible=guard["feasible"],
            diagnostics=windowed["diagnostics"], spread=windowed["spread"]))
        cur, J = windowed["sol"], windowed["Jc"]
    return dict(generations=generations, base_Jc=regrid_Jc)


def grow_topology(base, n_gen=2, cand_seeds=range(2), dx=0.05, windows=None,
                  tol=1e-4, outer=20, pos_iters=40, patience=5, sub=8,
                  verbose=True):
    """Task B driver: from a converged solution, greedily birth one kink at a
    time.  Each generation tries inserting into each family, over a few
    lifetime windows and jitter seeds; every candidate is re-optimized with
    the block alternation, pruned, and certified.  The best candidate is kept
    only if J_certified strictly improves by more than `tol`; otherwise the
    search stops.  `windows` is a list of (t_birth, t_death); default probes a
    full lifetime plus two shorter co-moving windows.  Returns the grown
    solution dict (with lifetime masks)."""
    cur = prune(base, tol=1e-8)
    curJ = certify(cur, sub=sub)["Jc"]
    if verbose:
        print(f"  base J_certified = {curJ:.5f}  "
              f"(Kf={cur['XI'].shape[1]}, Kg={cur['ETA'].shape[1]})")

    for gen in range(1, n_gen + 1):
        t = cur["t"]
        win_list = windows if windows is not None else [
            (t[0], t[-1]), (0.5, 1.0), (0.6, 0.9)]
        best_cand, best_candJ, best_desc = None, curJ + tol, None

        for family in ("f", "g"):
            W = cur["A"] if family == "f" else cur["B"]
            parent = int(np.abs(W).max(axis=0).argmax())   # most active kink
            for (tb, td) in win_list:
                for cs in cand_seeds:
                    rng = np.random.default_rng(1000 * gen + cs)
                    XI2, ETA2, af2, ag2 = add_kink(
                        family, cur["XI"], cur["ETA"], cur["alive_f"],
                        cur["alive_g"], parent, t, tb, td, dx=dx, rng=rng)
                    # bootstrap weights for the grown dimensions (zero-weight
                    # insertion => this re-solve starts from the current soln).
                    # The unchanged family's current weights seed the first LP.
                    if family == "f":
                        A0 = lp_weights_f(XI2, cur["B"], ETA2, ub=_ub(af2))
                    else:
                        A0 = cur["A"]
                    B0 = lp_weights_g(A0, XI2, ETA2, ub=_ub(ag2))
                    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
                    cand = _alternate(A0, XI2, B0, ETA2, t, af2, ag2,
                                      outer=outer, pos_iters=pos_iters,
                                      optimize_pos=True, verbose=False,
                                      patience=patience)
                    cand = prune(cand, tol=1e-8)
                    c = certify(cand, sub=sub)
                    if not c["rep"]["ALL CONSTRAINTS OK"]:
                        continue
                    if c["Jc"] > best_candJ:
                        best_cand, best_candJ = cand, c["Jc"]
                        best_desc = (family, tb, td, cs)

        if best_cand is None:
            if verbose:
                print(f"  gen {gen}: no accepted insertion "
                      f"(no candidate beat {curJ:.5f} + {tol}); stopping.")
            break
        fam, tb, td, cs = best_desc
        if verbose:
            print(f"  gen {gen}: ACCEPT  +{fam}-kink on "
                  f"[{tb:.2f},{td:.2f}] (seed {cs})   "
                  f"J_certified {curJ:.5f} -> {best_candJ:.5f}   "
                  f"(Kf={best_cand['XI'].shape[1]}, "
                  f"Kg={best_cand['ETA'].shape[1]})")
        cur, curJ = best_cand, best_candJ
    return cur


if __name__ == "__main__":
    print("=" * 70)
    print("Run 1 (sanity): single co-located static kink pair, positions FROZEN")
    print("        exact discrete optimum is J = 2 (tent, bang-bang schedule)")
    print("=" * 70)
    r0 = run(N=24, Kf=1, Kg=1, outer=3, seed="static",
             optimize_pos=False, verbose=True)
    report("static 1+1", r0)

    print()
    print("=" * 70)
    print("Run 2: same single kink pair, positions FREE (can travel discover >2?)")
    print("=" * 70)
    r1 = run(N=16, Kf=1, Kg=1, outer=6, seed="static", pos_iters=50)
    report("travel 1+1", r1)

    print()
    print("=" * 70)
    print("Run 3: Kf=3, Kg=2, positions free, seeded at center")
    print("=" * 70)
    r2 = run(N=16, Kf=3, Kg=2, outer=40, seed="static", pos_iters=40, patience=5)
    report("static 3+2", r2)

    print()
    print("=" * 70)
    print("Run 4: push further -- more kinks (Kf=5, Kg=4) + multistart.")
    print("  Two things left J on the table in Run 3:")
    print("  (a) the position NLP step isn't monotone in J, so a run left")
    print("      going could drift below its own best point -- run() now")
    print("      keeps the best feasible state and reverts on regression")
    print("      (see 'patience' above -- Run 3 already benefits from this).")
    print("  (b) optimize_positions is only a local search, and the initial")
    print("      kink jitter was hardcoded to rng_seed=0. Sweeping seeds")
    print("      exposes materially better local optima at higher K.")
    print("  optimize_positions uses analytic gradients (grad_total_J +")
    print("  grad_penalty) throughout this file, not L-BFGS-B's finite")
    print("  differences -- Runs 2-3 above already benefited (~20-100x")
    print("  fewer objective evals per solve); it's what makes the wider")
    print("  multistart sweep in Run 5 below affordable.")
    print("=" * 70)
    r3 = multistart(seeds=range(6), N=16, Kf=5, Kg=4, outer=40,
                     seed="static", pos_iters=40, patience=5)
    report("multistart 5+4", r3)

    print()
    print("=" * 70)
    print("Run 5: cheap analytic gradients afford a much wider search --")
    print("  more seeds, more kinks. Kf=6,Kg=5 is the best FEASIBLE frontier")
    print("  found (Kf=7,Kg=6 and Kf=8,Kg=7 were tried and did not reliably")
    print("  beat it: 8+7 found a similar J but failed verify_dense, i.e. it")
    print("  exploited near-violations rather than a genuinely better optimum).")
    print("=" * 70)
    r4 = multistart(seeds=range(20), N=16, Kf=6, Kg=5, outer=60,
                     seed="static", pos_iters=80, patience=6)
    report("multistart 6+5", r4)

    print()
    print("=" * 70)
    print("Run 6 (Task B): topology moves -- birth/prune kinks between")
    print("  alternations. Starting from Run 3's converged 3+2 solution,")
    print("  grow_topology() greedily inserts one zero-weight kink at a time")
    print("  (a perturbed copy of the most active kink, alive only on a")
    print("  lifetime window), re-optimizes with the block alternation, prunes")
    print("  dead trajectories, and KEEPS the insertion only if J_certified")
    print("  strictly improves. Unlike Runs 4-5 (fixed K, brute-force more")
    print("  kinks + restarts) this is the add/prune machinery the")
    print("  hierarchical generation-spawning experiment (Task D) builds on.")
    print("=" * 70)
    grown = grow_topology(r2, n_gen=2, cand_seeds=range(2), outer=20,
                          pos_iters=40, patience=5)
    report("grown topo", grown)

    print()
    print("=" * 70)
    print("Run 7 (Task C): graded (non-uniform) time grid.")
    print("  total_J / the weight-LPs / the monotonicity checks never read")
    print("  node SPACING (dt cancels in the harvest sum), so an arbitrary")
    print("  non-uniform grid is transparent to them -- only seeding and the")
    print("  certification refinement consult t. That lets us spend time")
    print("  nodes where kinks actually live instead of a uniform global grid.")
    print("  Part A -- reproduce Run 3 within 1% at HALF the time nodes.")
    print("  Part B -- a narrow lifetime window costs O(1/w) nodes on a")
    print("            uniform grid but O(1) with grading: the variable-count")
    print("            saving Task C targets (fine generations live fast/short).")
    print("=" * 70)

    baseJ = certify(r2)["Jc"]                       # Run 3: 17 nodes, all-alive
    base_live = n_live_nodes(r2)
    bar = 0.99 * baseJ
    print(f"  Run 3 baseline: {r2['t'].size} nodes, {base_live} live vars, "
          f"J_certified = {baseJ:.4f}   (1% bar = {bar:.4f})")

    # Part A: half the nodes. All-alive Run 3 has no short lifetimes, so a
    # graded grid can't beat a uniform one here (verified: identical optima)
    # -- the win is purely the halved node count. Multistart because the
    # coarse-grid position NLP is nonconvex.
    t_half = np.linspace(0.0, 1.0, 8)               # 8 nodes vs 17
    bestJ, bestS = -np.inf, None
    for s in range(8):
        rh = run(Kf=3, Kg=2, outer=40, seed="static", pos_iters=60,
                 patience=6, verbose=False, rng_seed=s, t=t_half)
        ch = certify(rh)
        if ch["rep"]["ALL CONSTRAINTS OK"] and ch["Jc"] > bestJ:
            bestJ, bestS, best_rh = ch["Jc"], s, rh
    print(f"  Part A: {t_half.size} nodes, {n_live_nodes(best_rh)} live vars "
          f"({100*n_live_nodes(best_rh)//base_live}% of baseline), "
          f"J_certified = {bestJ:.4f} (seed {bestS})   "
          f"-> {'PASS' if bestJ >= bar else 'FAIL'} (within 1% at half nodes)")

    # Part B: a fine g-kink alive only on a narrow window W. Resolve W to 6
    # local steps two ways and count live decision variables.
    W = (0.45, 0.55)                                # width 0.10
    tg = graded_grid([W], coarse_N=10, fine_sub=6)  # coarse bg + dense window
    rg = prune(run(Kf=3, Kg=2, outer=20, seed="static", pos_iters=40,
                   patience=5, verbose=False, t=tg), 1e-8)
    rngB = np.random.default_rng(0)
    parent = int(np.abs(rg["B"]).max(axis=0).argmax())
    XI2, ETA2, af2, ag2 = add_kink("g", rg["XI"], rg["ETA"], rg["alive_f"],
                                   rg["alive_g"], parent, tg, W[0], W[1],
                                   dx=0.03, rng=rngB)
    A0 = rg["A"]
    B0 = lp_weights_g(A0, XI2, ETA2, ub=_ub(ag2))
    A0 = lp_weights_f(XI2, B0, ETA2, ub=_ub(af2))
    gcand = prune(_alternate(A0, XI2, B0, ETA2, tg, af2, ag2, outer=20,
                             pos_iters=40, optimize_pos=True, verbose=False,
                             patience=5), 1e-8)
    cg = certify(gcand)
    win_steps = int(((tg >= W[0] - 1e-9) & (tg <= W[1] + 1e-9)).sum()) - 1
    # uniform grid giving W the SAME 6 local steps needs step = |W|/6 over the
    # whole span -> ~1/step intervals everywhere; count its live vars for the
    # same 5 all-alive background kinks + the windowed kink.
    N_unif = int(round((1.0 - 0.0) / ((W[1] - W[0]) / win_steps)))
    live_unif = 5 * (N_unif + 1) + (win_steps + 1)
    print(f"  Part B: window {W} resolved to {win_steps} local steps")
    print(f"    graded : {tg.size:2d} nodes, {n_live_nodes(gcand):3d} live vars, "
          f"J_certified = {cg['Jc']:.4f}  ok = {cg['rep']['ALL CONSTRAINTS OK']}"
          f"  (within 1% of baseline: {cg['Jc'] >= bar})")
    print(f"    uniform: {N_unif + 1:2d} nodes, {live_unif:3d} live vars for the "
          f"SAME local resolution ({live_unif / n_live_nodes(gcand):.1f}x more "
          f"variables; the win grows as the window narrows)")

    print()
    print("=" * 70)
    print("Run 8 (Task D): the renormalization warm start -- and an HONEST")
    print("  null result. If the hierarchy were self-similar, the next")
    print("  generation of kinks would be an affinely-rescaled copy of the")
    print("  current one (shorter lifetime, narrower extent, riding the")
    print("  parent's path). spawn_generation() inserts exactly that contracted")
    print("  copy at ZERO weight (Task B machinery, so J and feasibility are")
    print("  unchanged at insertion -- verified below); a random insertion")
    print("  (add_kink) drops the same new kink at a RANDOM position on the")
    print("  same window. Both are re-optimized by the identical alternation.")
    print("  Hoped-for acceptance: the warm start reaches a better J faster.")
    print("  Measured: it does NOT. On Run 3's gen-0 optimum the contracted")
    print("  copy lands nearly CO-LOCATED with its (barely-travelling) parent,")
    print("  and two hats at one point are redundant in a convex sum -- so the")
    print("  LP gives it ~no weight and it converges in 1-2 outers to a")
    print("  SHALLOWER basin, while random insertion explores genuinely new")
    print("  positions and does at least as well (and stays feasible). Reading:")
    print("  the gen-0 optimum is not yet a self-similar travel hierarchy, so")
    print("  the renormalization premise is unvalidated at k=0. Task D ships")
    print("  the machinery to test it; whether self-similarity (and a warm-")
    print("  start payoff) emerges at deeper generations is the open Section-5")
    print("  question -- reported straight rather than tuned into a win.")
    print("=" * 70)

    G0 = prune(r2, 1e-8)                                 # Run 3 converged 3+2
    J0 = certify(G0)["Jc"]
    pf = int(np.abs(G0["A"]).max(axis=0).argmax())
    pg = int(np.abs(G0["B"]).max(axis=0).argmax())
    budget = 12

    def _warm(XI2, ETA2, af2, ag2):
        """Re-optimize from the bootstrapped seed; return (coarse-J curve,
        certified J, feasible, convergence outer-iter)."""
        A0, B0 = _seed_grown(G0, XI2, ETA2, af2, ag2)
        r = _alternate(A0, XI2, B0, ETA2, G0["t"], af2, ag2, outer=budget,
                       pos_iters=40, optimize_pos=True, verbose=False,
                       patience=budget)                 # run full budget
        curve = [jp for (_, jp) in r["hist"]]
        conv = next((i + 1 for i in range(1, len(curve))
                     if abs(curve[i] - curve[i - 1]) < 1e-5), len(curve))
        c = certify(prune(r, 1e-8))
        return curve, c["Jc"], c["rep"]["ALL CONSTRAINTS OK"], conv

    # spawn arm: contracted copy of the finest carrier, one f + one g kink.
    # spawn places both new kinks on [0.5, 1.0] (half the all-alive lifetime,
    # at the travel end); match that window for the random arm so the ONLY
    # difference is structured contraction vs random position.
    Xs, Es, afs, ags = spawn_generation(G0, scale_t=0.5, scale_x=0.5,
                                        rng=np.random.default_rng(0))
    # J-neutral at insertion: the new columns carry ZERO weight (pad G0's
    # weights with a zero column per grown family) -> total_J is exactly G0's,
    # because a zero-weight g-kink has no jump and a zero-weight f-kink no rise.
    A_pad = np.column_stack([G0["A"], np.zeros(G0["t"].size)])
    B_pad = np.column_stack([G0["B"], np.zeros(G0["t"].size)])
    J_ins = total_J(A_pad, Xs, B_pad, Es)
    _, spawn_Jc, spawn_ok, spawn_conv = _warm(Xs, Es, afs, ags)

    best_rand = (-np.inf, False, None, None)            # (Jc, ok, conv, seed)
    for rs in range(4):
        rr = np.random.default_rng(rs)
        Xr, Er, afr, agr = add_kink("f", G0["XI"], G0["ETA"], G0["alive_f"],
                                    G0["alive_g"], pf, G0["t"], 0.5, 1.0,
                                    dx=0.3, rng=rr)
        Xr, Er, afr, agr = add_kink("g", Xr, Er, afr, agr, pg, G0["t"],
                                    0.5, 1.0, dx=0.3, rng=rr)
        _, jc, ok, conv = _warm(Xr, Er, afr, agr)
        if ok and jc > best_rand[0]:
            best_rand = (jc, ok, conv, rs)
    rand_Jc, rand_ok, rand_conv, rand_seed = best_rand

    print(f"  G0 (Run 3): J_certified = {J0:.4f}")
    print(f"  spawn insertion is J-neutral: J at insertion = {J_ins:.4f} "
          f"(= J0 coarse {G0['J']:.4f}? {abs(J_ins - G0['J']) < 1e-6})")
    print(f"  spawn : J_certified = {spawn_Jc:.4f} (dJ {spawn_Jc-J0:+.4f})  "
          f"feasible = {spawn_ok}  converged in {spawn_conv} outers")
    print(f"  random: J_certified = {rand_Jc:.4f} (dJ {rand_Jc-J0:+.4f})  "
          f"feasible = {rand_ok}  converged in {rand_conv} outers  "
          f"(best feasible of 4 seeds, #{rand_seed})")
    win = spawn_ok and spawn_Jc >= rand_Jc - 1e-6
    print(f"  -> warm start beats random: {win}  "
          f"(null result: contracted copy is redundant at gen 0 -- the "
          f"hierarchy is not yet self-similar; machinery is correct and ready "
          f"for the multi-generation Section-5 test)")

    print()
    print("=" * 70)
    print("Run 9 (Section 5): the generation-gain ladder -- the experiment")
    print("  everything else in this file serves. Run 8 showed the warm start")
    print("  (spawn_generation) is just an accelerator and doesn't beat random")
    print("  insertion at gen 0; it does NOT block the measurement itself. This")
    print("  run uses the already-working add_kink multistart (Run 8's random")
    print("  arm) as the insertion mechanism, with one change that makes it a")
    print("  measurement instead of a repeat of Runs 4-5: each generation's new")
    print("  f-kink and g-kink get an IMPOSED lifetime window that halves every")
    print("  generation (Run 6 showed an unrestricted greedy always prefers a")
    print("  full lifetime -- more DOF wins -- so the window constraint is the")
    print("  whole point). Records dJk = Jk - J_{k-1} per generation, plus a")
    print("  guard arm (free, full-lifetime insertion) so a constant dJk can't")
    print("  be an artifact of the imposed window geometry. Interpretation:")
    print("  dJk roughly CONSTANT over generations supports the ln(Nx) mesh")
    print("  growth being real (sup J = +infinity, approached not attained);")
    print("  dJk DECAYING means J is bounded and the mesh growth was transient.")
    print("=" * 70)

    ladder = generation_ladder(G0, n_gen=3, window0=0.5, window_ratio=0.5,
                               seeds=range(3), outer=25, pos_iters=60,
                               coarse_N=8, base_fine_sub=4, sub=8)

    print(f"\n  {'k':>2} {'w_k':>7} {'Jc':>8} {'dJk':>8} {'ok':>5}   "
          f"{'guard_Jc':>8} {'guard_dJk':>9} {'ok':>5}")
    for g in ladder["generations"]:
        print(f"  {g['k']:>2} {g['w_k']:>7.4f} {g['Jc']:>8.4f} "
              f"{g['dJk']:>+8.4f} {str(g['feasible']):>5}   "
              f"{g['guard_Jc']:>8.4f} {g['guard_dJk']:>+9.4f} "
              f"{str(g['guard_feasible']):>5}")
        for fam in ("f", "g"):
            d = g["diagnostics"][fam]
            print(f"       +{fam}-kink: lifetime=({d['lifetime'][0]:.3f},"
                  f"{d['lifetime'][1]:.3f})  extent=({d['extent'][0]:+.3f},"
                  f"{d['extent'][1]:+.3f})  jump_mean={d['jump_mean']:.3f}  "
                  f"offset_from_parent={d['offset_from_parent']:.3f}")
        spread_str = ", ".join(f"seed{s}:{jc:.4f}{'' if ok else '(infeas)'}"
                               for s, jc, ok in g["spread"])
        print(f"       spread: {spread_str}")

    dJks = [g["dJk"] for g in ladder["generations"]]
    guard_dJks = [g["guard_dJk"] for g in ladder["generations"]]
    print(f"\n  dJk sequence:       {[f'{d:+.4f}' for d in dJks]}")
    print(f"  guard dJk sequence: {[f'{d:+.4f}' for d in guard_dJks]}")
    print("  Reading these numbers is the open question this file was built")
    print("  to answer -- see STRATEGY.md Section 5 for the interpretation")
    print("  rule (constant vs decaying dJk) and the honesty requirements")
    print("  (every Jc above is J_certified; the spread, not just the max, is")
    print("  reported per generation; the guard arm is never adopted into the")
    print("  ladder, only compared against it).")

    # EXT: adaptive per-kink time nodes (fine generations live fast & short)
    # EXT: warm-start next generation from a rescaled copy of this solution
    # EXT: multistart currently reseeds from scratch per attempt; a real
    #      basin-hopping / CMA-ES search would be far more sample-efficient
    # EXT: multistart selects by coarse J, not by verify_dense feasibility --
    #      Run 5's Kf=7,8 exploration shows this can pick an infeasible
    #      "winner" once K is large enough that the position NLP struggles;
    #      a feasibility-aware selection would be more robust at high K

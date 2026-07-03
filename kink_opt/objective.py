"""Harvest-form objective, analytic gradients, and the position NLP block."""

import numpy as np
from scipy.optimize import minimize

from .geometry import MARGIN, GAP, PEN_W


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

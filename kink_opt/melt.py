"""Run 12 (plans/run12-renormalization-cell.md,
plans/run12-implementation-details.md) -- the melt-band cell construction.
Extends Run 11's constructive hierarchy (`construct.py`) from a single
traveling kink per generation to a BAND of `K` kinks per family per
generation, riding a drift path of absolute length `L` that does NOT shrink
with depth (unlike Run 11's affine-contracted, collapsing-to-one-point
anchoring). Positions are still set analytically -- weights are still solved
by LP-only alternation, zero position NLP anywhere, same discipline as
Run 11.

Kept separate from `construct.py` rather than folding in: `construct.py` is
a single-kink-per-generation carrier (its module constants `XI_OFFSET`/
`ETA_OFFSET`/`_travel_path` encode that directly), and Run 11's own numbers
are already cited in STRATEGY.md/CLAUDE.md -- a second, structurally
different ansatz belongs in its own module. Gen-0 seeding still reuses
`construct._travel_path`/`XI_OFFSET`/`ETA_OFFSET` unchanged (imported, not
reimplemented) -- see `build_melt_hierarchy`.
"""

import numpy as np

from .geometry import MARGIN, conv_eval
from .lp import lp_weights_f, lp_weights_g
from .solver import _alternate
from .verify import certify, graded_grid, _ub
from .topology import _insert_column, _seed_grown
from .construct import (fit_geometric, grid_convergence_check,
                        _travel_path, XI_OFFSET, ETA_OFFSET)


def build_band(t, c, w, window, K=8, x_hat=None, dead=False):
    """One generation's band ansatz: K hat offsets `x_hat` (shared by f and
    g) spread across a band of half-width `w`, centered on path `c(t)`,
    alive only on `window`. Returns (col_f, col_g, win_mask), each
    (Np1, K)/(Np1,) -- NOT yet inserted into any XI/ETA (the caller does
    that via `topology._insert_column`, once per family, matching Run 11's
    convention).

    `c` is a callable t (array) -> x (array), already resolved to absolute
    coordinates (the caller is responsible for anchoring it -- see
    `_drift_path`). `x_hat` defaults to `np.linspace(-1, 1, K)`, i.e. hats
    evenly spread across the band's own [-w, w] extent in the co-moving
    frame; f and g use the SAME x_hat grid by default (the LP, not the
    ansatz, is what should decide whether f's realized curvature ends up
    narrower/taller than g's).

    `dead=True` forces `win_mask` all-False regardless of `window` -- used
    by `check_band_neutral` exactly as `build_hierarchy`'s `dead_gens`
    forces a generation's columns dead for the insertion-neutral gate."""
    x_hat = np.linspace(-1.0, 1.0, K) if x_hat is None else np.asarray(x_hat, float)
    c_t = np.asarray(c(t), float)
    cols = np.column_stack([
        np.clip(c_t + w * xh, -1 + MARGIN, 1 - MARGIN) for xh in x_hat])
    tb, td = window
    win_mask = (np.zeros(t.shape, dtype=bool) if dead else
               (t >= tb - 1e-12) & (t <= td + 1e-12))
    return cols, cols.copy(), win_mask


def _drift_path(c_anchor, L, window, arc="linear"):
    """Band-center path for ONE generation, in absolute x: a path of length
    `L` (order O(1), the SAME absolute number at every generation -- "L_k ~
    O(1): does NOT shrink" is implemented literally as one global constant
    `L`, not a per-k schedule) over `window`, anchored so the path passes
    through `c_anchor` at the window's temporal midpoint. Returns a
    callable path(t) -> x (matching `build_band`'s `c` contract).

    arc="linear": path(t) = c_anchor + L * (that/s - 0.5), that = t - t_birth,
      s = window width -- straight sweep through c_anchor.
    arc="arc": path(t) = c_anchor + L * (4*u*(1-u) - 0.5), u = that/s --
      raised-parabola there-and-back, peaking at u=0.5 (a coarse proxy for
      the mesh's observed there-and-back band drift; NOT the default).
    Outside `window`, holds at the nearer endpoint value (the position array
    must still be defined at every t; only `win_mask` controls liveness)."""
    tb, td = window
    s = td - tb

    def path(tt):
        tt = np.asarray(tt, float)
        that = np.clip(tt - tb, 0.0, s)
        u = that / s
        if arc == "linear":
            return c_anchor + L * (u - 0.5)
        if arc == "arc":
            return c_anchor + L * (4.0 * u * (1.0 - u) - 0.5)
        raise ValueError("arc must be 'linear' or 'arc'")
    return path


def build_melt_hierarchy(n_gen, lambda_w=0.5, lambda_s=0.5, L=0.3, K=8,
                         x_hat=None, drift="linear", t0=0.0, t1=1.0,
                         w0=0.5, s0=0.5, coarse_N=8, base_fine_sub=4,
                         outer=20, dead_gens=None):
    """Build generations 0..n_gen as a single solution dict (schema: A, XI,
    B, ETA, t, alive_f, alive_g, J, hist -- identical to build_hierarchy's,
    PLUS an extra `band_specs` key: the list of per-generation BandSpec
    dicts (k, c, w, window, K, x_hat, L, drift) for generations 1..n_gen,
    needed by `read_environment`/`band_travel_sanity` -- harmless extra key,
    same convention persist.py's loaders already use for dense field data).

    Generation 0: the coarse carrier alone (one traveling f/g kink pair,
    `construct._travel_path` + `XI_OFFSET`/`ETA_OFFSET`, full lifetime,
    reused unchanged from Run 11) -- 1 f-column + 1 g-column.

    **DEVIATION from the top-level plan's Section 6 sketch and this file's
    own original Section 2.1/2.2/3 draft**: those called for an additional
    "boundary stock" -- one static, full-lifetime hat at x=0 per family,
    alongside the carrier. Dropped. Reason (found empirically while running
    Stage 1, not assumed): a full-lifetime hat at x=0 has Lipschitz
    coefficients 1/(1+0)=1 and 1/(1-0)=1 -- the WORST (most budget-hungry)
    position possible on BOTH of a family's per-node constraints
    (sum_i a_i/(1+xi_i)<=1, sum_i a_i/(1-xi_i)<=1) simultaneously. Since J is
    only BILINEAR (not jointly convex) in (A, B) even with positions fully
    fixed -- each weight-LP block is exactly optimal given the OTHER
    family's current weights, but alternating the two blocks is coordinate
    ascent on a bilinear objective, which has multiple fixed points -- the
    alternation reliably converges to the corner where the boundary hat
    alone saturates that budget (reaching Run 1's ~2.0 single-co-located-hat
    optimum) and every other same-family column (carrier AND every band)
    gets driven to EXACTLY zero forever, regardless of K/w0/anchor: verified
    directly, `build_melt_hierarchy(n)` with the boundary stock present gave
    Jc=2.0 for every n=0..3, i.e. dJk=0 for every band -- precisely the
    Stage 1 "stop and revisit the design" condition this plan's own
    checkpoint calls for. This is a structural property of the shared
    per-node Lipschitz constraint, not an optimizer or seeding bug (a second
    seeding strategy was tried first -- see the incremental-construction
    note below -- and still hit this same corner). Confirmed the carrier-
    only gen 0 does not have this problem (Jc matches `construct.py`'s own
    gen-0 number, 0.17568, exactly) and bands then measurably contribute
    (Stage 1 checkpoint below).

    Generation k (1 <= k <= n_gen): a band with
      w_k = w0 * lambda_w**(k-1)                    (half-width, absolute x)
      s_k = s0 * lambda_s**(k-1)                     (window width, absolute t)
    Every generation's window is centered on `mid = (t0+t1)/2` -- anchoring
    window_k at the MIDPOINT of generation (k-1)'s own window collapses
    recursively to that single point for every k (generation 0's own
    "window" is (t0, t1), so its midpoint already is `mid`), so this is
    implemented directly as one constant rather than an explicit recursion.
    `c_anchor` (the drift path's spatial center) is likewise the coarse
    carrier's raw path value at `mid`, one constant reused for every k. `L`
    itself does NOT shrink with k (Section 1's "does not shrink" point,
    implemented literally), so unlike Run 11's anchoring this does not force
    deeper generations toward zero spatial extent -- only `w_k` (band width)
    and `s_k` (window duration, hence sweep speed) change with depth.

    Grid: `graded_grid` over the n_gen windows, `fine_sub_k = base_fine_sub *
    s_1/s_k` (s_1 = generation 1's window width, the widest) -- the same
    convention `generation_ladder`/`scale_sweep`/`build_hierarchy` use.

    Weights: `_alternate(..., optimize_pos=False)` only -- monotone convex
    fixed point, deterministic, no position NLP, identical discipline to
    `build_hierarchy`.

    `dead_gens` (set of generation indices in 1..n_gen): force those bands'
    columns all-dead before the weight solve -- same purpose as
    `build_hierarchy`'s `dead_gens`, used by `check_band_neutral`."""
    dead_gens = set(dead_gens or ())
    x_hat = np.linspace(-1.0, 1.0, K) if x_hat is None else np.asarray(x_hat, float)

    mid = 0.5 * (t0 + t1)
    ws = [w0 * lambda_w ** (k - 1) for k in range(1, n_gen + 1)]
    ss = [s0 * lambda_s ** (k - 1) for k in range(1, n_gen + 1)]
    windows = [(mid - s / 2.0, mid + s / 2.0) for s in ss]
    fine_subs = [base_fine_sub * ss[0] / s for s in ss] if ss else []
    t = (graded_grid(windows, coarse_N=coarse_N, fine_sub=fine_subs,
                     t0=t0, t1=t1) if windows
         else np.linspace(t0, t1, coarse_N + 1))

    c_anchor = float(_travel_path(np.array([mid]))[0])
    p = _travel_path(t)
    col_f_carrier = np.clip(p + XI_OFFSET, -1 + MARGIN, 1 - MARGIN)
    col_g_carrier = np.clip(p + ETA_OFFSET, -1 + MARGIN, 1 - MARGIN)

    XI = col_f_carrier[:, None]
    ETA = col_g_carrier[:, None]
    alive_f = np.ones_like(XI, dtype=bool)
    alive_g = np.ones_like(ETA, dtype=bool)

    # Weights are built up INCREMENTALLY (gen 0 alternated to real
    # convergence first, then each band inserted at zero weight into that
    # already-converged solution and re-alternated -- the same
    # add_kink/_seed_grown bootstrap pattern `generation_ladder` uses),
    # rather than seeding the WHOLE final (1 + K*n_gen)-column structure
    # from one flat ramp and alternating once at the end the way
    # `build_hierarchy` does. Reason: J is only BILINEAR (not jointly
    # convex) in (A, B) with positions fixed, so LP-only alternation is
    # coordinate ascent on a bilinear objective -- its fixed point depends
    # on where it starts. `build_hierarchy` gets away with a single flat
    # seed because it never has more than a handful of columns (Kf_total =
    # n_gen+1); here Kf_total = 1 + K*n_gen (K=8 by default) is an order of
    # magnitude larger, and a flat seed over that many columns is more
    # exposed to landing in a worse fixed point. Building up incrementally
    # starts each new LP-alternation from a genuinely nonzero, already-
    # optimal base, which is markedly more robust in practice (verified
    # against the flat-seed alternative while developing this function).
    B0 = np.outer(t, np.ones(1)) * 0.4
    A0 = lp_weights_f(XI, B0, ETA, ub=_ub(alive_f))
    B0 = lp_weights_g(A0, XI, ETA, ub=_ub(alive_g))
    cur = _alternate(A0, XI, B0, ETA, t, alive_f, alive_g, outer=outer,
                     optimize_pos=False, verbose=False, patience=outer)

    band_specs = []
    for k in range(1, n_gen + 1):
        w_k, window_k = ws[k - 1], windows[k - 1]
        c_k = _drift_path(c_anchor, L, window_k, arc=drift)
        col_f, col_g, win = build_band(t, c_k, w_k, window_k, K=K,
                                       x_hat=x_hat, dead=(k in dead_gens))
        XI2, ETA2 = cur["XI"], cur["ETA"]
        af2, ag2 = cur["alive_f"], cur["alive_g"]
        for i in range(K):
            XI2, ETA2, af2, ag2 = _insert_column(
                "f", XI2, ETA2, af2, ag2, col_f[:, i], win)
            XI2, ETA2, af2, ag2 = _insert_column(
                "g", XI2, ETA2, af2, ag2, col_g[:, i], win)
        A0, B0 = _seed_grown(cur, XI2, ETA2, af2, ag2)
        cur = _alternate(A0, XI2, B0, ETA2, t, af2, ag2, outer=outer,
                         optimize_pos=False, verbose=False, patience=outer)
        band_specs.append(dict(k=k, c=c_k, w=w_k, window=window_k, K=K,
                               x_hat=x_hat, L=L, drift=drift))

    cur["band_specs"] = band_specs
    return cur


def melt_ladder(n_gen, lambda_w=0.5, lambda_s=0.5, L=0.3, K=8, sub=8,
               **kwargs):
    """Run 12 driver, direct analogue of `constructive_ladder`: for
    k=0..n_gen, `build_melt_hierarchy(k, ...)` then `certify()`. Records
    dJk = Jc_k - Jc_{k-1} and ratio = dJk/dJ_{k-1}.

    Returns dict(generations=[dict(k, Jc, constraints_ok, dJk, ratio,
    env, sol), ...], lambda_w, lambda_s, L, K) -- `env` is
    `read_environment`'s output for generation k's own band, read off the
    SAME certified sol (None for k=0, which has no band)."""
    generations = []
    Jprev, dJprev = None, None
    for k in range(0, n_gen + 1):
        sol = build_melt_hierarchy(k, lambda_w=lambda_w, lambda_s=lambda_s,
                                   L=L, K=K, **kwargs)
        c = certify(sol, sub=sub)
        Jc = c["Jc"]
        ok = bool(c["rep"]["ALL CONSTRAINTS OK"])
        dJk = None if Jprev is None else Jc - Jprev
        ratio = (dJk / dJprev if dJk is not None and dJprev not in (None, 0)
                else None)
        env = (read_environment(sol, k, sol["band_specs"][k - 1])
              if k >= 1 else None)
        generations.append(dict(k=k, Jc=Jc, constraints_ok=ok, dJk=dJk,
                                ratio=ratio, env=env, sol=sol))
        Jprev = Jc
        if dJk is not None:
            dJprev = dJk
    return dict(generations=generations, lambda_w=lambda_w, lambda_s=lambda_s,
               L=L, K=K)


def melt_sweep(n_gen, lambda_ws, lambda_ss, L=0.3, K=8, sub=8, **kwargs):
    """Sweep `melt_ladder` over a (lambda_w, lambda_s) grid -- direct
    analogue of `sweep_ratios`. Returns list of dict(lambda_w, lambda_s,
    dJk=[...], ratios=[...], ladder=<melt_ladder result>)."""
    out = []
    for lw in lambda_ws:
        for ls in lambda_ss:
            lad = melt_ladder(n_gen, lambda_w=lw, lambda_s=ls, L=L, K=K,
                              sub=sub, **kwargs)
            dJk = [g["dJk"] for g in lad["generations"] if g["dJk"] is not None]
            ratios = [g["ratio"] for g in lad["generations"]
                     if g["ratio"] is not None]
            out.append(dict(lambda_w=lw, lambda_s=ls, dJk=dJk, ratios=ratios,
                            ladder=lad))
    return out


def _slope_f(x_query, a, xi):
    """Exact piecewise-constant slope of f(x) = -sum_i a_i hat(x; xi_i) at
    each point in `x_query`, from the closed-form breakpoint formula (no
    finite differencing): on the interval between the j-th and (j+1)-th
    SORTED kink, slope = sum_{i<=j} a_i/(1-xi_i) - sum_{i>j} a_i/(1+xi_i)."""
    order = np.argsort(xi)
    xi_s, a_s = xi[order], a[order]
    left_term = a_s / (1.0 - xi_s)
    right_term = a_s / (1.0 + xi_s)
    cum_left = np.concatenate([[0.0], np.cumsum(left_term)])
    cum_right = np.concatenate([[0.0], np.cumsum(right_term)])
    right_total = right_term.sum()
    j = np.searchsorted(xi_s, x_query, side="right")
    return cum_left[j] - (right_total - cum_right[j])


def read_environment(sol, k, band_spec, n_sample=17, family="f"):
    """Read the slope-slack profile beta(x_hat) and remaining-rise profile
    rho(x_hat) that generation k's OWN band would see, sampled at `n_sample`
    points spanning band_spec['x_hat'] (i.e. in k's own co-moving frame),
    evaluated at t_read = temporal midpoint of band_spec['window'] (nearest
    existing grid node -- no time-interpolation, since `graded_grid`
    densifies exactly inside each window so an exact node is available
    nearby by construction), from the ALREADY-SOLVED (A, XI) or (B, ETA) of
    `sol` (a hierarchy certified through some generation <= k; typically
    generation k-1's solve for the INCOMING environment, or generation k's
    own solve for the OUTGOING one -- the caller decides which `sol` to
    pass, this function only reads).

    beta(x) = 1 - |slope of f(., t_read) at x|   (exact piecewise-constant
      formula, `_slope_f` -- no finite differencing)
    rho(x)  = (1 - |x|) + f(x, t_read)            (remaining Lipschitz rise
      budget at x, same formula `saturation_diagnostics` already uses for
      rise_budget_used = -f(x_birth, 0), generalized to any t)

    `family` selects which function's profile to read (f for beta/rho as
    defined; g's dual profile, if ever needed, is a symmetric call with
    family="g", not built by default).

    Returns dict(x=<absolute x sample points>, x_hat=<co-moving samples>,
    t_read=float, beta=np.ndarray, rho=np.ndarray)."""
    if band_spec["k"] != k:
        raise ValueError(f"band_spec is for generation {band_spec['k']}, not {k}")

    t = sol["t"]
    tb, td = band_spec["window"]
    idx = int(np.argmin(np.abs(t - 0.5 * (tb + td))))
    t_read = float(t[idx])

    x_hat = band_spec["x_hat"]
    x_hat_sample = np.linspace(x_hat.min(), x_hat.max(), n_sample)
    c_val = float(np.asarray(band_spec["c"](np.array([t_read])))[0])
    x_sample = np.clip(c_val + band_spec["w"] * x_hat_sample,
                       -1 + MARGIN, 1 - MARGIN)

    W = sol["A"] if family == "f" else sol["B"]
    P = sol["XI"] if family == "f" else sol["ETA"]
    a, xi = W[idx], P[idx]

    slope = _slope_f(x_sample, a, xi)
    beta = 1.0 - np.abs(slope)
    rho = (1.0 - np.abs(x_sample)) + conv_eval(x_sample, a, xi)

    return dict(x=x_sample, x_hat=x_hat_sample, t_read=t_read,
               beta=beta, rho=rho)


def env_distance(env_a, env_b):
    """Distance between two `read_environment` outputs sampled at the SAME
    x_hat grid (raises if grids differ) -- max-norm of the concatenated
    (beta, rho) differences. This is the map E -> E' self-reproduction
    metric: env_distance(E'_k, E_{k+1}) small means generation k+1's
    incoming environment matches generation k's outgoing one, i.e. near a
    fixed point."""
    if not np.allclose(env_a["x_hat"], env_b["x_hat"]):
        raise ValueError("env_distance requires the same x_hat sample grid")
    da = np.concatenate([env_a["beta"], env_a["rho"]])
    db = np.concatenate([env_b["beta"], env_b["rho"]])
    return float(np.max(np.abs(da - db)))


def check_band_neutral(n_gen, lambda_w=0.5, lambda_s=0.5, L=0.3, K=8,
                       sub=8, tol_rel=0.01, **kwargs):
    """Validation gate #1, direct analogue of `check_insertion_neutral`:
    build generation n_gen with ITS OWN band forced dead (`dead_gens=
    {n_gen}`), compare Jc against `certify(build_melt_hierarchy(n_gen-1))`.
    Returns dict(ok, Jc_with_dead, Jc_without, diff, diff_rel)."""
    if n_gen < 1:
        raise ValueError("check_band_neutral needs n_gen >= 1")
    with_dead = build_melt_hierarchy(n_gen, lambda_w=lambda_w,
                                     lambda_s=lambda_s, L=L, K=K,
                                     dead_gens={n_gen}, **kwargs)
    without = build_melt_hierarchy(n_gen - 1, lambda_w=lambda_w,
                                   lambda_s=lambda_s, L=L, K=K, **kwargs)
    Jc_with = certify(with_dead, sub=sub)["Jc"]
    Jc_without = certify(without, sub=sub)["Jc"]
    diff = Jc_with - Jc_without
    diff_rel = abs(diff) / max(abs(Jc_without), 1e-12)
    return dict(ok=diff_rel < tol_rel, Jc_with_dead=Jc_with,
               Jc_without=Jc_without, diff=diff, diff_rel=diff_rel)


def band_travel_sanity(sol, k, band_specs, tol_rel=0.3):
    """Validation gate #3, GENERALIZED from `travel_sanity`: confirm
    generation k's band actually drifts ~L (not co-located with its own
    center at window start) -- checks the REALIZED center displacement of
    the band's OWN columns over its window (median position at window start
    vs window end, across all K columns of one family), not a single
    kink's start/end like `travel_sanity` (a band could have zero center
    drift but nonzero column SPREAD across its width, or vice versa -- these
    are different failure modes and must not be conflated: this checks
    DRIFT, not WIDTH).

    Column identity: generation k's f-columns are at indices
    [1+(k-1)*K : 1+k*K] (1 = coarse carrier, then K columns per generation
    in insertion order -- `build_melt_hierarchy`'s documented column
    layout).

    Returns dict(ok, k, L_expected, L_observed)."""
    spec = band_specs[k - 1]
    K = spec["K"]
    lo, hi = 1 + (k - 1) * K, 1 + k * K
    t = sol["t"]
    tb, td = spec["window"]
    win_idx = np.flatnonzero((t >= tb - 1e-12) & (t <= td + 1e-12))
    XI_band = sol["XI"][:, lo:hi]
    if win_idx.size == 0:
        return dict(ok=False, k=k, L_expected=spec["L"], L_observed=0.0)
    center_start = float(np.median(XI_band[win_idx[0]]))
    center_end = float(np.median(XI_band[win_idx[-1]]))
    L_observed = abs(center_end - center_start)
    ok = abs(L_observed - spec["L"]) < tol_rel * spec["L"] + 1e-9
    return dict(ok=ok, k=k, L_expected=spec["L"], L_observed=L_observed)


def mesh_cross_check(dJk_sequence, target_per_octave=0.215, tol_factor=3.0):
    """Validation gate #4 (NEW relative to Run 11): at n_gen=2-3, dJk should
    land within `tol_factor`x of the mesh's measured +0.215/octave
    (order-of-magnitude, not a tight match -- Run 11's own toy construction
    (0.31 total J after 4 generations) is the cautionary example of an
    ansatz passing gates 1-3 while being off by ~1 order of magnitude from
    the mesh's structure). Returns dict(ok, dJk_sequence, target,
    ratio_to_target=[dJk_i / target for each i])."""
    ratio_to_target = [d / target_per_octave for d in dJk_sequence]
    ok = all(target_per_octave / tol_factor <= d <= target_per_octave * tol_factor
             for d in dJk_sequence)
    return dict(ok=ok, dJk_sequence=list(dJk_sequence), target=target_per_octave,
               ratio_to_target=ratio_to_target)


def fixed_point_sweep(n_gen, lambda_ws, lambda_ss, Ls, K=8, sub=8,
                      env_tol=0.05, **kwargs):
    """Stage 4 driver. For every (lambda_w, lambda_s, L) combination, run
    `melt_ladder`, then compute env_distance(E'_k, E_{k+1,in}) for every
    adjacent generation pair k=1..n_gen-1 (E'_k = generation k's own
    outgoing env, already in the ladder's `env` field; E_{k+1,in} =
    generation (k+1)'s band read off generation k's sol, i.e. what k+1
    would see arriving before it exists). A combination is "near a fixed
    point" if env_distance stays below `env_tol` (and roughly flat, not
    still trending down) for the last 2-3 generations.

    At a near-fixed-point combination, read gamma = dJk / dJ_{k-1} averaged
    over the last few generations (gamma close to 1 => log-growth; gamma < 1,
    stable => geometric decay => bounded, with 1-gamma the per-generation
    "re-arm tax").

    Returns list of dict(lambda_w, lambda_s, L, env_distances=[...],
    gamma_est, gamma_stable, ladder=<melt_ladder result>)."""
    out = []
    for lw in lambda_ws:
        for ls in lambda_ss:
            for L in Ls:
                lad = melt_ladder(n_gen, lambda_w=lw, lambda_s=ls, L=L, K=K,
                                  sub=sub, **kwargs)
                gens = lad["generations"]
                band_specs = gens[-1]["sol"]["band_specs"]
                env_distances = []
                for k in range(1, n_gen):
                    e_out_k = gens[k]["env"]
                    e_in_kp1 = read_environment(gens[k]["sol"], k + 1,
                                                band_specs[k])
                    env_distances.append(env_distance(e_out_k, e_in_kp1))
                stable = (len(env_distances) >= 2 and
                         all(d < env_tol for d in env_distances[-2:]))
                gamma_vals = [g["ratio"] for g in gens[-3:]
                             if g["ratio"] is not None]
                gamma_est = float(np.mean(gamma_vals)) if gamma_vals else float("nan")
                out.append(dict(lambda_w=lw, lambda_s=ls, L=L,
                                env_distances=env_distances,
                                gamma_est=gamma_est, gamma_stable=stable,
                                ladder=lad))
    return out


__all__ = [
    "build_band", "build_melt_hierarchy", "melt_ladder", "melt_sweep",
    "read_environment", "env_distance", "check_band_neutral",
    "band_travel_sanity", "mesh_cross_check", "fixed_point_sweep",
    "grid_convergence_check", "fit_geometric",
]

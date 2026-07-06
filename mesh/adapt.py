"""Adaptive grid construction: tau-gauge time regrid + band-only x refinement.

Pure grid arrays here — no LP surgery (see plans/mesh/00-primer.md §0.4). Two
builders:
  tau_regrid   — re-place time nodes to equidistribute HARVEST instead of time.
  band_refine  — bisect x-intervals only inside the frozen harvest band (I2).

See plans/mesh/01-grids.md for the derivation.
"""
import numpy as np

from .objective import harvest_per_interval

BAND = 0.4   # harvest-band half-width (I2)


def tau_regrid(f, g, x_grid, t_grid, M_new=None, tau_boost=0.0):
    """Re-place time nodes to equidistribute HARVEST instead of time.

    Returns t_new (shape (M_new+1,)), a monotone grid on [0,1] with t_new[0]=0,
    t_new[-1]=1, clustered where harvest is collected.

    Method:
      dJ_t = harvest_per_interval(f, g, x_grid, t_grid)      # (M,)
      tau  = concatenate([[0], cumsum(dJ_t)]); tau /= tau[-1]  # (M+1,) monotone 0..1
      # invert the CDF: equal-harvest targets -> their t positions
      targets = linspace(0, 1, (M_new or M) + 1)
      t_new   = interp(targets, tau, t_grid)

    tau_boost > 0 optionally over-concentrates near the melt peak by blending the
    equal-harvest grid with extra density around tau*~0.38 (I3); leave 0.0 for the
    plain equal-harvest grid and only add if 03/04 show the melt event under-
    resolved.
    """
    M = len(t_grid) - 1
    n_out = (M_new or M) + 1

    dJ_t = harvest_per_interval(f, g, x_grid, t_grid)   # (M,)
    dJ_t = np.clip(dJ_t, 0, None)

    J_total = np.sum(dJ_t)
    if J_total <= 0:
        # Trivial solution (J≈0): nothing to equidistribute, fall back to uniform.
        return np.linspace(0.0, 1.0, n_out)

    # Strictly-increasing CDF: tiny uniform floor so flat (dead) stretches don't
    # produce duplicate tau values, which would break np.interp's inversion.
    eps = 1e-12 * (t_grid[-1] - t_grid[0])
    tau = np.concatenate([[0.0], np.cumsum(dJ_t + eps)])
    tau = tau - tau[0]
    tau = tau / tau[-1]

    targets = np.linspace(0.0, 1.0, n_out)

    if tau_boost > 0.0:
        # Blend equal-harvest targets with extra density around the melt peak
        # tau* ~= 0.38 (I3). Simple approach: pull targets toward tau* by
        # tau_boost, keeping endpoints fixed.
        tau_star = 0.38
        targets = targets + tau_boost * (tau_star - targets) * (targets * (1 - targets)) * 4
        targets = np.clip(targets, 0.0, 1.0)
        targets[0], targets[-1] = 0.0, 1.0
        targets = np.sort(targets)

    t_new = np.interp(targets, tau, t_grid)
    t_new[0] = 0.0
    t_new[-1] = 1.0
    return t_new


def band_refine(x_grid, band=BAND):
    """One octave of x-refinement, INSIDE THE BAND ONLY.

    Bisect every x-interval whose midpoint lies in |x| < band; leave arm intervals
    (|midpoint| >= band) alone. Returns x_new (sorted, includes all old nodes).

    Result: band strand count roughly doubles per call (D1/D2); arms untouched.
    Endpoints ±1 preserved (they're in x_grid and never added twice).
    """
    mids = 0.5 * (x_grid[:-1] + x_grid[1:])
    add = mids[np.abs(mids) < band]
    x_new = np.union1d(x_grid, add)
    return x_new


if __name__ == "__main__":
    from mesh import make_grids, alternating_maximization, compute_J, harvest_per_interval

    # 1. tau_regrid on a real solved solution is monotone, hits endpoints, same length
    x, t = make_grids(4, 32)
    g0 = np.array([[0.5 * t[j] * (x[i] ** 2 - 1) for j in range(len(t))] for i in range(len(x))])
    f, g, _ = alternating_maximization(x, t, g_init=g0, max_iter=30)
    t2 = tau_regrid(f, g, x, t)
    assert t2[0] == 0 and abs(t2[-1] - 1) < 1e-12 and np.all(np.diff(t2) > 0)
    # tau grid should put >50% of its nodes in the live half of [0,1] near the melt
    # (sanity, not exact): median node time shifts toward tau*~0.38 region
    print("median t_new =", np.median(t2), " (uniform would be 0.5)")

    # 2. band_refine doubles band nodes, leaves arms, preserves endpoints
    xb = band_refine(x)
    assert xb[0] == -1 and xb[-1] == 1
    assert np.all(np.isin(x, xb))                       # old nodes preserved
    n_band_old = np.sum(np.abs(x) < BAND)
    n_band_new = np.sum(np.abs(xb) < BAND)
    n_arm_old = np.sum(np.abs(x) >= BAND)
    n_arm_new = np.sum(np.abs(xb) >= BAND)
    assert n_arm_new == n_arm_old                       # arms untouched
    print(f"band nodes {n_band_old}->{n_band_new}, arm nodes {n_arm_old} (unchanged)")

    print("adapt.py acceptance check: PASS")

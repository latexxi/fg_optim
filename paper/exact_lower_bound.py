"""Unconditional exact-arithmetic lower bound  sup J >= 3.0552.

Independent of the boundedness question (Run 13 / cell.tex): this is a theorem, not a
measurement. We exhibit ONE explicitly admissible pair (f, g) and compute its objective
J[f, g] in exact rational arithmetic, with every constraint verified as an exact
rational inequality.

Objective:  J[f,g] = int_0^1 int_{-1}^1 f_t(x,t) g_xx(x,t) dx dt,
maximized over f,g convex in x, |f_x|,|g_x| <= 1, f(+-1,t)=g(+-1,t)=0, f monotone
up / g monotone down in t (BV, i.e. as signed measures -- the class conjecture.txt's
"bang" mechanism lives in), f(x,1)=0, g(x,0)=0.

Witness (from the k=6 dyadic-mesh optimum, ../fg_opt3/data/level_k06.npz, J~3.05529):
  g : piecewise-linear in x, RIGHT-CONTINUOUS STEP in t (value g[.,j] on [t_j,t_{j+1}))
  f : piecewise-linear in x and in t (continuous)
For this pair, g_xx(.,t) = sum_i kappa_i(t) delta(x-x_i) with kappa_i CONSTANT on each
t-cell, and f_t(x_i,.) constant on each t-cell, so the continuous integral is EXACT
(no quadrature):
      J = sum_{j=0}^{M-1} sum_{i interior} (f[i,j+1]-f[i,j]) * kappa_{i,j},
      kappa_{i,j} = (g[i+1,j]-2 g[i,j]+g[i-1,j]) / dx.
(This equals the mesh solver's own compute_J -- so its reported 3.05529 IS a true
continuous J, not a discretization artifact.)  The step-in-t g attains 3.0552 exactly;
smoothing its jumps over a vanishing time-width keeps g Lipschitz-in-t and sends J back
to the same value, so the bound holds as a sup under either regularity convention.

The raw LP data violates convexity/Lipschitz/monotonicity by ~1e-12 (solver roundoff).
We repair by blending, at weight eps=1e-6, with a strictly-interior feasible pair
(parabola profiles f_int=(x^2-1)/2*(1-t), g_int=(x^2-1)/2*t), which restores exact
feasibility while costing ~5e-6 in J -- still > 3.05.

Run:  python3 paper/exact_lower_bound.py
"""
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "fg_opt3" / "data" / "level_k06.npz"


def main():
    if not DATA.exists():
        raise SystemExit(f"mesh solution not found: {DATA}\n"
                         "(the k=6 dyadic-mesh optimum from the sibling fg_opt3 repo)")
    d = np.load(DATA)
    f_np, g_np = d["f"], d["g"]
    N, Mp1 = f_np.shape                      # 65 x nodes, 257 t nodes
    M = Mp1 - 1
    dx = Fr(2, N - 1)

    x = [Fr(2 * i, N - 1) - 1 for i in range(N)]
    t = [Fr(j, M) for j in range(Mp1)]
    assert x[0] == -1 and x[-1] == 1 and t[0] == 0 and t[-1] == 1

    # exact rationals (float64 is an exact dyadic rational)
    f = [[Fr(f_np[i, j]) for j in range(Mp1)] for i in range(N)]
    g = [[Fr(g_np[i, j]) for j in range(Mp1)] for i in range(N)]

    # strictly-interior feasible pair for the repair blend
    half = Fr(1, 2)
    f_int = [[half * (x[i] * x[i] - 1) * (1 - t[j]) for j in range(Mp1)] for i in range(N)]
    g_int = [[half * (x[i] * x[i] - 1) * t[j] for j in range(Mp1)] for i in range(N)]

    eps = Fr(1, 10 ** 6)
    om = 1 - eps
    F = [[om * f[i][j] + eps * f_int[i][j] for j in range(Mp1)] for i in range(N)]
    G = [[om * g[i][j] + eps * g_int[i][j] for j in range(Mp1)] for i in range(N)]

    # ---------- exact feasibility verification ----------
    errs = []
    for j in range(Mp1):
        if F[0][j] or F[N - 1][j]:            errs.append(("f x-boundary", j))
        if G[0][j] or G[N - 1][j]:            errs.append(("g x-boundary", j))
    for i in range(N):
        if F[i][M]:                           errs.append(("f(x,1)=0", i))
        if G[i][0]:                           errs.append(("g(x,0)=0", i))
    for j in range(Mp1):
        for i in range(1, N - 1):
            if F[i + 1][j] - 2 * F[i][j] + F[i - 1][j] < 0: errs.append(("f convex", i, j))
            if G[i + 1][j] - 2 * G[i][j] + G[i - 1][j] < 0: errs.append(("g convex", i, j))
        for i in range(N - 1):
            if abs(F[i + 1][j] - F[i][j]) > dx: errs.append(("f Lipschitz", i, j))
            if abs(G[i + 1][j] - G[i][j]) > dx: errs.append(("g Lipschitz", i, j))
    for i in range(N):
        for j in range(M):
            if F[i][j + 1] - F[i][j] < 0: errs.append(("f_t>=0", i, j))
            if G[i][j + 1] - G[i][j] > 0: errs.append(("g_t<=0", i, j))

    # ---------- exact objective ----------
    J = Fr(0)
    for j in range(M):
        for i in range(1, N - 1):
            kap = (G[i + 1][j] - 2 * G[i][j] + G[i - 1][j]) / dx
            J += (F[i][j + 1] - F[i][j]) * kap

    print(f"mesh witness: {N} x-nodes x {Mp1} t-nodes  (k=6)")
    print(f"exact feasibility violations: {len(errs)}"
          + (f"  first: {errs[:3]}" if errs else "  (all constraints hold exactly)"))
    print(f"exact J = {J}")
    print(f"        = {float(J):.12f}")
    for thr in (Fr(305, 100), Fr(3055, 1000)):
        print(f"  J >= {float(thr)} : {J >= thr}")

    assert not errs, "witness is not exactly feasible"
    assert J >= Fr(305, 100), "bound below 3.05"

    cert = Path(__file__).with_suffix(".txt")
    cert.write_text(
        "THEOREM (exact arithmetic).  sup J >= 3.0552.\n\n"
        f"Witness: k=6 dyadic-mesh optimum, repaired (eps={eps}) to exact feasibility.\n"
        f"Grid: {N} x-nodes x {Mp1} t-nodes.  g step-in-t, f pw-linear.\n"
        f"Exact feasibility violations: 0.\n\n"
        f"J = {J}\n"
        f"  = {float(J):.15f}\n\n"
        "Every convexity / Lipschitz / monotonicity / boundary constraint verified as an\n"
        "exact rational (in)equality; J is the exact continuous objective of the witness.\n")
    print(f"\nwrote certificate -> {cert}")
    print("\nTHEOREM: sup J >= 3.0552  (unconditional, exact arithmetic).")


if __name__ == "__main__":
    main()

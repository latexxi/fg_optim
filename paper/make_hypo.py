"""Schematic of the HYPOTHETICAL blow-up construction (not data).

The anisotropic melting cascade of the paper's Section 4/5: each generation
is a drifting curvature band that contracts in width (w_k), lifetime (s_k)
and rise-share (r_k) but keeps an O(1) travel length. Everything here is
synthetic/idealized -- drawn by formula to illustrate the mechanism, not
solved. Writes paper/figs/fig_hypo.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

OUT = "/home/lauri/fg_opt4/paper/figs"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10,
                     "axes.labelsize": 10, "figure.dpi": 200,
                     "savefig.bbox": "tight"})

# sequential blue ramp for generations 0..3, finest highlighted amber
GEN_COL = ["#1b3a6b", "#2f6db0", "#5ba3d0", "#e08214"]
NGEN = 4

# ------------------------------------------------------------------ the map
# recursively build drift paths c_k(t): each generation tiles its parent's
# lifetime into 2 sub-windows and rides a faster O(1) excursion on the parent
# center. amplitude shrinks mildly (legibility); width & window halve (scaling).
def excursion(u):                       # there-and-back dip on [0,1] -> [0,-1,0]
    return -np.sin(np.pi * u)

def build(gen, t0, t1, base_fn, amp, segs):
    """append (gen, t-array, x-array) for this window, then recurse."""
    tt = np.linspace(t0, t1, 200)
    xx = base_fn(tt) + amp * excursion((tt - t0) / (t1 - t0))
    segs.append((gen, tt, xx))
    if gen + 1 >= NGEN:
        return
    path_fn = lambda t, b=base_fn, a=amp, s=t0, e=t1: \
        b(t) + a * excursion(np.clip((t - s) / (e - s), 0, 1))
    tm = 0.5 * (t0 + t1)
    build(gen + 1, t0, tm, path_fn, amp * 0.62, segs)
    build(gen + 1, tm, t1, path_fn, amp * 0.62, segs)

segs = []
build(0, 0.0, 1.0, lambda t: np.full_like(t, 0.05), 0.45, segs)
W0 = 0.13                                # band half-width at gen 0

# ------------------------------------------------------------ slice profiles
# f interior slice: convex (slope non-decreasing), arms exactly +-1, and the
# sharp vertex replaced by a widening staircase basin that refines with each
# generation. Built from an odd slope profile so f(+-1)=0 exactly.
W_GEN = [0.0, 0.30, 0.50, 0.62]                       # basin half-width per gen
def melt_slice(x, gen):
    W = W_GEN[gen]
    if W == 0.0:                                     # gen 0: the tent (jump)
        slope = np.sign(x)
    else:
        u = np.clip(x / W, -1, 1)                    # ramp across the band...
        n = 2 ** gen                                 # ...quantized to 2^gen steps
        slope = np.where(np.abs(x) < W, np.round(u * n) / n, np.sign(x))
    f = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1])
                                         * np.diff(x))])
    return f                                         # f(-1)=f(1)=0 by oddness

xg = np.linspace(-1, 1, 801)

# --------------------------------------------------------------- dJ ladders
kk = np.arange(1, 8)
dJ_const = 0.215 * np.ones_like(kk, dtype=float)          # blow-up scenario
dJ_decay = 0.30 * 0.55 ** (kk - 1)                        # bounded scenario

# ================================================================== figure
fig = plt.figure(figsize=(10.5, 6.6))
gs = gridspec.GridSpec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42,
                       wspace=0.30)

# ---- top: the (t,x) cascade map (spans all 3 columns) -------------------
axm = fig.add_subplot(gs[0, :])
for gen, tt, xx in segs:
    lw = 7.0 * 0.5 ** gen                            # band width ~ w_k
    axm.plot(tt, xx, color=GEN_COL[gen], lw=lw, solid_capstyle="round",
             alpha=0.9, zorder=gen)
# legend proxies
for gen in range(NGEN):
    axm.plot([], [], color=GEN_COL[gen], lw=3.5,
             label=f"gen {gen}:  $w_{{{gen}}}\\!\\sim\\!2^{{-{gen}}}$, "
                   f"$s_{{{gen}}}\\!\\sim\\!2^{{-{gen}}}$")
axm.axhline(0.05, color="gray", ls=":", lw=0.8)
axm.set_xlim(0, 1); axm.set_ylim(-0.85, 0.30)
axm.set_xlabel("$t$"); axm.set_ylabel("$x$  (support of $g_{xx}$)")
axm.set_title("Hypothetical cascade: support of $g_{xx}$ drifts and sub-divides "
              "(travel length $L\\sim O(1)$ fixed; width, lifetime, rise-share $\\sim 2^{-k}$)")
axm.legend(frameon=False, fontsize=8, loc="upper center", ncol=4,
           handlelength=1.6, columnspacing=1.2)
axm.text(0.015, 0.05, "each finer generation reaps fresh rise budget "
         "$1-|x|$ at new $(x,t)$ it visits",
         transform=axm.transAxes, fontsize=8.5, color="#444", va="bottom")

# ---- bottom-left: f slice refinement -----------------------------------
axf = fig.add_subplot(gs[1, 0])
for ng in range(NGEN):
    axf.plot(xg, melt_slice(xg, ng), color=GEN_COL[ng], lw=1.6,
             label=f"gen {ng}")
axf.plot(xg, -(1 - np.abs(xg)), color="gray", ls=":", lw=1, label="full tent")
axf.set_title("$f(\\cdot,t)$: widening basin, arms $\\pm1$ kept")
axf.set_xlabel("$x$")
axf.set_ylabel("depth"); axf.legend(frameon=False, fontsize=7.5, loc="upper center")

# ---- bottom-mid: g slice refinement (mirror in time role) --------------
axg = fig.add_subplot(gs[1, 1])
for ng in range(NGEN):
    axg.plot(xg, melt_slice(xg, ng), color=GEN_COL[ng], lw=1.6)
axg.plot(xg, -(1 - np.abs(xg)), color="gray", ls=":", lw=1)
axg.set_title("$g(\\cdot,t)$: widening basin, arms $\\pm1$ kept")
axg.set_xlabel("$x$")
axg.set_yticklabels([])

# ---- bottom-right: dJ ladder, the payoff -------------------------------
axd = fig.add_subplot(gs[1, 2])
axd.plot(kk, dJ_const, "o-", color="#b0413e",
         label="constant $\\Delta J_k$\n$\\Rightarrow\\ J\\to\\infty$")
axd.plot(kk, dJ_decay, "s--", color="#2f6db0", mfc="white",
         label="decaying $\\Delta J_k$\n$\\Rightarrow\\ J$ bounded")
axd.set_ylim(0, 0.35)
axd.set_title("per-generation gain $\\Delta J_k$"); axd.set_xlabel("generation $k$")
axd.set_ylabel("$\\Delta J_k$"); axd.legend(frameon=False, fontsize=7.5)

fig.savefig(f"{OUT}/fig_hypo.png")
print("wrote", f"{OUT}/fig_hypo.png")

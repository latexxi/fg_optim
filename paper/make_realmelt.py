"""Does the Fig.-2 hypothesis (arms +-1 preserved, vertex melting into a
widening multi-kink basin as generations accumulate) appear in the actual
computed slices? Shown at the levels where it is clearest: k6 (best resolved)
with k5 as confirmation. k7/k8 are time-starved and omitted. Reads
~/fg_opt3/data/. Writes paper/figs/fig_realmelt.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

DATA = "/home/lauri/fg_opt3/data"
OUT = "/home/lauri/fg_opt4/paper/figs"
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

def load(k):
    z = np.load(f"{DATA}/level_k0{k}.npz")
    return z["f"], z["g"], z["x_grid"], z["t_grid"]

fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.4))
fig.subplots_adjust(wspace=0.34)
cmap = plt.cm.viridis

# time samples: tent end -> interior melt (the real "generations" turning on)
TF = [0.0, 0.06, 0.12, 0.20, 0.30]        # f melts away from t=0
TG = [1.0, 0.94, 0.88, 0.80, 0.70]        # g melts away from t=1

for row, k in enumerate([6, 5]):
    f, g, x, t = load(k)
    dt = t[1] - t[0]
    normf = Normalize(0, 0.35); normg = Normalize(0.65, 1.0)
    # --- f slice family ---
    ax = axes[row, 0]
    for tv in TF:
        j = int(round(tv / dt))
        ax.plot(x, f[:, j], color=cmap(normf(tv)), lw=1.7)
    ax.plot([-1, 0, 1], [0, -1, 0], "k:", lw=1.2, label="tent ($t=0$)")
    ax.set_ylabel(f"level {k}\n\ndepth"); ax.set_title("$f(\\cdot,t)$: tent melts to basin")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    if row == 1: ax.set_xlabel("$x$")
    # --- g slice family ---
    ax = axes[row, 1]
    for tv in TG:
        j = int(round(tv / dt))
        ax.plot(x, g[:, j], color=cmap(normg(tv)), lw=1.7)
    ax.plot([-1, 0, 1], [0, -1, 0], "k:", lw=1.2, label="tent ($t=1$)")
    ax.set_title("$g(\\cdot,t)$: tent melts to basin")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    if row == 1: ax.set_xlabel("$x$")
    # --- slope staircase: arms +-1 kept, middle ramps through a band ---
    ax = axes[row, 2]
    xm = 0.5 * (x[:-1] + x[1:])
    j0 = 0                                     # tent
    jm = int(round(0.20 / dt))                 # melted
    ax.step(xm, np.diff(f[:, j0]) / (x[1]-x[0]), where="mid",
            color="#3b6fb6", lw=1.6, label="$t=0$ (tent: one jump)")
    ax.step(xm, np.diff(f[:, jm]) / (x[1]-x[0]), where="mid",
            color="#b0413e", lw=1.6, label="$t=0.20$ (melt: staircase)")
    ax.axhline(1, color="gray", ls=":", lw=0.7); ax.axhline(-1, color="gray", ls=":", lw=0.7)
    ax.set_ylabel("$f_x$"); ax.set_title("slope: arms $\\pm1$ kept, band ramps")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    if row == 1: ax.set_xlabel("$x$")

# colorbar for time, placed on the far right so it clears the f_x labels
sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
cb = fig.colorbar(sm, ax=axes[:, 2].tolist(), shrink=0.62, pad=0.02,
                  location="right")
cb.set_label("$t$")

fig.suptitle("Fig.-2 hypothesis in the computed data: arms stay $\\pm1$, the vertex "
             "melts into a widening multi-kink basin  (level 6 top, level 5 bottom)",
             fontsize=10.5, y=0.99)
fig.savefig(f"{OUT}/fig_realmelt.png")
print("wrote", f"{OUT}/fig_realmelt.png")

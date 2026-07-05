"""Generate the figures for melting.tex from the fg_opt3 mesh data.

Reads /home/lauri/fg_opt3/data/level_k0{1..8}.npz and writes PNGs into
paper/figs/.  Purely a plotting script: every quantity shown is computed
directly from the saved f, g fields by finite differences.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

DATA = "/home/lauri/fg_opt3/data"
OUT = "/home/lauri/fg_opt4/paper/figs"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "axes.grid": False,
    "savefig.bbox": "tight",
})

levels = {}
for k in range(1, 9):
    z = np.load(f"{DATA}/level_k0{k}.npz")
    levels[k] = dict(f=z["f"], g=z["g"], x=z["x_grid"], t=z["t_grid"],
                     J=float(z["J"]))

# ---------------------------------------------------------------- fig_levels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
ks = np.arange(1, 9)
J = np.array([levels[k]["J"] for k in ks])
Hf = np.array([np.sum(np.max(np.diff(levels[k]["f"], axis=1), axis=0))
               for k in ks])
clean = [1, 2, 3, 4, 5, 6]          # Mt kept in step with Nx
starved = [7, 8]                    # x refined, t starved

c_main, c_bad = "#3b6fb6", "#b0413e"
ax1.plot(clean, J[:6], "o-", color=c_main, label="$M_t$ scaled with $N_x$")
ax1.plot(starved, J[6:], "o--", color=c_bad, mfc="white",
         label="$t$-starved ($M_t$ not scaled)")
ax1.plot([6, 7], [J[5], J[6]], "--", color=c_bad, lw=1, zorder=1)
ax1.axhline(2.0, color="gray", lw=0.8, ls=":")
ax1.text(1.1, 2.03, "tent cap $J=2$", color="gray", fontsize=9)
ax1.annotate("+0.211", xy=(4.5, 2.73), fontsize=9, color=c_main, ha="center")
ax1.annotate("+0.219", xy=(5.5, 2.96), fontsize=9, color=c_main, ha="center")
ax1.set_xlabel("level $k$   ($N_x = 2^k{+}1$)")
ax1.set_ylabel("certified $J$")
ax1.set_title("(a)  $J$ per refinement level")
ax1.legend(frameon=False, fontsize=8, loc="lower right")

ax2.plot(clean, Hf[:6], "s-", color=c_main)
ax2.plot(starved, Hf[6:], "s--", color=c_bad, mfc="white")
ax2.plot([6, 7], [Hf[5], Hf[6]], "--", color=c_bad, lw=1, zorder=1)
ax2.axhline(1.0, color="gray", lw=0.8, ls=":")
ax2.text(1.1, 1.06, "tent value $H=1$", color="gray", fontsize=9)
ax2.set_xlabel("level $k$")
ax2.set_ylabel(r"$H[f]=\int_0^1 \sup_x f_t\,dt$")
ax2.set_title("(b)  peak-rise functional $H[f]$")
fig.savefig(f"{OUT}/fig_levels.png")
plt.close(fig)

# ---------------------------------------------------------------- k06 fields
z = levels[6]
f, g, x, t = z["f"], z["g"], z["x"], z["t"]
dx, dt = x[1] - x[0], t[1] - t[0]
ext = [t[0], t[-1], x[0], x[-1]]    # imshow with [x_index, t_index] layout

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), layout="constrained")
for ax, F, name in [(axes[0], f, "$f(x,t)$"), (axes[1], g, "$g(x,t)$")]:
    im = ax.imshow(F, origin="lower", aspect="auto", extent=ext,
                   cmap="viridis", vmin=-1, vmax=0)
    ax.set_xlabel("$t$"); ax.set_ylabel("$x$"); ax.set_title(name)
    fig.colorbar(im, ax=ax, shrink=0.9)
fig.savefig(f"{OUT}/fig_fields.png")
plt.close(fig)

# ---------------------------------------------------------------- fig_slices
# every time slice, colored by t, so the full melt evolution is visible.
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True,
                               layout="constrained")
Mt = t.size
cmap = plt.cm.viridis
norm = Normalize(0, 1)
for ax, F, name in [(ax1, f, "$f(\\cdot,t)$, all $t$"),
                    (ax2, g, "$g(\\cdot,t)$, all $t$")]:
    for j in range(Mt):
        ax.plot(x, F[:, j], color=cmap(norm(t[j])), lw=0.6, alpha=0.7)
    ax.plot([-1, 0, 1], [0, -1, 0], color="black", ls=":", lw=1.2,
            zorder=5, label="full tent")
    ax.set_xlabel("$x$"); ax.set_title(name)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
ax1.set_ylabel("depth")
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=(ax1, ax2),
                  shrink=0.9, pad=0.02)
cb.set_label("$t$")
fig.savefig(f"{OUT}/fig_slices.png")
plt.close(fig)

# ---------------------------------------------------------------- fig_slopes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.9))
xm = 0.5 * (x[:-1] + x[1:])
for tv, c in zip([0.0, 0.5], ["#3b6fb6", "#b0413e"]):
    j = int(round(tv / dt))
    ax1.step(xm, np.diff(f[:, j]) / dx, where="mid", color=c, lw=1.5,
             label=f"$t={tv:g}$")
ax1.axhline(1, color="gray", lw=0.7, ls=":"); ax1.axhline(-1, color="gray", lw=0.7, ls=":")
ax1.set_xlabel("$x$"); ax1.set_ylabel("$f_x$")
ax1.set_title("(a)  slope profile of $f$: jump vs. staircase")
ax1.legend(frameon=False, fontsize=9)

for tv, c in zip([0.0, 0.5], ["#3b6fb6", "#b0413e"]):
    j = int(round(tv / dt))
    curv = np.abs(np.diff(f[:, j], 2)) / dx      # slope change per cell
    ax2.bar(x[1:-1], curv, width=0.9 * dx, color=c, alpha=0.75,
            label=f"$t={tv:g}$")
ax2.set_xlabel("$x$"); ax2.set_ylabel("slope change per cell")
ax2.set_title("(b)  curvature distribution: delta vs. band")
ax2.legend(frameon=False, fontsize=9)
fig.savefig(f"{OUT}/fig_slopes.png")
plt.close(fig)

# ------------------------------------------------------------------ fig_melt
ft = np.diff(f, axis=1) / dt                      # (Nx, Mt-1), at t midpoints
gxx = np.abs(np.diff(g, 2, axis=0)) / dx**2       # (Nx-2, Mt), at interior x
harv = np.maximum(ft[1:-1, :], 0) * 0.5 * (gxx[:, 1:] + gxx[:, :-1])

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), layout="constrained")
panels = [
    (np.maximum(ft, 0), [t[0], t[-1], x[0], x[-1]],
     "$f_t$  (rise rate)"),
    (gxx, [t[0], t[-1], x[1], x[-2]],
     "$|g_{xx}|$  (curvature density)"),
    (harv, [t[0], t[-1], x[1], x[-2]],
     "$f_t \\cdot g_{xx}$  (harvest density)"),
]
for ax, (F, e, name) in zip(axes, panels):
    im = ax.imshow(F, origin="lower", aspect="auto", extent=e, cmap="magma",
                   norm=PowerNorm(0.35))
    ax.set_xlabel("$t$"); ax.set_ylabel("$x$"); ax.set_title(name)
    fig.colorbar(im, ax=ax, shrink=0.9)
fig.savefig(f"{OUT}/fig_melt.png")
plt.close(fig)

print("figures written to", OUT)

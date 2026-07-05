"""Visualize f, g, harvest, J and gamma across the levels where new melt
generations actually become visible: k04, k05, k06 (the regime above the
tent cap). Writes paper/figs/fig_gamma.png. All from ~/fg_opt3/data/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import PowerNorm

DATA = "/home/lauri/fg_opt3/data"
OUT = "/home/lauri/fg_opt4/paper/figs"
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

LEV = [4, 5, 6]
dat = {}
for k in LEV + [3]:
    z = np.load(f"{DATA}/level_k0{k}.npz")
    dat[k] = dict(f=z["f"], g=z["g"], x=z["x_grid"], t=z["t_grid"],
                  J=float(z["J"]))

def harvest(d):
    f, g, x, t = d["f"], d["g"], d["x"], d["t"]
    dx, dt = x[1] - x[0], t[1] - t[0]
    ft = np.maximum(np.diff(f, axis=1) / dt, 0)              # (Nx, Mt-1)
    gxx = np.abs(np.diff(g, 2, axis=0)) / dx**2              # (Nx-2, Mt)
    gxx_m = 0.5 * (gxx[:, 1:] + gxx[:, :-1])                 # -> Mt-1
    return ft[1:-1, :] * gxx_m, [t[0], t[-1], x[1], x[-2]]

fig = plt.figure(figsize=(10.6, 8.6))
gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1.25], hspace=0.5,
                       wspace=0.16)

# rows 0,1,2 : f, g, harvest ; cols : levels
for ci, k in enumerate(LEV):
    d = dat[k]
    ext = [d["t"][0], d["t"][-1], d["x"][0], d["x"][-1]]
    # f
    axf = fig.add_subplot(gs[0, ci])
    axf.imshow(d["f"], origin="lower", aspect="auto", extent=ext,
               cmap="viridis", vmin=-1, vmax=0)
    axf.set_title(f"level {k}:  $N_x={d['x'].size}$, $M_t={d['t'].size}$,  "
                  f"$J={d['J']:.3f}$")
    if ci == 0: axf.set_ylabel("$f(x,t)$\n\n$x$")
    axf.set_xticklabels([])
    # g
    axg = fig.add_subplot(gs[1, ci])
    axg.imshow(d["g"], origin="lower", aspect="auto", extent=ext,
               cmap="viridis", vmin=-1, vmax=0)
    if ci == 0: axg.set_ylabel("$g(x,t)$\n\n$x$")
    axg.set_xticklabels([])
    # harvest
    axh = fig.add_subplot(gs[2, ci])
    H, he = harvest(d)
    axh.imshow(H, origin="lower", aspect="auto", extent=he, cmap="magma",
               norm=PowerNorm(0.35))
    if ci == 0: axh.set_ylabel("$f_t\\,g_{xx}$\n\n$x$")
    axh.set_xlabel("$t$")

# bottom : J and gamma
axJ = fig.add_subplot(gs[3, :])
ks = [3, 4, 5, 6]
J = [dat[k]["J"] for k in ks]
axJ.plot(ks, J, "o-", color="#3b6fb6", lw=2, ms=7)
axJ.axhline(2.0, color="gray", ls=":", lw=0.9)
axJ.text(3.02, 2.02, "tent cap $J=2$", color="gray", fontsize=8.5)
# dJ labels on the clean melt octaves
dJ45 = dat[5]["J"] - dat[4]["J"]
dJ56 = dat[6]["J"] - dat[5]["J"]
axJ.annotate(f"$\\Delta J={dJ45:.3f}$", xy=(4.5, 2.72), ha="center",
             fontsize=9, color="#b0413e")
axJ.annotate(f"$\\Delta J={dJ56:.3f}$", xy=(5.5, 2.94), ha="center",
             fontsize=9, color="#b0413e")
gamma = dJ56 / dJ45
axJ.annotate(f"$\\gamma=\\dfrac{{\\Delta J_{{5\\to6}}}}{{\\Delta J_{{4\\to5}}}}"
             f"=\\dfrac{{{dJ56:.3f}}}{{{dJ45:.3f}}}\\approx {gamma:.2f}$",
             xy=(4.75, 2.25), fontsize=13, ha="center",
             bbox=dict(boxstyle="round", fc="#fdf3e7", ec="#e0a066"))
axJ.set_xticks(ks)
axJ.set_xlabel("level $k$  (one new melt generation per level)")
axJ.set_ylabel("certified $J$")
axJ.set_xlim(2.85, 6.15); axJ.set_ylim(1.9, 3.2)

fig.savefig(f"{OUT}/fig_gamma.png")
print(f"gamma = {gamma:.4f}  (dJ45={dJ45:.4f}, dJ56={dJ56:.4f})")
print("wrote", f"{OUT}/fig_gamma.png")

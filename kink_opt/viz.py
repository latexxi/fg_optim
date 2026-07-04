"""Shared plotting helpers for f(x,t)/g(x,t), used by visualize.py and
kink_opt/persist.py. Agg backend -- callers only ever savefig, never show,
so this must not require a display."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from .geometry import conv_eval


def field_grid(sol, nx=401):
    """Sample f(x,t), g(x,t) on a dense x-grid. -> (x, F, G)."""
    x = np.linspace(-1.0, 1.0, nx)
    Np1 = len(sol["t"])
    F = np.array([conv_eval(x, sol["A"][k], sol["XI"][k]) for k in range(Np1)])
    G = np.array([conv_eval(x, sol["B"][k], sol["ETA"][k]) for k in range(Np1)])
    return x, F, G


def plot_heatmaps(sol, nx=401, figsize=(14, 5)):
    """Heatmaps of f(x,t) and g(x,t)."""
    x, F, G = field_grid(sol, nx=nx)
    t = sol["t"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    im1 = ax1.contourf(x, t, F, levels=20, cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title("f(x,t)")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.contourf(x, t, G, levels=20, cmap="plasma")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title("g(x,t)")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    return fig

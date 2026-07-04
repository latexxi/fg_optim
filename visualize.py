#!/usr/bin/env python3
"""Visualize f and g functions from kink_opt results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from kink_opt import run, conv_eval, report
from kink_opt.viz import plot_heatmaps, field_grid


def plot_surfaces(r, nx=401, figsize=(14, 5)):
    """3D surface plots of f(x,t) and g(x,t)."""
    x = np.linspace(-1.0, 1.0, nx)
    t = r["t"]
    Np1 = len(t)

    # Evaluate f and g on grid
    F = np.array([conv_eval(x, r["A"][k], r["XI"][k]) for k in range(Np1)])
    G = np.array([conv_eval(x, r["B"][k], r["ETA"][k]) for k in range(Np1)])

    X, T = np.meshgrid(x, t)

    fig = plt.figure(figsize=figsize)

    # f(x,t)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, T, F, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("f(x,t)")
    ax1.set_title("f(x,t): harvesting function")

    # g(x,t)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, T, G, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("g(x,t)")
    ax2.set_title("g(x,t): weights function")

    plt.tight_layout()
    return fig


def plot_slices(r, t_indices=None, nx=401, figsize=(12, 6)):
    """Cross-sections of f and g at selected time points."""
    if t_indices is None:
        # Plot at t=0, middle, and t=1
        t_indices = [0, len(r["t"]) // 2, len(r["t"]) - 1]

    x = np.linspace(-1.0, 1.0, nx)
    t = r["t"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx in t_indices:
        F = conv_eval(x, r["A"][idx], r["XI"][idx])
        G = conv_eval(x, r["B"][idx], r["ETA"][idx])
        axes[0].plot(x, F, label=f"t={t[idx]:.2f}")
        axes[1].plot(x, G, label=f"t={t[idx]:.2f}")

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x,t)")
    axes[0].set_title("f at selected times")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("g(x,t)")
    axes[1].set_title("g at selected times")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_kink_trajectories(r, figsize=(10, 6)):
    """Plot kink position trajectories over time."""
    t = r["t"]
    XI = r["XI"]
    ETA = r["ETA"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # f-kinks (XI)
    for i in range(XI.shape[1]):
        ax1.plot(t, XI[:, i], marker=".", label=f"kink {i}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("position")
    ax1.set_title("f-kink trajectories (XI)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.05, 1.05)

    # g-kinks (ETA)
    for m in range(ETA.shape[1]):
        ax2.plot(t, ETA[:, m], marker=".", label=f"kink {m}")
    ax2.set_xlabel("t")
    ax2.set_ylabel("position")
    ax2.set_title("g-kink trajectories (ETA)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.05, 1.05)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Running optimization...")
    r = run(N=16, Kf=3, Kg=2, outer=5, seed="static", pos_iters=40, verbose=False)
    print(f"J = {r['J']:.5f}")

    print("\nGenerating visualizations...")
    plot_surfaces(r).savefig("surfaces.png", dpi=100)
    print("✓ surfaces.png")

    plot_heatmaps(r).savefig("heatmaps.png", dpi=100)
    print("✓ heatmaps.png")

    plot_slices(r).savefig("slices.png", dpi=100)
    print("✓ slices.png")

    plot_kink_trajectories(r).savefig("kink_trajectories.png", dpi=100)
    print("✓ kink_trajectories.png")

    print("\nRunning verification...")
    report("result", r)

    plt.show()

"""Matplotlib figure builders for Bohmian trajectory plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def create_trajectory_3d_figure(X, Y, Z, Lx, Ly, Lz, z_det, arr_sel=None):
    """Create the 3D selected-trajectory figure without saving or showing it."""
    fig = plt.figure(figsize=(10.0, 10.0))
    ax = fig.add_subplot(111, projection="3d")

    # ambient background (outside the box)
    fig.patch.set_facecolor("#fef7e7")   # same as your rcParams
    ax.set_facecolor("#fef7e7")

    # make the three panes (inside of the box) pure white and opaque
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")   # pure white
        axis.pane.set_edgecolor("black")   # box edges
        axis.pane.set_alpha(1.0)           # no transparency

    # light grid on top of that
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "_axinfo"):
            axis._axinfo["grid"]["linewidth"] = 0.4
            axis._axinfo["grid"]["color"] = (0.6, 0.6, 0.6, 0.25)

    # detector rectangle at top (no label -> no legend)
    xx = np.array([0, Lx, Lx, 0, 0])
    yy = np.array([0, 0, Ly, Ly, 0])
    zz = np.full_like(xx, z_det)
    ax.plot(xx, yy, zz, lw=1.0, alpha=0.3, color="k")   # very soft outline

    # trajectories
    for j in range(X.shape[1]):
        c = "#d62728" if (arr_sel is not None and arr_sel[j]) else "tab:blue"
        ax.plot(X[:, j], Y[:, j], Z[:, j], lw=0.8, alpha=0.9, color=c)

    # axis limits
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)

    ax.plot([0, 0], [0, 0], [0, Lz],
            color="gray", lw=1, zorder=10)

    # visually widen base
    xy_scale = 2.5
    z_scale = Lz / max(Lx, Ly)
    ax.set_box_aspect((xy_scale, xy_scale, z_scale))

    # nice ticks
    xticks = [0.0, 0.5 * Lx, Lx]
    yticks = [0.0, 0.5 * Ly, Ly]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    fmt = FormatStrFormatter("%.3g")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.tick_params(axis="y", labelsize=8, pad=2)
    ax.tick_params(axis="z", labelsize=10)

    ax.set_xlabel("x", labelpad=10)
    ax.set_ylabel("y", labelpad=10)
    ax.set_zlabel("z")
    ax.set_title("One randomly selected Bohmian trajectory", pad=18)

    # no title, no legend
    # let axes occupy more of the figure -> bigger box
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.08, top=0.98)

    return fig


def create_trajectory_projection_figure(X, Y, Z, Lx, Ly, Lz, z_det, arr_sel=None):
    """Create the XY/XZ/YZ selected-trajectory projection figure."""
    # make figure a bit taller (second number) so the panels are not so flat
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 6.0))

    # inside of each panel: white (or very light)
    box_bg = "white"          # or "#fef7e7" for a tiny warm tint
    for ax2 in axs:
        ax2.set_facecolor(box_bg)

    # define colors once
    color_arrived = "#d62728"   # nice strong red
    color_unarrived = "tab:blue"  # blue from tab palette

    # XY
    for j in range(X.shape[1]):
        c = color_arrived if (arr_sel is not None and arr_sel[j]) else color_unarrived
        axs[0].plot(X[:, j], Y[:, j], lw=0.8, alpha=0.9, color=c)

    axs[0].set_xlim(0, Lx)
    axs[0].set_ylim(0, Ly)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_title("XY")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(alpha=0.25)

    # XZ
    for j in range(X.shape[1]):
        c = color_arrived if (arr_sel is not None and arr_sel[j]) else color_unarrived
        axs[1].plot(X[:, j], Z[:, j], lw=0.8, alpha=0.9, color=c)

    axs[1].axhline(z_det, ls="--", lw=1.0, color="k")
    axs[1].set_xlim(0, Lx)
    axs[1].set_ylim(0, Lz)
    axs[1].set_title("XZ")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].grid(alpha=0.25)

    # YZ
    for j in range(X.shape[1]):
        c = color_arrived if (arr_sel is not None and arr_sel[j]) else color_unarrived
        axs[2].plot(Y[:, j], Z[:, j], lw=0.8, alpha=0.9, color=c)

    axs[2].axhline(z_det, ls="--", lw=1.0, color="k")
    axs[2].set_xlim(0, Ly)
    axs[2].set_ylim(0, Lz)
    axs[2].set_title("YZ")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("z")
    axs[2].grid(alpha=0.25)

    # title INSIDE the figure, with room reserved
    fig.suptitle("Projections of selected Bohmian trajectory", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])  # leave top 6% for the title

    return fig

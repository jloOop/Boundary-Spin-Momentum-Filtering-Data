# %% cell 0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-time-step probability-density plots for rho_prob_t*.npy
using original plotting style (midplanes, XY contour,
3D scatter, isosurface, optional slice bars).

Backward-compatible with:
  - old runs that saved:
      * mid_x, mid_y, mid_z in constants.npz
      * X_cpu.npy, Y_cpu.npy, Z_cpu.npy
  - new solver runs that save:
      * x.npy, y.npy, z.npy
      * rho_prob_t{tau:.5f}.npy
      * no mid_x/mid_y/mid_z in constants.npz

Now: only plots snapshots whose times are multiples of SNAP_DT
(e.g. 0.1, 0.2, 0.3, ...).

Added:
  - automatic computation of probability in each z-slice directly from rho_prob
    if prob_slices_t*.npy is not already present
"""

from pathlib import Path
import glob
import re
import imageio.v2 as imageio

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage import measure

from matplotlib.figure import Figure
from datetime import datetime
from matplotlib import transforms as mtransforms


# ──────────────────────────────────────────────────────────────
# Automatic in-axes signature on every saved figure
# ──────────────────────────────────────────────────────────────

SIGNATURE_TEXT = f"© J & T {datetime.now().year}"
SIGNATURE_KW = dict(
    ha="right", va="bottom", fontsize=8, alpha=0.6,
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor="white", alpha=0.3,
        edgecolor="none"
    ),
)

SIGNATURE_OFFSET_CM_3D = (-4.5, 1.8)  # (left, up) in centimeters
MARGIN_AXES = 0.006


def _signed_savefig(self, *args, **kwargs):
    """Patch Figure.savefig to add the signature inside main axes."""
    temp_artists = []
    try:
        for ax in self.get_axes():
            if not ax.get_visible():
                continue
            bbox = ax.get_position()

            # Skip small axes (e.g. colorbars)
            if bbox.width < 0.20 or bbox.height < 0.20 or getattr(ax, "name", "") == "colorbar":
                continue

            is_3d = getattr(ax, "name", "") == "3d" or hasattr(ax, "get_zlim3d")

            if not is_3d:
                artist = ax.text(
                    0.98, 0.02, SIGNATURE_TEXT,
                    transform=ax.transAxes,
                    **SIGNATURE_KW
                )
            else:
                dx_cm, dy_cm = SIGNATURE_OFFSET_CM_3D
                dx_in = dx_cm / 2.54
                dy_in = dy_cm / 2.54
                offset = mtransforms.ScaledTranslation(dx_in, dy_in, self.dpi_scale_trans)

                base_x = bbox.x1 - MARGIN_AXES
                base_y = bbox.y0 + MARGIN_AXES
                transform = self.transFigure + offset

                artist = self.text(
                    base_x, base_y, SIGNATURE_TEXT,
                    transform=transform,
                    zorder=1_000_000,
                    clip_on=False,
                    **SIGNATURE_KW
                )

            temp_artists.append(artist)

        return self.canvas.print_figure(*args, **kwargs)
    finally:
        for a in temp_artists:
            try:
                a.remove()
            except Exception:
                pass


Figure.savefig = _signed_savefig


# ──────────────────────────────────────────────────────────────
# Locate data_dir (folder that contains constants.npz)
# ──────────────────────────────────────────────────────────────

try:
    start_dir = Path(__file__).resolve().parent
except NameError:  # Jupyter / REPL
    start_dir = Path.cwd()

DATA_DIR = None
for cand in [start_dir] + list(start_dir.parents):
    if (cand / "constants.npz").exists():
        DATA_DIR = cand
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find 'constants.npz' starting from {start_dir} upwards.\n"
        "Run this script inside your simulation folder or a subfolder of it."
    )

data_dir = DATA_DIR
plot_dir = data_dir / "plots_general3"
plot_dir.mkdir(exist_ok=True)

print(f"[info] data_dir  = {data_dir}")
print(f"[info] plot_dir  = {plot_dir}")


# ──────────────────────────────────────────────────────────────
# Load constants & grids
# ──────────────────────────────────────────────────────────────

constants = np.load(data_dir / "constants.npz")
const_keys = set(constants.files)

Lx = float(constants["Lx"])
Ly = float(constants["Ly"])
Lz = float(constants["Lz"])
hx = float(constants["hx"])
hy = float(constants["hy"])
hz = float(constants["hz"])
Nx = int(constants["Nx"])
Ny = int(constants["Ny"])
Nz = int(constants["Nz"])

# Old runs had these; new solver does not
mid_x = int(constants["mid_x"]) if "mid_x" in const_keys else Nx // 2
mid_y = int(constants["mid_y"]) if "mid_y" in const_keys else Ny // 2
mid_z = int(constants["mid_z"]) if "mid_z" in const_keys else Nz // 2

# Prefer new 1D grid files if present
if (data_dir / "x.npy").exists() and (data_dir / "y.npy").exists() and (data_dir / "z.npy").exists():
    x = np.load(data_dir / "x.npy")
    y = np.load(data_dir / "y.npy")
    z = np.load(data_dir / "z.npy")
    print("[info] Using 1D grids: x.npy, y.npy, z.npy")

    # Build only the 2D meshes actually needed for contour/midplane plots
    X_xy, Y_xy = np.meshgrid(x, y, indexing="ij")
    X_xz, Z_xz = np.meshgrid(x, z, indexing="ij")
    Y_yz, Z_yz = np.meshgrid(y, z, indexing="ij")

    HAVE_FULL_3D_MESH = False
    X_cpu = Y_cpu = Z_cpu = None

elif (data_dir / "X_cpu.npy").exists() and (data_dir / "Y_cpu.npy").exists() and (data_dir / "Z_cpu.npy").exists():
    X_cpu = np.load(data_dir / "X_cpu.npy")
    Y_cpu = np.load(data_dir / "Y_cpu.npy")
    Z_cpu = np.load(data_dir / "Z_cpu.npy")
    print("[info] Using full 3D grids: X_cpu.npy, Y_cpu.npy, Z_cpu.npy")

    x = X_cpu[:, 0, 0]
    y = Y_cpu[0, :, 0]
    z = Z_cpu[0, 0, :]

    X_xy, Y_xy = X_cpu[:, :, mid_z], Y_cpu[:, :, mid_z]
    X_xz, Z_xz = X_cpu[:, mid_y, :], Z_cpu[:, mid_y, :]
    Y_yz, Z_yz = Y_cpu[mid_x, :, :], Z_cpu[mid_x, :, :]

    HAVE_FULL_3D_MESH = True

else:
    print("[warn] No saved grids found; reconstructing 1D grids from constants.")
    x = np.arange(Nx) * hx
    y = np.arange(Ny) * hy
    z = np.arange(Nz) * hz

    X_xy, Y_xy = np.meshgrid(x, y, indexing="ij")
    X_xz, Z_xz = np.meshgrid(x, z, indexing="ij")
    Y_yz, Z_yz = np.meshgrid(y, z, indexing="ij")

    HAVE_FULL_3D_MESH = False
    X_cpu = Y_cpu = Z_cpu = None

print(f"[info] Grid: Nx={Nx}, Ny={Ny}, Nz={Nz}")
print(f"[info] Mid indices: mid_x={mid_x}, mid_y={mid_y}, mid_z={mid_z}")


# ──────────────────────────────────────────────────────────────
# Per-time-step probability-density volumes
# ──────────────────────────────────────────────────────────────

pat = re.compile(r"rho_prob_t([0-9]+(?:\.\d+)?)\.npy$")

all_files = sorted(
    (
        Path(p) for p in glob.glob(str(data_dir / "rho_prob_t*.npy"))
        if pat.search(Path(p).name)
    ),
    key=lambda p: float(pat.search(p.name).group(1))
)

if not all_files:
    print("No rho_prob_t*.npy files found – nothing to plot.")
    raise SystemExit

SNAP_DT = 0.1
TOL = 1e-6

files = []
for f in all_files:
    m = pat.search(f.name)
    t = float(m.group(1))
    n_snap = round(t / SNAP_DT)
    if abs(t - n_snap * SNAP_DT) < TOL:
        files.append(f)

if not files:
    print(f"[warn] No files near multiples of {SNAP_DT} found; falling back to all files.")
    files = all_files

print(f"[info] Will plot {len(files)} snapshots (step ~ {SNAP_DT})")

ISO_LEVEL = 0.05
SMOOTH_SIGMA = 1.0


for f in files:
    m = pat.search(f.name)
    timestr = m.group(1)
    t_label = timestr.rstrip("0").rstrip(".")
    t = float(timestr)

    print(f"\n== t = {timestr} =============================")

    midplanes_file  = plot_dir / f"midplanes_t{timestr}.png"
    contour_file    = plot_dir / f"contour_xy_t{timestr}.png"
    scatter_file    = plot_dir / f"scatter_t{timestr}.png"
    isosurf_file    = plot_dir / f"isosurf_t{timestr}.png"
    slices_bar_file = plot_dir / f"slices_bar_t{timestr}.png"

    slices_file = data_dir / f"prob_slices_t{timestr}.npy"

    required_outputs = [midplanes_file, contour_file, scatter_file, isosurf_file, slices_bar_file]

    if all(p.exists() for p in required_outputs):
        print(" Plots already present – skipping.")
        continue

    rho_prob = np.load(f, mmap_mode="r")

    # --- compute probability in each z-slice if not already saved ---
    if slices_file.exists():
        prob_slices = np.load(slices_file)
    else:
        # Probability mass in each z-slab:
        # P_k ≈ sum_{i,j} rho[i,j,k] * hx * hy * hz
        prob_slices = np.sum(rho_prob, axis=(0, 1), dtype=np.float64) * hx * hy * hz
        np.save(slices_file, prob_slices)
        print(" Saved prob_slices_t*.npy")

    # slices for 2-D views
    rho_xy = np.clip(rho_prob[:, :, mid_z], 1e-10, None)
    rho_xz = np.clip(rho_prob[:, mid_y, :], 1e-10, None)
    rho_yz = np.clip(rho_prob[mid_x, :, :], 1e-10, None)

    # --- mid-plane panel ---
    if not midplanes_file.exists():
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
        cmargs = dict(levels=30, norm=LogNorm(), cmap="viridis")

        axs[0].contourf(X_xy, Y_xy, rho_xy, **cmargs)
        axs[0].set(xlabel="x", ylabel="y", title=f"z = {z[mid_z]:.3f}")

        axs[1].contourf(X_xz, Z_xz, rho_xz, **cmargs)
        axs[1].set(xlabel="x", ylabel="z", title=f"y = {y[mid_y]:.3f}")

        axs[2].contourf(Y_yz, Z_yz, rho_yz, **cmargs)
        axs[2].set(xlabel="y", ylabel="z", title=f"x = {x[mid_x]:.3f}")

        fig.suptitle(f"τ = {t_label}")
        fig.savefig(midplanes_file, dpi=300)
        plt.close(fig)
        print(" Mid-plane panel saved.")

    # --- XY contour at z = mid_z ---
    if not contour_file.exists():
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        pcm = ax.pcolormesh(
            X_xy, Y_xy, rho_xy,
            shading="auto",
            norm=LogNorm(),
            cmap="viridis"
        )
        ax.set(
            xlim=(0, Lx), ylim=(0, Ly),
            xlabel="x", ylabel="y",
            title=rf"$\rho_{{xy}}$ at slice $z={z[mid_z]:.3f}$ and $\tau={t_label}$"
        )
        fig.colorbar(pcm, ax=ax, label="|ψ|²")
        fig.savefig(contour_file, dpi=300)
        plt.close(fig)
        print(" XY contour saved.")

    # --- 3-D scatter (≥30% max) ---
    if not scatter_file.exists():
        fig = plt.figure(figsize=(7, 6), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        threshold = float(rho_prob.max()) * 0.3
        idx = np.argwhere(rho_prob > threshold)

        if idx.size:
            if len(idx) > 100_000:
                sel = np.random.choice(len(idx), size=100_000, replace=False)
                idx = idx[sel]

            ix = idx[:, 0]
            iy = idx[:, 1]
            iz = idx[:, 2]

            vals = rho_prob[ix, iy, iz]

            ax.scatter(
                x[ix], y[iy], z[iz],
                c=vals, cmap="viridis", s=1, alpha=0.05
            )

        ax.set(
            xlim=(0, Lx), ylim=(0, Ly), zlim=(0, Lz),
            xlabel="x", ylabel="y", zlabel="z",
            title=rf"Scatter ≥30 % of $\max|\psi|^2$ at $\tau={t_label}$"
        )
        fig.savefig(scatter_file, dpi=300)
        plt.close(fig)
        print(" Scatter saved.")

    # --- Iso-surface at fixed level (fraction of max after smoothing) ---
    if not isosurf_file.exists():
        rho_sm = gaussian_filter(np.asarray(rho_prob), sigma=SMOOTH_SIGMA)
        rho_max = float(rho_sm.max())

        if rho_max > 0.0:
            rho_n = rho_sm / rho_max

            if np.min(rho_n) < ISO_LEVEL < np.max(rho_n):
                verts, faces, *_ = measure.marching_cubes(rho_n, level=ISO_LEVEL)

                verts[:, 0] *= hx
                verts[:, 1] *= hy
                verts[:, 2] *= hz

                fig = plt.figure(figsize=(7, 6), tight_layout=True)
                ax = fig.add_subplot(111, projection="3d")

                mesh = Poly3DCollection(
                    verts[faces],
                    alpha=0.2,
                    facecolor="cyan",
                    edgecolor="k",
                    linewidths=0.1
                )
                ax.add_collection3d(mesh)

                ax.set(
                    xlim=(0, Lx), ylim=(0, Ly), zlim=(0, Lz),
                    xlabel="x", ylabel="y", zlabel="z",
                    title=(
                        rf"Iso-surface at {ISO_LEVEL*100:.0f}% of "
                        rf"$\max|\psi|^2$ and $\tau={t_label}$"
                    ),
                )

                fig.savefig(isosurf_file, dpi=300)
                plt.close(fig)
                print(" Iso-surface saved.")
            else:
                print(" Iso-surface skipped – threshold not met.")
        else:
            print(" Iso-surface skipped – zero density.")

    # --- slice bar: probability in each z-slice ---
    if not slices_bar_file.exists():
        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

#        ax.bar(z, prob_slices, width=0.9*hz)
#        ax.set(
#            xlabel="z",
#            ylabel="Probability",
#            title=rf"Probability per z-Slice for $\rho = |\psi_\uparrow|^2 + |\psi_\downarrow|^2,\ \tau={t_label}$"
#        )

        slice_idx = np.arange(len(prob_slices))
        ax.bar(slice_idx, prob_slices)
        ax.set(
            xlabel="Slice (increasing z)",
            ylabel="Probability",
            title=rf"Probability per z-Slice for $\rho = |\psi_\uparrow|^2 + |\psi_\downarrow|^2,\ \tau={t_label}$"
        )
        ax.grid(True, axis="y")

        fig.savefig(slices_bar_file, dpi=300)
        plt.close(fig)
        print(" Slice probs bar saved.")

print("\nAll plots written to", plot_dir)


# ──────────────────────────────────────────────────────────────
# GIF generation for each plot type
# ──────────────────────────────────────────────────────────────

def create_gif(pattern: str, gif_name: str, duration: float = 0.5):
    """Collect PNGs matching pattern, sort by time, and build a GIF."""
    png_files = list(plot_dir.glob(pattern))
    if not png_files:
        print(f"[gif] No files for pattern '{pattern}' – skipping {gif_name}")
        return

    def time_from_name(p: Path) -> float:
        m = re.search(r"_t([0-9.]+)\.png$", p.name)
        return float(m.group(1)) if m else 0.0

    png_files = sorted(png_files, key=time_from_name)

    images = [imageio.imread(p) for p in png_files]
    out_path = plot_dir / gif_name
    imageio.mimsave(out_path, images, duration=duration)
    print(f"[gif] Saved {gif_name} with {len(images)} frames")


create_gif("midplanes_t*.png",  "midplanes.gif",  duration=0.4)
create_gif("contour_xy_t*.png", "contour_xy.gif", duration=0.4)
create_gif("scatter_t*.png",    "scatter.gif",    duration=0.4)
create_gif("isosurf_t*.png",    "isosurf.gif",    duration=0.4)
create_gif("slices_bar_t*.png", "slices_bar.gif", duration=0.4)

print(f"\nAll plots and GIFs are in {plot_dir}")


# %% cell 1


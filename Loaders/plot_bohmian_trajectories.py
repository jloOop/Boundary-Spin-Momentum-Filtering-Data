
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pick N trajectories and analyze/plot them.

Supports (priority order):
  1) NEW SELECTED:
     - bohmian_traj_selected.npy (Nsteps, Ksel, 3)
     - bohm_times.npy (Nsteps,)
     - traj_indices.npy (Ksel,)                [optional but recommended]
     - bohm_arrived_mask_selected.npy (Ksel,)  [optional]
     - bohm_t_hit_selected.npy (Ksel,)         [optional]
     If selected-arrival files are missing but FULL arrivals exist,
     we'll map via traj_indices.npy.

  2) OLD SUBSET:
     - traj_subset.npy (Nsnap, K0, 3)
     - traj_subset_times.npy (Nsnap,)
     - traj_subset_idx.npy (K0,)
     - (optional) bohm_arrived_mask.npy (Mpart,), bohm_t_hit.npy (Mpart,)
       (we’ll map using traj_subset_idx.npy)

  3) FULL:
     - bohmian_traj.npy (Nsteps, Mpart, 3)
     - bohm_times.npy (Nsteps,)
     - (optional) bohm_arrived_mask.npy (Mpart,), bohm_t_hit.npy (Mpart,)
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import to_rgba

try:
    from trajectory_selection import pick_indices as select_trajectory_indices
except ImportError:
    from Loaders.trajectory_selection import pick_indices as select_trajectory_indices

try:
    from trajectory_data import load_trajectory_data
except ImportError:
    from Loaders.trajectory_data import load_trajectory_data

try:
    from trajectory_grid import load_trajectory_grid
except ImportError:
    from Loaders.trajectory_grid import load_trajectory_grid

try:
    from trajectory_view import prepare_selected_trajectory_view
except ImportError:
    from Loaders.trajectory_view import prepare_selected_trajectory_view

try:
    from trajectory_stats import format_selected_hit_time_stats, format_selected_pick_message
except ImportError:
    from Loaders.trajectory_stats import format_selected_hit_time_stats, format_selected_pick_message

try:
    from trajectory_plots import (
        create_trajectory_3d_figure,
        create_trajectory_projection_figure,
    )
except ImportError:
    from Loaders.trajectory_plots import (
        create_trajectory_3d_figure,
        create_trajectory_projection_figure,
    )

def main(argv=None):
    # ---- global plot style: default with white background ----
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "#fef7e7" #"lightgray" "f5f0e6" "fef7e7" "fdf6e3"
    plt.rcParams["axes.facecolor"] = "#fef7e7"

    # -------------------- args --------------------
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=1, help="number of particles to plot/analyze")
    p.add_argument("--mode", type=str, default="random",
                   choices=["random", "arrived", "unarrived", "earliest", "latest", "spread"],
                   help="how to select the K particles")
    p.add_argument("--seed", type=int, default=0, help="rng seed for random selection")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[warn] ignoring unknown args: {unknown}")
    base = Path(".").resolve()

    # -------------------- constants / grid --------------------
    grid = load_trajectory_grid(base)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    hx, hy, hz = grid.hx, grid.hy, grid.hz
    Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz
    z_det = grid.z_det  # detector ON the grid (matches solver)
    print(f"[info] grid=({Nx},{Ny},{Nz}), h=({hx:.4g},{hy:.4g},{hz:.4g}), L=({Lx},{Ly},{Lz})")
    print(f"[info] detector plane at z_det={z_det:.6f} (index k={Nz-1})")

    # -------------------- trajectory data source --------------------
    trajectory_data = load_trajectory_data(base)
    TRAJ = trajectory_data.trajectories
    T = trajectory_data.times
    idx_map = trajectory_data.idx_map
    label = trajectory_data.label
    arrived_vis = trajectory_data.arrived_vis
    t_hit_vis = trajectory_data.t_hit_vis

    Nsnap, Mvis = TRAJ.shape[0], TRAJ.shape[1]
    print(f"[info] loaded {label} with {Nsnap} frames and pool size {Mvis}")

    # -------------------- selection helpers --------------------
    rng = np.random.default_rng(args.seed)
    sel = select_trajectory_indices(
        args.mode,
        args.K,
        Mvis,
        arrived=arrived_vis,
        t_hit=t_hit_vis,
        rng=rng,
    )

    # Map to global indices (if we know the mapping)
    global_sel, pick_message = format_selected_pick_message(args.mode, sel, idx_map)
    print(pick_message)

    # quick stats
    hit_time_stats_message = format_selected_hit_time_stats(t_hit_vis, sel)
    if hit_time_stats_message is not None:
        print(hit_time_stats_message)

    XYZ, X, Y, Z, Tmin, Tmax, arr_sel, hit_sel = prepare_selected_trajectory_view(
        TRAJ,
        T,
        sel,
        arrived_vis=arrived_vis,
        t_hit_vis=t_hit_vis,
    )

    # -------------------- 3D plot --------------------
    fig = create_trajectory_3d_figure(X, Y, Z, Lx, Ly, Lz, z_det, arr_sel=arr_sel)

    fig.savefig(base / "traj_sel_3D.png", dpi=220)
    plt.close(fig)
    print("[save] traj_sel_3D.png")



    # -------------------- 2D projections: XY, XZ, YZ --------------------
    fig = create_trajectory_projection_figure(X, Y, Z, Lx, Ly, Lz, z_det, arr_sel=arr_sel)
    plt.show()
    fig.savefig(base / "traj_sel_projections.png", dpi=220)
    plt.close(fig)
    print("[save] traj_sel_projections.png")


if __name__ == "__main__":
    main()

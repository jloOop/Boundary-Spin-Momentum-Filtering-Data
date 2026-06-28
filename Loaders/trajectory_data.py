"""Trajectory data-source detection and loading helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrajectoryData:
    """Loaded trajectory arrays and metadata for plotting/selection."""

    trajectories: np.ndarray
    times: np.ndarray
    idx_map: Optional[np.ndarray]
    label: str
    arrived_vis: Optional[np.ndarray]
    t_hit_vis: Optional[np.ndarray]


def load_trajectory_data(base):
    """Load trajectories using the selected, subset, then full priority order."""
    base = Path(base)

    arrived_full = None
    t_hit_full = None
    if (base / "bohm_arrived_mask.npy").exists():
        arrived_full = np.load(base / "bohm_arrived_mask.npy").astype(bool)
    if (base / "bohm_t_hit.npy").exists():
        t_hit_full = np.load(base / "bohm_t_hit.npy")

    label = ""
    idx_map = None

    sel_new = (base / "bohmian_traj_selected.npy").exists()
    sel_old = (base / "traj_subset.npy").exists()

    if sel_new:
        trajectories = np.load(base / "bohmian_traj_selected.npy", mmap_mode="r")
        times = (
            np.load(base / "bohm_times.npy")
            if (base / "bohm_times.npy").exists()
            else np.arange(trajectories.shape[0])
        )
        arrived_sel = (
            np.load(base / "bohm_arrived_mask_selected.npy").astype(bool)
            if (base / "bohm_arrived_mask_selected.npy").exists()
            else None
        )
        t_hit_sel = (
            np.load(base / "bohm_t_hit_selected.npy")
            if (base / "bohm_t_hit_selected.npy").exists()
            else None
        )
        if (base / "traj_indices.npy").exists():
            idx_map = np.load(base / "traj_indices.npy")
        label = f"selected (new) Ksel={trajectories.shape[1]}"

        if (
            (arrived_sel is None or t_hit_sel is None)
            and (arrived_full is not None or t_hit_full is not None)
            and (idx_map is not None)
        ):
            if arrived_full is not None:
                arrived_sel = arrived_full[idx_map]
            if t_hit_full is not None:
                t_hit_sel = t_hit_full[idx_map]

        arrived_vis = arrived_sel
        t_hit_vis = t_hit_sel

    elif sel_old:
        trajectories = np.load(base / "traj_subset.npy", mmap_mode="r")
        times = np.load(base / "traj_subset_times.npy")
        idx_map = np.load(base / "traj_subset_idx.npy")
        label = f"subset (old) K0={trajectories.shape[1]}"

        arrived_vis = arrived_full[idx_map] if arrived_full is not None else None
        t_hit_vis = t_hit_full[idx_map] if t_hit_full is not None else None

    else:
        full_path = base / "bohmian_traj.npy"
        time_path = base / "bohm_times.npy"
        if not (full_path.exists() and time_path.exists()):
            raise FileNotFoundError(
                "Need one of:\n"
                "  NEW selected: bohmian_traj_selected.npy (+ bohm_times.npy)\n"
                "  OLD subset:   traj_subset.npy + traj_subset_times.npy + traj_subset_idx.npy\n"
                "  FULL:         bohmian_traj.npy + bohm_times.npy"
            )
        trajectories = np.load(full_path, mmap_mode="r")
        times = np.load(time_path)
        idx_map = np.arange(trajectories.shape[1], dtype=int)
        label = f"full M={trajectories.shape[1]}"

        arrived_vis = arrived_full
        t_hit_vis = t_hit_full

    return TrajectoryData(
        trajectories=trajectories,
        times=times,
        idx_map=idx_map,
        label=label,
        arrived_vis=arrived_vis,
        t_hit_vis=t_hit_vis,
    )

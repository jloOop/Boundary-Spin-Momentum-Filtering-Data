"""Trajectory array preparation helpers for plotting."""


def prepare_selected_trajectory_view(TRAJ, T, sel, arrived_vis=None, t_hit_vis=None):
    """Slice trajectories and optional arrival metadata for selected indices."""
    XYZ = TRAJ[:, sel, :]
    X = XYZ[..., 0]
    Y = XYZ[..., 1]
    Z = XYZ[..., 2]
    Tmin, Tmax = T[0], T[-1]

    arr_sel = arrived_vis[sel] if arrived_vis is not None else None
    hit_sel = t_hit_vis[sel] if t_hit_vis is not None else None

    return XYZ, X, Y, Z, Tmin, Tmax, arr_sel, hit_sel

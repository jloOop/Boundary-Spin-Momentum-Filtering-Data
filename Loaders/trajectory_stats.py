"""Pure statistics helpers for selected Bohmian trajectories."""

import numpy as np


def format_selected_pick_message(mode, sel, idx_map):
    """Return the global selection and pick message for selected trajectories."""
    if idx_map is not None:
        global_sel = idx_map[sel]
        message = (
            f"[pick] mode={mode}, K={sel.size} \u2192 global ids: "
            f"{global_sel[:min(10, sel.size)]}" + (" ..." if sel.size > 10 else "")
        )
    else:
        global_sel = sel
        message = f"[pick] mode={mode}, K={sel.size}"

    return global_sel, message


def format_selected_hit_time_stats(t_hit_vis, sel):
    """Return the selected hit-time stats message, or None if nothing is printed."""
    if (t_hit_vis is None) or (sel.size == 0):
        return None

    selected_hits = t_hit_vis[sel]
    finite_hits = selected_hits[np.isfinite(selected_hits)]
    if finite_hits.size > 0:
        return (
            f"[pick] hit-time stats (selected): mean={finite_hits.mean():.4g}, "
            f"p10={np.percentile(finite_hits, 10):.4g}, "
            f"p50={np.percentile(finite_hits, 50):.4g}, "
            f"p90={np.percentile(finite_hits, 90):.4g}"
        )

    return "[pick] none of the selected have finite t_hit"

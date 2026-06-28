"""Trajectory index selection helpers for Bohmian trajectory plotting."""

import numpy as np


def pick_indices(mode: str, K: int, pool_size: int, arrived=None, t_hit=None, rng=None) -> np.ndarray:
    """Pick visible trajectory indices without reading files or plotting.

    This preserves the selection behavior from ``plot_bohmian_trajectories.py``
    while making all formerly global inputs explicit.
    """
    if rng is None:
        rng = np.random.default_rng()

    pool = np.arange(pool_size)

    # If we lack arrival info, fall back to random.
    if arrived is None or t_hit is None:
        if K >= len(pool):
            return pool
        return rng.choice(pool, K, replace=False)

    A = np.asarray(arrived).astype(bool)
    H = np.asarray(t_hit)

    if mode == "random":
        return pool if K >= len(pool) else rng.choice(pool, K, replace=False)

    if mode == "arrived":
        cand = np.where(A)[0]
        return cand if K >= cand.size else rng.choice(cand, K, replace=False)

    if mode == "unarrived":
        cand = np.where(~A)[0]
        return cand if K >= cand.size else rng.choice(cand, K, replace=False)

    if mode == "earliest":
        cand = np.where(A & np.isfinite(H))[0]
        if cand.size == 0:
            return np.array([], dtype=int)
        order = np.argsort(H[cand])
        return cand[order[:min(K, cand.size)]]

    if mode == "latest":
        cand = np.where(A & np.isfinite(H))[0]
        if cand.size == 0:
            return np.array([], dtype=int)
        order = np.argsort(H[cand])[::-1]
        return cand[order[:min(K, cand.size)]]

    if mode == "spread":
        cand = np.where(A & np.isfinite(H))[0]
        if cand.size == 0:
            return np.array([], dtype=int)
        q = np.linspace(0.0, 1.0, K, endpoint=False) + 0.5 / K
        qs = np.quantile(H[cand], q)
        sel = []
        used = set()
        for target in qs:
            j = cand[np.argmin(np.abs(H[cand] - target))]
            if j in used:
                pool2 = [x for x in cand if x not in used]
                if not pool2:
                    break
                j = rng.choice(pool2)
            sel.append(j)
            used.add(j)
        return np.array(sel, dtype=int)

    # Fallback matches the previous script behavior for unexpected modes.
    return pool if K >= len(pool) else rng.choice(pool, K, replace=False)

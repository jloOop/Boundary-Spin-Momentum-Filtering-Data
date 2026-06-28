"""Pure bookkeeping helpers for solver time-series output."""

import numpy as np


def build_time_series_bookkeeping(T_final, dt, probability_samples=1000, view_samples=10):
    """Build time-step schedules and empty probability history containers."""
    num_steps = int(round(T_final / dt))
    prob_steps_cpu = set(np.linspace(1, num_steps, probability_samples, dtype=int).tolist())
    view_steps_cpu = set(np.linspace(1, num_steps, view_samples, dtype=int).tolist())
    total_probs = []
    prob_times = []

    return num_steps, prob_steps_cpu, view_steps_cpu, total_probs, prob_times


def build_probability_history_arrays(prob_times, total_probs):
    """Build NumPy arrays for saved probability history."""
    return np.array(prob_times), np.array(total_probs)

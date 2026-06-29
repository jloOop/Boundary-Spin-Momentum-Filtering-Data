# Post-processing loaders

This folder contains the public-facing post-processing layer for the spinor absorbing-boundary-condition simulations associated with:

> **A. Jozani, _Spin–Momentum Impedance and Filtering by a Spin-Coupled Absorbing Boundary Condition_**

The loaders do not solve the TDSE/Pauli equation and do not assemble the Crank–Nicolson/GMRES matrices. The heavy numerical work is done by the scripts in [`../Solvers/`](../Solvers/). This folder loads completed solver outputs and converts them into figure-level quantities, animations, summaries, and trajectory visualizations.

## Folder role

```text
Solvers/ -> run GPU/CuPy TDSE simulations and write arrays
Loaders/ -> load arrays and build figures, GIFs, summaries, trajectory plots
data/    -> store selected compact data and representative media
paper/   -> manuscript / preprint citation context
```

The main physical observable is the detector-present roof flux, equivalently the norm loss of the non-unitary spinor-ABC evolution. Bohmian trajectory histograms, when shown, are Monte Carlo samples of the same detector-present flux law. They should not be described as detector-free arrival-time predictions.

## Loader map

| Loader | Main role | Paper / repository connection |
|---|---|---|
| `plot_detection_time_distribution.py` | Reconstructs survival, roof flux, detected fraction, restricted mean, and optional Bohmian histogram overlay | Closest loader to the main detection-time figures |
| `make_density_gifs.py` | Builds PNG frames and GIFs from saved density snapshots | Qualitative visualization of propagation, confinement, absorption, and delayed structure |
| `plot_bohmian_trajectories.py` | Plots selected Pauli-current trajectories in 3D and projections | Trajectory visualization companion |
| `trajectory_*.py` | Utilities for trajectory selection, plotting, statistics, grid handling, and viewing | Support layer for trajectory visualization |

Each loader may also have a notebook version for interactive inspection. The Python exports are easier to skim on GitHub; notebooks are better for reproducing figures interactively.

## Recommended reading order

1. `plot_detection_time_distribution.py`
2. `make_density_gifs.py`
3. `plot_bohmian_trajectories.py`
4. `trajectory_*.py` utilities as needed

## Required inputs

Typical completed solver outputs include:

```text
constants.npz
prob_times.npy
total_probs.npy
rho_prob_t*.npy
x.npy
y.npy
z.npy
```

Trajectory post-processing may require:

```text
bohm_t_hit.npy
bohm_arrived_mask.npy
bohm_t_hit_selected.npy
bohm_arrived_mask_selected.npy
bohmian_traj_selected.npy
bohmian_traj.npy
bohm_times.npy
traj_indices.npy
```

Large raw run directories should remain in `runs/`, scratch storage, or external archive storage. Commit only selected compact arrays, final figures, GIF links, and documented summaries.

## Main quantitative loader

`plot_detection_time_distribution.py` loads the compact norm-monitoring output:

```text
prob_times.npy
total_probs.npy
constants.npz
```

and reconstructs

```math
S(t;\omega)=\|\Psi_t\|^2,
\qquad
g(t;\omega)=-\frac{dS}{dt},
\qquad
D_T(\omega)=1-S(T),
\qquad
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt.
```

When trajectory hit-time arrays are available, the loader can overlay a gray Bohmian histogram on top of the flux curve. The correct interpretation is:

```text
blue curve = detector-present roof-flux / norm-loss density
gray bars  = Monte Carlo trajectory samples of the same detector-present law
red curve  = detector-free Gaussian comparison current, not a detector model and not a fit input
```

## GIF and trajectory loaders

`make_density_gifs.py` is mainly qualitative. It helps a reader see the time evolution of the 3D probability density, confined transverse profile, propagation toward the roof, absorption, storage near the roof, and delayed oscillatory structure.

`plot_bohmian_trajectories.py` provides trajectory-level visualization of the same detector-present Pauli-current law. It is not evidence for a detector-free arrival-time proposal.

## Interpretation rules

- `total_probs.npy` is a survival curve, not a detection-time histogram.
- `g(t;omega)` is obtained by differentiating survival, so smoothing or the manuscript plotting pipeline may be required for publication-quality curves.
- `mu*(T;omega)` is a finite-window statistic.
- Trajectory histograms can be normalized conditionally over detected trajectories or over all sampled initial particles; the interpretation changes accordingly.

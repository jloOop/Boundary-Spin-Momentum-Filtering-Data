# Post-processing loaders

This folder contains the three GitHub-facing post-processing notebooks for the spinor-ABC simulation runs.  The name "loader" is kept because these notebooks load completed solver outputs, but the public-facing role is broader: they are the analysis and visualization layer between the GPU/HPC solvers and the figures, GIFs, and trajectory illustrations shown in the repository.

The notebooks do **not** run the TDSE solver. They assume that a solver run has already written arrays such as `constants.npz`, `prob_times.npy`, `total_probs.npy`, density snapshots, and optional Bohmian trajectory arrays.

## Recommended names

| New notebook name | Main role | Paper/repository connection |
|---|---|---|
| `make_density_gifs.ipynb` | Build PNG frames and GIFs from saved density snapshots `rho_prob_t*.npy`. | Repository media and representative density animations. |
| `plot_detection_time_distribution.ipynb` | Reconstruct survival, roof-flux density, finite-window restricted mean, trajectory histogram overlay, and summary files. | Closest post-processing notebook for Letter Fig. 1 and Supplement S2--S4. |
| `plot_bohmian_trajectories.ipynb` | Plot selected Pauli-current trajectories in 3D and as XY/XZ/YZ projections. | Trajectory visualization companion; in this project, trajectories are samples of the detector-present flux law. |


## Notebook roles

### `make_density_gifs.ipynb`

Use this notebook when a solver run saved density snapshots such as

```text
rho_prob_t0.10000.npy
rho_prob_t0.20000.npy
...
```

It creates frame images and GIFs:

```text
plots_general3/midplanes_t*.png
plots_general3/contour_xy_t*.png
plots_general3/scatter_t*.png
plots_general3/isosurf_t*.png
plots_general3/slices_bar_t*.png
plots_general3/midplanes.gif
plots_general3/contour_xy.gif
plots_general3/scatter.gif
plots_general3/isosurf.gif
plots_general3/slices_bar.gif
```

This notebook is mainly for visual communication: propagation, confinement, absorption, delayed density structure, and qualitative 3D wave-packet behavior.

### `plot_detection_time_distribution.ipynb`

Use this notebook for the main detection-time figure built from the compact solver outputs

```text
prob_times.npy
total_probs.npy
constants.npz
```

and, when available,

```text
bohm_t_hit.npy
bohm_arrived_mask.npy
bohm_t_hit_selected.npy
bohm_arrived_mask_selected.npy
bohmian_traj_selected.npy
bohm_times.npy
```

It reconstructs the survival curve

```text
S(t; omega) = ||Psi_t||^2
```

and the detector-present flux density

```text
g(t; omega) = -dS(t; omega)/dt
```

then computes the finite-window restricted mean

```text
mu*(T; omega) = integral_0^T S(t; omega) dt.
```

It also writes reusable summary files:

```text
arrival_stats_summary.npz
arrival_stats_summary.txt
combined_arrival2.png
combined_arrival2.pdf
```

This is the most important loader for the PRL-style spin--momentum filtering repository, because it connects the raw norm-loss arrays to the figure-level detector-time plot.

### `plot_bohmian_trajectories.ipynb`

Use this notebook to inspect individual trajectory geometry. It supports three storage layouts:

```text
# New selected trajectory format
bohmian_traj_selected.npy
bohm_times.npy
traj_indices.npy

# Older subset format
traj_subset.npy
traj_subset_times.npy
traj_subset_idx.npy

# Full trajectory format
bohmian_traj.npy
bohm_times.npy
```

The notebook can choose trajectories randomly or by hit-time status:

```bash
jupyter nbconvert --execute loaders/plot_bohmian_trajectories.ipynb -- --K 5 --mode spread --seed 1
```

It saves

```text
traj_sel_3D.png
traj_sel_projections.png
```

In the spinor-ABC repository, describe these trajectories carefully: they are visualization/sampling tools for the detector-present Pauli-current law, not a separate detector-free arrival-time law.

## How to run

The simplest workflow is to copy or open the notebooks from inside a completed run directory:

```bash
cd runs/<one-completed-run>
jupyter lab ../../loaders/plot_detection_time_distribution.ipynb
```

For the GIF notebook, running from a subfolder is also supported because it searches upward until it finds `constants.npz`.

For scripted execution, use:

```bash
jupyter nbconvert --to notebook --execute loaders/plot_detection_time_distribution.ipynb   --output executed_detection_time.ipynb
```

Large trajectory or density files should stay in `runs/` or external scratch storage. Commit only selected compact arrays, final figures, and the notebooks themselves.

## Dependencies

The loaders are CPU-side post-processing notebooks. They do not require CuPy unless you are re-running the solver. Typical dependencies are:

```bash
pip install numpy scipy matplotlib imageio scikit-image notebook
```

## What this folder demonstrates

For a scientific/industry reviewer, this folder shows the analysis layer after a large GPU simulation: robust loading of legacy and current output formats, survival/flux post-processing, finite-window statistics, Monte Carlo trajectory summaries, 3D visualization, GIF generation, and publication-quality plotting.

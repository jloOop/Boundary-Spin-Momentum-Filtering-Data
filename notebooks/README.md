# Notebooks

These notebooks are post-processing and visualization loaders. They assume that a simulation output directory already contains files such as `constants.npz`, `prob_times.npy`, `total_probs.npy`, `rho_prob_t*.npy`, and selected trajectory arrays.

## `Spin_Gif.ipynb`

Creates per-time density PNGs and GIFs from `rho_prob_t*.npy` files.

Expected inputs:

```text
constants.npz
rho_prob_t*.npy
x.npy, y.npy, z.npy      optional
```

Typical outputs:

```text
midplanes_t*.png
contour_xy_t*.png
scatter_t*.png
isosurf_t*.png
slices_bar_t*.png
midplanes.gif
contour_xy.gif
scatter.gif
isosurf.gif
slices_bar.gif
```

## `Paper_Presentation2.ipynb`

Creates arrival-time and roof-flux plots from saved probability and trajectory arrays.

Expected inputs:

```text
constants.npz
prob_times.npy
total_probs.npy
bohm_t_hit.npy
bohm_arrived_mask.npy
bohmian_traj_selected.npy
bohm_times.npy
```

Typical outputs:

```text
combined_arrival2.png
sample_traj_z_vs_t.png
arrival_stats_summary.npz
plots_compare/total_probability_vs_time.png
plots_compare/loss_vs_theory_raw.png
```

## `Traj3D.ipynb`

Plots selected 3D Bohmian trajectories and projections.

Expected inputs:

```text
bohmian_traj_selected.npy
bohm_times.npy
traj_indices.npy
bohm_arrived_mask_selected.npy
bohm_t_hit_selected.npy
```

Typical outputs:

```text
traj_sel_3D.png
traj_sel_projections.png
```

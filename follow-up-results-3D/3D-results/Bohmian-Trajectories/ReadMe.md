# Bohmian trajectories

This folder is for selected Bohmian trajectory visualizations and trajectory-related PNGs/GIFs.

The trajectories here are used as a Monte Carlo representation of the same detector-present spinor-ABC flux law. They are not a separate no-detector first-arrival proposal.

Generator:

```text
notebooks/Traj3D.ipynb
```

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
sample_traj_z_vs_t.png
```

Recommended layout:

```text
Bohmian-Trajectories/
  omega_001/
    traj_sel_3D.png
    traj_sel_projections.png
  omega_300/
    traj_sel_3D.png
    traj_sel_projections.png
```

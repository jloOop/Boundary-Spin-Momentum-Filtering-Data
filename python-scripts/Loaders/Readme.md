# Loader role map

This document links each post-processing notebook to the representative solver outputs.

```text
solver_spinor_abc_gaussian.py
        ↓ writes prob_times.npy, total_probs.npy, constants.npz,
          optional rho_prob_t*.npy and Bohmian arrays

loaders/plot_detection_time_distribution.ipynb
        ↓ reconstructs S(t), g(t), D_T, mu*(T), combined figure, summary files

loaders/make_density_gifs.ipynb
        ↓ turns rho_prob_t*.npy into PNG frames and GIFs

loaders/plot_bohmian_trajectories.ipynb
        ↓ inspects selected Pauli-current trajectories in 3D and projections
```

## Naming principle

The filenames use verbs:

- `make_...` for notebooks that generate media.
- `plot_...` for notebooks that turn arrays into figures.
- `detection_time_distribution` rather than `arrival_time` for the main PRL-style spinor-ABC repository, because the observable is the detector-present roof-flux/norm-loss law.

## GitHub placement

Recommended structure:

```text
solvers/
  solver_spinor_abc_gaussian.py
  diagnose_reflection_time_Decomposition.py
  diagnose_boundary_symbol.py

loaders/
  README.md
  make_density_gifs.ipynb
  plot_detection_time_distribution.ipynb
  plot_bohmian_trajectories.ipynb

figures/
  selected final PNG/PDF figures only

runs/
  local or scratch outputs; do not commit
```

## Commit guidance

Do commit:

```text
loaders/*.ipynb
loaders/*.py       # optional raw script exports, useful for quick code review
loaders/README.md
```

Do not commit full run outputs by default:

```text
runs/
*.npy
*.npz
*.gif
*.png generated in bulk
```

Commit only selected small arrays and selected final figures that are needed for reproducibility.

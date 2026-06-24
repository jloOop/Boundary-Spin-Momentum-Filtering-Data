# Post-processing loaders

This folder contains the public-facing post-processing layer for the spinor absorbing-boundary-condition (spinor-ABC) simulations in

> **A. Jozani, _Spin–Momentum Impedance and Filtering by a Spin-Coupled Absorbing Boundary Condition_**

and its Supplemental Material.

The word **loader** is kept because these files load completed solver outputs. Their broader role is to turn raw GPU/HPC run products into figure-level quantities, animations, and trajectory visualizations. They do **not** solve the TDSE/Pauli equation and do **not** assemble the Crank--Nicolson/GMRES matrices. The heavy numerical work is done by the scripts in `../Solvers/`; this folder is the analysis and visualization layer.

The main physical observable is the **detector-present roof flux**, equivalently the **norm loss** of the non-unitary spinor-ABC evolution. Bohmian trajectory histograms, when shown, are Monte Carlo samples of the same detector-present flux law. They should not be described as a separate detector-free arrival-time proposal.

---

## Folder role

A typical workflow is:

```text
Solvers/      -> run the GPU/CuPy TDSE simulation and write arrays
Loaders/      -> load those arrays and build figures, GIFs, summaries, and trajectory plots
data/         -> selected compact data and representative media for GitHub
paper/        -> manuscript / Supplement information when publicly available
```

The loaders assume that a completed run directory already contains arrays such as

```text
constants.npz
prob_times.npy
total_probs.npy
rho_prob_t*.npy
x.npy, y.npy, z.npy
```

and, when trajectory post-processing is used,

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

---

## Loader map

| Loader | Main role | Paper / repository connection |
|---|---|---|
| `make_density_gifs.ipynb` / `make_density_gifs.py` | Builds PNG frames and GIFs from saved density snapshots such as `rho_prob_t*.npy`. | Repository media and qualitative 3D visualization of propagation, confinement, absorption, and delayed density structure. |
| `plot_detection_time_distribution.ipynb` / `plot_detection_time_distribution.py` | Reconstructs the survival curve, detector-present flux density, detected fraction, finite-window restricted mean, and optional Bohmian histogram overlay. | Closest loader to Paper Fig. 1 and Supplemental Material S2--S4. |
| `plot_bohmian_trajectories.ipynb` / `plot_bohmian_trajectories.py` | Plots selected Pauli-current trajectories in 3D and in XY/XZ/YZ projections. | Trajectory visualization companion. In this project, trajectories sample the spinor-ABC detector-present flux law. |

Each loader may appear as a notebook (`.ipynb`) for interactive inspection and/or as an exported Python script (`.py`) for quick GitHub code review. The notebook is usually better for reproducing figures interactively; the `.py` export is easier for a reader to skim without opening Jupyter.

---

## Recommended reading order

1. Start with `plot_detection_time_distribution.*` because it reconstructs the main quantitative observables:

   ```math
   S(t;\omega)=\|\Psi_t\|^2,
   \qquad
   g(t;\omega)=-\frac{d}{dt}S(t;\omega),
   \qquad
   \mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt.
   ```

2. Then inspect `make_density_gifs.*` to see the qualitative 3D density evolution behind selected runs.

3. Finally inspect `plot_bohmian_trajectories.*` if you want the trajectory-level visualization of the same detector-present Pauli-current law.

---

## `plot_detection_time_distribution.*`

This is the most important loader for the main paper-level result. It loads the compact norm-monitoring output from the main solver,

```text
prob_times.npy
total_probs.npy
constants.npz
```

and reconstructs the survival probability

```math
S(t;\omega)=\|\Psi_t\|^2.
```

From this it computes the detector-present roof-flux density

```math
g(t;\omega)=-\frac{dS(t;\omega)}{dt},
```

as well as the detected fraction in a finite observation window,

```math
D_T(\omega)=1-S(T;\omega),
```

and the restricted mean detection time,

```math
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt=E[\min(\tau,T)].
```

This loader is the bridge between raw solver arrays and the figure-level detection-time plots. It is the closest post-processing file to the confinement sweep in Paper Fig. 1 and the numerical setup, convergence, full sweep, and finite-window bookkeeping in Supplemental Material S2--S4.

When trajectory hit-time arrays are available, this loader can overlay a gray Bohmian histogram on top of the flux curve. The correct interpretation is:

```text
blue curve  = detector-present roof-flux / norm-loss density
gray bars   = Monte Carlo trajectory samples of that same detector-present law
red curve   = detector-free Gaussian comparison current, not a detector model and not a fit input
```

The histogram should not be called a separate no-detector Bohmian arrival-time prediction. In this project the sampled trajectories use the Pauli-current velocity field generated by the spinor-ABC wavefunction,

```math
\dot Q(t)=\frac{j^P[\Psi_t^{\rm ABC}](Q(t))}{\rho[\Psi_t^{\rm ABC}](Q(t))},
```

and are binned by their first roof-hit times.

Typical outputs are

```text
arrival_stats_summary.npz
arrival_stats_summary.txt
combined_arrival2.png
combined_arrival2.pdf
```

or similarly named figure/summary products, depending on the run version.

### Notes for interpretation

- `total_probs.npy` is the survival curve, not a detection-time histogram.
- The flux density `g(t;omega)` is obtained by differentiating the survival curve, so smoothing or the paper's plotting pipeline may be needed for publication-quality curves.
- The restricted mean `mu*(T;omega)` is a finite-window statistic. It includes both the timing of detected probability and the probability still undetected at time `T`.
- If a histogram is normalized only over detected trajectories, it represents a conditional finite-window density. If it is normalized over all sampled initial particles, its area estimates the detected fraction.

---

## `make_density_gifs.*`

This loader builds visualizations from saved density snapshots such as

```text
rho_prob_t0.10000.npy
rho_prob_t0.20000.npy
rho_prob_t0.30000.npy
...
```

It is mainly for qualitative communication. It helps a reader see the time evolution of the 3D probability density, the confined transverse profile, the propagation toward the roof, absorption, storage near the roof, and delayed oscillatory structure.

Typical generated products include

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

These GIFs are useful for GitHub and presentations, but they are not the primary quantitative evidence. The quantitative evidence is the roof-flux / norm-loss analysis, convergence checks, confinement sweep, and boundary-symbol diagnostics.

### Notes for interpretation

- The density snapshots visualize `rho = |psi_up|^2 + |psi_down|^2`.
- A GIF may come from a representative parameter run rather than from the exact main Fig. 1 production sweep.
- Use captions carefully: describe GIFs as representative simulations or qualitative animations unless they are explicitly tied to a figure/table parameter set.

---

## `plot_bohmian_trajectories.*`

This loader plots selected Pauli-current trajectories from saved trajectory arrays. It supports several storage conventions used during the research workflow, for example

```text
# Selected trajectory format
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

Typical outputs are

```text
traj_sel_3D.png
traj_sel_projections.png
```

The goal is geometric inspection: how representative sampled trajectories move in the confined guide, how they approach the roof, and how selected detected/undetected trajectories differ visually.

### Correct interpretation

These trajectories are not an alternative physical detector model. They are sampled paths generated from the same spinor-ABC detector-present wavefunction and Pauli-current velocity field used in the flux calculation. Therefore:

```text
Do say:     Bohmian trajectory samples of the detector-present spinor-ABC flux law.
Do not say: Detector-free Bohmian arrival-time prediction.
```

This distinction matters because the paper compares detector-present absorbing-boundary dynamics with detector-free arrival-time proposals. The loader belongs to the detector-present side of that distinction.

---

## How to run

The safest workflow is to open or copy the loader into a completed run directory. Many research-loader notebooks assume that the current working directory contains the arrays they load.

From the repository root, for interactive use:

```bash
jupyter lab python-scripts/Loaders/plot_detection_time_distribution.ipynb
jupyter lab python-scripts/Loaders/make_density_gifs.ipynb
jupyter lab python-scripts/Loaders/plot_bohmian_trajectories.ipynb
```

From inside a completed run directory:

```bash
cd runs/<completed-run>
jupyter lab ../../python-scripts/Loaders/plot_detection_time_distribution.ipynb
```

For exported `.py` versions, run them only after checking the input path variables near the top of the script:

```bash
python python-scripts/Loaders/plot_detection_time_distribution.py
python python-scripts/Loaders/make_density_gifs.py
python python-scripts/Loaders/plot_bohmian_trajectories.py
```

If this folder is later moved to a top-level `loaders/` directory, replace `python-scripts/Loaders/` by `loaders/` in the commands above.

---

## Dependencies

The loaders are CPU-side post-processing tools. They normally do not require CuPy unless the loader has been modified to inspect GPU arrays directly.

Typical dependencies are

```bash
pip install numpy scipy matplotlib imageio scikit-image notebook jupyterlab
```

Optional, depending on the visualization mode:

```bash
pip install ipywidgets tqdm pandas
```

The solver environment additionally requires CUDA-compatible CuPy, but that is a solver requirement rather than a loader requirement.

---

## Data hygiene

Recommended practice:

```text
runs/                  local or scratch run output; do not commit
python-scripts/Loaders/ commit loader notebooks/scripts and this README
data/                  commit only selected compact arrays or documented media links
figures/               commit selected final figures only, if used
```

Avoid committing large raw arrays from every run. For GitHub, prefer:

```text
small summary .npz/.txt files
selected .csv tables
final .png/.pdf figures
README files explaining parameter sets
links to large GIFs or archived data when needed
```

---

## What this folder demonstrates

For a scientific or industry reader, this folder shows the analysis layer after large GPU/HPC simulations:

- robust loading of legacy and current simulation-output formats;
- survival, norm-loss, and roof-flux post-processing;
- finite-window detection statistics;
- Monte Carlo trajectory histogram handling;
- 3D trajectory visualization;
- density-frame and GIF generation;
- publication-quality Matplotlib/Jupyter workflows;
- disciplined separation between heavy numerical solvers and lightweight reproducibility/visualization scripts.

This is the folder to inspect after `../Solvers/` to understand how raw simulation arrays become the figures and media shown in the repository.

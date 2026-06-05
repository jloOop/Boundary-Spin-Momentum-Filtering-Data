# Boundary Spin–Momentum Filtering: Gaussian Waveguide Data

This repository contains the public numerical material for the follow-up project

**Boundary Spin–Momentum Filtering Delays Quantum Detection in a Harmonic Waveguide**

by **Alireza Jozani** and **Roderich Tumulka**.

The repository is meant to be a clean technical companion to the manuscript. It collects the scripts, selected summary data, PNG figures, and GIF animations used to document the detector-present spinor absorbing-boundary simulations for Gaussian initial wave packets.

## What this repository shows

The simulations solve the two-component time-dependent Schrödinger equation for a spin-1/2 particle in a harmonic waveguide with a spin-coupled absorbing boundary condition at the detecting roof. The detector-present detection-time density is the roof flux

```math
g(t;\omega)=\kappa\int_{\Sigma_L}|\Psi(x,y,L,t)|^2\,dx\,dy=-\frac{d}{dt}\|\Psi_t\|^2 .
```

The main finite-window statistic is

```math
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt,
\qquad S(t;\omega)=\|\Psi_t\|^2 .
```

For the main confinement sweep, increasing `omega` narrows the transverse harmonic ground state. The numerical result is that the prompt roof-flux peak is suppressed, the detected fraction in the fixed observation window decreases, and probability weight moves into a delayed oscillatory sector. In the finite window used in the paper, the statistic is fitted by

```math
\mu^*(20;\omega) \simeq 4.084 + 0.638\sqrt{\omega}.
```

The mechanism is local to the absorbing boundary. A tangential roof mode with wave vector `xi` is split by the spinor ABC into two spin–momentum branches with impedances

```math
\lambda_\pm(\xi)=i\kappa\pm |\xi|.
```

For the harmonic transverse state, the typical tangential scale is

```math
|\xi|\sim \ell_\perp^{-1}\sim \sqrt{\omega}.
```

This repository organizes the simulation scripts and visual outputs around that mechanism.

## Quick navigation

```text
paper/                                      manuscript PDFs or TeX source
python-scripts/                             production and diagnostic Python scripts
notebooks/                                  plotting and GIF-generation notebooks
follow-up-results-3D/3D-results/
  TimeEvolution-WaveFunction-Gifs/          density evolution GIFs and links to large ZIP assets
  Bohmian-Trajectories/                     trajectory PNGs/GIFs and trajectory notes
  Paper-Figures-and-PNGs/                   paper-level PNG figures
  Boundary-Symbol-Diagnostics/              boundary covariance and Duhamel diagnostic outputs
  Reflection-Diagnostics/                   spectral reflection/backflow diagnostic outputs
  Parameter-Scans/                          kappa, sigma_z, k_z, and large-box scans
data/
  selected-runs/                            small selected arrays for representative runs
  processed-summary-tables/                 CSV/JSON tables used in the manuscript
docs/                                       technical data guide and upload instructions
reproducibility/                            run manifest and checksums
```

## How to read the repository

Start with [`docs/TECHNICAL_DATA_GUIDE.md`](docs/TECHNICAL_DATA_GUIDE.md). It maps the manuscript and supplement to scripts, outputs, figures, PNGs, and GIFs.

For simple browsing, open these pages:

- [`follow-up-results-3D/3D-results/TimeEvolution-WaveFunction-Gifs/ReadMe.md`](follow-up-results-3D/3D-results/TimeEvolution-WaveFunction-Gifs/ReadMe.md)
- [`follow-up-results-3D/3D-results/Bohmian-Trajectories/ReadMe.md`](follow-up-results-3D/3D-results/Bohmian-Trajectories/ReadMe.md)
- [`follow-up-results-3D/3D-results/Paper-Figures-and-PNGs/ReadMe.md`](follow-up-results-3D/3D-results/Paper-Figures-and-PNGs/ReadMe.md)
- [`python-scripts/README.md`](python-scripts/README.md)

## Main scripts

| File | Purpose |
|---|---|
| `python-scripts/730SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py` | Main Gaussian spinor-ABC solver with density snapshots and selected Bohmian trajectories. |
| `python-scripts/953Reflection_Diag.py` | Spectral reflection and backflow diagnostics. |
| `python-scripts/972Cov_Diag_pro.py` | Boundary-layer covariance and full two-branch diagnostic. |
| `notebooks/Spin_Gif.ipynb` | Converts saved `rho_prob_t*.npy` density snapshots into PNGs and GIFs. |
| `notebooks/Paper_Presentation2.ipynb` | Arrival-time, roof-flux, survival, and paper-figure postprocessing. |
| `notebooks/Traj3D.ipynb` | Selected Bohmian trajectory plots. |

## Data policy

Large raw HPC outputs are not stored directly in the repository. Put only selected small arrays, processed summary files, PNGs, README files, and scripts in GitHub. Large GIF collections should be uploaded as ZIP files under **GitHub Releases**, then linked from `Links-to-Gifs.md`.

Recommended rule:

```text
small PNG/GIF files: commit directly
large GIF sets: zip them and upload as release assets
large raw .npy/.npz simulation folders: keep external; link them here
```

## Minimal reproduction path

A typical selected run directory contains

```text
constants.npz
prob_times.npy
total_probs.npy
rho_prob_t*.npy
bohm_t_hit.npy
bohm_arrived_mask.npy
bohmian_traj_selected.npy
bohm_times.npy
summary.json
```

Then the visual workflow is

```bash
# 1. Run or copy a selected simulation folder.
python python-scripts/730SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py

# 2. Generate density PNGs and GIFs from rho_prob_t*.npy.
jupyter notebook notebooks/Spin_Gif.ipynb

# 3. Generate arrival-time and survival plots.
jupyter notebook notebooks/Paper_Presentation2.ipynb

# 4. Generate trajectory PNGs.
jupyter notebook notebooks/Traj3D.ipynb
```

For cluster use, set `OUTDIR` to the folder where output should be written.

## Citation

If you use this repository, cite the associated manuscript once available.

## Author

Alireza Jozani  
GitHub: `jloOop`

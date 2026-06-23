# Boundary Spin–Momentum Filtering Data 

## Python scripts

- `Solvers/`: GPU/CuPy Crank--Nicolson/GMRES production and diagnostic solvers.
- `Loaders/`: CPU-side post-processing scripts/notebooks for plots, GIFs, detection-time distributions, and trajectory visualization.

Start with:
- `Solvers/README.md`
- `Loaders/README.md`


**Reproducibility material for:**  
**Spin–Momentum Filtering by an Absorbing Boundary Delays Quantum Detection in a Harmonic Waveguide**  

This repository contains selected scripts, reduced data, figures, and animations for the detector-present spinor absorbing-boundary simulations used in for a manuscript and Supplemental Material. It is a focused follow-up to the broader Schrödinger/Pauli detection-time simulations in the first repository.

The repository is intended to show the full scientific chain:

```text
spinor absorbing boundary  ->  GPU/CuPy finite-difference solver
                            ->  roof-flux detection statistics
                            ->  boundary-symbol diagnostics
                            ->  figures, tables, GIFs, and reproducibility notes
```

## Scientific point

The simulations solve the two-component Pauli/Schrödinger equation in a harmonic waveguide with a spin-coupled absorbing boundary at the detecting roof. The detector observable is the roof flux, equivalently the norm loss of the nonunitary spinor-ABC evolution,

```math
g(t;\omega)=\kappa\int_{\Sigma_L}\Psi^\dagger\Psi(x,y,L,t)\,dx\,dy
          =-\frac{d}{dt}\|\Psi_t\|^2.
```

For a finite observation window `T`, the main statistic is the restricted mean detection time

```math
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt,
\qquad S(t;\omega)=\|\Psi_t\|^2.
```

The mechanism is local to the spinor absorbing boundary. A tangential roof mode with wave vector `xi` is split into two spin–momentum branches with boundary impedances

```math
\lambda_\pm(\xi)=i\kappa\pm |\xi|.
```

In a harmonic transverse state, the oscillator length scales as

```math
\ell_\perp=\omega^{-1/2}, \qquad |\xi|\sim \ell_\perp^{-1}\sim \sqrt{\omega}.
```

Therefore increasing `omega` narrows the guide, strengthens the local boundary response, suppresses the prompt roof-flux peak, lowers the fixed-window detected fraction, and shifts probability weight into a delayed oscillatory detection sector. In the finite window used in the manuscript,

```math
\mu^*(20;\omega) \simeq 4.084 + 0.638\sqrt{\omega}.
```

This fit is a finite-window result for the stated packet, detector parameter, geometry, and observation time. The robust point is the boundary scale `sqrt(omega)`, not the numerical universality of the fitted coefficients.

## What this repository is, and is not

This is a **reproducibility and presentation repository**. It contains selected code, reduced outputs, summary tables, figures, and animations.

It is **not** intended to store all raw HPC output. Full raw `.npy`, `.npz`, `.h5`, and scratch directories should stay external or be uploaded as release assets when needed.

Bohmian trajectories, when shown, are used as Monte Carlo samples of the same detector-present roof-flux law. They are not used here as a separate no-detector arrival-time proposal.

## Repository map

```text
README.md
CITATION.cff
environment.yml
requirements.txt
.gitignore

paper/
  Submitted to PRL + and Supplemental Material PDFs

python-scripts/
  README.md
  solver_spinor_abc_gaussian.py   legacy production/diagnostic solver
  diagnose_reflection_time_Decomposition.py                         reflection/backflow diagnostics
  diagnose_boundary_symbol.py                            boundary-symbol covariance diagnostics

data/
  README.md
  processed-summary-tables/                     compact CSV/JSON tables used in plots
  selected-runs/                                small representative arrays only
  3DGifs_Simulations/                           selected GIF outputs or release links

docs/
  TECHNICAL_DATA_GUIDE.md                       map from paper sections to scripts/data
  FIGURE_MAP.md                                 which script/data made each figure
  REPRODUCIBILITY.md                            minimal workflow and output contracts
  DATA_POLICY.md                                what should and should not be committed

reproducibility/
  run_manifest.csv                              run labels, parameters, outputs, checksums
  checksums.sha256                              optional checksums for selected assets
```

## Main scripts

| File | Role | Main outputs |
|---|---|---|
| `python-scripts/solver_spinor_abc_gaussian.py` | Main CuPy finite-difference Crank–Nicolson/GMRES spinor-ABC run; also used for selected density snapshots and trajectory visualizations. | `constants.npz`, `prob_times.npy`, `total_probs.npy`, `rho_prob_t*.npy`, selected Bohmian arrays, `summary.json` |
| `python-scripts/diagnose_reflection_time_Decomposition.py` | Spectral reflection/backflow diagnostic below the detecting roof. | `roof_Jnet.npy`, `roof_Jplus.npy`, `roof_Jminus.npy`, `kz_P_plus.npy`, `kz_P_minus.npy`, `kz_R.npy`, `summary.json` |
| `python-scripts/diagnose_boundary_symbol.py` | Boundary-symbol covariance and Duhamel/one-grid continuation diagnostics. | `det_rate_toprow.npy`, `bl_rate0.npy`, `bl_rate_Ra.npy`, `bl_rate_sa.npy`, `bl_rate_R2.npy`, `bl_rate_s2.npy`, `duhamel_E_full_rel_true_depthSteps_*.npy`, `boundary_layer_covariance_integrands.npz`, `summary.json` |

Recommended future cleanup: keep the current filenames as legacy names for reproducibility, but add clearer aliases or wrappers such as `solve_spinor_abc_gaussian.py`, `diagnose_reflection.py`, and `diagnose_boundary_symbol.py`.

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate spinor-abc-filtering
```

Install the CuPy build matching the CUDA version on the cluster. For example, on CUDA 12 systems:

```bash
pip install cupy-cuda12x
```

Run a selected simulation or diagnostic:

```bash
export OUTDIR=/path/to/output/folder
python python-scripts/730SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py
python python-scripts/953Reflection_Diag.py
python python-scripts/972Cov_Diag_pro.py
```

For cluster runs, keep raw output under `runs/`, `scratch/`, or an external storage location. Commit only reduced tables, selected snapshots, and final figures.

## Minimal reproducibility path

A representative selected run should contain:

```text
constants.npz
summary.json
prob_times.npy
total_probs.npy
rho_prob_t*.npy                         optional selected snapshots only
bohm_t_hit.npy                          optional trajectory hit times
bohm_arrived_mask.npy                    optional trajectory mask
bohmian_traj_selected.npy                optional selected trajectories
bohm_times.npy                           optional trajectory time grid
```

A processed table should contain enough metadata to reproduce a figure without rerunning the HPC solver:

```text
omega, kappa, kz, sigma_z, theta, phi, Lz, Nx, Ny, Nz, dt, T_final,
detected_fraction, mu_star, prompt_weight, late_weight, notes
```

## Suggested citation

If you use this repository, cite the associated manuscript and this repository. Update `CITATION.cff` after the manuscript has a DOI, arXiv identifier, or journal reference.

## Author

Alireza Jozani  
Physics PhD candidate, University of Tübingen  
GitHub: `jloOop`

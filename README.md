# Boundary Spin–Momentum Filtering Data

**Focused reproducibility repository for spinor absorbing-boundary simulations in numerical quantum detection theory.**

This repository contains selected research code, reduced data, diagnostic outputs, figures, and representative animations for the project

> **A. Jozani, _Spin–Momentum Impedance and Filtering by a Spin-Coupled Absorbing Boundary Condition_**  
> Preprint / manuscript-related repository. See `paper/` and `CITATION.cff` for the current citation metadata.

The project studies detector-present Pauli/Schrödinger dynamics in a harmonic waveguide with a spin-coupled absorbing boundary at the detecting roof. The repository is intended as **research-code and reproducibility material**, not as a production software package.

---

## Scientific point

The central question is how a spin-coupled absorbing boundary changes the detection-time distribution of a Pauli particle in a confined waveguide.

The detector observable is the roof flux, equivalently the norm loss of the non-unitary spinor absorbing-boundary evolution,

```math
g(t;\omega)=\kappa\int_{\Sigma_L}\Psi^\dagger\Psi(x,y,L,t)\,dx\,dy
      = -\frac{d}{dt}\|\Psi_t\|^2 .
```

For a finite observation window `T`, the main finite-window statistic is the restricted mean detection time,

```math
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt,
\qquad S(t;\omega)=\|\Psi_t\|^2 .
```

The key mechanism is local to the spinor absorbing boundary. In tangential Fourier variables, the boundary matrix has two spin–momentum branches,

```math
\lambda_\pm(\xi)=i\kappa\pm |\xi| .
```

For a harmonic transverse ground state,

```math
\ell_\perp=\omega^{-1/2},
\qquad |\xi|\sim \ell_\perp^{-1}\sim \sqrt{\omega} .
```

Thus, increasing the transverse confinement parameter `omega` changes the local boundary response: it suppresses the prompt roof-flux peak, lowers the detected fraction in a fixed time window, and shifts part of the detector-present response to later times. The fitted form

```math
\mu^*(20;\omega) \simeq 4.084 + 0.638\sqrt{\omega}
```

is a finite-window diagnostic for the simulated packet, detector parameter, guide geometry, and observation time. The robust conclusion is the boundary scale `sqrt(omega)`, not universal numerical coefficients.

---

## What this repository contains

```text
Boundary-Spin-Momentum-Filtering-Data/
├── Solvers/              GPU/CuPy Crank–Nicolson/GMRES research solvers and diagnostics
├── Loaders/              CPU-side post-processing for figures, GIFs, and trajectory plots
├── data/                 Selected compact data, processed tables, and representative media
├── paper/                Manuscript / supplementary material when publicly available
├── environment.yml       Conda environment without CuPy pinned to a specific CUDA build
├── requirements.txt      Lightweight Python dependencies for inspection/post-processing
├── CITATION.cff          Citation metadata for the repository
└── README.md             This overview
```

### Start here

| Path | Purpose |
|---|---|
| `Solvers/READMe.md` | Explains the GPU/CuPy solver and diagnostic scripts. |
| `Loaders/README.md` | Explains how solver outputs are converted into figures, GIFs, and trajectory visualizations. |
| `data/README.md` | Notes on selected data and interpretation of the GIF material. |
| `paper/` | Manuscript-related PDF and citation context. |
| `CITATION.cff` | Repository citation metadata. |

---

## Main workflow

The scientific chain is:

```text
spinor absorbing boundary
  -> GPU/CuPy finite-difference simulation
  -> non-unitary roof-flux / norm-loss statistics
  -> boundary-symbol and reflection diagnostics
  -> figures, tables, GIFs, and reproducibility notes
```

The heavy numerical simulations are in `Solvers/`. The `Loaders/` directory is the analysis layer: it reads completed solver outputs and produces detection-time plots, representative density animations, and trajectory visualizations.

---

## Main scripts

| Script | Role | Typical outputs |
|---|---|---|
| `Solvers/solver_spinor_abc_gaussian.py` | Main Gaussian spinor-ABC simulation; evolves the 3D spinor wavefunction and records survival/norm-loss data. | `prob_times.npy`, `total_probs.npy`, `constants.npz`, `simulation_log.txt` |
| `Solvers/diagnose_reflection_time_Decomposition.py` | Reflection, time-window, near-roof, and finite-guide-memory diagnostics. | `roof_J*.npy`, `kz_*.npy`, `near_roof_mass.npy`, `summary.json` |
| `Solvers/diagnose_boundary_symbol.py` | Boundary-symbol, covariance, finite-epsilon, and Duhamel diagnostics. | `det_rate_toprow.npy`, `bl_rate*.npy`, `duhamel_*`, `boundary_layer_covariance_integrands.npz` |
| `Loaders/plot_detection_time_distribution.py` | Reconstructs survival, roof flux, detected fraction, restricted mean, and optional trajectory histograms. | `arrival_stats_summary.*`, detection-time figures |
| `Loaders/make_density_gifs.py` | Builds frames and GIFs from saved density snapshots. | `midplanes.gif`, `contour_xy.gif`, `slices_bar.gif`, etc. |
| `Loaders/plot_bohmian_trajectories.py` | Plots selected Pauli-current trajectories from saved trajectory arrays. | 3D trajectory and projection figures |

---

## Installation

Create the base environment:

```bash
conda env create -f environment.yml
conda activate spinor-abc-gaussian-followup
```

or install the lightweight dependencies directly:

```bash
pip install -r requirements.txt
```

The solver scripts require a CUDA-capable GPU and a CuPy build matching the local CUDA version. For example, on a CUDA 12 system:

```bash
pip install cupy-cuda12x
```

The `Loaders/` scripts are mainly CPU-side post-processing tools and can usually be inspected without CuPy, provided the required input arrays are available.

---

## Example commands

Run from the repository root.

```bash
mkdir -p runs

# Main Gaussian spinor-ABC confinement / norm-loss run
OMEGA=300 OUTDIR=./runs \
python Solvers/solver_spinor_abc_gaussian.py

# Reflection, timing-window, and near-roof diagnostics
OMEGA=100 OUTDIR=./runs \
python Solvers/diagnose_reflection_time_Decomposition.py

# Boundary-symbol covariance and Duhamel diagnostics
OMEGA=200 OUTDIR=./runs NX=100 NY=100 NZ=1500 DT=2.5e-4 TFINAL=20 \
python Solvers/diagnose_boundary_symbol.py
```

For post-processing, first inspect the input-path variables near the top of each loader. Many loaders assume that they are opened or copied into a completed run directory.

```bash
python Loaders/plot_detection_time_distribution.py
python Loaders/make_density_gifs.py
python Loaders/plot_bohmian_trajectories.py
```

For interactive inspection:

```bash
jupyter lab
```

---

## Minimal reproducibility path

A representative selected run should contain:

```text
constants.npz        run parameters and grid metadata
prob_times.npy       time grid for survival monitoring
total_probs.npy      survival probability S(t; omega)
rho_prob_t*.npy      optional selected density snapshots
summary.json         optional diagnostic metadata
```

From `prob_times.npy` and `total_probs.npy`, one can reconstruct:

```math
D_T(\omega)=1-S(T;\omega),
\qquad
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt,
\qquad
g(t;\omega)=-\frac{dS}{dt}.
```

For publication-quality figures, use the same smoothing and plotting pipeline used for the manuscript figures rather than relying only on raw numerical differentiation.

---

## Data policy

This repository is intentionally selective.

- Commit compact processed tables, reduced outputs, representative figures, and documented GIFs.
- Keep full raw GPU/HPC output in `runs/`, scratch storage, external archives, or GitHub releases when appropriate.
- Do not treat this repository as a complete raw simulation archive.
- Use `constants.npz` and `summary.json` as the source of truth for the parameters of a specific run.

---

## Correct interpretation of trajectories

Bohmian trajectories shown in this repository are used as Monte Carlo samples of the **same detector-present spinor-ABC flux law**. They are not a separate no-detector arrival-time proposal.

Use this wording:

```text
Bohmian trajectory samples of the detector-present spinor-ABC roof-flux law.
```

Avoid this wording:

```text
Detector-free Bohmian arrival-time prediction.
```

---

## What I personally implemented

For this manuscript-related repository, I prepared the public research-code and reproducibility material for the spinor absorbing-boundary follow-up project, including:

- spinor-ABC simulation and diagnostic workflows;
- boundary-symbol, reflection, and finite-window detection-time diagnostics;
- post-processing scripts for survival/norm-loss curves, detector-time plots, density animations, and trajectory visualizations;
- selected reduced data, figures, GIFs, and citation/environment files.

The code should be read as research software used to support a theoretical/numerical study, not as a packaged software product or commercial simulation platform.

---

## Safe claims and boundaries

This repository supports claims about:

- Python/CuPy scientific-computing workflows;
- finite-difference Pauli/Schrödinger simulations;
- Crank–Nicolson time stepping and GMRES/Krylov solves;
- non-Hermitian absorbing-boundary detector models;
- roof-flux / norm-loss detection-time statistics;
- boundary-symbol diagnostics and parameter scans;
- reduced-data, figure, GIF, and reproducibility organization.

This repository does **not** claim:

- a physical detector implementation;
- a universal experimental detector law;
- a detector-free Bohmian arrival-time model;
- a production software package;
- CUDA C/C++ expertise;
- complete archival storage of every raw HPC output;
- journal publication or PRL acceptance unless the citation metadata is later updated after confirmation.

---

## Related work

- A. Jozani, _Spin–Momentum Impedance and Filtering by a Spin-Coupled Absorbing Boundary Condition_, arXiv:2606.25650.
- A. Jozani and R. Tumulka, _Detection Time Distribution Predicted Using Absorbing Boundary Conditions and Imaginary Potentials_, arXiv:2603.22044.

---

## Citation

Use the repository citation metadata in `CITATION.cff`. Until journal metadata is finalized, cite the associated manuscript/preprint and this repository together.

```bibtex
@software{jozani_boundary_spin_momentum_filtering_data_2026,
  author = {Jozani, Alireza},
  title  = {Boundary Spin-Momentum Filtering: Gaussian Waveguide Data},
  year   = {2026},
  url    = {https://github.com/jloOop/Boundary-Spin-Momentum-Filtering-Data}
}
```

---

## Author

**Alireza Jozani**  
Physics PhD Candidate, University of Tübingen  
GitHub: [`jloOop`](https://github.com/jloOop)





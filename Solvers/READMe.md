# Representative spinor-ABC solvers

This folder contains the representative Python/CuPy research solvers and diagnostics used for the numerical side of:

> **A. Jozani, _Spin–Momentum Impedance and Filtering by a Spin-Coupled Absorbing Boundary Condition_**

The scripts are included as reproducibility and code-review material, not as a packaged Python library.

## What this folder demonstrates

- GPU/CuPy finite-difference simulation of 3D spinor quantum dynamics.
- Sparse non-Hermitian Hamiltonian assembly.
- Crank–Nicolson time stepping with GMRES/Krylov linear solves.
- Ghost-point implementation of a spin-coupled absorbing boundary.
- Probability-current, reflection, boundary-symbol, time-decomposition, and Duhamel diagnostics.
- Compact output suitable for post-processing and reproducibility.

The physical observable throughout the project is the detector-present roof flux, equivalently the norm loss of the non-unitary spinor absorbing-boundary evolution. These scripts should be read as simulations and diagnostics of a detector model, not as detector-free arrival-time simulations.

## Folder contents

```text
Solvers/
├── solver_spinor_abc_gaussian.py
├── diagnose_reflection_time_Decomposition.py
├── diagnose_boundary_symbol.py
├── solver_bookkeeping.py
├── solver_metadata.py
└── README.md
```

## Scientific background

The simulations solve a two-component Pauli/Schrödinger equation in a finite harmonic waveguide,

```math
i\partial_t\Psi=
\left[-\frac{1}{2}\Delta + \frac{1}{2}\omega^2((x-x_c)^2+(y-y_c)^2)\right]\Psi,
\qquad \Psi=(\psi_\uparrow,\psi_\downarrow)^T .
```

The detecting surface is the roof

```math
\Sigma_L=\{z=L\},
```

where the spin-coupled absorbing boundary condition is imposed:

```math
(\sigma\cdot\nabla)\Psi=i\kappa\sigma_z\Psi .
```

For the Pauli current, this boundary gives the outward roof-flux density

```math
g(t;\omega)=\kappa\int_{\Sigma_L}\Psi^\dagger\Psi\,dx\,dy
= -\frac{d}{dt}\|\Psi_t\|^2 .
```

The central mechanism is that the spinor ABC is not a scalar sink. At the roof, tangential Fourier modes see the boundary matrix

```math
C(\xi)=i\kappa I+(\hat z\times \xi)\cdot\sigma,
```

with eigenbranches

```math
\lambda_\pm(\xi)=i\kappa\pm |\xi| .
```

For the harmonic transverse ground state,

```math
|\xi|\sim \ell_\perp^{-1}\sim \sqrt{\omega} .
```

## Solver map

| Script | Main role | Main outputs |
|---|---|---|
| `solver_spinor_abc_gaussian.py` | Main Gaussian spinor-ABC solver. Evolves the 3D spinor wavefunction and records survival/norm-loss data. | `prob_times.npy`, `total_probs.npy`, `constants.npz`, `simulation_log.txt`, `stdout.txt` |
| `diagnose_reflection_time_Decomposition.py` | Reflection, time-window, longitudinal-k, near-roof, and finite-guide-memory diagnostics. | `summary.json`, `roof_J*.npy`, `kz_*.npy`, `near_roof_mass.npy` |
| `diagnose_boundary_symbol.py` | Boundary-symbol, covariance, finite-epsilon, and Duhamel diagnostics. | `summary.json`, `det_rate_toprow.npy`, `bl_rate*.npy`, `duhamel_*`, `boundary_layer_covariance_integrands.npz` |
| `solver_bookkeeping.py` | Shared output and metadata utilities where used. | Run bookkeeping files |
| `solver_metadata.py` | Shared parameter / metadata helpers where used. | Metadata summaries |

Recommended reading order:

1. `solver_spinor_abc_gaussian.py`
2. `diagnose_reflection_time_Decomposition.py`
3. `diagnose_boundary_symbol.py`

## Running the scripts

These scripts require a CUDA-capable GPU and a CuPy installation compatible with the local CUDA driver.

Run from the repository root:

```bash
mkdir -p runs

OMEGA=300 OUTDIR=./runs python Solvers/solver_spinor_abc_gaussian.py

OMEGA=100 OUTDIR=./runs python Solvers/diagnose_reflection_time_Decomposition.py

OMEGA=200 OUTDIR=./runs NX=100 NY=100 NZ=1500 DT=2.5e-4 TFINAL=20 \
  python Solvers/diagnose_boundary_symbol.py
```

For cluster runs, set `OUTDIR` to scratch or project storage. Do not commit raw run directories unless a small selected file is explicitly documented as reproducibility data.

## Interpreting survival output

The main solver writes:

```text
prob_times.npy
total_probs.npy
constants.npz
simulation_log.txt
stdout.txt
```

Here `total_probs.npy` is the discrete survival curve

```math
S(t;\omega)=\|\Psi_t\|^2 .
```

From this one obtains

```math
D_T(\omega)=1-S(T),
\qquad
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt,
\qquad
g(t;\omega)=-\frac{dS}{dt}.
```

## Interpretation rules

1. Detector-present, not detector-free: the blue flux curves and norm-loss data represent the detector-present spinor-ABC evolution.
2. Bohmian histograms, when shown, are Monte Carlo samples of the same detector-present flux law.
3. The finite-window fit is not universal; the robust feature is the boundary scale `|xi| ~ sqrt(omega)`.
4. The early/late split is diagnostic bookkeeping, not an exact microscopic decomposition.
5. Boundary-symbol inward-depth diagnostics are local probes, not a physical detector thickness.

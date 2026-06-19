
# Representative spinor-ABC solvers

This folder contains the three representative Python/CuPy solver scripts used for the numerical side of the spin--momentum filtering project.  They are included as reproducibility material for the paper

> **A. Jozani, _Spin--Momentum Filtering by an Absorbing Boundary Delays Quantum Detection in a Harmonic Waveguide_**

and for its Supplemental Material.  The scripts are not meant to be a polished Python package.  They are legacy production/diagnostic solvers kept close to the form used in the research runs, so that a reader can trace how the numerical quantities in the paper were generated.

The physical observable in this project is the **detector-present roof flux**, equivalently the **norm loss** of the non-unitary spinor absorbing-boundary evolution.  The scripts should therefore be read as simulations of the detector model itself, not as detector-free arrival-time simulations.

---

## Scientific background

The bulk evolution is the two-component Pauli/Schrödinger evolution in a harmonic waveguide,

$$
 i\partial_t\Psi = \left[-\frac12\Delta + \frac12\omega^2\big((x-x_c)^2+(y-y_c)^2\big)\right]\Psi,
 \qquad
 \Psi=(\psi_\uparrow,\psi_\downarrow)^T.
$$

The lower face and transverse side walls are reflecting Dirichlet walls.  The detecting roof is the plane

$$
\Sigma_L=\{z=L\},
$$

where the spin-coupled absorbing boundary condition is imposed:

$$
(\sigma\cdot\nabla)\Psi=i\kappa\sigma_z\Psi.
$$

In components this reads

$$
\partial_z\psi_\uparrow=i\kappa\psi_\uparrow-(\partial_x-i\partial_y)\psi_\downarrow,
\qquad
\partial_z\psi_\downarrow=i\kappa\psi_\downarrow+(\partial_x+i\partial_y)\psi_\uparrow.
$$

For the Pauli current this boundary gives the detector-present roof-flux density

$$
 g(t;\omega)=\kappa\int_{\Sigma_L}\Psi^\dagger\Psi\,dxdy
            =-\frac{d}{dt}\|\Psi_t\|^2.
$$

The survival probability is

$$
S(t;\omega)=\|\Psi_t\|^2,
$$

and the finite-window restricted mean detection time used in the paper is

$$
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt
               =E[\min(\tau,T)].
$$

The central mechanism studied by the diagnostics is that the spinor ABC is not a scalar sink.  At the roof, tangential Fourier modes see the boundary matrix

$$
C(\xi)=i\kappa I+(\hat z\times \xi)\cdot\sigma,
$$

with eigenbranches

$$
\lambda_\pm(\boldsymbol{\xi})=i\kappa\pm |\xi|.
$$

For the harmonic transverse ground state, the typical tangential scale is

$$
|\xi|\sim \ell_\perp^{-1}\sim \sqrt{\omega}.
$$

This is the local boundary scale behind the confinement-dependent detector response.

---

## Solver map

| Script | Role in this repository | Main outputs | Paper connection |
|---|---|---|---|
| `746SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py` | Main compact production solver for the confinement sweep and finite-window detection-time statistic.  It evolves the 3D spinor wave function with the spinor ABC using Crank--Nicolson and GMRES, records the total norm, and produces the data needed for `S(t;omega)`, `g(t;omega)`, `D_T`, and `mu^*(T;omega)`. | `prob_times.npy`, `total_probs.npy`, `constants.npz`, `simulation_log.txt`, `stdout.txt` | Letter Fig. 1; Supplemental Material S2--S4. |
| `953Reflection_Diag.py` | Reflection and finite-window diagnostic solver.  It keeps the same TDSE/CN/spinor-ABC evolution but adds diagnostics for roof backflow, a windowed longitudinal `k_z` decomposition, near-roof storage, tangential `Pi_+`/`Pi_-` branch content, and Duhamel-style comparison of the true near-roof field with the homogeneous boundary-symbol continuation. | `summary.json`, `diag_times.npy`, `roof_J*.npy`, `kz_*.npy`, `near_roof_mass.npy`, `W_plus_*.npy`, `W_minus_*.npy`, `Q_minus_*.npy`, `E_full_*.npy`, `E_plus_*.npy` | Supplemental Material S4, S7, and S8-style discussion of early/late windows, reflection diagnostics, and near-roof boundary response. |
| `973Cov_Diag_pro.py` | Boundary-symbol/covariance diagnostic solver.  This is the most direct numerical test of the two-branch mechanism.  It evaluates the full two-branch map `exp(-epsilon J)`, the covariance coefficient `beta_omega = Cov(t, s a_omega)`, exact finite-epsilon checks, RMST coefficients, detector-budget checks, and full two-branch Duhamel/slab remainders. | `summary.json`, `diag_times.npy`, `det_rate_toprow.npy`, `bl_rate*.npy`, `bl_*epsSteps_*.npy`, `duhamel_*depthSteps_*.npy`, `boundary_layer_covariance_integrands.npz` | Letter boundary-mechanism section; Supplemental Material S6--S7. |

Together these scripts separate three tasks:

1. **Production data:** compute the survival/norm-loss curves used for the confinement sweep.
2. **Reflection diagnostics:** test whether the delayed sector behaves like ordinary propagating reflection or near-roof boundary storage/filtering.
3. **Boundary-symbol diagnostics:** verify the local two-branch spin--momentum mechanism and its finite-grid Duhamel/covariance checks.

---

## Running the scripts

These scripts require a CUDA-capable GPU and a CuPy installation compatible with the local CUDA driver.  They were written for GPU/HPC execution, not for a small laptop CPU run.

A typical workflow is:

```bash
# from the repository root
mkdir -p runs

# Main confinement/norm-loss run
OMEGA=300 OUTDIR=./runs python solvers/746SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py

# Reflection / k_z / Pi diagnostic run
OMEGA=100 OUTDIR=./runs python solvers/953Reflection_Diag.py

# Boundary-symbol covariance and Duhamel diagnostic run
OMEGA=200 OUTDIR=./runs NX=100 NY=100 NZ=1500 DT=2.5e-4 TFINAL=20 \
python solvers/973Cov_Diag_pro.py
```

For cluster runs, set `OUTDIR` to a scratch or project-storage directory rather than committing raw output folders to GitHub.

---

## Interpreting the main production output

The main production solver writes

```text
prob_times.npy
 total_probs.npy
 constants.npz
 simulation_log.txt
 stdout.txt
```

The array `total_probs.npy` is the discrete survival curve

$$
S(t;\omega)=\|\Psi_t\|^2.
$$

From this one obtains

$$
D_T(\omega)=1-S(T;\omega),
$$

and, by numerical quadrature,

$$
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt.
$$

The roof-flux density can be reconstructed as

$$
 g(t;\omega)=-\frac{dS}{dt}.
$$

A minimal post-processing sketch is:



For publication-quality figures, the derivative used for `g(t;omega)` should be smoothed or obtained from the same post-processing pipeline used for the plotted data, because numerical differentiation amplifies small norm fluctuations.

---

## Interpreting the reflection diagnostic output

`953Reflection_Diag.py` is designed to answer the question:

> Is the delayed signal mainly ordinary propagating reflection, or is it a near-roof spinor-ABC boundary response with finite-guide memory?

The most important diagnostics are:

| Output | Meaning |
|---|---|
| `roof_Jnet.npy`, `roof_Jplus.npy`, `roof_Jminus.npy` | Pauli-current roof flux and local backflow bookkeeping. |
| `kz_P_plus.npy`, `kz_P_minus.npy`, `kz_R.npy` | Windowed longitudinal FFT diagnostic in a slab below the roof.  Positive `k_z` is upgoing; negative `k_z` is downgoing. |
| `near_roof_mass.npy` | Probability stored in a near-roof slab. |
| `W_plus_roof.npy`, `W_minus_roof.npy`, `Q_minus_roof.npy` | Tangential spin--momentum branch content at the roof. |
| `E_full_depthSteps_*.npy` | Relative error of the full two-branch homogeneous boundary-symbol continuation. |
| `E_plus_depthSteps_*.npy` | Historical plus-only comparison; not the central two-branch diagnostic. |
| `summary.json` | Integrated reflection/backflow/kz/Pi/Duhamel summaries. |

The conservative first-pass timing window in this script is only a diagnostic device.  It should not be interpreted as a separate detector model.

---

## Interpreting the boundary-symbol/covariance output

`973Cov_Diag_pro.py` is the main diagnostic script for the paper's local boundary mechanism.  It uses the top-row roof trace and the discrete tangential symbol to test the two-branch response

$$
B_{\mathrm{br}}(\epsilon,\xi)=e^{-R\epsilon}\Pi_+(\xi)+e^{R\epsilon}\Pi_-(\xi),
\qquad R=|\xi|.
$$

The central coefficient is the covariance form

$$
\Lambda_\omega=\sqrt{\omega}\,\beta_\omega,
\qquad
\beta_\omega=\mathrm{Cov}_{\nu_0}(t,s a_\omega),
\qquad
s=R/\sqrt{\omega}.
$$

Important outputs include:

| Output | Meaning |
|---|---|
| `det_rate_toprow.npy` | Primary discrete detector rate at the stored ABC row. |
| `P_total_series.npy` | Norm/survival monitor for detector-budget checks. |
| `bl_rate0.npy` | Zeroth-order boundary-layer rate. |
| `bl_rate_sa.npy`, `bl_rate_Ra.npy` | First-order covariance-rate integrands. |
| `bl_rate_R2.npy`, `bl_rate_s2.npy` | Second-order scale diagnostics. |
| `bl_exact_rate_epsSteps_*.npy` | Exact finite-epsilon two-branch density-rate check. |
| `bl_linear_rate_epsSteps_*.npy`, `bl_quadratic_rate_epsSteps_*.npy` | Taylor checks against the exact finite-epsilon expression. |
| `duhamel_E_full_rel_true_depthSteps_*.npy` | Full two-branch Duhamel/slab relative error. |
| `summary.json` | Detector budget, covariance coefficients, finite-epsilon checks, and Duhamel summaries. |

This script deliberately does **not** neglect `Pi_-`.  The full two-branch map is the central object.

---

## Recommended repository practice

These scripts can generate large output directories.  Keep the repository clean:

```text
solvers/                  # commit these scripts
runs/                     # local or scratch output; do not commit
figures/                  # selected final figures only
data/                     # selected compact arrays only
docs/                     # explanations, figure maps, and post-processing notes
```

Recommended `.gitignore` entries:

```gitignore
runs/
*.npy
*.npz
*.log
stdout.txt
simulation_log.txt
__pycache__/
```

Commit only selected small arrays or summary files that are needed to reproduce the figures in the paper.

---

## Citation

If you use this repository, cite the paper and the accompanying data/code repository:

```text
A. Jozani,
"Spin--Momentum Filtering by an Absorbing Boundary Delays Quantum Detection in a Harmonic Waveguide,"
PRL-format manuscript with Supplemental Material, 2026.

A. Jozani,
Boundary-Spin-Momentum-Filtering-Data,
GitHub repository: https://github.com/jloOop/Boundary-Spin-Momentum-Filtering-Data
```

For the numerical Crank--Nicolson/GMRES implementation background and the broader detection-time framework, also cite:

```text
A. Jozani and R. Tumulka,
"Detection Time Distribution Predicted Using Absorbing Boundary Conditions and Imaginary Potentials,"
Physical Review Research, soon.
```

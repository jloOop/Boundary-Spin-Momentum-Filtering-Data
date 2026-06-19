# Representative spinor-ABC solvers

This folder contains the three representative Python/CuPy solver and diagnostic scripts for the numerical side of

> **A. Jozani, _Spin--Momentum Filtering by an Absorbing Boundary Delays Quantum Detection in a Harmonic Waveguide_**

and its Supplemental Material.

The scripts are not presented as a polished Python package. They are research solvers and diagnostic programs kept close to the form used in the production runs, with public-facing names chosen to make their roles readable:

```text
solvers/
  solver_spinor_abc_gaussian.py
  diagnose_reflection_time_Decomposition.py
  diagnose_boundary_symbol.py
```

The physical observable throughout this project is the **detector-present roof flux**, equivalently the **norm loss** of the non-unitary spinor absorbing-boundary evolution. These scripts should therefore be read as simulations and diagnostics of the detector model itself, not as simulations of a detector-free arrival-time law.

---

## Scientific background

The bulk evolution is the two-component Pauli/Schrödinger evolution in a harmonic waveguide,

$$
 i\partial_t\Psi = \left[-\frac12\Delta
 + \frac12\omega^2\big((x-x_c)^2+(y-y_c)^2\big)\right]\Psi,
 \qquad
 \Psi=(\psi_\uparrow,\psi_\downarrow)^T.
$$

The lower face and transverse side walls are reflecting Dirichlet walls. The detecting roof is

$$
\Sigma_L = \{z=L\},
$$

where the spin-coupled absorbing boundary condition is imposed:

$$
(\sigma\cdot\nabla)\Psi = i\kappa\sigma_z\Psi.
$$

In components,

$$
\partial_z\psi_\uparrow
= i\kappa\psi_\uparrow-(\partial_x-i\partial_y)\psi_\downarrow,
$$

$$
\partial_z\psi_\downarrow
= i\kappa\psi_\downarrow+(\partial_x+i\partial_y)\psi_\uparrow.
$$

For the Pauli current, this boundary gives the detector-present roof-flux density

$$
g(t;\omega)
= \kappa\int_{\Sigma_L}\Psi^\dagger\Psi\,dxdy
= -\frac{d}{dt}\|\Psi_t\|^2.
$$

The survival probability is

$$
S(t;\omega)=\|\Psi_t\|^2,
$$

and the finite-window restricted mean detection time used in the paper is

$$
\mu^*(T;\omega)
= \int_0^T S(t;\omega)\,dt
= E[\min(\tau,T)].
$$

The central mechanism is that the spinor ABC is **not** a scalar sink. At the roof, tangential Fourier modes see the boundary matrix

$$
C(\xi)=i\kappa I+(\hat z\times \xi)\cdot\sigma,
$$

with eigenbranches

$$
\lambda_\pm(\xi)=i\kappa\pm |\xi|.
$$

For the harmonic transverse ground state, the typical tangential scale is

$$
|\xi|\sim \ell_\perp^{-1}\sim \sqrt{\omega}.
$$

This local boundary scale is the source of the confinement-dependent detector response studied in the paper.

---

## Solver map

| Script | Main role | Main outputs | Paper connection |
|---|---|---|---|
| `solver_spinor_abc_gaussian.py` | Main production solver for the Gaussian spinor-ABC confinement runs. It evolves the 3D spinor wave function with Crank--Nicolson and GMRES, records the total norm, and provides the compact data needed for `S(t;omega)`, `g(t;omega)`, `D_T`, and `mu^*(T;omega)`. | `prob_times.npy`, `total_probs.npy`, `constants.npz`, `simulation_log.txt`, `stdout.txt` | Letter Fig. 1; Supplemental Material S2--S4. |
| `diagnose_reflection_time_Decomposition.py` | Reflection and finite-window/time-decomposition diagnostic. It keeps the same TDSE/CN/spinor-ABC evolution but adds diagnostics for roof backflow, longitudinal `k_z` upgoing/downgoing content, near-roof storage, tangential `Pi_+`/`Pi_-` branch content, and Duhamel-style comparison with the local boundary-symbol continuation. | `summary.json`, `diag_times.npy`, `roof_J*.npy`, `kz_*.npy`, `near_roof_mass.npy`, `W_plus_*.npy`, `W_minus_*.npy`, `Q_minus_*.npy`, `E_full_*.npy`, `E_plus_*.npy` | Supplemental Material S4 for finite-window early/late bookkeeping; S7--S8 for boundary-response and finite-guide memory diagnostics. |
| `diagnose_boundary_symbol.py` | Boundary-symbol, covariance, finite-epsilon, and Duhamel diagnostic. This is the most direct numerical diagnostic for the two-branch spin--momentum mechanism. It tests the full two-branch map, the covariance coefficient, detector-budget consistency, and finite-grid Duhamel/slab remainders. | `summary.json`, `diag_times.npy`, `det_rate_toprow.npy`, `P_total_series.npy`, `bl_rate*.npy`, `bl_*epsSteps_*.npy`, `duhamel_*depthSteps_*.npy`, `boundary_layer_covariance_integrands.npz` | Letter boundary-mechanism section; Supplemental Material S6--S7, especially S7 for the finite-grid boundary-symbol diagnostic and Duhamel estimate. |

Together, the three scripts separate the reproducibility workflow into three layers:

1. **Production data:** compute the survival/norm-loss curves used for the confinement sweep.
2. **Reflection and time decomposition:** test whether delayed signal is ordinary propagating reflection or near-roof spinor-ABC boundary response with finite-guide memory.
3. **Boundary-symbol diagnostics:** verify the local two-branch spin--momentum mechanism and its finite-grid covariance/Duhamel checks.

---

---

## Running the scripts

These scripts require a CUDA-capable GPU and a CuPy installation compatible with the local CUDA driver. They were written for GPU/HPC execution, not for small CPU-only runs.

From the repository root:

```bash
mkdir -p runs

# Main Gaussian spinor-ABC confinement/norm-loss run
OMEGA=300 OUTDIR=./runs python solvers/solver_spinor_abc_gaussian.py

# Reflection, early/late timing, k_z, Pi_+/Pi_-, and near-roof diagnostics
OMEGA=100 OUTDIR=./runs python solvers/diagnose_reflection_time_Decomposition.py

# Boundary-symbol covariance, finite-epsilon, and Duhamel diagnostics
OMEGA=200 OUTDIR=./runs NX=100 NY=100 NZ=1500 DT=2.5e-4 TFINAL=20 \
python solvers/diagnose_boundary_symbol.py
```

For cluster runs, set `OUTDIR` to a scratch or project-storage directory. Do not commit raw run directories unless a small selected file is explicitly part of the reproducibility data.

---

## Interpreting `solver_spinor_abc_gaussian.py`

This is the main compact production solver for the confinement sweep. Its essential output is the survival curve,

```text
prob_times.npy
 total_probs.npy
 constants.npz
 simulation_log.txt
 stdout.txt
```

Here `total_probs.npy` is the discrete survival probability

$$
S(t;\omega)=\|\Psi_t\|^2.
$$

From this one obtains the detected fraction

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

```python
from pathlib import Path
import numpy as np

run = Path("runs/<your-run-directory>")
t = np.load(run / "prob_times.npy")
S = np.load(run / "total_probs.npy")

T = 20.0
mask = t <= T
tT = t[mask]
ST = S[mask]

# Include the known initial survival S(0)=1 if the saved arrays start after t=0.
if len(tT) == 0 or tT[0] > 0.0:
    tT = np.concatenate(([0.0], tT))
    ST = np.concatenate(([1.0], ST))

# Add an interpolated endpoint at T if necessary.
if tT[-1] < T and t[-1] >= T:
    ST_at_T = np.interp(T, t, S)
    tT = np.concatenate((tT, [T]))
    ST = np.concatenate((ST, [ST_at_T]))

trapz = getattr(np, "trapezoid", None)
if trapz is None:
    trapz = np.trapz

D_T = 1.0 - ST[-1]
mu_star = trapz(ST, tT)

# Useful diagnostic only; numerical differentiation amplifies small fluctuations.
g = -np.gradient(S, t)

print("D_T      =", D_T)
print("mu_star =", mu_star)
```

For publication-quality flux plots, use the same smoothing or post-processing pipeline used for the paper figures rather than relying only on a raw numerical derivative.

---

## Interpreting `diagnose_reflection_time_Decomposition.py`

This script asks:

> Is the delayed signal mainly ordinary propagating reflection, or is it a near-roof spinor-ABC boundary response with finite-guide memory?

It should be read together with the finite-window bookkeeping in Supplemental Material S4. In particular, the early/late split is a diagnostic decomposition of the detector-present roof-flux signal. It is not a new detector model and not a detector-free arrival-time law.

Important output families are:

| Output | Meaning |
|---|---|
| `roof_Jnet.npy`, `roof_Jplus.npy`, `roof_Jminus.npy` | Pauli-current roof flux and local backflow bookkeeping. |
| `roof_in_J*.npy`, `roof_in_rho_int.npy` | Same style of checks one grid point below the roof. |
| `kz_P_plus.npy`, `kz_P_minus.npy`, `kz_R.npy` | Windowed longitudinal FFT diagnostic in a slab below the roof. Positive `k_z` is upgoing; negative `k_z` is downgoing. |
| `kz_slab_mass.npy` | Mass in the slab used for the longitudinal FFT diagnostic. |
| `near_roof_mass.npy` | Probability stored in a near-roof slab. |
| `W_plus_roof.npy`, `W_minus_roof.npy`, `Q_minus_roof.npy` | Tangential spin--momentum branch content at the roof. |
| `E_full_depthSteps_*.npy` | Relative error of the full two-branch homogeneous boundary-symbol continuation. |
| `E_plus_depthSteps_*.npy` | Historical plus-only comparison. The full two-branch diagnostic is the central one. |
| `summary.json` | Integrated reflection/backflow/kz/Pi/Duhamel summaries. |

The conservative first-pass timing window used in the script is only a diagnostic timing device for separating early reflected signal from later bottom-return contamination. It should not be interpreted as an independent physical law.

---

## Interpreting `diagnose_boundary_symbol.py`

This is the main numerical diagnostic for the local spin--momentum filtering mechanism. It corresponds most directly to Supplemental Material S7, with the analytic mechanism derived in S6.

The script uses the stored top-row roof trace and the discrete tangential symbol to test the two-branch response

$$
B_{\mathrm{br}}(\epsilon,\xi)
= e^{-R\epsilon}\Pi_+(\xi)+e^{R\epsilon}\Pi_-(\xi),
\qquad R=|\xi|.
$$

The central first-order coefficient is the covariance form

$$
\Lambda_\omega=\sqrt{\omega}\,\beta_\omega,
\qquad
\beta_\omega=\mathrm{Cov}_{\nu_0}(t,s a_\omega),
\qquad
s=R/\sqrt{\omega}.
$$

Important output families are:

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
| `boundary_layer_covariance_integrands.npz` | Saved rate integrands used to inspect covariance contributions. |
| `summary.json` | Detector budget, covariance coefficients, finite-epsilon checks, and Duhamel summaries. |

This script deliberately does **not** neglect `Pi_-`. The full two-branch map is the central object. The auxiliary inward depth used in the diagnostic is a local boundary-symbol probe, not a physical detector thickness.

---

## Recommended repository practice

A clean repository layout is:

```text
solvers/        # the three representative scripts and this README
runs/           # local or scratch output; do not commit
figures/        # selected final figures only
data/           # selected compact arrays used for figures/tables
docs/           # explanations, figure maps, and post-processing notes
```

Recommended `.gitignore` entries:

```gitignore
runs/
scratch/
output/
outputs/
__pycache__/
*.pyc
*.log
**/stdout.txt
**/simulation_log.txt
```

Avoid globally ignoring `*.npy` and `*.npz` if the repository intentionally tracks selected compact data in `data/`. Instead, keep raw run output under `runs/` and commit only selected, documented arrays.

---

## Citation

If you use these scripts, cite the paper and the accompanying data/code repository:

```text
A. Jozani,
"Spin--Momentum Filtering by an Absorbing Boundary Delays Quantum Detection in a Harmonic Waveguide,"
PRL-format manuscript with Supplemental Material, 2026.

A. Jozani,
Boundary-Spin-Momentum-Filtering-Data,
GitHub repository: https://github.com/jloOop/Boundary-Spin-Momentum-Filtering-Data
```

For the broader detection-time framework and the Crank--Nicolson/GMRES implementation background, also cite:

```text
A. Jozani and R. Tumulka,
"Detection Time Distribution Predicted Using Absorbing Boundary Conditions and Imaginary Potentials,"
Physical Review Research, accepted / forthcoming.
```

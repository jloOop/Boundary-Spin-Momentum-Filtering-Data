# Representative spinor-ABC solvers

This folder contains the three representative Python/CuPy research solvers used for the numerical side of

> **A. Jozani, _Spin--Momentum Filtering by an Absorbing Boundary Delays Quantum Detection in a Harmonic Waveguide_**

and its Supplemental Material.

These scripts are included as **reproducibility and code-review material**, not as a polished Python package.  They are kept close to the production and diagnostic scripts used in the research workflow so that a reader can trace how the paper-level quantities were computed.

For an industry or scientific-computing reader, this folder demonstrates:

- GPU/CuPy finite-difference simulation of 3D spinor quantum dynamics,
- sparse non-Hermitian Hamiltonian assembly,
- Crank--Nicolson time stepping with GMRES/Krylov linear solves,
- ghost-point implementation of a spin-coupled absorbing boundary,
- probability-current, reflection, boundary-symbol, and Duhamel diagnostics,
- compact output suitable for post-processing and reproducibility.

The physical observable throughout the project is the **detector-present roof flux**, equivalently the **norm loss** of the non-unitary spinor absorbing-boundary evolution.  These scripts should therefore be read as simulations and diagnostics of a detector model, not as detector-free arrival-time simulations.

---

## Folder contents

```text
python-scripts/Solvers/
  solver_spinor_abc_gaussian.py
  diagnose_reflection_time_Decomposition.py
  diagnose_boundary_symbol.py
  README.md
```

If this folder is later moved to top-level `solvers/`, replace `python-scripts/Solvers/` by `solvers/` in the commands below.

---

## Scientific background

The simulations solve a two-component Pauli/Schrödinger equation in a finite harmonic waveguide,

$$
i\partial_t\Psi=
\left[
-\frac{1}{2}\Delta+
\frac{1}{2}\omega^2\bigl((x-x_c)^2+(y-y_c)^2\bigr)
\right]\Psi,
\qquad
\Psi=(\psi_\uparrow,\psi_\downarrow)^T.
$$

The lower face and transverse side walls are reflecting Dirichlet walls.  The detecting surface is the roof

$$
\Sigma_L=\{z=L\},
$$

where the spin-coupled absorbing boundary condition is imposed:

$$
(\sigma\cdot\nabla)\Psi=i\kappa\sigma_z\Psi.
$$

In components,

$$
\partial_z\psi_\uparrow
=i\kappa\psi_\uparrow-(\partial_x-i\partial_y)\psi_\downarrow,
$$

$$
\partial_z\psi_\downarrow
=i\kappa\psi_\downarrow+(\partial_x+i\partial_y)\psi_\uparrow.
$$

For the Pauli current, this boundary gives the outward roof-flux density

$$
g(t;\omega)
=\kappa\int_{\Sigma_L}\Psi^\dagger\Psi\,dx\,dy
=-\frac{d}{dt}\|\Psi_t\|^2
$$

The survival probability is

$$
S(t;\omega)=\||\Psi_t\||^2,
$$

and the finite-window restricted mean detection time used in the paper is

$$
\mu^*(T;\omega)
=\int_0^T S(t;\omega)\,dt
=E[\min(\tau,T)].
$$

The central mechanism is that the spinor ABC is **not a scalar sink**.  At the roof, tangential Fourier modes see the boundary matrix

$$
C(\xi)=i\kappa I+(\hat z\times\xi)\cdot\sigma,
$$

with eigenbranches

$$
\lambda_\pm(\xi)=i\kappa\pm |\xi|.
$$

For the harmonic transverse ground state,

$$
|\xi|\sim \ell_\perp^{-1}\sim\sqrt{\omega}.
$$

This local boundary scale is the source of the confinement-dependent detector response studied in the paper.

---

## Solver map

| Script | Main role | Paper/Supplement connection | Main outputs |
|---|---|---|---|
| `solver_spinor_abc_gaussian.py` | Main Gaussian spinor-ABC production solver.  Evolves the 3D spinor wavefunction, records the norm/survival curve, and provides the compact data needed for `S(t;omega)`, `g(t;omega)`, `D_T`, and `mu^*(T;omega)`. | Letter Fig. 1; Supplement S2 for model/grid setup, S3 for the confinement sweep, and S4 for the finite-window statistic. | `prob_times.npy`, `total_probs.npy`, `constants.npz`, `simulation_log.txt`, `stdout.txt` |
| `diagnose_reflection_time_Decomposition.py` | Reflection, time-window, and near-roof diagnostic.  Tests whether the delayed sector is best understood as ordinary propagating reflection or as a near-roof spinor-ABC boundary response with finite-guide memory. | Supplement S4 for finite-window early/late bookkeeping; S7 for near-roof two-branch/Duhamel diagnostics; S8 for finite-guide memory interpretation. | `summary.json`, `diag_times.npy`, `roof_J*.npy`, `kz_*.npy`, `near_roof_mass.npy`, `W_plus_*.npy`, `W_minus_*.npy`, `Q_minus_*.npy`, `E_full_*.npy`, `E_plus_*.npy` |
| `diagnose_boundary_symbol.py` | Boundary-symbol, covariance, finite-epsilon, and Duhamel diagnostic.  This is the most direct numerical diagnostic for the local two-branch spin--momentum mechanism. | Letter boundary-mechanism section; Supplement S6 for the analytic boundary symbol and covariance form; Supplement S7 for the finite-grid boundary-symbol and Duhamel checks. | `summary.json`, `diag_times.npy`, `det_rate_toprow.npy`, `P_total_series.npy`, `bl_rate*.npy`, `bl_*epsSteps_*.npy`, `duhamel_*depthSteps_*.npy`, `boundary_layer_covariance_integrands.npz` |

Recommended reading order:

1. `solver_spinor_abc_gaussian.py` for the production survival/norm-loss data.
2. `diagnose_reflection_time_Decomposition.py` for reflection, early/late, and near-roof diagnostics.
3. `diagnose_boundary_symbol.py` for the local spin--momentum boundary-symbol mechanism.

---

## Running the scripts

These scripts require a CUDA-capable GPU and a CuPy installation compatible with the local CUDA driver.  They were written for GPU/HPC execution, not for small CPU-only runs.

From the repository root:

```bash
mkdir -p runs

# Main Gaussian spinor-ABC confinement/norm-loss run
OMEGA=300 OUTDIR=./runs \
python python-scripts/Solvers/solver_spinor_abc_gaussian.py

# Reflection, timing-window, kz, Pi_+/Pi_-, and near-roof diagnostics
OMEGA=100 OUTDIR=./runs \
python python-scripts/Solvers/diagnose_reflection_time_Decomposition.py

# Boundary-symbol covariance, finite-epsilon, and Duhamel diagnostics
OMEGA=200 OUTDIR=./runs NX=100 NY=100 NZ=1500 DT=2.5e-4 TFINAL=20 \
python python-scripts/Solvers/diagnose_boundary_symbol.py
```

For cluster runs, set `OUTDIR` to a scratch or project-storage directory.  Do not commit raw run directories unless a small selected file is explicitly documented as reproducibility data.

Each run writes a `constants.npz` file.  Use this file as the source of truth for the actual grid, time step, confinement parameter, and packet parameters used in that run.

---

## Interpreting `solver_spinor_abc_gaussian.py`

This is the compact production solver for the Gaussian spinor-ABC confinement runs.  It writes the survival data

```text
prob_times.npy
 total_probs.npy
 constants.npz
 simulation_log.txt
 stdout.txt
```

Here `total_probs.npy` is the discrete survival curve

$$
S(t;\omega)=\|\Psi_t\|^2.
$$

From this one obtains the finite-window detected fraction

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

Minimal post-processing sketch:

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

# Add an interpolated endpoint at T if the saved grid does not land exactly on T.
if tT[-1] < T and t[-1] >= T:
    ST_at_T = np.interp(T, t, S)
    tT = np.concatenate((tT, [T]))
    ST = np.concatenate((ST, [ST_at_T]))

trapz = getattr(np, "trapezoid", None)
if trapz is None:
    trapz = np.trapz

D_T = 1.0 - ST[-1]
mu_star = trapz(ST, tT)

# Useful diagnostic only; raw numerical differentiation amplifies small fluctuations.
g = -np.gradient(S, t)

print("D_T      =", D_T)
print("mu_star =", mu_star)
```

For publication-quality flux plots, use the same smoothing/post-processing pipeline used for the paper figures rather than relying only on a raw numerical derivative.

---

## Interpreting `diagnose_reflection_time_Decomposition.py`

This diagnostic asks:

> Is the delayed signal mainly ordinary propagating reflection, or is it a near-roof spinor-ABC boundary response with finite-guide memory?

It should be read together with the finite-window bookkeeping in Supplement S4 and the boundary-response discussion in S7--S8.  The early/late split is a diagnostic decomposition of the detector-present roof-flux signal.  It is not a new detector model and not a detector-free arrival-time law.

Important output families:

| Output | Meaning |
|---|---|
| `roof_Jnet.npy`, `roof_Jplus.npy`, `roof_Jminus.npy` | Pauli-current roof flux and local backflow bookkeeping. |
| `roof_in_J*.npy`, `roof_in_rho_int.npy` | Same style of checks one grid point below the roof. |
| `kz_P_plus.npy`, `kz_P_minus.npy`, `kz_R.npy` | Windowed longitudinal FFT diagnostic in a slab below the roof.  Positive `k_z` is upgoing; negative `k_z` is downgoing. |
| `kz_slab_mass.npy` | Mass in the slab used for the longitudinal FFT diagnostic. |
| `near_roof_mass.npy` | Probability stored in a near-roof slab. |
| `W_plus_roof.npy`, `W_minus_roof.npy`, `Q_minus_roof.npy` | Tangential spin--momentum branch content at the roof. |
| `E_full_depthSteps_*.npy` | Relative error of the full two-branch homogeneous boundary-symbol continuation. |
| `E_plus_depthSteps_*.npy` | Historical plus-only comparison.  The full two-branch diagnostic is the central one. |
| `summary.json` | Integrated reflection/backflow/kz/Pi/Duhamel summaries. |

Important interpretation note: this script is a **diagnostic/reflection-oriented variant**, not the main production detector solver.  It is included to probe backflow, longitudinal `k_z` content, branch composition, and near-roof Duhamel/continuation errors.  The main detector-present law remains the roof flux/norm loss of the spinor-ABC evolution.

---

## Interpreting `diagnose_boundary_symbol.py`

This is the main numerical diagnostic for the local spin--momentum filtering mechanism.  It corresponds most directly to Supplement S7, with the analytic mechanism derived in Supplement S6.

The script uses the stored top-row roof trace and the discrete tangential symbol to test the full two-branch response

$$
B_{\mathrm{br}}(\epsilon,\xi)
=e^{-R\epsilon}\Pi_+(\xi)+e^{R\epsilon}\Pi_-(\xi),
\qquad
R=|\xi|.
$$

The central first-order coefficient is the covariance form

$$
\Lambda_\omega=\sqrt{\omega}\,\beta_\omega,
\qquad
\beta_\omega=\mathrm{Cov}_{\nu_0}(t,s a_\omega),
\qquad
s=R/\sqrt{\omega}.
$$

Important output families:

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

This script deliberately does **not** neglect `Pi_-`.  The full two-branch map is the central object.  The auxiliary inward depth used in the diagnostic is a local boundary-symbol probe, not a physical detector thickness.

---

## Interpretation rules

Use these conventions when describing figures or outputs generated from these scripts:

1. **Detector-present, not detector-free.**  The blue flux curves and norm-loss data represent the detector-present spinor-ABC evolution.
2. **Bohmian histograms, when shown elsewhere, are Monte Carlo samples of the same detector-present flux law.**  They should not be described as a separate no-detector Bohmian arrival-time proposal.
3. **The finite-window fit is not universal.**  The fit

   $$
   \mu^*(20;\omega)\simeq 4.084+0.638\sqrt{\omega}
   $$

   is specific to the chosen packet, detector parameter, guide geometry, and observation window.  The robust feature is the boundary scale

   $$
   R=|\xi|\sim\sqrt{\omega}.
   $$

4. **The early/late split is diagnostic bookkeeping.**  It separates finite-window contributions to the computed roof flux; it is not an exact microscopic decomposition into first-pass and return amplitudes.
5. **The boundary-symbol depth is auxiliary.**  The diagnostics at small inward depth test the local boundary relation and Duhamel remainder.  Physical detection remains at the roof `z=L`.

---

## Repository practice

These scripts can generate large output directories.  A clean workflow is:

```text
python-scripts/Solvers/   # commit these representative scripts and this README
runs/                     # local or scratch output; do not commit
figures/                  # selected final figures only
data/                     # selected compact arrays used for figures/tables
docs/                     # figure maps, reproducibility notes, and data policy
```

Recommended `.gitignore` style:

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

Avoid globally ignoring all `*.npy` and `*.npz` if the repository intentionally tracks selected compact data in `data/`.  Instead, keep raw run output under `runs/` and commit only selected, documented arrays.

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

---

## One-sentence summary

These solvers implement and diagnose a GPU/HPC finite-difference model of a spinor absorbing boundary whose local two-branch spin--momentum impedance converts transverse confinement, through `|xi| ~ sqrt(omega)`, into a detector-present delay of the roof-flux response.

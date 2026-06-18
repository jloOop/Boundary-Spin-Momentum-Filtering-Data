# Main solver scripts for Boundary-Spin-Momentum-Filtering-Data

This folder contains the three representative production/diagnostic scripts for the spinor absorbing-boundary project.  The files are intended to be readable on GitHub while preserving the working numerical solver statements from the uploaded versions.

## Script map

| Script | Public role | Paper / Supplement connection | Main outputs |
|---|---|---|---|
| `746SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py` | Main spinor-ABC norm-loss solver for confinement sweeps | Main roof-flux curves, detected fraction, and restricted-mean fit | `prob_times.npy`, `total_probs.npy`, `constants.npz`, logs |
| `953Reflection_Diag.py` | Reflection / finite-window diagnostic | Reflection checks and first-pass versus later-return interpretation | `kz_R.npy`, `roof_J*.npy`, `Q_minus_*.npy`, `E_*depth*.npy`, `summary.json` |
| `973Cov_Diag_pro.py` | Boundary-symbol covariance and Duhamel diagnostic | Supplemental boundary-symbol / finite-grid diagnostics | `bl_*.npy`, `duhamel_*.npy`, `boundary_layer_covariance_integrands.npz`, `summary.json` |

## Scientific convention used in these scripts

The detector observable is the roof flux, equivalently the norm loss of the nonunitary spinor-ABC evolution:

```text
g(t; omega) = - d ||Psi_t||^2 / dt
S(t; omega) = ||Psi_t||^2
D_T(omega) = integral_0^T g(t; omega) dt
mu*(T; omega) = integral_0^T S(t; omega) dt
```

Bohmian histograms, when present in figures outside these scripts, should be described as Monte Carlo samples of the same detector-present flux law, not as a separate no-detector arrival-time proposal.

## Suggested run pattern

Run into a scratch directory, not into the repository root:

```bash
export OUTDIR=/path/to/scratch/spinor_abc_runs
export OMEGA=300
python python-scripts/746SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py
```

The diagnostic scripts are heavier and should usually be run as controlled single-parameter checks or Slurm array jobs.



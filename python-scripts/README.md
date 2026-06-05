# Python scripts

This folder contains the main solver and diagnostics used for the Gaussian spinor-ABC follow-up project.

## Scripts

### `730SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py`

Main 3D CuPy Crank–Nicolson/GMRES solver for the Gaussian spinor-ABC simulations.

Main outputs include:

```text
constants.npz
prob_times.npy
total_probs.npy
rho_prob_t*.npy
bohm_t_hit.npy
bohm_arrived_mask.npy
bohmian_traj_selected.npy
bohm_times.npy
```

Use this script for selected density snapshots, arrival histograms, and trajectory visualizations.

### `953Reflection_Diag.py`

Reflection/backflow diagnostic script. It keeps the TDSE/CN/spinor-ABC implementation but adds spectral diagnostics:

```text
roof_Jnet.npy
roof_Jplus.npy
roof_Jminus.npy
kz_P_plus.npy
kz_P_minus.npy
kz_R.npy
near_roof_mass.npy
W_plus_roof.npy
W_minus_roof.npy
Q_minus_roof.npy
E_full_depthSteps_*.npy
summary.json
```

### `972Cov_Diag_pro.py`

Boundary-layer covariance diagnostic script for the two-branch spinor-ABC mechanism.

Main outputs include:

```text
det_rate_toprow.npy
bl_rate0.npy
bl_rate_Ra.npy
bl_rate_sa.npy
bl_rate_R2.npy
bl_rate_s2.npy
duhamel_E_full_rel_true_depthSteps_*.npy
boundary_layer_covariance_integrands.npz
summary.json
```

## Running on a cluster

Set `OUTDIR` before running:

```bash
export OUTDIR=/path/to/project/output
python python-scripts/730SpinorXY_Gauss_Bohm_DirichletABC_ZIB.py
```

For sweeps, set parameters by editing the script or using environment variables where supported.

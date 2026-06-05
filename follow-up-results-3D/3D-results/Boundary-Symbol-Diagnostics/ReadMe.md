# Boundary-symbol diagnostics

This folder is for the data and plots supporting the local boundary-symbol mechanism.

Relevant script:

```text
python-scripts/972Cov_Diag_pro.py
```

The diagnostic computes the two-branch boundary-layer map, the covariance coefficient, finite-epsilon checks, and full two-branch Duhamel/slab remainder diagnostics.

Suggested files:

```text
summary.json
boundary_layer_covariance_integrands.npz
bl_rate0.npy
bl_rate_Ra.npy
bl_rate_sa.npy
bl_rate_R2.npy
bl_rate_s2.npy
boundary_covariance_summary.csv
duhamel_summary.csv
```

Suggested figures:

```text
beta_cov_vs_omega.png
Lambda_vs_sqrt_omega.png
Duhamel_E_full_depths.png
boundary_budget_error.png
```

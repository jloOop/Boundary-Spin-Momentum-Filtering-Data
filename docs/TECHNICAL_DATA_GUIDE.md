# Technical data guide

This page maps the manuscript and supplemental material to the repository data, scripts, PNGs, and GIFs.

## 1. Main physical observable

The detector-present observable is the roof flux

```math
g(t;\omega)=\kappa\int_{\Sigma_L}\rho(x,y,L,t)\,dx\,dy=-\frac{d}{dt}\|\Psi_t\|^2 .
```

The survival probability is `S(t; omega)=||Psi_t||^2`, and the finite-window restricted mean survival time is

```math
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt.
```

Repository files connected to this observable:

```text
data/selected-runs/<run>/prob_times.npy
data/selected-runs/<run>/total_probs.npy
data/selected-runs/<run>/det_rate_toprow.npy
notebooks/Paper_Presentation2.ipynb
```

## 2. Main confinement sweep

Purpose: show that increasing `omega` narrows the transverse harmonic state, suppresses the prompt roof-flux peak, reduces the detected fraction, and enhances the delayed oscillatory sector.

Recommended folder:

```text
follow-up-results-3D/3D-results/Paper-Figures-and-PNGs/main-confinement-sweep/
```

Suggested files:

```text
omega_001_roof_flux.png
omega_100_roof_flux.png
omega_200_roof_flux.png
omega_300_roof_flux.png
mu_star_vs_sqrt_omega.png
main_confinement_sweep.csv
```

Suggested table columns:

```text
omega,sqrt_omega,D_T,mu_star,tau_cond_T,t_E,w_L,t_L,delta_t_cond_L
```

## 3. Density evolution GIFs

Purpose: show the actual 3D density evolution from the solver outputs `rho_prob_t*.npy`.

Generator:

```text
notebooks/Spin_Gif.ipynb
```

Inputs:

```text
constants.npz
x.npy, y.npy, z.npy          optional, if saved
rho_prob_t*.npy
```

Outputs:

```text
midplanes_t*.png
contour_xy_t*.png
scatter_t*.png
isosurf_t*.png
slices_bar_t*.png
midplanes.gif
contour_xy.gif
scatter.gif
isosurf.gif
slices_bar.gif
```

Public location:

```text
follow-up-results-3D/3D-results/TimeEvolution-WaveFunction-Gifs/
```

Use `Links-to-Gifs.md` when the GIFs are too large for direct GitHub upload.

## 4. Bohmian trajectory visualizations

Purpose: show that the Bohmian histograms and trajectory samples are used as Monte Carlo samples of the same detector-present spinor-ABC flux law, not as a separate no-detector arrival-time proposal.

Generator:

```text
notebooks/Traj3D.ipynb
```

Inputs:

```text
bohmian_traj_selected.npy
bohm_times.npy
traj_indices.npy
bohm_arrived_mask_selected.npy
bohm_t_hit_selected.npy
```

Outputs:

```text
traj_sel_3D.png
traj_sel_projections.png
```

Public location:

```text
follow-up-results-3D/3D-results/Bohmian-Trajectories/
```

## 5. Boundary-symbol and covariance diagnostics

Purpose: support the local boundary mechanism. The spinor absorbing boundary has two tangential branches

```math
\lambda_\pm(\xi)=i\kappa\pm |\xi|.
```

For the harmonic transverse family,

```math
|\xi|\sim \sqrt{\omega}.
```

Main diagnostic script:

```text
python-scripts/972Cov_Diag_pro.py
```

This script evaluates the full two-branch boundary-layer map, the covariance coefficient, finite-epsilon checks, RMST coefficients, and Duhamel/slab remainder checks.

Important outputs:

```text
summary.json
diag_times.npy
P_total_series.npy
det_rate_toprow.npy
bl_rate0.npy
bl_rate_Ra.npy
bl_rate_sa.npy
bl_rate_R2.npy
bl_rate_s2.npy
bl_branch_Q_minus.npy
duhamel_E_full_rel_true_depthSteps_*.npy
boundary_layer_covariance_integrands.npz
```

Public location:

```text
follow-up-results-3D/3D-results/Boundary-Symbol-Diagnostics/
```

## 6. Reflection diagnostics

Purpose: distinguish the spinor-ABC boundary response from ordinary propagating reflection.

Main diagnostic script:

```text
python-scripts/953Reflection_Diag.py
```

The script computes roof backflow, a windowed longitudinal `k_z` decomposition in a slab below the roof, tangential `Pi_+ / Pi_-` decomposition of the boundary matrix, and Duhamel relative-error diagnostics.

Important outputs:

```text
summary.json
diag_times.npy
P_total_series.npy
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
E_plus_depthSteps_*.npy
Q_minus_delta_depthSteps_*.npy
```

Public location:

```text
follow-up-results-3D/3D-results/Reflection-Diagnostics/
```

## 7. Parameter scans

Purpose: document that `kappa`, `sigma_z`, `k_z`, and box length change coefficients and tails, while the local confinement scale remains `|xi| ~ sqrt(omega)`.

Recommended folders:

```text
follow-up-results-3D/3D-results/Parameter-Scans/kappa-scan/
follow-up-results-3D/3D-results/Parameter-Scans/sigma-z-scan/
follow-up-results-3D/3D-results/Parameter-Scans/kz-scan/
follow-up-results-3D/3D-results/Parameter-Scans/large-box-check/
```

Suggested files:

```text
kappa_scan_summary.csv
sigma_z_fit_coefficients.csv
kz_scan_summary.csv
large_box_check_summary.csv
```

## 8. Recommended run naming

Use simple run names that preserve the parameter values:

```text
omega_001_kappa_pi_kz_pi_sigmaz_0p5
omega_100_kappa_pi_kz_pi_sigmaz_0p5
omega_200_kappa_pi_kz_pi_sigmaz_0p5
omega_300_kappa_pi_kz_pi_sigmaz_0p5
```

## 9. What to upload directly and what to link

Upload directly:

```text
README.md
*.py
*.ipynb
*.csv
*.json
small *.png
small representative *.gif
small selected *.npz
```

Do not upload huge raw output directories directly. For large files, use GitHub Releases or an external archive.

Use this page for release links:

```text
follow-up-results-3D/3D-results/TimeEvolution-WaveFunction-Gifs/Links-to-Gifs.md
```

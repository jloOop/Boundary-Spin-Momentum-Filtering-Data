# S4 time-window and reflection diagnostics

This folder contains compact CSV summaries for the Supplement S4 of the paper time-window bookkeeping and the related reflection / finite-guide-memory diagnostics.

These tables are derived from the small JSON outputs of `diagnose_reflection_time_decomposition.py`. They are intended for GitHub readability: a reviewer can inspect the key numerical checks without downloading large raw `.npy` arrays.

## Files

| File | Rows | Purpose |
|---|---:|---|
| `s4_timewindow_cutoff_sweep.csv` | one row per `omega` | Shows the monitoring-plane timing estimates used to justify the common cutoff `t_cut = 3.6`. |
| `s4_reflection_firstbounce_by_plane.csv` | one row per `omega` and monitoring plane | Records incident/reflection windows, first-bounce integrated fluxes, and first-bounce reflection ratios. |
| `s4_reflection_window_ratios_by_plane.csv` | one row per `omega` and monitoring plane | Separates the no-return window from the post-cut window. This is the cleanest table for showing early versus late finite-guide-memory behavior. |
| `s4_roof_detector_budget_sweep.csv` | one row per `omega` | Checks the detector-present roof flux against norm loss and records roof detection-time summary statistics. |
| `s4_timewindow_decomposition_highlights.csv` | one row per `omega` | Short human-readable summary for README previews. |

## Interpretation

The physical detector observable is always the roof flux, equivalently the norm loss of the spinor-ABC evolution. The interior monitoring planes are not additional detectors. They are diagnostic surfaces placed below the roof to track the first upward passage of the packet, estimate an effective axial speed, and identify when later finite-guide return signals can start to contaminate the early-time signal.

The quantities measured on the interior planes, such as $J_-/J_+$ ratios and upward/downward timing windows, are therefore diagnostic bookkeeping tools. They help justify the fixed cutoff $t_{\rm cut}=3.6$ used in Supplement S4 and separate the early first-pass sector from later finite-guide return/memory structure. They should not be interpreted as a second detection law or as an alternative arrival-time distribution.

The common cutoff is not fitted from the roof-flux data and is not a reflection coefficient. It is a conservative timing marker chosen to stay before the earliest estimated bottom-return contamination across the sweep.

Large time-series arrays such as `roof_J*.npy`, `kz_*.npy`, and `E_full_*.npy` should normally stay outside GitHub. The CSVs here are the public audit layer.



# Supplement S7 boundary-layer and Duhamel diagnostics

This file documents the S7 boundary-symbol tables in this folder.

The CSV files here are compact public summaries generated from the JSON outputs of
`diagnose_boundary_symbol.py`. They are intended as the readable data layer between the
large GPU/HPC diagnostic runs and the paper/Supplement discussion.

The physical observable remains the detector-present roof flux / norm loss of the
spinor-ABC evolution. The boundary-layer quantities here are **diagnostic probes** of the
local spinor absorbing boundary symbol. They are not additional detectors and not a
physical detector thickness.

## Files

| File | Rows | Purpose |
|---|---:|---|
| `s7_boundary_layer_highlights.csv` | one row per `omega` | Short public summary for README previews. |
| `s7_boundary_symbol_diagnostics.csv` | one row per `omega` | Main S7 one-grid table: detector budget, scaled tangential moment, covariance coefficient, and one-grid two-branch Duhamel error. |
| `s7_detector_budget_sweep.csv` | one row per `omega` | Roof detector-budget check: integrated detector rate versus norm loss. |
| `s7_finite_epsilon_checks.csv` | one row per `omega` and epsilon step | Exact finite-epsilon two-branch checks and Taylor diagnostics. |
| `s7_duhamel_depth_sweep.csv` | one row per `omega` and depth step | Duhamel/slab depth sweep. The one-grid depth is the central S7 diagnostic; larger depths are stress tests. |

## Interpretation

The S7 diagnostic uses the stored roof trace and the finite-difference tangential symbol
to test the local two-branch spin--momentum map

```math
B_{\rm br}(\epsilon,\xi)
= e^{-R\epsilon}\Pi_+(\xi)+e^{R\epsilon}\Pi_-(\xi),
\qquad R=|\xi|.
```

The one-grid auxiliary depth is

```math
\epsilon_g=h_z,
```

and is only a numerical probe of the boundary response. It should not be described as a
physical absorbing layer.

The central finite-window covariance coefficient is

```math
\Lambda_\omega = \sqrt{\omega}\,\beta^{\rm bl}_\omega,
\qquad
\beta^{\rm bl}_\omega = {\rm Cov}_{\nu_0}(t,s a_\omega),
\qquad s=R_h/\sqrt{\omega}.
```

The table checks that the scaled tangential moment `E_s2` remains order one, that the
detector budget closes, and that the one-grid full two-branch Duhamel diagnostic remains
controlled. The `Pi_-` branch is included in the central diagnostic; the plus-only value is
kept only as a historical comparison.

## Preview

|   omega |   eps_g_sqrt_omega |   detector_budget_rel_error |     E_s2 |    beta_bl |   Lambda_omega |   E2br_weighted_mean_active |   E_plus_only_historical_weighted_mean_active |
|--------:|-------------------:|----------------------------:|---------:|-----------:|---------------:|----------------------------:|----------------------------------------------:|
|       1 |          0.0133333 |                 1.82843e-05 | 0.978424 | -0.0802817 |     -0.0802817 |                  0.00179203 |                                      0.687879 |
|      50 |          0.0942809 |                 0.000400124 | 2.74361  | -0.0249063 |     -0.176114  |                  0.0148278  |                                      0.339313 |
|     100 |          0.133333  |                 0.000722047 | 3.13976  |  0.0322092 |      0.322092  |                  0.0299912  |                                      0.276732 |
|     200 |          0.188562  |                 0.000676803 | 2.91885  |  0.0214723 |      0.303665  |                  0.054401   |                                      0.223921 |
|     300 |          0.23094   |                 0.00023723  | 2.65044  |  0.0104942 |      0.181765  |                  0.0760119  |                                      0.223783 |

## Repository practice

Commit these CSV tables and the compact JSON summaries in
`data/representative-diagnostics/s7-boundary-layer-json/`.

Do **not** commit the large generated `.npy` arrays unless a small selected file is needed
for a specific figure or loader. Large arrays such as `bl_rate*.npy`,
`bl_*epsSteps_*.npy`, `duhamel_*depthSteps_*.npy`, and
`boundary_layer_covariance_integrands.npz` should normally stay in external run storage.


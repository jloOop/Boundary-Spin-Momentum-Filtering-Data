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

The physical observable remains the detector-present roof flux / norm loss. The interior-planes ratios and timing windows are used for diagnostics only such as e.g., measuring the speeed of packet and so on. They help justify the fixed cutoff used in Supplement S4 and separate early first-pass passage of wavepacket from late finite-guide return sectors structure.

The common cutoff is not fitted from the roof-flux data and is not a reflection coefficient. It is a conservative timing marker chosen to stay before the earliest estimated bottom-return contamination across the sweep.

Large time-series arrays such as `roof_J*.npy`, `kz_*.npy`, and `E_full_*.npy` should normally stay outside GitHub. The CSVs here are the public audit layer.

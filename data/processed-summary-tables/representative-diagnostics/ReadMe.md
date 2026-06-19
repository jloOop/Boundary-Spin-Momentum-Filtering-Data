# Representative S4 time-window diagnostics

This folder contains compact JSON audit files for the Supplement S4 time-window diagnostics.

The CSV files in `data/processed-summary-tables/` are the main public data products. They summarize the cutoff sweep, roof detector budget, first-bounce plane diagnostics, and no-return/post-cut current ratios.

The JSON files here provide the provenance behind those tables.

- `metrics/`: compact diagnostic summaries for each value of $\omega$. These are the recommended files for reproducing the public CSV tables.
- `full-summary/`: lower-level solver summaries with detailed window choices, thresholds, search intervals, and peak-finding diagnostics. These are included only as representative audit files.

The detector observable is always the roof flux / norm loss. Interior-plane timing windows and $J_-/J_+$ ratios are diagnostic tools used to estimate first-pass axial motion and separate early first-pass behavior from later finite-guide return structure. They are not additional detectors and not a separate arrival-time law.

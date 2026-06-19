
# S4 representative JSON diagnostics

This folder contains sanitized JSON outputs used to build the S4 time-window/reflection CSV tables.

The original absolute HPC scratch paths were replaced by run-directory basenames. These JSON files are small enough to keep in GitHub as an audit trail, while the large raw arrays should remain in external storage or ignored run directories.

Recommended public use:

1. Point readers first to `data/processed-summary-tables/`.
2. Keep these JSON files as provenance for scientific readers who want more detail.
3. Do not present these JSON files as the main result; the CSV tables are the public-facing summary.

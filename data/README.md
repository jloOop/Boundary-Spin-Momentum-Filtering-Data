# Data guide

| Collection | Role |
|---|---|
| [centered/](centered/README.md) | Revised detector-only saved records, selected figures, native reductions and supporting-results gallery. |
| [processed-summary-diagnostics-tables/](processed-summary-diagnostics-tables/) | Existing legacy summaries and diagnostics, preserved at their public paths. Their older numbering, run selection and T=20 fit are not the revised main result. |
| [3DGifs_Simulations/](3DGifs_Simulations/) | Protected legacy animation collection and release links, unchanged. Preparations and numerical implementations differ from the revised centered θ=0 reference family. |

The [manifest](centered/provenance/manifest.json) identifies each new distributed record, source hash, transformation, run selection and reproduction level. Compact NPZ exports preserve selected native columns and all rows; they are not byte-identical copies of the original CSVs. No large run folders or manuscript inventory reports are included.

Use native accepted-step roof events for the revised detector probability. Sampled survival, endpoint roof-density rate and terminal-cell inflow are not interchangeable. In lower-absorbing-boundary controls, total norm loss includes escape and cannot be assigned entirely to the upper detector. See [scope](../PROJECT_SCOPE.md) and [tested commands](../REPRODUCING.md).

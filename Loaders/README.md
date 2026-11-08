# Saved-data analysis and legacy loaders

The centered manuscript workflow is `reproduce_centered.py`, with checks in `verify_centered.py` and small CPU modules in `centered/`. See [REPRODUCING.md](../REPRODUCING.md) for the tested commands and [source distinctions](../PROJECT_SCOPE.md). It reads only committed compact records, never imports a production solver, and writes to ignored `build/`.

The original top-level loader programs and their behavior are retained. They belong to the legacy output formats:

| Existing path | Actual required inputs / limitation |
|---|---|
| `plot_detection_time_distribution.py` | Reads the current working directory; requires norm/constants records and Bohm arrival/mask arrays. The current implementation requires the trajectory pair even for its quantitative plot. It differentiates sampled survival and is not the centered native-event reducer. |
| `make_density_gifs.py` | Finds constants through ancestors of the script path and needs completed density snapshots. A repository-root invocation alone does not provide them. Per-frame intensity scaling makes it qualitative. |
| `plot_bohmian_trajectories.py` and `trajectory_*.py` | Require compatible trajectory arrays; these are not supplied by the centered saved-data package. |

No legacy source program was changed to repair these limitations. Existing [animation links](../data/3DGifs_Simulations/) remain untouched. Detector-free comparison current, native detector events, endpoint roof density and numerical terminal-cell inflow must not be substituted for each other. The centered workflow records these distinctions explicitly.

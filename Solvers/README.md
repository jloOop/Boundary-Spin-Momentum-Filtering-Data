# Generating sources and legacy solvers

The selected historical generating snapshots for the centered Cartesian study are under [centered/](centered/), linked to their actual configurations by the [single provenance manifest](../data/centered/provenance/manifest.json). They are provenance and fresh-simulation source material, not restart states or a tested GPU execution package. Source transformations, where necessary, only replace machine-specific output paths; the source and distributed hashes remain distinct.

The centered runs use a ghost-point roof closure with a compatible terminal longitudinal half-weight, endpoint-excluded stored coordinates and the recorded numerical roof position. Preserve the coordinate, state, matrix and solve precision fields in each run configuration. Norm loss, detector events and numerical cast/clamp corrections are separate records.

The original top-level files are retained unchanged:

- `solver_spinor_abc_gaussian.py`, with `solver_bookkeeping.py` and `solver_metadata.py`, is the legacy one-sided/full-weight Gaussian implementation.
- `diagnose_reflection_time_Decomposition.py` and `diagnose_boundary_symbol.py` supply the older diagnostic workflow and data conventions.

Production sources require a CUDA-capable GPU, a compatible CuPy build, NumPy and cupyx sparse/GMRES facilities. They have not been imported or executed for this release's CPU checks. Existing `environment.yml` and `requirements.txt` are legacy environments; the small tested saved-data environment is [requirements-centered.txt](../requirements-centered.txt).

Use [REPRODUCING.md](../REPRODUCING.md) for the working CPU route. A saved-data reproduction, an archived-image display, and a fresh simulation are different claims. No complete wavefunction checkpoints, external transverse environment or oscillator-channel discretization are included.

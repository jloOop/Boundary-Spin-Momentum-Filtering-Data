# Data

This directory contains selected compact data, processed diagnostic tables, and representative media.

The `slices_bar.gif` files show the longitudinal marginal \(P_z(z,t)\). Since this quantity is summed over the transverse directions, it does not directly show transverse narrowing. Its purpose is to test the longitudinal propagation profile.

The early longitudinal marginal can remain nearly unchanged across different \(\omega\). This separation is the visual counterpart of the mechanism discussed in the paper: the confinement dependence is not produced by ordinary first-pass longitudinal motion, but by the spinor absorbing boundary sampling the transverse scale.



# Data and representative media

This folder contains selected compact data, processed diagnostics, and representative media for the spinor absorbing-boundary simulations.

The repository is not intended to store every raw GPU/HPC output. Full raw output should remain in scratch/project storage, external archives, or GitHub releases when appropriate.

## Folder map

| Path | Purpose | Notes |
|---|---|---|
| `3DGifs_Simulations/` | Representative density animations and selected media | Qualitative visualization; not primary quantitative evidence |
| `processed-summary-diagnostics-tables/` | Reduced tables, summaries, and processed diagnostic outputs | Best place to inspect compact numerical outputs |

## How to interpret the media

Representative GIFs visualize the evolution of probability density and related slices. For example, a longitudinal marginal or `slices_bar.gif` should be read as qualitative support for propagation, absorption, and delayed structure. It should not be overinterpreted as direct proof of transverse narrowing or as a detector-free arrival-time result.

The quantitative observables are reconstructed from survival / norm-loss arrays and diagnostic summaries:

```math
S(t;\omega)=\|\Psi_t\|^2,
\qquad
g(t;\omega)=-\frac{dS}{dt},
\qquad
D_T(\omega)=1-S(T),
\qquad
\mu^*(T;\omega)=\int_0^T S(t;\omega)\,dt.
```

## Data policy

- Commit compact reduced data that is used directly for figures or reproducibility checks.
- Document selected arrays clearly.
- Keep raw run folders out of Git unless they are intentionally selected, small, and documented.
- Use `constants.npz` and `summary.json` as parameter records for individual runs.

# Data

This directory is for selected small data and processed tables.

Do not commit full raw HPC simulation folders. Keep full raw data externally and place only selected reduced files here.


---

## Note on interpreting the GIFs

The `slices_bar.gif` files show the longitudinal marginal \(P_z(z,t)\). Since this quantity is summed over the transverse directions, it does not directly show the transverse narrowing. Its purpose is to test the longitudinal propagation profile.

For comparing transverse confinement, use `midplanes.gif` and `isosurf.gif`. Together, the diagnostics show two complementary facts:

1. the early longitudinal marginal can remain nearly unchanged across different \(\omega\), and  
2. the transverse density narrows as the oscillator length \(\ell_\perp=\omega^{-1/2}\) decreases.

This separation is the visual counterpart of the mechanism discussed in the paper: the confinement dependence is not produced by ordinary first-pass longitudinal motion, but by the spinor absorbing boundary sampling the transverse scale.


### Supplement S4 time-window diagnostics

The files in `data/processed-summary-tables/` summarize the finite-window diagnostics used to justify the common cutoff \(t_{\rm cut}=3.6\) in Supplement S4.  The cutoff is a conservative bookkeeping marker chosen before the earliest estimated bottom-return contamination across the confinement sweep.  The tables also report first-bounce reflection ratios, no-return versus post-cut reflection ratios, and roof-flux/norm-loss budget checks.

These diagnostics do not define a second detector model.  The physical observable remains the detector-present roof flux / norm loss of the spinor-ABC evolution.

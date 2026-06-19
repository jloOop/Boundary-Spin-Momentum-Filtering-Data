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



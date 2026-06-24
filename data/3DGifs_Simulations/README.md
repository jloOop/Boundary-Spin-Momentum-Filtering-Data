# 3D GIF simulations — Gaussian spinor-ABC waveguide

This folder collects representative animation bundles for the Gaussian spin-coupled absorbing boundary condition simulations used in the paper.

The simulations evolve a two-component Pauli spinor wave packet in a finite harmonic waveguide. The side walls and the lower face are Dirichlet boundaries; the detecting roof uses the spin-coupled absorbing boundary condition

$$
(\boldsymbol{\sigma}\cdot\boldsymbol{\nabla})\Psi=i\kappa\sigma_z\Psi .
$$

The animations are qualitative visualizations of the detector-present evolution: propagation toward the roof, transverse confinement, absorption, near-roof storage, and delayed oscillatory structure. The quantitative detection-time analysis comes from the survival curve and roof-flux post-processing, not from the GIFs alone.

## How these ZIP files were produced

Typical pipeline:

```text
Solvers/solver_spinor_abc_gaussian.py
    -> constants.npz, x.npy, y.npy, z.npy, rho_prob_t*.npy, prob_times.npy, total_probs.npy
Loaders/make_density_gifs.py
    -> PNG frames and GIFs in plots_general3/
Loaders/plot_detection_time_distribution.py
    -> detector-present flux plots and arrival-time summaries
Loaders/plot_bohmian_trajectories.py
    -> selected trajectory plots, when trajectory arrays are available

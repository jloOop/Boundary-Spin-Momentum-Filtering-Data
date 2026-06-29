
# Fig. 1 confinement sweep run ledger

This folder records provenance for the eight simulation runs used in the main
Fig. 1 confinement sweep of the spinor absorbing-boundary project.

The stdout-derived table is not the full figure dataset. It records run parameters,
final survival probabilities, detected fractions, and runtime information. The
publication-quality roof-flux curves g(t; omega), restricted means mu*(T; omega),
and plotted detection-time densities are reconstructed from saved arrays such as
prob_times.npy and total_probs.npy using the manuscript plotting pipeline.

The final detected fraction is checked as:

D20(omega) = 1 - S(20).

Raw stdout logs are included only as sanitized provenance records.

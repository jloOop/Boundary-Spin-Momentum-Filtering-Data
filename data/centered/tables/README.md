# Supplementary tables

Readable Tables S1–S12 for the **28 October 2026** manuscript snapshot. Each panel links its canonical CSV. The CSV retains its available precision; only the Markdown display is rounded. Printed digits do not establish numerical accuracy. Probabilities, fractions and spin polarizations are dimensionless; other quantities use the manuscript's simulation units, with definitions below each table.

[Figure gallery](../README.md) · [Main results](../../../README.md) · [Reproduction instructions](../../../REPRODUCING.md) · [Source discrepancies](../../../PROJECT_SCOPE.md) · [Provenance](../provenance/manifest.json)

**Evidence labels:** native-record recomputation, archived derived results, and manuscript-reported rounded values describe different capabilities. Implementation labels (centered, legacy one-sided, or mixed) are stated separately. Missing native inputs are never counted as independently reproduced results.

| Item | Result |
|---|---|
| [Table S1](#table-s1) | Preparation and numerical settings |
| [Table S2](#table-s2) | Finite-window and early/late statistics |
| [Table S3](#table-s3) | Finite-grid boundary diagnostics |
| [Table S4](#table-s4) | Branch fractions and spin polarization |
| [Table S5](#table-s5) | Spatial and timestep refinement |
| [Table S6](#table-s6) | Bulk-spin and chiral-preparation controls |
| [Table S7](#table-s7) | Generic absorbing-layer controls |
| [Table S8](#table-s8) | Impedance-calibrated absorbing layers |
| [Table S9](#table-s9) | Fixed-confinement detector-parameter scan |
| [Table S10](#table-s10) | Longitudinal-momentum fit coefficients |
| [Table S11](#table-s11) | Fixed-source lower-wall displacement |
| [Table S12](#table-s12) | Reflecting versus absorbing lower boundary |
| [Long-time summary](#long-time-summary) | Fig. S13 support (not a numbered Supplement table) |

<a id="table-s1"></a>

## Table S1 — Preparation and numerical settings

The eight reference evolutions use the Cartesian spinor ABC and run through T=40. The T=10,20,30,40 statistics come from those same histories. Units are ℏ=m=1; times, lengths and momenta below use the manuscript's nondimensional units. The physical settings and numerical records are configuration evidence.

### Common longitudinal preparation

[Numerical CSV](settings/Table_S1_common_preparation.csv)

| Nominal L | Source zc | kz | κ | Amplitude σ∥ |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 10 | π | π | 0.5 |

### Common spin and solver settings

[Numerical CSV](settings/Table_S1_common_preparation.csv)

| θ | φ | T | Terminal weight | GMRES rtol | atol |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 40 | 0.5 | 1e-8 | 0 |

### Run-level timestep comparison

[Numerical CSV](settings/Table_S1_recorded_settings.csv)

| ω | Run | Δt: manuscript | Δt: recorded |
| ---: | ---: | ---: | ---: |
| 1 | 5540 | 0.00100 | 0.00100 |
| 25 | 5042 | 0.00080 | 0.00080 |
| 50 | 5047 | 0.00070 | 0.00080 |
| 75 | 5043 | 0.00050 | 0.00040 |
| 100 | 5541 | 0.00030 | 0.00050 |
| 150 | 5044 | 0.00025 | 0.00025 |
| 200 | 5045 | 0.00020 | 0.00025 |
| 300 | 7100 | 0.00015 | 0.00020 |

### Recorded grids and detector positions

[Numerical CSV](settings/Table_S1_recorded_settings.csv)

| ω | Run | Nx=Ny | Nz | Numerical zD |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5540 | 100 | 1500 | 19.98666573 |
| 25 | 5042 | 100 | 1500 | 19.98666573 |
| 50 | 5047 | 100 | 1500 | 19.98666573 |
| 75 | 5043 | 100 | 1500 | 19.98666573 |
| 100 | 5541 | 100 | 1500 | 19.98666573 |
| 150 | 5044 | 100 | 1500 | 19.98666573 |
| 200 | 5045 | 100 | 1500 | 19.98666573 |
| 300 | 7100 | 100 | 1500 | 19.98666573 |

**Definitions.** The initial spinor is (1,0)ᵀ, with θ=φ=0 and kz=κ=π. The transverse prescription is Lx=Ly=10/√ω with trap centers Lx/2,Ly/2. `dt_manuscript` is the separately [transcribed printed vector](settings/Table_S1_manuscript_timesteps.csv); `dt_recorded` is read from each bundled `run.json`. The CSV retains actual stencil spacings, launch norms, source-run links and precision. Coordinates/potential sampling use float32; stored states and matrix assembly use complex64; linear solves and extra spin/branch diagnostics use complex128/float64. GMRES uses Jacobi preconditioning, restart 30 and maxiter 1000. Event probabilities and masses are recorded every accepted step, plane records every 20 steps, and spin/branch records at nominal 0.02 intervals; reductions use actual saved times. These clocks are distinct from animation frame times.

**Interpretation.** Five timesteps in the printed vector differ from the retained run records (ω=50,75,100,200,300). Both are shown; the records determine the distributed calculations. The actual roof coordinate is below the nominal L=20.

**Implementation.** Centered ghost-point roof closure with a terminal half-weight norm; lower and lateral Dirichlet walls. The stored grid excludes nominal upper endpoints.

**Source and reproduction status.** Configuration/documentation, verified against bundled run records; no production simulation is claimed. [Summary preparation code](../../../Loaders/centered/supplement_tables_a.py) generates these compact configuration views.

[Back to table index](#supplementary-tables)


<a id="table-s2"></a>

## Table S2 — Finite-window and early/late statistics

T=40 with diagnostic split tcut=3.6. All eight spinor reference rows and both componentwise scalar controls are retained; the scalar rows are excluded from confinement fits.

### Spinor ABC: detection and means

[Numerical CSV](diagnostics/Table_S2_archived.csv)

| ω | D40 | Conditional mean | Restricted mean |
| ---: | ---: | ---: | ---: |
| 1 | 0.952 | 4.663 | 6.358 |
| 25 | 0.916 | 5.980 | 8.832 |
| 50 | 0.887 | 7.158 | 10.878 |
| 75 | 0.862 | 8.020 | 12.433 |
| 100 | 0.841 | 8.683 | 13.666 |
| 150 | 0.808 | 9.598 | 15.436 |
| 200 | 0.784 | 10.178 | 16.618 |
| 300 | 0.740 | 11.113 | 18.631 |

### Spinor ABC: early/late decomposition

[Numerical CSV](diagnostics/Table_S2_archived.csv)

| ω | tE | wL | tL | ΔtL,cond |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.595 | 0.394 | 7.839 | 2.067 |
| 25 | 2.561 | 0.405 | 10.999 | 3.419 |
| 50 | 2.543 | 0.447 | 12.864 | 4.615 |
| 75 | 2.533 | 0.479 | 13.982 | 5.488 |
| 100 | 2.526 | 0.504 | 14.735 | 6.157 |
| 150 | 2.517 | 0.538 | 15.677 | 7.081 |
| 200 | 2.511 | 0.559 | 16.234 | 7.666 |
| 300 | 2.501 | 0.590 | 17.088 | 8.613 |

### Scalar ABC: detection and means

[Numerical CSV](diagnostics/Table_S2_archived.csv)

| ω | D40 | Conditional mean | Restricted mean |
| ---: | ---: | ---: | ---: |
| 1 | 0.956 | 4.372 | 5.923 |
| 300 | 0.956 | 4.375 | 5.926 |

### Scalar ABC: early/late decomposition

[Numerical CSV](diagnostics/Table_S2_archived.csv)

| ω | tE | wL | tL | ΔtL,cond |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.598 | 0.386 | 7.198 | 1.774 |
| 300 | 2.600 | 0.386 | 7.195 | 1.775 |

**Definitions.** D40 is the unconditional detected fraction. The conditional mean averages clicks detected by 40; the restricted mean is ∫₀⁴⁰S(t)dt. tE and tL are click-time means conditioned on [0,3.6] and (3.6,40]. wL=DL/D40 is a fraction **among detections**, not an absolute probability. ΔtL,cond=wL(tL−tE) is the late contribution relative to the early conditional mean. The split is a diagnostic decomposition, not an encounter-resolved or strictly return-free interval. All means have time units; probabilities and weights are dimensionless.

**Interpretation.** Stronger confinement reduces prompt detection and increases the late conditional weight and mean. The restricted mean also includes survival at T=40. The scalar comparison remains almost unchanged across its two confinements under its own implementation.

**Implementation.** Mixed-source comparison: centered/half-weight spinor family and legacy one-sided scalar Robin closure with full terminal weight.

**Source and reproduction status.** Archived full-precision CSV; the eight spinor rows are independently recomputed from bundled accepted-step detector events and separately integrated survival by the [native diagnostics reducer](../../../Loaders/centered/diagnostics.py). Scalar rows are archived derived results only: normalized piecewise-linear **sampled survival**, checked in the source package against full-step histories, with launch norms and one-sided/full-weight provenance recorded [here](diagnostics/scalar_provenance.json). Those native scalar histories are not bundled.

[Back to table index](#supplementary-tables)


<a id="table-s3"></a>

## Table S3 — Finite-grid boundary diagnostics

T=40, all eight reference confinements. The endpoint budget uses native plane-record times; moments and continuation discrepancy use native spin/branch times.

### Probe scale, endpoint budget and momentum moment

[Numerical CSV](diagnostics/Table_S3_archived.csv)

| ω | hz√ω | qω | Δbud,rel | E[s²] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.013 | 1.75e-4 | 1.81e-5 | 0.982 |
| 25 | 0.067 | 8.74e-3 | 1.61e-4 | 1.966 |
| 50 | 0.094 | 2.44e-2 | 5.15e-4 | 2.745 |
| 75 | 0.115 | 4.44e-2 | 2.67e-4 | 3.333 |
| 100 | 0.133 | 6.77e-2 | 7.13e-4 | 3.808 |
| 150 | 0.163 | 1.21e-1 | 3.84e-4 | 4.522 |
| 200 | 0.189 | 1.78e-1 | 6.69e-4 | 5.009 |
| 300 | 0.231 | 2.95e-1 | 9.43e-4 | 5.524 |

### Covariance and inward continuation

[Numerical CSV](diagnostics/Table_S3_archived.csv)

| ω | βbl | Λω | E2br |
| ---: | ---: | ---: | ---: |
| 1 | -0.157960 | -0.157960 | 8.90e-4 |
| 25 | -0.254730 | -1.273649 | 3.47e-3 |
| 50 | -0.059895 | -0.423524 | 6.55e-3 |
| 75 | -0.015483 | -0.134087 | 9.78e-3 |
| 100 | 0.001241 | 0.012408 | 1.32e-2 |
| 150 | 0.008117 | 0.099409 | 2.07e-2 |
| 200 | 0.003756 | 0.053115 | 2.87e-2 |
| 300 | 0.001470 | 0.025455 | 4.56e-2 |

**Definitions.** Rh is the magnitude of the centered tangential-difference symbol and s=Rh/√ω. E[s²] is weighted by the time-integrated boundary density; qω=(hz√ω)²E[s²] is the dimensionless probe scale, not an error estimate. Δbud,rel=|∫κNDdt−[P0−P(T)]|/[P0−P(T)] uses the native **plane-record** density quadrature, not the accepted-step event sum. βbl=Cov(t,saω), Λω=Cov(t,Rhaω)=√ωβbl, with aω=−2W†ĴhW/(W†W), set to zero on neutral/zero-density modes. E2br is the boundary-density-weighted relative squared discrepancy between one-grid inward homogeneous continuation and the adjacent recorded plane (Eq. S102), integrating separate linear interpolants only where both endpoints are valid. The CSV keeps the unscaled covariance; βbl has time units and Λω time/length units. Other columns are dimensionless.

**Interpretation.** The covariance changes sign across the sampled family, while the scaled momentum moment increases. This auxiliary covariance is neither an empirical fit slope nor a physical absorbing-layer parameter, and it is not a universal predictor of click-time delay. Inward continuation is an auxiliary diagnostic; detailed values remain resolution-sensitive.

**Implementation.** Centered reference family with the half-weight terminal norm; the finite-grid tangential symbol uses centered differences.

**Source and reproduction status.** Archived full-precision results, recomputed from bundled native moment/diagnostic records by the [native diagnostics reducer](../../../Loaders/centered/diagnostics.py). Reconstructing the underlying Fourier fields or full Duhamel/monitor diagnostics would require unbundled fields. Plane-record accounting, spin/branch integrals, and accepted-step detector events remain separate.

[Back to table index](#supplementary-tables)


<a id="table-s4"></a>

## Table S4 — Branch fractions and spin polarization

T=40. Branch bars and detector polarization denote ratios of integrals over [0,40]; the final survivor polarization is evaluated at the endpoint. See [Fig. S1](../README.md#fig-s1) and [Table S5 refinement](#table-s5).

### Integrated branch fractions and laboratory-z ensembles

[Numerical CSV](diagnostics/Table_S4_archived.csv)

| ω | w+ | w− | w0 | Pz,det | Pz,survivor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.442 | 0.430 | 0.128 | 0.907 | 0.940 |
| 25 | 0.819 | 0.114 | 0.067 | 0.307 | 0.998 |
| 50 | 0.879 | 0.073 | 0.049 | 0.232 | 1.000 |
| 75 | 0.902 | 0.058 | 0.040 | 0.193 | 1.000 |
| 100 | 0.916 | 0.049 | 0.035 | 0.169 | 1.000 |
| 150 | 0.931 | 0.040 | 0.029 | 0.143 | 1.000 |
| 200 | 0.938 | 0.036 | 0.026 | 0.128 | 1.000 |
| 300 | 0.946 | 0.032 | 0.022 | 0.097 | 1.000 |

**Definitions.** w+, w− and w0 are ratios of integrated native boundary-density records, using linear interpolation on their actual times. Pz,det=(Pup−Pdown)/(Pup+Pdown) uses the integrated accepted-step event matrix. Pz,survivor is the z-polarization of the **entire final surviving state** from its diagnostic-time spin matrix. All columns are dimensionless. Independently rounded branch fractions need not sum exactly to one.

**Interpretation.** At strong confinement the plus branch dominates the evolved boundary trace, but the detector-associated laboratory-z polarization is small and the final whole-survivor polarization is near one. These are different weightings and spin axes. Unequal evolved branch weights do not imply unequal local branch absorption coefficients at fixed incident wave vector.

**Implementation.** Centered reference family. Neutral tangential-symbol modes remain in the total branch weight.

**Source and reproduction status.** Archived full-precision results independently recomputed by the [native diagnostics reducer](../../../Loaders/centered/diagnostics.py). **Printed-value mismatch:** at ω=50 the manuscript shows Pz,survivor=0.999; the retained value 0.99971289084281545 rounds to 1.000 at three decimals. This presentation follows the numerical CSV and retains the mismatch in the [printed-value reference](refinement/manuscript_reported_rounding_S2_S6.csv); it is not silently forced to the paper.

[Back to table index](#supplementary-tables)


<a id="table-s5"></a>

## Table S5 — Spatial and timestep refinement

ω=300, T=40, reference preparation and fixed numerical boundary positions. The four runs are baseline 7100, transverse 7102, longitudinal **7101 with Nz=2999**, and half timestep 7706. The later Nz=4000 result is excluded.

### Common run settings for panels (a) and (b)

[Numerical CSV](refinement/Table_S5_recomputed.csv)

| Run | Refinement | Nx=Ny | Nz | Δt |
| ---: | --- | ---: | ---: | ---: |
| 7100 | Baseline | 100 | 1500 | 2e-4 |
| 7102 | Transverse | 120 | 1500 | 2e-4 |
| 7101 | Longitudinal | 100 | 2999 | 2e-4 |
| 7706 | Half timestep | 100 | 1500 | 1e-4 |

### (a) Detection statistics and integrated plus branch

[Numerical CSV](refinement/Table_S5_recomputed.csv)

| Run | D40 | μ*(40) | εrel (%) | w+ |
| ---: | ---: | ---: | ---: | ---: |
| 7100 | 0.740 | 18.631 | 0.000 | 0.946 |
| 7102 | 0.743 | 18.510 | 0.654 | 0.946 |
| 7101 | 0.733 | 18.977 | 1.824 | 0.952 |
| 7706 | 0.740 | 18.626 | 0.031 | 0.946 |

### (a) Full-window rate discrepancies: [0,40]

[Numerical CSV](refinement/Table_S5_recomputed.csv)

| Run | εCN (%) | εbr (%) |
| ---: | ---: | ---: |
| 7100 | 0.000 | 0.000 |
| 7102 | 0.742 | 0.752 |
| 7101 | 6.188 | 6.320 |
| 7706 | 2.647 | 2.647 |

### (b) Spin and boundary-weighted diagnostics

[Numerical CSV](refinement/Table_S5_recomputed.csv)

| Run | Pz,det | Pz,survivor | E[s²] | 10³βbl |
| ---: | ---: | ---: | ---: | ---: |
| 7100 | 0.097 | 0.999975 | 5.524 | 1.470 |
| 7102 | 0.100 | 0.999974 | 5.607 | 0.987 |
| 7101 | 0.104 | 0.999959 | 6.167 | 14.856 |
| 7706 | 0.097 | 0.999963 | 5.524 | 1.661 |

### (b) Late-window rate discrepancies: [8,40]

[Numerical CSV](refinement/Table_S5_recomputed.csv)

| Run | εCN,late (%) | εbr,late (%) |
| ---: | ---: | ---: |
| 7100 | 0.000 | 0.000 |
| 7102 | 0.588 | 0.610 |
| 7101 | 9.580 | 9.857 |
| 7706 | 5.332 | 5.314 |

**Definitions.** All ε columns are numeric percentages (for example, 9.580 means 9.580%, not a fraction). εrel=100|μr−μb|/|μr| uses the **refined** restricted mean in its denominator. εCN(W)=100∫W|gCN,r−gCN,b|dt/∫WgCN,bdt uses native event probabilities divided by accepted-step durations and launch norms. εbr sums absolute differences of the +,−,neutral branch rates and divides by the integral of the baseline sampled roof-density rate Jκ,h over the same window. This denominator is distinct from the baseline accepted-step CN probability used for εCN. CN rates are piecewise constant; branch records are piecewise linear. Integrations use the union of time grids and split linear zero crossings, with no smoothing or time shift. The CSV retains raw βbl as well as the displayed 10³βbl scale; full and late windows are explicit columns.

**Interpretation.** Integrated plus-branch predominance and the detector/survivor spin contrast persist in these tests. Small restricted-mean differences do not establish full convergence: detailed late rates change substantially under longitudinal and timestep refinement, and auxiliary diagnostics are more sensitive.

**Implementation.** Centered/half-weight family, with longitudinal refinement retaining the original numerical roof and transverse walls.

**Source and reproduction status.** Recomputed from bundled native records using the first-pass [native diagnostics reducer](../../../Loaders/centered/diagnostics.py)'s Eq. S117 reconstruction; the original final four-run exporter was not located. [Summary preparation](../../../Loaders/centered/supplement_tables_a.py) joins its outputs without changing formulas. The earlier [three-run archive](diagnostics/Table_S5_three_run_archived.csv) and [pairwise archive](diagnostics/Table_S5_pairwise_archived.csv) remain as independent checks. **Printed discrepancies:** manuscript D40 is 0.739/0.742 for baseline/transverse, while native values round to 0.740/0.743; longitudinal εrel is printed 1.823%, while 1.823508241% rounds to 1.824%. The [printed-value reference](refinement/manuscript_reported_rounding_S2_S6.csv) keeps all inspected printed values and their display precision; no computed discrepancy is hardcoded.

[Back to table index](#supplementary-tables)


<a id="table-s6"></a>

## Table S6 — Bulk-spin and chiral-preparation controls

T=20, six controls at each of ω=1 and 100. The physical reference preparation is L=20, zc=10, kz=κ=π, amplitude width 0.5 and initial spinor (1,0)ᵀ; the chiral case changes the transverse preparation.

### All twelve control rows

[Numerical CSV](controls/T6/full_precision_results.csv)

| Control | ω | Run | D20 | Conditional mean | μ*(20) | DL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 1 | 5540 | 0.925 | 3.947 | 5.151 | 0.348 |
| Reference | 100 | 5541 | 0.724 | 5.548 | 9.534 | 0.307 |
| Uniform | 1 | 7692 | 0.928 | 3.938 | 5.091 | 0.350 |
| Uniform | 100 | 7693 | 0.724 | 5.549 | 9.536 | 0.307 |
| Radial | 1 | 7694 | 0.923 | 3.935 | 5.165 | 0.346 |
| Radial | 100 | 7695 | 0.723 | 5.556 | 9.554 | 0.307 |
| Rashba | 1 | 7696 | 0.920 | 4.111 | 5.375 | 0.356 |
| Rashba | 100 | 7697 | 0.707 | 5.690 | 9.883 | 0.306 |
| Rashba–D. | 1 | 7698 | 0.917 | 4.147 | 5.460 | 0.355 |
| Rashba–D. | 100 | 7699 | 0.710 | 5.663 | 9.818 | 0.307 |
| Chiral | 1 | 7001 | 0.914 | 4.060 | 5.435 | 0.348 |
| Chiral | 100 | 7000 | 0.598 | 6.434 | 11.890 | 0.287 |

**Definitions.** D20=1−S(20), μ*(20)=∫₀²⁰S(t)dt, and DL=S(3.6)−S(20) is an **absolute** late probability. The conditional mean follows from [μ*(20)−20S(20)]/D20. Means have time units; probabilities are dimensionless. The native-event cross-check columns in the CSV are separate quantities, not substituted for the survival reduction. Control key: Reference Ha=0; Uniform Ha=πσx+(1/2)σz; Radial Ha=0.2√ω(Xσx+Yσy), with X,Y relative to the trap center; Rashba Ha=0.5(σxpy−σypx); Rashba–D. adds 0.4(σxpy−σypx)+0.3(σxpx−σypy); Chiral has Ha=0 and transverse profile (χ10+iχ01)/√2.

**Interpretation.** The tested confinement contrast persists for these bulk additions and this excited preparation. Its magnitude depends on the dynamics and initial state; two confinement values do not establish a universal law or invariance under arbitrary spin couplings or preparations.

**Implementation.** Centered spinor-ABC controls. Statistics follow the original survival-history convention, using every-step stored mass and recorded launch norm.

**Source and reproduction status.** Archived derived numerical results, displayed from the existing full-precision CSV. The [historical reducer](../../../Solvers/centered/controls/analysis/T6/recalculate_s8_1.py) documents the every-step survival recipe, but its full control histories are not bundled; coarse sampled norms are not substituted. Historical source comments refer to an earlier manuscript snapshot; this presentation's printed-value comparison uses the supplied 28 October 2026 PDF (page 41).

[Back to table index](#supplementary-tables)


<a id="table-s7"></a>

## Table S7 — Generic absorbing-layer controls

At ω=100 and T=20, the source is z_c=10 with k_z=κ=π, Gaussian amplitude width σ∥=0.5, and initial spinor (1,0)ᵀ. Layer runs extend the guide to the nominal Dirichlet backing wall z=22 (N_z=1650, h_z=20/1500), remove the spinor roof ABC, and use manuscript-stated Δt=0.0004. The spinor-ABC reference is the Table S6 reference. See [Fig. S2](../README.md#fig-s2) and the calibrated comparison in [Table S8](#table-s8).

### All reported models

[Numerical CSV](manuscript_reported/Table_S7_manuscript_rounded.csv)

| Model | D₂₀ | Conditional mean | Restricted mean μ*(20) | D_L |
| --- | ---: | ---: | ---: | ---: |
| Spinor ABC reference | 0.724 | 5.548 | 9.534 | 0.307 |
| Scalar CAP | 0.857 | 4.240 | 6.487 | 0.304 |
| Uniform σₓ + CAP | 0.857 | 4.240 | 6.487 | 0.304 |
| Local χσₓ + CAP | 0.857 | 4.241 | 6.487 | 0.304 |
| Spin-selective y loss | 0.862 | 4.198 | 6.374 | 0.306 |
| Layer Rashba + CAP | 0.857 | 4.241 | 6.487 | 0.304 |
| Projected loss | 0.856 | 4.251 | 6.514 | 0.304 |

**Definitions.** All quantities are dimensionless. D₂₀=1−S(20); the conditional mean is conditioned on detection by 20; μ*(20)=∫₀²⁰S(t)dt; D_L=S(3.6)−S(20) is an absolute probability, not a fraction among detections. Layer statistics use normalized piecewise-linear survival. When the launch sample is absent, the manuscript normalizes by the first recorded norm and sets S=1 before that sample. The model key is: scalar CAP $-i40\chi I$; uniform $\sigma_x-i40\chi I$; local $\chi\sigma_x-i40\chi I$; spin-selective $-i\chi(40I+20\sigma_y)$; layer Rashba $0.5\chi K-i40\chi I$; projected loss $-i\chi(40I+0.5RQ_+)$. Here $\chi(z)=[1+\tanh((z-20)/0.165)]/2$, $K=\sigma_xp_y-\sigma_yp_x$, $R=\sqrt{K^2}$, and $Q_s=(I+sK/R)/2$ for R>0. The tangential symbol of Q_s is Π₋ₛ; its sign convention differs from the boundary-branch label.

**Interpretation.** At these strengths, the tested spin-dependent additions alter the scalar-layer statistics only weakly. Their restricted means remain below the spinor-ABC reference. This finite set does not establish equivalence of generic layers and the boundary detector, or exclude every possible layer model.

**Implementation.** Centered spinor-ABC reference; physical absorbing-layer controls without a spinor roof ABC. The actual layer implementation files are unavailable.

**Source and reproduction status.** Manuscript-reported rounded values, transcribed and visually checked against PDF page 42 (Supplement page 29). No verified numerical table export, layer solver, or native history was located in the mapped sources. The renderer reproduces this transcription only; it is not an independent numerical reconstruction. Entries were rounded independently, so displayed digits do not specify numerical accuracy.

[Back to table index](#supplementary-tables)


<a id="table-s8"></a>

## Table S8 — Impedance-calibrated absorbing layers

T=20 with the same reference preparation and nominal layer thickness ℓ=2 as Table S7. At ω=1 the manuscript states Δt=0.0008 for broadband cases and 0.001 for the remaining cases; at ω=100 it states 0.0004 for all cases. The bundled ω=100 reference 5541 instead records 0.0005; see [Table S1](#table-s1). The two panels share one canonical CSV and retain every reported model. [Fig. S2](../README.md#fig-s2) supplies the ω=100 profile comparison.

### ω=1

[Numerical CSV](manuscript_reported/Table_S8_manuscript_rounded.csv)

| Model | ω | D₂₀ | Conditional mean | Restricted mean μ*(20) | D_L |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spinor ABC reference | 1 | 0.925 | 3.947 | 5.151 | 0.348 |
| Central-exact | 1 | 0.805 | 4.817 | 7.773 | 0.379 |
| Broadband (w_c=2) | 1 | 0.911 | 4.064 | 5.480 | 0.382 |
| Broadband (w_c=0.5) | 1 | 0.905 | 4.003 | 5.522 | 0.363 |

### ω=100

[Numerical CSV](manuscript_reported/Table_S8_manuscript_rounded.csv)

| Model | ω | D₂₀ | Conditional mean | Restricted mean μ*(20) | D_L |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spinor ABC reference | 100 | 0.724 | 5.548 | 9.534 | 0.307 |
| Scalar CAP | 100 | 0.857 | 4.240 | 6.487 | 0.304 |
| Central-exact | 100 | 0.530 | 6.645 | 12.927 | 0.254 |
| Broadband (w_c=2) | 100 | 0.636 | 5.904 | 11.031 | 0.272 |

**Definitions.** D₂₀, the conditional mean, μ*(20), and absolute D_L use the survival convention in Table S7. For the uniform inverse layer, $H_{\rm layer}=\mathbf{1}_{[20,22)}\sum_s V_s(R)Q_s$ with $-\operatorname{Im}V_s\geq0$. Its entrance response is $\Lambda_s(E,R)=-q_s\cot(q_s\ell)$, $q_s^2=2(E-V_s)$; the target is $Z_s=i\kappa-sR$. Central-exact calibration matches at E_cal=k_z²/2. Weighted-broadband calibration additionally minimizes the passive complex-impedance mismatch over the positive-k Gaussian band, using flux weights proportional to k|φ̃₀(k)|² and central-energy penalty w_c. Calibration stops at R_cal=4√ω and holds the cutoff values above it. These are stationary matching relations, not propagated-wavepacket identities.

**Interpretation.** Broadband calibration improves the reported restricted-mean agreement relative to central-exact calibration at both tested confinements. At ω=100, the reported relative discrepancy decreases from approximately 35.6% to 15.7%, but the detected fraction and conditional mean still differ. Matching the auxiliary entrance response does not establish equality of the complete click-time distributions; entrance current also includes the time derivative of probability stored in the layer.

**Implementation.** Centered spinor-ABC references plus finite absorbing layers with a Dirichlet backing wall; the calibrated functions of K use the manuscript’s centered transverse Dirichlet operators. Layer inputs remain unavailable.

**Source and reproduction status.** Manuscript-reported rounded values, visually checked against PDF page 43 (Supplement page 30). No calibrated potentials, native curves, or verified numerical table export were located. This presentation preserves unfavorable as well as improved comparisons and is not a fresh computation.

[Back to table index](#supplementary-tables)


<a id="table-s9"></a>

## Table S9 — Fixed-confinement detector-parameter scan

Historical fixed-confinement scan at ω=1, k_z=π, amplitude width σ∥=0.5, T=20. [Fig. S3](../README.md#fig-s3) shows the associated legacy profile family. The CSV below transcribes the manuscript table rather than filling gaps from a centered campaign.

### Manuscript-reported restricted means

[Numerical CSV](manuscript_reported/Table_S9_manuscript_rounded.csv)

| κ | Restricted mean μ*(20) |
| ---: | ---: |
| 0.1 | 17.016 |
| 0.5 | 11.709 |
| 1 | 8.247 |
| 2 | 5.653 |
| π | 5.151 |
| 5 | 5.729 |
| 7 | 6.717 |
| 9 | 7.687 |
| 13 | 9.343 |

**Definitions.** κ is the detector parameter; μ*(20)=∫₀²⁰S(t)dt is dimensionless restricted time. The numerical κ field represents π by its double-precision value. [Located source audit](manuscript_reported/Table_S9_located_source_audit.csv) records eight available audit entries, including saved-window and trapezoidal survival areas, θ, Δt, and numerical detector position; those areas are archived audit quantities and are not asserted to be the manuscript’s exact reduction.

**Interpretation.** The smallest manuscript entry among the tabulated settings occurs at κ=π, the scalar matching point for the central longitudinal component. This is a finite-preparation comparison, not a globally optimal detector setting. The located π audit gives 5.153081498198618 (5.153 at three decimals), whereas the manuscript gives 5.151; this exceeds the displayed rounding and remains unresolved.

**Implementation.** Legacy one-sided normal-boundary family. The located source audit includes preparation and detector-position qualifications; it is not a uniform centered reference family.

**Source and reproduction status.** Manuscript-reported rounded values, visually checked against PDF page 45 (Supplement page 32). A complete verified Table S9 export was not located: the mapped audit lacks κ=5. The κ=1 record has θ=π/3 and Δt=0.0008 while the other audited records have θ=0 and Δt=0.001; its logged output-directory label says κ=2 despite recorded κ=1. The κ=π detector coordinate is approximately 19.986666 rather than the other audited records’ 20. No centered row or different preparation is substituted.

[Back to table index](#supplementary-tables)


<a id="table-s10"></a>

## Table S10 — Longitudinal-momentum fit coefficients

Centered spinor ABC at T=20, κ=π, amplitude width σ∥=0.5. Each unweighted fit uses exactly ω={1,25,50,100,150,200}, with μ*(20;ω)=A₂₀+B₂₀√ω. See [Fig. S6](../README.md#fig-s6), [fit-input metrics](controls/T10/table_S10_raw_metrics_T20.csv), and [Fig. S6 numerical data](controls/S6/fig_S6_centered_T20.csv).

### Three six-point fits

[Numerical CSV](controls/T10/table_S10_centered_T20.csv)

| k_z | A₂₀ | B₂₀ | r_rms | n |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8.394 | 0.451 | 0.212 | 6 |
| π | 4.557 | 0.475 | 0.179 | 6 |
| 5 | 2.093 | 0.291 | 0.335 | 6 |

**Definitions.** A₂₀ and r_rms have the dimensionless time units of μ*; B₂₀ is the coefficient of √ω. r_rms is the root mean square of the six fit residuals, not a numerical-discretization error. The canonical CSV additionally preserves R² and maximum absolute residual at available precision.

**Interpretation.** The fitted coefficients and residuals vary with longitudinal preparation even with the same detector and horizon. These empirical fits describe the stated six-point sample; they do not establish a universal confinement law.

**Implementation.** Centered ghost-point roof closure with its compatible terminal half-weight norm.

**Source and reproduction status.** Archived derived numerical results, rendered from the existing full-precision CSV. The rounded values agree with PDF page 47 (Supplement page 34). The [historical reducer](../../../Solvers/centered/controls/analysis/T10/recalculate_table_S10.py) refers to an unavailable provenance helper. The [portable controls workflow](../../../Loaders/centered/controls.py) displays the archived results; it does not independently reconstruct every sweep run.

[Back to table index](#supplementary-tables)


<a id="table-s11"></a>

## Table S11 — Fixed-source lower-wall displacement

ω=200, T=40, k_z=κ=π, amplitude width σ∥=0.5, initial spinor (1,0)ᵀ. Source z_c=10 and numerical upper detector z_D≈19.986666 remain fixed while z_B changes. Runs 7750–7753 correspond, in order, to z_B=0,−5,−10,−15. They retain the original nonnegative-z nodes, h_z≈0.013333 and N_x=N_y=100, with Δt=0.0003. [Fig. S11](../README.md#fig-s11) shows the signed longitudinal-current maps.

### Grid and full-window statistics

[Numerical CSV](controls/S11/table_S11_fixed_source_T40.csv)

| z_B | N_z | D₄₀ | Conditional mean | Restricted mean μ*(40) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1500 | 0.784 | 10.179 | 16.620 |
| -5 | 1875 | 0.750 | 10.431 | 17.833 |
| -10 | 2250 | 0.716 | 10.382 | 18.795 |
| -15 | 2625 | 0.687 | 10.263 | 19.571 |

### Late window (3.6,40]

[Numerical CSV](controls/S11/table_S11_fixed_source_T40.csv)

| z_B | D_L,40 | Late conditional mean |
| ---: | ---: | ---: |
| 0 | 0.438 | 16.234 |
| -5 | 0.404 | 17.216 |
| -10 | 0.370 | 17.740 |
| -15 | 0.341 | 18.125 |

**Definitions.** D₄₀=1−S(40), μ*(40)=∫₀⁴⁰S(t)dt, and the conditional mean uses the finite-window identity. D_L,40=S(3.6)−S(40) is the absolute late probability; its mean is conditioned only on detection in (3.6,40]. The common diagnostic split does not identify individual returns. Lengths and times are dimensionless physical coordinates, not oscillator-scaled axes.

**Interpretation.** The reported early probability is approximately 0.346 at all four positions. Moving the wall downward reduces the full-window and late detected probabilities while increasing the late conditional mean and restricted mean. The overall detected conditional mean is nonmonotonic across the four positions, so the result is not a uniform delay of all detections.

**Implementation.** Centered spinor-ABC roof, reflecting Dirichlet lower wall; fixed-source geometry family, distinct from a guide-length scan with a moving source.

**Source and reproduction status.** Archived derived numerical results, using the existing full-precision CSV and its survival-history convention. Displayed values agree with PDF page 50 (Supplement page 37). [Original analysis recipe](../../../Solvers/centered/controls/analysis/S11/analyze_fixed_source_T40.py) and [plot metadata](controls/S11/plot_metadata.json) are included; the four complete native histories and spatial current histories are not bundled, so this is archived-table presentation rather than native reconstruction.

[Back to table index](#supplementary-tables)


<a id="table-s12"></a>

## Table S12 — Reflecting versus absorbing lower boundary

T=40, ω=1 and 200, with the reference source and spin-coupled upper detector. The scalar lower ABC has κ_lower=π and is partially absorbing. Cases 7801 (ω=1) and 7800 (ω=200) supply its native two-channel records. See [Fig. S12](../README.md#fig-s12) and the [native endpoint metrics](controls/T12/endpoint_comparison.csv).

### Separate upper detection, lower escape and surviving mass

[Numerical CSV](controls/T12/four_way_statistics.csv)

| ω | Lower boundary | D_top | E_lower | S_in |
| ---: | --- | ---: | ---: | ---: |
| 1 | Dirichlet (rounded SI) | 0.952 | 0.000 | 0.048 |
| 1 | Scalar ABC (native) | 0.905 | 0.062 | 0.033 |
| 200 | Dirichlet (rounded SI) | 0.784 | 0.000 | 0.216 |
| 200 | Scalar ABC (native) | 0.476 | 0.472 | 0.052 |

### Upper-click times and late probability

[Numerical CSV](controls/T12/four_way_statistics.csv)

| ω | Lower boundary | Upper conditional mean | D_L | Late conditional mean |
| ---: | --- | ---: | ---: | ---: |
| 1 | Dirichlet (rounded SI) | 4.663 | 0.375 | 7.839 |
| 1 | Scalar ABC (native) | 3.865 | 0.329 | 6.092 |
| 200 | Dirichlet (rounded SI) | 10.178 | 0.438 | 16.234 |
| 200 | Scalar ABC (native) | 3.525 | 0.130 | 6.231 |

**Definitions.** D_top is cumulative upper-detector probability; E_lower is cumulative lower escape; S_in is mass still in the guide. Their sum is approximately one with the measured accounting defect retained. Probability of no upper click is S_in+E_lower, not S_in alone. D_L=D_top(40)−D_top(3.6) and the late mean refer only to upper events in (3.6,40]. All probabilities are absolute fractions of launch norm; conditional times are dimensionless.

**Interpretation.** The absorbing lower termination substantially reduces late upper clicks and their conditional mean, while the early upper response agrees at displayed precision. The termination is not transparent to the full wavepacket. Total norm loss includes lower escape and cannot be substituted for upper-detection probability; accounting closure is not a convergence test.

**Implementation.** Centered spinor upper ABC in all rows; reflecting Dirichlet or scalar absorbing lower closure. The different evidence conventions within the table are retained.

**Source and reproduction status.** Mixed evidence: the two scalar-lower-ABC rows are archived numerical results independently recomputed from bundled native accepted-step records by the [controls adapter](../../../Loaders/centered/controls.py). The two Dirichlet rows retain the rounded archived rounded-reference convention: S_in=1−D₄₀ and D_L=D₄₀w_L with w_L retained to four decimals in this CSV (0.3942 and 0.5587), beyond the three decimals printed in Table S2. They are not replaced by more precise native Dirichlet values. Their existing CSV arithmetic precision does not create new source precision. Values agree with PDF page 53 (Supplement page 40) after the declared display rounding.

[Back to table index](#supplementary-tables)


<a id="long-time-summary"></a>

## Long-time summary — Fig. S13 support (not a numbered Supplement table)

Case 7810: ω=200, centered spinor roof ABC and reflecting lower wall, k_z=κ=π, amplitude width σ∥=0.5, initial spinor (1,0)ᵀ; requested and saved support through T=200. [Fig. S13](../README.md#fig-s13) shows the profile. The manuscript estimate and current available support are intentionally separate.

### Historical manuscript estimates from logged survival

[Numerical CSV](manuscript_reported/FigS13_logged_estimates_manuscript_rounded.csv)

| Horizon | Reported detected fraction |
| ---: | ---: |
| 40 | 0.784 |
| 189.95 | 0.941 |

### Current bundled derived bins and sampled norms

[Numerical CSV](controls/7810/long_time_support.csv)

| Horizon | Binned-event D | Sampled S | D+S−1 |
| ---: | ---: | ---: | ---: |
| 40 | 0.784 | 0.216 | -3.202e-7 |
| 80 | 0.880 | 0.120 | -3.331e-7 |
| 100 | 0.901 | 0.099 | -3.390e-7 |
| 200 | 0.943 | 0.057 | -3.448e-7 |

**Definitions.** The historical D values are rounded estimates from the logged survival history at the manuscript endpoints. Current D integrates the included 0.01-wide accepted-event-derived bins; current S interpolates saved norms on their actual 0.025 cadence and divides by recorded launch norm. These are distinct clocks from the solver step Δt=0.00025. D+S−1 retains their measured accounting difference; it is not forced to zero.

**Interpretation.** The manuscript estimates approximately 15.7% of initial probability detected after 40 through 189.95, about 73% of the probability undetected at 40. The separately supported result at 200 is not a replacement for that historical estimate. Finite-window survival is not permanent nondetection; this single-confinement extension determines neither the unrestricted mean nor a long-horizon confinement law.

**Implementation.** Centered ghost-point roof with compatible terminal half-weight norm; one confinement only.

**Source and reproduction status.** The first panel is manuscript-reported rounded values, visually checked on PDF page 53. The second is recomputed from bundled derived bins and native sampled norm records, not the omitted complete accepted-step history. [Original plot data](../native/controls/7810/plot_data.csv), [metadata](../native/controls/7810/plot_metadata.json), and the [controls reducer](../../../Loaders/centered/controls.py) define the supported calculation. Run `python -m Loaders.centered.supplement_tables_b --check` to check the small current-support CSV without reading the research root.

[Back to table index](#supplementary-tables)

<!-- Generated by Loaders/render_supplement_tables.py. Edit the CSVs/specifications, then regenerate. -->

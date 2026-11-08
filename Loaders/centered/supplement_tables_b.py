"""Presentation specifications for Tables S7-S12 and Fig. S13 support.

Printed tables are explicit rounded transcriptions; this module never treats
those entries as independently reconstructed solver results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
from pathlib import Path

SECTIONS = [{'id': 'table-s7',
  'title': 'Generic absorbing-layer controls',
  'intro': 'At ω=100 and T=20, the source is z_c=10 with k_z=κ=π, Gaussian amplitude width σ∥=0.5, and '
           'initial spinor (1,0)ᵀ. Layer runs extend the guide to the nominal Dirichlet backing wall z=22 '
           '(N_z=1650, h_z=20/1500), remove the spinor roof ABC, and use manuscript-stated Δt=0.0004. The '
           'spinor-ABC reference is the Table S6 reference. See [Fig. S2](../README.md#fig-s2) and the '
           'calibrated comparison in [Table S8](#table-s8).',
  'implementation': 'Centered spinor-ABC reference; physical absorbing-layer controls without a spinor roof '
                    'ABC. The actual layer implementation files are unavailable.',
  'panels': [{'label': 'All reported models',
              'csv': 'data/centered/tables/manuscript_reported/Table_S7_manuscript_rounded.csv',
              'columns': [{'key': 'model',
                           'label': 'Model',
                           'format': 'text',
                           'values': {'spinor_abc': 'Spinor ABC reference',
                                      'scalar_cap': 'Scalar CAP',
                                      'uniform_x_cap': 'Uniform σₓ + CAP',
                                      'local_x_cap': 'Local χσₓ + CAP',
                                      'spin_y_loss': 'Spin-selective y loss',
                                      'layer_rashba_cap': 'Layer Rashba + CAP',
                                      'projected_loss': 'Projected loss'}},
                          {'key': 'D20', 'label': 'D₂₀', 'format': '.3f'},
                          {'key': 'tau_cond20', 'label': 'Conditional mean', 'format': '.3f'},
                          {'key': 'mu_star20', 'label': 'Restricted mean μ*(20)', 'format': '.3f'},
                          {'key': 'D_late', 'label': 'D_L', 'format': '.3f'}],
              'expected_rows': 7}],
  'dictionary': 'All quantities are dimensionless. D₂₀=1−S(20); the conditional mean is conditioned on '
                'detection by 20; μ*(20)=∫₀²⁰S(t)dt; D_L=S(3.6)−S(20) is an absolute probability, not a '
                'fraction among detections. Layer statistics use normalized piecewise-linear survival. When '
                'the launch sample is absent, the manuscript normalizes by the first recorded norm and sets '
                'S=1 before that sample. The model key is: scalar CAP $-i40\\chi I$; uniform '
                '$\\sigma_x-i40\\chi I$; local $\\chi\\sigma_x-i40\\chi I$; spin-selective '
                '$-i\\chi(40I+20\\sigma_y)$; layer Rashba $0.5\\chi K-i40\\chi I$; projected loss '
                '$-i\\chi(40I+0.5RQ_+)$. Here $\\chi(z)=[1+\\tanh((z-20)/0.165)]/2$, '
                '$K=\\sigma_xp_y-\\sigma_yp_x$, $R=\\sqrt{K^2}$, and $Q_s=(I+sK/R)/2$ for R>0. The '
                'tangential symbol of Q_s is Π₋ₛ; its sign convention differs from the boundary-branch '
                'label.',
  'interpretation': 'At these strengths, the tested spin-dependent additions alter the scalar-layer '
                    'statistics only weakly. Their restricted means remain below the spinor-ABC reference. '
                    'This finite set does not establish equivalence of generic layers and the boundary '
                    'detector, or exclude every possible layer model.',
  'status': 'Manuscript-reported rounded values, transcribed and visually checked against PDF page 42 '
            '(Supplement page 29). No verified numerical table export, layer solver, or native history was '
            'located in the mapped sources. The renderer reproduces this transcription only; it is not an '
            'independent numerical reconstruction. Entries were rounded independently, so displayed digits '
            'do not specify numerical accuracy.'},
 {'id': 'table-s8',
  'title': 'Impedance-calibrated absorbing layers',
  'intro': 'T=20 with the same reference preparation and nominal layer thickness ℓ=2 as Table S7. At ω=1 the '
           'manuscript states Δt=0.0008 for broadband cases and 0.001 for the remaining cases; at ω=100 it '
           'states 0.0004 for all cases. The bundled ω=100 reference 5541 instead records 0.0005; see [Table S1](#table-s1). The two panels share one canonical CSV and retain every reported '
           'model. [Fig. S2](../README.md#fig-s2) supplies the ω=100 profile comparison.',
  'implementation': 'Centered spinor-ABC references plus finite absorbing layers with a Dirichlet backing '
                    'wall; the calibrated functions of K use the manuscript’s centered transverse Dirichlet '
                    'operators. Layer inputs remain unavailable.',
  'panels': [{'label': 'ω=1',
              'csv': 'data/centered/tables/manuscript_reported/Table_S8_manuscript_rounded.csv',
              'columns': [{'key': 'model',
                           'label': 'Model',
                           'format': 'text',
                           'values': {'spinor_abc': 'Spinor ABC reference',
                                      'scalar_cap': 'Scalar CAP',
                                      'central_exact': 'Central-exact',
                                      'broadband_wc2': 'Broadband (w_c=2)',
                                      'broadband_wc0p5': 'Broadband (w_c=0.5)'}},
                          {'key': 'omega', 'label': 'ω', 'format': 'd'},
                          {'key': 'D20', 'label': 'D₂₀', 'format': '.3f'},
                          {'key': 'tau_cond20', 'label': 'Conditional mean', 'format': '.3f'},
                          {'key': 'mu_star20', 'label': 'Restricted mean μ*(20)', 'format': '.3f'},
                          {'key': 'D_late', 'label': 'D_L', 'format': '.3f'}],
              'expected_rows': 4,
              'row_slice': [0, 4],
              'expected_total_rows': 8},
             {'label': 'ω=100',
              'csv': 'data/centered/tables/manuscript_reported/Table_S8_manuscript_rounded.csv',
              'columns': [{'key': 'model',
                           'label': 'Model',
                           'format': 'text',
                           'values': {'spinor_abc': 'Spinor ABC reference',
                                      'scalar_cap': 'Scalar CAP',
                                      'central_exact': 'Central-exact',
                                      'broadband_wc2': 'Broadband (w_c=2)',
                                      'broadband_wc0p5': 'Broadband (w_c=0.5)'}},
                          {'key': 'omega', 'label': 'ω', 'format': 'd'},
                          {'key': 'D20', 'label': 'D₂₀', 'format': '.3f'},
                          {'key': 'tau_cond20', 'label': 'Conditional mean', 'format': '.3f'},
                          {'key': 'mu_star20', 'label': 'Restricted mean μ*(20)', 'format': '.3f'},
                          {'key': 'D_late', 'label': 'D_L', 'format': '.3f'}],
              'expected_rows': 4,
              'row_slice': [4, 8],
              'expected_total_rows': 8}],
  'dictionary': 'D₂₀, the conditional mean, μ*(20), and absolute D_L use the survival convention in Table '
                'S7. For the uniform inverse layer, $H_{\\rm layer}=\\mathbf{1}_{[20,22)}\\sum_s V_s(R)Q_s$ '
                'with $-\\operatorname{Im}V_s\\geq0$. Its entrance response is '
                '$\\Lambda_s(E,R)=-q_s\\cot(q_s\\ell)$, $q_s^2=2(E-V_s)$; the target is $Z_s=i\\kappa-sR$. '
                'Central-exact calibration matches at E_cal=k_z²/2. Weighted-broadband calibration '
                'additionally minimizes the passive complex-impedance mismatch over the positive-k Gaussian '
                'band, using flux weights proportional to k|φ̃₀(k)|² and central-energy penalty w_c. '
                'Calibration stops at R_cal=4√ω and holds the cutoff values above it. These are stationary '
                'matching relations, not propagated-wavepacket identities.',
  'interpretation': 'Broadband calibration improves the reported restricted-mean agreement relative to '
                    'central-exact calibration at both tested confinements. At ω=100, the reported relative '
                    'discrepancy decreases from approximately 35.6% to 15.7%, but the detected fraction and '
                    'conditional mean still differ. Matching the auxiliary entrance response does not '
                    'establish equality of the complete click-time distributions; entrance current also '
                    'includes the time derivative of probability stored in the layer.',
  'status': 'Manuscript-reported rounded values, visually checked against PDF page 43 (Supplement page 30). '
            'No calibrated potentials, native curves, or verified numerical table export were located. This '
            'presentation preserves unfavorable as well as improved comparisons and is not a fresh '
            'computation.'},
 {'id': 'table-s9',
  'title': 'Fixed-confinement detector-parameter scan',
  'intro': 'Historical fixed-confinement scan at ω=1, k_z=π, amplitude width σ∥=0.5, T=20. [Fig. '
           'S3](../README.md#fig-s3) shows the associated legacy profile family. The CSV below transcribes '
           'the manuscript table rather than filling gaps from a centered campaign.',
  'implementation': 'Legacy one-sided normal-boundary family. The located source audit includes preparation '
                    'and detector-position qualifications; it is not a uniform centered reference family.',
  'panels': [{'label': 'Manuscript-reported restricted means',
              'csv': 'data/centered/tables/manuscript_reported/Table_S9_manuscript_rounded.csv',
              'columns': [{'key': 'kappa', 'label': 'κ', 'format': 'g', 'values': {'3.141592653589793': 'π'}},
                          {'key': 'mu_star20', 'label': 'Restricted mean μ*(20)', 'format': '.3f'}],
              'expected_rows': 9}],
  'dictionary': 'κ is the detector parameter; μ*(20)=∫₀²⁰S(t)dt is dimensionless restricted time. The '
                'numerical κ field represents π by its double-precision value. [Located source '
                'audit](manuscript_reported/Table_S9_located_source_audit.csv) records eight available audit '
                'entries, including saved-window and trapezoidal survival areas, θ, Δt, and numerical '
                'detector position; those areas are archived audit quantities and are not asserted to be the '
                'manuscript’s exact reduction.',
  'interpretation': 'The smallest manuscript entry among the tabulated settings occurs at κ=π, the scalar '
                    'matching point for the central longitudinal component. This is a finite-preparation '
                    'comparison, not a globally optimal detector setting. The located π audit gives '
                    '5.153081498198618 (5.153 at three decimals), whereas the manuscript gives 5.151; this '
                    'exceeds the displayed rounding and remains unresolved.',
  'status': 'Manuscript-reported rounded values, visually checked against PDF page 45 (Supplement page 32). '
            'A complete verified Table S9 export was not located: the mapped audit lacks κ=5. The κ=1 record '
            'has θ=π/3 and Δt=0.0008 while the other audited records have θ=0 and Δt=0.001; its logged '
            'output-directory label says κ=2 despite recorded κ=1. The κ=π detector coordinate is '
            'approximately 19.986666 rather than the other audited records’ 20. No centered row or different '
            'preparation is substituted.'},
 {'id': 'table-s10',
  'title': 'Longitudinal-momentum fit coefficients',
  'intro': 'Centered spinor ABC at T=20, κ=π, amplitude width σ∥=0.5. Each unweighted fit uses exactly '
           'ω={1,25,50,100,150,200}, with μ*(20;ω)=A₂₀+B₂₀√ω. See [Fig. S6](../README.md#fig-s6), [fit-input '
           'metrics](controls/T10/table_S10_raw_metrics_T20.csv), and [Fig. S6 numerical '
           'data](controls/S6/fig_S6_centered_T20.csv).',
  'implementation': 'Centered ghost-point roof closure with its compatible terminal half-weight norm.',
  'panels': [{'label': 'Three six-point fits',
              'csv': 'data/centered/tables/controls/T10/table_S10_centered_T20.csv',
              'columns': [{'key': 'kz_label', 'label': 'k_z', 'format': 'text', 'values': {'pi': 'π'}},
                          {'key': 'A20', 'label': 'A₂₀', 'format': '.3f'},
                          {'key': 'B20', 'label': 'B₂₀', 'format': '.3f'},
                          {'key': 'r_rms', 'label': 'r_rms', 'format': '.3f'},
                          {'key': 'n', 'label': 'n', 'format': 'd'}],
              'expected_rows': 3}],
  'dictionary': 'A₂₀ and r_rms have the dimensionless time units of μ*; B₂₀ is the coefficient of √ω. r_rms '
                'is the root mean square of the six fit residuals, not a numerical-discretization error. The '
                'canonical CSV additionally preserves R² and maximum absolute residual at available '
                'precision.',
  'interpretation': 'The fitted coefficients and residuals vary with longitudinal preparation even with the '
                    'same detector and horizon. These empirical fits describe the stated six-point sample; '
                    'they do not establish a universal confinement law.',
  'status': 'Archived derived numerical results, rendered from the existing full-precision CSV. The rounded '
            'values agree with PDF page 47 (Supplement page 34). The [historical '
            'reducer](../../../Solvers/centered/controls/analysis/T10/recalculate_table_S10.py) refers to an '
            'unavailable provenance helper. The [portable controls '
            'workflow](../../../Loaders/centered/controls.py) displays the archived results; it does not '
            'independently reconstruct every sweep run.'},
 {'id': 'table-s11',
  'title': 'Fixed-source lower-wall displacement',
  'intro': 'ω=200, T=40, k_z=κ=π, amplitude width σ∥=0.5, initial spinor (1,0)ᵀ. Source z_c=10 and numerical '
           'upper detector z_D≈19.986666 remain fixed while z_B changes. Runs 7750–7753 correspond, in '
           'order, to z_B=0,−5,−10,−15. They retain the original nonnegative-z nodes, h_z≈0.013333 and '
           'N_x=N_y=100, with Δt=0.0003. [Fig. S11](../README.md#fig-s11) shows the signed '
           'longitudinal-current maps.',
  'implementation': 'Centered spinor-ABC roof, reflecting Dirichlet lower wall; fixed-source geometry '
                    'family, distinct from a guide-length scan with a moving source.',
  'panels': [{'label': 'Grid and full-window statistics',
              'csv': 'data/centered/tables/controls/S11/table_S11_fixed_source_T40.csv',
              'columns': [{'key': 'zB', 'label': 'z_B', 'format': 'd'},
                          {'key': 'Nz', 'label': 'N_z', 'format': 'd'},
                          {'key': 'D40', 'label': 'D₄₀', 'format': '.3f'},
                          {'key': 'tau_cond40', 'label': 'Conditional mean', 'format': '.3f'},
                          {'key': 'mu_star40', 'label': 'Restricted mean μ*(40)', 'format': '.3f'}],
              'expected_rows': 4},
             {'label': 'Late window (3.6,40]',
              'csv': 'data/centered/tables/controls/S11/table_S11_fixed_source_T40.csv',
              'columns': [{'key': 'zB', 'label': 'z_B', 'format': 'd'},
                          {'key': 'DL40', 'label': 'D_L,40', 'format': '.3f'},
                          {'key': 'tL40', 'label': 'Late conditional mean', 'format': '.3f'}],
              'expected_rows': 4}],
  'dictionary': 'D₄₀=1−S(40), μ*(40)=∫₀⁴⁰S(t)dt, and the conditional mean uses the finite-window identity. '
                'D_L,40=S(3.6)−S(40) is the absolute late probability; its mean is conditioned only on '
                'detection in (3.6,40]. The common diagnostic split does not identify individual returns. '
                'Lengths and times are dimensionless physical coordinates, not oscillator-scaled axes.',
  'interpretation': 'The reported early probability is approximately 0.346 at all four positions. Moving the '
                    'wall downward reduces the full-window and late detected probabilities while increasing '
                    'the late conditional mean and restricted mean. The overall detected conditional mean is '
                    'nonmonotonic across the four positions, so the result is not a uniform delay of all '
                    'detections.',
  'status': 'Archived derived numerical results, using the existing full-precision CSV and its '
            'survival-history convention. Displayed values agree with PDF page 50 (Supplement page 37). '
            '[Original analysis '
            'recipe](../../../Solvers/centered/controls/analysis/S11/analyze_fixed_source_T40.py) and [plot '
            'metadata](controls/S11/plot_metadata.json) are included; the four complete native histories and '
            'spatial current histories are not bundled, so this is archived-table presentation rather than '
            'native reconstruction.'},
 {'id': 'table-s12',
  'title': 'Reflecting versus absorbing lower boundary',
  'intro': 'T=40, ω=1 and 200, with the reference source and spin-coupled upper detector. The scalar lower '
           'ABC has κ_lower=π and is partially absorbing. Cases 7801 (ω=1) and 7800 (ω=200) supply its '
           'native two-channel records. See [Fig. S12](../README.md#fig-s12) and the [native endpoint '
           'metrics](controls/T12/endpoint_comparison.csv).',
  'implementation': 'Centered spinor upper ABC in all rows; reflecting Dirichlet or scalar absorbing lower '
                    'closure. The different evidence conventions within the table are retained.',
  'panels': [{'label': 'Separate upper detection, lower escape and surviving mass',
              'csv': 'data/centered/tables/controls/T12/four_way_statistics.csv',
              'columns': [{'key': 'omega', 'label': 'ω', 'format': 'd'},
                          {'key': 'lower_boundary',
                           'label': 'Lower boundary',
                           'format': 'text',
                           'values': {'Dirichlet (rounded SI)': 'Dirichlet (rounded SI)',
                                      'Scalar ABC (native records)': 'Scalar ABC (native)'}},
                          {'key': 'D_top', 'label': 'D_top', 'format': '.3f'},
                          {'key': 'E_lower', 'label': 'E_lower', 'format': '.3f'},
                          {'key': 'S', 'label': 'S_in', 'format': '.3f'}],
              'expected_rows': 4},
             {'label': 'Upper-click times and late probability',
              'csv': 'data/centered/tables/controls/T12/four_way_statistics.csv',
              'columns': [{'key': 'omega', 'label': 'ω', 'format': 'd'},
                          {'key': 'lower_boundary',
                           'label': 'Lower boundary',
                           'format': 'text',
                           'values': {'Dirichlet (rounded SI)': 'Dirichlet (rounded SI)',
                                      'Scalar ABC (native records)': 'Scalar ABC (native)'}},
                          {'key': 'mean_top', 'label': 'Upper conditional mean', 'format': '.3f'},
                          {'key': 'D_late', 'label': 'D_L', 'format': '.3f'},
                          {'key': 'mean_late', 'label': 'Late conditional mean', 'format': '.3f'}],
              'expected_rows': 4}],
  'dictionary': 'D_top is cumulative upper-detector probability; E_lower is cumulative lower escape; S_in is '
                'mass still in the guide. Their sum is approximately one with the measured accounting defect '
                'retained. Probability of no upper click is S_in+E_lower, not S_in alone. '
                'D_L=D_top(40)−D_top(3.6) and the late mean refer only to upper events in (3.6,40]. All '
                'probabilities are absolute fractions of launch norm; conditional times are dimensionless.',
  'interpretation': 'The absorbing lower termination substantially reduces late upper clicks and their '
                    'conditional mean, while the early upper response agrees at displayed precision. The '
                    'termination is not transparent to the full wavepacket. Total norm loss includes lower '
                    'escape and cannot be substituted for upper-detection probability; accounting closure is '
                    'not a convergence test.',
  'status': 'Mixed evidence: the two scalar-lower-ABC rows are archived numerical results independently '
            'recomputed from bundled native accepted-step records by the [controls '
            'adapter](../../../Loaders/centered/controls.py). The two Dirichlet rows retain the '
            'archived rounded-reference convention: S_in=1−D₄₀ and D_L=D₄₀w_L with w_L retained to four decimals in this CSV (0.3942 and 0.5587), beyond the three decimals printed in Table S2. They are not '
            'replaced by more precise native Dirichlet values. Their existing CSV arithmetic precision does '
            'not create new source precision. Values agree with PDF page 53 (Supplement page 40) after the '
            'declared display rounding.'},
 {'id': 'long-time-summary',
  'title': 'Fig. S13 support (not a numbered Supplement table)',
  'intro': 'Case 7810: ω=200, centered spinor roof ABC and reflecting lower wall, k_z=κ=π, amplitude width '
           'σ∥=0.5, initial spinor (1,0)ᵀ; requested and saved support through T=200. [Fig. '
           'S13](../README.md#fig-s13) shows the profile. The manuscript estimate and current available '
           'support are intentionally separate.',
  'implementation': 'Centered ghost-point roof with compatible terminal half-weight norm; one confinement '
                    'only.',
  'panels': [{'label': 'Historical manuscript estimates from logged survival',
              'csv': 'data/centered/tables/manuscript_reported/FigS13_logged_estimates_manuscript_rounded.csv',
              'columns': [{'key': 'horizon', 'label': 'Horizon', 'format': 'g'},
                          {'key': 'D_logged_estimate',
                           'label': 'Reported detected fraction',
                           'format': '.3f'}],
              'expected_rows': 2},
             {'label': 'Current bundled derived bins and sampled norms',
              'csv': 'data/centered/tables/controls/7810/long_time_support.csv',
              'columns': [{'key': 'horizon', 'label': 'Horizon', 'format': 'd'},
                          {'key': 'derived_bin_D', 'label': 'Binned-event D', 'format': '.3f'},
                          {'key': 'sampled_survival', 'label': 'Sampled S', 'format': '.3f'},
                          {'key': 'derived_D_plus_sampled_S_minus_one', 'label': 'D+S−1', 'format': '.3e'}],
              'expected_rows': 4}],
  'dictionary': 'The historical D values are rounded estimates from the logged survival history at the '
                'manuscript endpoints. Current D integrates the included 0.01-wide accepted-event-derived '
                'bins; current S interpolates saved norms on their actual 0.025 cadence and divides by '
                'recorded launch norm. These are distinct clocks from the solver step Δt=0.00025. D+S−1 '
                'retains their measured accounting difference; it is not forced to zero.',
  'interpretation': 'The manuscript estimates approximately 15.7% of initial probability detected after 40 '
                    'through 189.95, about 73% of the probability undetected at 40. The separately supported '
                    'result at 200 is not a replacement for that historical estimate. Finite-window survival '
                    'is not permanent nondetection; this single-confinement extension determines neither the '
                    'unrestricted mean nor a long-horizon confinement law.',
  'status': 'The first panel is manuscript-reported rounded values, visually checked on PDF page 53. The '
            'second is recomputed from bundled derived bins and native sampled norm records, not the omitted '
            'complete accepted-step history. [Original plot data](../native/controls/7810/plot_data.csv), '
            '[metadata](../native/controls/7810/plot_metadata.json), and the [controls '
            'reducer](../../../Loaders/centered/controls.py) define the supported calculation. Run `python '
            '-m Loaders.centered.supplement_tables_b --check` to check the small current-support CSV without '
            'reading the research root.'}]


def build_sections(repo_root: Path) -> list[dict]:
    """Return pure presentation specifications; numerical cells come from CSV."""
    return copy.deepcopy(SECTIONS)


def long_time_csv(repo_root: Path) -> str:
    """Reuse the established event-overlap convention on existing derived bins."""
    import numpy as np
    from .numerics import events

    base = repo_root / "data/centered/native/controls/7810"
    data = np.genfromtxt(base / "plot_data.csv", names=True, delimiter=",")
    config = json.loads((base / "run.json").read_text())
    times = np.load(base / "prob_times.npy", allow_pickle=False)
    mass = np.load(base / "total_probs.npy", allow_pickle=False)
    if (times.shape != mass.shape or times.size == 0 or not np.all(np.diff(times) > 0)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(mass))):
        raise ValueError("Invalid sampled-norm clock")
    rows = []
    for horizon in (40, 80, 100, 200):
        if not times[0] <= horizon <= times[-1]:
            raise ValueError(f"Sampled norms do not bracket horizon {horizon}")
        if data["t_left"][0] > 0 or data["t_right"][-1] < horizon:
            raise ValueError(f"Derived bins do not cover [0,{horizon}]")
        detected = events(data["t_left"], data["t_right"], data["detected_probability"], 0, horizon)["probability"]
        survival = float(np.interp(horizon, times, mass) / config["P0_init64"])
        rows.append({"horizon": horizon, "derived_bin_D": detected,
                     "sampled_survival": survival,
                     "derived_D_plus_sampled_S_minus_one": detected + survival - 1})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def check_generated_inputs(repo_root: Path) -> bool:
    path = repo_root / "data/centered/tables/controls/7810/long_time_support.csv"
    return path.read_text(encoding="utf-8") == long_time_csv(repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Detect stale current-support CSV without writing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    target = root / "data/centered/tables/controls/7810/long_time_support.csv"
    rendered = long_time_csv(root)
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != rendered:
            print("Stale Fig. S13 current-support CSV")
            return 1
        print("Fig. S13 current-support CSV matches bundled inputs")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(target.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

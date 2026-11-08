#!/usr/bin/env python3
"""Reproduce manuscript Fig. 2 from completed centered native mass histories.

Run: python plot_fig2_centered.py --run-root /path/to/1.Main --output /new/output
Requires numpy, scipy and matplotlib. Inputs are read-only; no solver is run.
Only Fig. 2 is exported. Omega=300 included; omega=400 excluded.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'fig2_centered_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
import numpy as np
import scipy
from scipy.integrate import simpson

HERE = Path(__file__).resolve().parent
RUNS = {
    1: '5540.Spin_Gauss_centeredABC_Omega=1_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3701390',
    25: '5042.Spin_Gauss_centeredABC_Omega=25_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704244',
    50: '5047.Spin_Gauss_centeredABC_Omega=50_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3715452',
    75: '5043.Spin_Gauss_centeredABC_Omega=75_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704245',
    100: '5541.Spin_Gauss_centeredABC_Omega=100_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3701391',
    150: '5044.Spin_Gauss_centeredABC_Omega=150_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704246',
    200: '5045.Spin_Gauss_centeredABC_Omega=200_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704247',
    300: '7100.Spin_Gauss_centeredABC_Omega=300_L=20_T=40_Nxy=100_Nz=1500_dt=0.0002_baseline_FG_v1_full_job3710751',
}
PREPARATION = dict(Lz=20., z0_init=10., k0=np.pi, k_bc=np.pi,
                   sigma_z=.5, theta=0., phi=0.)
FILES = ('revision_run.json', 'branch_refinement_case.json', 'constants.npz',
         'revision_step_data.csv', 'source_snapshot.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def scalar_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].item() for k in z.files}


def same(actual, expected, label):
    if actual == 'pi':
        actual = np.pi
    if isinstance(expected, str):
        require(actual == expected, f'{label}: {actual!r} != {expected!r}')
    else:
        require(np.isfinite(actual) and np.isclose(actual, expected, rtol=0, atol=1e-12),
                f'{label}: {actual!r} != {expected!r}')


def load_run(root, omega, folder):
    directory = (root / folder).resolve()
    require(directory.is_dir(), f'Missing selected run: {directory}')
    hashes = {name: sha256(directory / name) for name in FILES}
    run = json.loads((directory / 'revision_run.json').read_text())
    branch = json.loads((directory / 'branch_refinement_case.json').read_text())
    const = scalar_npz(directory / 'constants.npz')
    config = run['configuration']
    for label, record in [('revision_run.configuration', config),
                          ('branch_refinement_case', branch), ('constants', const)]:
        for key, value in PREPARATION.items():
            actual_key = 'z0' if label == 'constants' and key == 'z0_init' else key
            same(record[actual_key], value, f'{omega}: {label}.{actual_key}')
        for key, value in dict(omega=omega, endpoint_closure='centered',
                               endpoint_mass_weight=.5, metric='half_weight_last_z_node',
                               Nx=100, Ny=100, Nz=1500, grid_generator_nz=1500,
                               state_dtype='complex64', T_final=40.).items():
            same(record[key], value, f'{omega}: {label}.{key}')
        same(record['dt'], config['dt'], f'{omega}: {label}.dt')
    for key in ('hx', 'hy', 'hz', 'z_roof_actual', 'Lx', 'Ly'):
        same(const[key], config[key], f'{omega}: constants.{key}')
    same(config['real_dtype'], 'float32', 'coordinate precision')
    same(branch['coordinate_dtype'], 'float32', 'branch coordinate precision')
    same(config['matrix_assembly'], 'complex64_then_solve_complex128', 'matrix/solve precision')
    same(branch['matrix_dtype'], 'complex64', 'branch matrix precision')
    same(const['matrix_dtype'], 'complex64', 'constant matrix precision')
    same(config['initial_preparation_dtype'], 'complex64', 'preparation precision')
    same(config['potential_sampling_dtype'], 'float32', 'potential precision')
    same(run['actual_time'], 40., 'actual_time')
    require(run['status'] == 'COMPLETED_NUMERICAL_RUN_NOT_CONVERGENCE_CERTIFIED',
            f'Unexpected completion status: {directory}')
    require(hashes['source_snapshot.py'] == run['updated_source_sha256'],
            f'Source snapshot hash mismatch: {directory}')

    # All columns are checked, not just the columns used for plotting.
    a = np.genfromtxt(directory / 'revision_step_data.csv', delimiter=',', names=True,
                      dtype=np.float64, encoding='utf-8')
    require(a.ndim == 1 and len(a) > 1, f'Invalid step history: {directory}')
    require(all(np.isfinite(a[k]).all() for k in a.dtype.names),
            f'Nonfinite CSV value: {directory}')
    left, right, probability = a['t_left'], a['t_right'], a['roof_probability']
    dt = right - left
    require(np.all(probability >= 0), f'Negative detector probability: {directory}')
    require(np.all(dt > 0), f'Nonpositive duration: {directory}')
    require(np.array_equal(a['step'], np.arange(1, len(a) + 1)),
            f'Missing, duplicate, or unordered steps: {directory}')
    require(np.array_equal(left[1:], right[:-1]), f'Time gaps or overlaps: {directory}')
    require(left[0] == 0 and right[-1] == 40, f'Incomplete history: {directory}')
    require(len(a) == run['completed_step'], f'Completion count mismatch: {directory}')
    # Respect the solver's documented single-shorter-final-step endpoint policy.
    # The reconstruction always uses each actual duration, including the last.
    require(np.allclose(dt[:-1], config['dt'], rtol=0, atol=1e-11),
            f'Unexpected nonfinal step size: {directory}')
    require(0 < dt[-1] <= config['dt'] + 1e-11, f'Invalid final step size: {directory}')
    require(np.allclose(left, np.arange(len(a)) * config['dt'], rtol=0, atol=1e-11),
            f'Unexpected native step boundaries: {directory}')
    require(np.all(a['mass_stored'] >= 0), f'Negative stored mass: {directory}')
    p0 = float(run['P0_init64'])
    require(np.isfinite(p0) and p0 > 0, f'Invalid P0: {directory}')
    same(a['mass_before'][0], p0, 'initial stored mass')
    require(np.array_equal(a['mass_before'][1:], a['mass_stored'][:-1]),
            f'Discontinuous mass history: {directory}')

    # The ONLY detector normalization. roof_probability already includes kappa,
    # hx*hy and duration. The endpoint mass weight must NOT be applied here.
    t = (left + right) / 2
    g = probability / (p0 * dt)
    require(np.isfinite(g).all() and np.all(g >= 0), 'Invalid reconstructed density')
    roof_sum = math.fsum(probability)
    D40 = roof_sum / p0
    S40 = float(a['mass_stored'][-1]) / p0
    delta = D40 - (1 - S40)
    # Signed budget errors are reported, never used to alter the curves.
    late = (t >= 8) & (t <= 40)
    row = dict(omega=omega, selected_run_path=str(directory), status=run['status'],
               dt=config['dt'], dt_min=float(dt.min()), dt_max=float(dt.max()),
               dt_final=float(dt[-1]), final_step_shortened=bool(dt[-1] < config['dt'] - 1e-11),
               steps=len(a), Nx=config['Nx'], Ny=config['Ny'], Nz=config['Nz'],
               hx=config['hx'], hy=config['hy'], hz=config['hz'],
               state_precision=config['state_dtype'], coordinate_precision=config['real_dtype'],
               matrix_precision=const['matrix_dtype'], solve_precision='complex128',
               endpoint_closure=config['endpoint_closure'], endpoint_mass_weight=.5,
               L=20., z0=10., k0=np.pi, kappa=np.pi, gaussian_amplitude_width=.5,
               theta=0., phi=0., detector_z=config['z_roof_actual'],
               t_start=float(left[0]), t_end=float(right[-1]), P0_init64=p0,
               roof_probability_sum=roof_sum, mass_stored_final=float(a['mass_stored'][-1]),
               D40=D40, S40=S40, one_minus_S40=1-S40,
               D40_minus_one_minus_S40=delta, budget_residual_1_minus_D40_minus_S40=-delta,
               event_sum_minus_recorded_cumulative=roof_sum-run['cumulative_roof_probability'],
               final_mass_minus_recorded_mass=float(a['mass_stored'][-1])-run['current_accepted_prefix_mass64'],
               full_peak=float(g.max()), late_peak=float(g[late].max()),
               validation='PASS; completion is not convergence certification')
    # Fig. 2 retains the original survival-integral definition and Simpson
    # quadrature, now using every native accepted-step endpoint and P0_init64.
    endpoint_t = np.r_[left[0], right]
    survival = np.r_[p0, a['mass_stored']] / p0
    mu_simpson = float(simpson(survival, x=endpoint_t))
    mu_trapezoid = math.fsum(dt * (a['mass_before'] + a['mass_stored']) / (2 * p0))
    require(abs(mu_trapezoid - np.trapezoid(survival, endpoint_t)) < 1e-11,
            'Independent trapezoid summation failed')
    first_moment = math.fsum(t * probability) / p0
    mu_events = first_moment + 40 * (1 - D40)
    require(0 <= mu_simpson <= 40, 'Restricted mean outside [0,40]')
    row.update(sqrt_omega=float(np.sqrt(omega)), mu_star_T40=mu_simpson,
               mu_native_trapezoid=mu_trapezoid,
               mu_native_simpson_raw=mu_simpson * p0,
               simpson_minus_trapezoid=mu_simpson - mu_trapezoid,
               event_first_moment=first_moment,
               mu_from_events_M_plus_T_one_minus_D=mu_events,
               mu_simpson_minus_event_identity=mu_simpson - mu_events,
               mu_simpson_minus_M_plus_T_S40=mu_simpson - first_moment - 40*S40,
               positive_survival_increments=int(np.count_nonzero(np.diff(survival) > 0)),
               maximum_positive_survival_increment=float(max(0, np.diff(survival).max())))
    details = dict(input_sha256=hashes, configuration=config,
                   source_snapshot_matches_recorded_hash=True,
                   time_sha256=array_sha(t), density_sha256=array_sha(g),
                   endpoint_time_sha256=array_sha(endpoint_t), survival_sha256=array_sha(survival),
                   full_peak_time=float(t[np.argmax(g)]),
                   late_peak_time=float(t[late][np.argmax(g[late])]),
                   all_csv_values_finite=True, probabilities_nonnegative=True,
                   durations_positive=True, contiguous_nonduplicated_history=True)
    return dict(omega=omega, directory=directory, t=t, g=g, row=row, details=details)



def style_axes(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, width=.9)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(True, alpha=.10)


def render(report, output):
    plt.rcdefaults()
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12,
                         "axes.labelsize":13, "legend.fontsize":11,
                         "savefig.bbox":"tight", "pdf.fonttype":42, "ps.fonttype":42,
                         "figure.facecolor":"white", "axes.facecolor":"white",
                         "savefig.facecolor":"white", "savefig.transparent":False})
    w = np.array(tuple(RUNS), dtype=float)
    y = np.array([r["mu_star_T40"] for r in report["runs"]])
    a, b = report["fit"]["A"], report["fit"]["B"]
    grid = np.linspace(w.min(), w.max(), 500)
    root_grid = np.linspace(np.sqrt(w.min()), np.sqrt(w.max()), 500)
    fig, ax = plt.subplots(figsize=(8.6,5.2))
    style_axes(ax)
    ax.plot(grid, a+b*np.sqrt(grid), color="#050505", lw=.9,
            label=rf"fit: $\mu^*(40)\approx {a:.3f}+{b:.3f}\sqrt{{\omega}}$")
    points, = ax.plot(w,y,ls="none",marker="o",ms=7,mew=1.2,mec="#3F6AA8",mfc="#3F6AA8",
                     label=r"data: $\mu^*(40)$")
    ax.set_xlabel(r"$\omega$",fontweight="bold")
    ax.set_ylabel(r"$\mu^*(T)$",fontweight="bold")
    ax.set_xlim(-14,315)
    ax.set_ylim(4.5,25.5)
    ax.legend(loc="upper left",frameon=True)
    ins = ax.inset_axes([.58,.10,.36,.39])
    style_axes(ins)
    ins.plot(root_grid,a+b*root_grid,color="#050505",lw=.8)
    ipoints, = ins.plot(np.sqrt(w),y,ls="none",marker="o",ms=4.5,mew=.9,mec="#3F6AA8",mfc="#3F6AA8")
    ins.set_xlabel(r"$\sqrt{\omega}$",fontsize=10,fontweight="bold",labelpad=1)
    ins.set_ylabel(r"$\mu^*(T)$",fontsize=10,fontweight="bold",labelpad=1)
    ins.tick_params(which="major",labelsize=9,length=4)
    ins.tick_params(which="minor",length=2)
    ins.set_xlim(np.sqrt(w.min())-.5,np.sqrt(w.max())+.5)
    ins.set_ylim(4.5,25.5)
    np.testing.assert_array_equal(points.get_xdata(),w)
    np.testing.assert_array_equal(points.get_ydata(),y)
    np.testing.assert_array_equal(ipoints.get_xdata(),np.sqrt(w))
    np.testing.assert_array_equal(ipoints.get_ydata(),y)
    require(min(y.min(), (a+b*np.sqrt(grid)).min()) > 4.5 and
            max(y.max(), (a+b*np.sqrt(grid)).max()) < 25.5,
            'Data or fitted curve exceeds original y limits; revise and report limits')
    report['axes'] = dict(main_xlim=list(ax.get_xlim()), main_ylim=list(ax.get_ylim()),
                          inset_xlim=list(ins.get_xlim()), inset_ylim=list(ins.get_ylim()),
                          original_main_xlim=[-14,315], original_main_ylim=[4.5,25.5],
                          limit_changes='None: original main and inset limits retained.',
                          figsize_inches=[8.6,5.2], png_dpi=600)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(output/f"Fig2_centered.{ext}",dpi=600)
    plt.close(fig)
    report["validation"]["main_and_inset_all_eight_points"] = "PASS: plotted artists equal calculated arrays"
    np.savetxt(output/"fit_curve_main_axis.txt", np.column_stack((grid,a+b*np.sqrt(grid))),
               header="omega  fitted_mu_star_T40",fmt="%.17g")
    np.savetxt(output/"fit_curve_inset_axis.txt", np.column_stack((root_grid,a+b*root_grid)),
               header="sqrt_omega  fitted_mu_star_T40",fmt="%.17g")


def fit_values(omega, values):
    x = np.sqrt(omega)
    design = np.column_stack((np.ones(len(x)), x))
    beta, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    require(rank == 2, 'Rank-deficient fit')
    independent_b = math.fsum((x-x.mean())*(values-values.mean())) / math.fsum((x-x.mean())**2)
    independent_a = float(values.mean()-independent_b*x.mean())
    require(np.allclose(beta, [independent_a, independent_b], rtol=0, atol=1e-12),
            'Independent regression check failed')
    fitted = design @ beta
    residuals = values-fitted
    sse = float(residuals @ residuals)
    return dict(A=float(beta[0]), B=float(beta[1]), R_squared=1-sse/float(np.sum((values-values.mean())**2)),
                RMSE=float(np.sqrt(sse/len(values))), SSE=sse, model='A+B*sqrt(omega)',
                n_points=len(omega), method='Unweighted ordinary least squares on all eight calculated values',
                fit_domain=[float(omega.min()), float(omega.max())]), fitted, residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=HERE.parent)
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('Fig2_centered.pdf','Fig2_centered.png','README.txt','selected_runs_and_fit.csv',
                 'audit_provenance.json','fit_curve_main_axis.txt','fit_curve_inset_axis.txt'):
        require(not (args.output/name).exists(), f'Refusing to overwrite existing output: {args.output/name}')
    runs = [load_run(args.run_root, w, path) for w, path in RUNS.items()]
    require(tuple(r['omega'] for r in runs) == (1,25,50,75,100,150,200,300), 'Incorrect selected set')
    for r in runs:
        same(r['row']['detector_z'], 19.986665725708008, 'common actual detector plane')
    omega = np.array(tuple(RUNS), dtype=float)
    y = np.array([r['row']['mu_star_T40'] for r in runs])
    fit, fitted, residuals = fit_values(omega, y)
    trap_fit, _, _ = fit_values(omega, np.array([r['row']['mu_native_trapezoid'] for r in runs]))
    event_fit, _, _ = fit_values(omega, np.array([r['row']['mu_from_events_M_plus_T_one_minus_D'] for r in runs]))
    for r, f, residual in zip(runs, fitted, residuals):
        r['row'].update(fitted_mu=float(f), residual_data_minus_fit=float(residual))
    source = json.loads((HERE/'style_source_provenance.json').read_text())
    report = dict(scope='Only manuscript Fig. 2; completed centered runs, omega300 included, omega400 excluded.',
                  definition='mu*(40)=integral_0^40 S(t)dt; S=stored mass/P0_init64',
                  method='scipy.integrate.simpson on every native accepted-step endpoint, including initial mass at t=0',
                  historical_method='Original Fig.2 used Simpson on sparse unnormalized prob_times/total_probs samples. New centered values use native endpoint masses normalized once by P0_init64.',
                  event_check='M=sum(t_mid*roof_probability)/P0; mu_event=M+40*(1-D40). Discrepancies are reported without correction.',
                  processing='No smoothing, time shifts, extrapolation, binning, conditioning, or fabricated values. Fits use all eight points.',
                  fit=fit, alternate_fits=dict(native_trapezoid=trap_fit,event_identity=event_fit),
                  style_source=source, validation={},
                  runs=[dict(**r['row'], **r['details']) for r in runs],
                  numpy_version=np.__version__, scipy_version=scipy.__version__, matplotlib_version=matplotlib.__version__,
                  script_sha256=sha256(__file__),
                  limitations='Run completion and quadrature agreement are not convergence certificates. No error bars or confidence intervals inferred.')
    render(report, args.output)
    for r in runs:
        require(all(sha256(r['directory']/name) == digest for name, digest in r['details']['input_sha256'].items()),
                'Input changed during rendering')
    for key, digest in source['original_file_sha256'].items():
        path = Path(key)
        if path.exists():
            require(sha256(path) == digest, f'Original source changed: {path}')
    report['validation'].update(all_runs_pass_native_audit=True, original_inputs_unchanged=True,
                               independent_regression=True,
                               max_abs_simpson_minus_trapezoid=max(abs(r['row']['simpson_minus_trapezoid']) for r in runs),
                               max_abs_simpson_minus_event_identity=max(abs(r['row']['mu_simpson_minus_event_identity']) for r in runs))
    with (args.output/'selected_runs_and_fit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(runs[0]['row']))
        writer.writeheader();writer.writerows(r['row'] for r in runs)
    (args.output/'audit_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = [
        'Fig. 2 centered reproduction - 2026-10-18',
        'Omega=1,25,50,75,100,150,200,300. Omega400 excluded. Only Fig.2 generated.',
        'Restricted mean: mu*(40)=integral_0^40 S(t)dt, S=stored mass/P0_init64.',
        'Simpson quadrature on every accepted-step endpoint, including initial t=0.',
        'Original source used sparse raw norms; native mass/P0 is used for centered normalization consistency.',
        'No total_probs derivative, display bins, smoothing, shifts, extrapolation or conditional normalization.',
        'Independent native trapezoid and event identity checks are in the CSV and JSON.',
        'M=sum(t_mid*roof_probability)/P0; event identity mu=M+40*(1-D40).',
        'Budget and identity discrepancies are reported without forcing agreement.',
        '',
        'All eight inputs pass common preparation, centered closure, weight0.5, grid/precision,',
        'finite values, nonnegative roof events, positive durations, and contiguous0..40 history checks.',
        'Common L20,z0=10,k0=kappa=pi,Gaussian amplitude width0.5,theta=phi=0.',
        'Actual detector z=19.986665725708008. Native dt/grid/precision recorded per run in CSV.',
        '',
        'Original style source: exact RGB match between manuscript page5 image and mu_Fit.png.',
        'Fonts, markers, colors, line widths, inward ticks, grid, legend, inset and aspect preserved.',
        'Original main x range [-14,315] retained.',
        'Original inset x range [0.5,sqrt(300)+0.5] and both y ranges [4.5,25.5] retained.',
        'Original files and prior Fig.1 exports were not overwritten.',
        '',
        f"Fit: mu*(40) = {fit['A']:.12f} + {fit['B']:.12f} sqrt(omega)",
        f"R^2={fit['R_squared']:.12f}; RMSE={fit['RMSE']:.12f}; all eight points equally weighted.",
        f"Max |Simpson-trapezoid|={report['validation']['max_abs_simpson_minus_trapezoid']:.12g}",
        f"Max |Simpson-event identity|={report['validation']['max_abs_simpson_minus_event_identity']:.12g}",
        '',
        'omega       mu*(40)            residual (data-fit)',
        *[f"{r['omega']:5d}   {r['row']['mu_star_T40']:.12f}   {r['row']['residual_data_minus_fit']:+.12f}" for r in runs],
        '',
        'Run completion is not convergence certification. This is a descriptive finite-window fit.',
        'No numerical uncertainty estimates were supplied; no error bars or confidence intervals added.',
        'Rerun: python plot_fig2_centered.py --run-root /path/to/1.Main --output /new/output',
        'Dependencies: numpy, scipy, matplotlib. Keep style_source_provenance.json alongside script.',
        'Vector PDF and 600-dpi PNG.']
    (args.output/'README.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(fit=fit,validation=report['validation'],axes=report['axes']),indent=2))


if __name__ == '__main__':
    main()

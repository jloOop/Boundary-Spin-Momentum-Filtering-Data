#!/usr/bin/env python3
"""Reproduce manuscript Fig. 3 from completed centered native mass histories.

Run: python plot_fig3_centered.py --run-root /path/to/1.Main --output /new/output
Requires numpy, scipy and matplotlib. Inputs are read-only; no solver is run.
Only Fig. 3 is exported, with screenshot-derived horizon colors.
Refined omega=300 case7706 is included; omega=400 is excluded.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'fig3_centered_mpl'))
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
    300: '7706.Spin_Gauss_centeredABC_Omega=300_L=20_zc=10_T=40_Nxy=100_Nz=1500_dt=0.0001_baseline_half_dt_v1_full_job3724723',
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
    endpoint_t = np.r_[left[0], right]
    survival = np.r_[p0, a['mass_stored']] / p0
    details = dict(input_sha256=hashes, configuration=config,
                   source_snapshot_matches_recorded_hash=True,
                   time_sha256=array_sha(t), density_sha256=array_sha(g),
                   endpoint_time_sha256=array_sha(endpoint_t), survival_sha256=array_sha(survival),
                   full_peak_time=float(t[np.argmax(g)]),
                   late_peak_time=float(t[late][np.argmax(g[late])]),
                   all_csv_values_finite=True, probabilities_nonnegative=True,
                   durations_positive=True, contiguous_nonduplicated_history=True)
    return dict(omega=omega, directory=directory, t=t, g=g, row=row, details=details,
                steps=a, endpoint_t=endpoint_t, survival=survival)



HORIZONS = (10., 20., 30., 40.)
# Four colors selected from the supplied screenshot; colors encode horizons.
PALETTE = ('#1B0264', '#136384', '#72CE6D', '#E7E656')
ENDPOINT_ROUNDOFF_TOLERANCE = 1e-12


def horizon_statistics(run, T):
    a = run['steps']
    left, right, p = a['t_left'], a['t_right'], a['roof_probability']
    p0 = run['row']['P0_init64']
    require(0 < T <= right[-1], 'Horizon outside recorded support; no extrapolation')
    included = left < T
    l, r, p = left[included], right[included], p[included]
    clipped_right = np.minimum(r, T)
    fraction = (clipped_right-l)/(r-l)
    require(np.all((fraction > 0) & (fraction <= 1)), 'Invalid horizon overlap')
    # Constant native step rate, integrated only over the covered fraction.
    # Full events are not multiplied by dt, area, kappa or endpoint weight again.
    selected_probability = p*fraction
    D = math.fsum(selected_probability)/p0
    M = math.fsum((l+clipped_right)/2*selected_probability)/p0
    require(D > 0, 'Conditional mean undefined because no detections were recorded')
    conditional = M/D

    # Preserve original Fig. 3's piecewise-linear mass integration convention.
    # Interpolate one terminal value only when the horizon lies inside a step.
    endpoint_t, survival = run['endpoint_t'], run['survival']
    nearest = int(np.argmin(np.abs(endpoint_t-T)))
    raw_nearest = float(endpoint_t[nearest])
    offset = raw_nearest-T
    if abs(offset) <= ENDPOINT_ROUNDOFF_TOLERANCE:
        tt = endpoint_t[:nearest+1].copy()
        ss = survival[:nearest+1].copy()
        tt[-1] = T  # e.g. 29.999999999999996 is the nominal T=30 endpoint.
        method = 'exact_endpoint' if offset == 0 else 'roundoff_endpoint_match'
        bracket_left = bracket_right = raw_nearest
        interpolation_fraction = 1.
    else:
        k = int(np.searchsorted(endpoint_t, T))
        require(0 < k < len(endpoint_t), 'No bracket for requested horizon')
        bracket_left, bracket_right = float(endpoint_t[k-1]), float(endpoint_t[k])
        interpolation_fraction = (T-bracket_left)/(bracket_right-bracket_left)
        S_at_T = survival[k-1] + interpolation_fraction*(survival[k]-survival[k-1])
        tt = np.r_[endpoint_t[:k], T]
        ss = np.r_[survival[:k], S_at_T]
        method = 'linear_terminal_interpolation'
    require(np.all(np.diff(tt)>0), 'Nonpositive prefix duration')
    S = float(ss[-1])
    mu = math.fsum(np.diff(tt)*(ss[:-1]+ss[1:])/2)
    require(abs(mu - float(np.trapezoid(ss,tt))) < 1e-11, 'Independent trapezoid check failed')
    mu_simpson = float(simpson(ss, x=tt))
    mu_events = M + T*(1-D)
    require(0 <= D <= 1 and 0 <= conditional <= T and 0 <= mu <= T,
            'Horizon statistic outside probability/time bounds')
    partial_indices = np.flatnonzero(fraction < 1)
    partial = [dict(step=int(a['step'][included][j]),t_left=float(l[j]),t_right=float(r[j]),
                    included_duration=float(clipped_right[j]-l[j]),fraction=float(fraction[j]))
               for j in partial_indices]
    substantial_partial = [x for x in partial if ENDPOINT_ROUNDOFF_TOLERANCE < x['included_duration']
                            < x['t_right']-x['t_left']-ENDPOINT_ROUNDOFF_TOLERANCE]
    row = dict(omega=run['omega'], T=T, D_T=D, S_T=S, M_T=M,
               conditional_mean=conditional, mu_star=mu,
               survival_contribution=T*S, one_minus_S=1-S,
               D_minus_one_minus_S=D-(1-S), closure_1_minus_S_minus_D=1-S-D,
               decomposition_residual_mu_minus_M_minus_TS=mu-M-T*S,
               event_restricted_mean=mu_events, mu_minus_event_identity=mu-mu_events,
               mu_native_simpson=mu_simpson, simpson_minus_plotted_mu=mu_simpson-mu,
               horizon_survival_method=method, nearest_endpoint_raw=raw_nearest,
               nearest_endpoint_minus_T=offset, survival_bracket_left=bracket_left,
               survival_bracket_right=bracket_right, survival_interpolation_fraction=interpolation_fraction,
               partial_event_steps=json.dumps(partial,separators=(',',':')),
               substantial_partial_event_count=len(substantial_partial),
               selected_run_path=str(run['directory']), P0_init64=p0, nominal_dt=run['row']['dt'])
    return row


def style(ax):
    # Copied from original make_figures.py style() (lines16-19).
    ax.minorticks_on()
    ax.tick_params(direction='out',top=False,right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(.6)
    ax.margins(x=.01,y=.05)


def render(rows, output):
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.labelsize':11, 'pdf.fonttype':42,
                         'path.simplify':False, 'agg.path.chunksize':0,
                         'figure.facecolor':'white', 'savefig.facecolor':'white'})
    omega=np.array(tuple(RUNS),dtype=float)
    keys=('D_T','conditional_mean','mu_star')
    ylabels=(r'$D_T$',r'$\overline{\tau}_T^{\mathrm{cond}}$',r'$\mu^*(T)$')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for T,color in zip(HORIZONS,PALETTE):
        selected=[r for r in rows if r['T']==T]
        require([r['omega'] for r in selected]==list(RUNS),'Incomplete or unordered horizon')
        for ax,key,ylabel in zip(axes,keys,ylabels):
            line,=ax.plot(omega,[r[key] for r in selected],marker='o',lw=1,
                          color=color,label=f'T = {T:g}')
            ax.set(xlabel=r'$\omega$',ylabel=ylabel)
            style(ax)
            require(np.array_equal(line.get_xdata(),omega) and
                    np.array_equal(line.get_ydata(),[r[key] for r in selected]),
                    'Plotted points differ from computed statistics')
    for ax,letter in zip(axes,'abc'):
        ax.text(0,1.022,f'({letter})',transform=ax.transAxes,
                ha='left',va='bottom',fontsize=11)
    axes[0].legend(fontsize=9)
    require(fig._suptitle is None and all(not ax.get_title() for ax in axes),'Unexpected figure title')
    limits=[]
    for ax,key in zip(axes,keys):
        lo,hi=ax.get_ylim();xlo,xhi=ax.get_xlim()
        values=np.array([r[key] for r in rows])
        require(lo < values.min() and hi > values.max() and xlo < omega.min() and xhi > omega.max(),
                'A plotted point would be clipped')
        limits.append(dict(statistic=key,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())))
    fig.savefig(output/'Fig3_centered.pdf',metadata={'Creator':'Centered native runs; plot_fig3_centered.py',
                                                    'CreationDate':None,'ModDate':None})
    fig.savefig(output/'Fig3_centered.png',dpi=600)
    plt.close(fig)
    return dict(panel_labels=['(a)','(b)','(c)'],ylabels=ylabels,palette=PALETTE,
                figsize_inches=[13,4],png_dpi=600,limits=limits,
                axis_policy='Original autoscale margins x=.01,y=.05 retained; limits adapt to all eight centered runs through omega300, excluding omega400.',
                plotted_values=len(rows)*len(keys),all_plotted_artists_equal_calculated_arrays=True)


def write_table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=HERE.parent)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in ('Fig3_centered.pdf','Fig3_centered.png','selected_runs.csv',
                 'horizon_statistics.csv','audit_provenance.json','README.txt','plot_data.npz'):
        require(not (args.output/name).exists(), f'Refusing to overwrite existing output: {args.output/name}')
    runs=[load_run(args.run_root,w,path) for w,path in RUNS.items()]
    require(tuple(r['omega'] for r in runs)==(1,25,50,75,100,150,200,300),'Incorrect run set')
    for run in runs:
        same(run['row']['detector_z'],19.986665725708008,'actual detector position')
    rows=[horizon_statistics(run,T) for T in HORIZONS for run in runs]
    for run in runs:
        series=[r for r in rows if r['omega']==run['omega']]
        require(np.all(np.diff([r['D_T'] for r in series])>=0),'Detected fraction decreases with horizon')
        require(np.all(np.diff([r['mu_star'] for r in series])>=0),'Restricted mean decreases with horizon')
        last=series[-1]
        require(last['D_T']==run['row']['D40'] and last['S_T']==run['row']['S40'],'T40 identity check failed')
    # Prior Fig. 2 has an older omega300 run. Check identical inputs strictly,
    # and report the explicitly requested refinement separately.
    earlier=args.run_root/'Fig2_centered_comparison_2026-10-18/without400/audit_provenance.json'
    previous_check='Prior Fig. 2 audit not present; not required for standalone regeneration.'
    if earlier.exists():
        previous=json.loads(earlier.read_text())
        differences=[]; unchanged=[]; refined=[]
        for old in previous['runs']:
            current_run=next(r for r in runs if r['omega']==old['omega'])
            current=next(r for r in rows if r['omega']==old['omega'] and r['T']==40)
            same_path=Path(old['selected_run_path']).resolve()==current_run['directory']
            same_hashes=old['input_sha256']==current_run['details']['input_sha256']
            if same_path and same_hashes:
                require(abs(current['mu_native_simpson']-old['mu_star_T40'])<1e-12,
                        'Fig. 2 Simpson result changed for unchanged run')
                unchanged.append(old['omega'])
                differences.append(current['mu_star']-old['mu_star_T40'])
            else:
                require(old['omega']==300 and Path(old['selected_run_path']).name==
                        '7100.Spin_Gauss_centeredABC_Omega=300_L=20_T=40_Nxy=100_Nz=1500_dt=0.0002_baseline_FG_v1_full_job3710751' and current_run['directory'].name=='7706.Spin_Gauss_centeredABC_Omega=300_L=20_zc=10_T=40_Nxy=100_Nz=1500_dt=0.0001_baseline_half_dt_v1_full_job3724723',
                        'Unexpected run/input change relative to prior Fig. 2')
                refined.append(dict(omega=300,previous_run=old['selected_run_path'],
                    selected_run=str(current_run['directory']),previous_dt=old['dt'],
                    selected_dt=current_run['row']['dt'],
                    previous_mu_native_simpson=old['mu_star_T40'],
                    selected_mu_native_simpson=current['mu_native_simpson'],
                    selected_minus_previous=current['mu_native_simpson']-old['mu_star_T40']))
        previous_check=dict(status='PASS for unchanged inputs; requested refined run reported separately',
            source=str(earlier.resolve()),source_sha256=sha256(earlier),
            unchanged_omegas=unchanged,refined_runs=refined,
            max_abs_trapezoid_minus_Fig2_Simpson_unchanged=max((abs(x) for x in differences),default=0.),
            note='Fig. 3 retains native piecewise-linear integration; Fig. 2 used Simpson. No convergence claim is inferred from two timesteps.')
    source=json.loads((HERE/'style_source_provenance.json').read_text())
    figure=render(rows,args.output)
    for run in runs:
        require(all(sha256(run['directory']/name)==digest for name,digest in run['details']['input_sha256'].items()),
                'Selected input changed during analysis')
    for path,digest in source['original_file_sha256'].items():
        if Path(path).exists():
            require(sha256(path)==digest,f'Original source changed: {path}')
    write_table(args.output/'horizon_statistics.csv',rows)
    write_table(args.output/'selected_runs.csv',[r['row'] for r in runs])
    np.savez_compressed(args.output/'plot_data.npz',
        omega=np.array(tuple(RUNS)), horizons=np.array(HORIZONS),
        **{key:np.array([[r[key] for r in rows if r['T']==T] for T in HORIZONS])
           for key in ('D_T','conditional_mean','mu_star')})
    report=dict(scope='Only Fig. 3; eight completed centered runs through omega300; refined300 case7706; omega400 excluded; screenshot-derived horizon palette.',
        definitions=dict(D_T='sum covered roof_probability/P0_init64',M_T='sum clipped interval midpoint times covered roof_probability/P0_init64',
                         conditional_mean='M_T/D_T',mu_star='Exact integral of native piecewise-linear stored mass/P0_init64 through T'),
        endpoint_handling='Partial event intervals use recorded constant native rates; only the terminal survivor mass is linearly interpolated. No extrapolation.',
        roundoff_endpoint_tolerance=ENDPOINT_ROUNDOFF_TOLERANCE,
        processing='No smoothing, binning, time shifts of trajectories, conditional curve normalization, or probability-budget corrections. Lines connect computed points.',
        horizons=HORIZONS,figure=figure,style_source=source,statistics=rows,
        runs=[dict(**r['row'],**r['details']) for r in runs],prior_Fig2_check=previous_check,
        max_abs_probability_budget_discrepancy=max(abs(r['D_minus_one_minus_S']) for r in rows),
        max_abs_mu_minus_event_identity=max(abs(r['mu_minus_event_identity']) for r in rows),
        max_abs_simpson_minus_plotted_mu=max(abs(r['simpson_minus_plotted_mu']) for r in rows),
        original_inputs_unchanged=True,script_sha256=sha256(__file__),
        numpy_version=np.__version__,scipy_version=scipy.__version__,matplotlib_version=matplotlib.__version__,
        limitations='Completion is not a convergence certificate. Different nominal dt are recorded; no numerical error bars are inferred.')
    (args.output/'audit_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=[
        'Fig. 3 centered reproduction - screenshot palette, refined omega300 - 2026-10-23',
        'One combined 1x3 figure, matching the manuscript: (a) D_T, (b) M_T/D_T, (c) mu*(T).',
        'Omega=1,25,50,75,100,150,200,300; refined omega300 case7706 dt=0.0001; omega400 excluded.',
        'Horizons T=10,20,30,40 are prefixes of the same eight completed trajectories.',
        '',
        'D and M use roof events normalized once by P0_init64.',
        'Each full event is included once; a partial terminal event is apportioned by covered duration.',
        'The first moment uses the midpoint of the covered interval.',
        'mu*(T) is the piecewise-linear native stored-mass integral/P0_init64, as in originalFig. 3.',
        'Only a terminal value inside a step is interpolated; no extrapolation or data completion.',
        'Raw endpoint, offsets and partial-event fractions are retained in horizon_statistics.csv.',
        '',
        'Same centered closure, endpoint mass weight 0.5, common physical preparation and actual plane verified.',
        'Finite/nonnegative detector data, positive durations, exact contiguous time/mass histories 0..40 verified.',
        'Grid, precision, nominal/final dt and complete run paths are in selected_runs.csv.',
        'Source snapshots and input hashes are verified; originals and previous outputs are preserved.',
        '',
        'Exact manuscript page 5 image matched to original horizon_comparison.png.',
        'Original 13x4 inch geometry, panel labels, fonts, markers, line widths and legend retained; horizon colors replaced.',
        'Screenshot horizon palette: T10 purple#1B0264, T20 blue#136384, T30 green#72CE6D, T40 yellow#E7E656.',
        'Colors sampled after screenshot ICC-to-sRGB conversion; approximate screenshot colors, not asserted original paper RGB values.',
        'Middle axis retains the wide overline on tau. All other visual settings preserved.',
        'Original automatic axis margins x=.01,y=.05 retained. New numerical axis limits are in JSON.',
        '',
        f"Maximum |D_T-(1-S_T)|={report['max_abs_probability_budget_discrepancy']:.12g}",
        f"Maximum |mu-[M+T*(1-D)]|={report['max_abs_mu_minus_event_identity']:.12g}",
        f"Maximum |native Simpson-plotted linear integral|={report['max_abs_simpson_minus_plotted_mu']:.12g}",
        'All discrepancies reported without correction.',
        f'At T40 the native Simpson cross-check against the prior Fig. 2 (source-aware): {previous_check}.',
        'plot_data.npz stores the exact 96 values shown in all three panels.',
        'Completion and quadrature agreement are not convergence certificates.',
        '',
        'Rerun: python plot_fig3_centered.py --run-root /path/to/1.Main --output /new/output',
        'Dependencies: numpy, scipy, matplotlib. Keep style_source_provenance.json beside the script.',
        'Vector PDF and 600-dpi PNG (7800x2400 pixels).']
    (args.output/'README.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),figure=figure,
                         prior_Fig2_check=previous_check,max_budget_error=report['max_abs_probability_budget_discrepancy']),indent=2))


if __name__=='__main__':
    main()

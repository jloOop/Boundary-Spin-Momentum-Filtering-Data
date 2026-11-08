#!/usr/bin/env python3
"""Reproduce Supplementary Fig. S1 from completed centered native mass histories.

Run: python plot_figS1_centered.py --run-root /path/to/1.Main --output /new/output
Requires numpy and matplotlib. Inputs are read-only; no solver is run.
Only Fig. S1 is exported: omega=1 left; completed omega=300 case7100 on the right.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'figS1_centered_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {1: '5540.Spin_Gauss_centeredABC_Omega=1_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3701390',
 300: '7100.Spin_Gauss_centeredABC_Omega=300_L=20_T=40_Nxy=100_Nz=1500_dt=0.0002_baseline_FG_v1_full_job3710751'}
PREPARATION = dict(Lz=20., z0_init=10., k0=np.pi, k_bc=np.pi,
                   sigma_z=.5, theta=0., phi=0.)
FILES = ('revision_run.json', 'branch_refinement_case.json', 'constants.npz',
         'revision_step_data.csv', 'source_snapshot.py', 'revision_spin_branch_diagnostics.npz')


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



BLUE, ORANGE, GREEN = '#34566F', '#B5654D', '#65866D'


def divide_defined(numerator, denominator):
    result=np.full(np.shape(numerator),np.nan,dtype=np.float64)
    np.divide(numerator,denominator,out=result,where=denominator>0)
    return result


def derive_spin_curves(run):
    path=run['directory']/'revision_spin_branch_diagnostics.npz'
    with np.load(path,allow_pickle=False) as saved:
        d={k:saved[k] for k in saved.files}
    required=('time','step','roof_density','branch_plus_density','branch_minus_density',
              'neutral_density','no_click_spin_matrix','no_click_mass64','roof_spin_matrix')
    require(all(k in d for k in required),'Required spin diagnostics missing')
    require(all(np.isfinite(d[k]).all() for k in required),'Nonfinite relevant diagnostic values')
    t=d['time'];n=len(t);step=d['step'];rho=d['roof_density'];a=run['steps']
    require(t.ndim==1 and t[0]==0 and t[-1]==40 and np.all(np.diff(t)>0),'Invalid diagnostic times')
    require(step.shape==t.shape and np.issubdtype(step.dtype,np.integer) and
            step[0]==0 and step[-1]==len(a) and np.all(np.diff(step)>0),'Invalid diagnostic step indices')
    require(np.array_equal(t,run['endpoint_t'][step]),'Diagnostic times do not match accepted-step endpoints')
    masses=np.r_[run['row']['P0_init64'],a['mass_stored']]
    require(np.array_equal(d['no_click_mass64'],masses[step]),'Diagnostic mass mismatch')
    sectors=np.stack([d[k] for k in ('branch_plus_density','branch_minus_density','neutral_density')],axis=1)
    q=d['no_click_spin_matrix'];qr=d['roof_spin_matrix']
    require(rho.shape==(n,) and sectors.shape==(n,3) and q.shape==(n,2,2) and qr.shape==(n,2,2),
            'Incorrect diagnostic shapes')
    require(np.all(rho>=0) and np.all(sectors>=0),'Negative branch density')
    fractions=divide_defined(sectors,rho[:,None])
    positive=rho>0
    require(np.any(positive),'No defined branch fractions')
    partition_error=float(np.max(np.abs(fractions[positive].sum(axis=1)-1)))
    require(partition_error<1e-12,'Branch fractions do not partition boundary density')
    require(np.nanmin(fractions)>=-1e-12 and np.nanmax(fractions)<=1+1e-12,'Branch fraction outside[0,1]')
    trace_q=np.trace(q,axis1=1,axis2=2).real
    require(np.all(trace_q>=0),'Negative survivor trace')
    survivor=divide_defined((q[:,0,0]-q[:,1,1]).real,trace_q)
    up,down=a['p_spin_up'],a['p_spin_down']
    require(np.all(up>=0) and np.all(down>=0),'Negative spin-resolved event probability')
    event_trace=up+down
    require(np.array_equal(event_trace,a['roof_probability']),'Event trace does not equal recorded detector probability')
    detector=divide_defined(up-down,event_trace)
    # These ratios are dimensionless. Common P0, kappa, area and duration factors
    # cancel. The survivor matrix already contains the centered endpoint metric.
    require(np.isfinite(detector[event_trace>0]).all() and np.isfinite(survivor[trace_q>0]).all(),
            'Nonfinite defined polarization')
    require(np.nanmax(np.abs(detector))<=1+1e-12 and np.nanmax(np.abs(survivor))<=1+1e-12,
            'Polarization outside[-1,1]')
    matrix_checks={}
    for name,matrix in [('survivor',q),('roof',qr)]:
        herm=float(np.max(np.abs(matrix-np.conjugate(np.swapaxes(matrix,1,2)))))
        eigen_min=float(np.linalg.eigvalsh(matrix).min())
        require(herm<1e-12 and eigen_min>=-1e-12,'Spin matrix not Hermitian/positive semidefinite')
        matrix_checks[name]=dict(max_hermitian_error=herm,min_eigenvalue=eigen_min)
    mass_error=float(np.max(np.abs(trace_q-d['no_click_mass64'])))
    require(mass_error<1e-12,'Survivor matrix trace inconsistent with stored mass')
    data=dict(diagnostic_time=t,plus_fraction=fractions[:,0],minus_fraction=fractions[:,1],
              neutral_fraction=fractions[:,2],undetected_Pz=survivor,
              detector_time=run['t'],detector_Pz=detector,
              detector_interval_left=a['t_left'],detector_interval_right=a['t_right'])
    record=dict(omega=run['omega'],diagnostic_samples=n,accepted_step_samples=len(a),
                diagnostic_time_range=t[[0,-1]].tolist(),detector_midpoint_range=run['t'][[0,-1]].tolist(),
                diagnostic_dt_min=float(np.diff(t).min()),diagnostic_dt_max=float(np.diff(t).max()),
                undefined_branch_samples=int(np.count_nonzero(~positive)),
                undefined_detector_samples=int(np.count_nonzero(event_trace<=0)),
                undefined_undetected_samples=int(np.count_nonzero(trace_q<=0)),
                first_defined_branch_time=float(t[positive][0]),
                first_defined_detector_time=float(run['t'][event_trace>0][0]),
                max_branch_partition_error=partition_error,
                max_raw_branch_partition_error=float(np.max(np.abs(sectors.sum(axis=1)-rho))),
                detector_min=float(np.nanmin(detector)),detector_max=float(np.nanmax(detector)),
                survivor_min=float(np.nanmin(survivor)),survivor_max=float(np.nanmax(survivor)),
                event_trace_vs_probability_max_abs=0.,survivor_trace_vs_mass_max_abs=mass_error,
                matrix_checks=matrix_checks,diagnostic_times_and_masses_match_step_history=True,
                array_sha256={k:array_sha(v) for k,v in data.items()})
    return data,record


def style_axis(ax):
    ax.set_facecolor('white')
    for spine in ax.spines.values():spine.set_linewidth(.4)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='both',width=.9,direction='out',top=False,right=False)
    ax.tick_params(which='major',length=6)
    ax.tick_params(which='minor',length=2.5)
    ax.grid(True,alpha=.0001)
    ax.set_xlim(0,40)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel(r'$t$')


def checked_line(ax,t,y,**kwargs):
    line,=ax.plot(t,y,**kwargs)
    require(np.array_equal(line.get_xdata(),t,equal_nan=True) and
            np.array_equal(line.get_ydata(),y,equal_nan=True),'Plotted values changed')
    require(len(line.get_path().vertices)==len(t) and not line.get_path().should_simplify,
            'Plotted vertices simplified or removed')
    return line


def legend(ax,columns):
    return ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncol=columns,
                     frameon=True,facecolor='white',edgecolor='0.8',framealpha=1.,
                     fontsize=10.5,columnspacing=1.,handlelength=2.1,
                     handletextpad=.6,labelspacing=.3,borderpad=.35,borderaxespad=0)


def render(curves,output):
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,
        'axes.titlesize':14,'mathtext.fontset':'dejavusans','figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white','savefig.transparent':False,
        'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42,'path.simplify':False,
        'agg.path.chunksize':0,'svg.hashsalt':'PC2-native-four-panels-v1'})
    fig,axes=plt.subplots(2,2,figsize=(12.4,8.1))
    fig.subplots_adjust(left=.075,right=.985,bottom=.085,top=.86,wspace=.23,hspace=.66)
    labels=[]
    for col,(omega,data) in enumerate(curves.items()):
        upper,lower=axes[:,col];t=data['diagnostic_time']
        for key,name,color,linestyle,width in [
            ('plus_fraction',r'$\Pi_+$ branch',BLUE,'-',1.45),
            ('minus_fraction',r'$\Pi_-$ branch',ORANGE,'--',1.35),
            ('neutral_fraction','Neutral',GREEN,':',1.35)]:
            checked_line(upper,t,data[key],label=name,color=color,ls=linestyle,lw=width)
        checked_line(lower,data['detector_time'],data['detector_Pz'],
                     label=r'Detector $P_z^{\mathrm{det}}$',color=BLUE,lw=1.35)
        checked_line(lower,t,data['undetected_Pz'],label=r'Undetected $P_z^{\mathrm{nc}}$',
                     color=ORANGE,ls='--',lw=1.45)
        suffix=rf', $\omega = {omega}$'
        upper.set_ylim(-.04,1.06);upper.yaxis.set_major_locator(MultipleLocator(.2))
        upper.set_ylabel(r'$w_{\alpha}^{\mathrm{br}}(t)$')
        labels.append(upper.set_title(f'({chr(97+col)}) Branch fractions'+suffix,loc='left',y=1.235,pad=3))
        labels.append(legend(upper,3))
        lower.set_ylim(-1.06,1.075);lower.yaxis.set_major_locator(MultipleLocator(.5))
        lower.set_ylabel(r'$P_z(t)$')
        labels.append(lower.set_title(rf'({chr(99+col)}) $z$-spin polarization'+suffix,loc='left',y=1.235,pad=3))
        labels.append(legend(lower,2))
        style_axis(upper);style_axis(lower)
    require([len(ax.lines) for ax in axes.flat]==[3,3,2,2],'Unexpected panel curve counts')
    require(len(fig.axes)==4 and all(not ax.child_axes for ax in axes.flat),'Unexpected inset axes')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for i in range(0,len(labels),2):
        require(not labels[i].get_window_extent(renderer).overlaps(labels[i+1].get_window_extent(renderer)),
                'Panel title overlaps legend')
    for ax in axes.flat:
        ymin,ymax=ax.get_ylim()
        for line in ax.lines:
            require(np.nanmin(line.get_ydata())>ymin and np.nanmax(line.get_ydata())<ymax,
                    'Vertical limits clip defined data')
    for ext in ('pdf','png'):
        meta={'Creator':'Centered native S1; plot_figS1_centered.py','CreationDate':None,'ModDate':None} if ext=='pdf' else None
        fig.savefig(output/f'FigS1_centered.{ext}',dpi=600,metadata=meta)
    record=dict(figsize_inches=[12.4,8.1],png_dpi=600,panel_curve_counts=[3,3,2,2],
                left_omega=1,right_omega=300,xlim=[0,40],branch_ylim=[-.04,1.06],spin_ylim=[-1.06,1.075],
                original_limits_unchanged=True,palette=[BLUE,ORANGE,GREEN],
                no_insets=True,title_legend_overlap_check='PASS',all_native_vertices_retained=True,
                undefined_zero_denominator_ratios_retained_as_nan=True,path_simplification=False)
    plt.close(fig)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=HERE.parent)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in ('FigS1_centered.pdf','FigS1_centered.png','README.txt','selected_runs.csv',
                 'audit_provenance.json','plot_data.npz'):
        require(not (args.output/name).exists(), f'Refusing to overwrite existing output: {args.output/name}')
    require(tuple(RUNS)==(1,300), 'Unexpected run set')
    runs=[load_run(args.run_root,w,path) for w,path in RUNS.items()]
    curves={};records=[];arrays={}
    for run in runs:
        same(run['row']['detector_z'],19.986665725708008,'actual detector plane')
        data,record=derive_spin_curves(run)
        curves[run['omega']]=data;records.append(record)
        arrays.update({f"omega{run['omega']}__{key}":value for key,value in data.items()})
    source=json.loads((HERE/'style_source_provenance.json').read_text())
    figure=render(curves,args.output)
    np.savez_compressed(args.output/'plot_data.npz',**arrays)
    with np.load(args.output/'plot_data.npz',allow_pickle=False) as z:
        require(set(z.files)==set(arrays),'Plot archive fields changed')
        for key,value in arrays.items():
            require(z[key].dtype==value.dtype and z[key].shape==value.shape and z[key].tobytes()==value.tobytes(),
                    'Plot arrays failed byte-preserving roundtrip')
    for run in runs:
        require(all(sha256(run['directory']/name)==digest for name,digest in run['details']['input_sha256'].items()),
                'Selected input changed during rendering')
    for path,digest in source['original_file_sha256'].items():
        if Path(path).exists():require(sha256(path)==digest,f'Original source changed: {path}')
    table=[]
    for run,record in zip(runs,records):
        row=dict(run['row'])
        row.update({k:v for k,v in record.items() if not isinstance(v,(list,dict))})
        table.append(row)
    with (args.output/'selected_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    report=dict(scope='Only Supplementary Fig.S1: completed centered omega1 left; omega300 case7100 right; omega400 excluded.',
        definitions=dict(branch_fractions='branch sector densities / roof_density at native diagnostic times',
                         detector_Pz='(p_spin_up-p_spin_down)/(p_spin_up+p_spin_down) at actual accepted-step midpoints',
                         survivor_Pz='real(Q00-Q11)/real(trace(Q)) from no_click_spin_matrix at diagnostic times'),
        denominator_rule='Strictly positive only; zero denominators remain NaN gaps. No additional signal threshold.',
        normalization='Dimensionless ratios; common P0,kappa,area,duration factors cancel. Centered endpoint weight already included in survivor matrix.',
        processing='No smoothing, clipping of values, resampling, binning, downsampling, endpoint padding or field reconstruction.',
        branch_definition_note='Recorded Fourier diagnostics use centered discrete symbol sin(kh)/h and recorded neutral tolerance; not a finite-wall eigenbasis reconstruction.',
        figure=figure,style_source=source,diagnostic_audit=records,
        runs=[dict(**r['row'],**r['details']) for r in runs],
        input_files_unchanged=True,plot_data_byte_roundtrip=True,script_sha256=sha256(__file__),
        numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        limitations='Audit validates recorded native diagnostics and source definitions. Run completion is not convergence certification.')
    (args.output/'audit_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=[
        'Supplementary Fig. S1 centered reproduction - 2026-10-18',
        'Left: omega=1. Right: omega=300 using completed case7100. Only S1 generated.',
        'Panels(a,b): plus,minus,neutral boundary fractions at native diagnostic times.',
        'Panels(c,d): detector event Pz at accepted-step midpoints; survivor Pz at diagnostic times.',
        'Detector Pz=(p_spin_up-p_spin_down)/(p_spin_up+p_spin_down). Event trace equals roof_probability exactly.',
        'Survivor Pz=real(Q00-Q11)/real(trace(Q)) from no_click_spin_matrix.',
        'Branch fractions=recorded sector density/roof_density.',
        'Common normalization factors cancel in these ratios; no extra P0,kappa,area,dt or endpoint-weight factors.',
        '',
        'Zero-denominator ratios remain NaN gaps. No additional low-signal threshold.',
        'Every native plotted vertex retained. No smoothing, binning, downsampling, resampling or padding.',
        'All relevant recorded arrays finite, branches nonnegative and partitioned, polarizations within[-1,1].',
        'Diagnostic times match accepted endpoints; survivor masses match recorded CSV masses.',
        'Unrelated one_grid_E2br undefined values are not used in S1.',
        '',
        'Manuscript page34 image exactly matches preserved branches.png.',
        'The colocated original script was later changed to overlay extra confinements; its preserved helpers',
        'and original two-run arrangement were reconstructed and independently verified pixel-exact.',
        'Fonts, colors, widths, axis limits, ticks, panel labels, legends and geometry are retained.',
        'The original omega1/omega300 column titles are retained.',
        'Original files and previous centered exports remain unchanged.',
        '',
        *[f"omega={r['omega']}: {r['diagnostic_samples']} diagnostic samples; {r['accepted_step_samples']} events; undefined branch={r['undefined_branch_samples']}, detector={r['undefined_detector_samples']}, survivor={r['undefined_undetected_samples']}; max branch partition error={r['max_branch_partition_error']:.3g}." for r in records],
        '',
        'Selected run paths, dt, grid, precision and audit statistics: selected_runs.csv.',
        'Full plotted arrays, including undefined NaNs: plot_data.npz.',
        'Detailed source hashes and definitions: audit_provenance.json.',
        'Completion is not a convergence certificate.',
        '',
        'Rerun: python plot_figS1_centered.py --run-root /path/to/1.Main --output /new/output',
        'Dependencies: numpy and matplotlib. Keep style_source_provenance.json alongside script.',
        'Vector PDF and 600-dpi PNG.']
    (args.output/'README.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),figure=figure,diagnostic_audit=records),indent=2))


if __name__=='__main__':main()

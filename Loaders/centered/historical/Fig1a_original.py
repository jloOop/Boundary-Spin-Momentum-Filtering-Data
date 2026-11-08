#!/usr/bin/env python3
"""Reproduce Fig. 1(a,b) from raw centered step records using Paul Tol muted with omega=300 black.

Run with Python 3 + NumPy + Matplotlib:
    python plot_fig1_centered.py
    python plot_fig1_centered.py --run-root /path/to/1.Main --output /new/output

Panel a includes omega=1,50,100,150,200,300; panel b=1,50,100,300. Input runs and
original figures are read-only. The two panels share the exact same g arrays.
The reference directory contains ONLY the unchanged existing Gaussian reference
and provenance, not any original detector-present data.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'fig1_centered_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {
    1: '5540.Spin_Gauss_centeredABC_Omega=1_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3701390',
    50: '5047.Spin_Gauss_centeredABC_Omega=50_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3715452',
    100: '5541.Spin_Gauss_centeredABC_Omega=100_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3701391',
    150: '5044.Spin_Gauss_centeredABC_Omega=150_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704246',
    200: '5045.Spin_Gauss_centeredABC_Omega=200_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704247',
    300: '7706.Spin_Gauss_centeredABC_Omega=300_L=20_zc=10_T=40_Nxy=100_Nz=1500_dt=0.0001_baseline_half_dt_v1_full_job3724723',
}
# Palette B: swap omega100/150 colors; highest omega300 remains black.
# Annotated two-panel version restricted through omega300; omega400 omitted.
COLORS = {1: '#CC6677', 50: '#332288', 100: '#999933', 150: '#44AA99', 200: '#AA4499', 300: '#000000'}
FULL_WIDTHS = {1: 1.8, 50: 1.6, 100: 1.45, 150: 1.1, 200: .75, 300: .6}
LATE_WIDTHS = {1: 1.8, 50: 1.6, 100: 1.45, 300: .9}
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
    details = dict(input_sha256=hashes, configuration=config,
                   source_snapshot_matches_recorded_hash=True,
                   time_sha256=array_sha(t), density_sha256=array_sha(g),
                   full_peak_time=float(t[np.argmax(g)]),
                   late_peak_time=float(t[late][np.argmax(g[late])]),
                   all_csv_values_finite=True, probabilities_nonnegative=True,
                   durations_positive=True, contiguous_nonduplicated_history=True)
    return dict(omega=omega, directory=directory, t=t, g=g, row=row, details=details)


def free_gaussian_current(t, c):
    # Copied exactly from replot_deep_blue_300.py lines 92-98; validation only.
    sigma, k0 = c['sigma_z'], c['k0']
    xi = c['z_roof_actual'] - c['z0'] - k0 * t
    spread = 1 + (t / sigma**2)**2
    density = np.exp(-xi**2 / (sigma**2 * spread)) / (np.sqrt(np.pi) * sigma * np.sqrt(spread))
    return density * (k0 + t * xi / (sigma**4 + t**2))


def draw_panel(output, panel, runs, reference_t, reference_g):
    selected = tuple(RUNS) if panel == 'a' else (1, 50, 100, 300)
    widths = FULL_WIDTHS if panel == 'a' else LATE_WIDTHS
    fig = plt.figure(figsize=(8.8, 3.75))
    axes_height_inches = 7.5 * (0.952 - 0.085) / (2 + .39)
    ax = fig.add_axes([.115, .17, .863, axes_height_inches / 3.75])
    ax.set_xlabel(r'$t$', labelpad=4)
    ax.set_ylabel('Probability density', labelpad=9)
    ax.set_title(f'({panel})', loc='left', x=.025, fontsize=12, pad=10)
    ax.tick_params(which='both', direction='out', top=False, right=False, width=.7)
    ax.tick_params(which='major', length=4.2)
    ax.tick_params(which='minor', length=2.2)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    vertices = []
    for r in runs:
        w = r['omega']
        if w in selected:
            # Pass the SAME full native arrays to both panels. xlim alone zooms.
            line, = ax.plot(r['t'], r['g'], color=COLORS[w], lw=widths[w], label=rf'$\omega={w}$')
            require(np.array_equal(line.get_xdata(), r['t']) and
                    np.array_equal(line.get_ydata(), r['g']), 'Plotted vertices changed')
            require(line.get_color() == COLORS[w] and line.get_linewidth() == widths[w],
                    'Curve style changed')
            vertices.append(dict(omega=w, native_vertices=len(r['t']),
                                 density_sha256=array_sha(line.get_ydata())))
    if panel == 'a':
        ax.plot(reference_t, reference_g, color='#505050', lw=1.3,
                linestyle=(0, (5, 3)), label='Free Gaussian current')
        original_ylim = (-.0025, .405)
        ax.set(xlim=(0, 40), ylim=original_ylim)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(.10))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend(loc='upper right', ncol=3, frameon=False, fontsize=9.5,
                  handlelength=2.6, columnspacing=1.35, labelspacing=.6, borderaxespad=.7)
        peak = max(float(reference_g.max()), *(r['row']['full_peak'] for r in runs))
    else:
        original_ylim = (-.0002, .0355)
        ax.set(xlim=(8, 40), ylim=original_ylim)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(.005))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(loc='lower right', bbox_to_anchor=(1., 1.016), ncol=4,
                  frameon=False, fontsize=10, handlelength=2.6,
                  columnspacing=1.6, borderaxespad=0)
        peak = max(r['row']['late_peak'] for r in runs if r['omega'] in selected)
    if peak >= original_ylim[1]:
        ax.set_ylim(original_ylim[0], peak * 1.05)
    require(peak < ax.get_ylim()[1], 'Peak would be clipped')
    require(len(vertices) == len(selected), 'Incorrect panel series')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bounds = ax.get_legend().get_window_extent(renderer)
    title_bounds = ax._left_title.get_window_extent(renderer)
    require(not legend_bounds.overlaps(title_bounds), 'Legend overlaps panel label')
    for bounds in (legend_bounds, title_bounds):
        require(bounds.x0 >= 0 and bounds.y0 >= 0 and
                bounds.x1 <= fig.bbox.x1 and bounds.y1 <= fig.bbox.y1,
                'Legend or label would be clipped')
    stem = f'Fig1{panel}_centered'
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'{stem}.{ext}', dpi=600)
    record = dict(stem=stem, omegas=selected, xlim=[float(x) for x in ax.get_xlim()],
                  original_ylim=list(original_ylim), ylim=[float(y) for y in ax.get_ylim()],
                  limit_changed=list(ax.get_ylim()) != list(original_ylim),
                  maximum_visible_density=peak, vertices=vertices,
                  size_inches=[8.8, 3.75], png_dpi=600,
                  vector_pdf=True, axes_bounds=list(ax.get_position().bounds),
                  legend_columns=3 if panel == 'a' else 4,
                  colors={w: COLORS[w] for w in selected},
                  linewidths_pt={w: widths[w] for w in selected})
    plt.close(fig)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=HERE.parent)
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('Fig1a_centered.pdf', 'Fig1a_centered.png', 'Fig1b_centered.pdf',
                 'Fig1b_centered.png', 'selected_runs.csv', 'audit_provenance.json',
                 'plot_data.npz', 'README.txt'):
        require(not (args.output/name).exists(), f'Refusing to overwrite existing output: {args.output/name}')
    runs = [load_run(args.run_root, w, folder) for w, folder in RUNS.items()]
    source = json.loads((HERE / 'reference/source_provenance.json').read_text())
    reference_file = HERE / 'reference/free_gaussian_existing.npz'
    require(sha256(reference_file) == source['reference_copy_sha256'], 'Stored reference changed')
    with np.load(reference_file, allow_pickle=False) as z:
        reference_t, reference_g = z['free_gaussian_t'], z['free_gaussian_g']
    c = source['reference_constants']
    for key, expected in dict(Lz=20., z0=10., sigma_z=.5, k0=np.pi, k_bc=np.pi,
                              theta=0., phi=0.).items():
        same(c[key], expected, f'Gaussian reference {key}')
    for r in runs:
        same(r['row']['detector_z'], c['z_roof_actual'], 'reference evaluation plane')
    require(np.isfinite(reference_t).all() and np.isfinite(reference_g).all(), 'Nonfinite reference')
    require(np.array_equal(reference_t, np.linspace(0, 40, 4001)), 'Reference time grid changed')
    # Tiny cross-platform transcendental roundoff is allowed in validation only;
    # plotting always retains stored reference values unchanged.
    require(np.allclose(reference_g, free_gaussian_current(reference_t, c), rtol=2e-14, atol=1e-16),
            'Existing Gaussian reference does not match its preparation/plane')
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.labelsize': 12, 'legend.fontsize': 10, 'axes.linewidth': .65,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'path.simplify': False})
    figures = [draw_panel(args.output, panel, runs, reference_t, reference_g) for panel in ('a', 'b')]
    saved = {f'omega{r["omega"]}_{k}': r[k] for r in runs for k in ('t', 'g')}
    saved.update(free_gaussian_t=reference_t, free_gaussian_g=reference_g)
    np.savez_compressed(args.output/'plot_data.npz', **saved)
    with np.load(args.output/'plot_data.npz', allow_pickle=False) as z:
        require(all(np.array_equal(z[k], a) for k, a in saved.items()), 'Saved arrays changed')
    for r in runs:
        require(all(sha256(r['directory']/name) == digest
                    for name, digest in r['details']['input_sha256'].items()), 'Input changed during rendering')
    with (args.output/'selected_runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0]['row']))
        writer.writeheader()
        writer.writerows(r['row'] for r in runs)
    report = dict(scope='Only Fig. 1(a,b): a=1,50,100,150,200,300; b=1,50,100,300. Omega400 excluded to match the attached manuscript; accepted Paul Tol muted palette B with omega300 black.',
        detector_method='P0=P0_init64; dt=t_right-t_left; t=(t_left+t_right)/2; g=roof_probability/(P0*dt)',
        processing='Native step-midpoint vertices; no binning, smoothing, time shift, rescaling, differentiation, or conditional normalization.',
        D40='math.fsum(roof_probability)/P0_init64',
        S40='mass_stored_final/P0_init64',
        discrepancy='D40-(1-S40); retained and reported without correction',
        convergence='Run completion is not convergence certification; the selected runs have different dt.',
        run_replacement='Omega300 uses case7706 dt=0.0001 instead of case7100 dt=0.0002; all other selected runs and visual settings unchanged.',
        added_style='Annotated two-panel version through omega300. Omega100 is olive #999933 and omega150 teal #44AA99 (colors exchanged); omega300 black. Common omega colors match across panels. Panel a=1,50,100,150,200,300; b=1,50,100,300. Original fonts, line widths, axes, ticks, labels and geometry retained; b uses four legend columns.',
        visual_source=source, figures=figures,
        runs=[dict(**r['row'], **r['details']) for r in runs],
        numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
        script_sha256=sha256(__file__), input_files_unchanged_after_rendering=True)
    (args.output/'audit_provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = [
        'Fig. 1 centered reproduction - omega100/150 colors swapped, through300 - 2026-10-23',
        'Only panels (a) and (b). Refined omega=300 case7706 used in both panels. Omega=400 excluded.',
        'Panel (a): omega=1,50,100,150,200,300; panel (b): omega=1,50,100,300.',
        'Annotated two-panel selection with omega400 removed: a=1,50,100,150,200,300; b=1,50,100,300.',
        'Colors: 1=#CC6677, 50=#332288, 100=#999933, 150=#44AA99, 200=#AA4499, 300=#000000.',
        'Palette source: https://sronpersonalpages.nl/~pault/ (muted, Figure 4).',
        'Both panels use identical native unconditioned midpoint curves.',
        'P0=P0_init64; g=roof_probability/[P0*(t_right-t_left)].',
        'No extra kappa, area, step-duration, or endpoint-weight factor is applied.',
        'D40 is the event sum/P0, not trapezoidal integration of connected midpoints.',
        'S40=mass_stored_final/P0. Delta=D40-(1-S40). No budget correction.',
        '',
        'Verified common preparation: L=20, z0=10, k0=kappa=pi, amplitude width=0.5, theta=phi=0.',
        'All closures centered; endpoint mass weight=0.5; grid=100x100x1500.',
        'Actual detector and existing Gaussian evaluation plane: z=19.986665725708008.',
        'State/preparation complex64; coordinates/potential float32; matrix complex64; solve complex128.',
        'All numeric CSV columns finite, detector probabilities nonnegative, durations positive,',
        'consecutive steps and exactly contiguous nonduplicated time history 0..40.',
        'Selected paths, dt, spacings, precision, D40, S40 and signed errors: selected_runs.csv.',
        '',
        'Original style identified by exact decoded RGB match to manuscript page-4 embedded images.',
        'Original Gaussian arrays retained unchanged. Its original dark-gray #505050 dashed',
        'style is preserved (the manuscript caption calls it black). Preparation and plane match.',
        'Palette B with omega100/150 colors swapped; original fonts, widths, axes/ticks, labels and geometry reused.',
        'Omega=300 is black #000000 in both panels. No omega=400 curve plotted.',
        'Original widths retained. Omega=300 uses 0.6 pt in (a) and 0.9 pt in (b).',
        'Panel (b) uses the original four-column legend; font, spacing and anchor unchanged.',
        'Omega=300: nominal dt=0.0001, 400000 accepted steps through T=40; replaces case7100 dt=0.0002.',
        'All non-color styles unchanged. Original scripts, figures and previous exports preserved.',
        'The actual final duration is used for g and recorded in selected_runs.csv.',
    ]
    for f in figures:
        lines.append(f"{f['stem']}: xlim={f['xlim']}, ylim={f['ylim']}; limit change={f['limit_changed']}; max={f['maximum_visible_density']:.12g}.")
    lines += ['', 'omega       D40               S40               Delta',
              *[f"{r['omega']:>5}  {r['row']['D40']:.15f}  {r['row']['S40']:.15f}  {r['row']['D40_minus_one_minus_S40']:+.9e}" for r in runs],
              '', 'Completion is not a convergence certificate. Different native dt are recorded.',
              'Rerun: python plot_fig1_centered.py --run-root /path/to/1.Main --output /path/to/output',
              'Dependencies: numpy, matplotlib. Keep the reference/ folder alongside the script.',
              'plot_data.npz stores all native midpoint curves and the unchanged Gaussian comparison.',
              'PDF: vector paths/text. PNG: 5280x2250 pixels, 600 dpi. No path simplification.']
    (args.output/'README.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[-13:]))


if __name__ == '__main__':
    main()

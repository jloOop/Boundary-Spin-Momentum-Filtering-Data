#!/usr/bin/env python3
"""Recalculate and plot Fig. S10 using seven confinements per length at T=40.

2026-10-21 update: omega=75 is now included for BOTH lengths. Uses validated
accepted-step stored masses normalized by revision_run.json P0_init64.
The half-weight endpoint metric is already included in those masses.
No smoothing, clipping, time shift, or individual curve renormalization.

Usage: python scripts/reproduce_s10.py --raw-root /path/to/Output
Default input is the parent Output directory. The L20 omega75 run is loaded
from that directory if present, otherwise sibling 1.Main case5043.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, platform
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/s9_s10_matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
OMEGAS=(1,25,50,75,100,150,200)
T=40.0
REQUIRED=("revision_run.json","constants.npz","branch_refinement_case.json",
          "source_snapshot.py","revision_coordinates.npz","revision_step_data.csv")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def same(a, b, name):
    if isinstance(b, str):
        assert a == b, (name, a, b)
    else:
        assert np.isclose(a, b, rtol=0, atol=1e-12), (name, a, b)

def read_run(folder, expected_L, expected_omega):
    for f in REQUIRED:
        assert (folder / f).is_file(), f"Missing input: {folder/f}"
    hashes = {f: sha256(folder / f) for f in REQUIRED}
    meta = json.loads((folder / "revision_run.json").read_text())
    branch = json.loads((folder / "branch_refinement_case.json").read_text())
    cfg = meta["configuration"]
    with np.load(folder / "constants.npz", allow_pickle=False) as npz:
        constants = {k: npz[k].item() for k in npz.files}
    source = (folder / "source_snapshot.py").read_text()
    tree = ast.parse(source)
    embedded = [ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "BRANCH_REFINEMENT_CASE"
                        for t in n.targets)]
    assert len(embedded) == 1 and embedded[0] == branch
    assert hashes["source_snapshot.py"] == meta["updated_source_sha256"]
    expected = dict(Lz=expected_L, omega=expected_omega, T_final=T,
                    Nx=100, Ny=100, Nz=1500 if expected_L == 20 else 2000,
                    z0_init=expected_L/2, sigma_z=.5, k0=np.pi, k_bc=np.pi,
                    theta=0., phi=0., endpoint_closure="centered",
                    endpoint_mass_weight=.5, metric="half_weight_last_z_node",
                    state_dtype="complex64", state_representation="physical_nodal_spinor")
    for key, val in expected.items():
        same(cfg[key], val, f"configuration.{key}")
        ckey = "z0" if key == "z0_init" else key
        if ckey in constants:
            same(constants[ckey], val, f"constants.{ckey}")
        if key in branch:
            bval = np.pi if branch[key] == "pi" else branch[key]
            same(bval, val, f"branch.{key}")
    for key in ("hx", "hy", "hz", "dt", "z_roof_actual", "grid_generator_nz"):
        same(cfg[key], constants[key], key)
    same(cfg["matrix_assembly"], "complex64_then_solve_complex128", "matrix")
    assert cfg["initial_preparation_dtype"] == cfg["state_dtype"] == "complex64"
    assert cfg["potential_sampling_dtype"] == cfg["real_dtype"] == "float32"
    same(cfg["Lx"], 10/np.sqrt(expected_omega), "Lx")
    same(cfg["Ly"], 10/np.sqrt(expected_omega), "Ly")
    assert "COMPLETED" in meta["status"]
    same(meta["actual_time"], T, "actual_time")
    with np.load(folder / "revision_coordinates.npz", allow_pickle=False) as npz:
        coords = {k: npz[k] for k in ("x", "y", "z")}
    coord_info = {}
    for axis in ("x", "y", "z"):
        a = coords[axis]
        assert a.dtype == np.float32 and np.all(np.diff(a) > 0)
        assert len(a) == cfg["N"+axis]
        assert a[0] == 0 and a[-1] < cfg["L"+axis]
        same(cfg["h"+axis], float(a[1]-a[0]), "stencil_"+axis)
        coord_info[axis] = dict(nodes=len(a), dtype=str(a.dtype),
            first=float(a[0]), last=float(a[-1]),
            min_coordinate_spacing=float(np.min(np.diff(a))),
            max_coordinate_spacing=float(np.max(np.diff(a))),
            stencil_spacing=cfg["h"+axis],
            array_sha256=hashlib.sha256(a.tobytes()).hexdigest())
    same(float(coords["z"][-1]), cfg["z_roof_actual"], "roof_plane")
    # Preserve all digits from CSV; pandas default parser can lose an ulp.
    df = pd.read_csv(folder / "revision_step_data.csv", float_precision="round_trip")
    assert len(df) == meta["completed_step"]
    assert np.isfinite(df.to_numpy()).all()
    assert np.array_equal(df.step.to_numpy(), np.arange(1, len(df)+1))
    left, right = df.t_left.to_numpy(), df.t_right.to_numpy()
    dt = right-left
    assert left[0] == 0 and right[-1] == T and np.all(dt > 0)
    assert np.array_equal(left[1:], right[:-1])
    same(np.max(dt), cfg["dt"], "max_step")
    assert np.all(np.abs(dt[:-1]-cfg["dt"]) < 1e-12)
    assert dt[-1] <= cfg["dt"]+1e-12
    P0 = float(meta["P0_init64"])
    before, stored = df.mass_before.to_numpy(), df.mass_stored.to_numpy()
    assert before[0] == P0 and np.array_equal(before[1:], stored[:-1])
    assert stored[-1] == meta["current_accepted_prefix_mass64"]
    assert np.all(stored >= 0)
    raw_q = df.roof_probability.to_numpy()
    assert np.all(raw_q >= 0)
    q = raw_q/P0
    mid = (left+right)/2
    times = np.r_[0., right]
    survival = np.r_[P0, stored]/P0
    assert survival[0] == 1
    # Independent direct trapezoidal integral of every-step stored survival.
    mu = float(np.trapezoid(survival, times))
    D = math.fsum(q)
    moment = math.fsum(mid*q)
    tau = moment/D
    S = float(survival[-1])
    rhs = moment+T*S
    closure = 1-S-D
    defect = (df.solve_and_matrix_defect.to_numpy()
              + df.cast_mass_loss.to_numpy() + df.clamp_mass_loss.to_numpy())/P0
    closure_trace = 1-survival[1:]-np.cumsum(q)
    total_defect = math.fsum(defect)
    weighted_defect = math.fsum(mid*defect)
    assert abs(math.fsum(raw_q)-meta["cumulative_roof_probability"]) < 2e-14
    assert abs(closure-total_defect) < 2e-13
    assert abs(mu-rhs-weighted_defect) < 2e-12
    assert np.max(np.abs(closure_trace)) < 1e-6
    assert abs(mu-rhs) < 1e-5
    # The metadata's larger fail-stop guard is not an accuracy tolerance.
    assert df.relative_residual.max() < 2e-8
    summary = dict(case_id=int(branch["case_id"]), L=expected_L, omega=expected_omega,
        P0_init64=P0, N_steps=len(df), dt_config=cfg["dt"],
        dt_min=float(dt.min()), dt_max=float(dt.max()), dt_last=float(dt[-1]),
        t_start=float(left[0]), t_end=float(right[-1]), zc=cfg["z0_init"],
        z_detector=cfg["z_roof_actual"], Nz=cfg["Nz"], hz=cfg["hz"],
        D40=D, S40=S, tau_cond40=tau, mu40=mu, event_first_moment40=moment,
        restricted_mean_identity_rhs=rhs, restricted_mean_identity_residual=mu-rhs,
        closure_1_minus_S_minus_D=closure,
        max_abs_closure=float(np.max(np.abs(closure_trace))),
        numerical_defect_sum=total_defect, time_weighted_defect_sum=weighted_defect,
        identity_after_defect_residual=mu-rhs-weighted_defect,
        closure_after_defect_residual=closure-total_defect,
        max_solver_relative_residual=float(df.relative_residual.max()),
        max_survival_above_one=float(max(0., survival.max()-1)),
        max_survival_increase=float(max(0., np.diff(survival).max())),
        max_abs_clamp_loss=float(np.abs(df.clamp_mass_loss).max()),
        min_roof_probability=float(raw_q.min()),
        max_time_gap=float(np.max(np.abs(left[1:]-right[:-1]))),
        max_mass_gap=float(np.max(np.abs(before[1:]-stored[:-1]))),
        source_path=str(folder.resolve()), status=meta["status"])
    provenance = dict(case_id=summary["case_id"], path=str(folder.resolve()),
        input_sha256=hashes, configuration=cfg, constants=constants,
        branch_refinement_case=branch, grid=coord_info,
        source_hash_verified=True, source_embedded_case_matches_json=True,
        checks_passed=True, statistics=summary)
    return dict(summary=summary, provenance=provenance, df=df, t=mid,
                g=q/dt, q=q, left=left, right=right)

def fit(rows):
    x = np.sqrt([r["omega"] for r in rows])
    y = np.array([r["mu40"] for r in rows])
    X = np.column_stack((np.ones(len(x)), x))
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y-X@np.array([a,b])
    return dict(L=rows[0]["L"], n=len(x), omegas=[r["omega"] for r in rows],
                A40=float(a), B40=float(b),
                RMSE=float(np.sqrt(np.mean(residual**2))),
                R_squared=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
                maximum_absolute_residual=float(np.max(np.abs(residual))),
                residuals=residual.tolist(),
                method="separate unweighted least squares; RMSE=sqrt(SSE/n)")

def style_axis(ax, inset=False, s10=False):
    for spine in ax.spines.values():
        spine.set_linewidth(.4)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", width=.9,
        direction="in" if s10 else "out", top=s10, right=s10)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(True, alpha=.1 if s10 else .0001)
    if inset:
        ax.tick_params(labelsize=8, which="major", length=3.5, width=.6)
        ax.tick_params(which="minor", length=1.8, width=.6)

def save_figure(fig, out, stem):
    fig.savefig(out / "figures" / (stem+".pdf"))
    fig.savefig(out / "figures" / (stem+".png"), dpi=600)
    # Small local preview; the separately delivered PNG is always 600 dpi.
    fig.savefig(out / "qa" / (stem+"_preview.png"), dpi=160)
    plt.close(fig)

def plot_s10(rows, fits, out):
    fig,ax=plt.subplots(figsize=(8.6,5.2))
    style_axis(ax,s10=True)
    handles,labels,fit_ys=[],[],[]
    for L,marker,ls in ((20,"o","-"),(30,"s","--")):
        series=[r for r in rows if r["L"] == L]
        f=fits[L]
        w=np.array([r["omega"] for r in series])
        y=np.array([r["mu40"] for r in series])
        wg=np.linspace(w.min(),w.max(),500)
        yg=f["A40"]+f["B40"]*np.sqrt(wg)
        hfit,=ax.plot(wg,yg,color="#050505",lw=.9,linestyle=ls,zorder=2)
        hdata,=ax.plot(w,y,linestyle="none",marker=marker,markersize=7,
             markerfacecolor="none",markeredgecolor="#1F4E79",markeredgewidth=1.8,zorder=4)
        handles.append((hfit,hdata))
        labels.append(rf"$L={L}$: ${f['A40']:.3f}{f['B40']:+.3f}\sqrt{{\omega}}$")
        fit_ys.extend(yg)
        assert np.array_equal(hdata.get_ydata(),y)
    ax.set_xlabel(r"$\omega$",fontweight="bold")
    ax.set_ylabel(r"$\mu^*(T)$",fontweight="bold")
    ax.set_xlim(min(OMEGAS)-6,max(OMEGAS)+8)
    all_y=[r["mu40"] for r in rows]+fit_ys
    ax.set_ylim(min(all_y)-.8,max(all_y)+2.8)
    legend=ax.legend(handles,labels,handler_map={tuple:HandlerTuple(ndivide=1)},
        frameon=True,loc="upper left",
        title=r"$L;\ \mu^*(40)\approx A+B\sqrt{\omega}$",title_fontsize=11)
    legend.get_title().set_math_fontfamily("cm")
    fig.tight_layout()
    save_figure(fig,out,"Fig_S10_centered_T40")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default_out=Path(__file__).resolve().parents[1]
    parser.add_argument('--output',type=Path,default=default_out)
    parser.add_argument('--raw-root',type=Path,default=default_out.parent)
    parser.add_argument('--l20-omega75',type=Path)
    args=parser.parse_args()
    out,raw=args.output.resolve(),args.raw_root.resolve()
    for name in ('figures','tables','provenance','qa','latex'):
        (out/name).mkdir(parents=True,exist_ok=True)
    paths={}
    for L,name in ((20,'L=20_Main'),(30,'L=30_SweepCheck')):
        for p in sorted((raw/name).glob('*/revision_run.json')):
            cfg=json.loads(p.read_text())['configuration'];w=cfg['omega']
            if w not in OMEGAS: continue
            key=(L,int(w))
            assert key not in paths, f'Duplicate {key}'
            paths[key]=p.parent
    if (20,75) not in paths:
        candidates=([args.l20_omega75] if args.l20_omega75 else
            list((raw.parents[2]/'1.Main').glob('5043.Spin_Gauss_centeredABC_Omega=75_L=20_T=40*')))
        assert len(candidates)==1, f'Expected one completed L20 omega75 input: {candidates}'
        paths[20,75]=candidates[0]
    assert set(paths)=={(L,w) for L in (20,30) for w in OMEGAS}, 'Missing required cases'
    runs={k:read_run(p,*k) for k,p in sorted(paths.items())}
    rows=[runs[L,w]['summary'] for L in (20,30) for w in OMEGAS]
    fits={L:fit([r for r in rows if r['L']==L]) for L in (20,30)}
    old_fits={L:fit([r for r in rows if r['L']==L and r['omega']!=75]) for L in (20,30)}
    for L in (20,30):
        for r,e in zip([r for r in rows if r['L']==L],fits[L]['residuals']):
            r['fit_mu40']=fits[L]['A40']+fits[L]['B40']*np.sqrt(r['omega'])
            r['fit_residual']=e
    pd.DataFrame(rows).to_csv(out/'tables/S10_statistics_and_validation.csv',index=False,float_format='%.17g')
    pd.DataFrame([{k:v for k,v in f.items() if k not in ('omegas','residuals')}
                  for f in fits.values()]).to_csv(out/'tables/S10_fit_coefficients.csv',index=False,float_format='%.17g')
    pd.DataFrame([dict(case_id=r['case_id'],L=r['L'],omega=r['omega'],path=r['source_path'])
                  for r in rows]).to_csv(out/'provenance/selected_cases.csv',index=False)
    write_json(out/'tables/S10_full_precision_results.json',dict(rows=rows,fits=list(fits.values()),
        previous_six_point_fits=list(old_fits.values())))
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
        'axes.labelsize':13,'legend.fontsize':11,'axes.linewidth':1,
        'savefig.bbox':'tight','figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','savefig.transparent':False,'pdf.fonttype':42,
        'ps.fonttype':42,'path.simplify':False})
    plot_s10(rows,fits,out)
    provenance=dict(scope='Fig S10 only; seven confinement values per length at T=40',
        change='User requested inclusion of omega75 for BOTH lengths on 2026-10-21.',
        selected_omegas=list(OMEGAS),raw_root=str(raw),
        method='Integrate piecewise-linear every-step stored mass divided by P0_init64.',
        event_check='D40=sum(p_j)/P0; tau_cond40=sum(t_mid*p_j)/sum(p_j).',
        metric='half_weight_last_z_node; already incorporated, never weighted twice',
        fits='Separate unweighted least squares on all seven points; RMSE=sqrt(SSE/n).',
        precision='float32 coordinates/potential; complex64 assembly/storage; complex128 solve and norm reductions; float64 analysis',
        figures='Original hollow blue circle/square markers, solid/dashed black fits; vector PDF and 600dpi PNG.',
        script_sha256=sha256(__file__),runs=[runs[k]['provenance'] for k in sorted(runs)],
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        limitations='Completed runs, not convergence certified; midpoint-source geometry is distinct from fixed-source lower-wall controls.')
    for r in runs.values():
        for name,digest in r['provenance']['input_sha256'].items():
            assert sha256(Path(r['provenance']['path'])/name)==digest
    provenance['raw_inputs_unchanged_after_rendering']=True
    write_json(out/'provenance/run_provenance_manifest.json',provenance)
    f20,f30=fits[20],fits[30]
    equation=rf"""\begin{{equation}}
\mu^*(40;\omega)\simeq
\begin{{cases}}
{f20['A40']:.3f}+{f20['B40']:.3f}\sqrt{{\omega}}, & L=20,\quad r_{{\mathrm{{rms}}}}={f20['RMSE']:.3f},\\
{f30['A40']:.3f}+{f30['B40']:.3f}\sqrt{{\omega}}, & L=30,\quad r_{{\mathrm{{rms}}}}={f30['RMSE']:.3f}.
\end{{cases}}
\end{{equation}}
"""
    (out/'latex/S138_seven_point_replacement.tex').write_text(equation)
    caption=rf"""\caption{{Restricted means for centered spinor-ABC guides of nominal
lengths $L=20$ (circles) and $L=30$ (squares) at the common horizon $T=40$,
with midpoint sources $z_c=L/2$, $k_z=\kappa=\pi$, Gaussian amplitude width
$\sigma_\parallel=0.5$, and initial spinor $\eta=(1,0)^{{\mathsf T}}$.
Both series include $\omega=1,25,50,75,100,150,200$.
Markers are restricted means obtained by integrating the normalized,
piecewise-linear every-step stored-mass histories, with the centered
half-weight endpoint metric already incorporated. The solid ($L=20$)
and dashed ($L=30$) curves are separate unweighted fits to
$A_{{40}}(L)+B_{{40}}(L)\sqrt{{\omega}}$, with coefficients shown in the legend.
The RMS residuals are ${f20['RMSE']:.6f}$ and ${f30['RMSE']:.6f}$,
and $R^2={f20['R_squared']:.6f}$ and ${f30['R_squared']:.6f}$, respectively.}}
"""
    (out/'latex/Fig_S10_caption.tex').write_text(caption)
    body=rf"""% Replace the prior six-point S10 fit paragraph and corresponding Eq. (S138).
% Preserve the existing manuscript labels and automatic equation numbering.
Figure~S10 compares the restricted means at $T=40$ for nominal guide lengths
$L=20$ and $30$, with midpoint sources $z_c=10$ and $15$, respectively.
Both series use all seven confinements
$\omega\in\{{1,25,50,75,100,150,200\}}$.
The restricted means are integrals of the normalized, piecewise-linear
every-step stored-mass histories. Separate unweighted least-squares fits give
$A_{{40}}(20)={f20['A40']:.6f}$, $B_{{40}}(20)={f20['B40']:.6f}$,
$A_{{40}}(30)={f30['A40']:.6f}$, and $B_{{40}}(30)={f30['B40']:.6f}$.
The RMS residuals are ${f20['RMSE']:.6f}$ and ${f30['RMSE']:.6f}$,
with $R^2={f20['R_squared']:.6f}$ and ${f30['R_squared']:.6f}$,
and maximum absolute residuals ${f20['maximum_absolute_residual']:.6f}$ and
${f30['maximum_absolute_residual']:.6f}$ for $L=20$ and $30$, respectively.
The longer-guide fitted slope is {(f30['B40']/f20['B40']-1)*100:.2f}\% larger.
These fits summarize the finite observation window at the stated numerical
settings. The midpoint-source comparison changes the source--detector
distance and remains distinct from the separate fixed-source lower-wall controls.
"""
    (out/'latex/S9_4_fit_paragraph.tex').write_text(body)
    lines=['FIG S10 UPDATED WITH OMEGA=75, 2026-10-21',
        'All seven confinements {1,25,50,75,100,150,200} included for BOTH L=20 and L=30.',
        'All 14 centered-ABC raw histories recalculated through exactly T=40.',
        'Fig S9 and the original manuscript are unchanged.',
        '', 'L omega case D40 tau_cond40 mu40']
    for r in rows:
        lines.append(f"{r['L']} {r['omega']} {r['case_id']} {r['D40']:.12f} {r['tau_cond40']:.12f} {r['mu40']:.12f}")
    lines+=['','Fit diagnostics: L A40 B40 RMSE R2 max_abs_residual (unweighted; n=7).']
    for L in (20,30):
        f=fits[L]
        lines.append(f"{L} {f['A40']:.16g} {f['B40']:.16g} {f['RMSE']:.16g} {f['R_squared']:.16g} {f['maximum_absolute_residual']:.16g}")
    lines+=['',f"L30 slope increase over L20: {(f30['B40']/f20['B40']-1)*100:.6f}%.",
        f"Maximum closure |1-S-D|: {max(r['max_abs_closure'] for r in rows):.17g}",
        f"Maximum restricted-mean identity residual: {max(abs(r['restricted_mean_identity_residual']) for r in rows):.17g}",
        f"Maximum identity residual after logged numerical defect accounting: {max(abs(r['identity_after_defect_residual']) for r in rows):.17g}",
        'All steps contiguous; no time or stored-mass continuity gaps.',
        'All roof events nonnegative; half-weight endpoint metric already included.',
        'Normalizations are the recorded P0_init64; no clipping or renormalization applied.',
        'Source hashes, embedded case dictionaries, constants, physical detector planes and precision verified.',
        'All runs are completed numerical runs, not convergence certified.',
        'The midpoint-source comparison does not isolate lower-wall effects.',
        '', 'Reproduce: python scripts/reproduce_s10.py --raw-root /path/to/Output',
        'Optional: --l20-omega75 /path/to/case5043 if it is not in sibling 1.Main.',
        'Requires numpy>=2, pandas>=2, matplotlib>=3.8.',
        'All numerical CSVs use full-precision values before presentation rounding.',
        'The native every-step mass histories remain in their recorded raw case paths.',
        'run_provenance_manifest.json records all required paths and SHA256 checksums.']
    (out/'README_results.txt').write_text('\n'.join(lines)+'\n')
    (out/'requirements.txt').write_text('numpy>=2.0\npandas>=2.0\nmatplotlib>=3.8\n')
    print(json.dumps(dict(fits=list(fits.values()),added_points=[r for r in rows if r['omega']==75],
        output=str(out)),indent=2))

if __name__=='__main__':
    main()

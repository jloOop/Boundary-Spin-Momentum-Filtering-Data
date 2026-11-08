#!/usr/bin/env python3
"""Read-only raw-data post-processing for Tables S2--S4; never runs a solver.

Run with Python 3, numpy and pandas. Outputs are confined to this script's
directory by default. All CSV floating point numbers use 17 significant digits.
Source snapshots are parsed as text/AST, NEVER imported or executed.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import sys

import numpy as np
import pandas as pd

OMEGAS = [1, 25, 50, 75, 100, 150, 200, 300, 400]
CASES = [5540, 5042, 5047, 5043, 5541, 5044, 5045, 7100, 5046]
T, CUT = 40.0, 3.6


def total(x):
    return math.fsum(np.asarray(x, dtype=np.float64).ravel())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def linear_segments(t, y, a=0., b=T):
    """Clip interval geometry, never data; no extrapolation or smoothing."""
    t, y = np.asarray(t, np.float64), np.asarray(y, np.float64)
    assert t.ndim == y.ndim == 1 and len(t) == len(y)
    assert np.all(np.isfinite(t)) and np.all(np.isfinite(y))
    assert np.all(np.diff(t) > 0) and t[0] <= a < b <= t[-1]
    l, r = np.maximum(t[:-1], a), np.minimum(t[1:], b)
    keep = r > l
    l, r = l[keep], r[keep]
    slope = (np.diff(y)/np.diff(t))[keep]
    yl = y[:-1][keep] + slope*(l-t[:-1][keep])
    yr = y[:-1][keep] + slope*(r-t[:-1][keep])
    return l, r, yl, yr


def linear_integrals(t, y, a=0., b=T):
    """Exact zeroth and first moments of the linear interpolant."""
    l, r, yl, yr = linear_segments(t, y, a, b)
    h = r-l
    return total(h*(yl+yr)/2), total(h*((2*l+r)*yl+(l+2*r)*yr)/6)


def event_integrals(left, right, probability, a, b):
    """Exact interval overlap and first moment of constant event rate."""
    left, right, probability = map(lambda x: np.asarray(x, np.float64),
                                   (left, right, probability))
    assert np.all(right > left)
    l, r = np.maximum(left, a), np.minimum(right, b)
    keep = r > l
    rate = probability[keep]/(right-left)[keep]
    weight = rate*(r-l)[keep]
    return total(weight), total(weight*(l+r)[keep]/2)


def valid_product(t, n, e, valid, a=0., b=T):
    """Exact product of separate linear interpolants on valid intervals only."""
    t, n, e = map(lambda x: np.asarray(x, np.float64), (t, n, e))
    valid = np.asarray(valid, bool) & np.isfinite(e)
    assert t[0] <= a < b <= t[-1] and np.all(np.diff(t) > 0)
    l, r = np.maximum(t[:-1], a), np.minimum(t[1:], b)
    overlap = r > l
    good = overlap & valid[:-1] & valid[1:]
    bad = overlap & ~good
    def endpoints(y, mask):
        slope = (y[1:][mask]-y[:-1][mask])/np.diff(t)[mask]
        return (y[:-1][mask]+slope*(l[mask]-t[:-1][mask]),
                y[:-1][mask]+slope*(r[mask]-t[:-1][mask]))
    n0, n1 = endpoints(n, good)
    e0, e1 = endpoints(e, good)
    h = (r-l)[good]
    numerator = total(h*(2*n0*e0+n0*e1+n1*e0+2*n1*e1)/6)
    weight = total(h*(n0+n1)/2)
    b0, b1 = endpoints(n, bad)
    excluded_weight = total((r-l)[bad]*(b0+b1)/2)
    return dict(E_2br=numerator/weight, numerator=numerator, valid_weight=weight,
                excluded_duration=total((r-l)[bad]), excluded_boundary_weight=excluded_weight,
                valid_interval_count=int(good.sum()), excluded_interval_count=int(bad.sum()))


def analytic_tests():
    checks = {}
    def eq(name, got, expected):
        residual = float(got-expected)
        assert abs(residual) < 3e-14, (name, got, expected)
        checks[name] = dict(value=float(got), expected=float(expected), residual=residual)
    # Rates 2 on [0,2], 3 on [2,5], clipped at both ends.
    d, m = event_integrals([0,2], [2,5], [4,9], 1,4)
    eq('partial_event_mass',d,8); eq('partial_event_first_moment',m,21)
    # y=1+2t on an irregular grid, integrated on a strict subinterval.
    d,m = linear_integrals([0,0.7,2,5], [1,2.4,5,11], 0.2,4)
    eq('linear_integral',d,(4+4**2)-(.2+.2**2))
    eq('linear_first_moment',m,(4**2/2+2*4**3/3)-(.2**2/2+2*.2**3/3))
    # N=1+t, E=2-t: product=2+t-t^2. Final interval is deliberately invalid.
    v = valid_product([0,1,2,3], [1,2,3,4], [2,1,0,np.nan], [1,1,1,0], .5,3)
    F = lambda x: 2*x+x*x/2-x**3/3
    eq('linear_product',v['numerator'],F(2)-F(.5))
    eq('valid_weight',v['valid_weight'],(2+2**2/2)-(.5+.5**2/2))
    eq('invalid_interval_duration',v['excluded_duration'],1)
    eq('invalid_interval_weight',v['excluded_boundary_weight'],3.5)
    # Invalid middle node must invalidate both adjacent intervals, not be bridged.
    v = valid_product([0,1,2,3], [1,1,1,1], [2,np.nan,2,2], [1,0,1,1],0,3)
    eq('no_bridge_duration',v['excluded_duration'],2)
    eq('no_bridge_average',v['E_2br'],2)
    return checks


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    def display(v):
        if v is None: return 'pending'
        if isinstance(v,(float,np.floating)): return format(float(v),'.17g')
        return v
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys);w.writeheader()
        w.writerows({k:display(v) for k,v in row.items()} for row in rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def source_literal(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('Missing source literal '+name)


def verify_metadata(p, omega, case_id):
    run=json.loads((p/'revision_run.json').read_text()); c=run['configuration']
    case=json.loads((p/'branch_refinement_case.json').read_text())
    source=(p/'source_snapshot.py').read_text()
    assert source_literal(source,'BRANCH_REFINEMENT_CASE') == case
    assert sha(p/'source_snapshot.py') == run['updated_source_sha256']
    assert case['case_id'] == case_id and c['omega'] == case['omega'] == omega
    assert c['endpoint_closure'] == case['endpoint_closure'] == 'centered'
    assert c['endpoint_mass_weight'] == case['endpoint_mass_weight'] == .5
    assert c['metric'] == case['metric'] == 'half_weight_last_z_node'
    assert run['status'] == 'COMPLETED_NUMERICAL_RUN_NOT_CONVERGENCE_CERTIFIED'
    assert run['actual_time'] == c['T_final'] == case['T_final'] == T
    for key in ['Nx','Ny','Nz','dt','Lz','ell','sigma_z','z0_init','theta','phi',
                'state_dtype','initial_preparation_dtype','potential_sampling_dtype']:
        assert case[key] == c[key], (p,key)
    assert c['k0'] == c['k_bc'] == math.pi
    assert c['z0_init']==10 and c['sigma_z']==.5 and c['theta']==c['phi']==0
    assert c['matrix_assembly'] == case['matrix_dtype']+'_then_solve_complex128'
    with np.load(p/'constants.npz',allow_pickle=False) as z:
        constants={k:z[k].item() for k in z.files if z[k].ndim==0}
    for key in ['omega','T_final','theta','phi','sigma_z','k0','k_bc','z_roof_actual',
                'grid_generator_nz','hx','hy','hz','Lx','Ly','Lz','dt','Nx','Ny','Nz',
                'endpoint_mass_weight','endpoint_closure','metric','state_representation','state_dtype']:
        assert constants[key] == c[key], (p,key)
    assert constants['z0']==c['z0_init'] and constants['matrix_dtype']==case['matrix_dtype']
    coords={}
    with np.load(p/'revision_coordinates.npz',allow_pickle=False) as z:
        for axis in 'xyz':
            a=z[axis];n=c['N'+axis];h=c['h'+axis]
            assert len(a)==n and str(a.dtype)==case['coordinate_dtype']
            assert a[0]==0 and np.all(np.diff(a.astype(np.float64))>0)
            assert float(a[1]-a[0])==h
            coords[axis]=dict(dtype=str(a.dtype),nodes=n,first=float(a[0]),last=float(a[-1]),
                             min_actual_spacing=float(np.diff(a.astype(np.float64)).min()),
                             max_actual_spacing=float(np.diff(a.astype(np.float64)).max()))
        assert float(z['z'][-1])==c['z_roof_actual']
    summary=json.loads((p/'summary.json').read_text())
    assert summary['omega']==omega and summary['endpoint_closure']=='centered'
    assert summary['P0_init64']==run['P0_init64']
    # Job identity is corroborated by recorded HPC paths, not chosen from a filename.
    recorded_path=summary['run_dir']
    job=re.search(r'_job(\d+)',recorded_path).group(1)
    assert p.name==Path(recorded_path).name and '_job'+job in run['last_checkpoint']
    files=['revision_run.json','branch_refinement_case.json','constants.npz','revision_coordinates.npz',
           'source_snapshot.py','revision_step_data.csv','revision_spin_branch_diagnostics.npz',
           'prob_times.npy','total_probs.npy','summary.json','simulation_log.txt','stdout.txt']
    manifest=dict(omega=omega,new_case_id=case_id,job_id=job,run_id=run['version'],directory=str(p),
                  original_hpc_directory=recorded_path,source_sha256=sha(p/'source_snapshot.py'),
                  status=run['status'],P0_init64=run['P0_init64'],configuration=c,coordinates=coords,
                  source_was_executed=False,hashes={f:sha(p/f) for f in files if (p/f).exists()})
    return run,manifest


def matrix_checks(matrix, prefix):
    q=np.asarray(matrix,np.complex128)
    tr=np.trace(q,axis1=-2,axis2=-1).real
    eig=np.linalg.eigvalsh(q)
    active=tr>0
    pol=(q[...,0,0].real[active]-q[...,1,1].real[active])/tr[active]
    bloch=np.hypot(q[...,0,0].real[active]-q[...,1,1].real[active],
                   2*np.abs(q[...,0,1][active]))/tr[active]
    return {prefix+'_minimum_eigenvalue':float(eig.min()),
            prefix+'_negative_eigenvalues_count':int(np.count_nonzero(eig<0)),
            prefix+'_eigenvalues_below_minus_1e_minus14_count':int(np.count_nonzero(eig < -1e-14)),
            prefix+'_hermiticity_max_abs':float(np.max(np.abs(q-q.swapaxes(-1,-2).conj()))),
            prefix+'_trace_imag_max_abs':float(np.max(np.abs(np.trace(q,axis1=-2,axis2=-1).imag))),
            prefix+'_polarization_max_abs':float(np.abs(pol).max()) if len(pol) else 0.,
            prefix+'_polarization_violations_count':int(np.count_nonzero(np.abs(pol)>1)),
            prefix+'_bloch_norm_max':float(bloch.max()) if len(bloch) else 0.,
            prefix+'_bloch_norm_violations_count':int(np.count_nonzero(bloch>1))}


def compute_run(p, omega, case_id):
    run,manifest=verify_metadata(p,omega,case_id);c=run['configuration'];p0=float(run['P0_init64'])
    frame=pd.read_csv(p/'revision_step_data.csv',dtype=np.float64,float_precision='round_trip')
    d={k:frame[k].to_numpy(dtype=np.float64) for k in frame.columns}
    assert all(np.all(np.isfinite(a)) for a in d.values())
    l,r=d['t_left'],d['t_right'];n=len(r)
    assert np.array_equal(d['step'],np.arange(1,n+1,dtype=np.float64))
    assert n==run['completed_step'] and l[0]==0 and r[-1]==T
    assert np.array_equal(l[1:],r[:-1]) and np.all(r>l)
    prior=np.r_[p0,d['mass_stored'][:-1]]
    continuity=float(np.max(np.abs(d['mass_before']-prior)))
    assert continuity < 1e-13*p0, (p,continuity)
    prob=d['roof_probability']/p0
    de,me=event_integrals(l,r,prob,0,CUT);dl,ml=event_integrals(l,r,prob,CUT,T)
    det=de+dl;mt=me+ml;te=me/de;tl=ml/dl;wl=dl/det;tau=mt/det
    st=np.r_[0.,r];sv=np.r_[p0,d['mass_stored']]/p0
    mu=linear_integrals(st,sv)[0];s40=sv[-1]
    s2=dict(omega=omega,D40=det,tau_cond40=tau,mu_star40=mu,tE=te,wL40=wl,tL40=tl,
            Delta_tL_cond40=wl*(tl-te))
    with np.load(p/'revision_spin_branch_diagnostics.npz',allow_pickle=False) as zz:
        z={k:zz[k].copy() for k in zz.files}
    t=z['time'].astype(np.float64);N=z['roof_density'].astype(np.float64)
    assert t[0]==0 and t[-1]==T and np.all(np.diff(t)>0)
    steps=z['step'].astype(np.int64)
    assert np.all(np.diff(steps)>0) and np.array_equal(t,st[steps])
    Z,TN=linear_integrals(t,N);U2=linear_integrals(t,z['roof_R2_numerator'])[0]
    Ua,tUa=linear_integrals(t,z['roof_Ra_numerator'])
    mean_boundary=TN/Z;es2=U2/(omega*Z);lam=tUa/Z-mean_boundary*Ua/Z
    beta=lam/math.sqrt(omega);q=c['hz']**2*U2/Z
    e=valid_product(t,N,z['one_grid_E2br'],z['one_grid_valid'])
    native_budget=None;native_info={};plane_checks={}
    plane_files=['plane_times.npy','roof_rho_int.npy','P_total_on_plane_times.npy']
    if all((p/f).is_file() for f in plane_files):
        pt,pn,pm=[np.asarray(np.load(p/f,allow_pickle=False),np.float64) for f in plane_files]
        assert pt.ndim==pn.ndim==pm.ndim==1 and len(pt)==len(pn)==len(pm)
        assert all(np.all(np.isfinite(a)) for a in (pt,pn,pm))
        assert pt[0]==0 and pt[-1]==T and np.all(np.diff(pt)>0)
        assert np.all(pn>=0) and np.all(pm>0)
        pi=np.searchsorted(st,pt)
        assert np.all(pi<len(st)) and np.array_equal(st[pi],pt), 'Native plane records must match actual step endpoints exactly'
        assert pm[0]==run['P0_init_native'], 'Native launch reduction does not match same-run metadata'
        native_pt=np.asarray(np.load(p/'prob_times.npy',allow_pickle=False),np.float64)
        native_pm=np.asarray(np.load(p/'total_probs.npy',allow_pickle=False),np.float64)
        common_np,plane_idx,native_idx=np.intersect1d(pt,native_pt,return_indices=True)
        assert len(common_np)==len(native_pt) and np.array_equal(pm[plane_idx],native_pm[native_idx])
        # Eq(S96): native plane-time trapezoids and high-precision declared-metric
        # launch/final masses. P_total is an independent native-reduction check only.
        plane_integral=float(np.trapezoid(pn,pt))
        loss64=p0-float(d['mass_stored'][-1]);flux=c['k_bc']*plane_integral
        assert loss64>0
        native_budget=abs(flux-loss64)/loss64
        same_diag,plane_diag_idx,diag_idx=np.intersect1d(pt,t,return_indices=True)
        recorded_summary=json.loads((p/'summary.json').read_text())['roof_checks']['roof']
        plane_checks=dict(omega=omega,new_case_id=case_id,job_id=manifest['job_id'],
            Delta_bud_rel=native_budget,P0_init64=p0,P40_mass_stored=float(d['mass_stored'][-1]),
            mass_loss64=loss64,kappa=c['k_bc'],plane_density_integral=plane_integral,
            kappa_plane_density_integral=flux,signed_flux_minus_mass_loss=flux-loss64,
            plane_samples=len(pt),time_start=float(pt[0]),time_end=float(pt[-1]),
            plane_dt_min=float(np.diff(pt).min()),plane_dt_max=float(np.diff(pt).max()),
            plane_final_interval=float(pt[-1]-pt[-2]),
            plane_time_vs_step_endpoint_max_abs=float(np.max(np.abs(pt-st[pi]))),
            native_plane_P0=float(pm[0]),native_plane_P40=float(pm[-1]),
            native_P0_minus_P0_init_native=float(pm[0]-run['P0_init_native']),
            native_P0_minus_P0_init64=float(pm[0]-p0),
            native_P40_minus_mass_stored=float(pm[-1]-d['mass_stored'][-1]),
            native_plane_survival_vs_mass64_max_abs=float(np.max(np.abs(pm-p0*sv[pi]))),
            normalized_native_plane_survival_vs_step_survival_max_abs=float(np.max(np.abs(pm/p0-sv[pi]))),
            common_native_survival_samples=len(common_np),
            native_plane_vs_sampled_survival_common_max_abs=float(np.max(np.abs(pm[plane_idx]-native_pm[native_idx]))),
            common_spin_diagnostic_samples=len(same_diag),
            native_roof_vs_spin_density_common_max_abs=float(np.max(np.abs(pn[plane_diag_idx]-N[diag_idx]))),
            plane_integral_vs_linear_interpolant_abs=abs(plane_integral-linear_integrals(pt,pn)[0]),
            flux_minus_summary_int_kappa_rho=flux-recorded_summary['int_kappa_rho'],
            native_mass_loss_minus_summary_total_loss=float(pm[0]-pm[-1]-recorded_summary['total_loss']),
            budget_using_native_reduction_endpoints_CROSSCHECK_ONLY=abs(flux-(pm[0]-pm[-1]))/(pm[0]-pm[-1]))
        native_info=dict(plane_integral=plane_integral,plane_samples=len(pt))
        manifest['hashes'].update({f:sha(p/f) for f in plane_files})
        manifest['native_plane_inputs']={f:dict(local_path=str(p/f),
            recorded_cluster_path=manifest['original_hpc_directory']+'/'+f,
            sha256=sha(p/f),size_bytes=(p/f).stat().st_size,
            availability='found locally before any download',downloaded_this_task=False) for f in plane_files}
    s3=dict(omega=omega,hz_sqrt_omega=c['hz']*math.sqrt(omega),q_omega=q,
            Delta_bud_rel=native_budget,E_s2=es2,beta_bl_omega=beta,Lambda_omega=lam,E_2br=e['E_2br'])
    fractions=[linear_integrals(t,z[key])[0]/Z for key in ['branch_plus_density','branch_minus_density','neutral_density']]
    event_masses=[event_integrals(l,r,d[key]/p0,0,T)[0] for key in ['p_spin_up','p_spin_down']]
    det_pol=(event_masses[0]-event_masses[1])/sum(event_masses)
    Q=np.asarray(z['no_click_spin_matrix'],np.complex128);Q40=Q[-1]
    nc_pol=float((Q40[0,0]-Q40[1,1]).real/np.trace(Q40).real)
    s4=dict(omega=omega,wbar_br_plus40=fractions[0],wbar_br_minus40=fractions[1],
            wbar_br_neutral40=fractions[2],Pbar_det_z40=det_pol,P_nc_z40=nc_pol)
    eventq=np.zeros((n,2,2),np.complex128)
    eventq[:,0,0]=d['p_spin_up'];eventq[:,1,1]=d['p_spin_down']
    eventq[:,0,1]=d['p_spin_offdiag_real']+1j*d['p_spin_offdiag_imag']
    eventq[:,1,0]=eventq[:,0,1].conj()
    branch_error=z['branch_plus_density']+z['branch_minus_density']+z['neutral_density']-N
    cumulative=np.cumsum(prob,dtype=np.float64)+sv[1:]-1
    defects=d['cast_mass_loss']+d['clamp_mass_loss']+d['solve_and_matrix_defect']
    accounting=d['mass_before']-d['mass_stored']-d['roof_probability']-defects
    validation=dict(omega=omega,D40_plus_S40_minus_1=det+s40-1,
        max_cumulative_event_survival_discrepancy=float(np.abs(cumulative).max()),
        cumulative_discrepancy_t_at_max=float(r[np.argmax(np.abs(cumulative))]),
        mu_star_minus_event_restricted_identity=mu-(mt+T*(1-det)),
        conditional_mean_identity_residual=tau-(te+wl*(tl-te)),
        early_late_partition_residual=de+dl-total(prob),
        mass_before_preceding_stored_max_abs=continuity,
        branch_partition_max_abs=float(np.abs(branch_error).max()),
        branch_partition_signed_integral=linear_integrals(t,branch_error)[0],
        integrated_fraction_sum_residual=sum(fractions)-1,
        event_matrix_trace_minus_roof_probability_max_abs=float(np.max(np.abs(d['p_spin_up']+d['p_spin_down']-d['roof_probability']))),
        event_matrix_trace_minus_roof_probability_sum=total(d['p_spin_up']+d['p_spin_down']-d['roof_probability']),
        survivor_trace_minus_mass64_max_abs=float(np.max(np.abs(np.trace(Q,axis1=1,axis2=2).real-z['no_click_mass64']))),
        survivor_trace_minus_step_mass_max_abs=float(np.max(np.abs(np.trace(Q,axis1=1,axis2=2).real-p0*sv[steps]))),
        survivor_endpoint_trace_minus_mass=float(np.trace(Q40).real-d['mass_stored'][-1]),
        Lambda_minus_sqrt_omega_beta=lam-math.sqrt(omega)*beta,
        q_minus_hzsqrtomega_squared_Es2=q-(c['hz']*math.sqrt(omega))**2*es2,
        min_roof_probability=float(d['roof_probability'].min()),negative_roof_probability_count=int((d['roof_probability']<0).sum()),
        min_stored_mass=float(d['mass_stored'].min()),
        numerical_accounting_per_step_max_abs=float(np.abs(accounting).max()),
        cumulative_roof_probability_minus_run_metadata=total(d['roof_probability'])-run['cumulative_roof_probability'],
        final_mass_minus_run_metadata=float(d['mass_stored'][-1]-run['current_accepted_prefix_mass64']),
        relative_solve_residual_max=float(d['relative_residual'].max()),
        **matrix_checks(eventq,'event'),**matrix_checks(Q,'survivor'),
        **matrix_checks(z['roof_spin_matrix'],'boundary'))
    accounting_out=dict(omega=omega)
    for key in ['roof_probability','cast_mass_loss','clamp_mass_loss','solve_and_matrix_defect']:
        accounting_out[key+'_sum_over_P0']=total(d[key])/p0
        accounting_out[key+'_max_abs_cumulative_over_P0']=float(np.max(np.abs(np.cumsum(d[key]/p0))))
    accounting_out['closure_plus_separate_defects']=det+s40-1+total(defects)/p0
    nt=np.asarray(np.load(p/'prob_times.npy'),np.float64)
    nv=np.asarray(np.load(p/'total_probs.npy'),np.float64)
    assert nt[0]==0 and nt[-1]==T and np.all(np.diff(nt)>0)
    idx=np.searchsorted(st,nt)
    assert np.all(idx<len(st)) and np.allclose(st[idx],nt,rtol=0,atol=1e-12)
    native_info.update(native_norm_samples=len(nt),native_norm_initial=float(nv[0]),native_norm_final=float(nv[-1]),
        native_norm_minus_every_step_max_abs=float(np.max(np.abs(nv-p0*sv[idx]))),
        native_norm_initial_minus_P0_init_native=float(nv[0]-run['P0_init_native']),
        native_norm_survival_integral_crosscheck=linear_integrals(nt,nv/p0)[0])
    diagnostic_budget=abs(c['k_bc']*Z-(p0-d['mass_stored'][-1]))/(p0-d['mass_stored'][-1])
    supplementary=dict(omega=omega,D_E=de,D_L=dl,M_E=me,M_L=ml,S40=s40,Z=Z,
        mean_t_boundary=mean_boundary,U2_integral=U2,Ua_integral=Ua,tUa_integral=tUa,
        diagnostic_time_budget_DIFFERENT_SAMPLING_CONVENTION=diagnostic_budget,
        E2br_excluded_boundary_fraction=e['excluded_boundary_weight']/Z,
        **{'E2br_'+k:v for k,v in e.items() if k!='E_2br'},**native_info)
    manifest.update(accepted_steps=n,first_event_time=float(l[0]),last_event_time=float(r[-1]),
        diagnostic_nodes=len(t),diagnostic_first_time=float(t[0]),diagnostic_last_time=float(t[-1]),
        dt_min=float((r-l).min()),dt_max=float((r-l).max()),dt_final=float(r[-1]-l[-1]),
        final_step_shortened=bool(r[-1]-l[-1] < c['dt']-1e-12),
        native_plane_density_available=native_budget is not None,
        history_policy='Read the single consolidated accepted-step CSV and diagnostic NPZ once; require strictly increasing steps/times; no sorting, concatenation or restored-prefix duplication.')
    return s2,s3,s4,validation,accounting_out,supplementary,manifest,plane_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    write_json(out/'analytic_tests.json',analytic_tests())
    # Discover candidates through metadata, with the figure-selected case IDs fixed explicitly.
    catalogue=[]
    for f in sorted(root.glob('*/revision_run.json')):
        cf=f.parent/'branch_refinement_case.json'
        if cf.is_file():
            rr=json.loads(f.read_text());cc=json.loads(cf.read_text())
            catalogue.append((rr['configuration']['omega'],cc['case_id'],f.parent))
    main_figure_manifests=[root/'Fig1_centered_all7_2026-10-18/selected_runs.csv',
                           root/'Fig3_centered_2026-10-17/selected_runs.csv']
    figure_paths=set()
    for f in main_figure_manifests:
        with f.open() as stream:
            figure_paths.update(row['selected_run_path'] for row in csv.DictReader(stream))
    results=[]
    for omega,case in zip(OMEGAS,CASES):
        matches=[p for w,k,p in catalogue if w==omega and k==case]
        assert len(matches)==1,(omega,case,matches)
        p=matches[0];assert str(p) in figure_paths
        result=compute_run(p,omega,case);results.append(result)
        print(f'omega={omega}: case={case}; D40={result[0]["D40"]:.17g}; checks computed',flush=True)
    names=['Table_S2_centered_full_precision.csv','Table_S3_full_precision.csv','Table_S4_full_precision.csv',
           'validation_full_precision.csv','numerical_accounting.csv','supplementary_integrals.csv']
    for i,name in enumerate(names):write_csv(out/name,[r[i] for r in results])
    manifests=[r[6] for r in results]
    write_csv(out/'native_plane_budget_full_precision.csv',[r[7] for r in results if r[7]])
    write_json(out/'native_plane_budget_checks.json',[r[7] for r in results if r[7]])
    native_files=[dict(omega=m['omega'],new_case_id=m['new_case_id'],job_id=m['job_id'],filename=f,**info)
                  for m in manifests for f,info in m.get('native_plane_inputs',{}).items()]
    write_csv(out/'native_plane_file_manifest.csv',native_files)
    write_json(out/'native_plane_file_manifest.json',dict(downloads_performed=False,files=native_files))
    write_json(out/'manifest.json',dict(target_T=T,tcut=CUT,omega_order=OMEGAS,
        selection_references={str(f):sha(f) for f in main_figure_manifests},centered_runs=manifests))
    write_csv(out/'manifest.csv',[dict(omega=m['omega'],new_case_id=m['new_case_id'],job_id=m['job_id'],
        run_id=m['run_id'],directory=m['directory'],source_sha256=m['source_sha256'],status=m['status'],
        P0_init64=m['P0_init64'],dt=m['configuration']['dt'],hx=m['configuration']['hx'],
        hy=m['configuration']['hy'],hz=m['configuration']['hz'],
        actual_detector_z=m['configuration']['z_roof_actual'],endpoint_closure='centered',endpoint_mass_weight=.5,
        state_precision=m['configuration']['state_dtype'],matrix_assembly=m['configuration']['matrix_assembly'],
        coordinate_precision=m['configuration']['real_dtype'],accepted_steps=m['accepted_steps'],
        diagnostic_nodes=m['diagnostic_nodes'],dt_final=m['dt_final'],
        final_step_shortened=m['final_step_shortened']) for m in manifests])
    write_json(out/'environment.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,
        platform=platform.platform(),executable=sys.executable,analysis_script_sha256=sha(__file__),
        arithmetic='float64 / complex128; no simulation/source execution'))
    write_json(out/'centered_results.json',dict(S2=[r[0] for r in results],S3=[r[1] for r in results],
        S4=[r[2] for r in results],validation=[r[3] for r in results],
        numerical_accounting=[r[4] for r in results],supplementary=[r[5] for r in results],
        native_plane_budget=[r[7] for r in results]))


if __name__=='__main__':
    main()

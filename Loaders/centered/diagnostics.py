"""Native branch, spin, auxiliary-moment and Eq. S117 refinement reductions.

Adapted from the archived recalculate_tables.py and native_tools.py/compare.py.
Only the distributed saved records are read; no solver, network or source-root
access. Historical scalar results remain an archived-table route.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import numpy as np
from centered.numerics import events, linear_integral, linear_abs_difference, step_abs_difference

__all__ = ['run']
BASELINE = {1:5540,25:5042,50:5047,75:5043,100:5541,150:5044,200:5045,300:7100}
SECTORS = ('branch_plus_density','branch_minus_density','neutral_density')


def _csv(path, rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def _read_csv(path):
    with path.open(newline='') as f:
        return [{k:float(v) if v and _isfloat(v) else v for k,v in row.items()} for row in csv.DictReader(f)]


def _isfloat(s):
    try:float(s);return True
    except (ValueError,TypeError):return False


def _check(checks,name,actual,tolerance):
    value=float(actual)
    checks.append({'name':name,'passed':bool(np.isfinite(value) and abs(value)<=tolerance),
                   'actual':value,'tolerance':float(tolerance)})


def _load(root,case):
    path=root/'data/centered/native'/str(case)
    meta=json.loads((path/'run.json').read_text())
    with np.load(path/'steps.npz',allow_pickle=False) as z:s={k:z[k] for k in z.files}
    with np.load(path/'revision_spin_branch_diagnostics.npz',allow_pickle=False) as z:dg={k:z[k] for k in z.files}
    p0=float(meta['P0_init64']);t=np.r_[0.,s['t_right']];S=np.r_[p0,s['mass_stored']]/p0
    return dict(case=case,path=path,meta=meta,cfg=meta['configuration'],p0=p0,s=s,dg=dg,t=t,S=S)


def _validate(d,checks):
    s,dg,p0,c=d['s'],d['dg'],d['p0'],d['cfg'];n=len(s['t_left']);prefix=str(d['case'])
    required={'step','t_left','t_right','roof_probability','mass_before','mass_stored',
              'cast_mass_loss','clamp_mass_loss','solve_and_matrix_defect','p_spin_up','p_spin_down'}
    if not required.issubset(s):raise ValueError(f'{prefix}: missing native columns')
    if any(v.shape!=(n,) for v in s.values()):raise ValueError(f'{prefix}: inconsistent step shapes')
    if any(not np.isfinite(v).all() for v in s.values()):raise ValueError(f'{prefix}: nonfinite native records')
    if not (p0>0 and s['t_left'][0]==0 and s['t_right'][-1]==40):raise ValueError(f'{prefix}: unsupported launch/time support')
    if not np.all(s['t_right']>s['t_left']):raise ValueError(f'{prefix}: nonpositive intervals')
    _check(checks,prefix+'/step_sequence',not np.array_equal(s['step'],np.arange(1,n+1)),0)
    _check(checks,prefix+'/interval_chain',not np.array_equal(s['t_left'][1:],s['t_right'][:-1]),0)
    _check(checks,prefix+'/launch_norm',s['mass_before'][0]-p0,1e-14)
    _check(checks,prefix+'/mass_chain',np.max(abs(s['mass_before'][1:]-s['mass_stored'][:-1])),1e-14)
    _check(checks,prefix+'/centered_metric',not(c['endpoint_closure']=='centered' and c['endpoint_mass_weight']==.5 and c['metric']=='half_weight_last_z_node'),0)
    _check(checks,prefix+'/diagnostic_clock',not np.array_equal(dg['time'],d['t'][dg['step']]),0)
    _check(checks,prefix+'/event_spin_trace',np.max(abs(s['p_spin_up']+s['p_spin_down']-s['roof_probability'])),1e-14)
    _check(checks,prefix+'/branch_partition',np.max(abs(sum(dg[k] for k in SECTORS)-dg['roof_density'])),1e-12)
    _check(checks,prefix+'/survivor_trace',np.max(abs(np.trace(dg['no_click_spin_matrix'],axis1=1,axis2=2).real-dg['no_click_mass64'])),1e-12)
    _check(checks,prefix+'/diagnostic_mass',np.max(abs(dg['no_click_mass64']-np.r_[p0,s['mass_stored']][dg['step']])),1e-14)
    defects=s['cast_mass_loss']+s['clamp_mass_loss']+s['solve_and_matrix_defect']
    perstep=s['mass_before']-s['mass_stored']-s['roof_probability']-defects
    _check(checks,prefix+'/separate_correction_accounting',np.max(abs(perstep))/p0,1e-12)
    gap=1-d['S'][1:]-np.cumsum(s['roof_probability']/p0)
    _check(checks,prefix+'/cumulative_correction_accounting',np.max(abs(gap-np.cumsum(defects/p0))),1e-9)
    return {'case':d['case'],'event_survival_gap_final':float(gap[-1]),
            'event_survival_gap_max':float(np.max(abs(gap))),
            'casting_sum_over_P0':float(np.sum(s['cast_mass_loss'])/p0),
            'clamp_sum_over_P0':float(np.sum(s['clamp_mass_loss'])/p0),
            'solve_matrix_defect_sum_over_P0':float(np.sum(s['solve_and_matrix_defect'])/p0),
            'negative_event_count':int(np.count_nonzero(s['roof_probability']<0))}


def _linear_first_moment(t,y,a=0.,b=40.):
    x=np.r_[a,t[(t>a)&(t<b)],b];v=np.interp(x,t,y);l,r=x[:-1],x[1:]
    return float(np.sum((r-l)*((2*l+r)*v[:-1]+(l+2*r)*v[1:])/6))


def _valid_product(t,N,E,valid,a=0.,b=40.):
    """Exact product of separate linear traces; invalid intervals are excluded."""
    valid=np.asarray(valid,bool)&np.isfinite(E)
    left,right=np.maximum(t[:-1],a),np.minimum(t[1:],b)
    overlap=right>left;good=overlap&valid[:-1]&valid[1:];bad=overlap&~good
    def endpoints(y,mask):
        h=np.diff(t)[mask];slope=(y[1:][mask]-y[:-1][mask])/h
        return y[:-1][mask]+slope*(left[mask]-t[:-1][mask]),y[:-1][mask]+slope*(right[mask]-t[:-1][mask])
    n0,n1=endpoints(N,good);e0,e1=endpoints(E,good);h=(right-left)[good]
    numerator=np.sum(h*(2*n0*e0+n0*e1+n1*e0+2*n1*e1)/6)
    weight=np.sum(h*(n0+n1)/2)
    b0,b1=endpoints(N,bad)
    return {'E_2br':float(numerator/weight),'excluded_duration':float(np.sum((right-left)[bad])),
            'excluded_boundary_weight':float(np.sum((right-left)[bad]*(b0+b1)/2)),
            'valid_interval_count':int(good.sum()),'excluded_interval_count':int(bad.sum())}


def _stat(d):
    s,dg,c,p0=d['s'],d['dg'],d['cfg'],d['p0'];omega=float(c['omega']);t=dg['time'];N=dg['roof_density']
    def ev(key,a,b):return events(s['t_left'],s['t_right'],s[key],a,b,p0)
    full=ev('roof_probability',0,40);early=ev('roof_probability',0,3.6);late=ev('roof_probability',3.6,40)
    D=full['probability'];M=full['first_moment'];mu=linear_integral(d['t'],d['S'],0,40)
    wlate=late['probability']/D
    s2={'omega':omega,'D40':D,'tau_cond40':full['conditional_mean'],'mu_star40':mu,
        'tE':early['conditional_mean'],'wL40':wlate,'tL40':late['conditional_mean'],
        'Delta_tL_cond40':wlate*(late['conditional_mean']-early['conditional_mean'])}
    Z=linear_integral(t,N,0,40);TN=_linear_first_moment(t,N)
    U2=linear_integral(t,dg['roof_R2_numerator'],0,40)
    Ua=linear_integral(t,dg['roof_Ra_numerator'],0,40);tUa=_linear_first_moment(t,dg['roof_Ra_numerator'])
    lam=tUa/Z-TN*Ua/Z**2;aux=_valid_product(t,N,dg['one_grid_E2br'],dg['one_grid_valid'])
    pt=np.load(d['path']/'plane_times.npy',allow_pickle=False)
    pr=np.load(d['path']/'roof_rho_int.npy',allow_pickle=False)
    loss=p0-s['mass_stored'][-1];flux=c['k_bc']*linear_integral(pt,pr,0,40)
    s3={'omega':omega,'hz_sqrt_omega':c['hz']*math.sqrt(omega),'q_omega':c['hz']**2*U2/Z,
        'Delta_bud_rel':abs(flux-loss)/loss,'E_s2':U2/(omega*Z),
        'beta_bl_omega':lam/math.sqrt(omega),'Lambda_omega':lam,'E_2br':aux['E_2br']}
    sectors=[linear_integral(t,dg[k],0,40)/Z for k in SECTORS]
    up=ev('p_spin_up',0,40)['probability'];down=ev('p_spin_down',0,40)['probability']
    Q=dg['no_click_spin_matrix'][-1]
    s4={'omega':omega,'wbar_br_plus40':sectors[0],'wbar_br_minus40':sectors[1],
        'wbar_br_neutral40':sectors[2],'Pbar_det_z40':(up-down)/(up+down),
        'P_nc_z40':float((Q[0,0]-Q[1,1]).real/np.trace(Q).real)}
    aux.update({'omega':omega,'D_early':early['probability'],'D_late':late['probability'],
                'survival40':float(d['S'][-1]),'mu_minus_event_capped_mean':mu-(M+40*(1-D)),
                'sampled_endpoint_roof_probability':c['k_bc']*Z/p0,
                'native_CN_probability':D,'plane_roof_probability':flux/p0})
    if {'midpoint_interior_face_probability','roof_cell_mass_before','roof_cell_mass_solved','roof_cell_balance_defect'}.issubset(s):
        cell=s['midpoint_interior_face_probability']-s['roof_probability']-s['roof_cell_mass_solved']+s['roof_cell_mass_before']
        aux['terminal_cell_balance_reconstruction_max_abs']=float(np.max(abs(cell-s['roof_cell_balance_defect']))/p0)
        aux['terminal_cell_storage_sum']=float(np.sum(s['roof_cell_mass_solved']-s['roof_cell_mass_before'])/p0)
        aux['interior_face_probability_sum']=float(np.sum(s['midpoint_interior_face_probability'])/p0)
    return s2,s3,s4,aux


def _curves(d):
    dg,s=d['dg'],d['s'];N=dg['roof_density'];Q=dg['no_click_spin_matrix'];tr=np.trace(Q,axis1=1,axis2=2).real
    def divide(a,b):return np.divide(a,b,out=np.full(np.shape(a),np.nan),where=b>0)
    return dict(diagnostic_time=dg['time'],plus_fraction=divide(dg[SECTORS[0]],N),
                minus_fraction=divide(dg[SECTORS[1]],N),neutral_fraction=divide(dg[SECTORS[2]],N),
                undetected_Pz=divide((Q[:,0,0]-Q[:,1,1]).real,tr),
                detector_time=(s['t_left']+s['t_right'])/2,
                detector_Pz=divide(s['p_spin_up']-s['p_spin_down'],s['p_spin_up']+s['p_spin_down']),
                detector_interval_left=s['t_left'],detector_interval_right=s['t_right'])


def _compare_s5(b,r,a,z):
    bs,rs=b['s'],r['s'];bc,rc=b['cfg'],r['cfg'];bt,rt=b['dg']['time'],r['dg']['time']
    D=events(bs['t_left'],bs['t_right'],bs['roof_probability'],a,z,b['p0'])['probability']
    L1=step_abs_difference(bs['t_left'],bs['t_right'],bs['roof_probability'],rs['t_left'],rs['t_right'],rs['roof_probability'],a,z,b['p0'],r['p0'])
    denominator=bc['k_bc']/b['p0']*linear_integral(bt,b['dg']['roof_density'],a,z)
    sector_l1=[linear_abs_difference(bt,b['dg'][key]*bc['k_bc']/b['p0'],rt,r['dg'][key]*rc['k_bc']/r['p0'],a,z) for key in SECTORS]
    row={'baseline':b['case'],'refined':r['case'],'window_start':a,'window_end':z,
         'epsilon_CN_percent':100*L1/D,'epsilon_branch_percent':100*sum(sector_l1)/denominator,
         'CN_baseline_window_probability':D,'branch_baseline_roof_integral':denominator,
         **{key+'_absolute_integral':value for key,value in zip(SECTORS,sector_l1)}}
    if a==0:
        mb=linear_integral(b['t'],b['S'],0,z);mr=linear_integral(r['t'],r['S'],0,z)
        row['epsilon_mu_refined_denom_percent']=100*abs(mr-mb)/abs(mr)
    return row


def _archived_checks(expected,actual,keys,checks,prefix,tolerance=1e-10):
    residuals=[]
    for e,a in zip(expected,actual):
        for key in keys:
            error=float(a[key])-float(e[key]);residuals.append(abs(error))
    _check(checks,prefix,max(residuals,default=0),tolerance)


def run(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    checks=[];rows2=[];rows3=[];rows4=[];aux=[];accounting=[];curves={};cutoffs=[]
    tables=root/'data/centered/tables/diagnostics'
    for omega,case in BASELINE.items():
        d=_load(root,case);accounting.append(_validate(d,checks));s2,s3,s4,a=_stat(d)
        rows2.append(s2);rows3.append(s3);rows4.append(s4);aux.append(a)
        for cut in (3.0,3.1,3.6):
            E=events(d['s']['t_left'],d['s']['t_right'],d['s']['roof_probability'],0,cut,d['p0'])
            L=events(d['s']['t_left'],d['s']['t_right'],d['s']['roof_probability'],cut,40,d['p0'])
            cutoffs.append({'omega':omega,'cutoff':cut,'D_early':E['probability'],'D_late':L['probability'],
                            'early_conditional_mean':E['conditional_mean'],'late_conditional_mean':L['conditional_mean']})
        if omega in (1,300):curves[omega]=_curves(d)
    _archived_checks(_read_csv(tables/'Table_S2_archived.csv')[:8],rows2,list(rows2[0]),checks,'TableS2/native_vs_archived')
    _archived_checks(_read_csv(tables/'Table_S3_archived.csv'),rows3,list(rows3[0]),checks,'TableS3/native_vs_archived')
    _archived_checks(_read_csv(tables/'Table_S4_archived.csv'),rows4,list(rows4[0]),checks,'TableS4/native_vs_archived')
    with np.load(root/'data/centered/native/figures_diagnostics/FigS1_plot_data.npz',allow_pickle=False) as z:
        for omega,curve in curves.items():
            for key,value in curve.items():
                target=z[f'omega{omega}__{key}']
                _check(checks,f'FigS1/{omega}/{key}_undefined_mask',not np.array_equal(np.isnan(value),np.isnan(target)),0)
                finite=np.isfinite(value)&np.isfinite(target)
                _check(checks,f'FigS1/{omega}/{key}_values',np.max(abs(value[finite]-target[finite])) if np.any(finite) else 0,1e-12)
    from centered.s1_render import render
    render(curves,output)
    _csv(output/'Table_S2_spinor_recomputed.csv',rows2)
    _csv(output/'Table_S3_recomputed.csv',rows3)
    _csv(output/'Table_S4_recomputed.csv',rows4)
    _csv(output/'auxiliary_and_clock_distinctions.csv',aux)
    _csv(output/'numerical_accounting.csv',accounting)
    _csv(output/'cutoff_sensitivity_recomputed.csv',cutoffs)
    shutil.copyfile(tables/'scalar_archived.csv',output/'Table_S2_scalar_archived_ONLY.csv')
    baseline=_load(root,7100);refinement_stats=[];pairs=[]
    for case in (7100,7102,7101,7706):
        d=baseline if case==7100 else _load(root,case)
        if case!=7100:_validate(d,checks)
        s2,s3,s4,a=_stat(d)
        c=d['cfg'];refinement_stats.append(dict(case=case,Nx=c['Nx'],Nz=c['Nz'],dt=c['dt'],D=s2['D40'],
            mu=s2['mu_star40'],tcond=s2['tau_cond40'],Pzdet=s4['Pbar_det_z40'],Pznc=s4['P_nc_z40'],
            wplus=s4['wbar_br_plus40'],Es2=s3['E_s2'],beta=s3['beta_bl_omega'],E2br=s3['E_2br']))
        for a,z in ((0.,40.),(8.,40.)):pairs.append(_compare_s5(baseline,d,a,z))
        with np.load(d['path']/'revision_coordinates.npz',allow_pickle=False) as co, np.load(baseline['path']/'revision_coordinates.npz',allow_pickle=False) as bo:
            _check(checks,f'TableS5/{case}/fixed_roof_position',float(co['z'][-1])-float(bo['z'][-1]),0)
            for axis in ('x','y'):_check(checks,f'TableS5/{case}/fixed_{axis}_wall',float(co[axis][-1])-float(bo[axis][-1]),0)
    archive=_read_csv(tables/'Table_S5_three_run_archived.csv')
    lookup={x['case']:x for x in refinement_stats}
    _archived_checks(archive,[lookup[x['case']] for x in archive],list(archive[0]),checks,'TableS5/three_run_native_vs_archived',1e-10)
    original_pairs=_read_csv(tables/'Table_S5_pairwise_archived.csv')
    pair_errors=[]
    for row in original_pairs:
        if row.get('baseline')==7100 and row.get('refined') in (7101,7102) and row.get('b')==40 and row.get('a') in (0,8):
            got=next(x for x in pairs if x['refined']==row['refined'] and x['window_start']==row['a'])
            for key in ('epsilon_CN_percent','epsilon_branch_percent'):
                pair_errors.append(abs(got[key]-row[key]))
            if row['a']==0:pair_errors.append(abs(got['epsilon_mu_refined_denom_percent']-row['epsilon_mu_refined_denom_percent']))
    if not pair_errors:raise ValueError('No matching archived EqS117 pairs found')
    _check(checks,'TableS5/EqS117_vs_archived_pairwise',max(pair_errors),1e-9)
    _csv(output/'Table_S5_native_statistics.csv',refinement_stats)
    _csv(output/'Table_S5_EqS117_discrepancies.csv',pairs)
    # Mask regression: invalid nodes must exclude adjacent intervals rather than bridge.
    masked=_valid_product(np.array([0.,1.,2.,3.]),np.ones(4),np.array([2.,np.nan,2.,2.]),np.array([True,False,True,True]),0,3)
    _check(checks,'auxiliary_validity/no_bridge_duration',masked['excluded_duration']-2,0)
    _check(checks,'auxiliary_validity/no_bridge_mean',masked['E_2br']-2,0)
    result={'checks':checks,'reproduction_levels':{
        'FigS1':'native recomputation checked against archived plot arrays; reconstructed image, not exact embedded-manuscript identity',
        'TablesS2-S4':'spinor native recomputation; scalar archived-table display only',
        'TableS5':'native reconstruction of exact four-run EqS117; archived three-run checks; original final four-run exporter not located'},
        'rounding_discrepancy':{'paper_TableS5_baseline_D':0.739,'native_D':lookup[7100]['D'],'native_rounded_3dp':round(lookup[7100]['D'],3)},
        'printed_precision_discrepancies':[
            {'paper_item':'Table S5(a)','case':case,'quantity':quantity,'paper_displayed':printed,
             'native_value':value,'native_rounded_3dp':round(value,3)}
            for case,quantity,printed,value in (
                (7100,'D_T',0.739,lookup[7100]['D']),
                (7102,'D_T',0.742,lookup[7102]['D']),
                (7101,'epsilon_rel_percent',1.823,next(x['epsilon_mu_refined_denom_percent'] for x in pairs if x['refined']==7101 and x['window_start']==0)))],
        'limitations':['Scalar raw histories and complex spectral/monitor-current records are not bundled.',
                       'Auxiliary continuation is not a physical absorbing layer; exact local matching does not determine full propagated laws.',
                       'Detailed profiles and auxiliary diagnostics remain resolution-sensitive; these checks are data/reduction agreement, not new convergence evidence.'],
        'TableS5':refinement_stats,'TableS5_discrepancies':pairs,
        'check_count':len(checks),'passed':all(c['passed'] for c in checks)}
    result['output_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(output.iterdir()) if p.is_file() and p.suffix in ('.csv','.png','.pdf')}
    (output/'diagnostics_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

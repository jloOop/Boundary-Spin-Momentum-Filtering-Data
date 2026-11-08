#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centered two-exit control 7800: omega=200, T=40, Nz=1500, dt=0.00025.
Upper spinor ABC and supplied reference preparation/discretization are retained.
Only the lower termination changes: d_z psi=-i*k_lower*psi, k_lower=pi.
Physical nodal state uses half longitudinal mass at BOTH endpoint planes.
Top CN events are detector clicks; lower CN events are escape, never top clicks.
1-S is total loss. mu_top_capped integrates 1-D_top; mu_exit integrates S.
Scalar Robin termination is partially reflecting away from its matching wave number.
New source/configuration hashes prohibit continuation from reflecting-bottom runs.
--describe is CPU/import-free; --preflight checks GPU/API discovery.
ABC_PILOT_STEPS permits clean prefixes; no remote submission is performed by this source.
"""



# Configuration is available before importing NumPy or allocating GPU memory.
import sys as _br_sys, os as _br_os, json as _br_json, math as _br_math
BRANCH_REFINEMENT_CASE = {'case_id': 7800, 'refinement': 'centered_Omega200_LowerScalarABC_pi_T40_Nz1500_complex64_dt0.00025', 'Nx': 100, 'Ny': 100, 'Nz': 1500, 'grid_generator_nz': 1500, 'omega': 200.0, 'Lz': 20.0, 'T_final': 40.0, 'dt': 0.00025, 'ell': 10.0, 'z0_init': 10.0, 'sigma_z': 0.5, 'k0': 'pi', 'k_bc': 'pi', 'theta': 0.0, 'phi': 0.0, 'default_diagnostic_dt': 0.02, 'default_complex_trace_dt': 0.05, 'endpoint_closure': 'centered', 'endpoint_mass_weight': 0.5, 'state_dtype': 'complex64', 'matrix_dtype': 'complex64', 'initial_preparation_dtype': 'complex64', 'coordinate_dtype': 'float32', 'potential_sampling_dtype': 'float32', 'state_representation': 'physical_nodal_spinor', 'metric': 'half_weight_first_and_last_z_nodes', 'repair_parent_sha256': '76870569beb66f55ab4a8cbaadefbb6b1e7187eb28baccdc63ce4b79a94adc17', 'uploaded_example_sha256': 'c667a1421f2992d1333dcbc8f506e1dd6cf57b51e7953cbb00c61a0a79aca061', 'campaign_scope': 'Fixed-grid scalar lower-termination control. Same upper spinor ABC and supplied parent parameters. No transparency or convergence claim.', 'immediate_parent_case': 5045, 'immediate_parent_filename': '5045SpinorXY_Gauss_CenteredABC_Omega200_T40.py', 'immediate_parent_sha256': '8a353171eb7eac0a12d2a653b4b79e89930b511aa8a6ac7631f554a71e063625', 'lower_endpoint_mass_weight': 0.5, 'lower_endpoint_closure': 'centered_scalar_ABC', 'k_lower': 'pi', 'lower_boundary_position': 0.0, 'lower_normal': 'minus_z', 'lower_boundary_law': 'd_z_psi=-i*k_lower*psi_componentwise', 'detector_channels': 'top_spinor_detector_and_lower_scalar_escape'}
BRANCH_REFINEMENT_PARENT_SHA256 = '8a353171eb7eac0a12d2a653b4b79e89930b511aa8a6ac7631f554a71e063625'
if _br_sys.argv[1:] == ["--describe"]:
    _br_desc = dict(BRANCH_REFINEMENT_CASE)
    _br_desc.update(total_complex_components=2*BRANCH_REFINEMENT_CASE['Nx']*BRANCH_REFINEMENT_CASE['Ny']*BRANCH_REFINEMENT_CASE['Nz'],
                    planned_steps=_br_math.ceil(_br_math.nextafter(BRANCH_REFINEMENT_CASE['T_final']/BRANCH_REFINEMENT_CASE['dt'], -_br_math.inf)),
                    prompt_stop_step=round(3.6/BRANCH_REFINEMENT_CASE['dt']),
                    nominal_hz=20.0/BRANCH_REFINEMENT_CASE['grid_generator_nz'],
                    nominal_z_detector=(BRANCH_REFINEMENT_CASE['Nz']-1)*20.0/BRANCH_REFINEMENT_CASE['grid_generator_nz'],
                    parent_sha256=BRANCH_REFINEMENT_PARENT_SHA256,
                    failed_step_checkpoint_policy='disabled; resume clean checkpoints only')
    print(_br_json.dumps(_br_desc, indent=2, sort_keys=True))
    raise SystemExit(0)
_BRANCH_PREFLIGHT = _br_sys.argv[1:] == ["--preflight"]
if _br_sys.argv[1:] and not _BRANCH_PREFLIGHT:
    raise SystemExit("Usage: python this_script.py [--describe | --preflight]; runtime controls use environment variables.")
if "OMEGA" in _br_os.environ:
    _br_requested = float(_br_os.environ["OMEGA"])
    if not _br_math.isfinite(_br_requested) or _br_requested != 200.0:
        raise SystemExit("This source is fixed at omega=200. Unset OMEGA or set OMEGA=200.")

import sys, time, os, logging, json
from pathlib import Path
import numpy as np
def TRAPZ(y, x):
    # consistent trapezoidal integration; returns float
    f = getattr(np, "trapezoid", None)
    return float(f(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)) if f is not None else np.trapz(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)))
import cupy as cp
from cupyx.scipy.sparse import diags, eye, kron, coo_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import gmres, LinearOperator

ORIGINAL_SOURCE_SHA256 = '8a353171eb7eac0a12d2a653b4b79e89930b511aa8a6ac7631f554a71e063625'
# ---------------- BEGIN ADDITIVE REVISION SUPPORT v1 ----------------
# This block is embedded so each updated solver is independently runnable.
import csv as _csv
import hashlib as _hashlib
import inspect as _inspect
import math as _math
import signal as _signal
import shutil as _shutil
import platform as _platform
from datetime import datetime as _datetime, timezone as _timezone

_ABC_VERSION = 'Cartesian_7800_centered_LowerScalarABC_Omega200_T40_v1_20261021'
_ABC_STEP_FIELDS = ('step','t_left','t_right','roof_probability','mass_before','mass_solved',
 'mass_stored','cast_mass_loss','clamp_mass_loss','solve_and_matrix_defect','relative_residual',
 'p_spin_up','p_spin_down','p_spin_offdiag_real','p_spin_offdiag_imag','step_wall_seconds',
 'roof_cell_mass_before','roof_cell_mass_solved','midpoint_interior_face_probability','roof_cell_balance_defect',
 'lower_escape_probability','lower_cell_mass_before','lower_cell_mass_solved',
 'lower_midpoint_interior_face_probability','lower_cell_balance_defect',
 'lower_spin_up_probability','lower_spin_down_probability','lower_spin_offdiag_real','lower_spin_offdiag_imag',
 'D_top','E_lower','S','M1_top','M1_lower','mu_top_capped','mu_exit',
 'two_channel_budget_residual','mu_top_identity_residual','mu_exit_identity_residual')
_ABC_STEP_INDEX = {name: i for i, name in enumerate(_ABC_STEP_FIELDS)}
_ABC_CUMULATIVE_FIELDS = ('D_top','E_lower','S','M1_top','M1_lower','mu_top_capped','mu_exit',
 'two_channel_budget_residual','mu_top_identity_residual','mu_exit_identity_residual')

ENDPOINT_MASS_WEIGHT = BRANCH_REFINEMENT_CASE['endpoint_mass_weight']
LOWER_ENDPOINT_MASS_WEIGHT = BRANCH_REFINEMENT_CASE['lower_endpoint_mass_weight']
ENDPOINT_CLOSURE = BRANCH_REFINEMENT_CASE['endpoint_closure']
STATE_DTYPE = getattr(cp, BRANCH_REFINEMENT_CASE['state_dtype'])
ASSEMBLY_DTYPE = getattr(cp, BRANCH_REFINEMENT_CASE['matrix_dtype'])


def _abc_clean(x):
    if isinstance(x,dict): return {str(k):_abc_clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_abc_clean(v) for v in x]
    if isinstance(x,np.ndarray):return _abc_clean(x.tolist())
    if isinstance(x,np.generic):return _abc_clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x


def _abc_json(path,obj):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(_abc_clean(obj),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def _abc_npy(path,array):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def _abc_npz(path,**arrays):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def _abc_sha(path):
    h=_hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _abc_float_env(name,default,allow_zero=False):
    v=float(os.environ.get(name,default))
    if not np.isfinite(v) or (v<0 if allow_zero else v<=0):raise ValueError(f'{name} must be finite and positive')
    return v


def _abc_output(parent,legacy_label,script_id,default_omega):
    """Never write into an old results directory. Atomic ownership claim."""
    omega_tag=float(os.getenv('OMEGA',str(default_omega)))
    if not np.isfinite(omega_tag) or omega_tag<=0:raise ValueError('OMEGA must be finite and positive')
    tag=os.getenv('ABC_RUN_TAG','')
    import re as _re
    if tag and not _re.fullmatch(r'[A-Za-z0-9_.-]+',tag):raise ValueError('ABC_RUN_TAG: use letters/numbers/._- only')
    name=f"{script_id}.Spin_Gauss_centeredABC_LowerScalarABC_Omega={omega_tag:g}_L=20_k0=pi_kappa=pi_kappaLower=pi_sigma=0.5_T=40_Nz={BRANCH_REFINEMENT_CASE['Nz']}_complex64_LowerABC_v1"
    if tag:name+='_'+tag
    base=Path(parent).expanduser();base.mkdir(parents=True,exist_ok=True)
    dest=base/name
    dest.mkdir(exist_ok=False)  # existing output, even empty, is NOT silently reused
    (dest/'REVISION_OWNER.txt').write_text(_ABC_VERSION+'\n'+str(os.getpid())+'\n')
    return dest


def _abc_gpu_record():
    d={'numpy':np.__version__,'python':_platform.python_version(),'cupy':getattr(cp,'__version__','unknown')}
    try:
        i=int(cp.cuda.Device().id);p=cp.cuda.runtime.getDeviceProperties(i)
        name=p.get('name','unknown');name=name.decode() if isinstance(name,bytes) else name
        d.update(gpu_name=name,total_device_memory_bytes=int(p.get('totalGlobalMem',0)),device_index=i,
                 cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion())
    except Exception as exc:d['gpu_query_error']=str(exc)
    return d


def _abc_norm(v,dV):
    """Physical probability: both centered-ABC endpoint nodes have weight 1/2.
    Sum in complex128 without allocating a full-volume weight array.
    This also accepts any number of z-lines in tests; production has 2*Nx*Ny.
    """
    v128=v.astype(cp.complex128,copy=False)
    total=cp.vdot(v128,v128).real
    if ENDPOINT_MASS_WEIGHT != 1.0:
        roof=v128.reshape(-1,Nz)[:,-1].ravel()
        total=total-(1.0-ENDPOINT_MASS_WEIGHT)*cp.vdot(roof,roof).real
    if LOWER_ENDPOINT_MASS_WEIGHT != 1.0:
        lower=v128.reshape(-1,Nz)[:,0].ravel()
        total=total-(1.0-LOWER_ENDPOINT_MASS_WEIGHT)*cp.vdot(lower,lower).real
    return float(cp.asnumpy(total))*dV


def _abc_mass_native(up,down,dV):
    """Same physical quadrature, with the parent's native-dtype reduction convention."""
    total=cp.sum(cp.abs(up)**2+cp.abs(down)**2)
    if ENDPOINT_MASS_WEIGHT != 1.0:
        total=total-(1.0-ENDPOINT_MASS_WEIGHT)*cp.sum(cp.abs(up[:,:,-1])**2+cp.abs(down[:,:,-1])**2)
    if LOWER_ENDPOINT_MASS_WEIGHT != 1.0:
        total=total-(1.0-LOWER_ENDPOINT_MASS_WEIGHT)*cp.sum(cp.abs(up[:,:,0])**2+cp.abs(down[:,:,0])**2)
    return total*dV


def _abc_spin_matrix(u,d,weight):
    u=u.astype(cp.complex128,copy=False).ravel();d=d.astype(cp.complex128,copy=False).ravel()
    uu=float(cp.asnumpy(cp.vdot(u,u).real))*weight
    dd=float(cp.asnumpy(cp.vdot(d,d).real))*weight
    ud=complex(cp.asnumpy(cp.vdot(d,u)))*weight
    return np.array([[uu,ud],[ud.conjugate(),dd]],dtype=np.complex128)


def _abc_no_click_spin_matrix(up,down,dV):
    """Spin matrix of particles still inside both exits, with the survival metric.
    The legacy no_click name does not include particles already lost through the lower exit.
    """
    rho=_abc_spin_matrix(up,down,dV)
    if ENDPOINT_MASS_WEIGHT != 1.0:
        rho=rho-(1.0-ENDPOINT_MASS_WEIGHT)*_abc_spin_matrix(up[:,:,-1],down[:,:,-1],dV)
    if LOWER_ENDPOINT_MASS_WEIGHT != 1.0:
        rho=rho-(1.0-LOWER_ENDPOINT_MASS_WEIGHT)*_abc_spin_matrix(up[:,:,0],down[:,:,0],dV)
    return rho


def _abc_linear_interval(t,y,a,b):
    # Diagnostic quadrature only. Never changes the propagated field.
    t=np.asarray(t,float);y=np.asarray(y,float)
    if not np.isfinite(a) or not np.isfinite(b) or b<=a:return np.nan
    if len(t)!=len(y) or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)) or np.any(np.diff(t)<=0):
        raise ValueError('invalid time series for interval integration')
    a=max(float(a),float(t[0]));b=min(float(b),float(t[-1]))
    if b<=a:return 0.
    x=np.r_[a,t[(t>a)&(t<b)],b];v=np.interp(x,t,y)
    f=getattr(np,'trapezoid',None)
    return float(f(v,x) if f is not None else np.trapz(v,x))


class _ABCRevision:
    """Passive numerical/field diagnostics and atomic restart snapshots.

    Uses the two-endpoint metric and the parent CN precision. Top detector and
    scalar lower-escape probabilities are separate positive midpoint events.
    No renormalization, probability clipping, or bath is introduced. Cast loss,
    lateral clamp loss and solver/rounded-matrix defects remain separate.
    """
    def __init__(self,G,source_parent_sha):
        self.G=G;self.out=G['out_dir'];self.start_wall=time.monotonic();self.last_checkpoint=self.start_wall
        self.dt=float(G['dt']);self.T=float(G['T_final']);self.dV=G['hx']*G['hy']*G['hz']
        self.source=Path(__file__).resolve();self.source_sha=_abc_sha(self.source)
        sig=_inspect.signature(gmres)
        if 'rtol' not in sig.parameters:
            raise RuntimeError('This environment does not provide the original rtol GMRES API. Use the same working environment; no automatic library replacement is performed.')
        self.checkpoint_seconds=_abc_float_env('ABC_CHECKPOINT_SECONDS',10800)
        self.stop_seconds=_abc_float_env('ABC_STOP_AFTER_SECONDS',0,True)
        self.trace_dt=_abc_float_env('ABC_TRACE_DT',.05)
        self.diag_dt=_abc_float_env('ABC_DIAG_DT',.02)
        self.pilot_steps=int(os.getenv('ABC_PILOT_STEPS','0'))
        if self.pilot_steps<0:raise ValueError('ABC_PILOT_STEPS must be nonnegative')
        self.stop_signal=None
        for name in ('SIGTERM','SIGINT','SIGUSR1'):
            if hasattr(_signal,name):_signal.signal(getattr(_signal,name),self._signal)
        self.steps=[];self.diag=[];self.step=0;self.t=0.;self.p0=_abc_norm(G['psi_flat'],self.dV)
        self.p0_native=float(G['P0_init'])
        self.diag_next=0.;self.trace_next=0.;self.physical_checkpoints=[3.6,10.,20.,30.,40.]
        self.last_state_checkpoint=None;self.resumed_from=None
        config={k:G[k] for k in ('Nx','Ny','Nz','omega','ell','Lx','Ly','Lz','dt','T_final','hx','hy','hz','k_bc','k_lower','z0_init','sigma_z','k0','theta','phi')}
        config.update(diagnostic_dt=self.diag_dt,complex_trace_dt=self.trace_dt,real_dtype='float32',state_dtype=BRANCH_REFINEMENT_CASE['state_dtype'],matrix_assembly=BRANCH_REFINEMENT_CASE['matrix_dtype']+'_then_solve_complex128',
            gmres_rtol=1e-8,gmres_atol=0.,gmres_restart=30,gmres_maxiter=1000,
            monitor_every=int(G['monitor_every']),prob_every=int(G['prob_every']),
            z_roof_actual=float(cp.asnumpy(G['z'][-1])),
            boundary=ENDPOINT_CLOSURE+' scalar dissipative roof + matched Hermitian spin-tangential coupling',
            detector='top spinor ABC only; scalar lower ABC is a separate escape channel',
            lower_boundary='d_z psi=-i*k_lower*psi componentwise at z=0',
            lower_boundary_position=float(cp.asnumpy(G['z'][0])),
            lower_endpoint_mass_weight=LOWER_ENDPOINT_MASS_WEIGHT,
            lower_endpoint_closure=BRANCH_REFINEMENT_CASE['lower_endpoint_closure'],
            lower_monitor_target_z=[1.0,2.0],
            survivor_spin_conditioning='still inside the guide: neither top click nor lower escape')
        config.update(grid_generator_nz=BRANCH_REFINEMENT_CASE['grid_generator_nz'],
                      normal_grid_test=BRANCH_REFINEMENT_CASE['refinement'],
                      failure_checkpoint_policy='disabled_clean_checkpoints_only',
                      endpoint_mass_weight=ENDPOINT_MASS_WEIGHT,endpoint_closure=ENDPOINT_CLOSURE,
                      metric=BRANCH_REFINEMENT_CASE['metric'],state_representation='physical_nodal_spinor',
                      initial_preparation_dtype='complex64',potential_sampling_dtype='float32',
                      repair_parent_sha256=BRANCH_REFINEMENT_CASE['repair_parent_sha256'],
                      step_columns=list(_ABC_STEP_FIELDS))
        self.config=_abc_clean(config)
        # Changing T or diagnostics is not accepted during a restart of this release.
        self.fingerprint=_hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest()
        self.record=dict(version=_ABC_VERSION,configuration=self.config,source_parent_sha256=source_parent_sha,
            updated_source_sha256=self.source_sha,configuration_sha256=self.fingerprint,
            P0_init64=self.p0,P0_init_native=self.p0_native,status='RUNNING_NOT_VALIDATED',
            environment=_abc_gpu_record(),trace_dt=self.trace_dt,diagnostic_dt=self.diag_dt,
            endpoint_policy='base dt unchanged; a single shorter final Cayley step if needed to reach the requested T exactly',
             event_interpretation='top=detector click; lower=escape; 1-S=total exit loss plus numerical defects',
             mean_interpretation='mu_top_capped=integral(1-D_top); mu_exit=integral(S)',
             legacy_prob_loss_interpretation='raw total norm loss, NOT top detection probability',
             legacy_no_click_fields='remaining state inside the guide, conditioned on neither exit',
             lower_termination='passive scalar Robin boundary; not transparent to arbitrary wave numbers',
            numerical_guards={'true_relative_residual_max':2e-8,'cumulative_two_channel_closure_abs_max':1e-3,
                              'meaning':'fail-stop safety, not a claim of accuracy at these thresholds'},
            optional_pilot_steps=self.pilot_steps)
        (self.out/'complex_traces').mkdir()
        (self.out/'checkpoints').mkdir()
        _shutil.copyfile(self.source,self.out/'source_snapshot.py')
        resume=os.getenv('ABC_RESUME_FROM','').strip()
        if resume:self._restore(Path(resume).expanduser().resolve())
        _abc_json(self.out/'revision_run.json',self.record)
        self.stream=(self.out/'revision_step_data.csv').open('w',newline='',encoding='utf-8')
        self.writer=_csv.writer(self.stream);self.writer.writerow(_ABC_STEP_FIELDS)
        self.writer.writerows(self.steps);self.stream.flush()
        self._prepare_fourier()
        zcpu=cp.asnumpy(G['z']).astype(float)
        link_z=0.5*(zcpu[:-1]+zcpu[1:])
        self.lower_link_indices=np.asarray([int(np.argmin(abs(link_z-target))) for target in (1.0,2.0)])
        self.lower_link_positions=link_z[self.lower_link_indices]
        self.record['lower_monitor_actual_link_z']=self.lower_link_positions.tolist()
        _abc_json(self.out/'revision_run.json',self.record)
        if self.step==0:self.diagnostic(G['psi_flat'],0.,0,force=True)

    def _signal(self,number,frame):
        self.stop_signal=int(number)  # checked after a COMPLETED linear step only

    def _prepare_fourier(self):
        G=self.G;nx,ny=G['Nx'],G['Ny'];hx,hy=G['hx'],G['hy']
        kx=2*np.pi*np.fft.fftfreq(nx,d=hx);ky=2*np.pi*np.fft.fftfreq(ny,d=hy)
        self.xihx=np.sin(kx*hx)[:,None]/hx;self.xihy=np.sin(ky*hy)[None,:]/hy
        self.R=np.sqrt(self.xihx**2+self.xihy**2)
        self.rzero_tol=64*np.finfo(float).eps*max(1.,float(self.R.max()))
        self.neutral=self.R<=self.rzero_tol
        self.off=-self.xihy-1j*self.xihx

    def _restore(self,p):
        if not (p/'checkpoint.json').is_file():raise ValueError('ABC_RESUME_FROM must name a completed checkpoint directory')
        j=json.loads((p/'checkpoint.json').read_text())
        if j.get('configuration_sha256')!=self.fingerprint or j.get('updated_source_sha256')!=self.source_sha:
            raise ValueError('restart source/configuration mismatch; no silent continuation across changed algorithms')
        for key in ('numpy','cupy'):
            if j.get('environment',{}).get(key)!=self.record['environment'].get(key):
                raise ValueError('restart numerical-library version mismatch: '+key)
        for fn,expected in j['files'].items():
            if _abc_sha(p/fn)!=expected:raise ValueError('restart hash mismatch: '+fn)
        state=np.load(p/'psi_flat.npy',allow_pickle=False)
        if state.dtype!=np.dtype(BRANCH_REFINEMENT_CASE['state_dtype']) or state.shape!=(2*self.G['Nx']*self.G['Ny']*self.G['Nz'],) or not np.all(np.isfinite(state)):
            raise ValueError('invalid restart state')
        self.G['psi_flat']=cp.asarray(state);del state
        with np.load(p/'histories.npz',allow_pickle=False) as z:
            self.G['prob_t']=z['prob_t'].tolist();self.G['prob_val']=z['prob_val'].tolist()
            self.G['t_series']=z['t_series'].tolist();self.G['P_series']=z['P_series'].tolist()
            for k,d in self.G['plane_data'].items():
                for f,v in list(d.items()):
                    if isinstance(v,list) and k+'__'+f in z:d[f]=z[k+'__'+f].tolist()
            for k,d in self.G['roof_data'].items():
                for f,v in list(d.items()):
                    if isinstance(v,list) and k+'__'+f in z:d[f]=z[k+'__'+f].tolist()
            self.steps=z['step_rows'].tolist()
            if len(self.steps) and len(self.steps[0])!=len(_ABC_STEP_FIELDS):raise ValueError('restart step schema mismatch')
            names=z['diagnostic_keys'].tolist()
            for i in range(len(z['diagnostic_times'])):
                self.diag.append({k:z['diag__'+k][i].item() if z['diag__'+k][i].shape==() else z['diag__'+k][i] for k in names})
        self.step=int(j['step']);self.t=float(j['time']);self.p0=float(j['P0_init64']);self.p0_native=float(j['P0_init_native'])
        self.G['P0_init']=self.p0_native
        if len(self.steps)!=self.step or (self.step and int(self.steps[-1][0])!=self.step):raise ValueError('restart histories not complete')
        if abs(_abc_norm(self.G['psi_flat'],self.dV)-j['state_mass64'])>2e-12*max(1.,abs(j['state_mass64'])):raise ValueError('restart mass mismatch')
        self.trace_next=(np.floor(self.t/self.trace_dt)+1)*self.trace_dt
        self.diag_next=(np.floor(self.t/self.diag_dt)+1)*self.diag_dt
        self.physical_checkpoints=[x for x in self.physical_checkpoints if x>self.t+1e-12]
        self.resumed_from=str(p)
        self.record.update(resumed_from=str(p),restart_step=self.step,P0_init64=self.p0,P0_init_native=self.p0_native)
        # Diagnostics' earlier complex traces remain referenced in the parent, not silently duplicated.
        self.record['parent_trace_directory']=str(p.parent.parent/'complex_traces')
        self.record['parent_trace_handling']='prefix complex snapshots retained in parent; scalar/spin/branch histories copied into this run'
        _abc_json(self.out/'restart_provenance.json',j)

    def time_step(self,n):
        # All full steps reproduce the original dt grid; only terminal remainder differs.
        left=(n-1)*self.dt;right=min(n*self.dt,self.T)
        return float(left),float(right),float(right-left)

    def begin(self,old):
        self.step_start=time.monotonic()
        return _abc_norm(old,self.dV)

    def record_step(self,n,left,right,old,solved,stored,rhs,Astep,mass_before,mass_cast):
        G=self.G;step_dt=right-left
        residual=Astep@solved-rhs
        rr=float(cp.asnumpy(cp.linalg.norm(residual)))/max(float(cp.asnumpy(cp.linalg.norm(rhs))),np.finfo(float).tiny)
        mass_solved=_abc_norm(solved,self.dV);mass_stored=_abc_norm(stored,self.dV)
        old4=old.reshape(2,G['Nx'],G['Ny'],G['Nz'])
        new4=solved.reshape(2,G['Nx'],G['Ny'],G['Nz'])
        oldroof=old4[:,:,:,-1].astype(cp.complex128);newroof=new4[:,:,:,-1]
        q=(oldroof+newroof)/2
        pmat=_abc_spin_matrix(q[0],q[1],step_dt*G['k_bc']*G['hx']*G['hy'])
        pd=float(np.trace(pmat).real)
        qinside=(old4[:,:,:,-2].astype(cp.complex128)+new4[:,:,:,-2])/2
        cell_before=ENDPOINT_MASS_WEIGHT*self.dV*float(cp.asnumpy(cp.vdot(oldroof.ravel(),oldroof.ravel()).real))
        cell_solved=ENDPOINT_MASS_WEIGHT*self.dV*float(cp.asnumpy(cp.vdot(newroof.ravel(),newroof.ravel()).real))
        face_probability=step_dt*G['hx']*G['hy']/G['hz']*float(cp.asnumpy(cp.vdot(qinside.ravel(),q.ravel()).imag))
        cell_defect=face_probability-pd-(cell_solved-cell_before)

        oldlower=old4[:,:,:,0].astype(cp.complex128);newlower=new4[:,:,:,0]
        qlower=(oldlower+newlower)/2
        emat=_abc_spin_matrix(qlower[0],qlower[1],step_dt*G['k_lower']*G['hx']*G['hy'])
        pe=float(np.trace(emat).real)
        qlowerinside=(old4[:,:,:,1].astype(cp.complex128)+new4[:,:,:,1])/2
        lower_before=LOWER_ENDPOINT_MASS_WEIGHT*self.dV*float(cp.asnumpy(cp.vdot(oldlower.ravel(),oldlower.ravel()).real))
        lower_solved=LOWER_ENDPOINT_MASS_WEIGHT*self.dV*float(cp.asnumpy(cp.vdot(newlower.ravel(),newlower.ravel()).real))
        # Positive into the lower endpoint cell, i.e. toward -z.
        lower_face=step_dt*G['hx']*G['hy']/G['hz']*float(cp.asnumpy(cp.vdot(qlowerinside.ravel(),qlower.ravel()).imag))
        lower_defect=lower_face-pe-(lower_solved-lower_before)
        checked=(rr,mass_before,mass_solved,mass_cast,mass_stored,pd,pe,
                 cell_before,cell_solved,face_probability,cell_defect,
                 lower_before,lower_solved,lower_face,lower_defect)
        if not all(np.isfinite(v) for v in checked) or pd<0 or pe<0:
            raise RuntimeError(f'nonfinite/negative two-channel quantity at step {n}')
        if rr>2e-8:raise RuntimeError(f'true relative GMRES residual {rr:.6g} exceeds 2e-8 at step {n}')
        cast_loss=mass_solved-mass_cast;clamp_loss=mass_cast-mass_stored
        defect=mass_before-mass_solved-pd-pe
        if self.steps:
            prev={k:float(self.steps[-1][_ABC_STEP_INDEX[k]]) for k in _ABC_CUMULATIVE_FIELDS}
        else:
            prev={k:0.0 for k in _ABC_CUMULATIVE_FIELDS};prev['S']=1.0
        dtop=prev['D_top']+pd/self.p0
        elower=prev['E_lower']+pe/self.p0
        survival=mass_stored/self.p0
        midpoint=0.5*(left+right)
        mtop=prev['M1_top']+midpoint*pd/self.p0
        mlower=prev['M1_lower']+midpoint*pe/self.p0
        mu_top=prev['mu_top_capped']+step_dt*(1.0-0.5*(prev['D_top']+dtop))
        # Independent trapezoidal integration of the every-step stored survival.
        mu_exit=prev['mu_exit']+0.5*step_dt*(mass_before+mass_stored)/self.p0
        budget=1.0-survival-dtop-elower
        mu_top_residual=mu_top-(mtop+right*(1.0-dtop))
        mu_exit_residual=mu_exit-(mtop+mlower+right*survival)
        if abs(budget)>1e-3:
            raise RuntimeError(f'cumulative two-channel/norm discrepancy {budget:.6g} exceeds operational guard; no normalization applied')
        row=[int(n),left,right,pd,mass_before,mass_solved,mass_stored,cast_loss,clamp_loss,defect,rr,
             float(pmat[0,0].real),float(pmat[1,1].real),float(pmat[0,1].real),float(pmat[0,1].imag),time.monotonic()-self.step_start,
             cell_before,cell_solved,face_probability,cell_defect,
             pe,lower_before,lower_solved,lower_face,lower_defect,
             float(emat[0,0].real),float(emat[1,1].real),float(emat[0,1].real),float(emat[0,1].imag),
             dtop,elower,survival,mtop,mlower,mu_top,mu_exit,budget,mu_top_residual,mu_exit_residual]
        if len(row)!=len(_ABC_STEP_FIELDS):raise RuntimeError('two-channel step schema mismatch')
        self.steps.append(row);self.writer.writerow(row);self.step=int(n);self.t=float(right)
        if n%200==0:self.stream.flush()
        if n%400==0:
            sample=np.asarray(self.steps[-min(400,len(self.steps)):],float)[:,15]
            sec=float(np.median(sample));remaining=(self.T-right)/self.dt*sec
            _abc_json(self.out/'performance_estimate.json',dict(step=n,time=right,recent_median_step_seconds=sec,
                recent_p90_step_seconds=float(np.quantile(sample,.9)),remaining_step_only_hours=remaining/3600,
                projected_step_only_total_hours=(right/self.dt*sec+remaining)/3600,
                warning='excludes future iteration changes, checkpoint/diagnostic I/O, queue time; startup is separate'))
        return rr,mass_stored,pd

    def channel_arrays(self):
        """Every accepted endpoint, normalized once by the double-precision launch mass."""
        rows=np.asarray(self.steps,dtype=np.float64).reshape((-1,len(_ABC_STEP_FIELDS)))
        arrays={'time':np.r_[0.0,rows[:,_ABC_STEP_INDEX['t_right']]]}
        for key in _ABC_CUMULATIVE_FIELDS:
            arrays[key]=np.r_[1.0 if key=='S' else 0.0,rows[:,_ABC_STEP_INDEX[key]]]
        arrays['event_t_left']=rows[:,_ABC_STEP_INDEX['t_left']]
        arrays['event_t_right']=rows[:,_ABC_STEP_INDEX['t_right']]
        duration=arrays['event_t_right']-arrays['event_t_left']
        arrays['g_top']=rows[:,_ABC_STEP_INDEX['roof_probability']]/(self.p0*duration)
        arrays['g_lower']=rows[:,_ABC_STEP_INDEX['lower_escape_probability']]/(self.p0*duration)
        arrays['P0_init64']=np.float64(self.p0)
        return arrays

    def channel_summary(self):
        arrays=self.channel_arrays()
        result={k:float(arrays[k][-1]) for k in ('time',)+_ABC_CUMULATIVE_FIELDS}
        result.update(
            top_conditional_mean=(result['M1_top']/result['D_top'] if result['D_top']>0 else None),
            lower_conditional_mean=(result['M1_lower']/result['E_lower'] if result['E_lower']>0 else None),
            max_abs_two_channel_budget_residual=float(np.max(abs(arrays['two_channel_budget_residual']))),
            max_abs_mu_top_identity_residual=float(np.max(abs(arrays['mu_top_identity_residual']))),
            max_abs_mu_exit_identity_residual=float(np.max(abs(arrays['mu_exit_identity_residual']))),
            probability_semantics='D_top=top detector; E_lower=lower escape; S=still inside both channels',
            mean_semantics='mu_top_capped=integral(1-D_top), lower escapes assigned no top click; mu_exit=integral(S)',
            integration='CN event rates constant on accepted intervals; survival linearly interpolated on every accepted endpoint')
        return result

    def diagnostic(self,state,t,n,force=False):
        G=self.G;save_diag=force or t+1e-12>=self.diag_next;save_trace=force or t+1e-12>=self.trace_next
        if not(save_diag or save_trace):return
        u=state.reshape(2,G['Nx'],G['Ny'],G['Nz'])
        roof=cp.asnumpy(u[:,:,:,-1]);inside=cp.asnumpy(u[:,:,:,-2])
        if save_trace:
            _abc_npz(self.out/'complex_traces'/f'trace_step{n:09d}.npz',step=np.int64(n),time=np.float64(t),
                     roof=roof,one_grid_inside=inside,hx=np.float64(G['hx']),hy=np.float64(G['hy']),hz=np.float64(G['hz']))
            self.trace_next=(np.floor(t/self.trace_dt)+1)*self.trace_dt
        if not save_diag:return
        wr=np.fft.fft2(roof.astype(np.complex128),axes=(-2,-1),norm='ortho')
        wi=np.fft.fft2(inside.astype(np.complex128),axes=(-2,-1),norm='ortho')
        JW=np.stack((self.off*wr[1],self.off.conjugate()*wr[0]))
        gammaW=np.zeros_like(wr);nz=~self.neutral;gammaW[:,nz]=JW[:,nz]/self.R[nz]
        plus=(wr+gammaW)/2;minus=(wr-gammaW)/2
        weight=G['hx']*G['hy'];power=np.sum(abs(wr)**2,axis=0)
        z=float(np.sum(power)*weight)
        qp=float(np.sum(abs(plus[:,nz])**2)*weight);qm=float(np.sum(abs(minus[:,nz])**2)*weight)
        q0=float(np.sum(power[self.neutral])*weight)
        r2=float(np.sum(power*self.R**2)*weight)
        ra=float(-2*np.real(np.sum(wr.conjugate()*JW))*weight)
        # One-grid local-stencil continuation (NOT an exact diagonalization of finite Dirichlet walls).
        factor=np.zeros_like(self.R);factor[nz]=np.sinh(G['hz']*self.R[nz])/self.R[nz]
        hom=np.exp(-1j*G['k_bc']*G['hz'])*(np.cosh(G['hz']*self.R)[None,:,:]*wr-factor[None,:,:]*JW)
        den=float(np.linalg.norm(wi));e2=float(np.linalg.norm(wi-hom)/den) if den>0 else np.nan
        rho_roof=np.einsum('axy,bxy->ab',roof.astype(np.complex128),roof.astype(np.complex128).conjugate())*weight
        rho_nc=_abc_no_click_spin_matrix(u[0],u[1],self.dV)
        volume_mass=_abc_norm(state,self.dV)
        trace_defect=float(np.trace(rho_nc).real)-volume_mass
        if abs(trace_defect)>5e-12*max(1.,volume_mass):
            raise RuntimeError('Survivor spin trace and physical mass differ')
        lower_spin=_abc_spin_matrix(u[0,:,:,0],u[1,:,:,0],weight)
        lower_density=float(np.trace(lower_spin).real)
        lower_links=[]
        for link in self.lower_link_indices:
            lo=u[:,:,:,int(link)].astype(cp.complex128);hi=u[:,:,:,int(link)+1].astype(cp.complex128)
            lower_links.append(float(cp.asnumpy(cp.vdot(lo.ravel(),hi.ravel()).imag))*weight/(G['hz']*self.p0))
        entry=dict(time=float(t),step=int(n),roof_density=z,branch_plus_density=qp,branch_minus_density=qm,
            neutral_density=q0,branch_sum_error=qp+qm+q0-z,roof_R2_numerator=r2,
            roof_Ra_numerator=ra,one_grid_E2br=e2,one_grid_valid=bool(den>0),
            roof_spin_matrix=rho_roof,no_click_spin_matrix=rho_nc,
            no_click_mass64=volume_mass,no_click_trace_mass_defect=trace_defect,
            terminal_cell_mass=ENDPOINT_MASS_WEIGHT*G['hz']*z,
            lower_density=lower_density,lower_spin_matrix=lower_spin,
            lower_escape_rate=G['k_lower']*lower_density/self.p0,
            lower_terminal_cell_mass=LOWER_ENDPOINT_MASS_WEIGHT*G['hz']*lower_density,
            lower_monitor_link_z=self.lower_link_positions.copy(),
            lower_monitor_signed_link_current=np.asarray(lower_links,float),
            discrete_symbol_zero_tolerance=self.rzero_tol)
        if abs(entry['branch_sum_error'])>1e-11*max(1.,z):raise RuntimeError('Fourier branch partition failed')
        if self.diag and self.diag[-1]['step']==int(n):
            self.diag[-1]=entry
        else:
            self.diag.append(entry)
        self.diag_next=(np.floor(t/self.diag_dt)+1)*self.diag_dt

    def flush_native(self):
        G=self.G
        _abc_npy(self.out/'prob_times.npy',np.asarray(G['prob_t'],float));_abc_npy(self.out/'total_probs.npy',np.asarray(G['prob_val'],float))
        # Legacy probability-loss filename still means TOTAL raw norm loss, not top clicks.
        _abc_npy(self.out/'prob_loss.npy',self.p0_native-np.asarray(G['prob_val'],float))
        channels=self.channel_arrays()
        _abc_npz(self.out/'two_channel_history.npz',**channels)
        _abc_npy(self.out/'two_channel_times.npy',channels['time'])
        for key in _ABC_CUMULATIVE_FIELDS:_abc_npy(self.out/(key+'.npy'),channels[key])
        csv_path=self.out/'two_channel_history.csv';csv_tmp=csv_path.with_name(csv_path.name+'.tmp')
        with csv_tmp.open('w',newline='',encoding='utf-8') as f:
            writer=_csv.writer(f);writer.writerow(('time',)+_ABC_CUMULATIVE_FIELDS)
            writer.writerows(zip(channels['time'],*(channels[key] for key in _ABC_CUMULATIVE_FIELDS)))
            f.flush();os.fsync(f.fileno())
        os.replace(csv_tmp,csv_path)
        _abc_json(self.out/'two_channel_summary.json',self.channel_summary())
        _abc_npy(self.out/'plane_times.npy',np.asarray(G['t_series'],float));_abc_npy(self.out/'P_total_on_plane_times.npy',np.asarray(G['P_series'],float))
        for group in (G['plane_data'],G['roof_data']):
            for key,d in group.items():
                for f,v in d.items():
                    if isinstance(v,list) and f in ('Jnet','Jnet_conv','Jnet_spin','Jplus_loc','Jminus_loc','Jplus_loc_conv','Jminus_loc_conv','rho_int'):
                        _abc_npy(self.out/f'{key}_{f}.npy',np.asarray(v,float))
        self.stream.flush();os.fsync(self.stream.fileno())
        if self.diag:
            _abc_npz(self.out/'revision_spin_branch_diagnostics.npz',**{k:np.asarray([d[k] for d in self.diag]) for k in self.diag[0]})
        self.record.update(completed_step=self.step,actual_time=self.t,P0_init64=self.p0,P0_init_native=self.p0_native,
            current_accepted_prefix_mass64=float(self.steps[-1][6]) if self.steps else self.p0,
            cumulative_roof_probability=_math.fsum(row[3] for row in self.steps),
            cumulative_lower_escape_probability=_math.fsum(row[_ABC_STEP_INDEX['lower_escape_probability']] for row in self.steps),
            two_channel_statistics=self.channel_summary(),
            elapsed_this_invocation_seconds=time.monotonic()-self.start_wall)
        _abc_json(self.out/'revision_run.json',self.record)

    def checkpoint(self,state,label):
        # Validate the in-memory accepted prefix before writing a clean checkpoint.
        if len(self.steps)!=self.step or (self.step and int(self.steps[-1][0])!=self.step):
            raise RuntimeError('Refusing checkpoint: state/history step mismatch')
        if self.step and abs(float(self.steps[-1][2])-self.t)>1e-11:
            raise RuntimeError('Refusing checkpoint: history time mismatch')
        expected=float(self.steps[-1][6]) if self.step else self.p0
        if abs(_abc_norm(state,self.dV)-expected)>5e-12*max(1.,abs(expected)):
            raise RuntimeError('Refusing checkpoint: state mass differs from accepted history')
        self.diagnostic(state,self.t,self.step,force=True)
        self.flush_native();G=self.G
        target=self.out/'checkpoints'/f'checkpoint_step{self.step:09d}'
        if target.exists():
            self.last_state_checkpoint=str(target);return target
        temp=target.with_name(target.name+'.building');temp.mkdir(exist_ok=False)
        _abc_npy(temp/'psi_flat.npy',cp.asnumpy(state))
        arrays={'prob_t':np.asarray(G['prob_t']), 'prob_val':np.asarray(G['prob_val']),
                't_series':np.asarray(G['t_series']),'P_series':np.asarray(G['P_series']),
                'step_rows':np.asarray(self.steps,dtype=np.float64).reshape((-1,len(_ABC_STEP_FIELDS))),
                'diagnostic_keys':np.asarray(list(self.diag[0]) if self.diag else [],dtype='U80'),
                'diagnostic_times':np.asarray([d['time'] for d in self.diag])}
        for group in (G['plane_data'],G['roof_data']):
            for key,d in group.items():
                for f,v in d.items():
                    if isinstance(v,list) and f in ('Jnet','Jnet_conv','Jnet_spin','Jplus_loc','Jminus_loc','Jplus_loc_conv','Jminus_loc_conv','rho_int'):
                        arrays[key+'__'+f]=np.asarray(v,float)
        if self.diag:
            for k in self.diag[0]:arrays['diag__'+k]=np.asarray([d[k] for d in self.diag])
        _abc_npz(temp/'histories.npz',**arrays)
        j=dict(version=_ABC_VERSION,step=self.step,time=self.t,label=label,configuration_sha256=self.fingerprint,
               updated_source_sha256=self.source_sha,P0_init64=self.p0,P0_init_native=self.p0_native,
               environment=self.record['environment'],state_mass64=_abc_norm(state,self.dV),files={f:_abc_sha(temp/f) for f in ('psi_flat.npy','histories.npz')})
        _abc_json(temp/'checkpoint.json',j);os.replace(temp,target)
        self.last_checkpoint=time.monotonic();self.last_state_checkpoint=str(target)
        _abc_json(self.out/'latest_checkpoint.json',dict(directory=str(target),step=self.step,time=self.t))
        return target

    def after_step(self,state):
        self.diagnostic(state,self.t,self.step,force=(self.t>=self.T))
        need=False
        while self.physical_checkpoints and self.t+1e-12>=self.physical_checkpoints[0]:
            self.physical_checkpoints.pop(0);need=True
        if time.monotonic()-self.last_checkpoint>=self.checkpoint_seconds:need=True
        stop=bool(self.stop_signal) or (self.pilot_steps>0 and self.step>=self.pilot_steps) or (self.stop_seconds>0 and time.monotonic()-self.start_wall>=self.stop_seconds)
        if need or stop:self.checkpoint(state,'clean_stop' if stop else 'regular')
        if stop and self.t<self.T:
            self.record.update(status='CLEAN_STOP_INCOMPLETE_NONACCEPTING',stop_signal=self.stop_signal,last_checkpoint=self.last_state_checkpoint)
            self.flush_native();self.stream.close()
            print(f'[revision] CLEAN_STOP at step={self.step} t={self.t:.12g}; checkpoint={self.last_state_checkpoint}',flush=True)
            raise SystemExit(3)

    def fail(self,state,message):
        # A failed I/O/recorder operation may have partially advanced histories.
        # Never serialize that exceptional state as a resumable checkpoint.
        self.record.update(status='FAILED_NONACCEPTING',failure_message=str(message),
                           recovery_policy='No failure checkpoint written. Use last completed CLEAN checkpoint only.',
                           last_clean_checkpoint=self.last_state_checkpoint)
        try:
            _abc_json(self.out/'revision_run.json',self.record)
        except Exception as io_exc:
            print('[failure metadata write failed] '+str(io_exc),file=sys.stderr,flush=True)
        try:
            self.stream.close()
        except Exception as io_exc:
            print('[failure stream close failed] '+str(io_exc),file=sys.stderr,flush=True)
        raise RuntimeError(str(message))

    def complete(self,state):
        self.checkpoint(state,'final')
        self.record.update(status='COMPLETED_NUMERICAL_RUN_NOT_CONVERGENCE_CERTIFIED',last_checkpoint=self.last_state_checkpoint)
        self.flush_native();self.stream.close()

# ---------------- END ADDITIVE REVISION SUPPORT v1 ----------------


# Fail before constructing matrices when the selected environment is incompatible.
if 'rtol' not in _inspect.signature(gmres).parameters:
    raise RuntimeError("The supplied solver requires a CuPy GMRES API with rtol. Use the same working environment as the reference calculation; no tolerance or solver substitution is made.")
if _BRANCH_PREFLIGHT:
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No CUDA device is visible in this allocation.")
    _br_pre = _abc_gpu_record()
    if 'gpu_query_error' in _br_pre:
        raise RuntimeError(_br_pre['gpu_query_error'])
    _br_z = cp.linspace(0.0,20.0,BRANCH_REFINEMENT_CASE['grid_generator_nz'],endpoint=False,dtype=cp.float32)[:BRANCH_REFINEMENT_CASE['Nz']]
    _br_grid = dict(actual_hz=float(cp.asnumpy(_br_z[1]-_br_z[0])),actual_z_detector=float(cp.asnumpy(_br_z[-1])))
    _br_pre.update(normal_grid_check=_br_grid,case=BRANCH_REFINEMENT_CASE,gmres_signature=str(_inspect.signature(gmres)),
                   check="imports/API/device discovery only; no TDSE step or memory-fit claim")
    print(json.dumps(_br_pre, indent=2, sort_keys=True))
    raise SystemExit(0)

# -------------------- directories / logging --------------------


OMEGA_ENV = os.getenv("OMEGA", None)

backend = "CuPy"
print(f"[info] backend: {backend}")

out_dir = _abc_output(os.getenv("OUTDIR", "."), "lower_scalar_ABC", 7800, 200.0)
_abc_json(out_dir / "branch_refinement_case.json", BRANCH_REFINEMENT_CASE)
_abc_json(out_dir / "launch_environment.json", _abc_gpu_record())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(out_dir / "simulation_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# -------------------- utilities --------------------


def to_cpu(arr):
    return cp.asnumpy(arr)

# -------------------- frozen parameters / declared repair --------------------
DTYPE_R, DTYPE_C = cp.float32, cp.complex64

Nx = 100
Ny = 100
Nz = 1500


omega = float(OMEGA_ENV) if (OMEGA_ENV is not None) else float(200.0)

ell = 10.0

Lx = ell / np.sqrt(omega)
Ly = ell / np.sqrt(omega)
Lz = 20.0

dt = 0.00025
T_final = 40.0

Vxy0 = omega**2
V0z = 0.0
alpha = 0.0
k_bc = float(np.pi)
k_lower = float(np.pi)

z0_init  = 10.0
sigma_z  = 0.5
k0       = float(np.pi)

theta = float(0.0)
phi   = np.float32(0.0)

# Frozen case guard: this source is a separate metric-consistent repair/control.
for _br_key in ('Nx','Ny','Nz','omega','ell','Lz','dt','T_final','z0_init','sigma_z','theta','phi'):
    if float(globals()[_br_key]) != float(BRANCH_REFINEMENT_CASE[_br_key]):
        raise RuntimeError("Frozen refinement setting differs from manifest: " + _br_key)
if k0 != float(np.pi) or k_bc != float(np.pi) or k_lower != float(np.pi):
    raise RuntimeError("The control must retain k0=kappa_top=kappa_lower=pi.")

# -------------------- mesh / initial state (UNCHANGED) --------------------
# Mesh and initial condition
x = cp.linspace(0.0, Lx, Nx, endpoint=False, dtype=DTYPE_R)
y = cp.linspace(0.0, Ly, Ny, endpoint=False, dtype=DTYPE_R)
z = cp.linspace(0.0, Lz, Nz, endpoint=False, dtype=DTYPE_R)
hx = float(x[1] - x[0])
hy = float(y[1] - y[0])
hz = float(z[1] - z[0])

X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
mid_x = Nx // 2
mid_y = Ny // 2
mid_z = Nz // 2


Rho2 = (X - Lx/2)**2 + (Y - Ly/2)**2
psi_xy = cp.exp(-omega * Rho2 / 2)

psi_z = cp.exp(-(Z - z0_init)**2 / (2 * sigma_z**2)) * cp.exp(1j * k0 * (Z - z0_init))

psi_scalar = (psi_xy * psi_z).astype(DTYPE_C)

# Define initial spin state (general: both up and down)
c_up = cp.cos(theta / 2)
c_down = cp.sin(theta / 2) * cp.exp(1j * phi)

# Apply to scalar part (no phase term as requested)
psi_up = c_up * psi_scalar
psi_down = c_down * psi_scalar

# Enforce lateral Dirichlet BCs; the scalar lower ABC is not clamped.
psi_up[[0, -1], :, :] = 0
psi_up[:, [0, -1], :] = 0
psi_down[[0, -1], :, :] = 0
psi_down[:, [0, -1], :] = 0

# Normalize the wavefunction (now for full spinor)
def normalize_wavefunction(psi_up, psi_down, hx, hy, hz):
    """Normalize the full spinor wavefunction to ensure total probability = 1."""
    total_prob = _abc_mass_native(psi_up,psi_down,hx*hy*hz)
    norm = cp.sqrt(total_prob)
    psi_up /= norm
    psi_down /= norm
    return psi_up, psi_down

psi_up, psi_down = normalize_wavefunction(psi_up, psi_down, hx, hy, hz) # Normalize total
psi_flat = cp.concatenate((psi_up.ravel(), psi_down.ravel())).astype(STATE_DTYPE)



# Harmonic oscillator potential in x and y directions
V_real = 0.5 * Vxy0 * ((X - Lx/2)**2 + (Y - Ly/2)**2)

# Absorbing potential 

# ---- complex absorbing layer (smooth ramp) near the roof ----
        # strictly non-positive imaginary part
V_abs=0.0   #Now @ this *Moment there is no absorpin pot

V = V_real.astype(DTYPE_C) + V_abs            # <-- replace your old V definition
V_diag = V.ravel().astype(ASSEMBLY_DTYPE)
V_full = cp.concatenate((V_diag, V_diag))


# ── save coordinate grids & constants once ─────────────────────────
X_cpu, Y_cpu, Z_cpu = map(to_cpu,(X,Y,Z))
np.save(out_dir/"X_cpu.npy", X_cpu)
np.save(out_dir/"Y_cpu.npy", Y_cpu)
np.save(out_dir/"Z_cpu.npy", Z_cpu)
constants = {
    "omega": float(omega), "T_final": float(T_final), "theta": float(theta), "phi": float(phi),
    "z0": float(z0_init), "sigma_z": float(sigma_z), "k0": float(k0), "k_bc": float(k_bc),
    "z_roof_actual": float(to_cpu(z[-1])),
    "z_lower_actual": float(to_cpu(z[0])), "k_lower": float(k_lower),
    "lower_endpoint_mass_weight": LOWER_ENDPOINT_MASS_WEIGHT,
    "lower_endpoint_closure": BRANCH_REFINEMENT_CASE["lower_endpoint_closure"],
    "grid_generator_nz": int(BRANCH_REFINEMENT_CASE["grid_generator_nz"]),
    "hx": hx, "hy": hy, "hz": hz, "Lx": Lx, "Ly": Ly, "Lz": Lz, "dt": dt,
    "Nx": Nx, "Ny": Ny, "Nz": Nz, "mid_x": mid_x, "mid_y": mid_y, "mid_z": mid_z,
    "endpoint_mass_weight": ENDPOINT_MASS_WEIGHT, "endpoint_closure": ENDPOINT_CLOSURE,
    "metric": BRANCH_REFINEMENT_CASE["metric"], "state_representation": "physical_nodal_spinor",
    "state_dtype": BRANCH_REFINEMENT_CASE["state_dtype"], "matrix_dtype": BRANCH_REFINEMENT_CASE["matrix_dtype"],
}
np.savez(out_dir/"constants.npz", **constants)
logging.info("[info] Saved coordinate grids and constants")
# <<< END MOVED BLOCK

# frees hundreds of MB, no effect on solver or outputs
del X, Y, Z, Rho2, psi_xy, psi_z, psi_scalar


#Print all parameters to stdout (which is teed to stdout.txt)
print("Simulation Parameters:")
print(f"Nx: {Nx}")
print(f"Ny: {Ny}")
print(f"Nz: {Nz}")
print(f"theta: {theta}")
print(f"Lx: {Lx}")
print(f"Ly: {Ly}")
print(f"Lz: {Lz}")
print(f"k_bc: {k_bc}")
print(f"k_lower: {k_lower}")
print(f"dt: {dt}")
print(f"T_final: {T_final}")
print(f"V0z: {V0z}")
print(f"alpha: {alpha}")
print(f"DTYPE_R: {DTYPE_R}")
print(f"DTYPE_C: {DTYPE_C}")
print(f"omega: {omega}")
print(f"hx: {hx}")
print(f"hy: {hy}")
print(f"hz: {hz}")
print(f"mid_x: {mid_x}")
print(f"mid_y: {mid_y}")
print(f"mid_z: {mid_z}")
print(f"out_dir: {out_dir}")
#print(f"mask: (Z > 0) & (Z < 1) # Width 1, centered at ~14.5")
print("-------------------------")


def L_z(N_z, dz, k_bc, k_lower):
    """Centered scalar endpoint part of -1/2 d_z^2 at both ends.

    Lower outward normal is -z: d_z psi=-i*k_lower*psi. Eliminating the
    ghost psi[-1]=psi[1]+2i*h*k_lower*psi[0] gives H00=h^-2-i*k_lower/h,
    H01=-h^-2. The upper endpoint has the corresponding outward +z law.
    Both endpoint masses are 1/2; tangential spin coupling exists ONLY at top.
    """
    inv = 1.0 / (dz * dz)
    data, indices, indptr = [], [], [0]
    for i in range(N_z):
        if i == 0:
            scale=1.0/LOWER_ENDPOINT_MASS_WEIGHT
            indices += [i, i+1]
            data += [(0.5*inv-1j*k_lower/(2.0*dz))*scale,-0.5*inv*scale]
        elif i == N_z-1:
            scale=1.0/ENDPOINT_MASS_WEIGHT
            indices += [i-1,i]
            data += [-0.5*inv*scale,(0.5*inv-1j*k_bc/(2.0*dz))*scale]
        else:
            indices += [i-1,i,i+1]
            data += [-0.5*inv,inv,-0.5*inv]
        indptr.append(len(indices))
    data=cp.array(data,dtype=ASSEMBLY_DTYPE)
    indices=cp.array(indices,dtype=cp.int32)
    indptr=cp.array(indptr,dtype=cp.int32)
    return csr_matrix((data,indices,indptr),shape=(N_z,N_z))


def L_dirichlet(N, d):
    """
    Constructs a second-derivative operator in x/y with Dirichlet boundary
    conditions, using CuPy's CSR matrix format.
    """
    inv_d2 = 1 / d**2
    half_inv_d2 = 0.5 * inv_d2
    data = []
    indices = []
    indptr = [0]
    for i in range(N):
        if i == 0 or i == N - 1:
            indptr.append(indptr[-1]) # Empty row for Dirichlet
        else:
            left = i-1
            right = i+1
            if left == 0:
                indices.extend([i, right])
                data.append(inv_d2)
                data.append(-half_inv_d2)
                indptr.append(indptr[-1] + 2)
            elif right == N-1:
                indices.extend([left, i])
                data.append(-half_inv_d2)
                data.append(inv_d2)
                indptr.append(indptr[-1] + 2)
            else:
                indices.extend([left, i, right])
                data.append(-half_inv_d2)
                data.append(inv_d2)
                data.append(-half_inv_d2)
                indptr.append(indptr[-1] + 3)
    data = cp.array(data, dtype=ASSEMBLY_DTYPE)
    indices = cp.array(indices, dtype=cp.int32)
    indptr = cp.array(indptr, dtype=cp.int32)
    L_raw = csr_matrix((data, indices, indptr), shape=(N, N))
    return L_raw


# Laplacian with BCs
L1 = L_z(Nz, hz, k_bc, k_lower)
L2 = L_dirichlet(Nx, hx)
L3 = L_dirichlet(Ny, hy)
Ix = eye(Nx, dtype=ASSEMBLY_DTYPE, format='csr')
Iy = eye(Ny, dtype=ASSEMBLY_DTYPE, format='csr')
Iz = eye(Nz, dtype=ASSEMBLY_DTYPE, format='csr')

Lap_scalar = (kron(L2, kron(Iy, Iz)) + kron(Ix, kron(L3, Iz)) + kron(Ix, kron(Iy, L1))).tocsr()
Lap_diag = kron(eye(2, dtype=ASSEMBLY_DTYPE, format='csr'), Lap_scalar)

# Build coupling matrix C for spinor ABC at top boundary (Hermitian when k_bc=0)
rows_list, cols_list, data_list = [], [], []
coef = 0.5 / (hz * ENDPOINT_MASS_WEIGHT)
d_coef_x = 1.0 / (2.0 * hx)
d_coef_y = 1.0 / (2.0 * hy)
Ngrid = Nx * Ny * Nz

for ix in range(1, Nx-1):
    for iy in range(1, Ny-1):
        z_top = Nz - 1
        base = ix * Ny * Nz + iy * Nz + z_top

        # UP row
        row_up = base
        if ix > 1:
            col = Ngrid + (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (-d_coef_x))
        if ix < Nx - 2:
            col = Ngrid + (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (+d_coef_x))
        if iy > 1:
            col = Ngrid + ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 2:
            col = Ngrid + ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (-1j) * d_coef_y)

        # DOWN row
        row_down = Ngrid + base
        if ix > 1:
            col = (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (+d_coef_x))
        if ix < Nx - 2:
            col = (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (-d_coef_x))
        if iy > 1:
            col = ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 2:
            col = ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (-1j) * d_coef_y)


C = coo_matrix(
    (cp.array(data_list, dtype=ASSEMBLY_DTYPE), (cp.array(rows_list), cp.array(cols_list))),
    shape=(2 * Ngrid, 2 * Ngrid), dtype=ASSEMBLY_DTYPE
).tocsr()

Lap_spinor = (Lap_diag + C).tocsr()


# print('--------------------------------------------------------\n')
# # -----------------
# Crank-Nicolson matrices
Ntot = 2 * Nx * Ny * Nz
Id = eye(Ntot, dtype=ASSEMBLY_DTYPE, format='csr')

Potential = diags(V_full, 0, format='csr')
A = Id + 1j*dt/2 * (Lap_spinor + Potential).tocsr()
B = Id - 1j*dt/2 * (Lap_spinor + Potential).tocsr()
inv_diag = 1.0 / A.diagonal()
M = LinearOperator(shape=A.shape, matvec=lambda x: inv_diag * x)

# # build once outside loop
A64 = A.astype(cp.complex128)
B64 = B.astype(cp.complex128)
inv_diag64 = 1.0 / A64.diagonal()
M64 = LinearOperator(shape=A64.shape, matvec=lambda x: inv_diag64 * x)

# =============================================================================
# =============================================================================
# =============================================================================
# Diagnostics (UPDATED: roof reflection, first bounce; store NET + LOCAL splits)
# =============================================================================
# =============================================================================
# =============================================================================

Z_GUARD    = 4
Z_SEP_LIST = [60, 105]
AVG_PLANES = 1

z_roof_idx = Nz - 1
z_safe_top = z_roof_idx - Z_GUARD

plane_indices = []
for sep in Z_SEP_LIST:
    zp = z_safe_top - int(sep)
    if zp < 3:
        raise ValueError("Plane too close to bottom. Reduce Z_SEP_LIST or increase Nz.")
    plane_indices.append(int(zp))

# Window/threshold settings
EARLY_FRAC        = 0.25
INC_FRAC          = 0.10
WIDTH_FRAC        = 0.10
ARR_BASE_Q        = 0.10
ARR_MAD_MULT      = 15.0
ARR_MIN_BASE      = 20
ARRIVAL_ABS       = 1e-12
ARRIVAL_REL       = 1e-3
NO_RETURN_SAFETY  = 0.95
W_MIN_SAMPLES     = 5
BASE_MIN_SAMPLES  = 5

v_proxy = float(k0)  # same baseline value; correct source-speed fallback for future detuning

# monitoring cadence (keep yours if you like)
monitor_every = 20
prob_every    = 100
dt_mon        = monitor_every * dt

# -------------------- derivative helpers (keep your existing dz/dx/dy helpers if already defined) --------------------
def dz_slice_2nd(u, z_plane, hz):
    if z_plane == 0:
        return (u[:, :, 1] - u[:, :, 0]) / hz
    elif z_plane == u.shape[2] - 1:
        return (u[:, :, -1] - u[:, :, -2]) / hz
    else:
        return (u[:, :, z_plane+1] - u[:, :, z_plane-1]) / (2*hz)

def dx_2nd_2d(f2d, hx):
    out = cp.empty_like(f2d, dtype=cp.float64)
    f2d = f2d.astype(cp.float64, copy=False)
    out[1:-1, :] = (f2d[2:, :] - f2d[:-2, :]) / (2.0 * hx)
    out[0, :]    = (f2d[1, :] - f2d[0, :]) / hx
    out[-1, :]   = (f2d[-1, :] - f2d[-2, :]) / hx
    return out

def dy_2nd_2d(f2d, hy):
    out = cp.empty_like(f2d, dtype=cp.float64)
    f2d = f2d.astype(cp.float64, copy=False)
    out[:, 1:-1] = (f2d[:, 2:] - f2d[:, :-2]) / (2.0 * hy)
    out[:, 0]    = (f2d[:, 1] - f2d[:, 0]) / hy
    out[:, -1]   = (f2d[:, -1] - f2d[:, -2]) / hy
    return out

def plane_pauli_fluxes(psi_up, psi_down, zp):
    """
    Returns:
      Jnet_total, Jnet_conv, Jnet_spin,
      Jplus_loc_total, Jminus_loc_total,
      Jplus_loc_conv,  Jminus_loc_conv
    """
    u = psi_up[:, :, zp]
    d = psi_down[:, :, zp]

    duz = dz_slice_2nd(psi_up,   zp, hz)
    ddz = dz_slice_2nd(psi_down, zp, hz)

    j_conv = cp.imag(u.conj() * duz + d.conj() * ddz).astype(cp.float64)

    c  = u.conj() * d
    Sx = (2.0 * cp.real(c)).astype(cp.float64)
    Sy = (2.0 * cp.imag(c)).astype(cp.float64)

    dSydx = dx_2nd_2d(Sy, hx)
    dSxdy = dy_2nd_2d(Sx, hy)
    j_spin = (0.5 * (dSydx - dSxdy)).astype(cp.float64)

    jz = j_conv + j_spin

    Jnet_total = cp.sum(jz)     * (hx * hy)
    Jnet_conv  = cp.sum(j_conv) * (hx * hy)
    Jnet_spin  = cp.sum(j_spin) * (hx * hy)

    Jplus_loc_total  = cp.sum(cp.maximum(jz,     0.0)) * (hx * hy)
    Jminus_loc_total = cp.sum(cp.maximum(-jz,    0.0)) * (hx * hy)

    Jplus_loc_conv   = cp.sum(cp.maximum(j_conv, 0.0)) * (hx * hy)
    Jminus_loc_conv  = cp.sum(cp.maximum(-j_conv,0.0)) * (hx * hy)

    return (float(Jnet_total.get()), float(Jnet_conv.get()), float(Jnet_spin.get()),
            float(Jplus_loc_total.get()), float(Jminus_loc_total.get()),
            float(Jplus_loc_conv.get()),  float(Jminus_loc_conv.get()))


def avg_over_planes(func, psi_up, psi_down, zp, avg=0):
    vals = []
    for dzp in range(-avg, avg+1):
        zq = int(zp + dzp)
        if (zq < 2) or (zq > Nz-3):
            continue
        vals.append(func(psi_up, psi_down, zq))
    if not vals:
        return (np.nan,)*7
    return tuple(float(np.mean([v[i] for v in vals])) for i in range(7))

# -------------------- storage --------------------
t_series = []
P_series = []

z0 = float(z[0].get())
z_roof_pos = float(z[z_roof_idx].get())
plane_data = {}
for i, zp in enumerate(plane_indices):
    key = f"plane{i+1}"
    z_pos = float(z[zp].get())
    plane_data[key] = {
        "z_index": int(zp),
        "z_pos": float(z_pos),
        "dist_to_roof": float(z_roof_pos - z_pos),
        "dist_to_bottom": float(z_pos - z0),

        "Jnet": [],
        "Jnet_conv": [],
        "Jnet_spin": [],
        "Jplus_loc": [],         # total local +
        "Jminus_loc": [],        # total local -
        "Jplus_loc_conv": [],    # convective local +
        "Jminus_loc_conv": [], 
        # convective local -
        "analysis_total": {},
        "analysis_conv": {},
        "checks": {}    }

# original probability logging arrays (kept)
prob_t   = []
prob_val = []

# -------------------- roof diagnostics (NEW) --------------------
ROOF_DIAG = True

z_roof_idx = Nz - 1          # true boundary plane (Robin/ABC row in L_z)
z_roof_in  = Nz - 2          # one grid inside (recommended sanity check)

roof_data = {
    "roof": {
        "z_index": int(z_roof_idx),
        "z_pos": float(z[z_roof_idx].get()),
        "Jnet": [], "Jnet_conv": [], "Jnet_spin": [],
        "Jplus_loc": [], "Jminus_loc": [],
        "Jplus_loc_conv": [], "Jminus_loc_conv": [],
        "rho_int": []   # ∫_{roof} ρ dA
    },
    "roof_in": {
        "z_index": int(z_roof_in),
        "z_pos": float(z[z_roof_in].get()),
        "Jnet": [], "Jnet_conv": [], "Jnet_spin": [],
        "Jplus_loc": [], "Jminus_loc": [],
        "Jplus_loc_conv": [], "Jminus_loc_conv": [],
        "rho_int": []   # ∫_{roof_in} ρ dA
    }
}



# -------------------- time loop (original CN map; additive diagnostics) --------------------
# ceil avoids a truncated final horizon; only a terminal remainder uses a shorter step.
num_steps = int(np.ceil(np.nextafter(T_final / dt, -np.inf)))
print(f"[info] Starting time loop – {num_steps} steps; requested T={T_final:g}")
start_time = time.time()
psi0 = psi_flat.reshape(2, Nx, Ny, Nz)
P0_init = float(to_cpu(_abc_mass_native(psi0[0],psi0[1],hx*hy*hz)))
del psi0
rev = _ABCRevision(globals(), ORIGINAL_SOURCE_SHA256)
# Save exact 1-D coordinates without duplicating the three large broadcast grids.
# Old X_cpu/Y_cpu/Z_cpu files are still saved above for native compatibility.
_abc_npz(out_dir / "revision_coordinates.npz", x=to_cpu(x), y=to_cpu(y), z=to_cpu(z))


def _capture_native_endpoint(t_now, ensure_probability=False):
    # Native field names retained; volume probabilities use the declared physical metric.
    global psi_up, psi_down
    state = psi_flat.reshape(2, Nx, Ny, Nz)
    psi_up, psi_down = state[0], state[1]
    total = float(to_cpu(_abc_mass_native(psi_up,psi_down,hx*hy*hz)))
    if (ensure_probability or not prob_t) and (not prob_t or t_now > prob_t[-1]):
        prob_t.append(float(t_now)); prob_val.append(total)
    if t_series and t_now <= t_series[-1]:
        return
    t_series.append(float(t_now)); P_series.append(total)
    for key, dct in plane_data.items():
        vals = avg_over_planes(plane_pauli_fluxes, psi_up, psi_down, dct["z_index"], avg=AVG_PLANES)
        for field, value in zip(("Jnet","Jnet_conv","Jnet_spin","Jplus_loc","Jminus_loc","Jplus_loc_conv","Jminus_loc_conv"), vals):
            dct[field].append(value)
    if ROOF_DIAG:
        for key, dct in roof_data.items():
            zp=dct["z_index"]
            vals=plane_pauli_fluxes(psi_up,psi_down,zp)
            for field,value in zip(("Jnet","Jnet_conv","Jnet_spin","Jplus_loc","Jminus_loc","Jplus_loc_conv","Jminus_loc_conv"),vals):
                dct[field].append(value)
            rho2d=cp.abs(psi_up[:,:,zp])**2+cp.abs(psi_down[:,:,zp])**2
            dct["rho_int"].append(float(cp.sum(rho2d).get())*(hx*hy))


if rev.step == 0:
    _capture_native_endpoint(0.0, ensure_probability=True)
# Release construction-only matrices once. H_for_final is retained solely for a
# possible terminal remainder step; this is a memory optimization, not a new map.
H_for_final = (Lap_spinor + Potential).tocsr()
for _name in ('A','B','M','inv_diag','Lap_scalar','Lap_diag','L1','L2','L3','Ix','Iy','Iz','C','Lap_spinor','Potential','V','V_diag','V_full','V_real','X_cpu','Y_cpu','Z_cpu'):
    globals().pop(_name,None)
cp.get_default_memory_pool().free_all_blocks()

for n in range(rev.step + 1, num_steps + 1):
    previous = psi_flat  # last accepted subnormalized state, no renormalization
    try:
        left, right, step_dt = rev.time_step(n)
        mass_before = rev.begin(previous)
        Astep, Bstep, Mstep = A64, B64, M64
        # Ordinary steps use this case's fixed A64/B64/M64 objects.
        if n == num_steps and not np.isclose(step_dt, dt, rtol=1e-10, atol=1e-15):
            Afinal = Id + 1j*step_dt/2*H_for_final
            Bfinal = Id - 1j*step_dt/2*H_for_final
            Astep = Afinal.astype(cp.complex128); Bstep = Bfinal.astype(cp.complex128)
            inv_final = 1.0 / Astep.diagonal()
            Mstep = LinearOperator(shape=Astep.shape, matvec=lambda v: inv_final*v)
            _abc_json(out_dir/"terminal_step.json", {"step":n,"base_dt":dt,"terminal_dt":step_dt,"left":left,"right":right})
        rhs64 = (Bstep @ previous.astype(cp.complex128))
        psi64, info = gmres(Astep, rhs64, x0=previous.astype(cp.complex128),
                            rtol=1e-8, atol=0.0, maxiter=1000, M=Mstep, restart=30)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge at step {n}: info={info}; no failed state was committed")
        candidate = psi64.astype(STATE_DTYPE)
        mass_cast = _abc_norm(candidate, hx*hy*hz)
        psi_3d = candidate.reshape(2, Nx, Ny, Nz)
        psi_up, psi_down = psi_3d[0], psi_3d[1]
        # Only lateral Dirichlet planes are clamped; neither absorbing z endpoint is clamped.
        psi_up[[0,-1],:,:]=0; psi_up[:,[0,-1],:]=0
        psi_down[[0,-1],:,:]=0; psi_down[:,[0,-1],:]=0
        rev.record_step(n,left,right,previous,psi64,candidate,rhs64,Astep,mass_before,mass_cast)
        psi_flat=candidate
    except Exception as exc:
        # rev.step still identifies the last committed step if the solve/audit failed.
        psi_flat=previous
        rev.fail(previous,exc)
    t = right
    if (n % prob_every)==0 or n==num_steps:
        total_prob=float(to_cpu(_abc_mass_native(psi_up,psi_down,hx*hy*hz)))
        if t>prob_t[-1]:prob_t.append(t);prob_val.append(total_prob)
        logging.info(f"[Probability] t={t:.9f}, ∫|ψ|² dV = {total_prob:.10f}")
    if (n % monitor_every)==0 or n==num_steps:
        # This appends the native monitor records; the logger addition above is
        # de-duplicated, and legacy file names keep their original meaning.
        _capture_native_endpoint(t)
    if n%200==0:
        elapsed=time.time()-start_time
        completed_here=max(1,n-int(rev.record.get('restart_step',0)))
        eta=(num_steps-n)*(elapsed/completed_here)/3600
        print(f"[progress] Step {n}/{num_steps} t={t:.9f} | ETA {eta:.3f} h",flush=True)
    rev.after_step(psi_flat)

# Explicit terminal flush, including the exact requested horizon.
_capture_native_endpoint(float(T_final), ensure_probability=True)
rev.flush_native()

# All upper/lower monitor records and probability histories have been saved by flush_native.
# Old reflecting-bottom peak heuristics and sole-detector -dS/dt postprocessing do not apply.
summary = {
    'revision_version':_ABC_VERSION,
    'case':BRANCH_REFINEMENT_CASE,
    'run_dir':str(out_dir.resolve()),
    'P0_init64':float(rev.p0),
    'P0_init_native':float(rev.p0_native),
    'two_channel_statistics':rev.channel_summary(),
    'top_detector':'upper spinor ABC midpoint events in revision_step_data.csv',
    'lower_escape':'lower scalar ABC midpoint events, separate from top detection',
    'legacy_arrays':{
        'total_probs.npy':'raw surviving norm with half weights at both z endpoints',
        'P_total_on_plane_times.npy':'raw surviving norm at plane-record times',
        'prob_loss.npy':'TOTAL raw norm loss; not top detection probability',
        'roof_rho_int.npy':'top boundary density; multiply by k_bc for sampled top loss rate',
        'roof_Jnet.npy':'sampled derivative transport diagnostic; not authoritative top event rate'},
    'lower_monitor_actual_link_z':rev.lower_link_positions.tolist(),
    'lower_monitor_direction':'signed +z link currents; negative is toward lower escape plane',
    'lower_termination':'passive scalar Robin, partially reflecting except at matching discrete wave number',
    'reflection_analysis':'use separate characterized lower-boundary analysis; no reflecting-bottom heuristic applied',
    'elapsed_seconds':time.time()-start_time,
}
_abc_json(out_dir/'summary.json',summary)
print('[two-channel final] '+json.dumps(_abc_clean(rev.channel_summary()),sort_keys=True),flush=True)
print(f'[info] Outputs saved in: {out_dir.resolve()}',flush=True)
rev.complete(psi_flat)

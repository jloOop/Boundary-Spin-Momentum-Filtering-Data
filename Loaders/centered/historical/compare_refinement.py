import os
from pathlib import Path
import json,sys,numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'analysis'))
import native_tools as nt

def rate_values(d,x,field=None):
 idx=np.searchsorted(d['r'],x,side='right')
 return (d['p'] if field is None else d['f'][field].to_numpy()/d['P0'])[idx]/d['dt_actual'][idx]

def event_abs_diff(b,f,a,T,conditional=False):
 nt.require(a>=0 and T<=min(b['T'],f['T']),'unsupported pair horizon')
 t=np.unique(np.r_[a,b['r'][(b['r']>a)&(b['r']<T)],f['r'][(f['r']>a)&(f['r']<T)],T]);h=np.diff(t);mid=t[:-1]+h/2
 ub=rate_values(b,mid);uf=rate_values(f,mid)
 if conditional:
  ub=ub/nt.event_mom(b,a,T)[0];uf=uf/nt.event_mom(f,a,T)[0]
 return float(np.sum(h*abs(uf-ub)))

def branch_abs_diff(b,f,a,T):
 tb=b['diag']['time'];tf=f['diag']['time'];x=np.unique(np.r_[a,tb[(tb>a)&(tb<T)],tf[(tf>a)&(tf<T)],T])
 den=b['cfg']['k_bc']/b['P0']*nt.lin_mom(tb,b['diag']['roof_density'],a,T)[0]
 vals={}
 for k in nt.BR+['roof_density']:
  vb=np.interp(x,tb,b['diag'][k])*b['cfg']['k_bc']/b['P0'];vf=np.interp(x,tf,f['diag'][k])*f['cfg']['k_bc']/f['P0']
  vals[k]=nt.abs_linear(x,vf-vb)
 vals.update(den=den,epsilon_branch_percent=100*sum(vals[k] for k in nt.BR)/den,epsilon_roof_percent=100*vals['roof_density']/den)
 return vals

def pair(b,f,a,T):
 D=nt.event_mom(b,a,T)[0];Df=nt.event_mom(f,a,T)[0]
 l1=event_abs_diff(b,f,a,T);bro=branch_abs_diff(b,f,a,T)
 xb=np.r_[0,np.cumsum(b['p'])];xf=np.r_[0,np.cumsum(f['p'])]
 t=np.unique(np.r_[a,b['r'][(b['r']>a)&(b['r']<T)],f['r'][(f['r']>a)&(f['r']<T)],T]);delta=np.interp(t,f['t'],xf)-np.interp(t,b['t'],xb)
 imax=np.argmax(abs(delta));wb=nt.window(b,a,T);wf=nt.window(f,a,T)
 out=dict(baseline=b['case'],refined=f['case'],a=a,b=T,D_baseline=D,D_refined=Df,L1_rate=l1,epsilon_CN_percent=100*l1/D,epsilon_branch_percent=bro['epsilon_branch_percent'],epsilon_sampled_roof_percent=bro['epsilon_roof_percent'],conditional_TV=event_abs_diff(b,f,a,T,True)/2,max_abs_cumulative_diff=float(abs(delta[imax])),t_max_cumulative_diff=float(t[imax]),delta_Pz=wf['Pzdet']-wb['Pzdet'],delta_wplus=wf['wplus']-wb['wplus'])
 if a==0:
  out['censored_TV']=0.5*(l1+abs(Df-D)) # event+no-click measure, all norm corrections excluded
  mb=nt.lin_mom(b['t'],b['S'],0,T)[0];mf=nt.lin_mom(f['t'],f['S'],0,T)[0];out['epsilon_mu_refined_denom_percent']=100*abs(mf-mb)/abs(mf)
 return out


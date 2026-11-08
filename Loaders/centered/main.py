"""Portable main-results reconstruction from compact native detector records.

Archived images are separately tracked. Generated plots preserve archived curve
identities and settings but are numerical reconstructions, not claims of exact
embedded-manuscript image identity. No production solver is imported.
"""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from centered.numerics import events, linear_integral

__all__ = ['run', 'load_native', 'horizon_statistics']
BASELINE = {1:5540, 25:5042, 50:5047, 75:5043, 100:5541, 150:5044, 200:5045, 300:7100}
DISPLAY = BASELINE.copy()
DISPLAY[300] = 7706
HORIZONS = (10.,20.,30.,40.)
MANUSCRIPT_DT = (10.,8.,7.,5.,3.,2.5,2.,1.5)


def load_native(root, case):
    directory = Path(root)/'data/centered/native'/str(case)
    metadata=json.loads((directory/'run.json').read_text())
    with np.load(directory/'steps.npz',allow_pickle=False) as z:
        steps={k:z[k] for k in z.files}
    return metadata, steps


def _check(checks, name, residual, tolerance):
    checks.append(dict(name=name,actual=float(residual),tolerance=float(tolerance),
                       passed=bool(np.isfinite(residual) and residual <= tolerance)))


def _write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def horizon_statistics(metadata, s, T):
    P0=metadata['P0_init64']
    outcome=events(s['t_left'],s['t_right'],s['roof_probability'],0.,T,P0)
    times=np.r_[s['t_left'][0],s['t_right']]
    survival=np.r_[P0,s['mass_stored']]/P0
    mu=linear_integral(times,survival,0.,T)
    S=float(np.interp(T,times,survival));D=outcome['probability'];M=outcome['first_moment']
    keep=times<T
    t=np.r_[times[keep],T];v=np.r_[survival[keep],S]
    mu_simpson=float(simpson(v,x=t))
    return dict(T=T,D_T=D,conditional_mean=outcome['conditional_mean'],mu_star_linear=mu,
                mu_star_original_simpson=mu_simpson,simpson_minus_linear=mu_simpson-mu,
                S_T=S,one_minus_S_T=1-S,D_minus_one_minus_S=D-(1-S),
                M_T=M,mu_event_capped=M+T*(1-D),
                linear_minus_event_identity=mu-(M+T*(1-D)))


def _plot_profiles(z, out):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator,MultipleLocator,FormatStrFormatter
    colors={1:'#CC6677',50:'#332288',100:'#999933',150:'#44AA99',200:'#AA4499',300:'#000000'}
    full_width={1:1.8,50:1.6,100:1.45,150:1.1,200:.75,300:.6}
    late_width={1:1.8,50:1.6,150:1.1,300:.9}
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.labelsize':12,'axes.linewidth':.65,'path.simplify':False,'savefig.facecolor':'white'})
    for panel, selection in [('a',(1,50,100,150,200,300)),('b',(1,50,150,300))]:
        fig=plt.figure(figsize=(8.8,3.75))
        height=7.5*(.952-.085)/(2+.39)
        ax=fig.add_axes([.115,.17,.863,height/3.75])
        ax.set_xlabel(r'$t$',labelpad=4);ax.set_ylabel('Probability density',labelpad=9)
        ax.set_title(f'({panel})',loc='left',x=.025,fontsize=12,pad=10)
        ax.tick_params(which='both',direction='out',top=False,right=False,width=.7)
        ax.tick_params(which='major',length=4.2);ax.tick_params(which='minor',length=2.2)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        widths=full_width if panel=='a' else late_width
        for w in selection:
            ax.plot(z[f'omega{w}_t'],z[f'omega{w}_g'],color=colors[w],lw=widths[w],label=rf'$\omega={w}$')
        if panel=='a':
            ax.plot(z['free_gaussian_t'],z['free_gaussian_g'],color='#505050',lw=1.3,
                    linestyle=(0,(5,3)),label='Free Gaussian current')
            ax.set(xlim=(0,40),ylim=(-.0025,.405))
            ax.xaxis.set_major_locator(MultipleLocator(5));ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(.10));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.legend(loc='upper right',ncol=3,frameon=False,fontsize=9.5,handlelength=2.6,columnspacing=1.35,labelspacing=.6,borderaxespad=.7)
        else:
            ax.set(xlim=(8,40),ylim=(-.0002,.0355))
            ax.xaxis.set_major_locator(MultipleLocator(4));ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(.005));ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.legend(loc='lower right',bbox_to_anchor=(1.,1.016),ncol=4,frameon=False,fontsize=10,
                      handlelength=2.6,columnspacing=1.6,borderaxespad=0)
        fig.savefig(out/f'Fig1{panel}.png',dpi=200);plt.close(fig)


def _plot_fit(omega, values, A, B, out):
    import matplotlib.pyplot as plt
    def style(ax):
        for spine in ax.spines.values():spine.set_linewidth(.4)
        ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True,width=.9)
        ax.tick_params(which='major',length=6);ax.tick_params(which='minor',length=2.5);ax.grid(True,alpha=.10)
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,'legend.fontsize':11})
    fig,ax=plt.subplots(figsize=(8.6,5.2));style(ax)
    x=np.linspace(omega.min(),omega.max(),500)
    ax.plot(x,A+B*np.sqrt(x),color='#050505',lw=.9,label=rf'fit: $\mu^*(40)\approx {A:.3f}+{B:.3f}\sqrt{{\omega}}$')
    ax.plot(omega,values,ls='none',marker='o',ms=7,mew=1.2,mec='#3F6AA8',mfc='#3F6AA8',label=r'data: $\mu^*(40)$')
    ax.set_xlabel(r'$\omega$',fontweight='bold');ax.set_ylabel(r'$\mu^*(T)$',fontweight='bold')
    ax.set(xlim=(-14,315),ylim=(4.5,25.5));ax.legend(loc='upper left',frameon=True)
    ins=ax.inset_axes([.58,.10,.36,.39]);style(ins);rx=np.sqrt(x)
    ins.plot(rx,A+B*rx,color='#050505',lw=.8)
    ins.plot(np.sqrt(omega),values,ls='none',marker='o',ms=4.5,mew=.9,mec='#3F6AA8',mfc='#3F6AA8')
    ins.set_xlabel(r'$\sqrt{\omega}$',fontsize=10,fontweight='bold',labelpad=1)
    ins.set_ylabel(r'$\mu^*(T)$',fontsize=10,fontweight='bold',labelpad=1)
    ins.tick_params(which='major',labelsize=9,length=4);ins.tick_params(which='minor',length=2)
    ins.set(xlim=(np.sqrt(omega.min())-.5,np.sqrt(omega.max())+.5),ylim=(4.5,25.5))
    fig.savefig(out/'Fig2.png',dpi=200,bbox_inches='tight');plt.close(fig)


def _plot_horizons(rows,out):
    import matplotlib.pyplot as plt
    plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'path.simplify':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for T,color in zip(HORIZONS,('#1B0264','#136384','#72CE6D','#E7E656')):
        selected=[r for r in rows if r['T']==T]
        for ax,key,label in zip(axes,('D_T','conditional_mean','mu_star_linear'),(r'$D_T$',r'$\overline{\tau}^{\mathrm{cond}}_T$',r'$\mu^*(T)$')):
            ax.plot([r['omega'] for r in selected],[r[key] for r in selected],marker='o',lw=1,color=color,label=f'T = {T:g}')
            ax.set(xlabel=r'$\omega$',ylabel=label);ax.minorticks_on();ax.tick_params(direction='out',top=False,right=False)
            for spine in ax.spines.values():spine.set_linewidth(.6)
            ax.margins(x=.01,y=.05)
    for ax,letter in zip(axes,'abc'):ax.text(0,1.022,f'({letter})',transform=ax.transAxes,ha='left',va='bottom',fontsize=11)
    axes[0].legend(fontsize=9);fig.savefig(out/'Fig3.png',dpi=200);plt.close(fig)


def run(root, output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    checks=[];rows=[];settings=[];base=root/'data/centered';derived=base/'native/figures_main'
    with np.load(derived/'Fig1a_plot_data.npz',allow_pickle=False) as z:profile={k:z[k] for k in z.files}
    for case in (*BASELINE.values(),7706):
        metadata,s=load_native(root,case);cfg=metadata['configuration'];omega=int(cfg['omega']);P0=metadata['P0_init64']
        widths=s['t_right']-s['t_left'];n=len(widths)
        required=('step','t_left','t_right','roof_probability','mass_before','mass_stored','mass_solved',
                  'cast_mass_loss','clamp_mass_loss','solve_and_matrix_defect','p_spin_up','p_spin_down',
                  'p_spin_offdiag_real','p_spin_offdiag_imag')
        if not set(required)<=set(s):raise ValueError(f'case{case}: missing required schema')
        if any(v.shape!=(n,) for v in s.values()):raise ValueError(f'case{case}: inconsistent column shapes')
        if not all(np.isfinite(v).all() for v in s.values()):raise ValueError(f'case{case}: nonfinite native values')
        if s['step'].dtype!=np.dtype('int64'):raise ValueError('step dtype semantics changed')
        _check(checks,f'{case}: consecutive accepted steps',float(np.max(np.abs(s['step']-np.arange(1,n+1)))),0.)
        _check(checks,f'{case}: contiguous intervals',float(np.max(np.abs(s['t_left'][1:]-s['t_right'][:-1]))),0.)
        _check(checks,f'{case}: actual timestep vs configuration',float(np.max(np.abs(widths[:-1]-cfg['dt']))),1e-11)
        _check(checks,f'{case}: final support',abs(s['t_left'][0])+abs(s['t_right'][-1]-40.),1e-12)
        _check(checks,f'{case}: completed row count',abs(n-metadata['completed_step']),0.)
        _check(checks,f'{case}: recorded launch normalization',abs(s['mass_before'][0]-P0),1e-12)
        _check(checks,f'{case}: mass continuity',float(np.max(np.abs(s['mass_before'][1:]-s['mass_stored'][:-1]))),0.)
        if not np.all(s['roof_probability']>=0) or not (0<widths[-1]<=cfg['dt']+1e-11):raise ValueError('Invalid native event or final interval')
        expected={'Lz':20.,'z0_init':10.,'k0':np.pi,'k_bc':np.pi,'sigma_z':.5,'theta':0.,'phi':0.,'endpoint_mass_weight':.5}
        for key,value in expected.items():_check(checks,f'{case}: configuration {key}',abs(float(cfg[key])-value),1e-12)
        if cfg['endpoint_closure']!='centered' or cfg['metric']!='half_weight_last_z_node':raise ValueError('Wrong numerical implementation family')
        defect=s['mass_before']-s['mass_stored']-s['roof_probability']-s['cast_mass_loss']-s['clamp_mass_loss']-s['solve_and_matrix_defect']
        _check(checks,f'{case}: local accounting identity',float(np.max(np.abs(defect))),1e-12)
        _check(checks,f'{case}: spin event trace',float(np.max(np.abs(s['p_spin_up']+s['p_spin_down']-s['roof_probability']))),1e-12)
        _check(checks,f'{case}: cumulative event metadata',abs(float(np.sum(s['roof_probability']))-metadata['cumulative_roof_probability']),1e-11)
        settings.append(dict(case_id=case,omega=omega,actual_dt=cfg['dt'],actual_dt_min=float(widths.min()),actual_dt_max=float(widths.max()),
            steps=n,P0_init64=P0,Nx=cfg['Nx'],Nz=cfg['Nz'],actual_roof=cfg['z_roof_actual'],
            manuscript_baseline_dt=MANUSCRIPT_DT[list(BASELINE).index(omega)]*1e-4 if case!=7706 else '',
            state_dtype=cfg['state_dtype'],coordinate_dtype=cfg['real_dtype'],matrix_assembly=cfg['matrix_assembly']))
        for T in HORIZONS:rows.append(dict(case_id=case,omega=omega,**horizon_statistics(metadata,s,T)))
        if case==DISPLAY.get(omega) and f'omega{omega}_t' in profile:
            t=(s['t_left']+s['t_right'])/2;g=s['roof_probability']/(P0*widths)
            _check(checks,f'{case}: Fig1 archived native times',float(np.max(np.abs(t-profile[f'omega{omega}_t']))),0.)
            _check(checks,f'{case}: Fig1 archived native rates',float(np.max(np.abs(g-profile[f'omega{omega}_g']))),2e-14)
    _write_csv(output/'main_native_horizons.csv',rows);_write_csv(output/'actual_settings_vs_manuscript.csv',settings)
    baseline40=[next(r for r in rows if r['case_id']==case and r['T']==40) for case in BASELINE.values()]
    omega=np.array(list(BASELINE),dtype=float)
    y=np.array([r['mu_star_original_simpson'] for r in baseline40]);linear=np.array([r['mu_star_linear'] for r in baseline40])
    design=np.column_stack((np.ones(len(omega)),np.sqrt(omega)))
    A,B=np.linalg.lstsq(design,y,rcond=None)[0];Al,Bl=np.linalg.lstsq(design,linear,rcond=None)[0]
    recipe=json.loads((derived/'Fig2_recipe.json').read_text())['fit']
    _check(checks,'Fig2 original Simpson fit intercept',abs(A-recipe['A']),2e-11)
    _check(checks,'Fig2 original Simpson fit slope',abs(B-recipe['B']),2e-12)
    with (base/'tables/main_Fig2_selected_runs_and_fit.csv').open(newline='') as f:archived=list(csv.DictReader(f))
    for old,current in zip(archived,baseline40):
        if int(old['omega'])!=current['omega']:raise ValueError('Archived Fig2 row ordering mismatch')
        _check(checks,f"Fig2 omega{current['omega']}: archived Simpson observable",abs(float(old['mu_star_T40'])-current['mu_star_original_simpson']),2e-11)
    displayed=[next(r for r in rows if r['case_id']==case and r['T']==T) for T in HORIZONS for case in DISPLAY.values()]
    with np.load(derived/'Fig3_plot_data.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['omega'],list(DISPLAY));np.testing.assert_array_equal(z['horizons'],HORIZONS)
        for oldkey,newkey in [('D_T','D_T'),('conditional_mean','conditional_mean'),('mu_star','mu_star_linear')]:
            actual=np.array([r[newkey] for r in displayed]).reshape(4,8)
            _check(checks,f'Fig3 archived {oldkey}',float(np.max(np.abs(actual-z[oldkey]))),2e-11)
    _write_csv(output/'Fig2_baseline_T40.csv',baseline40);_write_csv(output/'Fig3_displayed_horizons.csv',displayed)
    fit=dict(original_Simpson=dict(A=float(A),B=float(B),RMSE=float(np.sqrt(np.mean((y-design@np.array([A,B]))**2)))),
             stated_piecewise_linear=dict(A=float(Al),B=float(Bl)),
             max_abs_Simpson_minus_linear=float(np.max(np.abs(y-linear))),
             note='Same baseline eight histories; integration methods kept separate. Fit residual is not convergence error.')
    (output/'fit_method_comparison.json').write_text(json.dumps(fit,indent=2)+'\n')
    _plot_profiles(profile,output);_plot_fit(omega,y,A,B,output);_plot_horizons(displayed,output)
    return dict(group='main',checks=checks,fit=fit,baseline_T40=baseline40,
                actual_settings=settings,max_abs_event_survival_discrepancy=max(abs(r['D_minus_one_minus_S']) for r in rows),
                figures=['Fig1a.png','Fig1b.png','Fig2.png','Fig3.png'],
                statuses={'main':'native reconstruction plus archived-derived replot; not exact manuscript-image reproduction',
                          'Fig2':'baseline7100; original Simpson fit and separate SI linear check',
                          'Fig1_and_Fig3':'7706 atomega300; other mapped histories unchanged'},
                limitations=['Finite-time event/survival mismatch is retained, not corrected into detector clicks.',
                             'Saved-data checks do not establish new simulation convergence or fresh restart capability.'])

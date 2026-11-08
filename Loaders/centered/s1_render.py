"""Archived S1 rendering functions; portable native-data adapter calls these.

Function bodies retained from the identified Oct18 S1 plotting recipe. Output is
numerical reconstruction, not a claim of exact embedded-manuscript image identity.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
BLUE, ORANGE, GREEN = '#34566F', '#B5654D', '#65866D'

def require(condition, message):
    if not condition:
        raise ValueError(message)

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

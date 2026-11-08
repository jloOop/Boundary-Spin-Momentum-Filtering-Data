"""Maintainer packaging: explicit mapped main inputs, never a production solver.

Usage: python Loaders/centered/package_main.py --research-root /path/to/R
This is not needed by the public saved-data workflow. Original files stay read-only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from extract import extract_run, copy_asset, asset, sanitize, sha256

RUNS = {
5540: '1.Main/5540.Spin_Gauss_centeredABC_Omega=1_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3701390',
5042: '1.Main/5042.Spin_Gauss_centeredABC_Omega=25_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704244',
5047: '1.Main/5047.Spin_Gauss_centeredABC_Omega=50_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3715452',
5043: '1.Main/5043.Spin_Gauss_centeredABC_Omega=75_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704245',
5541: '1.Main/5541.Spin_Gauss_centeredABC_Omega=100_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3701391',
5044: '1.Main/5044.Spin_Gauss_centeredABC_Omega=150_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704246',
5045: '1.Main/5045.Spin_Gauss_centeredABC_Omega=200_L=20_T=40_Nz=1500_complex64_Confinement_v1_full_job3704247',
7100: '1.Main/7100.Spin_Gauss_centeredABC_Omega=300_L=20_T=40_Nxy=100_Nz=1500_dt=0.0002_baseline_FG_v1_full_job3710751',
7706: '1.Main/7706.Spin_Gauss_centeredABC_Omega=300_L=20_zc=10_T=40_Nxy=100_Nz=1500_dt=0.0001_baseline_half_dt_v1_full_job3724723',
}
PACKAGES = {
'Fig1a': ('Fig1_Tol_swapped100_150_upto300_2026-10-23', 'Fig1a_centered.png', 'plot_fig1_centered.py'),
'Fig1b': ('Fig1b_Tol_teal150_refined300_2026-10-23', 'Fig1b_centered.png', 'plot_fig1b_centered.py'),
'Fig2': ('Fig2_centered_without400_lower_inset_2026-10-18', 'mu_Fit.png', 'plot_fig2_centered.py'),
'Fig3': ('Fig3_screenshot_palette_refined300_2026-10-23', 'horizon_comparison1.png.png', 'plot_fig3_centered.py'),
}


def package(research_root, root):
    research_root, root = Path(research_root).resolve(), Path(root).resolve()
    manifest = dict(group='main', assets=[], runs=[], paper_items=[], packaging_checks=[])
    for case, relative in RUNS.items():
        result = extract_run(research_root/relative, root, case, research_root)
        manifest['runs'].append(result['run']); manifest['assets'].extend(result['assets'])
        print(f'Packaged case {case}: {result["run"]["steps"]} native steps', flush=True)
    data = root/'data/centered'; derived=data/'native/figures_main'; derived.mkdir(parents=True,exist_ok=True)
    for label, (directory, image, script) in PACKAGES.items():
        source = research_root/'1.Main'/directory
        for filename, destination in [(image,data/'figures'/f'{label}.png'),
                                       (script,root/'Loaders/centered/historical'/f'{label}_original.py')]:
            manifest['assets'].append(copy_asset(source/filename,destination,root,research_root,sanitize_text=filename.endswith('.py')))
        if label in ('Fig1a','Fig3'):
            manifest['assets'].append(copy_asset(source/'plot_data.npz',derived/(label+'_plot_data.npz'),root,research_root))
        table = 'selected_runs_and_fit.csv' if label=='Fig2' else 'horizon_statistics.csv' if label=='Fig3' else 'selected_runs.csv'
        table_destination=data/'tables'/f'main_{label}_{table}'
        manifest['assets'].append(copy_asset(source/table,table_destination,root,research_root,sanitize_text=True))
        audit=json.loads((source/'audit_provenance.json').read_text())
        # Retain only the figure's scientific recipe/settings, not historic device paths or private working notes.
        keep = {k:audit[k] for k in ('fit','definitions','endpoint_handling','roundoff_endpoint_tolerance','processing','detector_method','D40','S40','discrepancy','figures','figure','horizons') if k in audit}
        keep['identity_status']='identified archived export; exact October28 embedded-image identity unconfirmed'
        keep['source_package']='R/1.Main/'+directory
        dest=derived/(label+'_recipe.json');dest.write_text(json.dumps(sanitize(keep,research_root),indent=2)+'\n')
        manifest['assets'].append(asset(source/'audit_provenance.json',dest,root,research_root,'selected scientific recipe fields; machine paths aliased'))
        selection=[5540,5047,5541,5044,5045,7706] if label=='Fig1a' else [5540,5047,5044,7706] if label=='Fig1b' else list(RUNS)[:8] if label=='Fig2' else list(RUNS)[:7]+[7706]
        manifest['paper_items'].append(dict(paper_item=label, displayed_asset=f'data/centered/figures/{label}.png',
             input_bundle='data/centered/native/figures_main', native_cases=selection,
             reduction_code='Loaders/centered/main.py', historical_recipe=f'Loaders/centered/historical/{label}_original.py',
             reproduction_level='archived figure display; derived replot; native observable reconstruction',
             identity_status=keep['identity_status'],
             qualification='Fig2 uses baseline7100 and original Simpson survival recipe; Fig1/3 use7706 atomega300; piecewise-linear method reported separately'))
    # Shared Fig1 data contains the separately selected b-panel's four exact curves.
    with np.load(research_root/'1.Main'/PACKAGES['Fig1a'][0]/'plot_data.npz',allow_pickle=False) as a, np.load(research_root/'1.Main'/PACKAGES['Fig1b'][0]/'plot_data.npz',allow_pickle=False) as b:
        assert all(np.array_equal(a[k],b[k]) for k in b.files)
    manifest['packaging_checks'].append({'name':'Fig1b exact subset of shared Fig1a archived arrays','passed':True,
        'source_path':'R/1.Main/'+PACKAGES['Fig1b'][0]+'/plot_data.npz',
        'source_sha256':sha256(research_root/'1.Main'/PACKAGES['Fig1b'][0]/'plot_data.npz')})
    manifest['paper_items'].append(dict(paper_item='Table S1',native_cases=list(RUNS)[:8],input_bundle='data/centered/native',reduction_code='Loaders/centered/main.py',reproduction_level='configuration/source verification',qualification='Manuscript dt differs from actual records; both vectors retained in output and PROJECT_SCOPE.md'))
    target=root/'build/packaging/main_manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--research-root',required=True,type=Path)
    parser.add_argument('--repository-root',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args()
    result=package(args.research_root,args.repository_root)
    print('Main packaged bytes:',sum(a['size_bytes'] for a in result['assets']))

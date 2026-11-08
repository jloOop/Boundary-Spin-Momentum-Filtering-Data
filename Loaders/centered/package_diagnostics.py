"""Opt-in maintainer packaging of mapped diagnostic inputs; no solver execution."""
from pathlib import Path
import argparse
import ast
import json
from centered.extract import copy_asset, extract_run, asset, sha256


def package(research_root, repository_root):
    R, root = Path(research_root).resolve(), Path(repository_root).resolve()
    if root.is_relative_to(R):
        raise ValueError('Repository output must be outside the original research tree')
    mapped = {
        7101: '3.Convergent_centered/7101.Spin_Gauss_centeredABC_Omega=300_L=20_T=40_Nxy=100_Nz=2999_dt=0.0002_normal_fixed_FG_v1_full_job3710752',
        7102: '3.Convergent_centered/7102.Spin_Gauss_centeredABC_Omega=300_L=20_T=40_Nxy=120_Nz=1500_dt=0.0002_transverse_fixed_FG_v1_full_job3710753',
    }
    assets, runs = [], []
    for case, folder in mapped.items():
        result = extract_run(R/folder, root, case, R)
        assets.extend(result['assets']); runs.append(result['run'])
    tables = R/'1.Main/Tables_S2_S3_S4_centered_2026-10-18'
    fixed = R/'ABC_Centered_EightPoint_Chiral_Evaluation_2026-10-17/ABC_FixedGeometry_7100_7101_7102_Evaluation_2026-10-19'
    cutoff = R/'1.Main/S1_S3_Recalculation_Omega400_CutoffComparison_2026-10-20'
    files = {
        tables/'excluding_omega400/Table_S2_full_precision.csv': 'data/centered/tables/diagnostics/Table_S2_archived.csv',
        tables/'excluding_omega400/Table_S3_full_precision.csv': 'data/centered/tables/diagnostics/Table_S3_archived.csv',
        tables/'excluding_omega400/Table_S4_full_precision.csv': 'data/centered/tables/diagnostics/Table_S4_archived.csv',
        tables/'reference_scalar_stats.csv': 'data/centered/tables/diagnostics/scalar_archived.csv',
        tables/'reference_scalar_provenance.json': 'data/centered/tables/diagnostics/scalar_provenance.json',
        tables/'native_plane_budget_full_precision.csv': 'data/centered/tables/diagnostics/native_plane_budget_archived.csv',
        tables/'numerical_accounting.csv': 'data/centered/tables/diagnostics/numerical_accounting_archived.csv',
        tables/'supplementary_integrals.csv': 'data/centered/tables/diagnostics/supplementary_integrals_archived.csv',
        fixed/'results/core_T40.csv': 'data/centered/tables/diagnostics/Table_S5_three_run_archived.csv',
        fixed/'results/pairwise_discrepancies.csv': 'data/centered/tables/diagnostics/Table_S5_pairwise_archived.csv',
        cutoff/'data/Table_S2_cut3.0_original_eight_full_precision.csv': 'data/centered/tables/diagnostics/cutoff3.0_archived.csv',
        cutoff/'data/Table_S2_cut3.6_original_eight_full_precision.csv': 'data/centered/tables/diagnostics/cutoff3.6_archived.csv',
        cutoff/'data/cutoff_summary.csv': 'data/centered/tables/diagnostics/cutoff_summary_archived.csv',
        R/'1.Main/FigS1_centered_omega300_2026-10-18/branches.png': 'data/centered/figures/FigS1.png',
        R/'1.Main/FigS1_centered_omega300_2026-10-18/plot_data.npz': 'data/centered/native/figures_diagnostics/FigS1_plot_data.npz',
        tables/'recalculate_tables.py': 'Loaders/centered/historical/recalculate_tables.py',
        tables/'reference_scalar_analysis.py': 'Loaders/centered/historical/reference_scalar_analysis.py',
        fixed/'analysis/compare.py': 'Loaders/centered/historical/compare_refinement.py',
        fixed/'analysis/native_tools.py': 'Loaders/centered/historical/native_tools.py',
        R/'1.Main/FigS1_centered_omega300_2026-10-18/plot_figS1_centered.py': 'Loaders/centered/historical/plot_figS1_centered.py',
    }
    for source, relative in files.items():
        assets.append(copy_asset(source, root/relative, root, R, sanitize_text=source.suffix in ('.py','.json','.csv')))
    # Reuse the archived rendering functions, separated from their native CSV loader.
    original = R/'1.Main/FigS1_centered_omega300_2026-10-18/plot_figS1_centered.py'
    text = original.read_text(); tree = ast.parse(text)
    names = {'require','style_axis','checked_line','legend','render'}
    functions = [ast.get_source_segment(text, node) for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    rendering = '''"""Archived S1 rendering functions; portable native-data adapter calls these.

Function bodies retained from the identified Oct18 S1 plotting recipe. Output is
numerical reconstruction, not a claim of exact embedded-manuscript image identity.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
BLUE, ORANGE, GREEN = '#34566F', '#B5654D', '#65866D'

'''+ '\n\n'.join(functions)+'\n'
    destination = root/'Loaders/centered/s1_render.py'
    destination.write_text(rendering)
    assets.append(asset(original,destination,root,R,'AST selection of archived rendering functions; no native CSV loader or source-path dependency'))
    paper_items = [
        {'paper_item':'Fig. S1', 'display':'data/centered/figures/FigS1.png',
         'inputs':['data/centered/native/5540','data/centered/native/7100','data/centered/native/figures_diagnostics/FigS1_plot_data.npz'],
         'reduction_code':'Loaders/centered/diagnostics.py', 'source_runs':[5540,7100],
         'reproduction_level':'archived export display; native reconstruction and archived-data comparison; no exact manuscript-image claim'},
        {'paper_item':'Tables S2-S4', 'inputs':['data/centered/native/'+str(k) for k in [5540,5042,5047,5043,5541,5044,5045,7100]],
         'tables':'data/centered/tables/diagnostics/', 'reduction_code':'Loaders/centered/diagnostics.py',
         'reproduction_level':'spinor rows recomputed from compact native records; scalar rows archived only; field-level spectral/monitor reconstruction unavailable'},
        {'paper_item':'Table S5', 'source_runs':[7100,7102,7101,7706],
         'inputs':['data/centered/native/'+str(k) for k in [7100,7102,7101,7706]],
         'reduction_code':'Loaders/centered/diagnostics.py',
         'reproduction_level':'new native-data reconstruction of Eq. S117; archived three-run product cross-check; four-run original exporter absent',
         'notes':'Nz2999 fixed numerical boundaries; later Nz4000 excluded. Printed baseline D=.739 differs from native .739745 rounding.'},
    ]
    result = {'assets':assets,'runs':runs,'paper_items':paper_items,
              'numerical_definitions':'Actual stored diagnostic clocks; piecewise-linear boundary moments; exact valid-interval product; native interval event rates; EqS117 baseline denominators and zero-crossing splits.',
              'limitations':['Scalar native histories absent; legacy one-sided full-weight scalar implementation.',
                             'Complex spectral fields and monitor currents not bundled; auxiliary continuation is not a physical absorbing layer.',
                             'Run completion and agreement of reductions are not convergence certification.']}
    out=root/'build/packaging/diagnostics_manifest.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Packaged diagnostic assets: {len(assets)}; native runs: {list(mapped)}')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root',type=Path,required=True)
    p.add_argument('--repository-root',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args();package(a.research_root,a.repository_root)

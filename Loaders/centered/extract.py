"""Lossless, explicit native-column packaging; never runs a generating solver.

This is an opt-in maintainer utility. The public CPU reproduction workflow reads
only the compact products and does not need the original research directory.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import io
import zipfile
import numpy as np

__all__ = ['sha256', 'asset', 'copy_asset', 'sanitize', 'extract_run']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def sanitize(value, research_root=None):
    """Alias path strings only; never change numerical arrays or status flags."""
    if isinstance(value, dict):
        return {k: sanitize(v, research_root) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, research_root) for v in value]
    if isinstance(value, str):
        if research_root:
            value = value.replace(str(Path(research_root).resolve()), 'R')
        if value.startswith(('/Users/', '/scratch/', '/home/', '/mnt/', '/cluster/')):
            return 'UNBUNDLED/' + Path(value).name
    return value


def asset(source, destination, repo_root, research_root, transform='byte-identical copy', **extra):
    source, destination = Path(source), Path(destination)
    return dict(path=str(destination.relative_to(repo_root)), sha256=sha256(destination),
                source_path='R/' + str(source.relative_to(research_root)),
                source_sha256=sha256(source), size_bytes=destination.stat().st_size,
                transformation=transform, **extra)


def copy_asset(source, destination, repo_root, research_root, sanitize_text=False):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transform = 'byte-identical copy'
    if sanitize_text:
        text = source.read_text()
        # Scientific strings/values remain unchanged; replace this machine's prefix.
        text = text.replace(str(Path(research_root).resolve()), 'R')
        # Only machine-specific absolute path tokens, not generic /dev or /tmp use.
        text = re.sub(r'/(?:Users|scratch|home|mnt|cluster)/[^\s\"\'<>]+',
                      lambda m: 'UNBUNDLED/' + Path(m.group(0)).name, text)
        destination.write_text(text)
        if destination.read_bytes() != source.read_bytes():
            transform = 'path strings aliased to R or UNBUNDLED; scientific values unchanged'
    else:
        shutil.copyfile(source, destination)
    return asset(source, destination, repo_root, research_root, transform)


def extract_run(source_dir, repo_root, case_id, research_root=None):
    """Export every row of explicitly selected CSV columns to steps.npz.

    All numerical/scientific CSV columns are retained except wall-clock timing.
    Decimal strings are parsed as IEEE binary64; step is verified integral then
    stored as int64. Export equality is checked column-by-column after reloading.
    """
    source_dir, repo_root = Path(source_dir).resolve(), Path(repo_root).resolve()
    research_root = Path(research_root).resolve() if research_root else source_dir.parent.parent
    case_id = int(case_id)
    if case_id <= 0 or repo_root.is_relative_to(research_root):
        raise ValueError('Destination must be a separate checkout outside the research root')
    if not source_dir.is_relative_to(research_root):
        raise ValueError('Source must be within the explicit research root')
    destination = repo_root / 'data' / 'centered' / 'native' / str(case_id)
    destination.mkdir(parents=True, exist_ok=True)
    src = source_dir / 'revision_step_data.csv'
    with src.open(newline='') as f:
        header = next(csv.reader(f))
    columns = [k for k in header if k != 'step_wall_seconds']
    required = {'step', 't_left', 't_right', 'roof_probability', 'mass_before',
                'mass_stored', 'cast_mass_loss', 'clamp_mass_loss', 'solve_and_matrix_defect'}
    if not required.issubset(columns):
        raise ValueError(f'Missing required native columns: {required - set(columns)}')
    source_hash = sha256(src)
    parsed = np.loadtxt(src, delimiter=',', skiprows=1, dtype=np.float64,
                        usecols=[header.index(k) for k in columns], ndmin=2)
    arrays = {k: np.array(parsed[:, i], dtype=np.float64) for i, k in enumerate(columns)}
    if not np.all(np.isfinite(parsed)):
        raise ValueError('Nonfinite source record; refusing silent alteration')
    if not np.array_equal(arrays['step'], arrays['step'].astype(np.int64)):
        raise ValueError('Nonintegral step identifier')
    arrays['step'] = arrays['step'].astype(np.int64)
    target = destination / 'steps.npz'
    # Standard NPZ container with LZMA members: NumPy uses Python zipfile/lzma.
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_LZMA) as archive:
        for key, values in arrays.items():
            stream = io.BytesIO()
            np.save(stream, values, allow_pickle=False)
            archive.writestr(key + '.npy', stream.getvalue())
    with np.load(target, allow_pickle=False) as z:
        for key, value in arrays.items():
            np.testing.assert_array_equal(z[key], value, err_msg=f'Export mismatch: {case_id}/{key}')
            assert z[key].dtype == value.dtype
    if sha256(src) != source_hash:
        raise ValueError('Original CSV changed during extraction')
    assets = [asset(src, target, repo_root, research_root,
                    'all selected columns/all rows parsed to binary64 (step int64), ZIP-LZMA; no resampling or numerical correction',
                    selected_columns=columns, omitted_columns=['step_wall_seconds'],
                    parser='numpy.loadtxt decimal text -> IEEE754 float64; integral step -> int64',
                    shapes={k: list(v.shape) for k, v in arrays.items()},
                    dtypes={k: v.dtype.str for k, v in arrays.items()},
                    array_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()},
                    source_column_equality=True)]
    run_src = source_dir / 'revision_run.json'
    original = json.loads(run_src.read_text())
    record = sanitize(original, research_root)
    (destination / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    assets.append(asset(run_src, destination / 'run.json', repo_root, research_root,
                        'machine paths aliased; original numerical configuration/status preserved'))
    for filename in ('branch_refinement_case.json', 'constants.npz', 'revision_coordinates.npz',
                     'revision_spin_branch_diagnostics.npz', 'plane_times.npy', 'roof_rho_int.npy',
                     'P_total_on_plane_times.npy', 'prob_times.npy', 'total_probs.npy'):
        source = source_dir / filename
        if source.is_file():
            if source.suffix == '.json':
                (destination/filename).write_text(json.dumps(sanitize(json.loads(source.read_text()), research_root), indent=2)+'\n')
                assets.append(asset(source, destination/filename, repo_root, research_root, 'machine paths aliased; numerical values unchanged'))
            else:
                assets.append(copy_asset(source, destination/filename, repo_root, research_root))
    source_target = repo_root/'Solvers'/'centered'/f'case_{case_id}.py'
    assets.append(copy_asset(source_dir/'source_snapshot.py', source_target, repo_root, research_root, sanitize_text=True))
    cfg = original['configuration']
    left, right = arrays['t_left'], arrays['t_right']
    run = dict(case_id=int(case_id), source_run='R/'+str(source_dir.relative_to(research_root)),
               distributed_directory=str(destination.relative_to(repo_root)),
               source_snapshot=str(source_target.relative_to(repo_root)),
               source_snapshot_original_sha256=sha256(source_dir/'source_snapshot.py'),
               source_matches_recorded=sha256(source_dir/'source_snapshot.py')==original['updated_source_sha256'],
               configuration=sanitize(cfg, research_root), P0_init64=original['P0_init64'],
               steps=len(left), support=[float(left[0]),float(right[-1])],
               dt_min=float(np.min(right-left)), dt_max=float(np.max(right-left)),
               dt_final=float(right[-1]-left[-1]),
               extraction_equality_verified=True,
               reproduction_level='recompute observables from compact native records; no restart state')
    return {'run':run, 'assets':assets}

#!/usr/bin/env python3
"""Reproduce signed-current figures from all native endpoint samples.

Example:
  MPLCONFIGDIR=/private/tmp/lowerwall_mpl /opt/anaconda3/bin/python \
    scripts/plot_currents.py --root /absolute/path/to/5.BottumWall/Output \
    --out /absolute/path/to/S9_4_fixed_source_T40_2026-10-20

The PDFs contain vector rectangles, never an image of the map. Adjacent native
cells with exactly the same fixed-LUT color are combined without altering the
color-mapped result. PNGs use the same native cells and the same color indices.
The common display cap is +/-0.65; larger magnitudes use endpoint colors, as
marked by the colorbar extensions. This changes only the display mapping.
Use --preview-only for 200-dpi previews without vector merging or final exports.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lowerwall_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.colors import ListedColormap, NoNorm, SymLogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np


VLIM = 0.65
LINTHRESH = 0.002
COLOR_TICKS = [-0.1, -0.01, 0.0, 0.01, 0.1]
STYLE_PRECEDENT = (
    "2_FirstAcuuracy_Scripts_Run/R4_ScalarABC_check_LowereWall_Check/Output/"
    "Fig_S11_reproduction/reproduce_Fig_S11.py"
)
XLIM = (0.0, 40.0)
YLIM = (-15.0, 20.3)
MONITORS = (16.006667137145996, 18.006667137145996)
CASE_IDS = (7750, 7751, 7752, 7753, 7754)
NORM = SymLogNorm(linthresh=LINTHRESH, linscale=1.0, base=10,
                  vmin=-VLIM, vmax=VLIM, clip=False)
CMAP = matplotlib.colormaps["RdBu_r"].resampled(256)
LUT = np.asarray(CMAP(np.arange(256)), dtype=np.float64)
INDEXED_CMAP = ListedColormap(LUT, name="RdBu_r_fixed256")


@dataclass
class Case:
    case_id: int
    path: Path
    omega: float
    wall: float
    z: np.ndarray
    t: np.ndarray
    t_edges: np.ndarray
    current: np.ndarray
    color_index: np.ndarray
    metadata: dict
    # rectangle arrays: temporal start/end, spatial start/end, color
    rectangles: np.ndarray | None = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_case(root: Path, case_id: int) -> Case:
    found = sorted(root.glob(f"{case_id}.Spin*"))
    if len(found) != 1:
        raise ValueError(f"Expected one case {case_id}; found {found}")
    path = found[0]
    configuration = json.loads((path / "revision_run.json").read_text())["configuration"]
    with np.load(path / "longitudinal_history.npz", allow_pickle=False) as data:
        t = np.array(data["profile_time"])
        z = np.array(data["z"])
        j = np.array(data["bond_current"])
        raw = np.array(data["bond_current_raw"])
        p0 = float(data["initial_norm_squared"])
        monitor_z = np.array(data["monitor_bond_z"])
        bond_z = np.array(data["bond_z"])
        state = str(data["profile_state"])
        normalization = str(data["normalization"])
    assert state == "stored_endpoint", state
    assert np.all(np.isfinite(j)) and np.all(np.diff(t) > 0) and np.all(np.diff(z) > 0)
    assert t[0] == 0 and t[-1] == 40
    assert j.shape == (len(t), len(z) - 1)
    assert np.array_equal(j, raw / p0), "Unexpected current normalization"
    assert np.array_equal(monitor_z, np.asarray(MONITORS))
    assert np.array_equal(bond_z, (z[:-1] + z[1:]) / 2)
    edges = np.empty(len(t) + 1)
    edges[1:-1] = (t[1:] + t[:-1]) / 2
    edges[0], edges[-1] = XLIM
    assert np.all(np.diff(edges) > 0)
    # This is exactly the fixed 256-entry Matplotlib LUT mapping used by cmap(norm).
    index = np.clip(np.floor(np.asarray(NORM(j)) * 256), 0, 255).astype(np.uint8)
    assert np.array_equal(LUT[index], CMAP(NORM(j))), "LUT mapping mismatch"
    positive_saturation_count = int(np.count_nonzero(j > VLIM))
    negative_saturation_count = int(np.count_nonzero(j < -VLIM))
    metadata = {
        "case_id": case_id,
        "source_directory": str(path),
        "source_npz_sha256": sha256(path / "longitudinal_history.npz"),
        "omega": float(configuration["omega"]),
        "z_bottom": float(z[0]),
        "z_detector": float(z[-1]),
        "profile_shape_time_by_bond": list(j.shape),
        "native_samples": int(j.size),
        "profile_time_first": float(t[0]),
        "profile_time_last": float(t[-1]),
        "profile_dt_min": float(np.diff(t).min()),
        "profile_dt_max": float(np.diff(t).max()),
        "initial_norm_squared": p0,
        "profile_state": state,
        "stored_normalization": normalization,
        "max_abs_raw_divided_P0_minus_normalized": float(np.max(np.abs(raw / p0 - j))),
        "raw_divided_P0_bitwise_equal_to_normalized": bool(np.array_equal(raw / p0, j)),
        "minimum_current": float(j.min()),
        "maximum_current": float(j.max()),
        "maximum_absolute_current": float(np.abs(j).max()),
        "samples_outside_shared_color_limits": int(np.count_nonzero(np.abs(j) > VLIM)),
        "maximum_excess_over_shared_color_limit": float(max(0, np.abs(j).max() - VLIM)),
        "positive_saturation_count": positive_saturation_count,
        "negative_saturation_count": negative_saturation_count,
        "positive_saturation_fraction": positive_saturation_count / int(j.size),
        "negative_saturation_fraction": negative_saturation_count / int(j.size),
        "total_saturation_fraction": (positive_saturation_count + negative_saturation_count) / int(j.size),
        "saturation_fraction_denominator": "all native endpoint-current cells; unweighted sample count",
        "monitor_bond_z": monitor_z.tolist(),
        "all_native_time_samples_retained": True,
        "all_native_bond_samples_retained": True,
    }
    return Case(case_id, path, float(configuration["omega"]), float(z[0]), z,
                t, edges, j, index, metadata)


def merge_identical_color_cells(case: Case) -> np.ndarray:
    """Merge z-runs, then extend only identical runs in adjacent time columns.

    Every rectangle is the union of same-color native cells. Verification expands
    each emitted rectangle into an integer grid and checks exact color identity
    and one-time coverage, so this operation cannot hide or alter native samples.
    """
    completed: list[tuple[int, int, int, int, int]] = []
    active: dict[tuple[int, int, int], int] = {}
    vertical_run_count = 0
    for it, row in enumerate(case.color_index):
        cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, row.size]
        vertical_run_count += len(cuts) - 1
        next_active = {}
        for lo, hi, color in zip(cuts[:-1].tolist(), cuts[1:].tolist(),
                                 row[cuts[:-1]].tolist()):
            key = (lo, hi, color)
            next_active[key] = active.pop(key, it)
        completed.extend((t0, it, lo, hi, color)
                         for (lo, hi, color), t0 in active.items())
        active = next_active
    completed.extend((t0, len(case.t), lo, hi, color)
                     for (lo, hi, color), t0 in active.items())
    rectangles = np.asarray(completed, dtype=np.int32)
    del completed
    reconstructed = np.empty_like(case.color_index)
    coverage = np.zeros_like(case.color_index)
    for t0, t1, z0, z1, color in rectangles:
        reconstructed[t0:t1, z0:z1] = color
        coverage[t0:t1, z0:z1] += 1
    assert np.all(coverage == 1), "Vector rectangles overlap or omit native cells"
    assert np.array_equal(reconstructed, case.color_index), "Vector colors changed"
    case.metadata.update({
        "vertical_color_runs": vertical_run_count,
        "merged_vector_rectangles": int(len(rectangles)),
        "color_mapping_reconstruction_exact": True,
        "native_cell_coverage_exactly_once": True,
    })
    case.rectangles = rectangles
    return rectangles


class NativeVectorCells(Artist):
    """Emit vector rectangles directly to Matplotlib's compressed PDF stream.

    Direct emission keeps memory bounded and avoids millions of Python Path
    objects. Graphics state is isolated by q/Q, and all numerical coordinates
    use round-trip precision. There are no images and no rasterization.
    """

    def __init__(self, case: Case):
        super().__init__()
        self.case = case
        self.set_zorder(1)
        self.set_rasterized(False)

    def draw(self, renderer):
        if not self.get_visible():
            return
        if not hasattr(renderer, "file"):
            raise RuntimeError("NativeVectorCells is intended only for PDF")
        case = self.case
        rects = case.rectangles
        assert rects is not None
        f = renderer.file
        matrix = self.axes.transData.get_affine().to_values()
        # Transform exact native data coordinates to PDF page coordinates.
        f.write(("q\n" + " ".join(format(v, ".17g") for v in matrix) + " cm\n"
                 + f"{XLIM[0]:.17g} {YLIM[0]:.17g} "
                 + f"{XLIM[1]-XLIM[0]:.17g} {YLIM[1]-YLIM[0]:.17g} re W n\n").encode())
        t_strings = [format(v, ".17g") for v in case.t_edges]
        z_strings = [format(v, ".17g") for v in case.z]
        # Group by color to set RGB at most 256 times per panel.
        order = np.argsort(rects[:, 4], kind="stable")
        sorted_rects = rects[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(sorted_rects[:, 4])) + 1, len(rects)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            rgb = LUT[sorted_rects[a, 4], :3]
            f.write((" ".join(format(v, ".17g") for v in rgb) + " rg\n").encode())
            batch: list[str] = []
            # Bound each compound path to 2048 rectangles for PDF-reader memory.
            for t0, t1, z0, z1, _ in sorted_rects[a:b]:
                width = case.t_edges[t1] - case.t_edges[t0]
                height = case.z[z1] - case.z[z0]
                batch.append(f"{t_strings[t0]} {z_strings[z0]} "
                             f"{width:.17g} {height:.17g} re\n")
                if len(batch) == 2048:
                    f.write(("".join(batch) + "f\n").encode())
                    batch.clear()
            if batch:
                f.write(("".join(batch) + "f\n").encode())
        f.write(b"Q\n")
        self.stale = False


def configure_axes(ax, case: Case, title: str):
    ax.set_facecolor("#e6e6e6")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15, 20])
    ax.set_title(title, loc="left", pad=6, fontsize=10.5)
    ax.tick_params(axis="both", length=3, width=0.6, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#404040")
    ax.axhline(case.wall, color="#414141", linewidth=0.75, zorder=3)
    ax.axhline(case.z[-1], color="#666666", linewidth=0.5, zorder=3)
    ax.axhline(MONITORS[0], color="#333333", linestyle="--", linewidth=0.65, zorder=3)
    ax.axhline(MONITORS[1], color="#333333", linestyle=":", linewidth=0.8, zorder=3)


def add_colorbar(fig, cax):
    scalar = ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(scalar, cax=cax, extend="both")
    # Matplotlib can rasterize colorbar solids automatically; explicitly forbid it.
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
    labels = [r"$-10^{-1}$", r"$-10^{-2}$", r"$0$", r"$10^{-2}$", r"$10^{-1}$"]
    cb.locator = FixedLocator(COLOR_TICKS)
    cb.update_ticks()
    cb.set_ticklabels(labels)
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.tick_params(labelsize=8, width=0.5, length=2.5, pad=3)
    cb.outline.set_linewidth(0.6)
    cb.set_label("Signed longitudinal current\n(positive toward detector)", fontsize=9, labelpad=8)
    return cb


def pdf_image_count(path: Path) -> tuple[int, str]:
    executable = shutil.which("pdfimages")
    if executable is None:
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfimages"
        if bundled.exists():
            executable = str(bundled)
    if executable is not None:
        result = subprocess.run([executable, "-list", str(path)], check=True,
                                capture_output=True, text=True)
        rows = [line for line in result.stdout.splitlines() if line.strip()
                and line.split()[0].isdigit()]
        return len(rows), result.stdout
    # The bundled runtime has pypdf even when the plotting interpreter does not.
    # Inspect resource dictionaries directly; avoid expanding millions of paths.
    bundled_python = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
    python = str(bundled_python) if bundled_python.exists() else sys.executable
    code = """
import json, sys
from pypdf import PdfReader
reader = PdfReader(sys.argv[1])
seen = set()
images = []
forms = 0
def visit(resources):
    global forms
    resources = resources.get_object() if hasattr(resources, 'get_object') else resources
    xobjects = resources.get('/XObject', {})
    xobjects = xobjects.get_object() if hasattr(xobjects, 'get_object') else xobjects
    for key, ref in xobjects.items():
        obj = ref.get_object()
        identity = (getattr(ref, 'idnum', None), getattr(ref, 'generation', None))
        if identity in seen:
            continue
        seen.add(identity)
        if obj.get('/Subtype') == '/Image':
            images.append(str(key))
        elif obj.get('/Subtype') == '/Form':
            forms += 1
            visit(obj.get('/Resources', {}))
for page in reader.pages:
    visit(page.get('/Resources', {}))
print(json.dumps({'image_xobjects': len(images), 'image_names': images,
                  'form_xobjects_examined': forms, 'pages': len(reader.pages)}))
"""
    result = subprocess.run([python, "-c", code, str(path)], check=True,
                            capture_output=True, text=True)
    record = json.loads(result.stdout)
    return record["image_xobjects"], result.stdout


def make_figure(cases: list[Case], out: Path, stem: str, multi: bool,
                preview_only: bool = False) -> dict:
    if multi:
        fig = plt.figure(figsize=(8.0, 6.5))
        grid = fig.add_gridspec(2, 2, left=.085, right=.805, bottom=.09,
                               top=.91, wspace=.12, hspace=.18)
        axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
        cax = fig.add_axes([.838, .18, .022, .67])
        for i, ax in enumerate(axes):
            if i < 2:
                ax.tick_params(labelbottom=False)
            if i % 2:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r"Longitudinal coordinate $z$", labelpad=6)
            if i >= 2:
                ax.set_xlabel(r"Time $t$", labelpad=4)
        fig.suptitle(r"$\omega=200$; fixed $z_c=10$", fontsize=11, y=.986)
    else:
        fig = plt.figure(figsize=(8.0, 3.8))
        ax = fig.add_axes([.085, .15, .72, .75])
        axes = [ax]
        cax = fig.add_axes([.838, .19, .022, .64])
        ax.set_xlabel(r"Time $t$", labelpad=4)
        ax.set_ylabel(r"Longitudinal coordinate $z$", labelpad=6)
    vectors = []
    for i, (ax, case) in enumerate(zip(axes, cases)):
        title = (f"({chr(97+i)}) " + rf"$z_B={case.wall:g}$") if multi else (
            rf"$\omega=1$, $z_B={case.wall:g}$; fixed $z_c=10$")
        configure_axes(ax, case, title)
        artist = NativeVectorCells(case)
        ax.add_artist(artist)
        vectors.append(artist)
    add_colorbar(fig, cax)
    pdf = out / f"{stem}.pdf"
    png = out / f"{stem}.png"
    png1200 = out / f"{stem}_1200dpi.png"
    preview = out / "validation" / f"{stem}_preview.png"
    meshes = []
    for ax, case, vector in zip(axes, cases, vectors):
        vector.set_visible(False)
        # NoNorm maps integer LUT indices directly, identical to the vector colors.
        # The x/z edges are native temporal midpoints and physical nodal z values.
        mesh = ax.pcolormesh(case.t_edges, case.z, case.color_index.T,
                             cmap=INDEXED_CMAP, norm=NoNorm(), shading="flat",
                             edgecolors="none", antialiased=False, rasterized=False,
                             zorder=1, snap=False)
        meshes.append(mesh)
    fig.savefig(preview, format="png", dpi=200)
    if preview_only:
        plt.close(fig)
        record = {"stem": stem, "preview": str(preview), "preview_dpi": 200,
                  "vmin": -VLIM, "vmax": VLIM, "linthresh": LINTHRESH,
                  "vector_merging_performed": False, "final_exports_written": False}
        print(json.dumps(record), flush=True)
        return record
    fig.savefig(png, format="png", dpi=600)
    fig.savefig(png1200, format="png", dpi=1200)
    for mesh in meshes:
        mesh.remove()
    for vector in vectors:
        vector.set_visible(True)
    start = time.monotonic()
    fig.savefig(pdf, format="pdf", metadata={
        "Title": "Fixed-source signed longitudinal-current maps",
        "Author": "Reproduced from saved native endpoint histories",
        "Subject": "Vector current cells; every native sample retained; common symmetric-log display cap +/-0.65",
        "Keywords": "omega200, omega1, fixed source, lower wall, signed current, native samples",
        "CreationDate": None, "ModDate": None,
    })
    pdf_seconds = time.monotonic() - start
    image_count, image_report = pdf_image_count(pdf)
    assert image_count == 0, f"PDF contains image XObjects: {pdf}"
    (out / "validation" / f"{stem}_pdfimages.txt").write_text(image_report)
    plt.close(fig)
    record = {
        "stem": stem,
        "case_ids": [case.case_id for case in cases],
        "size_inches": [8.0, 6.5] if multi else [8.0, 3.8],
        "png_dpi": 600,
        "png_pixels": [4800, 3900] if multi else [4800, 2280],
        "preview_dpi": 200,
        "png_1200dpi_file": png1200.name,
        "png_1200dpi_dpi": 1200,
        "png_1200dpi_pixels": [9600, 7800] if multi else [9600, 4560],
        "png_1200dpi_bytes": png1200.stat().st_size,
        "png_1200dpi_sha256": sha256(png1200),
        "pdf_image_xobjects": image_count,
        "pdf_true_vector": True,
        "pdf_bytes": pdf.stat().st_size,
        "png_bytes": png.stat().st_size,
        "pdf_render_generation_seconds": pdf_seconds,
        "pdf_sha256": sha256(pdf),
        "png_sha256": sha256(png),
        "preview": str(preview),
    }
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Raw lower-wall Output directory")
    parser.add_argument("--out", type=Path, required=True, help="Reproduction output directory")
    parser.add_argument("--preview-only", action="store_true",
                        help="Write only 200-dpi previews; no vector merging, PDFs, or final PNGs")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
        "pdf.compression": 9, "path.simplify": False, "savefig.facecolor": "white",
    })
    cases = [load_case(root, case_id) for case_id in CASE_IDS]
    assert np.isfinite([VLIM, LINTHRESH]).all() and VLIM > LINTHRESH > 0
    if args.preview_only:
        make_figure(cases[:4], out, "fig_S11_fixed_source_omega200_4panel", True, True)
        make_figure(cases[4:], out, "fig_S11_fixed_source_omega1_zBm15", False, True)
        return
    for case in cases:
        merge_identical_color_cells(case)
        print(f"case {case.case_id}: {case.current.size:,} native cells -> "
              f"{len(case.rectangles):,} exact-color vector rectangles", flush=True)
    figures = [
        make_figure(cases[:4], out, "fig_S11_fixed_source_omega200_4panel", True),
        make_figure(cases[4:], out, "fig_S11_fixed_source_omega1_zBm15", False),
    ]
    metadata = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "root": str(root),
        "output": str(out),
        "quantity": "stored endpoint bond_current = bond_current_raw / initial_norm_squared",
        "sign_convention": "positive toward the detector",
        "temporal_cell_edges": "midpoints of adjacent native profile_time samples; outer edges clipped to 0 and 40",
        "spatial_cell_edges": "native nodal z; each bond occupies its two endpoint nodes",
        "interpolation": "none",
        "smoothing": "none",
        "temporal_or_spatial_resampling": "none",
        "panel_shifting_or_panelwise_rescaling": "none",
        "normalization_applied_during_plot": "none; normalized bond_current validated against raw/P0",
        "same_color_cell_merging": "exact-color vertical runs, extended only when identical in adjacent time columns",
        "color_map": {
            "name": "RdBu_r",
            "lut_entries": 256,
            "lut_rgba_sha256": hashlib.sha256(LUT.tobytes()).hexdigest(),
            "normalization": "SymLogNorm",
            "vmin": -VLIM,
            "vmax": VLIM,
            "linthresh": LINTHRESH,
            "linscale": 1.0,
            "base": 10,
            "same_scale_for_both_figures": True,
            "per_panel_rescaling": False,
            "display_cap": True,
            "display_cap_note": "Common visual display limits +/-0.65; currents beyond these limits retain their numerical values and use the colormap endpoint colors.",
            "colorbar_extend": "both",
            "ticks": COLOR_TICKS,
            "style_precedent": STYLE_PRECEDENT,
            "style_precedent_lines": "38-40, 244-254, 270-276, 405",
            "normalization_provenance": "Matches the prior Fig_S11_reproduction display settings. That source explicitly reports inference from the reference PDF colorbar, not recovery of original plotting code. The display mapping is applied identically to all five unchanged current histories.",
        },
        "axes": {"time_limits": list(XLIM), "z_limits": list(YLIM),
                 "monitor_bond_z": list(MONITORS), "outside_guide_color": "#e6e6e6"},
        "cases": [case.metadata for case in cases],
        "figures": figures,
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

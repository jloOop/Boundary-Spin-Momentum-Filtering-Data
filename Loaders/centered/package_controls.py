#!/usr/bin/env python3
"""Package only mapped control assets; original research input is read-only.

Authoring-only command (not needed for saved-data reproduction):
python Loaders/centered/package_controls.py --source-root R --repo-root CHECKOUT
Requires NumPy, Pillow and pdftoppm for rendering existing PDFs.
No solver is imported or executed. No production directory is copied.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image

__all__ = ["package"]

FAMILIES = {
    "S4": ("4.Sweep_Scan_Controls/1.kappa_campaign/Output/Fig_S4_centered_T20_2026-10-20",
           "kappa_campaign.png", ["fig_S4_centered_T20.csv", "fig_S4_centered_fit_summary.csv"], ["recalculate_fig_S4.py"]),
    "S6": ("4.Sweep_Scan_Controls/2.kz_campaign/Output/Fig_S6_centered_T20_2026-10-20",
           "kz_campaign.png", ["fig_S6_centered_T20.csv", "fig_S6_centered_fit_summary.csv"], ["reproduce_fig_S6.py"]),
    "S8": ("4.Sweep_Scan_Controls/3.sigma_campaign/Output/Fig_S8_centered_T20",
           "Fig_S8_centered_T20_preview.png", ["results_full_precision.csv", "fit_coefficients.csv", "fit_residuals.csv"], ["reproduce_fig_s8.py"]),
    "S10": ("4.Sweep_Scan_Controls/4.Length_L=30_comparision/Output/S10_centered_T40_7points_2026-10-21",
            "figures/length_campaign.png", ["tables/S10_statistics_and_validation.csv", "tables/S10_fit_coefficients.csv"], ["scripts/reproduce_s10.py"]),
    "S11": ("4.Sweep_Scan_Controls/5.BottumWall/Output/S9_4_fixed_source_T40_2026-10-20",
            "fig_S11_fixed_source_omega200_4panel.png", ["table_S11_fixed_source_T40.csv", "table_S11_fixed_source_T40.tex", "plot_metadata.json"],
            ["scripts/plot_currents.py", "scripts/analyze_fixed_source_T40.py", "scripts/campaign_validation_support.py"]),
    "T6": ("2.S8/S8_1/S8_1_Table_S6_centered_2026-10-20", None,
           ["Table_S6.csv", "Table_S6.tex", "full_precision_results.csv", "numerical_validation.csv"],
           ["recalculate_s8_1.py", "independent_check.py"]),
    "T10": ("4.Sweep_Scan_Controls/2.kz_campaign/Output/Table_S10_centered_T20_2026-10-20", None,
            ["table_S10_centered_T20.csv", "table_S10_centered_T20.tex", "table_S10_raw_metrics_T20.csv"],
            ["recalculate_table_S10.py"]),
}
LOWER = "4.Sweep_Scan_Controls/6.Bottom_ABC_Check"
LONG = "4.Sweep_Scan_Controls/7.Long_Run_T=200"
LONG_STEM = "arrival_time_omega200_T200_inset_nested_black"
CONTROL_ITEMS = {
    "Fig. S2": "manuscript-reported; layer inputs and original export not bundled",
    "Fig. S3": "archived legacy one-sided profiles located but not bundled; unmatched preparation/native input qualifications retained",
    "Fig. S4": "archived figure/table display; parameter native recomputation and table-based curve regeneration not advertised",
    "Fig. S5": "archived legacy one-sided profiles located but not bundled; original native histories outside authorized source root",
    "Fig. S6": "archived figure/table display; parameter native recomputation and table-based curve regeneration not advertised",
    "Fig. S7": "manuscript-reported; original one-sided width inputs not located; incomplete centered reruns not substituted",
    "Fig. S8": "archived figure/table display; parameter native recomputation and table-based curve regeneration not advertised",
    "Fig. S9": "candidate export method conflicts with manuscript caption; not presented as exact reproduction",
    "Fig. S10": "archived figure/table display; parameter native recomputation and table-based curve regeneration not advertised",
    "Fig. S11": "archived current-map export and statistics; large spatial histories not bundled",
    "Fig. S12": "archived export; upper-detection observables recomputable from compact native channels",
    "Fig. S13": "raster of identified archived export; derived-bin replot and sampled-norm checks; full native step CSV not bundled",
    "Table S6": "archived bulk/chiral results; original every-step survival reduction documented, not independently reconstructed here",
    "Table S7": "manuscript-reported; generic layer inputs not bundled",
    "Table S8": "manuscript-reported; impedance-calibrated layer inputs not bundled",
    "Table S9": "manuscript-reported; legacy parameter-table input match incomplete",
    "Table S10": "archived table; original reducer has missing provenance helper",
    "Table S11": "archived fixed-source wall statistics; spatial/native large histories not bundled",
    "Table S12": "native upper/lower/survival statistics recomputable; archival comparison includes rounded Dirichlet rows",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def _sanitize(text, source_root):
    text = text.replace(str(source_root), "R")
    # Machine-local locations are provenance aliases, never changes to numeric fields.
    text = re.sub(r"/Users/[^/\s\"']+/Documents/EDIT_Spin-Momentum_filtering/3_CenteredABC_Run", "R", text)
    text = re.sub(r"/(?:scratch|home|Users|var|private/var)/[^\s\"'<>;,)\]}]+",
                  lambda m: "EXTERNAL/" + m.group().rstrip("/").split("/")[-1], text)
    return text

def package(source_root, repo_root, pdftoppm="pdftoppm"):
    source_root, repo_root = Path(source_root).resolve(), Path(repo_root).resolve()
    if "3DGifs_Simulations" in repo_root.parts:
        raise ValueError("Protected simulation media directory cannot be an output root")
    if source_root == repo_root or repo_root.is_relative_to(source_root):
        raise ValueError("Use a separate repository checkout outside the research root")
    assets, runs, equality_checks = [], [], []
    data = repo_root / "data/centered"
    for part in ["native/controls", "tables/controls", "figures"]:
        (data / part).mkdir(parents=True, exist_ok=True)

    def record(src, dest, transform, **extra):
        item = {"path": str(dest.relative_to(repo_root)), "size_bytes": dest.stat().st_size,
                "sha256": sha(dest), "source_path": "R/" + str(src.relative_to(source_root)),
                "source_sha256": sha(src), "transformation": transform, **extra}
        assets.append(item)
        return item

    def copy(src, dest, sanitize=False):
        src, dest = Path(src), Path(dest)
        if not src.resolve().is_relative_to(source_root):
            raise ValueError("Source outside authorized root")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if sanitize:
            text = src.read_bytes().decode("utf-8")
            cleaned = _sanitize(text, source_root)
            cosmetic_note_removed = False
            if src.name.endswith("_metadata.json") and "7810." in str(src):
                metadata = json.loads(cleaned)
                style = metadata.get("figure_style", {})
                for cosmetic in ("reference_image", "reference_figure"):
                    cosmetic_note_removed = (style.pop(cosmetic, None) is not None) or cosmetic_note_removed
                if cosmetic_note_removed:
                    cleaned = json.dumps(metadata, indent=2) + "\n"
            dest.write_bytes(cleaned.encode("utf-8"))
            if src.suffix.lower() == ".csv":
                before = list(csv.reader(io.StringIO(text)))
                after = list(csv.reader(io.StringIO(cleaned)))
                assert len(before) == len(after)
                numeric = 0
                for arow, brow in zip(before, after):
                    assert len(arow) == len(brow)
                    for a, b in zip(arow, brow):
                        try:
                            float(a)
                        except ValueError:
                            continue
                        assert a == b, "Numeric CSV token changed during path sanitization"
                        numeric += 1
                equality_checks.append({"name": str(dest.relative_to(repo_root)) + " numeric CSV tokens unchanged", "passed": True, "actual": numeric, "tolerance": 0})
            record(src, dest, ("machine-path text aliases; rendering-reference screenshot/conversational note omitted; numerical values and status flags unchanged" if cosmetic_note_removed else "machine-path text aliases only; numerical values and status flags unchanged") if cleaned != text else "byte-identical text copy")
        else:
            shutil.copyfile(src, dest)
            record(src, dest, "byte-identical copy")

    def image(src, dest, max_width=2400):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as image:
            original = image.size
            if image.width <= max_width:
                shutil.copyfile(src, dest)
                transform = "byte-identical archived PNG"
                final = image.size
            else:
                image = image.convert("RGB")
                image.thumbnail((max_width, 10000), Image.Resampling.LANCZOS)
                image.save(dest, optimize=True)
                transform = "complete-figure PNG resized with Pillow LANCZOS; no crop/recolor/data change"
                final = image.size
        record(src, dest, transform, source_dimensions=list(original), dimensions=list(final))

    for family, (relative, png, names, recipes) in FAMILIES.items():
        src = source_root / relative
        dest = data / "tables/controls" / family
        for name in names:
            copy(src / name, dest / Path(name).name, sanitize=True)
        for name in recipes:
            copy(src / name, repo_root / "Solvers/centered/controls/analysis" / family / Path(name).name, sanitize=True)
        if png:
            image(src / png, data / "figures" / f"Fig{family}.png")

    for case in ("7800", "7801"):
        found = list((source_root / LOWER / "Output").glob(case + ".*"))
        found = [p for p in found if p.is_dir()]
        if len(found) != 1:
            raise ValueError(f"Expected one mapped case {case}, found {len(found)}")
        src = found[0]
        dest = data / "native/controls" / case
        dest.mkdir(parents=True, exist_ok=True)
        original = src / "two_channel_history.npz"
        with np.load(original, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
        if any(a.dtype.hasobject for a in arrays.values()):
            raise ValueError("Object arrays are not permitted")
        np.savez_compressed(dest / "two_channel_history.npz", **arrays)
        schemas = {}
        with np.load(dest / "two_channel_history.npz", allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            for key, a in arrays.items():
                b = z[key]
                equal = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
                if not equal:
                    raise AssertionError(f"Recompression changed {case}:{key}")
                schemas[key] = {"shape": list(a.shape), "dtype": a.dtype.str,
                                "array_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        record(original, dest / "two_channel_history.npz", "lossless NPZ compression; all keys, rows, dtypes and values preserved; byte equality verified per array", arrays=schemas)
        equality_checks.append({"name": case + " NPZ all-array equality", "passed": True,
                                "actual": len(arrays), "tolerance": 0})
        for name in ["revision_run.json", "two_channel_summary.json", "branch_refinement_case.json"]:
            copy(src / name, dest / ("run.json" if name == "revision_run.json" else name), sanitize=True)
        for name in ["constants.npz", "revision_coordinates.npz"]:
            copy(src / name, dest / name)
        copy(src / "source_snapshot.py", repo_root / "Solvers/centered/controls" / ("case_" + case + ".py"), sanitize=True)
        config = json.loads((dest / "run.json").read_text())
        runs.append({"case": case, "source_run": "R/" + str(src.relative_to(source_root)), "configuration": config["configuration"],
                     "P0_init64": config["P0_init64"], "support": [float(arrays["time"][0]), float(arrays["time"][-1])],
                     "rate_normalization": "g_top/g_lower already divided by recorded P0_init64; never divide again",
                     "reproduction_level": "recompute observables from compact native channel records"})

    low_out = source_root / LOWER / "Output"
    image(low_out / "arrival_time_lower_boundary_comparison.png", data / "figures/FigS12.png")
    copy(low_out / "arrival_time_lower_boundary_comparison.json", data / "native/controls/S12_export.json", sanitize=True)
    copy(low_out / "plot_arrival_time.py", repo_root / "Solvers/centered/controls/analysis/S12/plot_arrival_time.py", sanitize=True)
    archive = source_root / LOWER / "LowerABC_7800_7801_Combined_Analysis.zip"
    with zipfile.ZipFile(archive) as z:
        for member in ["four_way_statistics.csv", "horizons_comparison.csv", "endpoint_comparison.csv"]:
            name = "LowerABC_Combined_Analysis/" + member
            raw = z.read(name)
            dest = data / "tables/controls/T12" / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            text = _sanitize(raw.decode(), source_root)
            dest.write_text(text)
            record(archive, dest, "selected ZIP member; machine-path text aliases only",
                   source_archive_member=name, source_member_sha256=hashlib.sha256(raw).hexdigest())

    src = [p for p in (source_root / LONG).glob("7810.*") if p.is_dir()]
    if len(src) != 1:
        raise ValueError("Expected one mapped long-time case")
    src = src[0]
    dest = data / "native/controls/7810"
    for name in ["revision_run.json", "prob_times.npy", "total_probs.npy", "constants.npz"]:
        copy(src / name, dest / ("run.json" if name == "revision_run.json" else name), sanitize=name.endswith(".json"))
    for suffix in ["_data.csv", "_metadata.json"]:
        copy(src / "output/pdf" / (LONG_STEM + suffix), dest / ("plot" + suffix), sanitize=True)
    copy(src / "source_snapshot.py", repo_root / "Solvers/centered/controls/case_7810.py", sanitize=True)
    copy(src / "plot_arrival_time.py", repo_root / "Solvers/centered/controls/analysis/S13/plot_arrival_time.py", sanitize=True)
    pdf = src / "output/pdf" / (LONG_STEM + ".pdf")
    # Preserve the complete identified archived export rather than redrawing it for display.
    target = data / "figures/FigS13"
    subprocess.run([pdftoppm, "-f", "1", "-singlefile", "-r", "150", "-png", str(pdf), str(target)], check=True)
    with Image.open(target.with_suffix(".png")) as im:
        dimensions = list(im.size)
    record(pdf, target.with_suffix(".png"), "first and only PDF page rasterized complete at150dpi using pdftoppm; no curve editing", dimensions=dimensions)
    cfg = json.loads((dest / "run.json").read_text())
    t = np.load(dest / "prob_times.npy", allow_pickle=False)
    runs.append({"case": "7810", "source_run": "R/" + str(src.relative_to(source_root)),
                 "configuration": cfg["configuration"], "P0_init64": cfg["P0_init64"],
                 "support": [float(t[0]), float(t[-1])], "manuscript_logged_estimate_endpoint": 189.95,
                 "reproduction_level": "derived-bin replot plus sampled-norm checks; original full accepted-step CSV not bundled"})

    fragment = {"assets": assets, "runs": runs, "paper_items": [
        {"paper_item": item, "reproduction_level": status,
         "reduction_code": "Loaders/centered/controls.py" if item not in ("Fig. S2", "Fig. S3", "Fig. S5", "Fig. S7", "Fig. S9", "Table S7", "Table S8", "Table S9") else None}
        for item, status in CONTROL_ITEMS.items()], "checks": equality_checks,
        "notes": [
            "All original research files remain unchanged; no protected media path accessed for writing.",
            "Parameter/bulk/chiral/fixed-wall tables are archived derived results, not independent native reconstruction.",
            "Historical recipes are retained for provenance and are never imported by the portable CPU workflow.",
            "FigS13 displayed PNG is a raster of the identified archived black-curve export; exact manuscript-image identity remains unconfirmed.",
            "FigS9 located candidate uses native events while the supplied caption says differentiated saved norms."
        ]}
    # One coherent paper-to-asset map, merged by the release verifier later.
    for item in fragment["paper_items"]:
        key = item["paper_item"]
        suffix = key.split()[-1]
        figure = data / "figures" / ("Fig" + suffix + ".png") if key.startswith("Fig.") else None
        item["displayed_asset"] = str(figure.relative_to(repo_root)) if figure and figure.exists() else None
        if key == "Table S11":
            selected = [a["path"] for a in assets if "/controls/S11/" in a["path"]]
        elif suffix in FAMILIES:
            folder = "T" + suffix[1:] if key.startswith("Table") else suffix
            selected = [a["path"] for a in assets if "/tables/controls/" + folder + "/" in a["path"]]
        elif key == "Fig. S12" or key == "Table S12":
            selected = [a["path"] for a in assets if any(x in a["path"] for x in ("/controls/7800/", "/controls/7801/", "/controls/T12/"))]
        elif key == "Fig. S13":
            selected = [a["path"] for a in assets if "/controls/7810/" in a["path"]]
        elif key == "Table S11":
            selected = [a["path"] for a in assets if "/controls/S11/" in a["path"]]
        elif key in ("Table S6", "Table S10"):
            folder = "T" + suffix[1:]
            selected = [a["path"] for a in assets if "/tables/controls/" + folder + "/" in a["path"]]
        else:
            selected = []
        item["distributed_inputs"] = selected
        item["generating_source_status"] = "Selected historical recipes retained under Solvers/centered/controls; no fresh-simulation or restart validation claimed."
    packaging = repo_root / "build/packaging"
    packaging.mkdir(parents=True, exist_ok=True)
    (packaging / "controls_manifest.json").write_text(json.dumps(fragment, indent=2, allow_nan=False) + "\n")
    print(f"Packaged {len(assets)} control assets, {sum(x['size_bytes'] for x in assets)} bytes")
    return fragment

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--pdftoppm", default="pdftoppm")
    a = p.parse_args()
    package(a.source_root, a.repo_root, a.pdftoppm)

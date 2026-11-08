#!/usr/bin/env python3
"""Recalculate ONLY Fig. S6 from native centered-ABC accepted-step histories.

Dependencies: Python >=3.10, numpy, matplotlib. No solver execution is needed.
Usage: python reproduce_fig_S6.py [--input-root ../] [--output-dir .]

The user approved the six available confinement values on 2026-10-20.
No omega=75 values are interpolated, substituted, or included in the fits.
Input validation failures stop generation; all calculations use float64.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fig_s6_mpl"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

T = 20.0
OMEGAS = (1, 25, 50, 100, 150, 200)
SERIES = (
    ("2", 2.0, "Kz=2_kappa=pi", "o", "-"),
    ("pi", math.pi, "kz=pi_kappa=pi_main", "^", "-."),
    ("5", 5.0, "kz=5_kappa=pi", "s", "--"),
)
INPUT_FILES = (
    "revision_run.json", "constants.npz", "branch_refinement_case.json",
    "revision_coordinates.npz", "source_snapshot.py", "revision_step_data.csv",
)
STYLE_REFERENCE = (
    "EXTERNAL/EDIT_Spin-Momentum_filtering"
    "2_FirstAcuuracy_Scripts_Run/R1_Kz_campaign/plot_kz_campaign2.py"
)
SI_REFERENCE = (
    "EXTERNAL/Downloads"
    "Spin_Momentum_Impedance_and_Filtering_Closed_Quantum_Open_System"
    " - 2026-10-20T100533.942.pdf"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value):
    return math.pi if isinstance(value, str) and value == "pi" else value


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_run(folder, label, kz, omega, root):
    """Validate all six inputs; compute the mean directly from stored masses."""
    for name in INPUT_FILES:
        require((folder / name).is_file(), f"{folder}: missing {name}")
    run = json.loads((folder / "revision_run.json").read_text())
    config = run["configuration"]
    branch = json.loads((folder / "branch_refinement_case.json").read_text())
    with np.load(folder / "constants.npz", allow_pickle=False) as z:
        constants = {key: z[key].item() for key in z.files}
    with np.load(folder / "revision_coordinates.npz", allow_pickle=False) as z:
        coords = {key: z[key] for key in ("x", "y", "z")}
    source = (folder / "source_snapshot.py").read_text()
    source_case = next(
        ast.literal_eval(node.value) for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BRANCH_REFINEMENT_CASE"
            for target in node.targets
        )
    )
    require(source_case == branch, f"{folder}: source/branch manifest mismatch")
    hashes = {name: sha256(folder / name) for name in INPUT_FILES}
    require(hashes["source_snapshot.py"] == run["updated_source_sha256"],
            f"{folder}: source hash differs from recorded executed source")

    expected = {
        "omega": omega, "k0": kz, "k_bc": math.pi, "sigma_z": 0.5, "Lz": 20,
        "Nx": 100, "Ny": 100, "Nz": 1500, "grid_generator_nz": 1500,
        "dt": config["dt"], "T_final": config["T_final"],
        "theta": 0, "phi": 0, "endpoint_closure": "centered",
        "endpoint_mass_weight": 0.5, "metric": "half_weight_last_z_node",
        "state_representation": "physical_nodal_spinor", "state_dtype": "complex64",
    }
    for name, data in (("configuration", config), ("constants", constants),
                       ("branch", branch)):
        for key, value in expected.items():
            require(scalar(data[key]) == value,
                    f"{folder}: {name}.{key}={data[key]!r}, expected {value!r}")
    require(config["z0_init"] == branch["z0_init"] == constants["z0"] == 10,
            f"{folder}: source center mismatch")
    require(constants["matrix_dtype"] == branch["matrix_dtype"] == "complex64",
            f"{folder}: assembly dtype mismatch")
    require(config["matrix_assembly"] == "complex64_then_solve_complex128",
            f"{folder}: unexpected matrix/solve precision")
    require(config["real_dtype"] == branch["coordinate_dtype"] == "float32",
            f"{folder}: coordinate dtype mismatch")
    for key, value in (("initial_preparation_dtype", "complex64"),
                       ("potential_sampling_dtype", "float32")):
        require(config[key] == branch[key] == value,
                f"{folder}: {key} mismatch")

    # Read source without importing or executing the GPU solver.
    for snippet in (
        "scale=1.0/ENDPOINT_MASS_WEIGHT",
        "coef = 0.5 / (hz * ENDPOINT_MASS_WEIGHT)",
        "total=total-(1.0-ENDPOINT_MASS_WEIGHT)*cp.vdot(roof,roof).real",
        "q=(oldroof+newroof)/2",
        "(right-left)*G['k_bc']*G['hx']*G['hy']",
        "z_roof_idx = Nz - 1",
    ):
        require(snippet in source, f"{folder}: centered/source evidence absent: {snippet}")
    require("sole roof ABC" in config["detector"], f"{folder}: unexpected detector")
    for axis in ("x", "y", "z"):
        values = coords[axis]
        require(len(values) == config["N" + axis], f"{folder}: {axis} grid count")
        require(values.dtype == np.dtype("float32"), f"{folder}: {axis} grid dtype")
        require(values[0] == 0 and np.all(np.diff(values) > 0),
                f"{folder}: invalid {axis} coordinates")
        require(float(values[1] - values[0]) == config["h" + axis]
                == constants["h" + axis], f"{folder}: {axis} spacing mismatch")
    roof = float(coords["z"][-1])
    require(roof == config["z_roof_actual"] == constants["z_roof_actual"],
            f"{folder}: actual detector mismatch")
    require(abs(roof - (config["Nz"] - 1) * 20 / config["grid_generator_nz"]) < 2e-6,
            f"{folder}: detector differs from declared endpoint=False geometry")
    require(run["status"] == "COMPLETED_NUMERICAL_RUN_NOT_CONVERGENCE_CERTIFIED",
            f"{folder}: run is not completed: {run['status']}")

    data = np.genfromtxt(folder / "revision_step_data.csv", delimiter=",", names=True,
                         dtype=np.float64)
    left, right = data["t_left"], data["t_right"]
    widths = right - left
    require(all(np.all(np.isfinite(data[key])) for key in data.dtype.names),
            f"{folder}: nonfinite accepted-step data")
    require(len(data) == run["completed_step"], f"{folder}: recorded step count mismatch")
    require(np.array_equal(data["step"], np.arange(1, len(data) + 1)),
            f"{folder}: duplicate/missing accepted steps")
    require(left[0] == 0 and np.array_equal(left[1:], right[:-1]),
            f"{folder}: noncontiguous recorded times")
    require(np.all(widths > 0) and np.max(abs(widths - config["dt"])) < 1e-12,
            f"{folder}: unexpected actual step width")
    require(right[-1] == run["actual_time"] == config["T_final"]
            and right[-1] >= T, f"{folder}: incomplete time history")
    require(np.array_equal(data["mass_before"][1:], data["mass_stored"][:-1]),
            f"{folder}: discontinuous stored mass")
    require(np.all(data["roof_probability"] >= 0), f"{folder}: negative roof probability")
    p0 = float(run["P0_init64"])
    require(p0 > 0 and data["mass_before"][0] == p0,
            f"{folder}: launch mass does not match P0_init64")
    require(data["mass_stored"][-1] == run["current_accepted_prefix_mass64"],
            f"{folder}: final mass disagrees with metadata")
    require(abs(math.fsum(data["roof_probability"]) - run["cumulative_roof_probability"])
            < 1e-12, f"{folder}: cumulative roof probability disagrees with metadata")
    require(np.max(data["relative_residual"]) <= 2e-8,
            f"{folder}: solver operational residual guard violated")

    # All selected runs store exactly T=20. Refuse an unseen endpoint convention.
    require(np.count_nonzero(right == T) == 1, f"{folder}: T=20 is not a stored endpoint")
    selected = right <= T
    times = np.concatenate(([0.0], right[selected]))
    survival = np.concatenate(([p0], data["mass_stored"][selected])) / p0
    mu = float(np.trapezoid(survival, times))
    # A midpoint event time is the recorded interval's midpoint, not a new sample.
    event_times = (left[selected] + right[selected]) / 2
    probabilities = data["roof_probability"][selected] / p0
    detection = math.fsum(probabilities)
    moment = math.fsum(probabilities * event_times)
    tau = moment / detection
    s20 = float(survival[-1])
    identity_rhs = moment + T * s20
    residual = mu - identity_rhs
    defect = (
        data["mass_before"][selected] - data["mass_stored"][selected]
        - data["roof_probability"][selected]
    ) / p0
    weighted_defect = math.fsum(event_times * defect)
    require(abs(residual - weighted_defect) < 1e-12,
            f"{folder}: identity residual disagrees with weighted mass defect")
    cumulative_closure = 1 - survival[1:] - np.cumsum(probabilities)

    metric = dict(
        kz_label=label, kz=kz, kappa=math.pi, omega=omega, T=T,
        mu_star20=mu, S20=s20, D20=detection, tau_cond20=tau,
        detector_first_moment20=moment, identity_rhs20=identity_rhs,
        identity_residual=residual, weighted_mass_defect=weighted_defect,
        balance_residual20=1 - s20 - detection,
        max_abs_prefix_balance_residual=float(np.max(abs(cumulative_closure))),
        max_positive_survival_step=float(np.max(np.diff(survival))),
        max_survival=float(np.max(survival)),
        P0_init64=p0, accepted_steps_T20=int(np.count_nonzero(selected)),
        case_id=branch["case_id"],
    )
    manifest = dict(
        omega=omega, kz_label=label, kz=kz, kappa=math.pi, sigma_z=0.5,
        zc=10, L=20, z_detector_actual=roof, case_id=branch["case_id"],
        folder=folder.name, dt=config["dt"],
        actual_dt_min=float(np.min(widths)), actual_dt_max=float(np.max(widths)),
        Nx=config["Nx"], Ny=config["Ny"], Nz=config["Nz"], hz=config["hz"],
        state_dtype="complex64", matrix_dtype="complex64", solve_dtype="complex128",
        coordinate_dtype="float32", initial_preparation_dtype="complex64",
        potential_sampling_dtype="float32", endpoint_closure="centered",
        endpoint_mass_weight=0.5, status=run["status"], validation="PASS",
        actual_time=run["actual_time"], completed_steps=len(data),
        accepted_steps_T20=metric["accepted_steps_T20"],
        P0_init64=p0, relative_path=str(folder.relative_to(root)), path=str(folder),
        **{name.replace(".", "_") + "_sha256": value for name, value in hashes.items()},
    )
    return metric, manifest


def calculate(root):
    rows, manifest = [], []
    for label, kz, group, _, _ in SERIES:
        candidates = {}
        for path in sorted((root / group).glob("*/revision_run.json")):
            conf = json.loads(path.read_text())["configuration"]
            if conf["omega"] in OMEGAS:
                candidates.setdefault(conf["omega"], []).append(path.parent)
        for omega in OMEGAS:
            paths = candidates.get(omega, [])
            require(len(paths) == 1,
                    f"{group}, omega={omega}: expected one run, found {len(paths)}")
            metric, record = read_run(paths[0], label, kz, omega, root)
            rows.append(metric)
            manifest.append(record)
    fits = []
    for label, kz, _, _, _ in SERIES:
        subset = [r for r in rows if r["kz_label"] == label]
        w = np.array([r["omega"] for r in subset], dtype=np.float64)
        y = np.array([r["mu_star20"] for r in subset], dtype=np.float64)
        design = np.column_stack((np.ones(w.size), np.sqrt(w)))
        a, b = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = design @ np.array([a, b])
        residuals = y - fitted
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        fits.append(dict(
            kz_label=label, kz=kz, kappa=math.pi, T=T, n=len(y),
            omega_values=";".join(map(str, OMEGAS)),
            A20=float(a), B20=float(b), RMSE=rmse, r_rms=rmse,
            R2=float(1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)),
            max_abs_residual=float(np.max(abs(residuals))),
            weighting="unweighted", model="A20+B20*sqrt(omega)",
        ))
        for row, fit_value, error in zip(subset, fitted, residuals):
            row["mu_star20_fit"] = float(fit_value)
            row["fit_residual"] = float(error)
    return rows, fits, manifest


def plot(out, rows, fits):
    # Style from the current paper's Fig. S6; no old numerical values are used.
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11,
        "axes.linewidth": 1.0, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, width=0.9)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(True, alpha=0.10)
    handles, labels, all_y = [], [], []
    grid = np.linspace(min(OMEGAS), max(OMEGAS), 1000)
    for (label, _, _, marker, line), fit in zip(SERIES, fits):
        values = [r["mu_star20"] for r in rows if r["kz_label"] == label]
        curve = fit["A20"] + fit["B20"] * np.sqrt(grid)
        hfit, = ax.plot(grid, curve, color="#050505", lw=0.9, linestyle=line)
        hdata, = ax.plot(OMEGAS, values, linestyle="none", marker=marker,
                        markersize=7, markerfacecolor="none", markeredgecolor="#1F4E79",
                        markeredgewidth=1.8, alpha=1, zorder=4)
        math_kz = r"\pi" if label == "pi" else label
        handles.append((hfit, hdata))
        labels.append(
            rf"$({math_kz},\pi)$: "
            rf"${fit['A20']:.3f}+{fit['B20']:.3f}\sqrt{{\omega}}$"
        )
        all_y.extend(values)
        all_y.extend(curve)
    ax.set_xlabel(r"$\omega$", fontweight="bold")
    ax.set_ylabel(r"$\mu^*(20)$", fontweight="bold")
    ax.set_xlim(min(OMEGAS) - 6, max(OMEGAS) + 8)
    ax.set_ylim(min(all_y) - 0.8, max(all_y) + 2.8)
    legend = ax.legend(
        handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)},
        frameon=True, loc="upper left",
        title=r"$(k_z,\kappa);\ \mu^*(20)\approx A_{20}+B_{20}\sqrt{\omega}$",
        title_fontsize=11,
    )
    legend.get_title().set_math_fontfamily("cm")
    fig.tight_layout()
    fig.savefig(out / "fig_S6_centered_T20.pdf", metadata={
        "Title": "Fig. S6: centered-ABC restricted means at T=20",
        "Subject": "Six measured omega values per kz; fixed kappa=pi",
        "Creator": "reproduce_fig_S6.py / matplotlib",
    })
    fig.savefig(out / "fig_S6_centered_T20.png", dpi=600)
    plt.close(fig)


def write_text(out, rows, fits, manifest):
    by_series = {label: [r for r in rows if r["kz_label"] == label]
                 for label, *_ in SERIES}
    require(all(np.all(np.diff([r["mu_star20"] for r in group]) > 0)
                for group in by_series.values()), "Update prose: monotonicity changed")
    matrix = np.array([[r["mu_star20"] for r in by_series[label]]
                       for label, *_ in SERIES])
    require(np.all(matrix[0] > matrix[1]) and np.all(matrix[1] > matrix[2]),
            "Update prose: sampled ordering changed")
    endpoints = np.array([[f["A20"] + f["B20"] * math.sqrt(w) for w in (1, 200)]
                          for f in fits])
    require(np.all(endpoints[0] > endpoints[1]) and np.all(endpoints[1] > endpoints[2]),
            "Update prose: fit crossing detected")
    caption = (
        "Fig. S6. Restricted means mu*(20) for centered spinor ABC with kz=2, pi, 5 "
        "at fixed kappa=pi, sigma_z=0.5, source center zc=10, and nominal guide "
        "length L=20. Hollow circles, triangles, and squares show values calculated "
        "at omega=1,25,50,100,150,200; solid, dash-dot, and dashed curves show "
        "separate unweighted fits A20+B20 sqrt(omega). The legend gives the updated "
        "fit equations with parameter pairs ordered as (kz,kappa). The separate "
        "(kz,kappa)=(5,5) matching control is not plotted.\n"
    )
    fit_phrases = [
        f"kz={f['kz_label']}: (A20,B20)=({f['A20']:.6f},{f['B20']:.6f}), "
        f"r_rms={f['RMSE']:.6f}, max|residual|={f['max_abs_residual']:.6f}, "
        f"R^2={f['R2']:.6f}" for f in fits
    ]
    paragraph = (
        "At fixed kappa=pi, sigma_z=0.5, zc=10, and nominal L=20, the centered-ABC "
        "restricted mean mu*(20) increases with omega for each of kz=2, pi, and 5 "
        "over the six calculated values omega=1,25,50,100,150,200. At every sampled "
        "confinement the ordering is mu*_{kz=2}>mu*_{kz=pi}>mu*_{kz=5}; the fitted "
        "curves also retain this ordering throughout 1<=omega<=200, with no "
        "crossing in that interval. Separate unweighted fits to A20+B20 sqrt(omega) "
        "give " + "; ".join(fit_phrases) + ". Here r_rms is the root mean square "
        "of the six fit residuals. The finite-window accounting residual is at "
        f"most {max(abs(r['identity_residual']) for r in rows):.3e}; this tests "
        "numerical consistency, while the fit residuals describe the empirical "
        "square-root approximation. Neither establishes discretization accuracy "
        "or convergence. The separate (kz,kappa)=(5,5) matching control is not "
        "plotted in Fig. S6.\n"
    )
    (out / "replacement_caption.txt").write_text(caption)
    (out / "replacement_S9_2_paragraph.txt").write_text(paragraph)

    report = [
        "FIG. S6 / SEC. S9.2 VALIDATION REPORT", "",
        "Scope: Fig. S6 only. No manuscript, Fig. S5, Table S10, or other results edited.",
        "Grid decision (2026-10-20): the user explicitly approved the six available",
        "values omega={1,25,50,100,150,200}. Omega=75 is absent for all three kz.",
        "It is omitted, with no interpolation, replacement run, or fitted pseudo-data.",
        "All 18 runs in the revised grid are present, completed, and internally consistent.",
        "",
        "RUN AND GEOMETRY CHECKS",
        "The manifest identifies each folder, case, actual dt range, Nz, status, and path.",
        "revision_run.json, constants.npz, branch_refinement_case.json,",
        "revision_coordinates.npz, and source_snapshot.py agree on the parameters.",
        "The snapshot SHA256 matches the executed-source hash; its embedded case",
        "manifest equals branch_refinement_case.json. Source code confirms the centered",
        "normal endpoint factor, matched spin coupling, and half-weight endpoint norm.",
        "kappa=pi, sigma_z=0.5, zc=10, nominal L=20; Nx=Ny=100, Nz=1500.",
        f"All detectors are the sole roof ABC at z[Nz-1]={manifest[0]['z_detector_actual']:.15g}.",
        "This is the actual endpoint=False grid node, not z=20 exactly.",
        "State/matrix assembly: complex64; solve: complex128; coordinates/potential:",
        "float32; initial preparation: complex64; norm reductions/analysis: float64.",
        "Actual dt at omega=1,25,50,100,150,200:",
        "  kz=2,5: 0.001,0.0008,0.0008,0.0005,0.00025,0.0002",
        "  kz=pi:  0.001,0.0008,0.0008,0.0005,0.00025,0.00025",
        "The dt difference at omega=200 is disclosed; it is not a metadata inconsistency.",
        "The kz=2,5 runs finish at T=20. The kz=pi runs finish at T=40;",
        "only their accepted prefix through exactly T=20 is used.",
        "",
        "CALCULATION AND INDEPENDENT IDENTITY CHECK",
        "P0 = revision_run.json['P0_init64']. Set times=(0,t_right[1],...) and",
        "S=(P0,mass_stored[1],...)/P0, dividing once. Direct trapezoidal integration",
        "is the exact integral of the piecewise-linear stored survival history.",
        "No coarse histories, smoothing, clipping, or further normalization are used.",
        "Every selected history stores T=20 exactly; no endpoint interpolation is needed.",
        "For accepted intervals through T20, q_i=roof_probability_i/P0 and",
        "t_mid_i=(t_left_i+t_right_i)/2. D20=sum(q_i),",
        "tau_cond20=sum(t_mid_i*q_i)/D20. The midpoint is the accepted-step event",
        "quadrature time implied by the recorded midpoint Cayley roof probability.",
        "Signed identity residual = mu_star20 - (D20*tau_cond20 + 20*S20).",
        "Independently, this residual agrees with sum[t_mid_i *",
        "(mass_before_i-mass_stored_i-roof_probability_i)/P0] to <1e-12.",
        "The small residual is retained; the survival curve is not forced to 1-D.",
        f"Max |identity residual|: {max(abs(r['identity_residual']) for r in rows):.12g}.",
        f"Max |1-S20-D20|: {max(abs(r['balance_residual20']) for r in rows):.12g}.",
        f"Max positive stored survival step: {max(r['max_positive_survival_step'] for r in rows):.12g}.",
        "Tiny stored survival increases are retained as recorded; no monotonic repair.",
        "",
        "PER-RUN VALUES (signed residual; full precision also in CSV)",
        "kz omega case     mu_star20          D20            tau_cond20      S20            identity_residual",
    ]
    for r in rows:
        report.append(
            f"{r['kz_label']:>2} {r['omega']:>5} {r['case_id']:>4} "
            f"{r['mu_star20']:17.12f} {r['D20']:.12f} {r['tau_cond20']:14.12f} "
            f"{r['S20']:.12f} {r['identity_residual']:+.12e}"
        )
    report.extend([
        "", "FITS",
        "Ordinary unweighted least squares with design columns [1,sqrt(omega)], n=6.",
        "r_rms = RMSE = sqrt(sum(residual_i^2)/6), matching the paper's convention.",
        "R^2 = 1-SSE/sum((mu_i-mean(mu))^2); residual_i=measured-fitted.",
        *fit_phrases,
        "All sampled means strictly increase. Sample ordering: kz=2 > pi > 5.",
        "For each pair, the fit difference is affine in sqrt(omega) and positive",
        "at both range endpoints, so the fitted curves do not cross on [1,200].",
        "No inference about uncalculated physical curves between samples is required.",
        "", "STYLE AND LIMITATIONS",
        "Matched the latest SI Fig. S6: hollow blue (#1F4E79) o,^,s markers;",
        "black (#050505) solid/dash-dot/dashed fits; 8.6x5.2 inch base canvas;",
        "font 12, labels 13, legend 11; inward ticks and faint grid; vector PDF",
        "and 600-dpi PNG. Legend/caption pair order is consistently (kz,kappa).",
        "Axis limits encompass all new points and fits. The (5,5) control is excluded.",
        "These are numerical consistency and fit checks, not discretization error",
        "estimates. No convergence claim or seven-point fit is made.",
        "Missing original-grid runs: omega=75 for kz=2,pi,5 (user-approved omission).",
        "Inconsistent or incomplete selected runs: none.",
        "", "Reference SI (Fig. S6 on PDF page 34):", SI_REFERENCE,
        "Style source (read only):", STYLE_REFERENCE, "",
    ])
    (out / "validation_report.txt").write_text("\n".join(report))
    (out / "README.txt").write_text(
        "Fig. S6, centered ABC, T=20\n\n"
        "Reproduce from the unchanged run folders one directory above this package:\n"
        "  python reproduce_fig_S6.py\n"
        "Or:\n"
        "  python reproduce_fig_S6.py --input-root /path/to/Output --output-dir /path/to/result\n"
        "Dependencies: Python >=3.10, numpy >=2.0, matplotlib >=3.8.\n"
        "The supplied environment used /opt/anaconda3/bin/python.\n"
        "The script fails on missing, duplicate, incomplete, or inconsistent selected runs.\n"
        "Input folders are read only. Re-running replaces the generated files in output-dir.\n"
        "The explicit user-approved omega set has six values, excluding absent omega=75.\n\n"
        "fig_S6_centered_T20.csv: 18 measurements, fit residuals, identity checks.\n"
        "fig_S6_centered_fit_summary.csv: 3 fits; r_rms is absolute RMSE.\n"
        "run_manifest.csv: selected input paths, metadata, and SHA256 for all six inputs.\n"
        "validation_report.txt: method, all identity residuals, limitations.\n"
        "replacement_caption.txt and replacement_S9_2_paragraph.txt: standalone proposed text.\n"
        "No manuscript edits were made.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.input_root.resolve(), args.output_dir.resolve()
    rows, fits, manifest = calculate(root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fig_S6_centered_T20.csv", rows)
    write_csv(out / "fig_S6_centered_fit_summary.csv", fits)
    write_csv(out / "run_manifest.csv", manifest)
    write_text(out, rows, fits, manifest)
    plot(out, rows, fits)
    provenance = {
        "analysis_date": "2026-10-20", "input_root": str(root),
        "scope": "Fig. S6 only", "horizon": T, "omega_values": list(OMEGAS),
        "grid_change_authorization": "User: Use the six available values",
        "missing_original_grid": [{"kz": v, "omega": 75} for v in ("2", "pi", "5")],
        "selected_runs": len(rows), "numerical_validation": "PASS",
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "script_sha256": sha256(Path(__file__)),
        "style_source": STYLE_REFERENCE, "latest_SI_reference": SI_REFERENCE,
        "SI_page_1_based": 34,
    }
    for key, path in (("style_source_sha256", Path(STYLE_REFERENCE)),
                      ("SI_reference_sha256", Path(SI_REFERENCE))):
        if path.is_file():
            provenance[key] = sha256(path)
    (out / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Validated {len(rows)} runs; wrote Fig. S6 and supporting files to {out}")
    for fit in fits:
        print(fit)


if __name__ == "__main__":
    main()

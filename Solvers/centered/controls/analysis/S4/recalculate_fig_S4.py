#!/usr/bin/env python3
"""Recalculate only Fig. S4 from native centered-ABC accepted-step records.

Run: python recalculate_fig_S4.py
Requires Python >=3.10, NumPy >=2.0, Matplotlib >=3.8.
Optional: --input-root /path/to/Output --output-dir /path/to/results

The user explicitly excluded omega=75 on 2026-10-20. All three unweighted
fits therefore use the SAME SIX omega values. No simulation is launched.
No old numerical values or fitted coefficients are used.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

OMEGAS = (1, 25, 50, 100, 150, 200)
KAPPAS = (("2", 2.0), ("pi", math.pi), ("7", 7.0))
T = 20.0
REQUIRED = (
    "revision_run.json", "constants.npz", "branch_refinement_case.json",
    "revision_coordinates.npz", "source_snapshot.py", "revision_step_data.csv",
)
STYLE_REFERENCE = (
    "EXTERNAL/EDIT_Spin-Momentum_filtering"
    "2_FirstAcuuracy_Scripts_Run/R1_kappa_campaign/plot_kappa_campaign.py"
)
SI_REFERENCE = (
    "EXTERNAL/Downloads"
    "Spin_Momentum_Impedance_and_Filtering_Closed_Quantum_Open_System"
    " - 2026-10-20T100533.942.pdf"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b, description):
    if isinstance(a, str) or isinstance(b, str):
        ok = a == b
    else:
        ok = math.isclose(float(a), float(b), rel_tol=2e-13, abs_tol=2e-13)
    require(ok, f"{description}: {a!r} != {b!r}")


def scalar(value):
    if value == "pi":
        return math.pi
    return value


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)  # Python floats retain round-trip precision.


def source_case(source):
    """Inspect source statically; never import/execute a GPU simulation."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BRANCH_REFINEMENT_CASE"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Source has no literal BRANCH_REFINEMENT_CASE")


def validate_metadata(folder, label, kappa, omega):
    for filename in REQUIRED:
        require((folder / filename).is_file(), f"Missing {folder / filename}")
    run = json.loads((folder / "revision_run.json").read_text())
    c = run["configuration"]
    branch = json.loads((folder / "branch_refinement_case.json").read_text())
    source = (folder / "source_snapshot.py").read_text()
    embedded = source_case(source)
    require(embedded == branch, f"Snapshot/branch manifest differs: {folder}")
    with np.load(folder / "constants.npz", allow_pickle=False) as z:
        constants = {k: v.item() for k, v in z.items()}
    expected = dict(
        omega=omega, k_bc=kappa, k0=math.pi, sigma_z=0.5,
        Lz=20.0, endpoint_closure="centered", endpoint_mass_weight=0.5,
        metric="half_weight_last_z_node", state_representation="physical_nodal_spinor",
        theta=0.0, phi=0.0, Nz=1500, grid_generator_nz=1500,
        Nx=100, Ny=100, state_dtype="complex64",
    )
    for key, val in expected.items():
        for name, data in (("configuration", c), ("branch", branch), ("constants", constants)):
            require(key in data, f"Missing {name}.{key} in {folder}")
            equal(scalar(data[key]), val, f"{folder.name}: {name}.{key}")
    equal(c["z0_init"], 10.0, "configuration zc")
    equal(branch["z0_init"], 10.0, "branch zc")
    equal(constants["z0"], 10.0, "constants zc")
    for key in ("dt", "T_final"):
        equal(c[key], branch[key], f"branch {key}")
        equal(c[key], constants[key], f"constants {key}")
    for key in ("matrix_dtype", "initial_preparation_dtype", "coordinate_dtype",
                "potential_sampling_dtype"):
        equal(branch[key], "float32" if "coordinate" in key or "potential" in key
              else "complex64", key)
    equal(constants["matrix_dtype"], branch["matrix_dtype"], "matrix dtype")
    equal(c["matrix_assembly"], "complex64_then_solve_complex128", "matrix/solve precision")
    equal(c["initial_preparation_dtype"], "complex64", "initial preparation")
    equal(c["potential_sampling_dtype"], "float32", "potential sampling")
    equal(c["real_dtype"], "float32", "real dtype")
    equal(sha256(folder / "source_snapshot.py"), run["updated_source_sha256"], "source SHA256")
    # Verify centered normal closure AND the matched spin-tangential endpoint scale.
    for token in (
        "ENDPOINT_MASS_WEIGHT = BRANCH_REFINEMENT_CASE['endpoint_mass_weight']",
        "ENDPOINT_CLOSURE = BRANCH_REFINEMENT_CASE['endpoint_closure']",
        "scale=1.0/ENDPOINT_MASS_WEIGHT",
        "coef = 0.5 / (hz * ENDPOINT_MASS_WEIGHT)",
        "total=total-(1.0-ENDPOINT_MASS_WEIGHT)*cp.vdot(roof,roof).real",
        "q=(oldroof+newroof)/2",
        "mass_stored=_abc_norm(stored,self.dV)",
    ):
        require(token in source, f"Source convention absent: {token}: {folder}")
    # The actual detector is the last z node, not the nominal L=20 endpoint.
    with np.load(folder / "revision_coordinates.npz", allow_pickle=False) as z:
        for axis, count, extent, spacing in (
            ("x", c["Nx"], c["Lx"], c["hx"]),
            ("y", c["Ny"], c["Ly"], c["hy"]),
            ("z", c["Nz"], c["Lz"], c["hz"]),
        ):
            v = z[axis]
            require(v.shape == (count,) and v.dtype == np.dtype("float32"),
                    f"Coordinate shape/precision: {folder}/{axis}")
            require(v[0] == 0 and np.all(np.diff(v) > 0), f"Coordinate order {axis}")
            equal(float(v[1] - v[0]), spacing, f"actual spacing {axis}")
            require(abs(float(v[-1]) - (count - 1) * extent / count) < 3e-6,
                    f"Endpoint differs from declared endpoint=False mesh: {axis}")
        roof = float(z["z"][-1])
    equal(roof, c["z_roof_actual"], "configuration detector position")
    equal(roof, constants["z_roof_actual"], "constants detector position")
    require(run["status"].startswith("COMPLETED_NUMERICAL_RUN"), f"Incomplete run: {folder}")
    require(run["actual_time"] >= T and c["T_final"] >= T, f"Run too short: {folder}")
    equal(run["actual_time"], c["T_final"], "completion time")
    require(run["P0_init64"] > 0, f"Invalid launch norm: {folder}")
    return run, dict(
        omega=omega, kappa=label, kappa_numeric=kappa, case_id=branch["case_id"],
        folder=folder.name, dt=c["dt"], Nz=c["Nz"], Nx=c["Nx"], Ny=c["Ny"],
        status=run["status"], actual_time=run["actual_time"], requested_horizon=T,
        zc=10.0, L=20.0, z_detector_actual=roof, hz=c["hz"],
        endpoint_closure=c["endpoint_closure"], endpoint_mass_weight=0.5,
        state_dtype=c["state_dtype"], matrix_dtype=branch["matrix_dtype"],
        solve_dtype="complex128", coordinate_dtype="float32", kz=c["k0"],
        sigma_z=c["sigma_z"], P0_init64=run["P0_init64"],
        metadata_validation="PASS", path=str(folder),
    )


def calculate(folder, run, manifest):
    names = (
        "step", "t_left", "t_right", "roof_probability", "mass_before",
        "mass_stored", "cast_mass_loss", "clamp_mass_loss",
        "solve_and_matrix_defect", "relative_residual",
    )
    # Binary64 parsed directly from decimal CSV; no coarse probability histories.
    data = np.loadtxt(folder / "revision_step_data.csv", delimiter=",",
                      skiprows=1, usecols=(0, 1, 2, 3, 4, 6, 7, 8, 9, 10))
    require(data.ndim == 2 and np.all(np.isfinite(data)), f"Invalid step data: {folder}")
    d = dict(zip(names, data.T))
    p0 = run["P0_init64"]
    left, right, mass = d["t_left"], d["t_right"], d["mass_stored"]
    require(np.array_equal(d["step"], np.arange(1, len(data) + 1)), "Missing/duplicate step")
    require(left[0] == 0 and np.array_equal(left[1:], right[:-1]), "Noncontiguous times")
    require(np.all(right > left), "Nonpositive step duration")
    require(d["mass_before"][0] == p0 and
            np.array_equal(d["mass_before"][1:], mass[:-1]), "Discontinuous mass chain")
    require(np.all(d["roof_probability"] >= 0), "Negative roof probability")
    equal(right[-1], run["actual_time"], "CSV completion time")
    equal(len(data), run["completed_step"], "CSV completion step")
    require(right[-1] >= T, "Step data do not reach T20")
    require(np.max(np.abs((right - left) - manifest["dt"])) < 2e-12, "Unexpected dt")
    # These retained runs all have an exact recorded boundary at T=20. Stop on
    # mismatch rather than silently round the horizon or guess partial events.
    boundary = np.flatnonzero(right == T)
    require(len(boundary) == 1, f"T=20 not a unique recorded boundary: {folder}")
    n = int(boundary[0]) + 1
    sl = slice(0, n)
    times = np.r_[0.0, right[sl]]
    survival = np.r_[p0, mass[sl]] / p0
    # Primary calculation: exact integral of the piecewise-linear stored history.
    mu = float(np.trapezoid(survival, times))
    S20 = float(survival[-1])
    # Independent accepted-step event quadrature with accepted interval midpoints.
    tm = (left[sl] + right[sl]) / 2
    events = d["roof_probability"][sl] / p0
    D20 = math.fsum(events)
    M20 = math.fsum(tm * events)
    tau = M20 / D20
    identity = mu - (M20 + T * S20)
    defects = (d["cast_mass_loss"][sl] + d["clamp_mass_loss"][sl] +
               d["solve_and_matrix_defect"][sl]) / p0
    defect_moment = math.fsum(tm * defects)
    closure = 1.0 - D20 - S20
    require(abs(identity - defect_moment) < 2e-12, "Identity residual accounting mismatch")
    require(abs(closure - math.fsum(defects)) < 2e-12, "Closure accounting mismatch")
    require(abs(identity) < 1e-4, "Unexpectedly large identity residual")
    all_roof = math.fsum(d["roof_probability"])
    equal(all_roof, run["cumulative_roof_probability"], "Full-run event total")
    manifest.update(
        steps_total=len(data), steps_through_T20=n,
        dt_recorded_min=float(np.min(right - left)),
        dt_recorded_max=float(np.max(right - left)),
        native_history_validation="PASS",
    )
    return dict(
        omega=manifest["omega"], kappa=manifest["kappa"],
        kappa_numeric=manifest["kappa_numeric"], case_id=manifest["case_id"],
        T=T, mu_star_20=mu, D20=D20, tau_cond20=tau, S20=S20,
        event_first_moment_20=M20, identity_rhs=M20 + T * S20,
        identity_residual=identity, roof_closure_residual=closure,
        defect_time_moment=defect_moment,
        residual_minus_defect_moment=identity - defect_moment,
        P0_init64=p0, steps_through_T20=n,
        increasing_survival_steps=int(np.count_nonzero(np.diff(survival) > 0)),
        max_survival_increase=float(np.max(np.diff(survival))),
        max_relative_solver_residual=float(np.max(d["relative_residual"][sl])),
        dt=manifest["dt"], Nz=manifest["Nz"], path=str(folder),
    )


def fit_series(rows):
    fits = []
    for label, kappa in KAPPAS:
        series = [r for r in rows if r["kappa"] == label]
        omega = np.array([r["omega"] for r in series], dtype=float)
        y = np.array([r["mu_star_20"] for r in series])
        X = np.column_stack((np.ones(len(omega)), np.sqrt(omega)))
        A, B = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ np.array([A, B])
        residual = y - fitted
        rss = float(residual @ residual)
        r2 = 1 - rss / float(np.sum((y - y.mean()) ** 2))
        for row, f, r in zip(series, fitted, residual):
            row["fitted_mu_star_20"] = float(f)
            row["fit_residual_data_minus_fit"] = float(r)
        fits.append(dict(
            kappa=label, kappa_numeric=kappa, n=len(series), omegas=";".join(map(str, OMEGAS)),
            A20=float(A), B20=float(B), RMSE=math.sqrt(rss / len(series)),
            R_squared=r2, max_abs_residual=float(np.max(np.abs(residual))),
            residual_sum_squares=rss, fit="unweighted A20+B20*sqrt(omega)",
        ))
    return fits


def draw(rows, fits, out):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "s4_centered_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13,
        "legend.fontsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, width=0.9)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(True, alpha=0.10)
    handles, labels, all_y = [], [], []
    for fit, marker, linestyle in zip(fits, ("o", "s", "^"), ("-", "--", "-.")):
        series = [r for r in rows if r["kappa"] == fit["kappa"]]
        x = np.array([r["omega"] for r in series])
        y = np.array([r["mu_star_20"] for r in series])
        grid = np.linspace(x.min(), x.max(), 800)
        fy = fit["A20"] + fit["B20"] * np.sqrt(grid)
        h_fit, = ax.plot(grid, fy, color="#050505", lw=0.9, linestyle=linestyle)
        h_data, = ax.plot(x, y, linestyle="none", marker=marker, markersize=7,
                         markerfacecolor="none", markeredgecolor="#1F4E79",
                         markeredgewidth=1.8, zorder=4)
        handles.append((h_fit, h_data))
        k = r"\pi" if fit["kappa"] == "pi" else fit["kappa"]
        labels.append(fr"$({k},\pi)$: ${fit['A20']:.3f}+{fit['B20']:.3f}\sqrt{{\omega}}$")
        all_y.extend(y)
        all_y.extend(fy)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\mu^{*}(20)$")
    # Adapt margins to the new range while containing all data AND curves.
    ax.set_xlim(-0.02 * max(OMEGAS), 1.02 * max(OMEGAS))
    ymin, ymax = min(all_y), max(all_y)
    ax.set_ylim(ymin - 0.06 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
    leg = ax.legend(
        handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)},
        title=r"$(\kappa,k_z);\ \mu^{*}(20)\approx A_{20}+B_{20}\sqrt{\omega}$",
        loc="upper left", frameon=True, handlelength=3.5,
    )
    leg.get_title().set_fontsize(11)
    leg.get_title().set_math_fontfamily("cm")
    fig.tight_layout()
    fig.savefig(out / "fig_S4_centered_T20.pdf",
                metadata={"Title": "Fig. S4: centered-ABC detector parameter sweep",
                          "Subject": "T=20; six confinement values; omega=75 excluded",
                          "Author": "Recalculated from accepted-step centered-ABC records"})
    fig.savefig(out / "fig_S4_centered_T20.png", dpi=600)
    plt.close(fig)
    return matplotlib.__version__


def text_outputs(rows, fits, manifests, out):
    roof = manifests[0]["z_detector_actual"]
    max_identity = max(abs(r["identity_residual"]) for r in rows)
    max_closure = max(abs(r["roof_closure_residual"]) for r in rows)
    fit_text = "; ".join(
        f"kappa={f['kappa']}: A20={f['A20']:.6f}, B20={f['B20']:.6f}, "
        f"r_rms={f['RMSE']:.6f}, R^2={f['R_squared']:.6f}, "
        f"max|residual|={f['max_abs_residual']:.6f}" for f in fits
    )
    caption = (
        "Figure S4. Restricted means mu*(20) for the centered spinor ABC at "
        "kappa=2, pi, and 7, with kz=pi, sigma_z=0.5, zc=10, and L=20. "
        "Markers show the integrals of normalized every-step stored survival "
        "at omega=1,25,50,100,150,200; lines are separate unweighted empirical "
        "fits A20+B20 sqrt(omega), with updated equations in the legend. "
        "Legend parameter pairs are ordered as (kappa,kz)."
    )
    paragraph = (
        "At T=20, the calculated restricted mean increases with omega for all "
        "three detector parameters. The sampled ordering is "
        "mu_pi < mu_2 < mu_7 at omega=1,25,50; mu_pi < mu_7 < mu_2 at "
        "omega=100; and mu_2 < mu_7 < mu_pi at omega=150,200. Thus kappa=pi "
        "still gives the smallest mean at omega=1, but not at the largest "
        "confinements. The kappa=2 and 7 series exchange order between "
        "omega=50 and 100 and again between 100 and 150; each exchanges order "
        "with kappa=pi between 100 and 150. These are changes between sampled "
        "values, not resolved crossing locations. Over the six retained "
        f"confinements, the unweighted fits give {fit_text}. "
        "Here r_rms=sqrt(sum(residual^2)/6). The empirical square-root curves "
        "do not reproduce every sampled reordering. The finite-window "
        f"identity has maximum absolute residual {max_identity:.3e}; this "
        "checks consistency of the stored norm and accepted-step detector "
        "probabilities, and does not establish discretization accuracy or convergence."
    )
    (out / "replacement_caption_and_S9_1_paragraph.txt").write_text(
        caption + "\n\nReplacement paragraph for Sec. S9.1\n\n" + paragraph +
        "\n\nEditorial note: omega=75 excluded from every series by the user's "
        "2026-10-20 instruction. No manuscript file was edited.\n"
    )
    lines = [
        "Fig. S4 centered-ABC validation report",
        "",
        "Scope: only Fig. S4 / Sec. S9.1, horizon T=20.",
        "User amendment (2026-10-20): exclude omega=75 from ALL series.",
        f"Retained omega: {OMEGAS}; kappa: 2, pi, 7; 18/18 runs available.",
        "Missing/inconsistent retained runs: none.",
        "The original seven-point request lacked kappa=2,7 omega=75 runs.",
        "A pi omega=75 run exists in 1.Main but is excluded by the amendment.",
        "No original first-order runs, substitutions, interpolated confinement data,",
        "coarse prob_times/total_probs, smoothing, clipping, or renormalization used.",
        "",
        "Metadata and source validation",
        "For each selected run, revision_run.json, constants.npz,",
        "branch_refinement_case.json, revision_coordinates.npz, and",
        "source_snapshot.py agree on centered closure, half-weight final z node,",
        "omega, kappa, kz=pi, sigma_z=0.5, zc=10, L=20, grid and precision.",
        "The source's embedded case dict equals branch_refinement_case.json;",
        "its SHA256 equals revision_run.json updated_source_sha256.",
        "Nx=Ny=100; Nz=1500. State/matrix assembly complex64; linear solve",
        "complex128; coordinates and sampled potential float32.",
        f"Detector: sole upper roof at actual last node z={roof:.17g}.",
        "Nominal L=20; endpoint=False mesh gives nominal roof 20*(1499/1500),",
        "represented in float32. It is not a detector exactly at z=20.",
        "The centered normal endpoint and matched spin-tangential assembly",
        "use the reciprocal half-node weight. Original one-sided closure excluded.",
        "All histories have contiguous step IDs/times and exact stored-to-next",
        "mass_before continuity; all roof probabilities nonnegative.",
        "Every T=20 is an exact recorded boundary. kappa=pi runs extend to T=40;",
        "only their [0,20] histories enter the means/events/fits.",
        "dt by omega=1,25,50,100,150,200:",
        "  kappa=2,7: 0.001,0.0008,0.0008,0.0005,0.00025,0.0002",
        "  kappa=pi:  0.001,0.0008,0.0008,0.0005,0.00025,0.00025",
        "The omega=200 pi time step differs from kappa=2,7 and is retained as run.",
        "Full precision dt extrema, completion, case IDs and paths: run_manifest.csv.",
        "",
        "Calculation",
        "P0 = revision_run.json['P0_init64']. Raw mass sequence =",
        "[P0, mass_stored[1], ..., mass_stored[n]]. Divide once by P0.",
        "S values are kept as recorded, including tiny early positive increments.",
        "mu*(20) = trapezoidal integral on actual [0,t_right...] timestamps,",
        "i.e. exact integral of the normalized piecewise-linear stored history.",
        "Independently: p_j=roof_probability_j/P0; t_j=(t_left_j+t_right_j)/2;",
        "D20=sum p_j; M20=sum t_j*p_j; tau_cond20=M20/D20.",
        "Identity residual R=mu*(20) - [D20*tau_cond20 + 20*S20].",
        "Raw step defect e_j=(mass_before-mass_stored-roof_probability)/P0",
        "also equals (cast_mass_loss+clamp_mass_loss+solve_and_matrix_defect)/P0.",
        "R equals sum t_j*e_j; 1-D20-S20 equals sum e_j. These are diagnostic",
        "cross-checks only; they do not replace the survival integral.",
        f"Maximum |R| = {max_identity:.17g}.",
        f"Maximum |1-D20-S20| = {max_closure:.17g}.",
        "The source's 1e-3 closure guard is a fail-stop bound, not an accuracy target.",
        "",
        "Per-run numerical consistency (signed residuals)",
        "kappa omega case mu*(20) D20 tau_cond20 S20 R 1-D20-S20",
    ]
    lines.extend(
        f"{r['kappa']:>4} {r['omega']:>3} {r['case_id']} "
        f"{r['mu_star_20']:.12f} {r['D20']:.12f} {r['tau_cond20']:.12f} "
        f"{r['S20']:.12f} {r['identity_residual']:+.9e} "
        f"{r['roof_closure_residual']:+.9e}" for r in rows
    )
    lines += [
        "", "Fit statistics", fit_text,
        "Fit residual=data-fit; RMSE uses divisor n=6 (SI r_rms convention).",
        "R^2=1-sum(residual^2)/sum((data-mean(data))^2).",
        "",
        "Ordering at measured confinement values",
    ]
    for w in OMEGAS:
        ordered = sorted((r for r in rows if r["omega"] == w), key=lambda r: r["mu_star_20"])
        lines.append(f"omega={w}: " + " < ".join(f"kappa={r['kappa']}" for r in ordered))
    lines.append("Intersections of empirical fitted curves (not measured crossings):")
    for f1, f2 in itertools.combinations(fits, 2):
        root = (f2["A20"] - f1["A20"]) / (f1["B20"] - f2["B20"])
        if root >= 0:
            w = root * root
            lines.append(f"  {f1['kappa']} / {f2['kappa']}: omega={w:.9f}, "
                         + ("inside" if min(OMEGAS) <= w <= max(OMEGAS) else "outside")
                         + " retained range")
    lines += [
        "Fits cannot be used to infer all the sampled order changes.",
        "Close orderings (e.g. kappa=2 vs pi at omega=25) are properties of",
        "these stored calculations; no discretization error bars are available.",
        "",
        "Style and output",
        "Original Fig. S4 style: " + STYLE_REFERENCE,
        "Latest located SI: " + SI_REFERENCE + " (Fig. S4 p.33; r_rms p.32).",
        "Preserved 8.6x5.2-inch canvas, hollow navy #1F4E79 circles/squares/",
        "triangles for kappa=2/pi/7, black #050505 solid/dashed/dash-dot fits,",
        "13pt labels, 12pt ticks, 11pt legend, inward ticks, faint grid.",
        "Updated y label mu*(20), fit equations, and limits to contain all data/curves.",
        "PDF is vector with embedded TrueType fonts; PNG uses 600 dpi.",
        "",
        "Limitations",
        "Checks establish numerical accounting consistency of completed runs.",
        "No mesh/time-step/precision refinement study was performed here.",
        "No convergence or continuum accuracy claim is made. The different",
        "omega=200 dt and finite-grid detector position are explicitly disclosed.",
        "No manuscript, Table S9, Fig. S3, or other section was changed.",
        "",
        "Reproduce: python recalculate_fig_S4.py",
        "Optional --input-root points to the campaign Output directory;",
        "--output-dir selects the regenerated deliverables directory.",
        "All computed numbers are saved at Python float round-trip precision in CSV.",
        "input_provenance.json records SHA256 checksums of every calculation input.",
    ]
    (out / "validation_report.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input-root", type=Path, default=here.parent)
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    root, out = args.input_root.resolve(), args.output_dir.resolve()
    # Select by actual metadata, not by trusting folder names. Ambiguity is fatal.
    candidates = []
    for file in sorted(root.glob("*/*/revision_run.json")):
        r = json.loads(file.read_text())
        candidates.append((file.parent, r["configuration"]))
    selected = []
    for label, kappa in KAPPAS:
        for omega in OMEGAS:
            matches = [p for p, c in candidates if
                       c.get("omega") == omega and c.get("k_bc") == kappa]
            require(len(matches) == 1,
                    f"Expected one run for kappa={label}, omega={omega}, found {matches}")
            selected.append((matches[0], label, kappa, omega))
    rows, manifests, provenance = [], [], []
    for folder, label, kappa, omega in selected:
        run, manifest = validate_metadata(folder, label, kappa, omega)
        result = calculate(folder, run, manifest)
        rows.append(result)
        manifests.append(manifest)
        provenance.append({
            "path": str(folder),
            "sha256": {filename: sha256(folder / filename) for filename in REQUIRED},
        })
    fits = fit_series(rows)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fig_S4_centered_T20.csv", rows)
    write_csv(out / "fig_S4_centered_fit_summary.csv", fits)
    write_csv(out / "run_manifest.csv", manifests)
    text_outputs(rows, fits, manifests, out)
    mpl_version = draw(rows, fits, out)
    provenance_output = dict(
        analysis="Fig. S4 only; centered ABC; normalized native every-step mass",
        T=T, omega=list(OMEGAS), excluded_omega=[75],
        exclusion_reason="Explicit user instruction on 2026-10-20",
        python=platform.python_version(), numpy=np.__version__, matplotlib=mpl_version,
        script_sha256=sha256(Path(__file__)),
        style_reference=STYLE_REFERENCE,
        supplementary_information_reference=SI_REFERENCE,
        reference_sha256={
            str(p): sha256(p) for p in map(Path, (STYLE_REFERENCE, SI_REFERENCE))
            if p.is_file()
        },
        inputs=provenance,
    )
    (out / "input_provenance.json").write_text(json.dumps(provenance_output, indent=2) + "\n")
    (out / "requirements.txt").write_text("numpy>=2.0\nmatplotlib>=3.8\n")
    print(json.dumps({"output": str(out), "validated_runs": len(rows), "fits": fits}, indent=2))


if __name__ == "__main__":
    main()

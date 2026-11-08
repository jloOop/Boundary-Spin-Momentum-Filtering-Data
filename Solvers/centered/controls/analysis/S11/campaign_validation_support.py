#!/usr/bin/env python3
"""Analyze completed centered-ABC lower-wall runs 7750--7754.

NumPy is required. Matplotlib is optional. Inputs are read-only; output files
contain new calculations from the supplied histories, never reference values.
Use actual accepted interval endpoints, including the clipped final CN step.
An operational accounting pass is not a spatial or time convergence result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


COMPLETED = "COMPLETED_NUMERICAL_RUN_NOT_CONVERGENCE_CERTIFIED"
FAMILY = {
    7750: (200.0, 0.0, 1500, 0.0003),
    7751: (200.0, -5.0, 1875, 0.0003),
    7752: (200.0, -10.0, 2250, 0.0003),
    7753: (200.0, -15.0, 2625, 0.0003),
    7754: (1.0, -15.0, 2625, 0.001),
}
REQUIRED_FILES = (
    "revision_run.json", "revision_step_data.csv",
    "revision_coordinates.npz", "longitudinal_history.npz",
)
TIME_ATOL = 1e-10
ACCOUNTING_GUARD = 1e-3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def maximum_abs(a):
    return float(np.max(np.abs(a), initial=0.0))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def trapz(y, x):
    fn = getattr(np, "trapezoid", None)
    return float(fn(y, x) if fn is not None else np.trapz(y, x))


def window_grid(time, start, stop):
    require(0 <= start < stop, "Window must satisfy 0 <= start < stop.")
    require(start >= time[0] and stop <= time[-1],
            "Requested window exceeds actual recorded support; no extrapolation is allowed.")
    return np.r_[start, time[(time > start) & (time < stop)], stop]


def linear_integral(time, values, start, stop):
    """Exact integral of the piecewise-linear recorded history on a window."""
    q = window_grid(time, start, stop)
    return trapz(np.interp(q, time, values), q)


def event_moments(left, right, probability, start, stop):
    """Probability and first moment of constant per-accepted-step rates.

    A window may cut either end of an interval. The moment is evaluated at the
    midpoint of its OVERLAP, not at the midpoint of the original whole step.
    """
    require(0 <= start < stop and start >= left[0] and stop <= right[-1],
            "Event window exceeds recorded support.")
    duration = right - left
    require(np.all(duration > 0), "Nonpositive accepted interval.")
    a = np.maximum(left, start)
    b = np.minimum(right, stop)
    overlap = np.maximum(b - a, 0.0)
    weighted = probability * (overlap / duration)
    return float(np.sum(weighted)), float(np.sum(weighted * (a + 0.5 * overlap)))


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def statistics(time, survival, left, right, probability, horizon, cutoff=3.6):
    """Table S11 quantities from survival, plus independent CN-event values."""
    require(0 < cutoff < horizon, "Early/late cutoff must lie inside the window.")
    mu = linear_integral(time, survival, 0.0, horizon)
    s_t = float(np.interp(horizon, time, survival))
    s_cut = float(np.interp(cutoff, time, survival))
    detected = 1.0 - s_t
    moment = mu - horizon * s_t
    late = s_cut - s_t
    late_moment = (cutoff * s_cut - horizon * s_t
                   + linear_integral(time, survival, cutoff, horizon))
    cn_d, cn_m = event_moments(left, right, probability, 0.0, horizon)
    cn_early, cn_early_m = event_moments(left, right, probability, 0.0, cutoff)
    cn_late, cn_late_m = event_moments(left, right, probability, cutoff, horizon)
    endpoint_cumulative = np.r_[0.0, np.cumsum(probability)]
    q = window_grid(time, 0.0, horizon)
    closure = 1.0 - np.interp(q, time, survival) - np.interp(q, time, endpoint_cumulative)
    return {
        "T": float(horizon), "cutoff": float(cutoff), "S_T": s_t,
        "D_T": detected, "mean_cond": safe_ratio(moment, detected),
        "mu_star": mu, "detected_first_moment_survival": moment,
        "D_E": 1.0 - s_cut, "D_L": late,
        "late_mean": safe_ratio(late_moment, late),
        "late_first_moment_survival": late_moment,
        "D_T_CN": cn_d, "first_moment_CN": cn_m,
        "mean_cond_CN": safe_ratio(cn_m, cn_d),
        "mu_star_CN": cn_m + horizon * (1.0 - cn_d),
        "D_E_CN": cn_early, "early_first_moment_CN": cn_early_m,
        "D_L_CN": cn_late, "late_first_moment_CN": cn_late_m,
        "late_mean_CN": safe_ratio(cn_late_m, cn_late),
        "detector_survival_closure_T": detected - cn_d,
        "max_abs_detector_survival_closure_window": maximum_abs(closure),
        "detected_moment_difference_survival_minus_CN": moment - cn_m,
        "late_moment_difference_survival_minus_CN": late_moment - cn_late_m,
        "restricted_mean_identity_residual_using_recorded_S": mu - cn_m - horizon * s_t,
        "restricted_mean_difference_survival_minus_CN": mu - cn_m - horizon * (1.0 - cn_d),
    }


def load_postchecker(path):
    if path is None or not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("centered_lowerwall_postcheck", path)
    require(spec is not None and spec.loader is not None, "Cannot load postcheck.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "check", None)), "postcheck.py must expose check(out, code).")
    return module.check


def load_run(path, postchecker=None):
    path = Path(path).expanduser().resolve()
    for name in REQUIRED_FILES:
        require((path / name).is_file(), f"Missing input: {path / name}")
    meta = json.loads((path / "revision_run.json").read_text())
    cfg = meta["configuration"]
    require(meta.get("status") == COMPLETED,
            f"{path.name}: run must have completed T=40; status is {meta.get('status')!r}.")
    require(abs(float(meta["actual_time"]) - 40.0) <= TIME_ATOL
            and abs(float(cfg["T_final"]) - 40.0) <= TIME_ATOL,
            f"{path.name}: both requested and completed horizon must be T=40.")
    require(cfg.get("endpoint_closure") == "centered"
            and float(cfg.get("endpoint_mass_weight", -1)) == 0.5,
            f"{path.name}: centered closure and half-weight roof metric are required.")
    case_id = cfg.get("case_id", cfg.get("fixed_geometry", {}).get("case_id"))
    casefile = path / "branch_refinement_case.json"
    if casefile.is_file():
        case_meta = json.loads(casefile.read_text())
        candidate = case_meta.get("case_id")
        if candidate is not None:
            require(case_id is None or int(case_id) == int(candidate), "Case identifiers disagree.")
            case_id = candidate
    if case_id is None:
        try:
            case_id = int(path.name.split(".", 1)[0])
        except ValueError as exc:
            raise ValueError("Cannot determine campaign case ID.") from exc
    case_id = int(case_id)
    require(case_id in FAMILY, f"Unexpected case {case_id}; expected 7750--7754.")
    omega, z_bottom, nz, nominal_dt = FAMILY[case_id]
    for key, expected in (("omega", omega), ("dt", nominal_dt), ("k0", math.pi),
                          ("k_bc", math.pi), ("sigma_z", 0.5), ("theta", 0.0), ("phi", 0.0)):
        require(abs(float(cfg[key]) - expected) <= 1e-12, f"Case {case_id}: {key} mismatch.")
    z_source = float(cfg.get("z0_init", cfg.get("z0", float("nan"))))
    require(z_source == 10.0, f"Case {case_id}: physical source must remain at z=10.")
    require(int(cfg["Nx"]) == 100 and int(cfg["Ny"]) == 100 and int(cfg["Nz"]) == nz,
            f"Case {case_id}: unexpected production grid; smoke outputs are not accepted.")
    p0 = float(meta["P0_init64"])
    require(math.isfinite(p0) and p0 > 0, "Invalid recorded launch norm.")
    a = np.genfromtxt(path / "revision_step_data.csv", delimiter=",", names=True,
                      dtype=np.float64, ndmin=1)
    columns = ("step", "t_left", "t_right", "roof_probability", "mass_before",
               "mass_stored", "relative_residual", "p_spin_up", "p_spin_down",
               "p_spin_offdiag_real", "p_spin_offdiag_imag")
    require(a.size > 0 and all(k in (a.dtype.names or ()) for k in columns), "Incomplete event schema.")
    for name in a.dtype.names:
        require(np.all(np.isfinite(a[name])), f"Nonfinite accepted-step column: {name}")
    n = len(a)
    require(n == int(meta["completed_step"])
            and np.array_equal(a["step"], np.arange(1, n + 1)), "Accepted-step numbering mismatch.")
    left, right = a["t_left"], a["t_right"]
    require(left[0] == 0.0 and np.array_equal(left[1:], right[:-1])
            and np.all(right > left), "Accepted-step intervals have gaps or invalid endpoints.")
    require(right[-1] == float(meta["actual_time"]), "CSV/metadata terminal time mismatch.")
    width = right - left
    require(maximum_abs(width[:-1] - nominal_dt) < TIME_ATOL
            and 0 < width[-1] <= nominal_dt + TIME_ATOL,
            "Unexpected step-size change before the clipped final step.")
    require(maximum_abs(a["mass_before"][1:] - a["mass_stored"][:-1]) < 1e-10 * max(1.0, p0),
            "Discontinuous accepted mass history.")
    require(abs(float(a["mass_before"][0]) / p0 - 1.0) < 5e-10, "Launch norm mismatch.")
    require(np.all(a["roof_probability"] >= 0.0), "Negative detector event probability.")
    require(np.min(a["mass_stored"]) >= 0.0, "Negative stored norm.")
    max_residual = float(np.max(a["relative_residual"]))
    require(max_residual <= 2e-8, "Linear residual exceeds the inherited operational guard.")
    time = np.r_[0.0, right]
    survival = np.r_[a["mass_before"][0], a["mass_stored"]] / p0
    probability = a["roof_probability"] / p0
    cumulative = np.r_[0.0, np.cumsum(probability)]
    closure = maximum_abs(1.0 - survival - cumulative)
    require(closure <= ACCOUNTING_GUARD, "Detector/survival accounting exceeds the operational guard.")
    spin_trace_defect = maximum_abs(a["p_spin_up"] + a["p_spin_down"] - a["roof_probability"]) / p0
    require(spin_trace_defect <= 1e-12, "Spin-event trace does not equal detector probability.")
    u, d = a["p_spin_up"] / p0, a["p_spin_down"] / p0
    off2 = (a["p_spin_offdiag_real"] / p0) ** 2 + (a["p_spin_offdiag_imag"] / p0) ** 2
    require(np.min(u) >= -1e-14 and np.min(d) >= -1e-14
            and np.min(u * d - off2) >= -1e-12, "Nonpositive detector spin matrix.")
    with np.load(path / "revision_coordinates.npz", allow_pickle=False) as z:
        coords = {axis: z[axis].copy() for axis in ("x", "y", "z")}
    with np.load(path / "longitudinal_history.npz", allow_pickle=False) as h:
        longitudinal = {key: h[key].copy() for key in h.files}
    require(len(coords["z"]) == nz and float(coords["z"][0]) == z_bottom,
            "Lower-wall coordinate or node count mismatch.")
    for axis in coords:
        require(np.all(np.isfinite(coords[axis])) and np.all(np.diff(coords[axis]) > 0),
                f"Invalid {axis} coordinate array.")
    checks = validate_longitudinal(longitudinal, coords, cfg, time, survival,
                                   probability, left, right)
    external = None
    if postchecker is not None:
        external = postchecker(path, 0)
    files = {name: sha256(path / name) for name in REQUIRED_FILES}
    source_check = "source_snapshot.py unavailable; source hash was not independently verified"
    if (path / "source_snapshot.py").is_file():
        actual_hash = sha256(path / "source_snapshot.py")
        require(actual_hash == meta.get("updated_source_sha256"), "Source snapshot hash mismatch.")
        files["source_snapshot.py"] = actual_hash
        source_check = "source snapshot matches revision_run.json SHA256"
    validation = {
        "status": "ACCOUNTING_CHECKS_PASSED_NOT_CONVERGENCE",
        "postcheck": external,
        "postcheck_note": ("Adjacent/selected postcheck.py passed" if postchecker else
                           "No adjacent postcheck.py; internal record and metric checks only"),
        "source_verification": source_check,
        "max_abs_detector_survival_closure": closure,
        "max_true_relative_linear_residual": max_residual,
        "max_abs_event_spin_trace_defect": spin_trace_defect,
        "accepted_steps": n, "actual_terminal_time": float(right[-1]),
        "actual_last_step_duration": float(width[-1]),
        "operational_guards_not_accuracy_bounds": {
            "max_detector_survival_closure": ACCOUNTING_GUARD,
            "max_true_relative_linear_residual": 2e-8,
        },
        **checks,
    }
    for field in ("solve_and_matrix_defect", "cast_mass_loss", "clamp_mass_loss", "roof_cell_balance_defect"):
        if field in a.dtype.names:
            validation["sum_signed_" + field] = float(np.sum(a[field]) / p0)
            validation["max_abs_" + field] = maximum_abs(a[field]) / p0
    return {"path": path, "case_id": case_id, "omega": omega, "z_bottom": z_bottom,
            "configuration": cfg, "coords": coords, "longitudinal": longitudinal,
            "time": time, "survival": survival, "left": left, "right": right,
            "probability": probability, "validation": validation, "source_files": files,
            "initial_norm": p0}


def validate_longitudinal(h, coords, cfg, time, survival, probability, left, right):
    required = ("profile_time", "profile_step", "z", "bond_z", "plane_density", "bond_current",
                "monitor_bond_z", "monitor_left_index", "interval_midtime", "monitor_flux_midpoint",
                "monitor_cumulative", "layer_probability", "layer_balance_residual")
    require(all(key in h for key in required), "longitudinal_history.npz has an incomplete schema.")
    for key in required:
        require(np.all(np.isfinite(h[key])), f"Nonfinite longitudinal record: {key}")
    z = coords["z"].astype(np.float64)
    require(np.array_equal(h["z"], z), "Longitudinal/coordinate z arrays disagree.")
    require(np.array_equal(h["bond_z"], 0.5 * (z[:-1] + z[1:])), "Bond coordinates disagree.")
    ps = h["profile_step"].astype(np.int64)
    pt = h["profile_time"]
    require(np.array_equal(ps, h["profile_step"]) and np.all(np.diff(ps) > 0)
            and ps[0] == 0 and ps[-1] == len(left), "Incomplete endpoint profile steps.")
    require(np.array_equal(pt, time[ps]), "Profile times are not their accepted-step endpoints.")
    require(h["plane_density"].shape == (len(pt), len(z))
            and h["bond_current"].shape == (len(pt), len(z) - 1), "Invalid profile shapes.")
    require(h["monitor_flux_midpoint"].shape == (len(left), 2)
            and h["monitor_cumulative"].shape == (len(left) + 1, 2), "Invalid monitor-history shapes.")
    require(maximum_abs(h["interval_midtime"] - 0.5 * (left + right)) < TIME_ATOL,
            "Monitor times are not accepted CN midpoints.")
    mleft = h["monitor_left_index"].astype(np.int64)
    expected_left = np.array([int(np.argmin(abs(z - v))) for v in (16.0, 18.0)])
    require(np.array_equal(mleft, expected_left), "Monitors must use left nodes nearest z=16,18.")
    expected_mz = 0.5 * (z[mleft] + z[mleft + 1])
    require(np.array_equal(h["monitor_bond_z"], expected_mz), "Monitor bond-coordinate mismatch.")
    dens = h["plane_density"]
    hz = float(cfg["hz"])
    mass_from_profiles = hz * (np.sum(dens, axis=1) - 0.5 * dens[:, -1])
    norm_defect = maximum_abs(mass_from_profiles - survival[ps])
    require(norm_defect <= 2e-8, "Profile density normalization/half-roof metric disagrees with survival.")
    layers = np.stack([hz * (np.sum(dens[:, i + 1:], axis=1) - 0.5 * dens[:, -1])
                       for i in mleft], axis=1)
    require(h["layer_probability"].shape == layers.shape, "Invalid layer-probability shape.")
    layer_metric_defect = maximum_abs(h["layer_probability"] - layers)
    require(layer_metric_defect <= 2e-10, "Layer probability is not the centered half-roof mass.")
    mon_integral = np.vstack((np.zeros(2), np.cumsum((right - left)[:, None]
                            * h["monitor_flux_midpoint"], axis=0)))
    monitor_integral_defect = maximum_abs(mon_integral - h["monitor_cumulative"])
    require(monitor_integral_defect <= 2e-10, "Monitor cumulative integral uses incorrect interval widths.")
    roof_cumulative = np.r_[0.0, np.cumsum(probability)]
    balance = layers - layers[0] - mon_integral[ps] + roof_cumulative[ps, None]
    require(h["layer_balance_residual"].shape == balance.shape, "Invalid layer-balance shape.")
    balance_recompute_defect = maximum_abs(balance - h["layer_balance_residual"])
    require(balance_recompute_defect <= 2e-10, "Stored layer balance disagrees with independent reconstruction.")
    require(maximum_abs(balance) <= ACCOUNTING_GUARD, "Layer probability accounting exceeds operational guard.")
    return {"max_abs_profile_mass_vs_survival": norm_defect,
            "max_abs_layer_metric_reconstruction_defect": layer_metric_defect,
            "max_abs_monitor_integral_reconstruction_defect": monitor_integral_defect,
            "max_abs_layer_balance_reconstruction_defect": balance_recompute_defect,
            "max_abs_layer_balance": maximum_abs(balance),
            "monitor_bond_z": expected_mz.tolist(),
            "profile_time_spacing_min": float(np.min(np.diff(pt))),
            "profile_time_spacing_max": float(np.max(np.diff(pt)))}


def validate_family(runs):
    require(len(runs) == 5 and {r["case_id"] for r in runs} == set(FAMILY),
            "Supply exactly one completed output for each of 7750--7754.")
    by_id = {r["case_id"]: r for r in runs}
    base = by_id[7750]
    for run in runs:
        common = run["coords"]["z"][run["coords"]["z"] >= 0]
        require(np.array_equal(common, base["coords"]["z"]), "Nonnegative reference nodes are not identical.")
        require(float(run["configuration"]["hz"]) == float(base["configuration"]["hz"]),
                "Longitudinal stencil spacing differs within the family.")
        require(np.array_equal(run["longitudinal"]["monitor_bond_z"],
                               base["longitudinal"]["monitor_bond_z"]), "Monitor positions differ within family.")
        if run["omega"] == 200.0:
            for axis in ("x", "y"):
                require(np.array_equal(run["coords"][axis], base["coords"][axis]),
                        "Transverse grid differs between omega=200 wall controls.")
    require(np.array_equal(by_id[7753]["coords"]["z"], by_id[7754]["coords"]["z"]),
            "The omega comparison must share exactly the same longitudinal geometry.")
    return [by_id[k] for k in sorted(by_id)]


def monitor_summary(run, start, stop):
    h = run["longitudinal"]
    active = (run["right"] > start) & (run["left"] < stop)
    width = np.maximum(0.0, np.minimum(run["right"], stop) - np.maximum(run["left"], start))
    out = []
    for j, z in enumerate(h["monitor_bond_z"]):
        rate = h["monitor_flux_midpoint"][:, j]
        out.append({"z": float(z), "start": start, "stop": stop,
                    "current_min_on_overlapping_intervals": float(np.min(rate[active])),
                    "current_max_on_overlapping_intervals": float(np.max(rate[active])),
                    "signed_integrated_transport": float(np.dot(width, rate)),
                    "positive_integrated_transport": float(np.dot(width, np.maximum(rate, 0.0))),
                    "negative_integrated_transport_magnitude": float(np.dot(width, np.maximum(-rate, 0.0))),
                    "meaning": "signed probability transport, not an encounter or passage count"})
    return out


def row_for(run, horizon):
    values = statistics(run["time"], run["survival"], run["left"], run["right"],
                        run["probability"], horizon)
    return {"case_id": run["case_id"], "omega": run["omega"], "z_B": run["z_bottom"],
            "z_source": 10.0, "z_detector_actual": float(run["coords"]["z"][-1]),
            "Nz": len(run["coords"]["z"]), "hz": float(run["configuration"]["hz"]),
            "nominal_dt": float(run["configuration"]["dt"]),
            "last_step_duration": float(run["right"][-1] - run["left"][-1]), **values}


def write_table(path, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def profile_window(time, values, horizon):
    """Endpoint profiles, with only a necessary terminal plotting interpolation."""
    require(time[0] == 0 and horizon <= time[-1], "Profile window exceeds recorded support.")
    use = time <= horizon
    t, v = time[use], values[use]
    if t[-1] < horizon:
        i = int(np.searchsorted(time, horizon))
        weight = (horizon - time[i - 1]) / (time[i] - time[i - 1])
        terminal = (1.0 - weight) * values[i - 1] + weight * values[i]
        t, v = np.r_[t, horizon], np.concatenate((v, terminal[None, ...]), axis=0)
    return t, v


def stairs_window(run, rate, horizon):
    keep = run["left"] < horizon
    require(np.any(keep) and horizon <= run["right"][-1], "Invalid plotting horizon.")
    return np.r_[run["left"][0], np.minimum(run["right"][keep], horizon)], rate[keep]


def save_figure(fig, stem):
    paths = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix("." + suffix)
        temp = path.with_name(path.name + ".tmp")
        fig.savefig(temp, format=suffix, dpi=180 if suffix == "png" else 150,
                    bbox_inches="tight", facecolor="white")
        temp.replace(path)
        paths.append(path.name)
    return paths


def make_figures(runs, out, synthetic_label=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
        from matplotlib.lines import Line2D
    except ImportError:
        return {"status": "SKIPPED_MATPLOTLIB_UNAVAILABLE", "files": [],
                "note": "Tables/JSON were calculated; install Matplotlib and rerun to create figures."}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "savefig.facecolor": "white"})
    wall_runs = [r for r in runs if r["omega"] == 200.0]
    require(len(wall_runs) == 4, "Four omega=200 runs are needed for the wall maps.")
    vmax = max(maximum_abs(r["longitudinal"]["bond_current"]) for r in wall_runs)
    vmax = max(vmax, 1e-300) if vmax > 0 else 1.0
    norm = SymLogNorm(linthresh=vmax * 1e-3, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    files = []
    colors = ["#264E86", "#008A8A", "#CA6836", "#763C92"]
    for horizon in (30.0, 40.0):
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True, sharey=True)
        for i, (ax, run) in enumerate(zip(axes.flat, wall_runs)):
            h = run["longitudinal"]
            t, current = profile_window(h["profile_time"], h["bond_current"], horizon)
            ax.set_facecolor("#e7e7e7")
            mesh = ax.pcolormesh(t, h["bond_z"], current.T, cmap="RdBu_r", norm=norm,
                                 shading="nearest", rasterized=True)
            for z, style in zip(h["monitor_bond_z"], ("--", ":")):
                ax.axhline(float(z), color="#191919", linestyle=style, linewidth=0.9)
            ax.set_title(f"({chr(97+i)})  $z_B={run['z_bottom']:g}$  |  run {run['case_id']}")
            ax.set_xlim(0, horizon)
            ax.set_ylim(-15, float(run["coords"]["z"][-1]))
            if i >= 2:
                ax.set_xlabel("Time $t$")
            if i % 2 == 0:
                ax.set_ylabel("Longitudinal coordinate $z$")
        fig.suptitle(rf"Centered ABC: lower-wall dependence, $\omega=200$, $T={horizon:g}$", y=0.99)
        fig.subplots_adjust(left=0.07, right=0.86, bottom=0.13, top=0.91, hspace=0.19, wspace=0.10)
        cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
        fig.colorbar(mesh, cax=cax, label="Signed longitudinal current (positive toward detector)")
        monitor_z = wall_runs[0]["longitudinal"]["monitor_bond_z"]
        handles = [Line2D([0], [0], color="#191919", ls="--", label=f"Monitor at z={monitor_z[0]:.6f}"),
                   Line2D([0], [0], color="#191919", ls=":", label=f"Monitor at z={monitor_z[1]:.6f}")]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.45, 0.04), frameon=False)
        fig.text(0.07, 0.015, "Endpoint profiles; gray outside guide. Common symmetric-log color scale; no smoothing or time shift.", fontsize=8)
        if synthetic_label:
            fig.text(0.5, 0.5, synthetic_label, ha="center", va="center", color="black", fontsize=22, alpha=0.55, rotation=25)
        files += save_figure(fig, out / f"Fig_S11_centered_T{horizon:g}")
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
        for j, ax in enumerate(axes):
            for run, color in zip(wall_runs, colors):
                h = run["longitudinal"]
                edges, rate = stairs_window(run, h["monitor_flux_midpoint"][:, j], horizon)
                ax.stairs(rate, edges, linewidth=0.75, color=color,
                          label=rf"$z_B={run['z_bottom']:g}$", rasterized=True)
            ax.axhline(0.0, color="#333333", linewidth=0.6)
            ax.set_ylabel("Signed current")
            ax.set_title(f"Monitor bond at z={wall_runs[0]['longitudinal']['monitor_bond_z'][j]:.6f}")
            ax.grid(alpha=0.18)
            ax.set_xlim(0, horizon)
        axes[0].legend(ncol=4, frameon=False)
        axes[-1].set_xlabel("Time $t$")
        fig.suptitle(rf"Accepted CN-midpoint monitor currents: $\omega=200$, $T={horizon:g}$", y=0.99)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.text(0.07, 0.015, "Every accepted interval is shown as a constant midpoint rate. Positive current points toward the detector.", fontsize=8)
        if synthetic_label:
            fig.text(0.5, 0.5, synthetic_label, ha="center", fontsize=22, alpha=0.55, rotation=25)
        files += save_figure(fig, out / f"Monitor_currents_centered_T{horizon:g}")
        plt.close(fig)

    pair = [next(r for r in runs if r["case_id"] == cid) for cid in (7753, 7754)]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
    for run, color in zip(pair, ("#763C92", "#167C80")):
        roof_rate = run["probability"] / (run["right"] - run["left"])
        rates = [roof_rate] + [run["longitudinal"]["monitor_flux_midpoint"][:, j] for j in range(2)]
        for ax, rate in zip(axes, rates):
            edges, y = stairs_window(run, rate, 40.0)
            ax.stairs(y, edges, color=color, linewidth=0.75,
                      label=rf"$\omega={run['omega']:g}$ (run {run['case_id']})", rasterized=True)
            ax.axhline(0, color="#333333", linewidth=0.6)
            ax.grid(alpha=0.18)
            ax.set_xlim(0, 40)
    axes[0].set_title("Detector click rate from accepted CN events")
    axes[0].set_ylabel("Detector density")
    axes[0].legend(ncol=2, frameon=False)
    for j in range(2):
        axes[j + 1].set_title(f"Monitor bond at z={pair[0]['longitudinal']['monitor_bond_z'][j]:.6f}")
        axes[j + 1].set_ylabel("Signed current")
    axes[-1].set_xlabel("Time $t$")
    fig.suptitle(r"Confinement comparison at fixed $z_B=-15$, $z_c=10$, detector position, and $T=40$", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.text(0.07, 0.015, "Accepted-step rates on their actual time intervals; dt differs between the two preparations. No time alignment or smoothing.", fontsize=8)
    if synthetic_label:
        fig.text(0.5, 0.5, synthetic_label, ha="center", fontsize=22, alpha=0.55, rotation=25)
    files += save_figure(fig, out / "Omega_comparison_zBminus15_T40")
    plt.close(fig)
    return {"status": "CREATED", "files": files,
            "map_color_scale": {"type": "symmetric logarithmic", "vmin": -vmax, "vmax": vmax,
                                "linear_threshold": vmax * 1e-3,
                                "shared_across": "four omega=200 cases and both horizons"},
            "map_time_rule": "recorded endpoint profiles; interpolate only a missing terminal plotting profile",
            "trace_time_rule": "constant rate on each actual accepted CN interval; no thinning"}


def analyze(runs, out, no_plots=False):
    runs = validate_family(runs)
    out = Path(out).expanduser().resolve()
    require(all(out != r["path"] and r["path"] not in out.parents for r in runs),
            "Choose an analysis destination outside every input run directory.")
    out.mkdir(parents=True, exist_ok=True)
    rows = {str(int(t)): [row_for(r, t) for r in runs] for t in (30.0, 40.0)}
    for horizon, table in rows.items():
        write_table(out / f"Table_S11_centered_T{horizon}.csv", table)
    baseline = runs[0]
    early = []
    for run in runs[:4]:
        q = np.unique(np.r_[window_grid(baseline["time"], 0.0, 3.6), window_grid(run["time"], 0.0, 3.6)])
        delta = np.interp(q, baseline["time"], baseline["survival"]) - np.interp(q, run["time"], run["survival"])
        early.append({"case_id": run["case_id"], "z_B": run["z_bottom"],
                      "reference_case_id": 7750, "window": [0.0, 3.6],
                      "max_abs_survival_and_survival_defined_detection_difference": maximum_abs(delta)})
    record = {
        "schema": "CENTERED_FIXED_SOURCE_LOWERWALL_ANALYSIS_V1",
        "status": "COMPLETED_DATA_ANALYSIS_NOT_CONVERGENCE_CERTIFICATION",
        "normalization": "once by each recorded P0_init64; supplied normalized longitudinal arrays checked against this norm",
        "survival_rule": "piecewise linear on accepted stored-norm endpoints",
        "event_rule": "piecewise constant CN event probability divided by its actual interval duration; exact clipped overlap and first moment",
        "table_columns": {
            "D_T": "1-S(T)", "mean_cond": "[mu_star-T*S(T)]/D_T", "mu_star": "integral_0^T S(t) dt",
            "D_L": "S(3.6)-S(T)", "late_mean": "[3.6*S(3.6)-T*S(T)+integral_3.6^T S(t) dt]/D_L",
            "CN_suffix": "independent accepted detector-event estimate", "undefined_ratio": "null in JSON; empty CSV cell",
        },
        "scope": "7750--7753 compare four lower walls at omega=200; 7754 is a separate omega=1 control at z_B=-15",
        "limitations": ["Accounting guards do not certify physical, spatial, or temporal convergence.",
                        "Signed currents are net probability transport, not counts of returns or trajectories.",
                        "No individual peak is assigned a branch-specific round-trip interpretation.",
                        "The omega=1 and omega=200 cases use different transverse preparations and time steps."],
        "tables": rows, "early_wall_comparison": early,
        "runs": [{"case_id": r["case_id"], "omega": r["omega"], "z_B": r["z_bottom"],
                  "source_directory": str(r["path"]), "source_files_sha256": r["source_files"],
                  "validation": r["validation"],
                  "monitor_transport_T30": monitor_summary(r, 0.0, 30.0),
                  "monitor_transport_T40": monitor_summary(r, 0.0, 40.0),
                  "monitor_transport_3p6_to_6": monitor_summary(r, 3.6, 6.0)} for r in runs],
    }
    record["figures"] = ({"status": "SKIPPED_BY_REQUEST", "files": []} if no_plots else make_figures(runs, out))
    write_json(out / "lowerwall_analysis.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", metavar="RUN_DIR", type=Path, nargs="*", help="five extracted completed output directories")
    parser.add_argument("--runs", metavar="RUN_DIR", type=Path, nargs="+", help="alternative to positional directories")
    parser.add_argument("--out", type=Path, default=Path("centered_lowerwall_analysis"), help="analysis destination, outside inputs")
    parser.add_argument("--no-plots", action="store_true", help="produce CSV and JSON only")
    parser.add_argument("--postcheck", type=Path, help="explicit postcheck.py; by default use an adjacent file if present")
    args = parser.parse_args(argv)
    require(not (args.directories and args.runs), "Use positional directories or --runs, not both.")
    paths = args.runs or args.directories
    require(len(paths) == 5, "Exactly five completed directories, 7750--7754, are required.")
    checkpath = args.postcheck if args.postcheck is not None else Path(__file__).resolve().with_name("postcheck.py")
    if args.postcheck is not None:
        require(checkpath.is_file(), "The explicitly selected postcheck.py does not exist.")
    postchecker = load_postchecker(checkpath)
    runs = []
    for path in paths:
        print(f"Checking completed input: {path}", flush=True)
        runs.append(load_run(path, postchecker))
    record = analyze(runs, args.out, args.no_plots)
    print(json.dumps({"status": record["status"], "output_directory": str(args.out.expanduser().resolve()),
                      "tables": ["Table_S11_centered_T30.csv", "Table_S11_centered_T40.csv"],
                      "json": "lowerwall_analysis.json", "figures": record["figures"]["status"],
                      "convergence_established": False}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)

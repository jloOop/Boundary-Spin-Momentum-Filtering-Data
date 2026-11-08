#!/usr/bin/env python3
"""Recalculate ONLY Fig. S8 from accepted centered-ABC every-step histories.

Requires Python >=3.10, numpy and matplotlib. No simulation code is executed.
Run from any directory:
    python reproduce_fig_s8.py
Optional --data-root, --reference-root and --output-dir relocate inputs/outputs.
The default reference root is 3_CenteredABC_Run/1.Main (adds case 5043, omega=75).
Original figure files are used only for style/historical comparison, never as
inputs to a centered value or fit. Missing/incomplete runs are never extrapolated.
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
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fig_s8_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np

T = 20.0
WIDTHS = (0.25, 0.5, 0.75)
ORIGINAL_OMEGAS = (1, 25, 50, 75, 100, 150, 200)
EXCLUDED_FIGURE_OMEGAS = frozenset({75})  # User-requested exclusion for all widths.
INPUTS = ("revision_run.json", "constants.npz", "branch_refinement_case.json",
          "revision_coordinates.npz", "source_snapshot.py", "revision_step_data.csv")
COMPLETE = "COMPLETED_NUMERICAL_RUN_NOT_CONVERGENCE_CERTIFIED"
STYLE = {0.25: ("o", "-"), 0.5: ("s", "--"), 0.75: ("^", "-.")}
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent
DEFAULT_REFERENCE = HERE.parents[3] / "1.Main"
PROJECT = HERE.parents[4]
HISTORICAL = PROJECT / "2_FirstAcuuracy_Scripts_Run/R1_Sigma_Campaign/Output_"
SI = Path("EXTERNAL/Spin_Momentum_Now"
          "Spin_Momentum_Impedance_and_Filtering_RecentOne")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def same(a, b):
    a = np.pi if isinstance(a, str) and a == "pi" else a
    b = np.pi if isinstance(b, str) and b == "pi" else b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.isclose(a, b, rtol=0, atol=1e-12))


def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format(v, ".17g") if isinstance(v, float) else v
                             for k, v in row.items()})


def source_evidence(source, branch):
    """Inspect literal manifest and formula evidence without executing CuPy."""
    module = ast.parse(source)
    values = {n.targets[0].id: n.value for n in module.body
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name)}
    literal = ast.literal_eval(values["BRANCH_REFINEMENT_CASE"])
    if literal != branch:
        raise ValueError("source literal manifest differs from branch_refinement_case.json")
    patterns = {
        "Gaussian_amplitude_width":
            "psi_z = cp.exp(-(Z - z0_init)**2 / (2 * sigma_z**2))",
        "transverse_preparation": "psi_xy = cp.exp(-omega * Rho2 / 2)",
        "spin_up": "c_up = cp.cos(theta / 2)",
        "spin_down": "c_down = cp.sin(theta / 2) * cp.exp(1j * phi)",
        "norm_complex128": "v128=v.astype(cp.complex128,copy=False)",
        "norm_endpoint_weight":
            "total=total-(1.0-ENDPOINT_MASS_WEIGHT)*cp.vdot(roof,roof).real",
        "native_normalization": "total_prob = _abc_mass_native(psi_up,psi_down,hx*hy*hz)",
        "centered_normal_stencil": "scale=1.0/ENDPOINT_MASS_WEIGHT",
        "matched_tangential_factor": "coef = 0.5 / (hz * ENDPOINT_MASS_WEIGHT)",
        "detector_midpoint": "q=(oldroof+newroof)/2",
        "detector_probability":
            "pmat=_abc_spin_matrix(q[0],q[1],(right-left)*G['k_bc']*G['hx']*G['hy'])",
        "mass_defect": "defect=mass_before-mass_solved-pd",
    }
    evidence = {}
    for key, pattern in patterns.items():
        matches = [i for i, line in enumerate(source.splitlines(), 1) if pattern in line]
        if not matches:
            raise ValueError(f"unrecognized source implementation: {key}")
        evidence[key] = {"line": matches[0], "text": pattern}
    return evidence


def analyze(folder, expected_sigma):
    rec = {"run_id": folder.name.split(".")[0], "path": str(folder.resolve()),
           "expected_sigma_parallel": expected_sigma, "accepted": False,
           "rejection_reasons": [], "files": {}, "checks": {}}
    reasons, checks = rec["rejection_reasons"], rec["checks"]

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            reasons.append(name)

    try:
        for name in INPUTS:
            p = folder / name
            rec["files"][name] = {"sha256": sha(p), "bytes": p.stat().st_size}
        r = json.loads((folder / "revision_run.json").read_text())
        c = r["configuration"]
        b = json.loads((folder / "branch_refinement_case.json").read_text())
        with np.load(folder / "constants.npz", allow_pickle=False) as a:
            constants = {k: a[k].item() for k in a.files}
        rec.update(configuration=c, branch_refinement_case=b, constants=constants,
                   status=r["status"], metadata_actual_time=r["actual_time"],
                   metadata_completed_step=r["completed_step"], P0_init64=r["P0_init64"],
                   sigma_parallel=c["sigma_z"], omega=c["omega"],
                   source_environment=r.get("environment"), version=r["version"])
        check("source_sha256", rec["files"]["source_snapshot.py"]["sha256"]
              == r["updated_source_sha256"])
        check("configuration_sha256",
              hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
              == r["configuration_sha256"])
        rec["source_evidence"] = source_evidence(
            (folder / "source_snapshot.py").read_text(), b)
        fixed = dict(sigma_z=expected_sigma, k0=np.pi, k_bc=np.pi, Lz=20.0,
                     z0_init=10.0, theta=0.0, phi=0.0, Nx=100, Ny=100, Nz=1500,
                     grid_generator_nz=1500, ell=10.0, endpoint_closure="centered",
                     endpoint_mass_weight=0.5, metric="half_weight_last_z_node",
                     state_representation="physical_nodal_spinor",
                     state_dtype="complex64", initial_preparation_dtype="complex64",
                     potential_sampling_dtype="float32")
        for key, wanted in fixed.items():
            check("configuration_" + key, key in c and same(c[key], wanted))
            check("branch_" + key, key in b and same(b[key], wanted))
        for key in ("omega", "dt", "T_final"):
            check("branch_matches_" + key, same(c[key], b[key]))
        for key in ("omega", "T_final", "theta", "phi", "sigma_z", "k0", "k_bc",
                    "z_roof_actual", "grid_generator_nz", "hx", "hy", "hz",
                    "Lx", "Ly", "Lz", "dt", "Nx", "Ny", "Nz",
                    "endpoint_mass_weight", "endpoint_closure", "metric",
                    "state_representation", "state_dtype"):
            check("constants_matches_" + key, same(constants[key], c[key]))
        check("constants_z0", same(constants["z0"], c["z0_init"]))
        check("matrix_dtype", constants["matrix_dtype"] == b["matrix_dtype"] == "complex64"
              and c["matrix_assembly"] == "complex64_then_solve_complex128")
        check("transverse_domain", same(c["Lx"], 10 / np.sqrt(c["omega"]))
              and same(c["Ly"], 10 / np.sqrt(c["omega"])))
        check("spinor_boundary", c["boundary"] ==
              "centered scalar dissipative roof + matched Hermitian spin-tangential coupling")
        check("sole_detector", c["detector"] == "sole roof ABC; no bath and no Bohmian propagation")
        check("positive_base_dt", c["dt"] > 0)
        check("requested_horizon", c["T_final"] >= T)
        check("completed_status", r["status"] == COMPLETE)
        check("metadata_reaches_T20", r["actual_time"] >= T)
        check("metadata_reaches_requested_end", same(r["actual_time"], c["T_final"]))
        with np.load(folder / "revision_coordinates.npz", allow_pickle=False) as coords:
            rec["coordinate_summary"] = {}
            for axis in "xyz":
                x = coords[axis]
                n, length, h = c["N" + axis], c["L" + axis], c["h" + axis]
                check("coordinates_" + axis,
                      x.shape == (n,) and str(x.dtype) == "float32"
                      and float(x[0]) == 0 and np.all(np.diff(x) > 0)
                      and same(float(x[1] - x[0]), h))
                # GPU float32 linspace differs by at most a few coordinate ULPs.
                check("coordinate_grid_" + axis,
                      np.max(np.abs(x.astype(float) - np.arange(n) * length / n))
                      <= 4 * np.finfo(np.float32).eps * length)
                rec["coordinate_summary"][axis] = dict(
                    dtype=str(x.dtype), n=int(x.size), first=float(x[0]), last=float(x[-1]))
            check("detector_is_last_node", same(float(coords["z"][-1]), c["z_roof_actual"]))

        with (folder / "revision_step_data.csv").open() as f:
            header = next(csv.reader(f))
        check("step_columns", header == c["step_columns"])
        data = np.loadtxt(folder / "revision_step_data.csv", delimiter=",", skiprows=1,
                          dtype=np.float64, ndmin=2)
        col = {name: data[:, i] for i, name in enumerate(header)}
        left, right = col["t_left"], col["t_right"]
        p0 = float(r["P0_init64"])
        rec.update(csv_rows=len(data), csv_last_time=float(right[-1]),
                   csv_last_step=int(col["step"][-1]))
        check("finite_every_step_data", np.isfinite(data).all())
        check("consecutive_step_numbers",
              np.array_equal(col["step"], np.arange(1, len(data) + 1)))
        check("starts_at_zero", left[0] == 0)
        check("continuous_times", np.max(np.abs(left[1:] - right[:-1])) < 1e-12)
        check("positive_steps", np.all(right > left))
        check("dt_agrees", np.max(np.abs((right-left)[:-1] - c["dt"])) < 1e-11
              and right[-1] - left[-1] <= c["dt"] + 1e-11)
        check("csv_reaches_T20", right[-1] >= T)
        check("csv_matches_metadata_end", same(right[-1], r["actual_time"]))
        check("csv_matches_metadata_steps", len(data) == r["completed_step"])
        check("initial_mass_matches_P0_init64",
              p0 > 0 and same(col["mass_before"][0], p0))
        continuity = float(np.max(np.abs(col["mass_before"][1:] - col["mass_stored"][:-1])))
        check("stored_mass_continuity", continuity < 1e-12)
        check("nonnegative_detector_and_mass",
              np.min(col["roof_probability"]) >= 0 and np.min(col["mass_stored"]) >= 0)
        check("metadata_terminal_mass", same(col["mass_stored"][-1],
                                             r["current_accepted_prefix_mass64"]))
        check("metadata_detector_sum", abs(math.fsum(col["roof_probability"])
              - r["cumulative_roof_probability"]) < 2e-12)
        check("solver_guard", np.max(col["relative_residual"]) <= 2e-8)
        balance = (col["mass_before"] - col["mass_stored"] - col["roof_probability"])
        components = (col["solve_and_matrix_defect"] + col["cast_mass_loss"]
                      + col["clamp_mass_loss"])
        check("step_defect_components", np.max(np.abs(balance - components)) < 1e-13)
        rec["max_mass_continuity_error"] = continuity
        # Incomplete/mismatched candidates receive no mu*(20), detector mean or fit.
        if reasons:
            return None, rec

        # Include P0 at t=0; divide by P0_init64 exactly once. Never clip, enforce
        # monotonicity, smooth or reweight the already centered-metric stored mass.
        t = np.r_[0.0, right]
        mass = np.r_[p0, col["mass_stored"]]
        nodes = np.r_[t[t < T], T]
        survival = np.interp(nodes, t, mass) / p0
        mu = math.fsum(np.diff(nodes) * (survival[:-1] + survival[1:]) / 2)

        # Independent accepted-step detector route. If T cuts a step, allocate
        # its probability uniformly over that step and integrate only [left,T].
        mask = left < T
        l, u = left[mask], np.minimum(right[mask], T)
        fraction = (u-l) / (right[mask]-l)
        mid = (u+l) / 2
        pd = col["roof_probability"][mask] * fraction / p0
        d20 = math.fsum(pd)
        moment = math.fsum(mid * pd)
        tau = moment / d20
        s20 = float(survival[-1])
        budget = d20 + s20 - 1
        identity = mu - (d20 * tau + T * s20)
        epsilon = balance[mask] * fraction / p0
        epsilon_sum = math.fsum(epsilon)
        epsilon_moment = math.fsum(mid * epsilon)
        corrected_budget = budget + epsilon_sum
        corrected_identity = identity - epsilon_moment
        mass_ibp = math.fsum((survival[:-1]-survival[1:]) *
                            (nodes[:-1]+nodes[1:])/2) + T*s20
        max_budget = float(np.max(np.abs(np.cumsum(pd) + survival[1:] - 1)))
        check("T20_budget_operational_guard", max_budget <= 1e-3)
        check("budget_explained_by_recorded_defects", abs(corrected_budget) < 1e-11)
        check("mean_identity_explained_by_recorded_defects", abs(corrected_identity) < 1e-10)
        check("mass_integration_by_parts", abs(mu - mass_ibp) < 1e-10)
        check("restricted_mean_bounds", 0 <= mu <= T)
        if reasons:
            return None, rec
        rec["accepted"] = True
        result = dict(run_id=rec["run_id"], sigma_parallel=float(c["sigma_z"]),
                      omega=float(c["omega"]), T=T, mu_star_20=float(mu),
                      D20=float(d20), tau_cond20=float(tau), detector_first_moment_20=float(moment),
                      S20=s20, P0_init64=p0, probability_budget_residual=float(budget),
                      restricted_mean_identity_residual=float(identity),
                      normalized_nondetector_mass_loss=float(epsilon_sum),
                      time_weighted_nondetector_mass_loss=float(epsilon_moment),
                      budget_after_defect_accounting=float(corrected_budget),
                      identity_after_defect_accounting=float(corrected_identity),
                      mass_integration_by_parts_residual=float(mu-mass_ibp),
                      max_abs_prefix_probability_budget_residual=max_budget,
                      max_positive_survival_increment=float(max(0, np.diff(survival).max())),
                      max_relative_solver_residual=float(col["relative_residual"][mask].max()),
                      Nx=int(c["Nx"]), Ny=int(c["Ny"]), Nz=int(c["Nz"]), dt=float(c["dt"]),
                      hz=float(c["hz"]), detector_z=float(c["z_roof_actual"]),
                      full_history_end=float(right[-1]), steps_used=int(mask.sum()),
                      source_folder=str(folder.resolve()))
        for key in ("cast_mass_loss", "clamp_mass_loss", "solve_and_matrix_defect"):
            result[key + "_20_normalized"] = float(math.fsum(col[key][mask]*fraction/p0))
        rec["statistics"] = result.copy()
        return result, rec
    except (OSError, ValueError, KeyError, SyntaxError, IndexError) as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
        return None, rec


def fit_all(rows):
    coverage = {s: {r["omega"] for r in rows if r["sigma_parallel"] == s} for s in WIDTHS}
    common = sorted(set.intersection(*coverage.values()) - EXCLUDED_FIGURE_OMEGAS)
    if len(common) < 3:
        raise ValueError("Fewer than three common verified confinement points")
    fits, residuals = [], []
    for sigma in WIDTHS:
        series = [r for r in rows if r["sigma_parallel"] == sigma]
        chosen = [r for r in series if r["omega"] in common]
        w = np.array([r["omega"] for r in chosen])
        y = np.array([r["mu_star_20"] for r in chosen])
        x = np.column_stack([np.ones(len(w)), np.sqrt(w)])
        a, b = np.linalg.lstsq(x, y, rcond=None)[0]
        e = y - x @ [a, b]
        fits.append(dict(sigma_parallel=sigma, T=T, n_fit=len(w), A20=float(a), B20=float(b),
                         RMSE=float(np.sqrt(np.mean(e**2))),
                         R_squared=float(1 - np.sum(e**2)/np.sum((y-y.mean())**2)),
                         max_abs_residual=float(np.max(np.abs(e))),
                         omega_used=";".join(f"{v:g}" for v in common)))
        for r in series:
            included = r["omega"] in common
            pred = float(a + b*np.sqrt(r["omega"]))
            r["in_common_fit"] = included
            r["in_figure"] = r["omega"] not in EXCLUDED_FIGURE_OMEGAS
            r["figure_exclusion_reason"] = "omega=75 excluded by request" if not r["in_figure"] else ""
            residuals.append(dict(sigma_parallel=sigma, omega=r["omega"], run_id=r["run_id"],
                                  in_common_fit=included, in_figure=r["in_figure"],
                                  evaluation="fit" if included else
                                  ("outside_fit_range" if r["omega"] > max(common)
                                   else "held_out_noncommon"),
                                  mu_star_20=r["mu_star_20"], fitted_mu_star_20=pred,
                                  residual_mu_minus_fit=r["mu_star_20"] - pred))
    return common, coverage, fits, residuals


def plot(rows, fits, common, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 13, "legend.fontsize": 11,
                         "axes.linewidth": 1.0, "savefig.bbox": "tight",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, width=.9)
    ax.tick_params(axis="both", which="major", length=6)
    ax.tick_params(axis="both", which="minor", length=2.5)
    ax.grid(True, alpha=.10)
    handles, labels, vals = [], [], []
    for f in fits:
        sigma, a, b = f["sigma_parallel"], f["A20"], f["B20"]
        series = [r for r in rows if r["sigma_parallel"] == sigma
                  and r["omega"] not in EXCLUDED_FIGURE_OMEGAS]
        w = [r["omega"] for r in series]
        y = [r["mu_star_20"] for r in series]
        # Curves cover the common fitted range only; no invented simulation values.
        grid = np.linspace(min(common), max(common), 500)
        curve = a + b*np.sqrt(grid)
        marker, ls = STYLE[sigma]
        hf, = ax.plot(grid, curve, color="#050505", lw=.9, linestyle=ls)
        hd, = ax.plot(w, y, linestyle="none", marker=marker, markersize=7,
                      markerfacecolor="none", markeredgecolor="#1F4E79",
                      markeredgewidth=1.8, zorder=4)
        handles.append((hf, hd))
        labels.append(fr"${sigma:g}$: ${a:.3f}{b:+.3f}\sqrt{{\omega}}$")
        vals.extend([*y, *curve])
    ax.set_xlabel(r"$\omega$", fontweight="bold")
    ax.set_ylabel(r"$\mu^*(20)$", fontweight="bold")
    ax.set_xlim(-5, 208)
    ax.set_ylim(min(vals)-.8, max(vals)+2.8)
    ax.set_xticks(np.arange(0, 201, 25))
    legend = ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)},
                       frameon=True, loc="upper left",
                       title=r"$\sigma_{\parallel};\ \mu^*(20)\approx A+B\sqrt{\omega}$",
                       title_fontsize=11)
    legend.get_title().set_math_fontfamily("cm")
    fig.tight_layout()
    fig.savefig(out / "Fig_S8_centered_T20.pdf",
                metadata={"Title": "Fig. S8: longitudinal packet width, centered ABC, T=20",
                          "Creator": "reproduce_fig_s8.py (NumPy/Matplotlib)"})
    fig.savefig(out / "Fig_S8_centered_T20.png", dpi=600)
    # Small inspection copy; delivered publication PNG above remains 600 dpi.
    fig.savefig(out / "Fig_S8_centered_T20_preview.png", dpi=160)
    plt.close(fig)


def old_comparison(fits):
    """Historical displayed coefficients only; never used by fit_all or plot."""
    fp = HISTORICAL / "mu_star_sigma_campaign_plot_fits.csv"
    dp = HISTORICAL / "mu_star_sigma_campaign_plot_data.csv"
    if not fp.exists() or not dp.exists():
        return []
    with fp.open() as f:
        prior = list(csv.DictReader(f))
    with dp.open() as f:
        points = list(csv.DictReader(f))
    comparison = []
    for f in fits:
        s = f["sigma_parallel"]
        old = next(v for v in prior if float(v["sigma_parallel"]) == s)
        a, b = float(old["A"]), float(old["B"])
        old_res = [float(v["mu_star"]) - (a+b*np.sqrt(float(v["omega"])))
                   for v in points if float(v["sigma_parallel"]) == s]
        comparison.append(dict(sigma_parallel=s, old_n=7, new_n=f["n_fit"],
                               old_A20=a, new_A20=f["A20"], delta_A20=f["A20"]-a,
                               old_B20=b, new_B20=f["B20"], delta_B20=f["B20"]-b,
                               old_RMSE=float(old["rms_residual"]), new_RMSE=f["RMSE"],
                               old_R_squared=float(old["R_squared"]), new_R_squared=f["R_squared"],
                               old_max_abs_residual=float(max(abs(v) for v in old_res)),
                               new_max_abs_residual=f["max_abs_residual"],
                               caveat="Historical 7-point first-order figure; sigma=0.5 embedded example. "
                               f"New fit uses {f['n_fit']} common centered points; "
                               "not a controlled method comparison."))
    return comparison


def markdown_table(rows, keys, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(format(r[k], formats.get(k, ".12g"))
                     if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


def write_report(out, rows, audits, common, coverage, fits, residuals, comparison):
    accepted = [a for a in audits if a["accepted"]]
    rejected = [a for a in audits if not a["accepted"]]
    lines = [
        "# Fig. S8 / Sec. S9.3 numerical validation",
        "",
        "Scope: Fig. S8 only. Manuscript unchanged. These fixed-grid runs do not certify "
        "spatial or temporal convergence.",
        "",
        f"Accepted {len(accepted)} of {len(audits)} candidates through T=20.",
        "Common unweighted fit set: " + ", ".join(f"{w:g}" for w in common) + ".",
        "Omega=75 is excluded from markers and fitting for all three widths by request. "
        "The verified sigma=0.5, omega=75 result remains in the audit CSV and provenance "
        "with in_figure=False. Other verified numerical points are shown. "
        "Fits use only the common set and curves "
        "are drawn only over its range. Other verified points do not enter any coefficients "
        "or fit-quality statistic.",
        "",
        "## Coverage and exclusions", "",
    ]
    for s in WIDTHS:
        missing = sorted(set(ORIGINAL_OMEGAS)-coverage[s])
        lines.append(f"- sigma={s:g}: verified omega={sorted(coverage[s])}; "
                     f"missing/unusable original points={missing}.")
    lines += [
        "- Case 5043 (sigma=0.5, omega=75) was recovered from the completed main confinement "
        "sweep in 3_CenteredABC_Run/1.Main. It was absent from the supplied width folder.",
    ]
    for a in accepted:
        if a["run_id"] == "7681":
            lines.append(f"- Case 7681 is now completed and accepted: metadata and CSV "
                         f"both reach t={a['csv_last_time']:g}, with {a['csv_rows']} "
                         "consecutive accepted steps. Its omega=200 point enters the common fit.")
    for a in rejected:
        lines.append(f"- REJECTED case {a['run_id']}: status={a.get('status')}; "
                     f"metadata t={a.get('metadata_actual_time')}; CSV t={a.get('csv_last_time')}; "
                     f"reasons={'; '.join(a['rejection_reasons'])}. No T20 value assigned.")
    lines += [
        "", "## Preparation and discrete conventions", "",
        "Cross-checked revision_run.json, constants.npz, branch_refinement_case.json, "
        "revision_coordinates.npz and source_snapshot.py for each candidate. Source and "
        "configuration SHA256 fingerprints match. The script records individual checks and "
        "source line evidence in run_provenance_manifest.json.",
        "",
        "All accepted runs use kz=kappa=pi, nominal L=20, zc=10, eta=(1,0)^T "
        "(theta=phi=0), Nx=Ny=100, Nz=1500, centered spinor ABC with matched "
        "tangential coupling, float32 coordinates/potential and complex64 preparation/state/"
        "matrix assembly with complex128 solves. Transverse Gaussian amplitude is "
        "exp[-omega*r_perp^2/2] and Lx=Ly=10/sqrt(omega). Longitudinal amplitude is "
        "exp[-(z-zc)^2/(2*sigma_parallel^2)] exp[ikz(z-zc)]; thus sigma_parallel "
        "is the amplitude width (density standard deviation sigma_parallel/sqrt(2)).",
        "",
        "The stored mass is hx*hy*hz times the spinor squared norm with HALF weight on "
        "the final z node. It is already in the centered endpoint metric; no second "
        "endpoint correction is applied. Actual detector z=19.986665725708008, "
        "hz=0.013333333656191826 (the nominal guide length is 20).",
        "",
        "The sigma=0.5 reference histories run to 40 and are restricted to 20; width "
        "histories run to 20. All T=20 endpoints used here are actual accepted-step nodes. "
        "Every row, including times, step numbering, stored-mass continuity, metadata "
        "terminal mass/probability and solver residual, is checked.",
        "",
        "## Calculation and residual definitions", "",
        "S(0)=P0_init64/P0_init64=1. At every accepted right endpoint "
        "S(t_n)=mass_stored_n/P0_init64. Integrate the piecewise-linear interpolant "
        "by trapezoids from 0 to 20. No clipping, smoothing, tail extension or "
        "substitution from coarse sampled histories is used.",
        "",
        "Independently, p_n=roof_probability_n/P0_init64; D20=sum p_n, "
        "M20=sum [(t_left+t_right)/2] p_n, tau_cond20=M20/D20. "
        "These are accepted-step midpoint detector probabilities, not sampled currents "
        "or mass differences. Normalize each physical mass/probability once.",
        "",
        "Signed probability-budget residual r_P = D20+S20-1. Signed restricted-mean "
        "residual r_mu = mu_star_20-[D20*tau_cond20+20*S20].",
        "",
        "Finite-precision mass defects prevent an exactly zero detector identity. "
        "epsilon_n=(mass_before-mass_stored-roof_probability)/P0_init64 "
        "equals (solve_and_matrix_defect+cast_mass_loss+clamp_mass_loss)/P0_init64. "
        "The independently checked relations are r_P=-sum epsilon_n and "
        "r_mu=sum t_mid,n*epsilon_n. Neither residual is forcibly zeroed. "
        "The exact discrete integration-by-parts check using mass differences is also "
        "reported separately and is not substituted for the detector check.",
        "",
        "Operational acceptance limits: recorded solver relative residual <=2e-8 "
        "and cumulative normalized probability discrepancy <=1e-3. These are the "
        "simulation fail-stop guards, not accuracy/convergence claims. Postprocessing "
        "requires defect-accounted budget <1e-11 and defect-accounted mean identity "
        "<1e-10; metadata and mass continuity are checked at 1e-12 absolute tolerance.",
        "",
        "## Per-run results and validation", "",
        markdown_table(rows, ["run_id", "sigma_parallel", "omega", "dt", "steps_used",
                              "mu_star_20", "D20", "tau_cond20", "S20",
                              "probability_budget_residual",
                              "restricted_mean_identity_residual"]),
        "",
        f"Maximum |r_P|={max(abs(r['probability_budget_residual']) for r in rows):.12g}; "
        f"maximum |r_mu|={max(abs(r['restricted_mean_identity_residual']) for r in rows):.12g}.",
        f"Maximum defect-accounted |r_P|="
        f"{max(abs(r['budget_after_defect_accounting']) for r in rows):.12g}; "
        f"maximum defect-accounted |r_mu|="
        f"{max(abs(r['identity_after_defect_accounting']) for r in rows):.12g}.",
        "",
        "## Unweighted square-root fits", "",
        "Residual means numerical mu minus A20+B20*sqrt(omega). RMSE uses denominator "
        "n (not n-2); R_squared=1-SSE/SST. Coefficients use NumPy lstsq, no weights.",
        "", markdown_table(fits, ["sigma_parallel", "n_fit", "omega_used", "A20", "B20",
                                  "RMSE", "R_squared", "max_abs_residual"]), "",
        markdown_table(residuals, ["sigma_parallel", "omega", "in_common_fit", "evaluation",
                                   "mu_star_20", "fitted_mu_star_20", "residual_mu_minus_fit"]),
        "",
        "Noncommon residuals are held-out evaluations and do not enter fit statistics. "
        "Any prediction labeled outside_fit_range is an extrapolation of the fitted "
        "formula, not an additional simulation value. Curves cover the common range only.",
        "", "## Interpretation", "",
    ]
    for s in WIDTHS:
        series = [r for r in rows if r["sigma_parallel"] == s]
        diffs = np.diff([r["mu_star_20"] for r in series])
        lines.append(f"- sigma={s:g}: mu increases from {series[0]['mu_star_20']:.9f} "
                     f"at omega={series[0]['omega']:g} to {series[-1]['mu_star_20']:.9f} "
                     f"at omega={series[-1]['omega']:g}; all successive sampled increments "
                     f"are positive={bool(np.all(diffs > 0))}.")
    order = []
    for w in common:
        at_w = sorted((r for r in rows if r["omega"] == w),
                      key=lambda r: r["mu_star_20"], reverse=True)
        order.append({"omega": w, "largest_to_smallest_sigma":
                      ">".join(f"{r['sigma_parallel']:g}" for r in at_w)})
    lines += ["", markdown_table(order, ["omega", "largest_to_smallest_sigma"]), ""]
    by_pair = {(r["sigma_parallel"], r["omega"]): r for r in rows}
    if (.25, 200.) in by_pair and (.5, 200.) in by_pair:
        delta = by_pair[(.5, 200.)]["mu_star_20"] - by_pair[(.25, 200.)]["mu_star_20"]
        lines.append(f"At omega=200, mu(sigma=0.5)-mu(sigma=0.25)={delta:.12g}. "
                     "The ordering of these two numerical series reverses between "
                     "omega=150 and 200. No simulation locates the precise crossing. "
                     "The omega=200 runs use dt=0.00025 (sigma=0.5) and dt=0.0002 "
                     "(sigma=0.25); the small reversal is a feature of these finite-grid "
                     "results, not a convergence-certified crossing.")
    for i, fa in enumerate(fits):
        for fb in fits[i+1:]:
            root = (fb["A20"]-fa["A20"])/(fa["B20"]-fb["B20"])
            if root >= 0:
                w = root**2
                lines.append(f"Fitted-formula crossing sigma={fa['sigma_parallel']:g}/"
                             f"{fb['sigma_parallel']:g}: omega={w:.9g}; "
                             f"inside common fitted range={min(common) <= w <= max(common)}. "
                             "This is a fit intersection, not an observed simulation crossing.")
            else:
                lines.append(f"No nonnegative-omega fitted crossing for sigma="
                             f"{fa['sigma_parallel']:g}/{fb['sigma_parallel']:g}.")
    lines += [
        "", "Ordering refers only to sampled common points. Exact crossing locations "
        "are not determined by these data; fitted-curve intersections need not agree "
        "with the ordering of individual numerical markers.",
        "",
        "## Historical Fig. S8 comparison", "",
        "The original seven-point figure is a historical display comparison only. "
        "It used older first-order histories, and its plotting script substituted "
        "embedded example values for sigma=0.5 by default. Both the numerical method "
        "and fit point set differ. Changes below cannot isolate an ABC effect, "
        "nor establish convergence. No historical values enter the new fits.",
        "",
        markdown_table(comparison, ["sigma_parallel", "old_A20", "new_A20", "delta_A20",
                                    "old_B20", "new_B20", "delta_B20",
                                    "old_RMSE", "new_RMSE", "old_R_squared", "new_R_squared"])
        if comparison else "Historical files unavailable; comparison omitted.",
        "",
        "## Figure and reproduction", "",
        "Matches the current Fig. S8's 8.6 x 5.2 inch nominal canvas, tight crop, "
        "blue open circles/squares/triangles, black solid/dashed/dash-dot curves, "
        "DejaVu Sans fonts (12/13/11 pt), Computer Modern legend title, 0.4 pt "
        "axes, inward ticks, light grid and upper-left equation legend. Changes: "
        "ylabel explicitly mu*(20), recalculated markers/equations, and fit curves "
        f"span omega={min(common):g} to {max(common):g}, the common fit range.",
        "",
        "Run `python reproduce_fig_s8.py` with numpy and matplotlib installed. "
        "Default paths resolve relative to this script; --data-root, --reference-root "
        "and --output-dir can relocate data. Inputs are read-only. CSV numbers use "
        "17 significant digits; this reports numerical precision, not physical accuracy.",
        "",
        "Deliverables: vector PDF; 600-dpi PNG; results_full_precision.csv; "
        "fit_coefficients.csv; fit_residuals.csv; historical_fit_comparison.csv; "
        "validation_checks.csv; run_provenance_manifest.json; this report; "
        "Fig_S8_caption.tex; S9_3_replacement_paragraph.tex; reproduce_fig_s8.py.",
    ]
    (out / "numerical_validation_report.md").write_text("\n".join(lines) + "\n")


def write_latex(out, fits, rows, common):
    def tex_set(values):
        return r"\{" + ",".join(f"{v:g}" for v in values) + r"\}"

    extra_points, missing_points = [], []
    for sigma in WIDTHS:
        available = {r["omega"] for r in rows if r["sigma_parallel"] == sigma}
        extra = sorted(available - set(common) - EXCLUDED_FIGURE_OMEGAS)
        missing = sorted(set(ORIGINAL_OMEGAS) - available - EXCLUDED_FIGURE_OMEGAS)
        if extra:
            extra_points.append(r"$\omega\in" + tex_set(extra)
                                + rf"$ for $\sigma_\parallel={sigma:g}$")
        if missing:
            missing_points.append(r"$\omega\in" + tex_set(missing)
                                  + rf"$ for $\sigma_\parallel={sigma:g}$")
    extra_note = ("Additional verified markers at " + "; ".join(extra_points)
                  + " are excluded from all fits.\n") if extra_points else ""
    missing_note = ("Unavailable points from the original confinement set are "
                    + "; ".join(missing_points) + ".\n") if missing_points else ""
    (out / "Fig_S8_caption.tex").write_text(
        r"""\caption{Restricted mean detection times
$\mu^*(20;\omega,\sigma_\parallel)=\int_0^{20}S(t;\omega,\sigma_\parallel)\,dt$
for Gaussian amplitude widths $\sigma_\parallel=0.25,0.5,0.75$,
with $L=20$, $z_c=10$, $k_z=\kappa=\pi$ and initial spinor
$\eta=(1,0)^{\mathsf T}$, using the centered spinor absorbing boundary condition.
Open circles, squares and triangles show the independently integrated,
every-step stored-mass histories, normalized by the initial mass.
The solid, dashed and dash-dotted curves are unweighted fits
$A_{20}+B_{20}\sqrt{\omega}$ on the same common set
""" + r"$\omega\in" + tex_set(common)
        + "$; the fitted equations appear in the legend.\n"
        + extra_note + missing_note
        + r"The confinement $\omega=75$ is excluded for all three widths." + "\n"
        + r"""Fit curves are shown only over the fitted confinement range.}
\label{fig:width-confinement}
""")
    coeff = ", ".join(f"$({f['A20']:.6f},{f['B20']:.6f})$" for f in fits)
    rmse = ", ".join(f"${f['RMSE']:.6f}$" for f in fits)
    r2 = ", ".join(f"${f['R_squared']:.6f}$" for f in fits)
    m200 = {r["sigma_parallel"]: r["mu_star_20"] for r in rows if r["omega"] == 200}
    order_groups = {}
    for omega in common:
        series = sorted((r for r in rows if r["omega"] == omega),
                        key=lambda r: r["mu_star_20"], reverse=True)
        order = tuple(r["sigma_parallel"] for r in series)
        order_groups.setdefault(order, []).append(omega)
    order_text = ""
    for order, omegas in order_groups.items():
        expression = ">".join(rf"\mu^*_{{\sigma_\parallel={s:g}}}" for s in order)
        order_text += ("The numerical ordering is $" + expression + "$ at "
                       + r"$\omega\in" + tex_set(omegas) + "$. ")
    terminal_text = ""
    if all(s in m200 for s in WIDTHS):
        terminal_text = (r"At $\omega=200$, the means are "
                         + ", ".join(f"${m200[s]:.6f}$" for s in WIDTHS)
                         + r" for $\sigma_\parallel=0.25,0.5,0.75$, respectively. ")
    by_pair = {(r["sigma_parallel"], r["omega"]): r["mu_star_20"] for r in rows}
    if all((s, w) in by_pair for s in (.25, .5) for w in (150, 200)):
        d150 = by_pair[(.5, 150)] - by_pair[(.25, 150)]
        d200 = by_pair[(.5, 200)] - by_pair[(.25, 200)]
        if d150 * d200 < 0:
            terminal_text += (r"The $\sigma_\parallel=0.25$ and $0.5$ series "
                              r"therefore reverse their ordering between "
                              r"$\omega=150$ and $200$ without locating an exact crossing. ")
    (out / "S9_3_replacement_paragraph.tex").write_text(
        "% Replace only the Fig. S8 confinement discussion in Sec. S9.3.\n"
        "% Leave the Fig. S7 discussion untouched.\n"
        r"Figure~\ref{fig:width-confinement} shows that the restricted mean at $T=20$ "
        r"increases with confinement at every verified sampled point for all three "
        r"Gaussian amplitude widths. On the common set "
        + r"$\omega\in" + tex_set(common) + "$, unweighted fits "
        r"$\mu^*(20)=A_{20}+B_{20}\sqrt{\omega}$ give "
        f"the coefficient pairs $(A_{{20}},B_{{20}})$ as {coeff}, respectively, for "
        r"$\sigma_\parallel=0.25,0.5,0.75$; "
        f"the corresponding RMSEs are {rmse}, with corresponding $R^2$ values {r2}. "
        + order_text + terminal_text
        + r"The fitted intercept decreases and the fitted slope increases with width. "
        r"These are empirical finite-horizon fits; they do not establish an asymptotic "
        r"square-root law or spatial or temporal convergence. "
        r"Only completed centered-ABC histories are included. The confinement "
        r"$\omega=75$ is excluded from all three plotted series and all fits."
        "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output-dir", type=Path, default=HERE)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for sigma in WIDTHS:
        group = args.data_root / (f"sigma={sigma:g}_kz=kappa=pi" + ("_main" if sigma == .5 else ""))
        for meta in sorted(group.glob("*/revision_run.json")):
            candidates.append((meta.parent, sigma))
    # The supplied sigma=.5 folder lacks omega75; add the identified main run.
    for meta in sorted(args.reference_root.glob("5043.*/revision_run.json")):
        candidates.append((meta.parent, .5))
    if not candidates:
        raise SystemExit("No candidate datasets found")
    rows, audits = [], []
    for folder, sigma in candidates:
        row, audit = analyze(folder, sigma)
        audits.append(audit)
        print(f"{audit['run_id']}: {'ACCEPTED' if row else 'REJECTED'} "
              f"{'; '.join(audit['rejection_reasons'])}", flush=True)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (r["sigma_parallel"], r["omega"]))
    if len({(r["sigma_parallel"], r["omega"]) for r in rows}) != len(rows):
        raise ValueError("Ambiguous duplicate width/confinement pair; choose a single verified source")
    common, coverage, fits, residuals = fit_all(rows)
    comparison = old_comparison(fits)
    write_csv(out / "results_full_precision.csv", rows)
    write_csv(out / "fit_coefficients.csv", fits)
    write_csv(out / "fit_residuals.csv", residuals)
    if comparison:
        write_csv(out / "historical_fit_comparison.csv", comparison)
    write_csv(out / "validation_checks.csv", [
        dict(run_id=a["run_id"], sigma_parallel=a.get("sigma_parallel"),
             omega=a.get("omega"), accepted=a["accepted"], status=a.get("status"),
             metadata_actual_time=a.get("metadata_actual_time"), csv_last_time=a.get("csv_last_time"),
             csv_rows=a.get("csv_rows"), rejection_reasons="; ".join(a["rejection_reasons"]),
             **a["checks"]) for a in audits])
    plot(rows, fits, common, out)
    write_report(out, rows, audits, common, coverage, fits, residuals, comparison)
    write_latex(out, fits, rows, common)
    reference_paths = [
        SI / "AJ_MainBody_Supp_Revision10.tex", SI / "mu_star_sigma_campaign.png",
        HISTORICAL / "plot_sigma_campaign.py",
        HISTORICAL / "mu_star_sigma_campaign_plot_fits.csv",
        HISTORICAL / "mu_star_sigma_campaign_plot_data.csv"]
    manifest = dict(
        scope="Only Fig. S8 / Sec. S9.3; no manuscript edits",
        horizon=T, common_fit_omegas=common,
        figure_excluded_omegas=sorted(EXCLUDED_FIGURE_OMEGAS),
        plotted_coverage={str(k): sorted(v - EXCLUDED_FIGURE_OMEGAS) for k,v in coverage.items()},
        coverage={str(k): sorted(v) for k, v in coverage.items()},
        missing_original_points={str(k): sorted(set(ORIGINAL_OMEGAS)-v) for k,v in coverage.items()},
        preparation="Gaussian amplitude exp(-(z-zc)^2/(2 sigma_parallel^2))*exp(i kz(z-zc)); eta=(1,0)^T",
        normalization="P0_init64 once; stored masses already use centered endpoint metric",
        residual_signs=dict(probability="D20+S20-1",
                            restricted_mean="mu20-(D20*tau_cond20+20*S20)"),
        fitting="unweighted numpy.linalg.lstsq on identical common set",
        curve_domain=[min(common), max(common)],
        runs=audits,
        reference_assets=[dict(path=str(f), sha256=sha(f), purpose="style/historical comparison only")
                          for f in reference_paths if f.exists()],
        software=dict(python=platform.python_version(), numpy=np.__version__,
                      matplotlib=matplotlib.__version__, executable=sys.executable),
        calculation_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        output_sha256={f.name: sha(f) for f in out.iterdir() if f.is_file()
                       and f.name != "run_provenance_manifest.json"},
        convergence_claim=False)
    dump_json(out / "run_provenance_manifest.json", manifest)
    print("Common fit confinement values:", common)
    print(markdown_table(fits, ["sigma_parallel", "A20", "B20", "RMSE", "R_squared", "max_abs_residual"]))


if __name__ == "__main__":
    main()

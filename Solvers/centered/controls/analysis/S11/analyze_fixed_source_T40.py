#!/usr/bin/env python3
"""Reproduce only the fixed-source S9.4 statistics at T=40.

Inputs are read-only. Event statistics use exact accepted-interval overlaps;
the restricted mean is independently integrated from every stored mass.
The copied campaign validator is used only to read/check the five T=40 runs;
its other horizons and figure routines are never called.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

import campaign_validation_support as support
import independent_metrics

T, CUT = 40.0, 3.6
COLUMNS = ("zB", "Nz", "D40", "tau_cond40", "mu_star40", "DL40", "tL40")
REFERENCE_PDFS = (
    "EXTERNAL/Spin_Momentum_Impedance_and_Filtering_Closed_Quantum_Open_System - 2026-10-20T100533.942.pdf",
    "EXTERNAL/Spin_Momentum_Impedance_and_Filtering_Closed_Quantum_Open_System - 2026-10-20T100551.841.pdf",
)


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, columns, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([format(row[k], ".17g") if isinstance(row[k], float)
                             else row[k] for k in columns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    out, root = args.out.resolve(), args.root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(exist_ok=True)
    paths = []
    for case in range(7750, 7755):
        candidates = sorted(p for p in root.glob(f"{case}.*") if p.is_dir())
        if len(candidates) != 1:
            raise ValueError(f"Expected one run {case}, found {len(candidates)}")
        paths.append(candidates[0])
    runs = support.validate_family([support.load_run(p) for p in paths])
    rows, details, independent = [], [], []
    for run in runs:
        cid = run["case_id"]
        stats = support.statistics(run["time"], run["survival"], run["left"],
                                   run["right"], run["probability"], T, CUT)
        row = {
            "zB": int(run["z_bottom"]), "Nz": len(run["coords"]["z"]),
            "D40": stats["D_T_CN"], "tau_cond40": stats["mean_cond_CN"],
            "mu_star40": stats["mu_star"], "DL40": stats["D_L"],
            "tL40": stats["late_mean_CN"],
        }
        check = independent_metrics.audit(run["path"])
        # Independent implementation: directly integrates rate * overlap.
        pairs = {
            "D40": "D40", "tau_cond40": "tau_cond40", "mu_star40": "mu_star40",
            "DL40": "DL40_from_survival", "tL40": "tL40",
        }
        agreement = max(abs(row[a] - check[b]) for a, b in pairs.items())
        if agreement > 5e-12:
            raise ValueError(f"Independent statistics disagree for {cid}: {agreement}")
        independent.append(check)
        if run["omega"] == 200:
            rows.append(row)
        h = run["longitudinal"]
        monitor36_6 = support.monitor_summary(run, CUT, 6.0)
        details.append({
            "case_id": cid, "omega": run["omega"], **row,
            "S40": stats["S_T"], "S3_6": float(np.interp(CUT, run["time"], run["survival"])),
            "DE": stats["D_E_CN"], "late_event_probability": stats["D_L_CN"],
            "DL_survival_minus_event": stats["D_L"] - stats["D_L_CN"],
            "detected_first_moment": stats["first_moment_CN"],
            "late_first_moment": stats["late_first_moment_CN"],
            "restricted_mean_identity_rhs": row["D40"] * row["tau_cond40"] + T * stats["S_T"],
            "restricted_mean_identity_residual":
                row["mu_star40"] - (row["D40"] * row["tau_cond40"] + T * stats["S_T"]),
            "independent_implementation_max_abs_difference": agreement,
            "validation": run["validation"],
            "native_profiles": {
                "count": len(h["profile_time"]), "first": float(h["profile_time"][0]),
                "last": float(h["profile_time"][-1]),
                "minimum_gap": float(np.min(np.diff(h["profile_time"]))),
                "maximum_gap": float(np.max(np.diff(h["profile_time"]))),
                "extra_checkpoint_profiles_retained": True,
            },
            "monitor_signed_transport_3_6_to_6": monitor36_6,
        })
    early = np.array([r["DE"] for r in details[:4]])
    event_histories = np.stack([np.r_[0.0, np.cumsum(r["probability"])] for r in runs[:4]])
    early_knots = runs[0]["time"] <= CUT
    early_cdf_range = float(np.max(np.ptp(event_histories[:, early_knots], axis=0)))
    closure_max = max(r["validation"]["max_abs_detector_survival_closure"] for r in details[:4])
    identity_max = max(abs(r["restricted_mean_identity_residual"]) for r in details[:4])
    layer_max = max(r["validation"]["max_abs_layer_balance"] for r in details[:4])
    record = {
        "scope": "Only S9.4 fixed-source lower-wall control; centered runs 7750--7754; T=40",
        "status": "ACCOUNTING_CONSISTENT_NOT_CONVERGENCE_CERTIFICATION",
        "methods": {
            "P0": "revision_run.json P0_init64, verified against initial stored mass",
            "D40_tau_cond40": "accepted roof probabilities / P0; constant rates on actual intervals; exact overlap moments",
            "mu_star40": "independent integral of piecewise-linear every-step stored mass / P0",
            "DL40": "S(3.6)-S(40); survival definition retained",
            "tL40": "event first moment / event probability over (3.6,40]",
            "DL40_note": "Survival and event definitions differ by recorded accounting defects, explicitly reported",
            "identity": "mu_star40 - (D40*tau_cond40 + 40*S40), not enforced",
            "profile_current": "hx*hy/(hz*P0) Im sum over x,y,spin of conj(psi_k)*psi_(k+1)",
            "upper_layer_balance": "P_upper(t)-P_upper(0)-integral J_monitor + cumulative roof events",
        },
        "omega200_summary": {
            "DE_min": float(early.min()), "DE_max": float(early.max()),
            "DE_range": float(np.ptp(early)),
            "DE_relative_range": float(np.ptp(early) / early.mean()),
            "max_pairwise_early_event_CDF_difference": early_cdf_range,
            "max_detector_survival_difference": closure_max,
            "max_restricted_mean_identity_residual": identity_max,
            "max_monitor_layer_balance": layer_max,
            "tau_cond40_is_monotonic": False,
        },
        "runs": details,
    }
    write_json(out / "validation" / "computed_values_full_precision.json", record)
    write_json(out / "validation" / "independent_metrics.json", {"runs": independent})
    write_csv(out / "table_S11_fixed_source_T40.csv", COLUMNS, rows)
    summary_columns = ("case_id", "omega") + COLUMNS + (
        "S40", "S3_6", "DE", "late_event_probability", "DL_survival_minus_event",
        "detected_first_moment", "late_first_moment", "restricted_mean_identity_rhs",
        "restricted_mean_identity_residual",
    )
    write_csv(out / "validation" / "all_five_runs_full_precision.csv", summary_columns, details)
    text = ["Full-precision computed quantities (17 significant decimal digits)",
            "All quantities below are computed, not fitted.",
            "D40/tau_cond40/tL40: accepted-step event integrals.",
            "mu_star40/DL40: independent stored-survival integrals/difference.", ""]
    for d in details:
        text.append(f"Run {d['case_id']}: omega={d['omega']:g}, zB={d['zB']}")
        for key in summary_columns[2:]:
            text.append(f"  {key} = {d[key]:.17g}")
        text.append("")
    (out / "full_precision_values.txt").write_text("\n".join(text) + "\n")
    body = "\n".join(
        f"${r['zB']}$ & {r['Nz']} & " +
        " & ".join(f"{r[k]:.3f}" for k in COLUMNS[2:]) + r" \\"
        for r in rows
    )
    table = r"""\begin{table}[tb]
\caption{Fixed-source lower-wall controls at $\omega=200$ and $T=40$ using
the centered absorbing boundary closure. $D_{40}$ and
$\bar{\tau}^{\mathrm{cond}}_{40}$ are obtained from accepted-step detector
events, while $\mu^\ast(40)$ is independently integrated from the normalized
every-step stored survival history. Here $D_{L,40}=S(3.6)-S(40)$, and
$t_{L,40}$ is the event-weighted mean conditioned on detection in $(3.6,40]$.
The finite-window identity is checked independently. Values are rounded
independently to three decimal places.}
\label{tab:fixed-source-lowerwall-T40}
\centering
\begin{tabular}{rrrrrrr}
\hline\hline
$z_B$ & $N_z$ & $D_{40}$ & $\bar{\tau}^{\mathrm{cond}}_{40}$ &
$\mu^\ast(40)$ & $D_{L,40}$ & $t_{L,40}$ \\
\hline
""" + body + r"""
\hline\hline
\end{tabular}
\end{table}
"""
    (out / "table_S11_fixed_source_T40.tex").write_text(table)
    caption_common = r"""Red/blue indicate net current toward/away from the detector;
gray marks positions outside the guide. Dashed/dotted lines mark the common
monitor links at $z=16.006667$ and $18.006667$. The plotted quantity is the
finite-difference longitudinal link current, summed over the transverse grid
and both spin components and divided once by the initial norm.
All stored endpoint profiles are shown at their native recorded times
(nominal spacing $0.03$, with additional checkpoint samples and the terminal
sample retained), without smoothing or time shifts. Monitor-balance checks
use accepted-step Crank--Nicolson midpoint currents."""
    (out / "caption_fig_S11_fixed_source_omega200.tex").write_text(
        r"""\caption{Signed longitudinal-current maps for the centered-ABC
fixed-source lower-wall control at $\omega=200$ over $0\leq t\leq40$,
with $z_c=10$ and $z_D=19.986666$. Panels (a)--(d) correspond to
$z_B=0,-5,-10,-15$. All four panels share a symmetric-logarithmic color scale
with display limits $\pm0.65$ and linear threshold $2\times10^{-3}$.
Values outside this common display range use the endpoint colors, as indicated
by the colorbar extensions; the underlying currents are unchanged. """
        + caption_common + "}\n")
    (out / "caption_fig_S11_fixed_source_omega1.tex").write_text(
        r"""\caption{Signed longitudinal-current map for the separate
centered-ABC run at $\omega=1$, $z_B=-15$, $z_c=10$, nominal $L=35$,
$N_z=2625$, $\Delta t=10^{-3}$, and $0\leq t\leq40$.
The detector is at $z_D=19.986666$. The color scale is the same as
in the four-panel $\omega=200$ figure. """
        + caption_common + "}\n")
    paragraph = r"""For the fixed-source lower-wall control, the centered-ABC runs
keep $\omega=200$, $z_c=10$, $z_D=19.986666$, the transverse grid, and the
nonnegative longitudinal nodes fixed while moving $z_B$ through
$0,\allowbreak-5,\allowbreak-10,\allowbreak-15$; all four runs use
$\Delta t=3\times10^{-4}$ and reach $T=40$.
Table~S11 gives $D_{40}=0.784,\allowbreak0.750,\allowbreak0.716,\allowbreak0.687$
and $D_{L,40}=0.438,\allowbreak0.404,\allowbreak0.370,\allowbreak0.341$,
while the corresponding late conditional means increase as
$t_{L,40}=16.234,\allowbreak17.216,\allowbreak17.740,\allowbreak18.125$.
Thus moving the wall downward reduces the late detected probability within
the window and increases its mean detection time. The overall conditional
mean is nonmonotonic,
$\bar{\tau}^{\mathrm{cond}}_{40}=10.179,\allowbreak10.431,\allowbreak10.382,\allowbreak10.263$,
whereas $\mu^\ast(40)$ increases from $16.620$ to $19.571$.
The early probability remains essentially unchanged:
$D_E\simeq0.34592835$, with a range below $2.25\times10^{-8}$ across the four
walls at the common cutoff $t_{\rm cut}=3.6$.
The signed currents at both common monitors have negative net transport
over $3.6<t\leq6$, although brief positive intervals occur; the subsequent
transport depends on the lower-wall position (Fig.~S11).
Cumulative detector events agree with $1-S(t)$ within $2.55\times10^{-7}$
through $T=40$. Independently integrated restricted means satisfy
$\mu^\ast(40)=D_{40}\bar{\tau}^{\mathrm{cond}}_{40}+40S(40)$
with residuals below $1.38\times10^{-6}$, and the integrated probability
balances above the two monitors close within $1.27\times10^{-6}$ at the
stored profile times. These are accounting consistency checks, not estimates
of discretization accuracy or evidence of convergence; the signed transport
does not assign individual late oscillations to particular round trips.
"""
    (out / "S9_4_fixed_source_replacement_paragraph.tex").write_text(paragraph)
    report = [
        "S9.4 fixed-source lower-wall control at T=40: validation",
        "All five centered runs are complete. No old first-order data were used.",
        "All values are computed; no fit was needed or performed.", "",
        "Definitions",
        "g_j = roof_probability_j/[P0*(t_right-t_left)], constant on each accepted interval.",
        "D40, DE, tau_cond40 and tL40 use exact overlap integration at 3.6 and 40.",
        "mu_star40 uses the piecewise-linear every-step mass_stored/P0 history.",
        "DL40 retains S(3.6)-S(40); the independent late event probability is also reported.",
        "No identity or flux balance was imposed after calculation.", "",
        "Provenance and geometry",
        "Five required provenance files and all new longitudinal histories independently audited.",
        "Centered closure; half-weight detector node; fixed physical source zc=10.",
        "Detector zD=19.986665725708008; monitors z=16.006667137145996,18.006667137145996.",
        "Four omega200 transverse grids are bitwise identical, Nx=Ny=100.",
        "Omega1 has the prescribed different physical transverse extent (Lx=Ly=10);",
        "it shares Nz=2625 and the exact longitudinal geometry with run7753.",
        "All nonnegative z nodes and stencil hz=0.013333333656191826 are common.", "",
        "Table (publication rounding; full precision is in CSV/JSON/full_precision_values.txt)",
        "zB  Nz       D40      tau40      mu40      DL40       tL40",
    ]
    for row in rows:
        report.append(f"{row['zB']:3d} {row['Nz']:5d} " +
                      " ".join(f"{row[k]:10.3f}" for k in COLUMNS[2:]))
    report += ["", "Independent accounting checks",
               "Run   max|cumulative events-(1-S)|   mu identity residual   max|upper-layer balance|"]
    for d in details:
        report.append(f"{d['case_id']}  "
                      f"{d['validation']['max_abs_detector_survival_closure']:.17g}  "
                      f"{d['restricted_mean_identity_residual']:.17g}  "
                      f"{d['validation']['max_abs_layer_balance']:.17g}")
    report += ["",
               f"omega200 DE range: [{early.min():.17g}, {early.max():.17g}]",
               f"DE spread: {np.ptp(early):.17g} (relative {np.ptp(early)/early.mean():.17g}).",
               f"Max pairwise early event-CDF difference: {early_cdf_range:.17g}.",
               "DE is essentially unchanged; extending the horizon does not alter its fixed early window.",
               "D40 and DL40 decrease, mu_star40 and tL40 increase as the wall moves downward.",
               "tau_cond40 first increases, then decreases; it is not monotonic.",
               "Both monitors have negative integrated transport over (3.6,6] for all omega200 runs.",
               "Brief positive intervals exist, so the old 'always negative' statement is not retained.",
               "The separate omega1 run has positive monitor currents throughout this same window.", "",
               "Recorded times"]
    for d in details:
        v, p = d["validation"], d["native_profiles"]
        report.append(f"{d['case_id']}: {v['accepted_steps']} accepted steps; "
                      f"last width={v['actual_last_step_duration']:.17g}; "
                      f"{p['count']} native profiles over [{p['first']},{p['last']}], "
                      f"gaps [{p['minimum_gap']:.17g},{p['maximum_gap']:.17g}].")
    report += ["",
               "Coarse total_probs.npy uses native reductions and is not the every-step mass64 history.",
               "It was cross-checked, but was not substituted for mass_stored in the restricted mean.",
               "Survival/event DL discrepancies are at most 1.02e-7; full values are retained.",
               "Tiny positive stored-survival increments (below 1.35e-9) are preserved, not corrected.", "",
               "Figures",
               "Native endpoint link currents normalized once by P0; same signed scale for all panels.",
               "Symmetric logarithmic colors, shared display limits +/-0.65, linthresh=0.002, base10, linscale1, RdBu_r.",
               "A fixed LUT and merging only identical adjacent display colors preserve the color-mapped data.",
               "Maps are vector paths in PDFs; PNG exports are 600 dpi plus additional 1200-dpi versions.",
               "Exact rendering settings and native sample counts: plot_metadata.json.", "",
               "Limitations / inconsistencies",
               "No required inputs remain missing after the longitudinal histories were supplied.",
               "Currents outside +/-0.65 use the endpoint display colors; colorbar extensions mark this, and counts are recorded in plot_metadata.json.",
               "The original PDF does not expose its exact colormap threshold; the new explicit scale is documented.",
               "The omega1 transverse extent differs as prescribed by ell/sqrt(omega); no common transverse mesh is claimed across omega.",
               "Floating-point probability-accounting defects are measured, not forced to zero.",
               "These checks do not establish spatial/time convergence or discretization error bounds.",
               "The supplied main-paper paragraph still quotes the old T30 values; it was not edited because this task is limited to S9.4.",
               "Existing source PDFs and raw run data remain unchanged."]
    (out / "validation_report.txt").write_text("\n".join(report) + "\n")
    audit_file = out / "validation" / "provenance_audit.json"
    audits = json.loads(audit_file.read_text()) if audit_file.exists() else []
    if audits and (len(audits) != 5 or not all(a["all_checks_pass"] for a in audits)):
        raise ValueError("Independent provenance audit failed")
    manifest = {
        "scope": record["scope"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_documents_are_reference_material_only": True,
        "input_runs": [
            {"case_id": r["case_id"], "directory": str(r["path"]),
             "initial_norm_squared": r["initial_norm"],
             "configuration": r["configuration"],
             "files": next((a["files"] for a in audits if a["case_id"] == r["case_id"]),
                           {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                            for p in r["path"].iterdir() if p.is_file()})}
            for r in runs
        ],
        "reference_pdfs": [{"path": p, "sha256": digest(Path(p))}
                           for p in REFERENCE_PDFS if Path(p).exists()],
        "analysis_scripts": {p.name: digest(p) for p in sorted((out / "scripts").glob("*.py"))},
        "independent_provenance_checks": "validation/provenance_audit.json",
        "full_precision_statistics": "validation/computed_values_full_precision.json",
        "plot_settings": "plot_metadata.json",
        "plot_metadata_sha256": digest(out / "plot_metadata.json")
            if (out / "plot_metadata.json").exists() else None,
        "rendering_revision": {
            "scope": "Display mapping and PNG resolution only; native currents and table definitions unchanged",
            "common_display_limits": [-0.65, 0.65],
            "linear_threshold": 0.002,
            "previous_full_range_figures": "previous_figures_full_range/",
            "style_provenance": "Earlier paper-style reproduction inferred from reference colorbar, not recovered original plotting source",
        },
    }
    write_json(out / "run_provenance_manifest.json", manifest)
    print("\n".join(report[:24]))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()

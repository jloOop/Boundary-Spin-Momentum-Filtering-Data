#!/usr/bin/env python3
"""Calculate Table S10 directly from the centered-ABC accepted-step histories.

Run: python recalculate_table_S10.py [--input-root ..] [--output-dir .]
Requirements: Python >=3.10, numpy >=2.0, matplotlib >=3.8.
The user approved six omega values and explicitly excluded the (5,5) control.
The vendored source supplies the same strict raw-run validation used for Fig. S6.
No numerical results are read from the existing figure CSVs or the manuscript.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "provenance" / "fig_S6_calculation_source.py"
spec = importlib.util.spec_from_file_location("s6_raw_calculation", SOURCE)
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)
import numpy as np
import matplotlib.pyplot as plt


def render_table(out, fits):
    """A booktabs-style table: three decimal places for manuscript display."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.0, 2.85))
    fig.text(
        0.06, 0.91,
        r"TABLE S10. Unweighted fits of $\mu^*(20;\omega)$ to "
        r"$A_{20}+B_{20}\sqrt{\omega}$.",
        ha="left", va="top", fontsize=11,
    )
    fig.text(
        0.06, 0.79,
        r"Centered spinor ABC; $\kappa=\pi$, $\sigma_z=0.5$, $z_c=10$, "
        r"nominal $L=20$." "\n"
        r"Fits use $\omega=1,25,50,100,150,200$ and correspond to Fig. S6.",
        ha="left", va="top", fontsize=10, linespacing=1.5,
    )
    ax = fig.add_axes([0.20, 0.14, 0.60, 0.44])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = (0.15, 0.48, 0.71, 0.96)
    headings = (r"$(k_z,\kappa)$", r"$A_{20}$", r"$B_{20}$", r"$r_{\rm rms}$")
    for x, heading in zip(xs, headings):
        ax.text(x, 0.85, heading, ha="center" if x == xs[0] else "right", va="center")
    for y, fit in zip((0.61, 0.38, 0.15), fits):
        kz = r"\pi" if fit["kz_label"] == "pi" else fit["kz_label"]
        ax.text(xs[0], y, rf"$({kz},\pi)$", ha="center", va="center")
        for x, key in zip(xs[1:], ("A20", "B20", "r_rms")):
            ax.text(x, y, f"{fit[key]:.3f}", ha="right", va="center")
    for y, width in ((0.99, 0.9), (0.73, 0.5), (0.02, 0.9)):
        ax.plot([0, 1], [y, y], color="black", lw=width, clip_on=False)
    fig.savefig(out / "table_S10_centered_T20.pdf", metadata={
        "Title": "Table S10: centered-ABC restricted-mean fits at T=20",
        "Subject": "Three fixed-kappa=pi rows; six confinement values; control excluded",
        "Creator": "recalculate_table_S10.py",
    })
    fig.savefig(out / "table_S10_centered_T20.png", dpi=600)
    plt.close(fig)


def write_tex(out, fits):
    lines = [
        "% Replacement table fragment; requires booktabs. Manuscript not edited.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\caption{Unweighted fits of \(\mu^*(20;\omega)\) to "
        r"\(A_{20}+B_{20}\sqrt{\omega}\) for the centered spinor ABC at fixed "
        r"\(\kappa=\pi\), \(\sigma_z=0.5\), \(z_c=10\), and nominal \(L=20\). "
        r"The three rows correspond to Fig.~\ref{fig:SM_kz_fixedpi}, using "
        r"\(\omega=1,25,50,100,150,200\). The residual \(r_{\rm rms}\) is the "
        r"root mean square of the six fit residuals.}",
        r"\label{tab:SM_kz_fits}",
        r"\begin{tabular}{@{}c@{\hspace{1.2cm}}rrr@{}}",
        r"\toprule",
        r"$(k_z,\kappa)$ & $A_{20}$ & $B_{20}$ & $r_{\rm rms}$ \\",
        r"\midrule",
    ]
    for fit in fits:
        kz = r"\pi" if fit["kz_label"] == "pi" else fit["kz_label"]
        lines.append(
            rf"$({kz},\pi)$ & {fit['A20']:.3f} & {fit['B20']:.3f}"
            rf" & {fit['r_rms']:.3f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table}", ""]
    (out / "table_S10_centered_T20.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=HERE.parent)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    root, out = args.input_root.resolve(), args.output_dir.resolve()

    # This rereads revision_step_data.csv and all metadata for all 18 runs.
    metrics, fits, manifest = raw.calculate(root)
    out.mkdir(parents=True, exist_ok=True)
    raw.write_csv(out / "table_S10_centered_T20.csv", fits)
    raw.write_csv(out / "table_S10_raw_metrics_T20.csv", metrics)
    raw.write_csv(out / "run_manifest.csv", manifest)
    write_tex(out, fits)
    render_table(out, fits)
    max_identity = max(abs(r["identity_residual"]) for r in metrics)
    report = [
        "TABLE S10: RAW-DATA RECALCULATION, 2026-10-20", "",
        "Scope: three fixed-kappa=pi rows, ordered (2,pi),(pi,pi),(5,pi).",
        "User instruction: exclude the separate (5,5) matching control.",
        "Fit grid: omega=1,25,50,100,150,200; six values previously approved by user.",
        "No omega75 interpolation and no reuse of old manuscript coefficients.", "",
        "All 18 raw runs were reread and passed the same strict metadata,",
        "source-hash, centered closure, coordinate, precision, and completion",
        "checks as Fig. S6. SHA256 hashes and exact paths appear in run_manifest.csv.",
        "P0=revision_run.json['P0_init64']. The input is the every-step mass_stored",
        "column in revision_step_data.csv, normalized once by P0.",
        "mu_star20 is the direct trapezoidal integral of the piecewise-linear",
        "survival through the actual recorded T20 endpoint; no coarse history,",
        "smoothing, clipping, or additional renormalization is used.",
        "D20=sum(roof_probability/P0); tau_cond20 uses recorded interval midpoints.",
        "The per-run signed residual mu_star20-D20*tau_cond20-20*S20 is included",
        "in table_S10_raw_metrics_T20.csv and independently reconciled against",
        "the time-weighted per-step mass/roof defect.",
        f"Maximum absolute identity residual: {max_identity:.12g}.", "",
        "Model: mu_star20=A20+B20*sqrt(omega), unweighted least squares.",
        "r_rms=RMSE=sqrt(sum((mu_star20-A20-B20*sqrt(omega))^2)/6).",
        "Full precision is retained in CSV; the table displays three decimals.",
        "r_rms measures fit disagreement, not discretization error.", "",
        "Nominal L=20; actual detector z=19.986665725708008 at the final grid node.",
        "All Nz=1500; state/assembly complex64, solve complex128.",
        "At omega200, dt=0.00025 for kz=pi and 0.0002 for kz=2,5.",
        "kz=pi uses the T20 prefix of the completed T40 runs.",
        "No convergence claim. No incomplete or inconsistent selected runs.",
        "Fig. S6 and the manuscript were not modified.", "",
        "Replacement prose preceding Fig. S6, to replace the now-obsolete",
        "sentence referring to a fourth Table S10 row:",
        "Table S10 reports the unweighted square-root fits for the three",
        "fixed-kappa=pi series shown in Fig. S6.", "",
    ]
    (out / "validation_report.txt").write_text("\n".join(report))
    (out / "replacement_S9_2_table_sentence.txt").write_text(
        "Table S10 reports the unweighted square-root fits for the three "
        "fixed-kappa=pi series shown in Fig. S6.\n"
    )
    (out / "README.txt").write_text(
        "Table S10: three centered-ABC fit rows, fixed kappa=pi, T=20.\n"
        "The (5,5) control is excluded at the user's instruction.\n\n"
        "Run from any directory: python /path/to/recalculate_table_S10.py\n"
        "Optional flags: --input-root /path/to/Output --output-dir /path/to/results\n"
        "Requirements: Python >=3.10, numpy >=2.0, matplotlib >=3.8.\n"
        "Environment used here: /opt/anaconda3/bin/python.\n"
        "Keep provenance/fig_S6_calculation_source.py beside this script. It is\n"
        "imported for raw data validation and calculation; its figure-writing\n"
        "entry point is never called by the table script.\n\n"
        "table_S10_centered_T20.tex is a manuscript table fragment (booktabs),\n"
        "not a standalone document. Table numbering is supplied by the manuscript.\n"
        "PDF and 600-dpi PNG provide a rendered preview; CSV retains full precision.\n"
        "table_S10_raw_metrics_T20.csv includes all 18 means and identity checks.\n"
        "No manuscript or Fig. S6 files are edited.\n"
    )
    provenance = {
        "scope": "Table S10, three fixed-kappa=pi rows",
        "excluded_control": "(kz,kappa)=(5,5), explicitly excluded by user",
        "omega_values": list(raw.OMEGAS), "horizon": 20, "raw_run_count": 18,
        "input_root": str(root),
        "script_sha256": raw.sha256(Path(__file__)),
        "raw_calculation_source_sha256": raw.sha256(SOURCE),
        "numpy_version": np.__version__,
        "max_abs_identity_residual": max_identity,
        "raw_validation": "PASS",
    }
    (out / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for fit in fits:
        print(f"({fit['kz_label']},pi): A20={fit['A20']:.15g}, "
              f"B20={fit['B20']:.15g}, r_rms={fit['r_rms']:.15g}")
    print(f"Recalculated all 18 raw histories and saved Table S10 in {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot the accepted-step detection density with a late-time inset.

Example (from this run directory):
    /opt/anaconda3/bin/python plot_arrival_time.py
    /opt/anaconda3/bin/python plot_arrival_time.py --nested-inset-start 80

The saved roof events, rather than sampled transport currents or numerical
derivatives of the survival probability, define the detector observable.
Only the initial norm is divided out: the curve is not conditioned on a click.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "arrival_time_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator
import numpy as np


def free_gaussian_current(t, *, z, z0, sigma, k0):
    """Full-line free current, hbar=m=1, psi0 ~ exp[-(z-z0)^2/(2 sigma^2)].

    sigma is the amplitude width: the initial position standard deviation is
    sigma/sqrt(2). The longitudinal state has unit norm on the full real line.
    No reflecting wall, detector, or finite-horizon normalization is included.
    """
    denominator = sigma**4 + t**2
    rho = sigma / np.sqrt(np.pi * denominator) * np.exp(
        -(sigma**2) * (z - z0 - k0 * t) ** 2 / denominator
    )
    velocity = (k0 * sigma**4 + (z - z0) * t) / denominator
    return rho * velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--main-ylim", type=float, nargs=2, default=None,
                        help="Preserve explicit main-axis limits when toggling the comparison.")
    parser.add_argument("--inset-start", type=float, default=8.0)
    parser.add_argument("--inset-end", type=float, default=None,
                        help="Inset end time, independent of the main panel (default: --tmax).")
    parser.add_argument("--nested-inset-start", type=float, default=None,
                        help="Add a second inset inside an enlarged first inset.")
    parser.add_argument("--nested-inset-end", type=float, default=None,
                        help="Second inset end time (default: first inset end time).")
    parser.add_argument("--third-inset-start", type=float, default=None,
                        help="Add a third inset inside the second inset.")
    parser.add_argument("--third-inset-end", type=float, default=None,
                        help="Third inset end time (default: second inset end time).")
    parser.add_argument("--bin-width", type=float, default=0.01)
    parser.add_argument("--show-free-comparison", action="store_true",
                        help="Optionally add the dashed free-Gaussian reference; default is one black detector curve.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "output" / "pdf"
    output_dir.mkdir(parents=True, exist_ok=True)

    run = json.loads((run_dir / "revision_run.json").read_text())
    config = run["configuration"]
    p0 = float(run["P0_init64"])
    # Columns: left/right endpoints, detector event, stored mass, two spin events.
    rows = np.loadtxt(
        run_dir / "revision_step_data.csv", delimiter=",", skiprows=1,
        usecols=(1, 2, 3, 6, 11, 12),
    )
    left, right, event, mass, spin_up, spin_down = rows.T
    dt = right - left
    if not np.all(np.isfinite(rows)) or not np.all(dt > 0) or np.any(event < 0):
        raise ValueError("Detector data contain invalid entries.")
    if not np.allclose(left[1:], right[:-1], atol=1e-12, rtol=0):
        raise ValueError("Accepted-step intervals are not contiguous.")
    if not np.allclose(event, spin_up + spin_down, atol=1e-15, rtol=1e-12):
        raise ValueError("Spin-resolved events do not sum to detector events.")
    inset_end = args.tmax if args.inset_end is None else args.inset_end
    if not (0 < args.tmax <= right[-1] + 1e-10):
        raise ValueError("Require 0 < tmax <= saved horizon.")
    if not (0 <= args.inset_start < inset_end <= right[-1] + 1e-10):
        raise ValueError("Require 0 <= inset-start < inset-end <= saved horizon.")
    has_nested = args.nested_inset_start is not None
    nested_end = inset_end if args.nested_inset_end is None else args.nested_inset_end
    if args.nested_inset_end is not None and not has_nested:
        raise ValueError("--nested-inset-end requires --nested-inset-start.")
    if has_nested and not (args.inset_start <= args.nested_inset_start < nested_end <= inset_end):
        raise ValueError("The nested inset interval must lie within the first inset interval.")
    has_third = args.third_inset_start is not None
    third_end = nested_end if args.third_inset_end is None else args.third_inset_end
    if args.third_inset_end is not None and not has_third:
        raise ValueError("--third-inset-end requires --third-inset-start.")
    if has_third and (not has_nested or not
                      (args.nested_inset_start <= args.third_inset_start < third_end <= nested_end)):
        raise ValueError("The third inset interval must lie within an enabled second inset.")

    probability = event / p0
    native_time = (left + right) / 2
    native_rate = probability / dt
    width_in_steps = max(1, round(args.bin_width / float(np.median(dt))))
    starts = np.arange(0, len(left), width_in_steps)
    ends = np.minimum(starts + width_in_steps, len(left)) - 1
    bin_left, bin_right = left[starts], right[ends]
    bin_probability = np.add.reduceat(probability, starts)
    bin_duration = bin_right - bin_left
    time = (bin_left + bin_right) / 2
    rate = bin_probability / bin_duration
    if not np.isclose(np.sum(rate * bin_duration), probability.sum(), atol=1e-13, rtol=0):
        raise ValueError("Binning did not preserve detected probability.")
    if not np.isclose(event.sum(), run["cumulative_roof_probability"], atol=1e-10, rtol=0):
        raise ValueError("Saved event sum disagrees with the run metadata.")

    # Preserve the existing numerical-plane convention: use the stored float32
    # grid endpoint z_roof_actual, not nominal Lz or a reconstructed ideal grid.
    free_rate = free_gaussian_current(
        time, z=config["z_roof_actual"], z0=config["z0_init"],
        sigma=config["sigma_z"], k0=config["k0"],
    )
    visible = (time >= 0) & (time <= args.tmax)
    tail = (time >= args.inset_start) & (time <= inset_end)
    omega = float(config["omega"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "mathtext.fontset": "dejavusans", "axes.labelsize": 13,
        "axes.linewidth": 0.65, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "path.simplify": False,
    })
    figure_size = (12.0, 7.4) if has_nested else (10.5, 6.0)
    fig, ax = plt.subplots(figsize=figure_size)
    fig.subplots_adjust(left=0.105, right=0.977, bottom=0.13, top=0.97)
    # The latest screenshot request specifies a single black detector curve.
    color = "#000000"
    line_args = {"color": color, "linewidth": 1.1}
    free_line_args = {"color": "#505050", "linewidth": 1.6,
                      "linestyle": (0, (5, 3))}
    ax.plot(time[visible], rate[visible], label=rf"$\omega={omega:g}$", **line_args)
    if args.show_free_comparison:
        ax.plot(time[visible], free_rate[visible], label="Detector-free Gaussian current", **free_line_args)
    main_peak = float(rate[visible].max())
    if args.show_free_comparison:
        main_peak = max(main_peak, float(free_rate[visible].max()))
    main_scale = np.ceil(main_peak / 0.05) * 0.05
    ax.set(xlim=(0, args.tmax), ylim=args.main_ylim or (-0.05 * main_scale, 1.06 * main_scale), xlabel=r"$t$",
           ylabel="Probability density")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10,
              handlelength=2.8, columnspacing=2.2, borderaxespad=0.8)

    inset_bounds = [0.17, 0.18, 0.79, 0.63] if has_nested else [0.30, 0.32, 0.665, 0.46]
    inset = ax.inset_axes(inset_bounds)
    inset.plot(time[tail], rate[tail], **line_args)
    if args.show_free_comparison:
        inset.plot(time[tail], free_rate[tail], **free_line_args)
    tail_top = float(rate[tail].max())
    if args.show_free_comparison:
        tail_top = max(tail_top, float(free_rate[tail].max()))
    inset.set(xlim=(args.inset_start, inset_end), ylim=(0, tail_top * 1.12),
              xlabel=r"$t$", ylabel="Probability density")
    inset.set_title(rf"Late-time detail: ${args.inset_start:g}\leq t\leq {inset_end:g}$",
                    fontsize=11, pad=7, loc="left")
    inset.tick_params(labelsize=9)
    inset.xaxis.label.set_size(11)
    inset.yaxis.label.set_size(11)
    inset.xaxis.set_major_locator(MaxNLocator(nbins=5))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if tail_top < 0.02 else "%.2f"))
    inset.xaxis.set_minor_locator(AutoMinorLocator(5))
    inset.yaxis.set_minor_locator(AutoMinorLocator(2))

    nested_bounds = None
    third_bounds = None
    if has_nested:
        nested_mask = (time >= args.nested_inset_start) & (time <= nested_end)
        # Axes coordinates are relative to the FIRST inset, so this is a true
        # inset-within-an-inset. The upper-right area is above the late signal.
        nested_bounds = [0.42, 0.43, 0.55, 0.47]
        nested = inset.inset_axes(nested_bounds)
        nested.plot(time[nested_mask], rate[nested_mask], **line_args)
        if args.show_free_comparison:
            nested.plot(time[nested_mask], free_rate[nested_mask], **free_line_args)
        nested_top = float(rate[nested_mask].max())
        if args.show_free_comparison:
            nested_top = max(nested_top, float(free_rate[nested_mask].max()))
        nested.set(xlim=(args.nested_inset_start, nested_end),
                   ylim=(0, nested_top * 1.15), xlabel=r"$t$", ylabel="Probability density")
        nested.set_title(rf"${args.nested_inset_start:g}\leq t\leq {nested_end:g}$",
                         fontsize=10.5, pad=7, loc="center")
        nested.tick_params(labelsize=9)
        nested.xaxis.label.set_size(10.5)
        nested.yaxis.label.set_size(10.5)
        nested.set_xticks(np.linspace(args.nested_inset_start, nested_end, 4))
        nested.yaxis.set_major_locator(MaxNLocator(nbins=4))
        nested.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        nested.yaxis.get_offset_text().set_fontsize(9)
        nested.xaxis.set_minor_locator(AutoMinorLocator(5))
        nested.yaxis.set_minor_locator(AutoMinorLocator(2))

        if has_third:
            third_mask = (time >= args.third_inset_start) & (time <= third_end)
            third_bounds = [0.43, 0.47, 0.54, 0.42]
            third = nested.inset_axes(third_bounds)
            third.plot(time[third_mask], rate[third_mask], **line_args)
            third_top = float(rate[third_mask].max())
            if args.show_free_comparison:
                third.plot(time[third_mask], free_rate[third_mask], **free_line_args)
                third_top = max(third_top, float(free_rate[third_mask].max()))
            third.set(xlim=(args.third_inset_start, third_end),
                      ylim=(0, third_top * 1.15), xlabel=r"$t$", ylabel=r"$g(t)$")
            third.set_title(rf"${args.third_inset_start:g}\leq t\leq {third_end:g}$",
                            fontsize=8, pad=4)
            third.tick_params(labelsize=7, length=3)
            third.tick_params(which="minor", length=1.5)
            third.xaxis.label.set_size(8)
            third.yaxis.label.set_size(8)
            third.xaxis.labelpad = 1
            third.yaxis.labelpad = 1
            third.set_xticks(np.linspace(args.third_inset_start, third_end, 3))
            third.yaxis.set_major_locator(MaxNLocator(nbins=3))
            third.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            third.yaxis.get_offset_text().set_fontsize(7)
            third.xaxis.set_minor_locator(AutoMinorLocator(4))
            third.yaxis.set_minor_locator(AutoMinorLocator(2))

    stem = f"arrival_time_omega{omega:g}_T{args.tmax:g}_inset"
    if has_third:
        stem += "_triple"
    elif has_nested:
        stem += "_nested"
    if not args.show_free_comparison:
        stem += "_black"
    else:
        stem += "_black_with_free_gaussian"
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"{stem}.{extension}", facecolor="white", **kwargs)
    plt.close(fig)
    np.savetxt(
        output_dir / f"{stem}_data.csv",
        np.column_stack((bin_left, bin_right, time, rate, free_rate, bin_probability)),
        delimiter=",", fmt="%.12g", comments="",
        header="t_left,t_right,t_mid,g_detector,g_free_gaussian,detected_probability",
    )

    def detected_by(horizon):
        fraction = np.clip((horizon - left) / dt, 0, 1)
        return float(np.sum(probability * fraction))

    displayed_windows = [("Main plot", 0.0, args.tmax),
                         ("First inset", args.inset_start, inset_end)]
    if has_nested:
        displayed_windows.append(("Nested inset", args.nested_inset_start, nested_end))
    if has_third:
        displayed_windows.append(("Third inset", args.third_inset_start, third_end))
    detection_windows = []
    for name, start_time, end_time in displayed_windows:
        window_probability = detected_by(end_time) - detected_by(start_time)
        detection_windows.append({"panel": name, "time_start": start_time,
                                  "time_end": end_time, "probability": window_probability,
                                  "percent": 100 * window_probability})
    import csv
    with (output_dir / f"{stem}_detection_windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detection_windows[0]))
        writer.writeheader()
        writer.writerows(detection_windows)

    peak = int(np.argmax(native_rate))
    budget_residual = 1 - mass / p0 - np.cumsum(probability)
    result = {
        "source_csv": str(run_dir / "revision_step_data.csv"),
        "source_metadata": str(run_dir / "revision_run.json"),
        "density_definition": "roof_probability / (t_right - t_left) / P0_init64",
        "normalization": "Unconditioned; divide once by initial norm, not by detected fraction.",
        "time_units": "Simulation natural units (hbar = m = 1); no SI conversion applied.",
        "gaussian_definition": "psi0(z) proportional to exp[-(z-z0)^2/(2*sigma_z^2) + i*k0*(z-z0)]; free full-line current evaluated at actual detector coordinate.",
        "comparison_caveat": "Comparison flux, not a detector-click probability density: free full-line evolution with no detector or reflecting lower boundary, not exact finite-guide evolution.",
        "free_comparison_plane_convention": "Stored numerical detecting plane, preserving the original plotting code's z_roof_actual convention.",
        "free_comparison_z_plane": float(config["z_roof_actual"]),
        "free_comparison_plotted": args.show_free_comparison,
        "configuration": {k: config[k] for k in (
            "omega", "Lz", "z_roof_actual", "k_bc", "k0", "sigma_z", "z0_init", "theta", "phi",
        )},
        "plot_xlim": [0, args.tmax],
        "plot_ylim": list(ax.get_ylim()),
        "inset_xlim": [args.inset_start, inset_end],
        "nested_inset_xlim": [args.nested_inset_start, nested_end] if has_nested else None,
        "third_inset_xlim": [args.third_inset_start, third_end] if has_third else None,
        "figure_size_inches": list(figure_size),
        "inset_bounds_in_main_axes": inset_bounds,
        "nested_inset_bounds_in_first_inset": nested_bounds,
        "third_inset_bounds_in_second_inset": third_bounds,
        "displayed_window_detection_probabilities": detection_windows,
        "window_probability_note": "Unconditional detector probability within each interval. The displayed windows overlap and must not be summed.",
        "figure_style": {
            "reference_image": "EXTERNAL/Screenshot 2026-10-27 at 20.01.20.png",
            "reference_figure": "User screenshot, with a single black detector curve as requested",
            "detector_color": color, "detector_linewidth_pt": line_args["linewidth"],
            "free_current_color": free_line_args["color"],
            "free_current_linewidth_pt": free_line_args["linewidth"],
            "axes_linewidth_pt": 0.65, "tick_direction": "out",
            "tick_width_pt": 0.7, "major_tick_length_pt": 4, "minor_tick_length_pt": 2,
            "font_note": "DejaVu Sans, matching the screenshot's Matplotlib styling.",
        },
        "P0_init64": p0,
        "native_steps": int(len(left)),
        "plot_bins": int(len(time)),
        "nominal_bin_width": float(np.median(bin_duration)),
        "binning": "Sum detector events in consecutive bins and divide by bin duration; no fitted smoothing.",
        "native_peak_time": float(native_time[peak]),
        "native_peak_density": float(native_rate[peak]),
        "binned_peak_density": float(rate.max()),
        "D_40": detected_by(40),
        "D_80": detected_by(80),
        "D_100": detected_by(100),
        "D_200": detected_by(200),
        "D_plot_horizon": detected_by(args.tmax),
        "D_saved_horizon": float(probability.sum()),
        "final_survival": float(mass[-1] / p0),
        "max_abs_probability_budget_residual": float(np.max(np.abs(budget_residual))),
        "binning_integral_error": float(np.sum(rate * bin_duration) - probability.sum()),
        "run_status": run["status"],
    }
    (output_dir / f"{stem}_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved PNG, PDF, SVG, CSV, and metadata in {output_dir}")


if __name__ == "__main__":
    main()

"""CPU control checks and explicitly labeled archived-data display routes.

This module never imports generating GPU sources. Channel histories store already
launch-normalized g_top/g_lower. Large state/spatial-history files are unnecessary
for the advertised native two-boundary reductions; other control tables remain
archived derived evidence rather than independent reconstruction.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from centered.numerics import events, linear_integral

__all__ = ["run"]

# Numerical recomputation tolerances are separate from the measured native
# probability-accounting defect. No tolerance is fitted to rounded paper targets.
ALGEBRA_ATOL = 1e-9
CHANNEL_CLOSURE_ATOL = 1e-6
DERIVED_BIN_ATOL = 5e-10

def _rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def _write_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def _add(checks, name, actual, tolerance):
    actual = float(actual)
    checks.append({"name": name, "passed": bool(np.isfinite(actual) and abs(actual) <= tolerance),
                   "actual": actual, "tolerance": tolerance})

def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}

def _channels(root, output, checks):
    base = root / "data/centered/native/controls"
    rows, curves = [], {}
    for case in ("7800", "7801"):
        data = _load(base / case / "two_channel_history.npz")
        config = json.loads((base / case / "run.json").read_text())
        required = {"time", "D_top", "E_lower", "S", "event_t_left", "event_t_right",
                    "g_top", "g_lower", "P0_init64", "mu_top_capped", "mu_exit", "M1_top", "M1_lower"}
        if not required.issubset(data):
            raise ValueError(f"{case}: missing native channel keys {required-set(data)}")
        left, right, time = data["event_t_left"], data["event_t_right"], data["time"]
        width = right-left
        if not (len(time) == len(left)+1 and np.array_equal(time[1:], right) and
                np.array_equal(time[:-1], left) and np.all(width > 0)):
            raise ValueError(f"{case}: inconsistent cumulative/event time support")
        if not (float(data["P0_init64"]) > 0 and np.isfinite(data["P0_init64"])):
            raise ValueError("Invalid launch norm")
        _add(checks, f"{case}: launch norm agrees with source configuration",
             float(data["P0_init64"])-config["P0_init64"], 1e-14)
        _add(checks, f"{case}: actual support ends at40", right[-1]-40, 1e-10)
        for channel, ratekey, cumulative in (("upper", "g_top", "D_top"), ("lower", "g_lower", "E_lower")):
            rate = data[ratekey]
            if rate.shape != left.shape or not np.all(np.isfinite(rate)):
                raise ValueError(f"{case}: invalid {ratekey}")
            reconstructed = np.r_[0., np.cumsum(rate*width)]
            _add(checks, f"{case}: {channel} cumulative event reconstruction",
                 np.max(np.abs(reconstructed-data[cumulative])), ALGEBRA_ATOL)
        top = events(left, right, data["g_top"]*width, 0, 40)
        lower = events(left, right, data["g_lower"]*width, 0, 40)
        late = events(left, right, data["g_top"]*width, 3.6, 40)
        early = events(left, right, data["g_top"]*width, 0, 3.6)
        top_cap = top["first_moment"]+40*(1-top["probability"])
        no_top_integral = linear_integral(time, 1-data["D_top"], 0, 40)
        mu_exit = linear_integral(time, data["S"], 0, 40)
        _add(checks, f"{case}: upper first moment",
             top["first_moment"]-data["M1_top"][-1], ALGEBRA_ATOL)
        _add(checks, f"{case}: lower first moment",
             lower["first_moment"]-data["M1_lower"][-1], ALGEBRA_ATOL)
        _add(checks, f"{case}: upper capped mean event/1-D identity",
             top_cap-no_top_integral, ALGEBRA_ATOL)
        _add(checks, f"{case}: archived upper capped mean",
             top_cap-data["mu_top_capped"][-1], ALGEBRA_ATOL)
        _add(checks, f"{case}: archived exit restricted mean",
             mu_exit-data["mu_exit"][-1], ALGEBRA_ATOL)
        budget = data["S"]+data["D_top"]+data["E_lower"]-1
        _add(checks, f"{case}: native three-channel probability accounting",
             np.max(np.abs(budget)), CHANNEL_CLOSURE_ATOL)
        # Finite numerical accounting remains visible; it is not converted to events.
        rows.append({"case": case, "omega": config["configuration"]["omega"],
                     "D_top": top["probability"], "E_lower": lower["probability"],
                     "S_in": float(data["S"][-1]), "P_no_upper_state_channels": float(data["S"][-1]+data["E_lower"][-1]),
                     "P_no_upper_from_events": 1-top["probability"],
                     "conditional_top_mean": top["conditional_mean"], "D_early_0_3p6": early["probability"],
                     "early_conditional_mean": early["conditional_mean"], "D_late_3p6_40": late["probability"],
                     "mu_top_capped": top_cap, "mu_exit": mu_exit,
                     "integral_S_plus_E": linear_integral(time, data["S"]+data["E_lower"], 0, 40),
                     "budget_residual_max": float(np.max(np.abs(budget)))})
        curves[case] = (left, right, data["g_top"])
    _write_rows(output / "two_channel_statistics.csv", rows)

    # Matched one-boundary comparison uses main case5045's event record, not 1-S.
    path = root / "data/centered/native/5045"
    main = _load(path / "steps.npz")
    run = json.loads((path / "run.json").read_text())
    p0 = run["P0_init64"]
    left, right, p = (main[k] for k in ("t_left", "t_right", "roof_probability"))
    reference = events(left, right, p, 0, 40, p0)
    reference_late = events(left, right, p, 3.6, 40, p0)
    comparison = [{"case": "5045", "lower_boundary": "Dirichlet", "D40_upper": reference["probability"],
                   "D_late_3p6_40_upper": reference_late["probability"]},
                  {"case": "7800", "lower_boundary": "scalar ABC", "D40_upper": rows[0]["D_top"],
                   "D_late_3p6_40_upper": rows[0]["D_late_3p6_40"]}]
    _write_rows(output / "lower_boundary_comparison.csv", comparison)
    metadata = json.loads((base / "S12_export.json").read_text())
    for row, archived in zip(comparison, metadata["curves"]):
        _add(checks, f"S12:{row['case']} event probability vs identified export",
             row["D40_upper"]-archived["top_probability_0_40"], ALGEBRA_ATOL)
    curves["5045"] = (left, right, p/(right-left)/p0)
    _plot_lower(curves, metadata, output)
    return rows, comparison

def _plot_lower(curves, metadata, output):
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "path.simplify": False}):
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        fig.subplots_adjust(left=.10, right=.98, bottom=.13, top=.97)
        inset = ax.inset_axes([.30, .33, .66, .46])
        for case, entry in zip(("5045", "7800"), metadata["curves"]):
            left, right, rate = curves[case]
            t = (left+right)/2
            ax.plot(t, rate, color=entry["color"], lw=1.1, label=entry["label"])
            inset.plot(t, rate, color=entry["color"], lw=1.1)
        ax.set(xlim=metadata["main_limits"]["x"], ylim=metadata["main_limits"]["y"],
               xlabel=r"$t$", ylabel="Probability density")
        inset.set(xlim=metadata["inset_limits"]["x"], ylim=metadata["inset_limits"]["y"],
                  xlabel=r"$t$", ylabel="Probability density")
        ax.legend(frameon=False, loc="upper right", fontsize=10)
        fig.savefig(output / "FigS12_native_replot.png", dpi=160)
        plt.close(fig)

def _long(root, output, checks):
    base = root / "data/centered/native/controls/7810"
    data = np.genfromtxt(base / "plot_data.csv", names=True, delimiter=",")
    metadata = json.loads((base / "plot_metadata.json").read_text())
    config = json.loads((base / "run.json").read_text())
    times = np.load(base / "prob_times.npy", allow_pickle=False)
    mass = np.load(base / "total_probs.npy", allow_pickle=False)
    if times.shape != mass.shape or not np.all(np.diff(times) > 0) or not np.all(np.isfinite(mass)):
        raise ValueError("Invalid long-time sampled norm support")
    left, right = data["t_left"], data["t_right"]
    _add(checks, "7810: saved norm support200", times[-1]-200, 1e-10)
    _add(checks, "7810: plotted support200", right[-1]-200, 1e-10)
    _add(checks, "7810: plotted bin widths", np.max(np.abs((right-left)-.01)), 5e-11)
    _add(checks, "7810: derived bin probability/rate agreement",
         np.max(np.abs(data["detected_probability"]-data["g_detector"]*(right-left))), DERIVED_BIN_ATOL)
    values = []
    for horizon in (40, 80, 100, 200):
        result = events(left, right, data["detected_probability"], 0, horizon)
        s = float(np.interp(horizon, times, mass)/config["P0_init64"])
        _add(checks, f"7810: derived D{horizon} vs archived metadata",
             result["probability"]-metadata[f"D_{horizon}"], DERIVED_BIN_ATOL)
        values.append({"horizon": horizon, "derived_bin_D": result["probability"], "sampled_survival": s,
                       "derived_D_plus_sampled_S_minus_one": result["probability"]+s-1,
                       "source_level": "existing binned events and sampled norms; not full-step native reconstruction"})
    _write_rows(output / "long_time_support.csv", values)
    _plot_long(data, metadata, output)
    return {"rows": values, "manuscript_logged_estimate_endpoint": 189.95,
            "available_plot_and_norm_endpoint": 200,
            "unrestricted_mean_status": "not determined; do not substitute the later200 endpoint for manuscript estimate",
            "saved_norm_cadence": float(np.median(np.diff(times))),
            "archived_plot_bins": len(data)}

def _plot_long(data, metadata, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 13, "axes.linewidth": .65,
                         "xtick.direction": "out", "ytick.direction": "out", "path.simplify": False}):
        fig, ax = plt.subplots(figsize=metadata["figure_size_inches"])
        fig.subplots_adjust(left=.105, right=.977, bottom=.13, top=.97)
        t, rate = data["t_mid"], data["g_detector"]
        ax.plot(t, rate, color="#000000", linewidth=1.1, label=r"$\omega=200$")
        ax.set(xlim=metadata["plot_xlim"], ylim=metadata["plot_ylim"], xlabel=r"$t$", ylabel="Probability density")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        inset = ax.inset_axes(metadata["inset_bounds_in_main_axes"])
        mask = (t >= 8) & (t <= 200)
        inset.plot(t[mask], rate[mask], color="#000000", linewidth=1.1)
        inset.set(xlim=metadata["inset_xlim"], ylim=(0, float(rate[mask].max())*1.12),
                  xlabel=r"$t$", ylabel="Probability density")
        inset.set_title(r"Late-time detail: $8\leq t\leq200$", fontsize=11, pad=7, loc="left")
        inset.xaxis.set_major_locator(MaxNLocator(nbins=5))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
        inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        inset.tick_params(labelsize=9)
        inset.xaxis.label.set_size(11); inset.yaxis.label.set_size(11)
        nested = inset.inset_axes(metadata["nested_inset_bounds_in_first_inset"])
        mask = (t >= 80) & (t <= 200)
        nested.plot(t[mask], rate[mask], color="#000000", linewidth=1.1)
        nested.set(xlim=metadata["nested_inset_xlim"], ylim=(0, float(rate[mask].max())*1.15),
                   xlabel=r"$t$", ylabel="Probability density")
        nested.set_title(r"$80\leq t\leq200$", fontsize=10.5, pad=7)
        nested.set_xticks(np.linspace(80, 200, 4))
        nested.yaxis.set_major_locator(MaxNLocator(nbins=4))
        nested.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        nested.tick_params(labelsize=9)
        nested.xaxis.label.set_size(10.5); nested.yaxis.label.set_size(10.5)
        for a in (ax, inset, nested):
            a.xaxis.set_minor_locator(AutoMinorLocator(5))
            a.yaxis.set_minor_locator(AutoMinorLocator(2))
        fig.savefig(output / "FigS13_derived_replot.png", dpi=150)
        plt.close(fig)

def run(root: Path, output: Path):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    channels, comparison = _channels(root, output, checks)
    long = _long(root, output, checks)
    archived = output / "archived"
    archived.mkdir(exist_ok=True)
    for item in ("S4", "S6", "S8", "S10", "S11"):
        shutil.copyfile(root / f"data/centered/figures/Fig{item}.png", archived / f"Fig{item}_archived.png")
    tables = root / "data/centered/tables/controls"
    displayed_tables = []
    for folder in ("T6", "T10", "S11", "T12", "S4", "S6", "S8", "S10"):
        destination = archived / "tables" / folder
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted((tables / folder).glob("*.csv")):
            shutil.copyfile(source, destination / source.name)
            displayed_tables.append(str(source.relative_to(root)))
    unavailable = {
        "Fig. S2 / Tables S7-S8": "manuscript-reported layers; numerical inputs not bundled",
        "Fig. S3 / Fig. S5 / Table S9": "legacy one-sided comparisons, source/preparation qualifications and absent native inputs",
        "Fig. S7": "original one-sided width family not located; incomplete centered runs not substituted",
        "Fig. S9": "caption/source-method conflict; no exact reproduction claim",
        "Table S10 original reducer": "provenance/fig_S6_calculation_source.py missing; archived derived table display only",
        "Fig. S11 spatial-map regeneration": "large native longitudinal histories not bundled; complete archived image displayed",
        "Bulk/chiral and parameter native reconstruction": "archived tables displayed; complete every-step native runs not bundled",
    }
    return {"checks": checks, "two_channel_statistics": channels, "lower_boundary_comparison": comparison,
            "long_time": long, "archived_table_display": displayed_tables, "unavailable": unavailable,
            "reproduction_levels": {
                "FigS12/TableS12": "native channel observable reconstruction; native-data replot",
                "FigS13": "existing derived-bin replot plus sampled-norm checks",
                "FigsS4/S6/S8/S10/S11 and TablesS6/S10/S11": "archived export/table display"},
            "interpretation": "Accounting defects are retained, not reclassified as detector events. These checks are not new simulation convergence."}

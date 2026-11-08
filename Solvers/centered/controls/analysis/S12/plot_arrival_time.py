#!/usr/bin/env python3
"""Compare upper-detector arrival densities for matched Omega=200 runs.

Run with Python, NumPy and Matplotlib. Outputs are saved beside this script.
The density is the accepted-step top event probability divided by the step
duration and initial launch mass, evaluated at the interval midpoint. Neither
curve is conditioned on detection, smoothed, or reconstructed from norm loss.

Palette: Paul Tol, muted (green and indigo),
https://sronpersonalpages.nl/~pault/#sec:qualitative
Figure styling and axis names follow Fig. S9 (p. 37) of the supplied paper.
The main plot and inset deliberately have no titles.
"""

from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "arrival-time-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
STEM = "arrival_time_lower_boundary_comparison"
CURVES = (
    ("5045.", "Dirichlet lower wall", "#117733", "-"),
    ("7800.", "Scalar absorbing lower boundary", "#332288", "-"),
)


def load_curve(prefix):
    matches = sorted(p for p in ROOT.glob(prefix + "*") if p.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one run matching {prefix!r}, found {len(matches)}")
    run = matches[0]
    metadata = json.loads((run / "revision_run.json").read_text())
    events = np.genfromtxt(
        run / "revision_step_data.csv", delimiter=",", names=True,
        usecols=("t_left", "t_right", "roof_probability"),
    )
    dt = events["t_right"] - events["t_left"]
    p0 = float(metadata["P0_init64"])
    assert p0 > 0 and np.all(dt > 0)
    assert np.allclose(events["t_right"][:-1], events["t_left"][1:], rtol=0, atol=1e-12)
    assert np.isclose(events["t_left"][0], 0) and np.isclose(events["t_right"][-1], 40)
    t = (events["t_left"] + events["t_right"]) / 2
    density = events["roof_probability"] / (p0 * dt)
    assert np.all(np.isfinite(density)) and np.all(density >= 0)
    integrated = float(np.sum(density * dt))
    expected = float(metadata["cumulative_roof_probability"]) / p0
    assert np.isclose(integrated, expected, rtol=1e-12, atol=1e-14)
    return t, density, metadata, {
        "source_directory": run.name,
        "source_file": "revision_step_data.csv",
        "initial_mass": p0,
        "samples": len(t),
        "top_probability_0_40": integrated,
        "peak_time": float(t[np.argmax(density)]),
        "peak_density": float(np.max(density)),
        "max_density_5_35": float(np.max(density[(t >= 5) & (t <= 35)])),
    }


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    loaded = [load_curve(prefix) for prefix, *_ in CURVES]
    t0, _, meta0, _ = loaded[0]
    t1, _, meta1, _ = loaded[1]
    assert np.array_equal(t0, t1)
    for key in ("omega", "Lz", "k0", "k_bc", "sigma_z", "dt", "Nx", "Ny", "Nz"):
        assert meta0["configuration"][key] == meta1["configuration"][key], key
    assert meta0["configuration"]["omega"] == 200

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.13, top=0.97)
    inset = ax.inset_axes([0.36, 0.30, 0.61, 0.42])

    for (_, label, color, linestyle), (t, density, _, _) in zip(CURVES, loaded):
        for axis in (ax, inset):
            axis.plot(t, density, color=color, linestyle=linestyle,
                      linewidth=1.2 if color == "#117733" else 1.5,
                      label=label, solid_capstyle="round")

    ax.set(xlim=(0, 40), ylim=(-0.0125, 0.25),
           xlabel=r"$t$", ylabel="Probability density")
    ax.set_xticks(np.arange(0, 41, 5))
    ax.set_yticks(np.arange(0, 0.251, 0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=False, right=False, which="both")
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.985),
              frameon=False, fontsize=10.5, handlelength=2.8, labelspacing=0.55)

    inset.set(xlim=(5, 35), ylim=(0, 0.032))
    inset.set_xticks([5, 10, 15, 20, 25, 30, 35])
    inset.set_yticks([0, 0.01, 0.02, 0.03])
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.xaxis.set_minor_locator(AutoMinorLocator(5))
    inset.yaxis.set_minor_locator(AutoMinorLocator(2))
    inset.tick_params(labelsize=9, top=False, right=False, which="both", pad=3)
    inset.set_xlabel(r"$t$", fontsize=10, labelpad=1)
    inset.set_ylabel("Probability density", fontsize=10, labelpad=4)

    for extension in ("pdf", "svg", "png"):
        output = ROOT / f"{STEM}.{extension}"
        fig.savefig(output, dpi=600)
        print(output)
    plt.close(fig)

    provenance = {
        "observable": "Unconditional upper-detector arrival-time density",
        "formula": "roof_probability / (P0_init64 * (t_right - t_left))",
        "time": "(t_left + t_right) / 2",
        "conditioning_on_top_detection": False,
        "smoothing": False,
        "palette_source": "https://sronpersonalpages.nl/~pault/#sec:qualitative",
        "style_reference": "Supplied paper, Fig. S9, page 37",
        "axis_labels": {"x": "t", "y": "Probability density"},
        "titles": False,
        "main_limits": {"x": [0, 40], "y": [-0.0125, 0.25]},
        "inset_limits": {"x": [5, 35], "y": [0, 0.032]},
        "curves": [
            dict(label=label, color=color, **loaded_curve[3])
            for (_, label, color, _), loaded_curve in zip(CURVES, loaded)
        ],
    }
    (ROOT / f"{STEM}.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

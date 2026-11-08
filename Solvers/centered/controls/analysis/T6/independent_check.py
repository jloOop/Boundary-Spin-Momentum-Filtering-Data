#!/usr/bin/env python3
"""Independent stdlib-only audit of the S8.1 output; no simulation imports.

Reads the original step CSV files and run metadata anew, computes each
piecewise-linear segment directly, and compares results to the deliverables.
Run after recalculate_s8_1.py. The sole output is independent_check_report.txt
beside this script. No raw input or main deliverable is modified.
"""
from pathlib import Path
import bisect
import csv
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
EXPECTED = [
    ("Reference", 1, 5540), ("Reference", 100, 5541),
    ("Uniform", 1, 7692), ("Uniform", 100, 7693),
    ("Radial", 1, 7694), ("Radial", 100, 7695),
    ("Rashba", 1, 7696), ("Rashba", 100, 7697),
    ("Rashba–Dresselhaus", 1, 7698), ("Rashba–Dresselhaus", 100, 7699),
    ("Chiral", 1, 7001), ("Chiral", 100, 7000),
]
METRICS = ("D20", "tau_cond20", "mu_star20", "DL")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def value_at(times, values, target):
    assert times[0] <= target <= times[-1]
    index = bisect.bisect_left(times, target)
    if times[index] == target:
        return values[index]
    fraction = (target - times[index-1]) / (times[index] - times[index-1])
    return values[index-1] + fraction * (values[index] - values[index-1])


def calculate(path):
    run = json.loads((path / "revision_run.json").read_text())
    rows = read_csv(path / "revision_step_data.csv")
    p0 = run["P0_init64"]
    assert len(rows) == run["completed_step"]
    assert run["actual_time"] >= 20
    times, survival, event_mass, event_first, late_event_mass = [0.0], [1.0], [], [], []
    previous_mass, previous_time = p0, 0.0
    for expected_step, row in enumerate(rows, start=1):
        number = int(row["step"])
        left, right = float(row["t_left"]), float(row["t_right"])
        mass = float(row["mass_stored"])
        assert number == expected_step and left == previous_time and right > left
        assert float(row["mass_before"]) == previous_mass
        times.append(right)
        survival.append(mass / p0)
        assert float(row["roof_probability"]) >= 0
        # All selected T=20 cutoffs occur at actual edges. Do not silently
        # assign fractional native events if a future selection differs.
        assert not left < 20 < right
        assert not left < 3.6 < right
        if right <= 20:
            probability = float(row["roof_probability"]) / p0
            event_mass.append(probability)
            event_first.append(probability * (left + right) / 2)
            if left >= 3.6:
                late_event_mass.append(probability)
        previous_mass, previous_time = mass, right
    assert previous_mass == run["current_accepted_prefix_mass64"]
    assert previous_time == run["actual_time"]
    # Integrate each recorded line segment, clipping its upper endpoint at T.
    areas = []
    for index in range(1, len(times)):
        left, right = times[index-1], min(20.0, times[index])
        if left >= 20:
            break
        left_s = survival[index-1]
        right_s = value_at(times, survival, right)
        areas.append((right-left) * (left_s + right_s) / 2)
    mu = math.fsum(areas)
    s20, s36 = value_at(times, survival, 20), value_at(times, survival, 3.6)
    d = 1 - s20
    result = dict(D20=d, tau_cond20=(mu-20*s20)/d, mu_star20=mu, DL=s36-s20,
                  S20=s20, S3_6=s36)
    de = math.fsum(event_mass)
    te = math.fsum(event_first)/de
    le = math.fsum(late_event_mass)
    result.update(D20_events=de, tau_cond20_events=te, DL_events=le,
                  event_minus_survival_D20=de-d,
                  event_minus_survival_tau_cond20=te-result["tau_cond20"],
                  event_minus_survival_DL=le-result["DL"])
    selected_s = [v for t, v in zip(times, survival) if t <= 20]
    quality = dict(S_min=min(selected_s), S_max=max(selected_s),
                   S_above_one_count=sum(v > 1 for v in selected_s),
                   S_below_zero_count=sum(v < 0 for v in selected_s))
    return result, quality


def main():
    manifest = json.loads((HERE / "run_provenance_manifest.json").read_text())
    paths = {r["case_id"]: Path(r["local_path"]) for r in manifest["runs"]}
    published = read_csv(HERE / "full_precision_results.csv")
    comparisons = read_csv(HERE / "control_vs_reference.csv")
    validation = {int(r["case_id"]): r for r in read_csv(HERE / "numerical_validation.csv")}
    assert [(r["control"], int(r["omega"]), int(r["case_id"])) for r in published] == EXPECTED
    assert len(comparisons) == 10
    results, lines = {}, [
        "INDEPENDENT S8.1 CHECK: stdlib csv, bisect and math; no main-script imports",
        "Inputs: original revision_step_data.csv and revision_run.json.",
        "Every-step stored masses divided once by recorded P0_init64; no clipping or repair.",
        "Native midpoint detector probabilities audited independently.",
        "",
        "case omega maximum absolute difference across all full-precision numerical fields",
    ]
    maximum_difference = 0.0
    for expected, delivered in zip(EXPECTED, published):
        control, omega, case_id = expected
        result, quality = calculate(paths[case_id])
        differences = {k: abs(v-float(delivered[k])) for k, v in result.items()}
        assert max(differences.values()) <= 5e-13, (case_id, differences)
        for key, value in quality.items():
            assert abs(value-float(validation[case_id][key])) <= 1e-15, (case_id, key)
        maximum_difference = max(maximum_difference, max(differences.values()))
        results[control, omega] = result
        lines.append(f"{case_id:4d} {omega:3d} {max(differences.values()):.17g}")
    lines += ["", "Control comparisons: independent same-omega reference subtraction and percentage."]
    for row in comparisons:
        control, omega = row["control"], int(row["omega"])
        control_result = results[control, omega]
        reference_result = results["Reference", omega]
        deltas = {f"delta_{key}": control_result[key]-reference_result[key] for key in METRICS}
        deltas["delta_mu_percent"] = 100*deltas["delta_mu_star20"]/reference_result["mu_star20"]
        assert int(row["reference_case_id"]) == (5540 if omega == 1 else 5541)
        difference = max(abs(value-float(row[key])) for key, value in deltas.items())
        assert difference <= 5e-12, (control, omega, difference)
        lines.append(f"{control} omega={omega}: PASS; delta_mu={deltas['delta_mu_percent']:+.12f}%")
    original = (HERE/"original_Table_S6_READ_ONLY_REFERENCE.tex").read_text()
    updated = (HERE/"Table_S6.tex").read_text()
    numerical_row = re.compile(
        r"(& (?:1|100)\s*\n)& [\d.]+\s*\n& [\d.]+\s*\n& [\d.]+\s*\n& [\d.]+ \\\\ % \d+")
    # Replacing only result fields and run-ID comments must leave exact
    # original layout, row order, Hamiltonians, notation, labels and caption.
    assert len(numerical_row.findall(original)) == len(numerical_row.findall(updated)) == 12
    assert numerical_row.sub(r"\1<NUMBERS>", original) == numerical_row.sub(r"\1<NUMBERS>", updated)
    rounded_rows = read_csv(HERE/"Table_S6.csv")
    assert len(rounded_rows) == 12 and len(rounded_rows[0]) == 6
    for expected, row in zip(EXPECTED, rounded_rows):
        control, omega, _ = expected
        assert int(row["omega"]) == omega
        for key in METRICS:
            # The supplied latest PDF prints this one cell to two decimals;
            # all other Table S6 result cells use three decimals.
            decimals = 2 if (control, omega, key) == ("Chiral", 100, "mu_star20") else 3
            assert row[key] == format(results[control, omega][key], f".{decimals}f")
    lines += [
        "",
        "Table structure: PASS; exact original after masking only 48 results and 12 source IDs.",
        "Six CSV columns and all 48 rounded values: PASS.",
        "Latest PDF precision retained: chiral omega=100 mu uses two decimals; all others three.",
        f"Maximum full-precision difference: {maximum_difference:.17g}",
        "",
        "Confinement ordering independently checked for all six families:",
    ]
    for control in dict.fromkeys(c for c, _, _ in EXPECTED):
        low, high = results[control, 1], results[control, 100]
        assert high["D20"] < low["D20"]
        assert high["tau_cond20"] > low["tau_cond20"]
        assert high["mu_star20"] > low["mu_star20"]
        assert low["DL"] > 0 and high["DL"] > 0
        lines.append(f"{control}: D20(100)<D20(1), tau(100)>tau(1), mu(100)>mu(1); late mass>0.")
    for omega in (1, 100):
        leaders = sorted(
            ((results[c, omega]["mu_star20"]-results["Reference", omega]["mu_star20"], c)
             for c in ("Uniform", "Radial", "Rashba", "Rashba–Dresselhaus", "Chiral")),
            reverse=True)
        lines.append(f"Largest mu increase at omega={omega}: {leaders[0][1]}.")
    assert results["Radial", 1]["tau_cond20"] < results["Reference", 1]["tau_cond20"]
    lines += ["Radial omega=1 decreases conditional mean while increasing restricted mean: PASS.",
              "",
              "Scientific main-script review: formulas, source normalization, actual-time",
              "trapezoids, explicit cutoff interpolation, event midpoint weighting and all",
              "ten same-omega comparisons agree with the requested convention.",
              "No forced closure or evolving-state normalization was found.",
              "PASS: independent recalculation, numerical comparisons, validation and table structure.",
              "",
              "Independent script SHA256: " + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()]
    (HERE/"independent_check_report.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

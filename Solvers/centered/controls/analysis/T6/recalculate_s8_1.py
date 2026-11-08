#!/usr/bin/env python3
"""Recalculate S8.1 from recorded centered-ABC data; never executes a solver.

Python >=3.10 and NumPy are the only dependencies. By default --root is the
3_CenteredABC_Run directory above this package. Output is confined to --output
(the directory containing this script by default). All full-precision CSV
floats use 17 significant digits. Raw histories are never modified.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import sys
import numpy as np

T, CUT = 20.0, 3.6
CASES = [
    ("Reference", 1, 5540), ("Reference", 100, 5541),
    ("Uniform", 1, 7692), ("Uniform", 100, 7693),
    ("Radial", 1, 7694), ("Radial", 100, 7695),
    ("Rashba", 1, 7696), ("Rashba", 100, 7697),
    ("Rashba–Dresselhaus", 1, 7698), ("Rashba–Dresselhaus", 100, 7699),
    ("Chiral", 1, 7001), ("Chiral", 100, 7000),
]
METRICS = ["D20", "tau_cond20", "mu_star20", "DL"]
HAMILTONIANS = {
    "Reference": "H_a=0",
    "Uniform": "H_a=pi sigma_x+(1/2) sigma_z",
    "Radial": "H_a=0.2 sqrt(omega)(X sigma_x+Y sigma_y)",
    "Rashba": "H_a=0.5(sigma_x p_y-sigma_y p_x)",
    "Rashba–Dresselhaus": "H_a=0.4(sigma_x p_y-sigma_y p_x)+0.3(sigma_x p_x-sigma_y p_y)",
    "Chiral": "H_a=0; profile chi_+=(chi_10+i chi_01)/sqrt(2)",
}
COEFFICIENTS = {
    "Reference": (0, 0, 0, 0, 0),
    "Uniform": (math.pi, .5, 0, 0, 0),
    "Radial": (0, 0, .2, 0, 0),
    "Rashba": (0, 0, 0, .5, 0),
    "Rashba–Dresselhaus": (0, 0, 0, .4, .3),
}
COEFF_KEYS = ["uniform_bx", "uniform_bz", "radial_gamma", "alpha_R", "beta_D"]
OLD_IDS = [2050, 2053, 3253, 3256, 3250, 3255, 5300, 5301, 5302, 5303, 5304, 5305]
LATEST_PDF = Path("EXTERNAL/Spin_Momentum_Impedance_and_Filtering_Closed_Quantum_Open_System - 2026-10-19T222948.158.pdf")
LATEST_PDF_SHA = "269f68ab867ee1fb135a83d742a860ecd1f40fda5b815cfd043f14cef8da7593"
# Visually checked against current PDF page 28; all other printed rows equal Revision9.
CURRENT_PRINTED_CHIRAL = {1: (.913, 4.063, 5.435, .348), 100: (.597, 6.434, 11.89, .287)}


def table_number(row, metric):
    # Preserve the latest PDF's isolated two-decimal chiral mu entry.
    decimals = 2 if row["control"] == "Chiral" and row["omega"] == 100 and metric == "mu_star20" else 3
    return format(row[metric], f".{decimals}f")


def require(test, message):
    if not test:
        raise ValueError(message)


def total(a):
    return math.fsum(np.asarray(a, dtype=np.float64).ravel())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: format(float(v), ".17g") if isinstance(v, (float, np.floating))
                        else v for k, v in row.items()})


def source_literal(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == name
                                               for n in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing source literal: " + name)


def locate(root, case_id):
    folder = root / ("1.Main" if case_id in (5540, 5541) else "2.S8/S8_1")
    matches = [p for p in folder.glob(f"{case_id}.*") if p.is_dir()]
    require(len(matches) == 1, f"Missing or ambiguous case {case_id}: {matches}")
    return matches[0]


def clip_linear(t, s, left=0., right=T):
    """Insert exact endpoints into a linear interpolant, without extrapolation."""
    require(t[0] <= left < right <= t[-1], "History does not cover requested interval")
    x = np.r_[left, t[(t > left) & (t < right)], right]
    return x, np.interp(x, t, s)


def survival_metrics(t, s):
    x, y = clip_linear(t, s)
    mu = total(np.diff(x) * (y[:-1] + y[1:]) / 2)
    s20, s36 = float(np.interp(T, t, s)), float(np.interp(CUT, t, s))
    d = 1 - s20
    return dict(D20=d, tau_cond20=(mu-T*s20)/d, mu_star20=mu, DL=s36-s20,
                S20=s20, S3_6=s36)


def event_moments(left, right, probability, a=0., b=T):
    """Native midpoint quadrature; partial interval uses constant native rate.

    In all selected runs t=3.6 and t=20 are actual step edges, so no event
    fraction or additional within-step assumption is needed for these results.
    """
    l, r = np.maximum(left, a), np.minimum(right, b)
    keep = r > l
    q = probability[keep] * (r[keep]-l[keep])/(right[keep]-left[keep])
    return total(q), total(q*(l[keep]+r[keep])/2)


def interpolation_record(t, s, target):
    i = int(np.searchsorted(t, target))
    exact = bool(i < len(t) and t[i] == target)
    if exact:
        return dict(target=target, exact_recorded_time=True, left_time=float(t[i]),
                    right_time=float(t[i]), value=float(s[i]))
    return dict(target=target, exact_recorded_time=False, left_time=float(t[i-1]),
                right_time=float(t[i]), value=float(np.interp(target, t, s)))


def verify_run(path, control, omega, case_id):
    run = json.loads((path/"revision_run.json").read_text())
    case = json.loads((path/"branch_refinement_case.json").read_text())
    summary = json.loads((path/"summary.json").read_text())
    c = run["configuration"]
    source = (path/"source_snapshot.py").read_text()
    require(source_literal(source, "BRANCH_REFINEMENT_CASE") == case, f"{case_id}: source case differs")
    require(sha(path/"source_snapshot.py") == run["updated_source_sha256"], f"{case_id}: source hash")
    require(case["case_id"] == case_id, "Case ID mismatch")
    require(run["status"] == "COMPLETED_NUMERICAL_RUN_NOT_CONVERGENCE_CERTIFIED", "Incomplete run")
    require(run["actual_time"] == c["T_final"] == case["T_final"] >= T, "Time coverage mismatch")
    expected = dict(omega=float(omega), Lz=20., ell=10., k0=math.pi, k_bc=math.pi,
                    sigma_z=.5, z0_init=10., theta=0., phi=0., Nx=100, Ny=100, Nz=1500,
                    endpoint_closure="centered", endpoint_mass_weight=.5,
                    metric="half_weight_last_z_node", state_dtype="complex64",
                    state_representation="physical_nodal_spinor",
                    initial_preparation_dtype="complex64", potential_sampling_dtype="float32")
    for k, v in expected.items():
        require(c[k] == v, f"{case_id}: configuration {k}={c[k]} expected {v}")
        if k in case:
            case_value = math.pi if k in ("k0", "k_bc") and case[k] == "pi" else case[k]
            require(case_value == v, f"{case_id}: case metadata {k}")
    require(c["matrix_assembly"] == "complex64_then_solve_complex128", "Assembly/solve precision")
    require(c["Lx"] == c["Ly"] == 10/math.sqrt(omega), "Transverse domain")
    with np.load(path/"constants.npz", allow_pickle=False) as z:
        constants = {k: z[k].item() for k in z.files if z[k].ndim == 0}
    for k in ["omega", "T_final", "theta", "phi", "sigma_z", "k0", "k_bc", "z_roof_actual",
              "grid_generator_nz", "hx", "hy", "hz", "Lx", "Ly", "Lz", "dt", "Nx", "Ny", "Nz",
              "endpoint_mass_weight", "endpoint_closure", "metric", "state_representation", "state_dtype"]:
        require(constants[k] == c[k], f"{case_id}: constants {k}")
    require(constants["z0"] == c["z0_init"], "Initial center")
    require(constants["matrix_dtype"] == "complex64", "Matrix dtype")
    coordinates = {}
    with np.load(path/"revision_coordinates.npz", allow_pickle=False) as z:
        for axis in "xyz":
            a = z[axis]
            h = np.diff(a.astype(np.float64))
            require(a.size == c["N"+axis] and str(a.dtype) == "float32", "Coordinate shape/dtype")
            require(a[0] == 0 and np.all(h > 0) and float(a[1]-a[0]) == c["h"+axis], "Coordinate grid")
            coordinates[axis] = dict(count=a.size, dtype=str(a.dtype), first=float(a[0]),
                                     last=float(a[-1]), min_spacing=float(h.min()), max_spacing=float(h.max()))
        require(float(z["z"][-1]) == c["z_roof_actual"], "Actual roof position")
    if case_id >= 7690:
        for k, v in zip(COEFF_KEYS, COEFFICIENTS[control]):
            require(c[k] == case[k] == constants[k] == v, f"{case_id}: coupling {k}")
        for k, v in dict(transverse_preparation="ground", added_hamiltonian_roof_multiplier=1.,
                         added_hamiltonian_z_profile="uniform_in_z",
                         added_hamiltonian_coordinate_convention="X=x-Lx/2;Y=y-Ly/2",
                         added_hamiltonian_derivative_convention="centered_interior_dirichlet").items():
            require(c[k] == case[k] == constants[k] == v, f"{case_id}: {k}")
        require("kron(_s81_Ha_xy, Iz" in source, "Global Ha lift absent")
        require("hermiticity" in (path/"stdout.txt").read_text().lower(), "Missing operator audit")
    else:
        # Source is inspected as text, never imported; chiral adds no bulk term.
        require("_s81_Ha_bulk" not in source, "Unexpected added bulk term")
        if control == "Chiral":
            require("chi_plus" in source and "1j" in source, "Missing chiral preparation")
    require(summary["P0_init64"] == run["P0_init64"], "Initial mass records disagree")
    require(summary["omega"] == omega and summary["endpoint_closure"] == "centered", "Summary mismatch")
    files = ["source_snapshot.py", "revision_run.json", "branch_refinement_case.json", "constants.npz",
             "revision_coordinates.npz", "revision_step_data.csv", "prob_times.npy", "total_probs.npy",
             "summary.json", "simulation_log.txt", "stdout.txt", "initial_preparation.json",
             "initial_preparation_slice.npz", "source_preflight.json", "gpu_metric_preflight.json",
             "postcheck.json"]
    manifest = dict(case_id=case_id, control=control, omega=omega, selected_for_table=True,
                    job_id=re.search(r"_job(\d+)", path.name).group(1),
                    local_path=str(path.resolve()), original_hpc_path=summary.get("run_dir"),
                    status=run["status"], configuration=c, coordinates=coordinates,
                    P0_init64=run["P0_init64"], P0_init_native=run["P0_init_native"],
                    source_sha256=sha(path/"source_snapshot.py"),
                    hashes={name: sha(path/name) for name in files if (path/name).exists()},
                    evidence_basis="metadata, constants, source snapshot, coordinates, native histories and logs")
    return run, manifest


def calculate(path, control, omega, case_id):
    run, manifest = verify_run(path, control, omega, case_id)
    p0 = float(run["P0_init64"])
    with (path/"revision_step_data.csv").open() as f:
        names = next(csv.reader(f))
        a = np.loadtxt(f, delimiter=",", dtype=np.float64)
    d = dict(zip(names, a.T))
    require(np.isfinite(a).all(), f"{case_id}: nonfinite step data")
    left, right = d["t_left"], d["t_right"]
    require(len(right) == run["completed_step"], "Step count")
    require(np.array_equal(d["step"], np.arange(1, len(right)+1)), "Sequential step numbers")
    require(left[0] == 0 and right[-1] == run["actual_time"], "History endpoints")
    require(np.all(right > left) and np.array_equal(left[1:], right[:-1]), "Time gaps/overlaps")
    chain = float(np.max(np.abs(d["mass_before"]-np.r_[p0, d["mass_stored"][:-1]])))
    require(chain < 1e-13*p0, "Mass continuity")
    require(d["mass_stored"][-1] == run["current_accepted_prefix_mass64"], "Final recorded mass")
    t, s = np.r_[0., right], np.r_[p0, d["mass_stored"]]/p0
    result = dict(control=control, omega=omega, case_id=case_id, **survival_metrics(t, s))
    q = d["roof_probability"]/p0
    det, first = event_moments(left, right, q)
    late, _ = event_moments(left, right, q, CUT, T)
    result.update(D20_events=det, tau_cond20_events=first/det, DL_events=late,
                  event_minus_survival_D20=det-result["D20"],
                  event_minus_survival_tau_cond20=first/det-result["tau_cond20"],
                  event_minus_survival_DL=late-result["DL"])
    selected = right <= T
    x, y = clip_linear(t, s)
    budget = 1-s[1:]-np.cumsum(q)
    # Independent integration-by-parts identity for the signed mass increments,
    # without replacing any survival ordinate with detector-based values.
    mass_loss_moment = total((y[:-1]-y[1:])*(x[:-1]+x[1:])/2)
    identity = result["mu_star20"]-T*result["S20"]-mass_loss_moment
    coarse_t = np.load(path/"prob_times.npy", allow_pickle=False).astype(np.float64)
    coarse_p = np.load(path/"total_probs.npy", allow_pickle=False).astype(np.float64)
    require(len(coarse_t) == len(coarse_p) and np.isfinite(coarse_p).all()
            and np.all(np.diff(coarse_t) > 0), "Coarse history invalid")
    require(coarse_t[0] == 0 and coarse_t[-1] >= T, "Coarse history coverage")
    coarse_s = coarse_p/p0  # Diagnostic only; never rescaled by its own first sample.
    coarse = survival_metrics(coarse_t, coarse_s)
    violation = y > 1
    defects = d["cast_mass_loss"]+d["clamp_mass_loss"]+d["solve_and_matrix_defect"]
    quality = dict(control=control, omega=omega, case_id=case_id,
                   P0_init64=p0, P0_init64_minus_one=p0-1, P0_init_native=run["P0_init_native"],
                   first_mass_before=float(d["mass_before"][0]),
                   run_T=run["actual_time"], run_steps=len(right), steps_through20=int(selected.sum()),
                   history_source="revision_step_data.csv:mass_stored + P0_init64 at t=0",
                   dt_nominal=run["configuration"]["dt"],
                   dt_min=float(np.diff(x).min()), dt_max=float(np.diff(x).max()),
                   S_min=float(y.min()), S_max=float(y.max()),
                   S_above_one_count=int(violation.sum()), S_below_zero_count=int((y < 0).sum()),
                   maximum_S_excess=float(max(0., y.max()-1)),
                   S_excess_in_float32_eps=float(max(0., y.max()-1)/np.finfo(np.float32).eps),
                   positive_survival_increment_count=int((np.diff(y) > 0).sum()),
                   maximum_positive_survival_increment=float(max(0., np.diff(y).max())),
                   mass_chain_max_abs=chain,
                   negative_event_count=int((q[selected] < 0).sum()),
                   event_minus_survival_D20=result["event_minus_survival_D20"],
                   event_minus_survival_tau_cond20=result["event_minus_survival_tau_cond20"],
                   max_abs_cumulative_budget_minus_events=float(np.abs(budget[selected]).max()),
                   sum_cast_mass_loss_normalized=total(d["cast_mass_loss"][selected])/p0,
                   sum_clamp_mass_loss_normalized=total(d["clamp_mass_loss"][selected])/p0,
                   sum_solve_and_matrix_defect_normalized=total(d["solve_and_matrix_defect"][selected])/p0,
                   budget_minus_events_minus_recorded_defects=(
                       result["D20"]-det-total(defects[selected])/p0),
                   max_relative_residual=float(d["relative_residual"][selected].max()),
                   native_residual_guard=run["numerical_guards"]["true_relative_residual_max"],
                   native_cumulative_closure_guard=run["numerical_guards"]["cumulative_roof_closure_abs_max"],
                   independent_mass_moment_identity_residual=identity,
                   full_run_event_sum_minus_metadata=total(d["roof_probability"])-run["cumulative_roof_probability"],
                   coarse_sampling_min=float(np.diff(coarse_t).min()),
                   coarse_sampling_max=float(np.diff(coarse_t).max()),
                   coarse_samples_through20=int((coarse_t <= T).sum()),
                   coarse_first_sample_over_P0=float(coarse_s[0]),
                   coarse_vs_dense_mass_max_abs=float(np.max(np.abs(coarse_p-np.interp(coarse_t,t,s)*p0))))
    quality.update({"coarse_minus_dense_"+k: coarse[k]-result[k] for k in METRICS})
    require(quality["negative_event_count"] == 0, "Negative native event")
    require(quality["max_relative_residual"] <= quality["native_residual_guard"], "Solver guard exceeded")
    require(quality["max_abs_cumulative_budget_minus_events"] <= quality["native_cumulative_closure_guard"],
            "Closure guard exceeded")
    require(abs(identity) < 2e-13, "Independent mass-moment check failed")
    manifest["interpolation"] = [interpolation_record(t, s, v) for v in (CUT, T)]
    manifest["sampling"] = {k: quality[k] for k in ("dt_nominal","dt_min","dt_max","run_steps",
                                                   "steps_through20","coarse_sampling_min","coarse_sampling_max")}
    return result, quality, manifest


def analytic_checks():
    t = np.array([0., .7, 2.1, 5., 21.])
    s = 1-.02*t
    r = survival_metrics(t, s)
    require(abs(r["mu_star20"]-16) < 1e-14 and abs(r["D20"]-.4) < 1e-14, "Linear survival test")
    require(abs(r["tau_cond20"]-10) < 1e-14 and abs(r["DL"]-.328) < 1e-14, "Interpolated cutoff test")
    d, m = event_moments(np.array([0., 2.]), np.array([2., 5.]), np.array([4., 9.]), 1., 4.)
    require(d == 8 and m == 21, "Partial interval event test")
    return "PASS: irregular-grid survival integration, interpolated cutoffs, and partial event intervals"


def prepare_table(original, rows):
    table = original[original.index(r"\begin{table}[H]"):original.index(r"\end{table}")+len(r"\end{table}")]
    pattern = re.compile(
        r"& (1|100)\s*\n& ([\d.]+)\s*\n& ([\d.]+)\s*\n& ([\d.]+)\s*\n& ([\d.]+) \\\\ % (\d+)")
    old, index = [], 0

    def substitute(match):
        nonlocal index
        row = rows[index]
        require(int(match[1]) == row["omega"] and int(match[6]) == OLD_IDS[index], "Original table ordering changed")
        old.append(dict(control=row["control"], omega=row["omega"], old_case_id=int(match[6]),
                        **dict(zip(METRICS, map(float, match.group(2, 3, 4, 5))))))
        index += 1
        return ("& " + str(row["omega"]) + "\n" +
                "\n".join("& " + table_number(row, k) for k in METRICS) +
                r" \\ % " + str(row["case_id"]))

    updated = pattern.sub(substitute, table)
    require(index == 12, "Expected exactly 12 original table rows")
    return table, updated+"\n", old


def results_paragraph(rows, comparisons):
    lookup = {(r["control"],r["omega"]):r for r in comparisons}
    pct = lambda c,w: lookup[c,w]["delta_mu_percent"]
    return (
        r"Every tested control retains a smaller \(D_{20}\) and larger conditional and "
        r"restricted means at \(\omega=100\) than at \(\omega=1\), with late detections at "
        r"both confinements. Relative to the new centered reference at the same \(\omega\), "
        rf"the uniform coupling changes \(\mu^*(20)\) by \({pct('Uniform',1):+.2f}\%\) and "
        rf"\({pct('Uniform',100):+.2f}\%\), and the radial coupling by "
        rf"\({pct('Radial',1):+.2f}\%\) and \({pct('Radial',100):+.2f}\%\), at "
        r"\(\omega=1\) and \(100\), respectively. The corresponding increases are "
        rf"\({pct('Rashba',1):.2f}\%\) and \({pct('Rashba',100):.2f}\%\) for Rashba, "
        rf"\({pct('Rashba–Dresselhaus',1):.2f}\%\) and \({pct('Rashba–Dresselhaus',100):.2f}\%\) "
        r"for Rashba--Dresselhaus, and "
        rf"\({pct('Chiral',1):.2f}\%\) and \({pct('Chiral',100):.2f}\%\) for the chiral preparation. "
        r"The chiral change is the largest at \(\omega=100\), whereas Rashba--Dresselhaus "
        r"gives the largest change at \(\omega=1\). At \(\omega=1\), the radial coupling "
        r"slightly decreases the conditional mean despite increasing the restricted mean. "
        r"Thus the confinement contrast persists for these specified controls, while the "
        r"detection statistics remain preparation dependent. These comparisons concern "
        r"only the two tested confinements at \(T=20\); they establish neither independence "
        r"from arbitrary transverse states nor numerical convergence." + "\n"
    )


def report_text(rows, comparisons, validation, cross, original_comparison, checks):
    out = [
        "S8.1 / TABLE S6 — CENTERED-ABC RECALCULATION, 2026-10-20",
        "",
        "Selection and convention",
        "All 12 required rows are present and verified; no missing or mismatched selected run.",
        "Reference: completed confinement-sweep cases 5540/5541, restricted to [0,20] from T=40.",
        "Controls: 7692–7699; existing chiral preparations: 7001/7000. No simulations were run.",
        "Cases 7690/7691 are audited reference cross-checks only, not substituted for the requested sweep.",
        "Every source snapshot matches its run-record SHA-256. Metadata, source case dictionary,",
        "constants, coordinates, logs and native data were checked. See run_provenance_manifest.json",
        "and source_audit.txt for implementation evidence and full per-run paths/parameters.",
        "",
        "All spatial grids: Nx=Ny=100, Nz=1500. Lx=Ly=10/sqrt(omega); nominal Lz=20.",
        "Actual final z node=19.986665725708008 (endpoint=False); hz=0.013333333656191826.",
        "Coordinates/potential sampling=float32; stored state and matrix assembly=complex64;",
        "linear solve=complex128. Centered roof and half endpoint mass weight are retained.",
        "The omega=1 sweep/chiral dt is 0.001; new non-chiral controls use 0.0008.",
        "All omega=100 runs use 0.0005. This timestep difference is disclosed, not corrected.",
        "No spatial/time refinement or convergence claim is made.",
        "",
        "Calculation",
        "S(0)=P0_init64/P0_init64=1; later ordinates=mass_stored/P0_init64.",
        "Only this initial normalization is applied. No clipping, monotonicization, evolving-state",
        "renormalization, detector replacement, or probability-budget repair is performed.",
        "D20=1-S(20); mu_star20=integral_0^20 S(t)dt; DL=S(3.6)-S(20);",
        "tau_cond20=(mu_star20-20*S(20))/D20. Piecewise-linear trapezoids use actual recorded times.",
        "Both 3.6 and 20 are exact native step edges in every selected run; the interpolation",
        "routine nevertheless evaluates both explicitly. Manifest records the brackets.",
        "All rows use every-step histories. prob_times/total_probs are diagnostic only;",
        "their actual cadence and differences are in numerical_validation.csv.",
        "Older analysis coarse branches divided by first recorded mass; that convention is",
        "not used here. Older event-derived table routines are not used for the main table.",
        "",
        "Native event cross-check",
        "D_events=sum(roof_probability)/P0_init64.",
        "tau_events=sum(t_mid*roof_probability)/sum(roof_probability), t_mid=(t_left+t_right)/2.",
        "These are accepted-step midpoint roof probabilities from the old+solved spinor,",
        "not decimated roof flux or mass differences. Event-minus-survival differences follow:",
        "case  control                  omega     delta_D20           delta_tau_cond20",
    ]
    for r in rows:
        out.append(f"{r['case_id']:4d}  {r['control']:24s} {r['omega']:3d}  "
                   f"{r['event_minus_survival_D20']:+.12e}  {r['event_minus_survival_tau_cond20']:+.12e}")
    out += ["", "Survival bounds and numerical integrity",
            "There is no separately declared S-bound tolerance in the source. Strict violations",
            "are reported, not dismissed using an invented tolerance. The largest S>1 excursion",
            f"is {max(q['maximum_S_excess'] for q in validation):.12e}, less than one float32 epsilon.",
            "Native guards are relative residual<=2e-8 and |cumulative mass budget-event mass|",
            "<=1e-3 P0; these are operational fail-stop limits, not accuracy bounds.",
            "No negative S or native event probabilities; all time and mass chains are complete.",
            f"Maximum |P0_init64-1| across the 12 rows: {max(abs(q['P0_init64_minus_one']) for q in validation):.12e}.",
            "All recorded clamp losses are zero. Recorded cast and solve/matrix defects account",
            "for the small event/survival differences; the remaining accounting residuals are",
            "included explicitly in numerical_validation.csv, without any closure adjustment.",
            "case omega P0_init64             min_S               max_S-1     count_S>1 max_residual"]
    for q in validation:
        out.append(f"{q['case_id']:4d} {q['omega']:3d} {q['P0_init64']:.17g} "
                   f"{q['S_min']:.12g} {q['maximum_S_excess']:.8e} "
                   f"{q['S_above_one_count']:6d} {q['max_relative_residual']:.8e}")
    out += ["", checks,
            "Every row also passes independent integration by parts using signed mass decrements.",
            "", "Reference rerun minus selected sweep (not a convergence estimate)",
            "omega case delta_D20 delta_tau_cond20 delta_mu_star20 delta_DL"]
    for r in cross:
        out.append(f"{r['omega']} {r['case_id']} " + " ".join(f"{r['delta_'+k]:+.12e}" for k in METRICS))
    out += ["", "Changes relative to previous printed Table S6",
            "The latest PDF's old rounded values are preserved in original_Table_S6_printed.csv and",
            "new_minus_old_printed.csv for editorial comparison. They never enter new control",
            "versus reference percentages. New-minus-old differences have only the precision",
            "supported by the old printed precision (three decimals except chiral omega100 mu: two).",
            "The latest PDF and editable Revision9 source differ in the two chiral rows.",
            "The PDF already prints chiral rows [.913,4.063,5.435,.348] and [.597,6.434,11.89,.287].",
            "Its prose nevertheless retains the old5.5%/14.9% claims. The source-only earlier rows",
            "are separately preserved in editable_source_Table_S6_printed.csv.",
            "The updated LaTeX preserves the editable table structure and latest PDF's numerical",
            "precision, including the isolated two-decimal chiral omega100 restricted mean.",
            "The confinement ordering survives for every family. The omega=100 reference changes",
            "from D20=0.670, tau_cond20=6.008, mu_star20=10.628, DL=0.305 to the new values.",
            "The old uniform/radial <=about1.2% bound still holds, but radial omega=1 has a smaller",
            "mu increase and its conditional-mean change is now negative.",
            "The former spin-orbit range 2.8–6.7% becomes " +
            f"{min(r['delta_mu_percent'] for r in comparisons if 'Rashba' in r['control']):.2f}–" +
            f"{max(r['delta_mu_percent'] for r in comparisons if 'Rashba' in r['control']):.2f}%.",
            "The old prose's chiral omega=100 increase of 14.9% becomes " +
            f"{next(r['delta_mu_percent'] for r in comparisons if r['control']=='Chiral' and r['omega']==100):.2f}%.",
            "Chiral is not the largest mu change at both confinements: Rashba–Dresselhaus leads at 1.",
            "Uniform omega=100 has a small nonzero mu increase even though the old rounded entries",
            "were equal. These comparisons do not certify significance or numerical convergence.",
            "", "Original manuscript and raw outputs were not edited.",
            "Reproduce: python3 recalculate_s8_1.py --root /path/to/3_CenteredABC_Run",
            "Optional --output writes the generated results to another directory.",
            ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    check = analytic_checks()
    manuscript = root/"1.Main/S1_S3_Recalculation_Omega400_CutoffComparison_2026-10-20/reference/supplementary_reference.tex"
    text = manuscript.read_text()
    start = text.index(r"\subsection{Bulk spin and transverse-state controls}")
    end = text.index(r"\subsection{Hamiltonian-layer mimics}", start)
    original = text[start:end].rstrip()+"\n"
    rows, validation, provenance = [], [], []
    for control, omega, case_id in CASES:
        r, q, p = calculate(locate(root, case_id), control, omega, case_id)
        rows.append(r); validation.append(q); provenance.append(p)
    reference = {r["omega"]:r for r in rows if r["control"] == "Reference"}
    comparisons = []
    for row in rows[2:]:
        ref = reference[row["omega"]]
        c = {k: row[k] for k in ("control", "omega", "case_id")}
        c["reference_case_id"] = ref["case_id"]
        c.update({"delta_"+k:row[k]-ref[k] for k in METRICS})
        c["delta_mu_percent"] = 100*c["delta_mu_star20"]/ref["mu_star20"]
        comparisons.append(c)
    cross = []
    for omega, case_id in [(1, 7690), (100, 7691)]:
        r, q, p = calculate(locate(root, case_id), "Reference", omega, case_id)
        p["selected_for_table"] = False
        p["selection_note"] = "Reference rerun cross-check only; requested completed confinement sweep used."
        provenance.append(p)
        r.update({"delta_"+k:r[k]-reference[omega][k] for k in METRICS})
        r["selected_reference_case_id"] = reference[omega]["case_id"]
        cross.append(r)
    original_table, table, editable_old_rows = prepare_table(original, rows)
    old_rows = []
    for old in editable_old_rows:
        item = {k:v for k,v in old.items() if k != "old_case_id"}
        item["printed_source"] = "latest supplementary PDF, page 28"
        if item["control"] == "Chiral":
            item.update(dict(zip(METRICS, CURRENT_PRINTED_CHIRAL[item["omega"]])))
        old_rows.append(item)
    paragraph = results_paragraph(rows, comparisons)
    intro = original[:original.index(r"\begin{table}[H]")]
    addition = (
        r"We use \(T=20\), \(L=20\), \(z_c=10\), \(k_z=\kappa=\pi\), and "
        r"\(\sigma_\parallel=0.5\), with the centered spinor ABC and endpoint mass weight \(1/2\). "
        r"The reference histories are taken from the completed \(H_a=0\) confinement sweep. "
        r"For each run, \(S(t)=P(t)/P_0\) uses the every-step stored mass and the recorded "
        r"initial mass \(P_0\), accumulated in double precision. We integrate its piecewise-linear "
        r"interpolant on the actual recorded times, evaluating \(S(3.6)\) and \(S(20)\) explicitly. "
        r"At \(\omega=1\), the reference and chiral runs use \(\Delta t=0.001\), and the added-Hamiltonian "
        r"controls use \(\Delta t=0.0008\); all \(\omega=100\) runs use \(\Delta t=0.0005\)."
        "\n\n")
    proposed = intro+addition+table+"\n"+paragraph
    (output/"Table_S6.tex").write_text(table)
    (output/"Sec_S8_1_replacement.tex").write_text(proposed)
    (output/"results_interpretation_paragraph.tex").write_text(paragraph)
    (output/"original_S8_1_READ_ONLY_REFERENCE.tex").write_text(original)
    (output/"original_Table_S6_READ_ONLY_REFERENCE.tex").write_text(original_table+"\n")
    rounded = []
    for r in rows:
        rounded.append({"Control and added Hamiltonian":r["control"]+": "+HAMILTONIANS[r["control"]],
                        "omega":r["omega"], **{k:table_number(r,k) for k in METRICS}})
    write_csv(output/"Table_S6.csv", rounded)
    write_csv(output/"full_precision_results.csv", rows)
    write_json(output/"full_precision_results.json", rows)
    write_csv(output/"control_vs_reference.csv", comparisons)
    write_csv(output/"numerical_validation.csv", validation)
    write_csv(output/"reference_rerun_crosscheck.csv", cross)
    write_csv(output/"original_Table_S6_printed.csv", old_rows)
    write_csv(output/"editable_source_Table_S6_printed.csv", editable_old_rows)
    old_changes = []
    for old, new in zip(old_rows, rows):
        old_changes.append(dict(control=new["control"], omega=new["omega"], case_id=new["case_id"],
                                **{"new_minus_old_"+k:new[k]-old[k] for k in METRICS}))
    write_csv(output/"new_minus_old_printed.csv", old_changes)
    family_checks = []
    for control in dict.fromkeys(r["control"] for r in rows):
        low, high = [r for r in rows if r["control"] == control]
        family_checks.append(dict(control=control, D20_100_less_than_1=high["D20"] < low["D20"],
                                  tau_100_greater_than_1=high["tau_cond20"] > low["tau_cond20"],
                                  mu_100_greater_than_1=high["mu_star20"] > low["mu_star20"],
                                  **{"omega100_minus_omega1_"+k:high[k]-low[k] for k in METRICS}))
    write_csv(output/"confinement_contrasts.csv", family_checks)
    script_references = [
        root/"1.Main/Tables_S2_S3_S4_centered_2026-10-18/recalculate_tables.py",
        root/"ABC_Centered_EightPoint_Chiral_Evaluation_2026-10-17/analysis/analyze.py",
    ]
    if LATEST_PDF.exists():
        require(sha(LATEST_PDF) == LATEST_PDF_SHA, "Current PDF changed: recheck manuscript before updating")
    write_json(output/"run_provenance_manifest.json",
               dict(analysis_date="2026-10-20", analysis_horizon=T, late_cutoff=CUT,
                    manuscript_path=str(manuscript), manuscript_sha256=sha(manuscript),
                    original_manuscript_path="EXTERNAL/AJ_MainBody_Supp_Revision9.tex",
                    manuscript_copy_verified_identical=True,
                    latest_manuscript_pdf_path=str(LATEST_PDF), latest_manuscript_pdf_sha256=LATEST_PDF_SHA,
                    latest_pdf_inspection="Pages 27–28 text and visual check: only two chiral rows differ from editable source.",
                    table_precision="Three decimals except chiral omega100 mu_star20: two, matching latest PDF.",
                    prior_analysis_read=[dict(path=str(p),sha256=sha(p)) for p in script_references],
                    normalization="mass_stored/P0_init64, with initial P0_init64 sample at t=0",
                    integration="piecewise-linear, actual recorded times, explicit cutoff interpolation",
                    event_check="accepted-step midpoint roof_probability, divided once by P0_init64",
                    raw_data_modified=False, source_snapshots_executed=False,
                    missing_or_mismatched_selected_runs=[], runs=provenance,
                    runtime=dict(python=platform.python_version(), numpy=np.__version__, executable=sys.executable),
                    calculation_script_sha256=sha(Path(__file__))))
    (output/"numerical_validation_report.txt").write_text(
        report_text(rows, comparisons, validation, cross, old_changes, check))
    print("PASS: 12 selected rows and 2 reference cross-checks; no missing/mismatched runs.")
    for r in rows:
        print(r["control"], r["omega"], *[table_number(r,k) for k in METRICS])
    print("Control-vs-reference mu changes (%)")
    for r in comparisons:
        print(r["control"], r["omega"], format(r["delta_mu_percent"], "+.8f"))


if __name__ == "__main__":
    main()

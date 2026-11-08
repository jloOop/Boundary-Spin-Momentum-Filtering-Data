"""Presentation specifications and compact summaries for Tables S1--S6.

Rendering only reads canonical CSVs. ``--prepare`` explicitly rebuilds the two
settings summaries and the four-run refinement summary from bundled records.
The printed-value CSV is a transcription, never a native recomputation target.
No production solver or original research source directory is accessed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BASELINE = {1: 5540, 25: 5042, 50: 5047, 75: 5043, 100: 5541,
            150: 5044, 200: 5045, 300: 7100}
TABLES = "data/centered/tables/"
SETTINGS = TABLES + "settings/"
REFINEMENT = TABLES + "refinement/"
DIAGNOSTICS = TABLES + "diagnostics/"


def _read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format(v, ".17g") if isinstance(v, float) else v
                             for k, v in row.items()})


def settings_rows(root):
    """Read recorded configuration and the separately transcribed S1 vector."""
    root = Path(root)
    reported = {int(r["omega"]): r for r in _read(root / SETTINGS / "Table_S1_manuscript_timesteps.csv")}
    rows = []
    configs = []
    for omega, case in BASELINE.items():
        record = json.loads((root / "data/centered/native" / str(case) / "run.json").read_text())
        c = record["configuration"]
        assert c["omega"] == omega
        configs.append(c)
        rows.append(dict(
            omega=omega, case_id=case, Nx=c["Nx"], Ny=c["Ny"], Nz=c["Nz"],
            dt_recorded=c["dt"], dt_manuscript=float(reported[omega]["dt_manuscript"]),
            T_final=c["T_final"], z_detector_actual=c["z_roof_actual"],
            Lx_nominal=c["Lx"], Ly_nominal=c["Ly"], Lz_nominal=c["Lz"],
            hx_stencil=c["hx"], hy_stencil=c["hy"], hz_stencil=c["hz"],
            endpoint_mass_weight=c["endpoint_mass_weight"], norm_convention=c["metric"],
            coordinate_dtype=c["real_dtype"], state_dtype=c["state_dtype"],
            matrix_assembly=c["matrix_assembly"], diagnostic_dt_nominal=c["diagnostic_dt"],
            plane_record_every_steps=c["monitor_every"], survival_record_every_steps=c["prob_every"],
            launch_norm=record["P0_init64"], source_run=f"data/centered/native/{case}/run.json"))
    fields = {"L_nominal": "Lz", "source_z": "z0_init", "kz": "k0", "kappa": "k_bc",
              "sigma_amplitude": "sigma_z", "theta": "theta", "phi": "phi", "T_final": "T_final",
              "terminal_mass_weight": "endpoint_mass_weight", "gmres_rtol": "gmres_rtol",
              "gmres_atol": "gmres_atol", "gmres_restart": "gmres_restart", "gmres_maxiter": "gmres_maxiter"}
    for key in fields.values():
        assert all(c[key] == configs[0][key] for c in configs), key
    common = [{key: configs[0][source] for key, source in fields.items()}]
    return rows, common


def refinement_rows(root, diagnostics_output=None):
    """Join the first-pass four-run reducer outputs without changing formulas.

    ``diagnostics_output`` lets verification reuse an already rerun diagnostics
    group; otherwise only the four runs required by S5 are read and reduced.
    """
    root = Path(root)
    if diagnostics_output is not None:
        out = Path(diagnostics_output)
        stats = _read(out / "Table_S5_native_statistics.csv")
        pairs = _read(out / "Table_S5_EqS117_discrepancies.csv")
        stats = [{k: float(v) for k, v in row.items()} for row in stats]
        pairs = [{k: float(v) if v else None for k, v in row.items()} for row in pairs]
    else:
        import sys
        loaders = str(Path(__file__).resolve().parents[1])
        if loaders not in sys.path:
            sys.path.insert(0, loaders)
        from centered.diagnostics import _load, _stat, _compare_s5
        baseline = _load(root, 7100)
        stats, pairs = [], []
        for case in (7100, 7102, 7101, 7706):
            d = baseline if case == 7100 else _load(root, case)
            s2, s3, s4, _ = _stat(d)
            c = d["cfg"]
            stats.append(dict(case=case, Nx=c["Nx"], Nz=c["Nz"], dt=c["dt"], D=s2["D40"],
                              mu=s2["mu_star40"], tcond=s2["tau_cond40"], Pzdet=s4["Pbar_det_z40"],
                              Pznc=s4["P_nc_z40"], wplus=s4["wbar_br_plus40"], Es2=s3["E_s2"],
                              beta=s3["beta_bl_omega"], E2br=s3["E_2br"]))
            pairs.extend(_compare_s5(baseline, d, a, 40.) for a in (0., 8.))
    labels = {7100: "Baseline", 7102: "Transverse", 7101: "Longitudinal", 7706: "Half timestep"}
    assert {int(r["case"]) for r in stats} == set(labels)
    result = []
    for row in stats:
        case = int(row["case"])
        full = next(r for r in pairs if int(r["refined"]) == case and r["window_start"] == 0)
        late = next(r for r in pairs if int(r["refined"]) == case and r["window_start"] == 8)
        result.append(dict(
            case_id=case, refinement=labels[case], Nx=int(row["Nx"]), Ny=int(row["Nx"]),
            Nz=int(row["Nz"]), dt=row["dt"], omega=300, T=40, D=row["D"], mu=row["mu"],
            tcond=row["tcond"], epsilon_mu_refined_denom_percent=full["epsilon_mu_refined_denom_percent"],
            wplus=row["wplus"], epsilon_CN_full_percent=full["epsilon_CN_percent"],
            epsilon_branch_full_percent=full["epsilon_branch_percent"], Pzdet=row["Pzdet"],
            Pznc=row["Pznc"], Es2=row["Es2"], beta=row["beta"], beta_times_1000=1000*row["beta"],
            E2br=row["E2br"], epsilon_CN_late_percent=late["epsilon_CN_percent"],
            epsilon_branch_late_percent=late["epsilon_branch_percent"],
            full_window_start=0, full_window_end=40, late_window_start=8, late_window_end=40))
    return result


def prepare(root):
    root = Path(root)
    settings, common = settings_rows(root)
    _write(root / SETTINGS / "Table_S1_recorded_settings.csv", settings)
    _write(root / SETTINGS / "Table_S1_common_preparation.csv", common)
    _write(root / REFINEMENT / "Table_S5_recomputed.csv", refinement_rows(root))


def summary_checks(root, diagnostics_output=None):
    """Check canonical derived summaries against their bundled source records."""
    root = Path(root)
    settings, common = settings_rows(root)
    targets = [(SETTINGS+"Table_S1_recorded_settings.csv", settings),
               (SETTINGS+"Table_S1_common_preparation.csv", common),
               (REFINEMENT+"Table_S5_recomputed.csv", refinement_rows(root, diagnostics_output))]
    checks = []
    for name, expected in targets:
        actual = _read(root / name)
        ok = len(actual) == len(expected) and all(set(a) == set(b) for a, b in zip(actual, expected))
        max_error = 0.
        if ok:
            for a, b in zip(actual, expected):
                for key, value in b.items():
                    if isinstance(value, (int, float)):
                        error = abs(float(a[key]) - value)
                        max_error = max(max_error, error)
                        ok = ok and error <= 1e-12 * max(1., abs(value))
                    else:
                        ok = ok and a[key] == value
        checks.append(dict(name="supplement summary: " + name, passed=bool(ok),
                           actual=max_error, tolerance=1e-12))
    return checks


def manuscript_comparisons(root):
    """Return agreement at the exact displayed precision, including conflicts.

    These comparisons are evidence, not numerical normalization identities.
    A disagreement is retained; no numeric result is adjusted to the PDF.
    """
    root = Path(root)
    data = {}
    s2 = _read(root / DIAGNOSTICS / "Table_S2_archived.csv")
    for i, row in enumerate(s2):
        data[("S2", ("spinor_" if i < 8 else "scalar_") + row["omega"])] = row
    for table in ("S3", "S4"):
        for row in _read(root / DIAGNOSTICS / f"Table_{table}_archived.csv"):
            data[(table, row["omega"])] = row
    for row in _read(root / REFINEMENT / "Table_S5_recomputed.csv"):
        data[("S5", row["case_id"])] = row
    for row in _read(root / TABLES / "controls/T6/full_precision_results.csv"):
        data[("S6", row["case_id"])] = row
    result = []
    for row in _read(root / REFINEMENT / "manuscript_reported_rounding_S2_S6.csv"):
        value = float(data[(row["table_id"], row["row_id"])][row["quantity"]])
        reported = float(row["reported_value"])
        fmt = row["display_format"]
        def display(v):
            return format(int(v), fmt) if fmt == "d" and int(v) == v else format(v, fmt)
        actual_display, reported_display = display(value), display(reported)
        if fmt.endswith("e"):
            exponent = int(reported_display.split("e")[1])
            precision = int(fmt[1:-1])
            tolerance = 0.5 * 10 ** (exponent - precision)
        elif fmt.endswith("f"):
            tolerance = 0.5 * 10 ** -int(fmt[1:-1])
        else:
            tolerance = 0.
        result.append({**row, "numerical_value": value, "numerical_display": actual_display,
                       "manuscript_display": reported_display, "absolute_error": abs(value-reported),
                       "rounding_half_unit": tolerance, "agrees_at_printed_precision": actual_display == reported_display})
    for row in _read(root / SETTINGS / "Table_S1_recorded_settings.csv"):
        actual, printed = float(row["dt_recorded"]), float(row["dt_manuscript"])
        result.append(dict(table_id="S1", row_id=row["omega"], quantity="dt", reported_value=printed,
                           numerical_value=actual, display_format=".5g", source_pdf_page=17,
                           numerical_display=format(actual,".5g"), manuscript_display=format(printed,".5g"),
                           absolute_error=abs(actual-printed), rounding_half_unit=0,
                           agrees_at_printed_precision=actual == printed))
    return result


def _column(key, label, fmt=".3f", **extra):
    return dict(key=key, label=label, format=fmt, **extra)


def _panel(label, path, columns, count, **extra):
    return dict(label=label, csv=path, columns=columns, expected_rows=count, **extra)


def build_sections(repo_root):
    """Pure presentation specification; never recompute or write scientific data."""
    c = _column
    omega = c("omega", "ω", "d")
    case = c("case_id", "Run", "d")
    physical = SETTINGS + "Table_S1_common_preparation.csv"
    settings = SETTINGS + "Table_S1_recorded_settings.csv"
    s2 = DIAGNOSTICS + "Table_S2_archived.csv"
    s3 = DIAGNOSTICS + "Table_S3_archived.csv"
    s4 = DIAGNOSTICS + "Table_S4_archived.csv"
    s5 = REFINEMENT + "Table_S5_recomputed.csv"
    s6 = TABLES + "controls/T6/full_precision_results.csv"
    recipe = "[native diagnostics reducer](../../../Loaders/centered/diagnostics.py)"
    rounding = "[printed-value reference](refinement/manuscript_reported_rounding_S2_S6.csv)"
    sections = [dict(
        id="table-s1", title="Preparation and numerical settings",
        intro="The eight reference evolutions use the Cartesian spinor ABC and run through T=40. The T=10,20,30,40 statistics come from those same histories. Units are ℏ=m=1; times, lengths and momenta below use the manuscript's nondimensional units. The physical settings and numerical records are configuration evidence.",
        implementation="Centered ghost-point roof closure with a terminal half-weight norm; lower and lateral Dirichlet walls. The stored grid excludes nominal upper endpoints.",
        panels=[
            _panel("Common longitudinal preparation",physical,[c("L_nominal","Nominal L",".0f"),c("source_z","Source zc",".0f"),c("kz","kz",".15g",values={"3.1415926535897931":"π"}),c("kappa","κ",".15g",values={"3.1415926535897931":"π"}),c("sigma_amplitude","Amplitude σ∥",".1f")],1),
            _panel("Common spin and solver settings",physical,[c("theta","θ",".0f"),c("phi","φ",".0f"),c("T_final","T",".0f"),c("terminal_mass_weight","Terminal weight",".1f"),c("gmres_rtol","GMRES rtol",".0e"),c("gmres_atol","atol",".0f")],1),
            _panel("Run-level timestep comparison",settings,[omega,case,c("dt_manuscript","Δt: manuscript",".5f"),c("dt_recorded","Δt: recorded",".5f")],8),
            _panel("Recorded grids and detector positions",settings,[omega,case,c("Nx","Nx=Ny","d"),c("Nz","Nz","d"),c("z_detector_actual","Numerical zD",".8f")],8)],
        dictionary="The initial spinor is (1,0)ᵀ, with θ=φ=0 and kz=κ=π. The transverse prescription is Lx=Ly=10/√ω with trap centers Lx/2,Ly/2. `dt_manuscript` is the separately [transcribed printed vector](settings/Table_S1_manuscript_timesteps.csv); `dt_recorded` is read from each bundled `run.json`. The CSV retains actual stencil spacings, launch norms, source-run links and precision. Coordinates/potential sampling use float32; stored states and matrix assembly use complex64; linear solves and extra spin/branch diagnostics use complex128/float64. GMRES uses Jacobi preconditioning, restart 30 and maxiter 1000. Event probabilities and masses are recorded every accepted step, plane records every 20 steps, and spin/branch records at nominal 0.02 intervals; reductions use actual saved times. These clocks are distinct from animation frame times.",
        interpretation="Five timesteps in the printed vector differ from the retained run records (ω=50,75,100,200,300). Both are shown; the records determine the distributed calculations. The actual roof coordinate is below the nominal L=20.",
        status="Configuration/documentation, verified against bundled run records; no production simulation is claimed. [Summary preparation code](../../../Loaders/centered/supplement_tables_a.py) generates these compact configuration views."),
        dict(id="table-s2",title="Finite-window and early/late statistics",
             intro="T=40 with diagnostic split tcut=3.6. All eight spinor reference rows and both componentwise scalar controls are retained; the scalar rows are excluded from confinement fits.",
             implementation="Mixed-source comparison: centered/half-weight spinor family and legacy one-sided scalar Robin closure with full terminal weight.",
             panels=[
                 _panel("Spinor ABC: detection and means",s2,[omega,c("D40","D40"),c("tau_cond40","Conditional mean"),c("mu_star40","Restricted mean")],8,row_slice=[0,8],expected_total_rows=10),
                 _panel("Spinor ABC: early/late decomposition",s2,[omega,c("tE","tE"),c("wL40","wL"),c("tL40","tL"),c("Delta_tL_cond40","ΔtL,cond")],8,row_slice=[0,8],expected_total_rows=10),
                 _panel("Scalar ABC: detection and means",s2,[omega,c("D40","D40"),c("tau_cond40","Conditional mean"),c("mu_star40","Restricted mean")],2,row_slice=[8,10],expected_total_rows=10),
                 _panel("Scalar ABC: early/late decomposition",s2,[omega,c("tE","tE"),c("wL40","wL"),c("tL40","tL"),c("Delta_tL_cond40","ΔtL,cond")],2,row_slice=[8,10],expected_total_rows=10)],
             dictionary="D40 is the unconditional detected fraction. The conditional mean averages clicks detected by 40; the restricted mean is ∫₀⁴⁰S(t)dt. tE and tL are click-time means conditioned on [0,3.6] and (3.6,40]. wL=DL/D40 is a fraction **among detections**, not an absolute probability. ΔtL,cond=wL(tL−tE) is the late contribution relative to the early conditional mean. The split is a diagnostic decomposition, not an encounter-resolved or strictly return-free interval. All means have time units; probabilities and weights are dimensionless.",
             interpretation="Stronger confinement reduces prompt detection and increases the late conditional weight and mean. The restricted mean also includes survival at T=40. The scalar comparison remains almost unchanged across its two confinements under its own implementation.",
             status=f"Archived full-precision CSV; the eight spinor rows are independently recomputed from bundled accepted-step detector events and separately integrated survival by the {recipe}. Scalar rows are archived derived results only: normalized piecewise-linear **sampled survival**, checked in the source package against full-step histories, with launch norms and one-sided/full-weight provenance recorded [here](diagnostics/scalar_provenance.json). Those native scalar histories are not bundled."),
        dict(id="table-s3",title="Finite-grid boundary diagnostics",
             intro="T=40, all eight reference confinements. The endpoint budget uses native plane-record times; moments and continuation discrepancy use native spin/branch times.",
             implementation="Centered reference family with the half-weight terminal norm; the finite-grid tangential symbol uses centered differences.",
             panels=[
                 _panel("Probe scale, endpoint budget and momentum moment",s3,[omega,c("hz_sqrt_omega","hz√ω"),c("q_omega","qω",".2e"),c("Delta_bud_rel","Δbud,rel",".2e"),c("E_s2","E[s²]")],8),
                 _panel("Covariance and inward continuation",s3,[omega,c("beta_bl_omega","βbl",".6f"),c("Lambda_omega","Λω",".6f"),c("E_2br","E2br",".2e")],8)],
             dictionary="Rh is the magnitude of the centered tangential-difference symbol and s=Rh/√ω. E[s²] is weighted by the time-integrated boundary density; qω=(hz√ω)²E[s²] is the dimensionless probe scale, not an error estimate. Δbud,rel=|∫κNDdt−[P0−P(T)]|/[P0−P(T)] uses the native **plane-record** density quadrature, not the accepted-step event sum. βbl=Cov(t,saω), Λω=Cov(t,Rhaω)=√ωβbl, with aω=−2W†ĴhW/(W†W), set to zero on neutral/zero-density modes. E2br is the boundary-density-weighted relative squared discrepancy between one-grid inward homogeneous continuation and the adjacent recorded plane (Eq. S102), integrating separate linear interpolants only where both endpoints are valid. The CSV keeps the unscaled covariance; βbl has time units and Λω time/length units. Other columns are dimensionless.",
             interpretation="The covariance changes sign across the sampled family, while the scaled momentum moment increases. This auxiliary covariance is neither an empirical fit slope nor a physical absorbing-layer parameter, and it is not a universal predictor of click-time delay. Inward continuation is an auxiliary diagnostic; detailed values remain resolution-sensitive.",
             status=f"Archived full-precision results, recomputed from bundled native moment/diagnostic records by the {recipe}. Reconstructing the underlying Fourier fields or full Duhamel/monitor diagnostics would require unbundled fields. Plane-record accounting, spin/branch integrals, and accepted-step detector events remain separate."),
        dict(id="table-s4",title="Branch fractions and spin polarization",
             intro="T=40. Branch bars and detector polarization denote ratios of integrals over [0,40]; the final survivor polarization is evaluated at the endpoint. See [Fig. S1](../README.md#fig-s1) and [Table S5 refinement](#table-s5).",
             implementation="Centered reference family. Neutral tangential-symbol modes remain in the total branch weight.",
             panels=[_panel("Integrated branch fractions and laboratory-z ensembles",s4,[omega,c("wbar_br_plus40","w+"),c("wbar_br_minus40","w−"),c("wbar_br_neutral40","w0"),c("Pbar_det_z40","Pz,det"),c("P_nc_z40","Pz,survivor")],8)],
             dictionary="w+, w− and w0 are ratios of integrated native boundary-density records, using linear interpolation on their actual times. Pz,det=(Pup−Pdown)/(Pup+Pdown) uses the integrated accepted-step event matrix. Pz,survivor is the z-polarization of the **entire final surviving state** from its diagnostic-time spin matrix. All columns are dimensionless. Independently rounded branch fractions need not sum exactly to one.",
             interpretation="At strong confinement the plus branch dominates the evolved boundary trace, but the detector-associated laboratory-z polarization is small and the final whole-survivor polarization is near one. These are different weightings and spin axes. Unequal evolved branch weights do not imply unequal local branch absorption coefficients at fixed incident wave vector.",
             status=f"Archived full-precision results independently recomputed by the {recipe}. **Printed-value mismatch:** at ω=50 the manuscript shows Pz,survivor=0.999; the retained value 0.99971289084281545 rounds to 1.000 at three decimals. This presentation follows the numerical CSV and retains the mismatch in the {rounding}; it is not silently forced to the paper."),
        dict(id="table-s5",title="Spatial and timestep refinement",
             intro="ω=300, T=40, reference preparation and fixed numerical boundary positions. The four runs are baseline 7100, transverse 7102, longitudinal **7101 with Nz=2999**, and half timestep 7706. The later Nz=4000 result is excluded.",
             implementation="Centered/half-weight family, with longitudinal refinement retaining the original numerical roof and transverse walls.",
             panels=[
                 _panel("Common run settings for panels (a) and (b)",s5,[case,c("refinement","Refinement","text"),c("Nx","Nx=Ny","d"),c("Nz","Nz","d"),c("dt","Δt",".0e")],4),
                 _panel("(a) Detection statistics and integrated plus branch",s5,[case,c("D","D40"),c("mu","μ*(40)"),c("epsilon_mu_refined_denom_percent","εrel (%)"),c("wplus","w+")],4),
                 _panel("(a) Full-window rate discrepancies: [0,40]",s5,[case,c("epsilon_CN_full_percent","εCN (%)"),c("epsilon_branch_full_percent","εbr (%)")],4),
                 _panel("(b) Spin and boundary-weighted diagnostics",s5,[case,c("Pzdet","Pz,det"),c("Pznc","Pz,survivor",".6f"),c("Es2","E[s²]"),c("beta_times_1000","10³βbl")],4),
                 _panel("(b) Late-window rate discrepancies: [8,40]",s5,[case,c("epsilon_CN_late_percent","εCN,late (%)"),c("epsilon_branch_late_percent","εbr,late (%)")],4)],
             dictionary="All ε columns are numeric percentages (for example, 9.580 means 9.580%, not a fraction). εrel=100|μr−μb|/|μr| uses the **refined** restricted mean in its denominator. εCN(W)=100∫W|gCN,r−gCN,b|dt/∫WgCN,bdt uses native event probabilities divided by accepted-step durations and launch norms. εbr sums absolute differences of the +,−,neutral branch rates and divides by the integral of the baseline sampled roof-density rate Jκ,h over the same window. This denominator is distinct from the baseline accepted-step CN probability used for εCN. CN rates are piecewise constant; branch records are piecewise linear. Integrations use the union of time grids and split linear zero crossings, with no smoothing or time shift. The CSV retains raw βbl as well as the displayed 10³βbl scale; full and late windows are explicit columns.",
             interpretation="Integrated plus-branch predominance and the detector/survivor spin contrast persist in these tests. Small restricted-mean differences do not establish full convergence: detailed late rates change substantially under longitudinal and timestep refinement, and auxiliary diagnostics are more sensitive.",
             status=f"Recomputed from bundled native records using the first-pass {recipe}'s Eq. S117 reconstruction; the original final four-run exporter was not located. [Summary preparation](../../../Loaders/centered/supplement_tables_a.py) joins its outputs without changing formulas. The earlier [three-run archive](diagnostics/Table_S5_three_run_archived.csv) and [pairwise archive](diagnostics/Table_S5_pairwise_archived.csv) remain as independent checks. **Printed discrepancies:** manuscript D40 is 0.739/0.742 for baseline/transverse, while native values round to 0.740/0.743; longitudinal εrel is printed 1.823%, while 1.823508241% rounds to 1.824%. The {rounding} keeps all inspected printed values and their display precision; no computed discrepancy is hardcoded."),
        dict(id="table-s6",title="Bulk-spin and chiral-preparation controls",
             intro="T=20, six controls at each of ω=1 and 100. The physical reference preparation is L=20, zc=10, kz=κ=π, amplitude width 0.5 and initial spinor (1,0)ᵀ; the chiral case changes the transverse preparation.",
             implementation="Centered spinor-ABC controls. Statistics follow the original survival-history convention, using every-step stored mass and recorded launch norm.",
             panels=[_panel("All twelve control rows",s6,[c("control","Control","text",values={"Rashba–Dresselhaus":"Rashba–D."}),omega,case,c("D20","D20"),c("tau_cond20","Conditional mean"),c("mu_star20","μ*(20)"),c("DL","DL")],12)],
             dictionary="D20=1−S(20), μ*(20)=∫₀²⁰S(t)dt, and DL=S(3.6)−S(20) is an **absolute** late probability. The conditional mean follows from [μ*(20)−20S(20)]/D20. Means have time units; probabilities are dimensionless. The native-event cross-check columns in the CSV are separate quantities, not substituted for the survival reduction. Control key: Reference Ha=0; Uniform Ha=πσx+(1/2)σz; Radial Ha=0.2√ω(Xσx+Yσy), with X,Y relative to the trap center; Rashba Ha=0.5(σxpy−σypx); Rashba–D. adds 0.4(σxpy−σypx)+0.3(σxpx−σypy); Chiral has Ha=0 and transverse profile (χ10+iχ01)/√2.",
             interpretation="The tested confinement contrast persists for these bulk additions and this excited preparation. Its magnitude depends on the dynamics and initial state; two confinement values do not establish a universal law or invariance under arbitrary spin couplings or preparations.",
             status="Archived derived numerical results, displayed from the existing full-precision CSV. The [historical reducer](../../../Solvers/centered/controls/analysis/T6/recalculate_s8_1.py) documents the every-step survival recipe, but its full control histories are not bundled; coarse sampled norms are not substituted. Historical source comments refer to an earlier manuscript snapshot; this presentation's printed-value comparison uses the supplied 28 October 2026 PDF (page 41).")]
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prepare", action="store_true", help="explicitly regenerate canonical settings and S5 summaries")
    parser.add_argument("--check", action="store_true", help="check compact summaries against bundled native/configuration records")
    parser.add_argument("--diagnostics-output", type=Path, help="reuse outputs from an existing diagnostics group reduction")
    parser.add_argument("--comparison-output", type=Path, help="write printed-vs-numerical evidence as JSON")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    if args.check:
        checks = summary_checks(args.root, args.diagnostics_output)
        print(json.dumps(checks, indent=2))
        if not all(c["passed"] for c in checks):
            raise SystemExit(1)
    if args.comparison_output:
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_output.write_text(json.dumps(manuscript_comparisons(args.root), indent=2) + "\n")
    if not (args.prepare or args.check or args.comparison_output):
        parser.error("choose --prepare, --check or --comparison-output")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute the separate legacy scalar-control S2 block without propagation.

Uses the manuscript's normalized sampled-survival convention. Full-step native
norms and detector events are independently cross-checked, never substituted for
the sampled convention. Source scripts are read and hashed, never imported.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
BASE = PROJECT / "2_FirstAcuuracy_Scripts_Run/R4_ConvergentTest_Resolved_Boundary/Archive"
NAMES = {
    1: "5000.Spin_Gauss_DirichletABC_theta=0_ScalarABC_Omega=1_L=20_zB=0_zc=10_k0=pi_kappa=pi_sigma=0.5_Nz=1500_dt=0.001_T=40_ZIB_R4_v1",
    300: "5001.Spin_Gauss_DirichletABC_theta=0_ScalarABC_Omega=300_L=20_zB=0_zc=10_k0=pi_kappa=pi_sigma=0.5_Nz=1500_dt=0.0002_T=40_ZIB_R4_v1",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integral(t, s, left, right):
    assert t[0] <= left < right <= t[-1]
    x = np.r_[left, t[(t > left) & (t < right)], right]
    return float(np.trapezoid(np.interp(x, t, s), x))


def stats(t, raw, p0):
    assert t.ndim == raw.ndim == 1 and len(t) == len(raw)
    assert np.all(np.isfinite(t)) and np.all(np.isfinite(raw))
    assert np.all(np.diff(t) > 0) and t[0] == 0.0 and t[-1] == 40.0
    assert raw[0] == p0 and p0 > 0.0
    s = raw / p0
    sc, sf = np.interp([3.6, 40.0], t, s)
    mu_e = integral(t, s, 0.0, 3.6)
    mu = integral(t, s, 0.0, 40.0)
    de, dl = 1.0 - sc, sc - sf
    me = mu_e - 3.6 * sc
    ml = integral(t, s, 3.6, 40.0) + 3.6 * sc - 40.0 * sf
    d, m = de + dl, me + ml
    te, tl = me / de, ml / dl
    return {"D40": float(d), "tau_cond40": float(m/d), "mu_star40": mu,
            "tE": float(te), "wL40": float(dl/d), "tL40": float(tl),
            "Delta_tL_cond40": float(dl/d*(tl-te)), "S40": float(sf),
            "D_E": float(de), "D_L": float(dl), "M_E": float(me), "M_L": float(ml),
            "D40_plus_S40_minus_1": float(d+sf-1),
            "restricted_mean_identity_residual": float(mu-(m+40*(1-d))),
            "early_late_identity_residual": float(m/d-(te+dl/d*(tl-te))),
            "survival_increase_count": int(np.count_nonzero(np.diff(s) > 0)),
            "maximum_survival_increase": float(np.max(np.diff(s))),
            "minimum_survival": float(np.min(s)),
            "maximum_survival_above_one": float(np.max(s)-1),
            "survival_node_count": int(len(t))}


def main():
    # A linear survival function is integrated analytically, including split nodes.
    tt = np.array([0., 2., 5.]); ss = 1.-0.1*tt
    assert abs(integral(tt, ss, 0.7, 3.6) - ((3.6-.05*3.6**2)-(.7-.05*.7**2))) < 1e-14
    result, manifest = [], []
    for omega, name in NAMES.items():
        d = BASE / name
        meta = json.loads((d/"run_metadata.json").read_text())
        geom = json.loads((d/"geometry_audit.json").read_text())
        operator = json.loads((d/"operator_audit.json").read_text())
        summary = json.loads((d/"summary.json").read_text())
        source = (d/"script_used.py").read_text()
        source_hash = sha(d/"script_used.py")
        assert source_hash == meta["script_sha256"]
        with np.load(d/"constants.npz", allow_pickle=False) as const:
            assert float(const["omega"]) == omega == meta["omega"] == summary["omega"]
            assert int(const["run_id"]) == meta["run_id"] == summary["run_id"]
            for key in ("Nx", "Ny", "Nz", "dt"):
                assert float(const[key]) == float(meta[key])
            assert float(const["z_det"]) == meta["z_detector_actual"] == geom["z_detector_actual"]
            assert float(const["ABC_ALPHA"]) == meta["boundary_alpha"] == 0.0
            assert str(const["source_sha256"]) == meta["source_sha256"]
        assert summary["completed"] and meta["T_requested"] == meta["T_stepped"] == 40.0
        assert meta["resume_checkpoint"] is None
        assert operator["passed"] and operator["inner_product"] == "uniform hx*hy*hz, as in supplied source"
        assert "This is NOT the standard centered-ghost row with a half-weight endpoint" in source
        assert "data += [-0.5 * inv, 0.5 * inv - 1j * k_bc / (2.0 * dz)]" in source
        assert "Lap_spinor = (Lap_diag + ABC_ALPHA*C).tocsr()" in source
        assert "PASS_R4_CARTESIAN_COMPLETED" in (d/"stdout.txt").read_text()
        t = np.asarray(np.load(d/"prob_times.npy"), dtype=np.float64)
        raw = np.asarray(np.load(d/"total_probs.npy"), dtype=np.float64)
        p0 = float(meta["initial_norm_squared"])
        row = {"omega": omega, "block": "legacy_scalar_ABC", **stats(t, raw, p0)}
        with np.load(d/"field_history.npz", allow_pickle=False) as hist:
            ft = np.asarray(hist["time"], dtype=np.float64)
            fp = np.asarray(hist["norm_squared"], dtype=np.float64)
            assert len(ft) == meta["num_steps"] + 1
            assert np.allclose(np.diff(ft), meta["dt"], rtol=0, atol=1e-14)
            full = stats(ft, fp, p0)
            sample_mass_error = float(np.max(np.abs(np.interp(t, ft, fp)-raw)))
            assert sample_mass_error < 2e-15
            assert np.max(np.abs(hist["survival"]-fp/p0)) < 2e-15
            # These flux arrays are already divided by S0 in script_used.py.
            events = np.asarray(hist["roof_flux_midpoint"], dtype=np.float64)*np.diff(ft)
            cum = np.r_[0., np.cumsum(events)]
            budget = cum + fp/p0 - 1.
            trace = np.load(d/"branch_spin_history.npz", allow_pickle=False)
            spin_keys = list(trace.files)
            event_matrix = np.asarray(trace["event_spin_probability"], dtype=np.complex128)
            event_trace = np.trace(event_matrix, axis1=1, axis2=2).real
            row["event_matrix_trace_vs_roof_probability_max_abs"] = float(np.max(np.abs(event_trace-events)))
            row["event_matrix_minimum_eigenvalue"] = float(np.min(np.linalg.eigvalsh(event_matrix)))
            trace.close()
            with np.load(d/"longitudinal_history.npz", allow_pickle=False) as lng:
                q = np.asarray(lng["survivor_spin_matrix"], dtype=np.complex128)
                qtrace = np.trace(q, axis1=1, axis2=2).real
                qexpected = fp[np.asarray(lng["profile_step"], dtype=int)]/p0
                row["survivor_matrix_trace_vs_mass_max_abs"] = float(np.max(np.abs(qtrace-qexpected)))
                row["survivor_matrix_minimum_eigenvalue"] = float(np.min(np.linalg.eigvalsh(q)))
                assert float(lng["z"][-1]) == meta["z_detector_actual"]
            event_moment = float(np.sum(events*(ft[:-1]+ft[1:])/2.))
            row["native_event_independent_restricted_mean_residual_sampled_survival"] = float(row["mu_star40"]-(event_moment+40*(1-cum[-1])))
            row["native_event_independent_restricted_mean_residual_full_step_survival"] = float(full["mu_star40"]-(event_moment+40*(1-cum[-1])))
            row["scalar_survival_identity_note"] = "D and moments reconstructed from the same linear survival interpolant by manuscript convention, so their identity residual is algebraic, not independent detector validation. The separately reported native-event residual is independent. No records are altered."
            row["full_step_crosscheck"] = full
            row["full_step_minus_manuscript_sampled_mu_star40"] = full["mu_star40"]-row["mu_star40"]
            row["full_step_minus_manuscript_sampled_tE"] = full["tE"]-row["tE"]
            row["native_event_D40"] = float(cum[-1])
            row["native_event_D40_plus_S40_minus_1"] = float(budget[-1])
            row["native_maximum_cumulative_event_survival_discrepancy"] = float(np.max(np.abs(budget)))
            row["sampled_vs_full_step_recorded_mass_max_abs"] = sample_mass_error
            row["summary_mu_vs_full_step_abs_difference"] = abs(float(summary["mu_star_T"])-full["mu_star40"])
            row["branch_spin_history_keys"] = spin_keys
        result.append(row)
        inputs = ["run_metadata.json", "geometry_audit.json", "operator_audit.json", "constants.npz",
                  "script_used.py", "prob_times.npy", "total_probs.npy", "field_history.npz",
                  "branch_spin_history.npz", "longitudinal_history.npz", "summary.json", "stdout.txt"]
        manifest.append({"omega": omega, "case_id": str(meta["run_id"]), "run_id": meta["run_id"],
                         "job_id": None, "job_id_note": "Not recorded in retained metadata/stdout.",
                         "directory": str(d), "source_hash": source_hash, "source_hash_algorithm": "sha256",
                         "parent_source_hash": meta["source_sha256"], "P0": p0,
                         "closure": "legacy scalar Robin last row; NOT centered-ghost",
                         "endpoint_mass_weight": 1.0, "volume_metric": "uniform hx*hy*hz",
                         "z_detector_actual": meta["z_detector_actual"],
                         "grid": {k: meta[k] for k in ["Nx", "Ny", "Nz", "hx_actual", "hy_actual", "hz_actual"]},
                         "dt": meta["dt"], "wavefunction_dtype": meta["wavefunction_dtype"],
                         "field_matrices": meta["field_matrices"], "linear_solve_dtype": meta["linear_solve_dtype"],
                         "preparation": {k: meta[k] for k in ["theta", "phi", "z0", "sigma_z_amplitude", "k0", "kappa"]},
                         "status": "verified complete through T=40; native full-step and sampled histories agree at common nodes",
                         "statistics_convention": "Manuscript normalized piecewise-linear sampled survival: prob_times.npy/total_probs.npy. Full-step records used as independent cross-check.",
                         "legacy_metadata_note": "revision_run.json, branch_refinement_case.json, revision_coordinates.npz are not this legacy campaign's schema. Identity checked against run_metadata, constants, geometry/operator audits, script hash, full-step raw history and completion record.",
                         "inputs": {f: {"path": str(d/f), "sha256": sha(d/f)} for f in inputs}})
    (OUT/"reference_scalar_stats.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (OUT/"reference_scalar_provenance.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    cols = ["omega", "D40", "tau_cond40", "mu_star40", "tE", "wL40", "tL40", "Delta_tL_cond40"]
    with (OUT/"reference_scalar_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader(); writer.writerows(result)
    print(json.dumps([{k:r[k] for k in cols} for r in result], indent=2))


if __name__ == "__main__":
    main()

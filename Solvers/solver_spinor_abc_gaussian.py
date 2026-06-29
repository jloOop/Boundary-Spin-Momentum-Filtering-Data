#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main production solver: spinor-ABC confinement sweep / restricted-mean data.

Repository role
---------------
This script is the representative time-dependent Pauli-spinor solver used for
main spinor-ABC confinement runs.  It evolves a Gaussian wave packet in the
harmonic waveguide with the spin-coupled absorbing boundary at the roof.

The script intentionally keeps the full research run light: no Bohmian trajectory
sampling and no dense density snapshots are written by default.  It stores the
survival/norm history, from which the paper-level detector observables are
post-processed:

    S(t; omega)      = ||Psi_t||^2
    g(t; omega)      = - dS(t; omega) / dt
    D_T(omega)       = 1 - S(T)
    mu*(T; omega)    = integral_0^T S(t; omega) dt

These are the quantities behind the confinement sweep and the finite-window
restricted-mean detection-time fit.

Physics implemented
-------------------
Bulk Hamiltonian:
    H = -1/2 Delta + 1/2 omega^2 [(x-Lx/2)^2 + (y-Ly/2)^2]

Boundary conditions:
    - Dirichlet on sidewalls and bottom.
    - Spinor absorbing boundary at the roof, implemented through the roof-layer
      finite-difference spinor coupling matrix C.

Numerics
--------
    - Cartesian finite differences.
    - Crank--Nicolson time stepping.
    - CuPy sparse matrices on GPU.
    - GMRES/Krylov solve with diagonal preconditioner.

Important reader note
---------------------
The executable solver statements below are left as in the working research
script.  The comments are added for GitHub readability and to connect the code
to the notation used in the paper and Supplemental Material.
"""

import sys, time, os, logging
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags, eye, kron, coo_matrix, csr_matrix, bmat
from cupyx.scipy.sparse.linalg import gmres, LinearOperator

try:
    from Solvers.solver_bookkeeping import build_probability_history_arrays, build_time_series_bookkeeping
    from Solvers.solver_metadata import build_solver_constants
except ImportError:
    from solver_bookkeeping import build_probability_history_arrays, build_time_series_bookkeeping
    from solver_metadata import build_solver_constants



# =============================================================================
# Run configuration and output directory
# =============================================================================
# OMEGA can be supplied by the shell for parameter sweeps, e.g.
#     OMEGA=300 OUTDIR=/scratch/$USER/spinor_runs python 746SpinorXY...
# The directory name is intentionally descriptive so that HPC outputs can be
# identified without opening the log files.
# -------------------- directories / logging --------------------
OMEGA_ENV = os.getenv("OMEGA", None)
omega = float(OMEGA_ENV) if (OMEGA_ENV is not None) else float(300.0)

backend = "CuPy"
print(f"[info] backend: {backend}")

run_name = os.getenv(
    "RUN_NAME",
    f"2557.Spin_Gauss_DirichletABC_theta=0+C_Omega={omega:g}_L=20_sigma=0.5_ConvergentNz",
)

out_dir = Path(os.getenv("OUTDIR", ".")) / run_name
out_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(out_dir / "simulation_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ── tee stdout → log file, like the cylindrical code ───────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, open(out_dir / "stdout.txt", "w"))


# -------------------- utilities --------------------
def to_cpu(arr):
    return cp.asnumpy(arr)


# =============================================================================
# Physical and numerical parameters
# =============================================================================
# The transverse box scales as Lx=Ly=ell/sqrt(omega), keeping the wall distance
# and transverse resolution fixed in oscillator units.  This is essential for
# the confinement sweep: the observed omega-dependence should not come from
# lateral-wall artifacts or changing transverse resolution.
# -------------------- parameters (keep yours) --------------------
DTYPE_R, DTYPE_C = cp.float32, cp.complex64

Nx = 100
Ny = 100
Nz = 2000
ell = 10.0

Lx = ell / np.sqrt(omega)
Ly = ell / np.sqrt(omega)
Lz = 20.0

dt = float(2e-4)
T_final = 20.0
#SNAPSHOT_DT = 0.1  # save snapshots every 0.05

Vxy0 = omega**2
V0z = 0.0
alpha = 0.0
k_bc = float(cp.pi)
# --- Gaussian packet parameters (choose these) ---
z0      = 10.0          # initial center in z (must be > a few sigma_z from z=0)
sigma_z = 0.5         # longitudinal width (in your dimensionless z-units)
k0      = float(cp.pi) # mean momentum along +z; adjust as needed
#-----Spinor parameter---------
theta = float(0.0)
phi   = np.float32(0.0)


# =============================================================================
# Mesh and factorized initial spinor
# =============================================================================
# The initial state is a transverse harmonic-ground-state profile times a
# right-moving longitudinal Gaussian, with a constant spinor (theta, phi).
# -------------------- mesh / initial state (keep yours) --------------------
x = cp.linspace(0.0, Lx, Nx, endpoint=False, dtype=DTYPE_R)
y = cp.linspace(0.0, Ly, Ny, endpoint=False, dtype=DTYPE_R)
z = cp.linspace(0.0, Lz, Nz, endpoint=False, dtype=DTYPE_R)



hx = float(x[1] - x[0])
hy = float(y[1] - y[0])
hz = float(z[1] - z[0])

# -------------------- Gaussian initial condition (recommended) --------------------
# Transverse HO ground state centered at (Lx/2, Ly/2) with width ~ 1/sqrt(omega)
# Longitudinal Gaussian centered at z0 with width sigma_z and mean momentum k0 > 0



# Optional: avoid putting appreciable weight too close to the roof
# (not necessary if z0 is far enough below Lz-hz)

# --- build mesh ---
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

# --- transverse part (HO ground state) ---
Rho2 = (X - Lx/2)**2 + (Y - Ly/2)**2
psi_xy = cp.exp(-omega * Rho2 / 2)  # unnormalized

# --- longitudinal Gaussian with plane-wave phase ---
# NOTE: use (Z - z0), not a mask. Keep it smooth.
psi_z = cp.exp(-(Z - z0)**2 / (2*sigma_z**2)) * cp.exp(1j * k0 * (Z - z0))

# --- scalar spatial wavefunction ---
psi_scalar = (psi_xy * psi_z).astype(DTYPE_C)

# --- spinor coefficients ---
c_up   = cp.cos(theta/2)
c_down = cp.sin(theta/2) * cp.exp(1j * phi)

psi_up   = (c_up   * psi_scalar).astype(DTYPE_C)
psi_down = (c_down * psi_scalar).astype(DTYPE_C)



# --- enforce Dirichlet BCs on sidewalls and bottom ---
psi_up[[0, -1], :, :] = 0
psi_up[:, [0, -1], :] = 0
psi_up[:, :, 0] = 0
psi_down[[0, -1], :, :] = 0
psi_down[:, [0, -1], :] = 0
psi_down[:, :, 0] = 0

# --- normalize full spinor ---
def normalize_wavefunction(psi_up, psi_down, hx, hy, hz):
    total_prob = cp.sum(cp.abs(psi_up)**2 + cp.abs(psi_down)**2) * (hx*hy*hz)
    norm = cp.sqrt(total_prob)
    psi_up   /= norm
    psi_down /= norm
    return psi_up, psi_down

psi_up, psi_down = normalize_wavefunction(psi_up, psi_down, hx, hy, hz)
psi_flat = cp.concatenate((psi_up.ravel(), psi_down.ravel())).astype(DTYPE_C)

# (optional) free memory if you won't reuse X,Y,Z
# del X, Y, Z, Rho2, psi_xy, psi_z, psi_scalar
# cp.get_default_memory_pool().free_all_blocks()



rho0 = cp.abs(psi_up)**2 + cp.abs(psi_down)**2
print("rho at bottom plane max =", float(cp.max(rho0[:, :, 0])))
print("rho at sidewall x=0 max =", float(cp.max(rho0[0, :, :])))
print("rho at sidewall y=0 max =", float(cp.max(rho0[:, 0, :])))
print("rho at detector plane initial max =", float(cp.max(rho0[:, :, -1])))



# Free memory: large construction-only arrays (correct for Gaussian init)
del X, Y, Z, Rho2, psi_xy, psi_z, psi_scalar, rho0
cp.get_default_memory_pool().free_all_blocks()


# =============================================================================
# Bulk potential
# =============================================================================
# There is no CAP in this script.  All detector back-action comes from the
# spinor absorbing boundary at the roof, not from an imaginary bulk potential.
# -------------------- potentials (keep yours; no absorber) --------------------
# Harmonic oscillator potential in x,y (built originally from X,Y, but you deleted them; rebuild minimal)
# Rebuild X,Y on the fly for V only (small overhead once)
Xv = (x - Lx/2)**2                      # (Nx,)
Yv = (y - Ly/2)**2                      # (Ny,)
V_xy = 0.5 * Vxy0 * (Xv[:,None] + Yv[None,:])   # (Nx,Ny)
V_real = V_xy[:,:,None]                 # broadcast to (Nx,Ny,Nz) when used


V_abs  = 0.0
V = V_real.astype(DTYPE_C) + V_abs
V_diag = cp.repeat(V_xy.ravel(), Nz).astype(DTYPE_C)  # (Nx*Ny*Nz,)
V_full = cp.concatenate((V_diag, V_diag))


constants = build_solver_constants(
    Nx=Nx,
    Ny=Ny,
    Nz=Nz,
    Lx=Lx,
    Ly=Ly,
    Lz=Lz,
    hx=hx,
    hy=hy,
    hz=hz,
    dt=dt,
    T_final=T_final,
    omega=omega,
    Vxy0=Vxy0,
    V0z=V0z,
    alpha=alpha,
    k_bc=k_bc,
    theta=theta,
    phi=phi,
    z0=z0,
    sigma_z=sigma_z,
    k0=k0,
)
np.savez(out_dir/"constants.npz", **constants)
logging.info("[info] Saved coordinate grids and constants")

del Xv, Yv, V_real, V_abs, V
cp.get_default_memory_pool().free_all_blocks()


#Print all parameters to stdout (which is teed to stdout.txt)
print("Simulation Parameters:")
print(f"Nx: {Nx}")
print(f"Ny: {Ny}")
print(f"Nz: {Nz}")
print(f"theta: {theta}")
print(f"Lx: {Lx}")
print(f"Ly: {Ly}")
print(f"Lz: {Lz}")
print(f"k_bc: {k_bc}")
print(f"dt: {dt}")
print(f"T_final: {T_final}")
print(f"V0z: {V0z}")
print(f"alpha: {alpha}")
print(f"DTYPE_R: {DTYPE_R}")
print(f"DTYPE_C: {DTYPE_C}")
print(f"omega: {omega}")
print(f"hx: {hx}")
print(f"hy: {hy}")
print(f"hz: {hz}")
print(f"out_dir: {out_dir}")
#print(f"SNAPSHOT_DT: {SNAPSHOT_DT}")
print(f"k0: {k0}")
print(f"z0: {z0}")
print(f"sigma_z: {sigma_z}")

print("-------------------------------------------")




# =============================================================================
# Sparse finite-difference Hamiltonian assembly
# =============================================================================
# L_z implements the z kinetic term with bottom Dirichlet and roof Robin/ABC row.
# L_dirichlet implements the transverse sidewall Dirichlet kinetic terms.
# The spinor part is introduced below by adding the roof-layer coupling C.
# -------------------- operators (UNCHANGED from your solver) --------------------
def L_z(N_z, dz, k_bc):
    """
    Discretizes H = -(1/2) d^2/dz^2 with:
      z=0 : Dirichlet
      z=L : Robin ABC (∂z ψ + i k_bc ψ = 0)
    Returns CSR matrix.
    """
    inv = 1.0 / (dz * dz)
    data, indices, indptr = [], [], [0]
    for i in range(N_z):
        if i == 0:
            indptr.append(len(indices))
        elif i == N_z - 1:
            indices += [i - 1, i]
            data += [-0.5 * inv, 0.5 * inv - 1j * k_bc / (2.0 * dz)]
            indptr.append(len(indices))
        else:
            left = i - 1
            right = i + 1
            if left == 0:
                indices += [i, right]
                data += [inv, -0.5 * inv]
                indptr.append(len(indices))
            else:
                indices += [left, i, right]
                data += [-0.5 * inv, inv, -0.5 * inv]
                indptr.append(len(indices))
    data = cp.array(data, dtype=DTYPE_C)
    indices = cp.array(indices, dtype=cp.int32)
    indptr = cp.array(indptr, dtype=cp.int32)
    return csr_matrix((data, indices, indptr), shape=(N_z, N_z))

def L_dirichlet(N, d):
    inv_d2 = 1 / d**2
    half_inv_d2 = 0.5 * inv_d2
    data, indices, indptr = [], [], [0]
    for i in range(N):
        if i == 0 or i == N - 1:
            indptr.append(indptr[-1])
        else:
            left = i-1
            right = i+1
            if left == 0:
                indices.extend([i, right])
                data.extend([inv_d2, -half_inv_d2])
                indptr.append(indptr[-1] + 2)
            elif right == N-1:
                indices.extend([left, i])
                data.extend([-half_inv_d2, inv_d2])
                indptr.append(indptr[-1] + 2)
            else:
                indices.extend([left, i, right])
                data.extend([-half_inv_d2, inv_d2, -half_inv_d2])
                indptr.append(indptr[-1] + 3)
    data = cp.array(data, dtype=DTYPE_C)
    indices = cp.array(indices, dtype=cp.int32)
    indptr = cp.array(indptr, dtype=cp.int32)
    return csr_matrix((data, indices, indptr), shape=(N, N))

L1 = L_z(Nz, hz, k_bc)
L2 = L_dirichlet(Nx, hx)
L3 = L_dirichlet(Ny, hy)

Ix = eye(Nx, dtype=DTYPE_C, format='csr')
Iy = eye(Ny, dtype=DTYPE_C, format='csr')
Iz = eye(Nz, dtype=DTYPE_C, format='csr')

Lap_scalar = (kron(L2, kron(Iy, Iz)) + kron(Ix, kron(L3, Iz)) + kron(Ix, kron(Iy, L1))).tocsr()
Lap_diag = kron(eye(2, dtype=DTYPE_C, format='csr'), Lap_scalar)


# =============================================================================
# Spinor absorbing-boundary coupling at the roof
# =============================================================================
# This is the physically central block.  It discretizes the tangential derivative
# terms in
#     d_z psi_up   = i*kappa psi_up   - (d_x - i d_y) psi_down,
#     d_z psi_down = i*kappa psi_down + (d_x + i d_y) psi_up,
# only on the top z-layer.  These off-diagonal roof entries are what make the
# boundary a spin--momentum filter rather than a scalar sink.
# Spinor ABC coupling matrix C (UNCHANGED)
rows_list, cols_list, data_list = [], [], []
coef = 0.5 / hz
d_coef_x = 1.0 / (2.0 * hx)
d_coef_y = 1.0 / (2.0 * hy)
Ngrid = Nx * Ny * Nz

for ix in range(1, Nx-1):
    for iy in range(1, Ny-1):
        z_top = Nz - 1
        base = ix * Ny * Nz + iy * Nz + z_top

        # UP row
        row_up = base
        if ix > 1:
            col = Ngrid + (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (-d_coef_x))
        if ix < Nx - 2:
            col = Ngrid + (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (+d_coef_x))
        if iy > 1:
            col = Ngrid + ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 2:
            col = Ngrid + ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_up); cols_list.append(col); data_list.append(coef * (-1j) * d_coef_y)

        # DOWN row
        row_down = Ngrid + base
        if ix > 1:
            col = (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (+d_coef_x))
        if ix < Nx - 2:
            col = (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (-d_coef_x))
        if iy > 1:
            col = ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 2:
            col = ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_down); cols_list.append(col); data_list.append(coef * (-1j) * d_coef_y)


C = coo_matrix(
    (cp.array(data_list, dtype=DTYPE_C), (cp.array(rows_list), cp.array(cols_list))),
    shape=(2 * Ngrid, 2 * Ngrid), dtype=DTYPE_C
).tocsr()

Lap_spinor = (Lap_diag + C).tocsr()




Potential = diags(V_full, 0, format="csr")




# =============================================================================
# Crank--Nicolson matrices
# =============================================================================
# One time step solves
#     (I + i dt H/2) psi_{n+1} = (I - i dt H/2) psi_n.
# The non-Hermitian boundary makes GMRES appropriate.
# CN matrices (UNCHANGED)
Ntot = 2 * Nx * Ny * Nz
Id = eye(Ntot, dtype=DTYPE_C, format='csr')
Potential = diags(V_full, 0, format='csr')

A = Id + 1j*dt/2 * (Lap_spinor + Potential).tocsr()
B = Id - 1j*dt/2 * (Lap_spinor + Potential).tocsr()

# GMRES preconditioner (UNCHANGED idea)
A64 = A.astype(cp.complex128)
B64 = B.astype(cp.complex128)
inv_diag64 = 1.0 / A64.diagonal()
M64 = LinearOperator(shape=A64.shape, matvec=lambda x: inv_diag64 * x)


# ================================================================
#  OPTIMISED gradient routines (slicing, no cp.roll)
# ================================================================


# #--- define physical snapshot times, independent of dt --------------------------------------------------- 
# snapshot_times = np.arange(SNAPSHOT_DT, T_final + 1e-9, SNAPSHOT_DT)

# # map from time-step index n to the *nominal* snapshot time τ
# snapshot_map = {
#     int(round(t_snap / dt)): float(t_snap)
#     for t_snap in snapshot_times
# }
# snapshot_steps = set(snapshot_map.keys())

# print("[info] snapshot times:", snapshot_times)
# print("[info] snapshot steps:", sorted(list(snapshot_steps))[:10], "...")




# =============================================================================
# Time-series storage
# =============================================================================
# total_probs and prob_times are the main outputs of this production solver.
# They are sufficient to reconstruct S(t), g(t), D_T, and mu*(T).
# ── simulation bookkeeping ──────────────────────────────────────---------------------------------------------- 
num_steps, prob_steps_cpu, view_steps_cpu, total_probs, prob_times = build_time_series_bookkeeping(
    T_final,
    dt,
)
# Include the known initial survival point.
# This makes prob_times.npy / total_probs.npy a self-contained S(t) curve.
initial_total_prob = float(
    to_cpu(cp.sum(cp.abs(psi_up)**2 + cp.abs(psi_down)**2) * (hx * hy * hz))
)
prob_times.append(0.0)
total_probs.append(initial_total_prob)
logging.info(f"[Probability] t=0.00, ∫|ψ|² dV = {initial_total_prob:.6f}")



# ---------- detector on-grid ----------
z_det = float(cp.asnumpy(z[-1]))   # ON-GRID: last z node (not Lz)
DET_EPS = float(min(1e-6, 0.1*hz)) # tiny cushion below the plane



# =============================================================================
# Main CN/GMRES evolution loop
# =============================================================================
# Each step advances the full two-component spinor, reapplies the hard walls,
# and periodically records the remaining norm.  The norm loss is the detector
# click-time density after post-processing.
# ── main time loop ─────────────────────────────────────────────────
print(f"[info] Starting time loop – {num_steps} steps")
start_time = time.time()
for n in range(1, num_steps+1):
    # progress every 100 steps
    if n % 100 == 0:    
        elapsed = time.time()-start_time
        eta = (num_steps-n)*(elapsed/n)/60
        print(f"[progress] Step {n}/{num_steps} (t={n*dt:.2f}) | ETA: {eta:.2f} min", flush=True)
    # Crank–Nicolson advance (UNCHANGED)
    rhs64 = (B64 @ psi_flat.astype(cp.complex128))
    psi64, info = gmres(
    A64, rhs64,
    x0=psi_flat.astype(cp.complex128),
    rtol=1e-8,
    atol=0.0,          # recommended: control accuracy via rtol
    maxiter=1000,
    M=M64,
    restart=30,
)
    psi_flat = psi64.astype(cp.complex64)
    psi_3d = psi_flat.reshape(2, Nx, Ny, Nz)
    psi_up = psi_3d[0]
    psi_down = psi_3d[1]
    
    # Enforce Dirichlet BCs after solve (UNCHANGED)
    psi_up[[0, -1], :, :] = 0
    psi_up[:, [0, -1], :] = 0
    psi_up[:, :, 0] = 0
    psi_down[[0, -1], :, :] = 0
    psi_down[:, [0, -1], :] = 0
    psi_down[:, :, 0] = 0
        

    # ── diagnostics / I-O cadence (UNCHANGED) ────────────────────────────
    if (n % 100 == 0) or (n in prob_steps_cpu):
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)*(hx*hy*hz)))
        total_probs.append(total_prob)
        prob_times.append(n*dt)
        logging.info(f"[Probability] t={n*dt:.2f}, ∫|ψ|² dV = {total_prob:.6f}")
        # save to disk (overwrite each time)
        prob_times_array, total_probs_array = build_probability_history_arrays(prob_times, total_probs)
        np.save(out_dir/"prob_times.npy", prob_times_array)
        np.save(out_dir/"total_probs.npy", total_probs_array)
        
        
    
    # # (2) density snapshots at fixed physical times τ = SNAPSHOT_DT, 2*SNAPSHOT_DT, ...
    # if n in snapshot_steps:
    #     tau_snap = snapshot_map[n]  # nominal snapshot time
    #     rho_prob = to_cpu(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)
    #     # filename uses the *nominal* snapshot time with 5 decimals
    #     np.save(out_dir / f"rho_prob_t{tau_snap:.5f}.npy", rho_prob)


       
    # optional live-view at sparse intervals (unchanged numerics)
    if n in view_steps_cpu:
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)*(hx*hy*hz)))
        print(f"[Probability] t={n*dt:.2f}, Total Probability = {total_prob:.6f}", flush=True)
    
    

# =============================================================================
# Final output flush
# =============================================================================
# The final arrays are intentionally compact and GitHub/HPC-friendly.
# ── final flush (UNCHANGED) ────────────────────────────────────────────────────
prob_times_array, total_probs_array = build_probability_history_arrays(prob_times, total_probs)
np.save(out_dir/"prob_times.npy", prob_times_array)
np.save(out_dir/"total_probs.npy", total_probs_array)
#np.save(out_dir/"bohm_t_hit.npy",        cp.asnumpy(t_hit))
#np.save(out_dir/"bohm_arrived_mask.npy", cp.asnumpy(arrived))
# save only the selected trajectories (compact)

elapsed = time.time()-start_time
print(f"Total execution time: {elapsed:.2f} s", flush=True)
print(f"[info] All output files are in: {out_dir.resolve()}")
print(f"[info] Log saved to {out_dir/'simulation_log.txt'}")

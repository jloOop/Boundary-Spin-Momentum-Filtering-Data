#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary-symbol, covariance, and Duhamel diagnostic solver for spinor ABC.

Repository role
---------------
This is the representative Supplemental-Material diagnostic script for the
finite-grid boundary-symbol test.  It is mainly connected to the two-branch
analysis and Duhamel/finite-epsilon checks discussed in the Supplemental
Material, especially the boundary-symbol diagnostic section.

The script evolves the same spinor-ABC TDSE system and, at diagnostic times,
computes quantities such as

    beta_omega = Cov_{nu0}(t, s a_omega),
    sqrt(omega) * beta_omega,
    exact finite-epsilon boundary-layer densities,
    Taylor consistency errors,
    detector-budget checks,
    full two-branch Duhamel/slab errors.

Central idea
------------
The spinor absorbing boundary has tangential branches i*kappa +/- |xi|.  In the
harmonic guide, |xi| ~ sqrt(omega), so this script checks the finite-grid version
of that local boundary-symbol mechanism.

Important reader note
---------------------
The TDSE/CN/GMRES evolution and spinor-ABC matrix assembly are left as in the
working diagnostic solver.  Added comments identify the mathematical purpose of
each diagnostic block for readers of the GitHub repository.
"""

import sys
import time
import os
import logging
import json
from pathlib import Path

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags, eye, kron, coo_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import gmres, LinearOperator



# =============================================================================
# Utility helpers
# =============================================================================
# Environment-variable helpers make this script easier to use in Slurm arrays
# without editing the source file for each omega/grid/time-step run.
# =============================================================================
# Utilities
# =============================================================================


def to_cpu(arr):
    return cp.asnumpy(arr)


def env_float(name, default):
    val = os.getenv(name, None)
    return float(val) if val is not None else float(default)


def env_int(name, default):
    val = os.getenv(name, None)
    return int(val) if val is not None else int(default)


def safe_tag(x):
    return (f"{float(x):.12g}").replace("-", "m").replace(".", "p")



# =============================================================================
# Run parameters
# =============================================================================
# Most important environment variables:
#     OMEGA, NX, NY, NZ, DT, TFINAL, KBC, OUTDIR,
#     DIAG_EVERY, DEPTH_STEPS, BL_EPS_STEPS.
# This lets the same file serve as a reproducible diagnostic script and an HPC
# parameter-sweep worker.
# =============================================================================
# Parameters: core solver values are the same as in 954Reflection_Diag.py unless
# overridden by environment variables.
# =============================================================================

backend = "CuPy"
print(f"[info] backend: {backend}")

DTYPE_R, DTYPE_C = cp.float32, cp.complex64

Nx = env_int("NX", 100)
Ny = env_int("NY", 100)
Nz = env_int("NZ", 1500)

omega = env_float("OMEGA", 200.0)
ell = env_float("ELL", 10.0)

Lx = ell / np.sqrt(omega)
Ly = ell / np.sqrt(omega)
Lz = env_float("LZ", 20.0)

dt = env_float("DT", 25e-5)
T_final = env_float("TFINAL", 20.0)

Vxy0 = omega**2
V0z = 0.0
alpha = 0.0
k_bc = env_float("KBC", np.pi)

z0_init = env_float("Z0_INIT", 10.0)
sigma_z = env_float("SIGMA_Z", 0.5)
k0 = env_float("K0", np.pi)

theta = env_float("THETA", 0.0)
phi = np.float32(env_float("PHI", 0.0))

run_name = (
    f"3053.BoundaryLayerCovariance_spinorABC"
    f"_theta={safe_tag(theta)}_omega={safe_tag(omega)}"
    f"_dt={safe_tag(dt)}"
)
out_root = Path(os.getenv("OUTDIR", "."))
out_dir = out_root / run_name
out_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(out_dir / "simulation_log.txt"),
        logging.StreamHandler(sys.stdout),
    ],
)



# =============================================================================
# Mesh and initial spinor state
# =============================================================================
# The transverse box scales like ell/sqrt(omega), preserving the oscillator-unit
# resolution across confinement runs.  The longitudinal packet is Gaussian.
# =============================================================================
# Mesh / initial state (unchanged)
# =============================================================================

x = cp.linspace(0.0, Lx, Nx, endpoint=False, dtype=DTYPE_R)
y = cp.linspace(0.0, Ly, Ny, endpoint=False, dtype=DTYPE_R)
z = cp.linspace(0.0, Lz, Nz, endpoint=False, dtype=DTYPE_R)
hx = float(x[1] - x[0])
hy = float(y[1] - y[0])
hz = float(z[1] - z[0])

z_cpu = np.asarray(to_cpu(z))
z_roof_idx = Nz - 1          # stored top row, i.e. the actual discrete ABC row
z_roof_in = Nz - 2           # one grid point below the ABC row
z_roof_pos = float(z_cpu[z_roof_idx])

X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
mid_x = Nx // 2
mid_y = Ny // 2
mid_z = Nz // 2

Rho2 = (X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2
psi_xy = cp.exp(-omega * Rho2 / 2)
psi_z = cp.exp(-(Z - z0_init) ** 2 / (2 * sigma_z**2)) * cp.exp(
    1j * k0 * (Z - z0_init)
)
psi_scalar = (psi_xy * psi_z).astype(DTYPE_C)

c_up = cp.cos(theta / 2)
c_down = cp.sin(theta / 2) * cp.exp(1j * phi)

psi_up = c_up * psi_scalar
psi_down = c_down * psi_scalar

psi_up[[0, -1], :, :] = 0
psi_up[:, [0, -1], :] = 0
psi_up[:, :, 0] = 0
psi_down[[0, -1], :, :] = 0
psi_down[:, [0, -1], :] = 0
psi_down[:, :, 0] = 0


def normalize_wavefunction(psi_up, psi_down, hx, hy, hz):
    total_prob = cp.sum(cp.abs(psi_up) ** 2 + cp.abs(psi_down) ** 2) * (hx * hy * hz)
    norm = cp.sqrt(total_prob)
    psi_up /= norm
    psi_down /= norm
    return psi_up, psi_down


psi_up, psi_down = normalize_wavefunction(psi_up, psi_down, hx, hy, hz)
psi_flat = cp.concatenate((psi_up.ravel(), psi_down.ravel())).astype(DTYPE_C)

V_real = 0.5 * Vxy0 * ((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2)
V_abs = 0.0
V = V_real.astype(DTYPE_C) + V_abs
V_diag = V.ravel().astype(DTYPE_C)
V_full = cp.concatenate((V_diag, V_diag))

np.save(out_dir / "x.npy", to_cpu(x))
np.save(out_dir / "y.npy", to_cpu(y))
np.save(out_dir / "z.npy", to_cpu(z))

constants = {
    "hx": hx,
    "hy": hy,
    "hz": hz,
    "Lx": Lx,
    "Ly": Ly,
    "Lz_nominal": Lz,
    "z_roof_pos_discrete": z_roof_pos,
    "dt": dt,
    "T_final": T_final,
    "Nx": Nx,
    "Ny": Ny,
    "Nz": Nz,
    "omega": float(omega),
    "k_bc": float(k_bc),
    "k0": float(k0),
    "z0_init": float(z0_init),
    "sigma_z": float(sigma_z),
    "theta": float(theta),
    "phi": float(phi),
}
np.savez(out_dir / "constants.npz", **constants)
logging.info("[info] Saved coordinate grids and constants")

del X, Y, Z, Rho2, psi_xy, psi_z, psi_scalar

print("Simulation Parameters:")
print(f"Nx: {Nx}")
print(f"Ny: {Ny}")
print(f"Nz: {Nz}")
print(f"theta: {theta}")
print(f"Lx: {Lx}")
print(f"Ly: {Ly}")
print(f"Lz nominal: {Lz}")
print(f"stored ABC-row z position: {z_roof_pos}")
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
print(f"mid_x: {mid_x}")
print(f"mid_y: {mid_y}")
print(f"mid_z: {mid_z}")
print(f"out_dir: {out_dir}")
print("-------------------------")



# =============================================================================
# Sparse Hamiltonian and Crank--Nicolson matrices
# =============================================================================
# This block constructs the same finite-difference evolution operator as the
# production solver.  The diagnostic analysis starts only after the CN matrices
# have been assembled.
# =============================================================================
# Hamiltonian / CN matrices: unchanged from the original script
# =============================================================================


def L_z(N_z, dz, k_bc):
    """
    Discretizes H = -(1/2) d^2/dz^2 with:
      z=0 : Dirichlet (modified stencil at i=1, empty row at i=0)
      z=L : Robin ABC (∂z ψ = i k_bc ψ)
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
    L_z_raw = csr_matrix((data, indices, indptr), shape=(N_z, N_z))
    return L_z_raw



def L_dirichlet(N, d):
    """
    Constructs a second-derivative operator in x/y with Dirichlet boundary
    conditions, using CuPy's CSR matrix format.
    """
    inv_d2 = 1 / d**2
    half_inv_d2 = 0.5 * inv_d2
    data = []
    indices = []
    indptr = [0]
    for i in range(N):
        if i == 0 or i == N - 1:
            indptr.append(indptr[-1])
        else:
            left = i - 1
            right = i + 1
            if left == 0:
                indices.extend([i, right])
                data.append(inv_d2)
                data.append(-half_inv_d2)
                indptr.append(indptr[-1] + 2)
            elif right == N - 1:
                indices.extend([left, i])
                data.append(-half_inv_d2)
                data.append(inv_d2)
                indptr.append(indptr[-1] + 2)
            else:
                indices.extend([left, i, right])
                data.append(-half_inv_d2)
                data.append(inv_d2)
                data.append(-half_inv_d2)
                indptr.append(indptr[-1] + 3)
    data = cp.array(data, dtype=DTYPE_C)
    indices = cp.array(indices, dtype=cp.int32)
    indptr = cp.array(indptr, dtype=cp.int32)
    L_raw = csr_matrix((data, indices, indptr), shape=(N, N))
    return L_raw


L1 = L_z(Nz, hz, k_bc)
L2 = L_dirichlet(Nx, hx)
L3 = L_dirichlet(Ny, hy)
Ix = eye(Nx, dtype=DTYPE_C, format="csr")
Iy = eye(Ny, dtype=DTYPE_C, format="csr")
Iz = eye(Nz, dtype=DTYPE_C, format="csr")

Lap_scalar = (
    kron(L2, kron(Iy, Iz))
    + kron(Ix, kron(L3, Iz))
    + kron(Ix, kron(Iy, L1))
).tocsr()
Lap_diag = kron(eye(2, dtype=DTYPE_C, format="csr"), Lap_scalar)

rows_list, cols_list, data_list = [], [], []
coef = 0.5 / hz
d_coef_x = 1.0 / (2.0 * hx)
d_coef_y = 1.0 / (2.0 * hy)
Ngrid = Nx * Ny * Nz

for ix in range(1, Nx - 1):
    for iy in range(1, Ny - 1):
        z_top = Nz - 1
        base = ix * Ny * Nz + iy * Nz + z_top

        row_up = base
        if ix > 1:
            col = Ngrid + (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up)
            cols_list.append(col)
            data_list.append(coef * (-d_coef_x))
        if ix < Nx - 2:
            col = Ngrid + (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_up)
            cols_list.append(col)
            data_list.append(coef * (+d_coef_x))
        if iy > 1:
            col = Ngrid + ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_up)
            cols_list.append(col)
            data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 2:
            col = Ngrid + ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_up)
            cols_list.append(col)
            data_list.append(coef * (-1j) * d_coef_y)

        row_down = Ngrid + base
        if ix > 1:
            col = (ix - 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down)
            cols_list.append(col)
            data_list.append(coef * (+d_coef_x))
        if ix < Nx - 2:
            col = (ix + 1) * Ny * Nz + iy * Nz + z_top
            rows_list.append(row_down)
            cols_list.append(col)
            data_list.append(coef * (-d_coef_x))
        if iy > 1:
            col = ix * Ny * Nz + (iy - 1) * Nz + z_top
            rows_list.append(row_down)
            cols_list.append(col)
            data_list.append(coef * (+1j) * d_coef_y)
        if iy < Ny - 2:
            col = ix * Ny * Nz + (iy + 1) * Nz + z_top
            rows_list.append(row_down)
            cols_list.append(col)
            data_list.append(coef * (-1j) * d_coef_y)

C = coo_matrix(
    (cp.array(data_list, dtype=DTYPE_C), (cp.array(rows_list), cp.array(cols_list))),
    shape=(2 * Ngrid, 2 * Ngrid),
    dtype=DTYPE_C,
).tocsr()

Lap_spinor = (Lap_diag + C).tocsr()

Ntot = 2 * Nx * Ny * Nz
Id = eye(Ntot, dtype=DTYPE_C, format="csr")

Potential = diags(V_full, 0, format="csr")
A = Id + 1j * dt / 2 * (Lap_spinor + Potential).tocsr()
B = Id - 1j * dt / 2 * (Lap_spinor + Potential).tocsr()
inv_diag = 1.0 / A.diagonal()
M = LinearOperator(shape=A.shape, matvec=lambda x: inv_diag * x)

A64 = A.astype(cp.complex128)
B64 = B.astype(cp.complex128)
inv_diag64 = 1.0 / A64.diagonal()
M64 = LinearOperator(shape=A64.shape, matvec=lambda x: inv_diag64 * x)



# =============================================================================
# Boundary-symbol diagnostic controls
# =============================================================================
# DEPTH_STEPS tests true slab rows against homogeneous two-branch continuation.
# BL_EPS_STEPS tests exact finite-epsilon density formulas and Taylor expansion.
# USE_FD_SYMBOL chooses the finite-difference tangential symbol sin(kh)/h.
# =============================================================================
# Boundary-layer / covariance diagnostics for the new two-branch approach
# =============================================================================

DIAG_EVERY = env_int("DIAG_EVERY", 20)
PROB_EVERY = env_int("PROB_EVERY", 100)

# Depths used to test the homogeneous full two-branch Duhamel approximation
# from the stored top row into the boundary slab.
DEPTH_STEPS = [int(s) for s in os.getenv("DEPTH_STEPS", "1,2,3,4,8,16,32,64").split(",") if s.strip()]

# Epsilon values, in units of hz, used in the exact/taylor boundary-layer
# density and moment diagnostics.
BL_EPS_STEPS = [int(s) for s in os.getenv("BL_EPS_STEPS", "1,2,3,4").split(",") if s.strip()]

# Symbol and numerical-cutoff controls.
USE_FD_SYMBOL = bool(env_int("USE_FD_SYMBOL", 1))
REL_POWER_CUTOFF = env_float("REL_POWER_CUTOFF", 1e-13)
R_BRANCH_MIN = env_float("R_BRANCH_MIN", 1e-12)
RDELTA_MAX = env_float("RDELTA_MAX", 50.0)
EDGE_BAND = env_int("EDGE_BAND", 4)
ACTIVE_REL_THRESHOLD = env_float("ACTIVE_REL_THRESHOLD", 1e-4)
RMST_T = env_float("RMST_T", T_final)
EPS_NUM = 1e-300

# Optional consistency check only. The stored top row is always the primary
# discrete ABC trace.
ENABLE_GHOST_CHECK = bool(env_int("ENABLE_GHOST_CHECK", 1))

# If true, store some final 2D/1D coefficient arrays for later inspection.
SAVE_EXTRA_ARRAYS = bool(env_int("SAVE_EXTRA_ARRAYS", 1))



# =============================================================================
# Plane-current and trace diagnostics
# =============================================================================
# The stored top row is the primary discrete ABC trace.  The ghost trace below is
# kept only as a consistency check, not as the main detector-budget observable.
# =============================================================================
# Differential, flux, and trace helpers
# =============================================================================


def dz_slice_2nd(u, z_plane, hz):
    if z_plane == 0:
        return (u[:, :, 1] - u[:, :, 0]) / hz
    if z_plane == u.shape[2] - 1:
        return (u[:, :, -1] - u[:, :, -2]) / hz
    return (u[:, :, z_plane + 1] - u[:, :, z_plane - 1]) / (2 * hz)



def dx_2nd_2d(f2d, hx):
    out = cp.empty_like(f2d, dtype=cp.float64)
    f2d = f2d.astype(cp.float64, copy=False)
    out[1:-1, :] = (f2d[2:, :] - f2d[:-2, :]) / (2.0 * hx)
    out[0, :] = (f2d[1, :] - f2d[0, :]) / hx
    out[-1, :] = (f2d[-1, :] - f2d[-2, :]) / hx
    return out



def dy_2nd_2d(f2d, hy):
    out = cp.empty_like(f2d, dtype=cp.float64)
    f2d = f2d.astype(cp.float64, copy=False)
    out[:, 1:-1] = (f2d[:, 2:] - f2d[:, :-2]) / (2.0 * hy)
    out[:, 0] = (f2d[:, 1] - f2d[:, 0]) / hy
    out[:, -1] = (f2d[:, -1] - f2d[:, -2]) / hy
    return out



def dx_complex_2d(f2d, hx):
    f2d = f2d.astype(cp.complex128, copy=False)
    out = cp.empty_like(f2d, dtype=cp.complex128)
    out[1:-1, :] = (f2d[2:, :] - f2d[:-2, :]) / (2.0 * hx)
    out[0, :] = (f2d[1, :] - f2d[0, :]) / hx
    out[-1, :] = (f2d[-1, :] - f2d[-2, :]) / hx
    return out



def dy_complex_2d(f2d, hy):
    f2d = f2d.astype(cp.complex128, copy=False)
    out = cp.empty_like(f2d, dtype=cp.complex128)
    out[:, 1:-1] = (f2d[:, 2:] - f2d[:, :-2]) / (2.0 * hy)
    out[:, 0] = (f2d[:, 1] - f2d[:, 0]) / hy
    out[:, -1] = (f2d[:, -1] - f2d[:, -2]) / hy
    return out



def plane_pauli_fluxes(psi_up, psi_down, zp):
    """
    Plane-integrated Pauli current.  This is a diagnostic consistency check;
    the exact discrete detector budget for the spinor ABC is computed from
    kappa * int |psi_top|^2 dxdy.
    """
    u = psi_up[:, :, zp]
    d = psi_down[:, :, zp]

    duz = dz_slice_2nd(psi_up, zp, hz)
    ddz = dz_slice_2nd(psi_down, zp, hz)

    j_conv = cp.imag(u.conj() * duz + d.conj() * ddz).astype(cp.float64)

    c = u.conj() * d
    Sx = (2.0 * cp.real(c)).astype(cp.float64)
    Sy = (2.0 * cp.imag(c)).astype(cp.float64)

    dSydx = dx_2nd_2d(Sy, hx)
    dSxdy = dy_2nd_2d(Sx, hy)
    j_spin = (0.5 * (dSydx - dSxdy)).astype(cp.float64)

    jz = j_conv + j_spin

    Jnet_total = cp.sum(jz) * (hx * hy)
    Jnet_conv = cp.sum(j_conv) * (hx * hy)
    Jnet_spin = cp.sum(j_spin) * (hx * hy)

    Jplus_loc_total = cp.sum(cp.maximum(jz, 0.0)) * (hx * hy)
    Jminus_loc_total = cp.sum(cp.maximum(-jz, 0.0)) * (hx * hy)

    return {
        "Jnet_total": float(Jnet_total.get()),
        "Jnet_conv": float(Jnet_conv.get()),
        "Jnet_spin": float(Jnet_spin.get()),
        "Jplus_local_total": float(Jplus_loc_total.get()),
        "Jminus_local_total": float(Jminus_loc_total.get()),
    }



def ghost_trace_from_abc(psi_up, psi_down):
    """
    One-step continuum-style ghost reconstruction from the stored top row.
    This is not the primary trace used in the argument; it is only a check.
    """
    u = psi_up[:, :, z_roof_idx].astype(cp.complex128, copy=False)
    d = psi_down[:, :, z_roof_idx].astype(cp.complex128, copy=False)

    dxd = dx_complex_2d(d, hx)
    dyd = dy_complex_2d(d, hy)
    dxu = dx_complex_2d(u, hx)
    dyu = dy_complex_2d(u, hy)

    Dminus_d = dxd - 1j * dyd
    Dplus_u = dxu + 1j * dyu

    uL = u + hz * (1j * k_bc * u - Dminus_d)
    dL = d + hz * (1j * k_bc * d + Dplus_u)

    uL[[0, -1], :] = 0
    uL[:, [0, -1]] = 0
    dL[[0, -1], :] = 0
    dL[:, [0, -1]] = 0

    return uL, dL



def transverse_sidewall_mass_2d(u, d, band=EDGE_BAND):
    rho = cp.abs(u) ** 2 + cp.abs(d) ** 2
    mask = cp.zeros(rho.shape, dtype=cp.bool_)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    edge = cp.sum(cp.where(mask, rho, 0.0)) * hx * hy
    total = cp.sum(rho) * hx * hy
    edge_f = float(edge.get())
    total_f = float(total.get())
    return edge_f, total_f, edge_f / total_f if total_f > 0 else np.nan



# =============================================================================
# Tangential spin--momentum symbol J(xi)
# =============================================================================
# J is the finite-grid version of the boundary matrix part with eigenvalues
# +/- |xi|.  The code uses this to form Pi_+/Pi_- projectors and exact maps
# exp(-epsilon J) without dropping the Pi_- branch.
# =============================================================================
# Tangential Fourier symbols and J-operator helpers
# =============================================================================

kx = 2.0 * np.pi * cp.fft.fftfreq(Nx, d=hx)
ky = 2.0 * np.pi * cp.fft.fftfreq(Ny, d=hy)
KX, KY = cp.meshgrid(kx, ky, indexing="ij")

if USE_FD_SYMBOL:
    XI_X = cp.sin(KX * hx) / hx
    XI_Y = cp.sin(KY * hy) / hy
else:
    XI_X = KX
    XI_Y = KY

R_TAN = cp.sqrt(XI_X**2 + XI_Y**2).astype(cp.float64)
R2_TAN = R_TAN**2
BRANCH_MODE_MASK = R_TAN > R_BRANCH_MIN
R_SAFE = cp.where(BRANCH_MODE_MASK, R_TAN, 1.0)
XI_PLUS = (XI_X + 1j * XI_Y).astype(cp.complex128)
XI_MINUS = (XI_X - 1j * XI_Y).astype(cp.complex128)

sqrt_omega = float(np.sqrt(float(omega))) if omega > 0 else np.nan
S_TAN = R_TAN / sqrt_omega if sqrt_omega > 0 else cp.full_like(R_TAN, cp.nan)
S2_TAN = S_TAN**2



def fft2_trace(u, d):
    uhat = cp.fft.fft2(u.astype(cp.complex128, copy=False), norm="ortho")
    dhat = cp.fft.fft2(d.astype(cp.complex128, copy=False), norm="ortho")
    return uhat, dhat



def apply_J_to_hat(uhat, dhat):
    """
    J(xi) W, where
        J = [[0, -i xi_-], [i xi_+, 0]].
    """
    jw_u = -1j * XI_MINUS * dhat
    jw_d = 1j * XI_PLUS * uhat
    return jw_u, jw_d



def wdag_j_w(uhat, dhat):
    jw_u, jw_d = apply_J_to_hat(uhat, dhat)
    val = cp.conj(uhat) * jw_u + cp.conj(dhat) * jw_d
    return cp.real(val).astype(cp.float64), jw_u, jw_d



def apply_projectors(uhat, dhat):
    """
    Pi_+ and Pi_- projectors for R>0.  For R=0, the split is not canonical;
    branch ratios exclude those modes.
    """
    jw_u, jw_d = apply_J_to_hat(uhat, dhat)
    nr_u = cp.where(BRANCH_MODE_MASK, jw_u / R_SAFE, 0.0)
    nr_d = cp.where(BRANCH_MODE_MASK, jw_d / R_SAFE, 0.0)

    p_u = 0.5 * (uhat + nr_u)
    p_d = 0.5 * (dhat + nr_d)
    m_u = 0.5 * (uhat - nr_u)
    m_d = 0.5 * (dhat - nr_d)
    return p_u, p_d, m_u, m_d



def spinor_norm2_hat(uhat, dhat, mask=None):
    dens = cp.abs(uhat) ** 2 + cp.abs(dhat) ** 2
    if mask is not None:
        dens = cp.where(mask, dens, 0.0)
    return cp.sum(dens)



def exp_minus_epsJ_on_hat(uhat, dhat, eps):
    """
    Exact full two-branch map e^{-eps J(xi)} W without explicitly splitting
    into Pi_+ and Pi_-.  This is regular at R=0.
    """
    jw_u, jw_d = apply_J_to_hat(uhat, dhat)
    rd = R_TAN * eps
    c = cp.cosh(rd)
    sinh_over_R = cp.where(BRANCH_MODE_MASK, cp.sinh(rd) / R_SAFE, eps)
    out_u = c * uhat - sinh_over_R * jw_u
    out_d = c * dhat - sinh_over_R * jw_d
    return out_u, out_d



def exp_minus_2epsJ_density(uhat, dhat, eps):
    """
    rho_exact = W^dagger e^{-2 eps J} W.
    Regular at R=0.  This is exactly equivalent to the full Pi_+/Pi_- formula.
    """
    rho0 = (cp.abs(uhat) ** 2 + cp.abs(dhat) ** 2).astype(cp.float64)
    wjw, _, _ = wdag_j_w(uhat, dhat)
    rd2 = 2.0 * R_TAN * eps
    c2 = cp.cosh(rd2)
    sinh2_over_R = cp.where(BRANCH_MODE_MASK, cp.sinh(rd2) / R_SAFE, 2.0 * eps)
    return c2 * rho0 - sinh2_over_R * wjw



# =============================================================================
# Central covariance diagnostic
# =============================================================================
# This is the main Supplemental-Material diagnostic block.  It computes the
# stable product s*a*rho0 without pointwise division by small spectral density,
# then integrates time-weighted rates to obtain beta_omega-style coefficients.
# =============================================================================
# Central boundary-layer moment diagnostics
# =============================================================================


def boundary_layer_observables_from_trace(u_trace, d_trace, eps_steps=BL_EPS_STEPS):
    """
    Computes exactly the quantities entering

        tau_bl = tau_0 + eps sqrt(omega) beta_omega + O(eps^2 omega),
        beta_omega = Cov_{nu0}(t, s a_omega),  s=R/sqrt(omega),

    without dividing by R or by rho0.  The stable identity used is

        s a rho0 = -(2/sqrt(omega)) W^dagger J W.

    The exact finite-epsilon density uses

        rho_exact(eps) = W^dagger exp(-2 eps J) W,

    so Pi_- is included exactly.
    """
    u = u_trace.astype(cp.complex128, copy=False)
    d = d_trace.astype(cp.complex128, copy=False)
    edge_mass, total_mass_xy, edge_fraction = transverse_sidewall_mass_2d(u, d)

    uhat, dhat = fft2_trace(u, d)
    rho0 = (cp.abs(uhat) ** 2 + cp.abs(dhat) ** 2).astype(cp.float64)
    wjw, _, _ = wdag_j_w(uhat, dhat)

    max_rho = float(cp.max(rho0).get()) if rho0.size else 0.0
    if max_rho > 0 and np.isfinite(max_rho):
        power_mask = rho0 > (REL_POWER_CUTOFF * max_rho)
    else:
        power_mask = cp.zeros_like(rho0, dtype=cp.bool_)

    branch_mask = power_mask & BRANCH_MODE_MASK
    neutral_mask = power_mask & (~BRANCH_MODE_MASK)

    rho0_sum = cp.sum(rho0)
    rho0_power_sum = cp.sum(cp.where(power_mask, rho0, 0.0))
    neutral_sum = cp.sum(cp.where(neutral_mask, rho0, 0.0))

    # Physical detector rate at the stored ABC row.  With norm='ortho',
    # Parseval gives sum_hat rho0 = sum_xy |psi|^2.
    rate0 = k_bc * hx * hy * rho0_sum

    # Stable products for the covariance and RMST coefficients.
    # R a rho0 = -2 W^dagger J W; s a rho0 = (R/sqrt(omega)) a rho0.
    Ra_rho0 = -2.0 * wjw
    if sqrt_omega > 0:
        sa_rho0 = Ra_rho0 / sqrt_omega
    else:
        sa_rho0 = cp.full_like(Ra_rho0, cp.nan)

    rate_Ra = k_bc * hx * hy * cp.sum(Ra_rho0)
    rate_sa = k_bc * hx * hy * cp.sum(sa_rho0)
    rate_abs_sa_bound = k_bc * hx * hy * cp.sum(2.0 * S_TAN * rho0) if sqrt_omega > 0 else cp.nan
    rate_R2 = k_bc * hx * hy * cp.sum(R2_TAN * rho0)
    rate_s2 = k_bc * hx * hy * cp.sum(S2_TAN * rho0) if sqrt_omega > 0 else cp.nan

    # Optional branch weights.  These are not used to neglect Pi_-; they are
    # reported only to show what the split contains.
    p_u, p_d, m_u, m_d = apply_projectors(uhat, dhat)
    Wp0 = spinor_norm2_hat(p_u, p_d, mask=branch_mask)
    Wm0 = spinor_norm2_hat(m_u, m_d, mask=branch_mask)
    Wp0_f = float(Wp0.get())
    Wm0_f = float(Wm0.get())
    branch_denom = Wp0_f + Wm0_f

    out = {
        "rate0": float(rate0.get()),
        "rate_Ra": float(rate_Ra.get()),
        "rate_sa": float(rate_sa.get()),
        "rate_abs_sa_bound": float(rate_abs_sa_bound.get()) if hasattr(rate_abs_sa_bound, "get") else float(rate_abs_sa_bound),
        "rate_R2": float(rate_R2.get()),
        "rate_s2": float(rate_s2.get()) if hasattr(rate_s2, "get") else float(rate_s2),
        "rho0_sum": float(rho0_sum.get()),
        "rho0_power_sum": float(rho0_power_sum.get()),
        "power_mask_fraction_of_rho0": float((rho0_power_sum / (rho0_sum + EPS_NUM)).get()),
        "neutral_fraction_of_power_mask": float((neutral_sum / (rho0_power_sum + EPS_NUM)).get()),
        "edge_mass": float(edge_mass),
        "total_mass_xy": float(total_mass_xy),
        "edge_fraction": float(edge_fraction),
        "branch_W_plus": Wp0_f,
        "branch_W_minus": Wm0_f,
        "branch_Q_minus": float(Wm0_f / branch_denom) if branch_denom > 0 else np.nan,
        "branch_eta_minus": float(np.sqrt(Wm0_f / (Wp0_f + EPS_NUM))) if Wp0_f >= 0 and Wm0_f >= 0 else np.nan,
        "eps": {},
    }

    for step in eps_steps:
        step = int(step)
        eps = float(step * hz)
        rdelta = R_TAN * eps
        valid_mask = power_mask & (rdelta <= RDELTA_MAX)
        valid_power_sum = cp.sum(cp.where(valid_mask, rho0, 0.0))
        valid_power_fraction = valid_power_sum / (rho0_sum + EPS_NUM)

        rho_exact = exp_minus_2epsJ_density(uhat, dhat, eps)
        rho_linear = rho0 + eps * Ra_rho0
        rho_quad = rho0 + eps * Ra_rho0 + 2.0 * (eps**2) * R2_TAN * rho0

        rho_exact_v = cp.where(valid_mask, rho_exact, 0.0)
        rho_linear_v = cp.where(valid_mask, rho_linear, 0.0)
        rho_quad_v = cp.where(valid_mask, rho_quad, 0.0)
        rho0_v = cp.where(valid_mask, rho0, 0.0)

        exact_rate = k_bc * hx * hy * cp.sum(rho_exact_v)
        linear_rate = k_bc * hx * hy * cp.sum(rho_linear_v)
        quad_rate = k_bc * hx * hy * cp.sum(rho_quad_v)
        rate0_valid = k_bc * hx * hy * cp.sum(rho0_v)

        abs_l1_linear = k_bc * hx * hy * cp.sum(cp.abs(rho_exact_v - rho_linear_v))
        abs_l1_quad = k_bc * hx * hy * cp.sum(cp.abs(rho_exact_v - rho_quad_v))

        neg_linear = k_bc * hx * hy * cp.sum(cp.where(valid_mask, cp.maximum(-rho_linear, 0.0), 0.0))
        neg_quad = k_bc * hx * hy * cp.sum(cp.where(valid_mask, cp.maximum(-rho_quad, 0.0), 0.0))

        # Branch-based eta/rem at finite epsilon, for comparison only.
        branch_valid = branch_mask & (rdelta <= RDELTA_MAX)
        exp_plus = cp.exp(-rdelta)
        exp_minus = cp.exp(+rdelta)
        Wp_eps = spinor_norm2_hat(exp_plus * p_u, exp_plus * p_d, mask=branch_valid)
        Wm_eps = spinor_norm2_hat(exp_minus * m_u, exp_minus * m_d, mask=branch_valid)
        Wp_eps_f = float(Wp_eps.get())
        Wm_eps_f = float(Wm_eps.get())
        denom_eps = Wp_eps_f + Wm_eps_f
        eta_eps = np.sqrt(Wm_eps_f / (Wp_eps_f + EPS_NUM)) if Wp_eps_f >= 0 and Wm_eps_f >= 0 else np.nan

        out["eps"][step] = {
            "epsilon": eps,
            "valid_power_fraction": float(valid_power_fraction.get()),
            "exact_rate": float(exact_rate.get()),
            "linear_rate": float(linear_rate.get()),
            "quadratic_rate": float(quad_rate.get()),
            "rate0_valid": float(rate0_valid.get()),
            "linear_abs_L1_rate_error": float(abs_l1_linear.get()),
            "quadratic_abs_L1_rate_error": float(abs_l1_quad.get()),
            "linear_negative_rate_part": float(neg_linear.get()),
            "quadratic_negative_rate_part": float(neg_quad.get()),
            "branch_W_plus_eps": Wp_eps_f,
            "branch_W_minus_eps": Wm_eps_f,
            "branch_Q_minus_eps": float(Wm_eps_f / denom_eps) if denom_eps > 0 else np.nan,
            "branch_eta_minus_eps": float(eta_eps),
            "branch_rem_ratio_eps": float(abs(k_bc * eps) * eta_eps) if np.isfinite(eta_eps) else np.nan,
        }

    return out



# =============================================================================
# Full two-branch Duhamel/slab diagnostic
# =============================================================================
# Compares actual rows below the roof with the homogeneous continuation from
# the top-row trace.  E_full is the central diagnostic; E_plus_only is retained
# as a historical comparison, not as the new argument.
# =============================================================================
# Full two-branch Duhamel / homogeneous slab diagnostic
# =============================================================================


def duhamel_boundary_slab_diagnostic_top(psi_up, psi_down):
    """
    Compares actual rows L-delta with the homogeneous full two-branch map

        exp(-i kappa delta) exp(-delta J) W_top.

    E_full is the direct diagnostic for whether the Duhamel remainder is small.
    E_linear_wave tests the wavefunction-level linearization exp(-delta J)
    approx I-delta J.  E_plus is retained only as a historical comparison.
    """
    uB = psi_up[:, :, z_roof_idx].astype(cp.complex128, copy=False)
    dB = psi_down[:, :, z_roof_idx].astype(cp.complex128, copy=False)
    uhat, dhat = fft2_trace(uB, dB)
    rho0 = (cp.abs(uhat) ** 2 + cp.abs(dhat) ** 2).astype(cp.float64)

    max_rho = float(cp.max(rho0).get()) if rho0.size else 0.0
    if max_rho > 0 and np.isfinite(max_rho):
        power_mask = rho0 > (REL_POWER_CUTOFF * max_rho)
    else:
        power_mask = cp.zeros_like(rho0, dtype=cp.bool_)

    p_u, p_d, _, _ = apply_projectors(uhat, dhat)
    jw_u, jw_d = apply_J_to_hat(uhat, dhat)

    out = {}
    for step in DEPTH_STEPS:
        step = int(step)
        zidx = z_roof_idx - step
        if step < 1 or zidx < 0:
            continue

        delta = float(step * hz)
        rdelta = R_TAN * delta
        valid_mask = power_mask & (rdelta <= RDELTA_MAX)

        u_true = psi_up[:, :, zidx].astype(cp.complex128, copy=False)
        d_true = psi_down[:, :, zidx].astype(cp.complex128, copy=False)
        u_true_hat, d_true_hat = fft2_trace(u_true, d_true)

        phase = cp.exp(-1j * k_bc * delta)

        u_exp, d_exp = exp_minus_epsJ_on_hat(uhat, dhat, delta)
        u_hom = phase * u_exp
        d_hom = phase * d_exp

        u_lin = phase * (uhat - delta * jw_u)
        d_lin = phase * (dhat - delta * jw_d)

        # Historical plus-only diagnostic.  It is not used in the new proof.
        u_plus = phase * (cp.exp(-rdelta) * p_u)
        d_plus = phase * (cp.exp(-rdelta) * p_d)

        true_norm = spinor_norm2_hat(u_true_hat, d_true_hat, mask=valid_mask)
        diff_full = spinor_norm2_hat(u_true_hat - u_hom, d_true_hat - d_hom, mask=valid_mask)
        diff_linear = spinor_norm2_hat(u_true_hat - u_lin, d_true_hat - d_lin, mask=valid_mask)
        diff_plus = spinor_norm2_hat(u_true_hat - u_plus, d_true_hat - d_plus, mask=valid_mask)
        hom_norm = spinor_norm2_hat(u_hom, d_hom, mask=valid_mask)

        true_norm_f = float(true_norm.get())
        hom_norm_f = float(hom_norm.get())
        rho0_valid_sum = float(cp.sum(cp.where(valid_mask, rho0, 0.0)).get())
        rho0_sum = float(cp.sum(rho0).get())

        out[step] = {
            "delta": delta,
            "z_index": int(zidx),
            "z_pos": float(z_cpu[zidx]),
            "valid_power_fraction": float(rho0_valid_sum / (rho0_sum + EPS_NUM)),
            "E_full_rel_true": float(cp.sqrt(diff_full / true_norm).get()) if true_norm_f > 0 else np.nan,
            "E_full_rel_hom": float(cp.sqrt(diff_full / hom_norm).get()) if hom_norm_f > 0 else np.nan,
            "E_linear_wave_rel_true": float(cp.sqrt(diff_linear / true_norm).get()) if true_norm_f > 0 else np.nan,
            "E_plus_only_rel_true": float(cp.sqrt(diff_plus / true_norm).get()) if true_norm_f > 0 else np.nan,
        }

    return out



# =============================================================================
# Time-series storage for diagnostics
# =============================================================================
# Arrays are kept separately so post-processing can reproduce every summary.json
# quantity without rerunning the simulation.
# =============================================================================
# Time-series storage
# =============================================================================

prob_t = []
prob_val = []

diag_times = []
P_total_series = []

# Detector and Pauli current sanity checks.
det_rate_toprow_series = []
det_rate_ghost_series = []
pauli_roof_net_series = []
pauli_roof_local_minus_series = []
pauli_roof_local_plus_series = []

# Central zeroth-order and covariance-rate series.
bl_rate0_series = []
bl_rate_Ra_series = []
bl_rate_sa_series = []
bl_rate_abs_sa_bound_series = []
bl_rate_R2_series = []
bl_rate_s2_series = []
bl_edge_fraction_series = []
bl_neutral_fraction_series = []
bl_branch_Q_minus_series = []
bl_branch_eta_minus_series = []

# Exact finite-epsilon and Taylor density-rate diagnostics.
bl_exact_rate = {int(s): [] for s in BL_EPS_STEPS}
bl_linear_rate = {int(s): [] for s in BL_EPS_STEPS}
bl_quadratic_rate = {int(s): [] for s in BL_EPS_STEPS}
bl_rate0_valid = {int(s): [] for s in BL_EPS_STEPS}
bl_linear_abs_L1_error_rate = {int(s): [] for s in BL_EPS_STEPS}
bl_quadratic_abs_L1_error_rate = {int(s): [] for s in BL_EPS_STEPS}
bl_linear_negative_rate_part = {int(s): [] for s in BL_EPS_STEPS}
bl_quadratic_negative_rate_part = {int(s): [] for s in BL_EPS_STEPS}
bl_valid_power_fraction = {int(s): [] for s in BL_EPS_STEPS}
bl_branch_Q_minus_eps = {int(s): [] for s in BL_EPS_STEPS}
bl_branch_eta_minus_eps = {int(s): [] for s in BL_EPS_STEPS}
bl_branch_rem_ratio_eps = {int(s): [] for s in BL_EPS_STEPS}

# Full homogeneous/Duhamel checks.
duhamel_E_full = {int(s): [] for s in DEPTH_STEPS}
duhamel_E_full_hom = {int(s): [] for s in DEPTH_STEPS}
duhamel_E_linear_wave = {int(s): [] for s in DEPTH_STEPS}
duhamel_E_plus_only = {int(s): [] for s in DEPTH_STEPS}
duhamel_valid_power_fraction = {int(s): [] for s in DEPTH_STEPS}



# =============================================================================
# Main CN/GMRES evolution loop with boundary diagnostics
# =============================================================================
# The solver advances the spinor field exactly as in the production code; all
# covariance/Duhamel diagnostics are evaluated at the chosen cadence.
# =============================================================================
# Time loop: evolution unchanged
# =============================================================================

num_steps = int(round(T_final / dt))
print(f"[info] Starting time loop – {num_steps} steps")
start_time = time.time()

psi0 = psi_flat.reshape(2, Nx, Ny, Nz)
P0_init = float(to_cpu(cp.sum(cp.abs(psi0[0]) ** 2 + cp.abs(psi0[1]) ** 2) * (hx * hy * hz)))
del psi0

for n in range(1, num_steps + 1):
    if n % 200 == 0:
        elapsed = time.time() - start_time
        eta = (num_steps - n) * (elapsed / n) / 60.0
        print(f"[progress] Step {n}/{num_steps} (t={n*dt:.3f}) | ETA: {eta:.2f} min", flush=True)

    rhs64 = B64 @ psi_flat.astype(cp.complex128)
    psi64, info = gmres(
        A64,
        rhs64,
        x0=psi_flat.astype(cp.complex128),
        rtol=1e-8,
        atol=0.0,
        maxiter=1000,
        M=M64,
        restart=30,
    )

    if info != 0:
        logging.warning(f"[GMRES] not converged at step {n}, info={info}")
    psi_flat = psi64.astype(cp.complex64)

    psi_3d = psi_flat.reshape(2, Nx, Ny, Nz)
    psi_up = psi_3d[0]
    psi_down = psi_3d[1]

    # Keep the original imposed zero side/bottom boundary clean-up.
    psi_up[[0, -1], :, :] = 0
    psi_up[:, [0, -1], :] = 0
    psi_up[:, :, 0] = 0
    psi_down[[0, -1], :, :] = 0
    psi_down[:, [0, -1], :] = 0
    psi_down[:, :, 0] = 0

    t = n * dt

    if (n % PROB_EVERY) == 0:
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_up) ** 2 + cp.abs(psi_down) ** 2) * (hx * hy * hz)))
        prob_t.append(t)
        prob_val.append(total_prob)
        logging.info(f"[Probability] t={t:.3f}, ∫|ψ|² dV = {total_prob:.8f}")

    if (n % DIAG_EVERY) == 0:
        diag_times.append(t)

        Ptot = float(to_cpu(cp.sum(cp.abs(psi_up) ** 2 + cp.abs(psi_down) ** 2) * (hx * hy * hz)))
        P_total_series.append(Ptot)

        u_top = psi_up[:, :, z_roof_idx].astype(cp.complex128, copy=False)
        d_top = psi_down[:, :, z_roof_idx].astype(cp.complex128, copy=False)

        bl = boundary_layer_observables_from_trace(u_top, d_top, eps_steps=BL_EPS_STEPS)

        bl_rate0_series.append(bl["rate0"])
        bl_rate_Ra_series.append(bl["rate_Ra"])
        bl_rate_sa_series.append(bl["rate_sa"])
        bl_rate_abs_sa_bound_series.append(bl["rate_abs_sa_bound"])
        bl_rate_R2_series.append(bl["rate_R2"])
        bl_rate_s2_series.append(bl["rate_s2"])
        bl_edge_fraction_series.append(bl["edge_fraction"])
        bl_neutral_fraction_series.append(bl["neutral_fraction_of_power_mask"])
        bl_branch_Q_minus_series.append(bl["branch_Q_minus"])
        bl_branch_eta_minus_series.append(bl["branch_eta_minus"])
        det_rate_toprow_series.append(bl["rate0"])

        if ENABLE_GHOST_CHECK:
            u_ghost, d_ghost = ghost_trace_from_abc(psi_up, psi_down)
            rho_ghost = cp.sum(cp.abs(u_ghost) ** 2 + cp.abs(d_ghost) ** 2) * hx * hy
            det_rate_ghost_series.append(float((k_bc * rho_ghost).get()))
        else:
            det_rate_ghost_series.append(np.nan)

        pauli = plane_pauli_fluxes(psi_up, psi_down, z_roof_idx)
        pauli_roof_net_series.append(pauli["Jnet_total"])
        pauli_roof_local_minus_series.append(pauli["Jminus_local_total"])
        pauli_roof_local_plus_series.append(pauli["Jplus_local_total"])

        for s in BL_EPS_STEPS:
            s = int(s)
            ep = bl["eps"][s]
            bl_exact_rate[s].append(ep["exact_rate"])
            bl_linear_rate[s].append(ep["linear_rate"])
            bl_quadratic_rate[s].append(ep["quadratic_rate"])
            bl_rate0_valid[s].append(ep["rate0_valid"])
            bl_linear_abs_L1_error_rate[s].append(ep["linear_abs_L1_rate_error"])
            bl_quadratic_abs_L1_error_rate[s].append(ep["quadratic_abs_L1_rate_error"])
            bl_linear_negative_rate_part[s].append(ep["linear_negative_rate_part"])
            bl_quadratic_negative_rate_part[s].append(ep["quadratic_negative_rate_part"])
            bl_valid_power_fraction[s].append(ep["valid_power_fraction"])
            bl_branch_Q_minus_eps[s].append(ep["branch_Q_minus_eps"])
            bl_branch_eta_minus_eps[s].append(ep["branch_eta_minus_eps"])
            bl_branch_rem_ratio_eps[s].append(ep["branch_rem_ratio_eps"])

        duh = duhamel_boundary_slab_diagnostic_top(psi_up, psi_down)
        for s in DEPTH_STEPS:
            s = int(s)
            if s in duh:
                duhamel_E_full[s].append(duh[s]["E_full_rel_true"])
                duhamel_E_full_hom[s].append(duh[s]["E_full_rel_hom"])
                duhamel_E_linear_wave[s].append(duh[s]["E_linear_wave_rel_true"])
                duhamel_E_plus_only[s].append(duh[s]["E_plus_only_rel_true"])
                duhamel_valid_power_fraction[s].append(duh[s]["valid_power_fraction"])
            else:
                duhamel_E_full[s].append(np.nan)
                duhamel_E_full_hom[s].append(np.nan)
                duhamel_E_linear_wave[s].append(np.nan)
                duhamel_E_plus_only[s].append(np.nan)
                duhamel_valid_power_fraction[s].append(np.nan)



# =============================================================================
# Save arrays and assemble summary.json
# =============================================================================
# The summary is intentionally verbose: it records detector budget, covariance
# coefficients, exact finite-epsilon checks, and Duhamel errors in one file.
# =============================================================================
# Save outputs and build summary
# =============================================================================

def _get_numpy_trapezoid():
    """Return a NumPy trapezoidal integrator compatible with old and new NumPy.

    Do not use getattr(np, "trapezoid", np.trapz): Python evaluates the
    default argument immediately, so it crashes if np.trapz is absent.
    """
    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return fn
    fn = getattr(np, "trapz", None)
    if fn is not None:
        return fn

    def _manual_trapezoid(y, x=None, dx=1.0, axis=-1):
        y = np.asarray(y, dtype=np.float64)
        if x is None:
            d = dx
        else:
            x = np.asarray(x, dtype=np.float64)
            d = np.diff(x, axis=axis) if x.ndim > 1 else np.diff(x)
            if np.ndim(d) == 1 and y.ndim > 1:
                shape = [1] * y.ndim
                shape[axis] = d.shape[0]
                d = d.reshape(shape)
        sl0 = [slice(None)] * y.ndim
        sl1 = [slice(None)] * y.ndim
        sl0[axis] = slice(None, -1)
        sl1[axis] = slice(1, None)
        return np.sum(0.5 * (y[tuple(sl0)] + y[tuple(sl1)]) * d, axis=axis)

    return _manual_trapezoid


trapz_fn = _get_numpy_trapezoid()


def arr(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype)



def trapz_safe(y, x):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return np.nan
    return float(trapz_fn(y, x))



def prepend_zero_time(t, y, y0=0.0):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size == 0:
        return np.asarray([0.0]), np.asarray([float(y0)])
    if t[0] > 0.0:
        return np.concatenate(([0.0], t)), np.concatenate(([float(y0)], y))
    return t, y



def restrict_series_to_T(t, y, T):
    """Return arrays on [0,T], adding a linearly interpolated endpoint if needed."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T = float(T)
    if t.size == 0:
        return np.asarray([0.0, T]), np.asarray([0.0, 0.0])
    if T <= t[0]:
        yT = np.interp(T, t, y)
        return np.asarray([0.0, T]), np.asarray([0.0, yT])
    mask = t < T
    tt = t[mask]
    yy = y[mask]
    if t[-1] < T:
        # We cannot extrapolate trustworthy dynamics.  Use the available endpoint.
        T_eff = t[-1]
        return t, y
    yT = np.interp(T, t, y)
    tt = np.concatenate((tt, [T]))
    yy = np.concatenate((yy, [yT]))
    return tt, yy



def integrate_rate(y, t=None, y0=0.0):
    if t is None:
        t = diag_times
    tt, yy = prepend_zero_time(t, y, y0=y0)
    return trapz_safe(yy, tt)



def integrate_time_rate(y, t=None, y0=0.0):
    if t is None:
        t = diag_times
    tt, yy = prepend_zero_time(t, y, y0=y0)
    return trapz_safe(tt * yy, tt)



def integrate_rate_to_T(y, T, t=None, y0=0.0):
    if t is None:
        t = diag_times
    tt, yy = prepend_zero_time(t, y, y0=y0)
    tt, yy = restrict_series_to_T(tt, yy, T)
    return trapz_safe(yy, tt)



def integrate_time_rate_to_T(y, T, t=None, y0=0.0):
    if t is None:
        t = diag_times
    tt, yy = prepend_zero_time(t, y, y0=y0)
    tt, yy = restrict_series_to_T(tt, yy, T)
    return trapz_safe(tt * yy, tt)



def integrate_shifted_time_rate_to_T(y, T, t=None, y0=0.0):
    if t is None:
        t = diag_times
    tt, yy = prepend_zero_time(t, y, y0=y0)
    tt, yy = restrict_series_to_T(tt, yy, T)
    return trapz_safe((tt - T) * yy, tt)



def stats_weighted_active(y, weight, active):
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    finite = np.isfinite(y) & np.isfinite(weight) & (weight >= 0)
    active_finite = finite & active
    out = {
        "median_all": float(np.nanmedian(y)) if y.size else np.nan,
        "max_all": float(np.nanmax(y)) if y.size else np.nan,
        "median_active": float(np.nanmedian(y[active_finite])) if np.any(active_finite) else np.nan,
        "max_active": float(np.nanmax(y[active_finite])) if np.any(active_finite) else np.nan,
        "weighted_mean_active": np.nan,
    }
    if np.any(active_finite):
        num = integrate_rate(np.where(active_finite, y * weight, 0.0), y0=0.0)
        den = integrate_rate(np.where(active_finite, weight, 0.0), y0=0.0)
        out["weighted_mean_active"] = float(num / den) if den > 0 else np.nan
    return out


# Convert to arrays.
diag_times = arr(diag_times)
P_total_series = arr(P_total_series)
prob_t = arr(prob_t)
prob_val = arr(prob_val)

det_rate_toprow_series = arr(det_rate_toprow_series)
det_rate_ghost_series = arr(det_rate_ghost_series)
pauli_roof_net_series = arr(pauli_roof_net_series)
pauli_roof_local_minus_series = arr(pauli_roof_local_minus_series)
pauli_roof_local_plus_series = arr(pauli_roof_local_plus_series)

bl_rate0_series = arr(bl_rate0_series)
bl_rate_Ra_series = arr(bl_rate_Ra_series)
bl_rate_sa_series = arr(bl_rate_sa_series)
bl_rate_abs_sa_bound_series = arr(bl_rate_abs_sa_bound_series)
bl_rate_R2_series = arr(bl_rate_R2_series)
bl_rate_s2_series = arr(bl_rate_s2_series)
bl_edge_fraction_series = arr(bl_edge_fraction_series)
bl_neutral_fraction_series = arr(bl_neutral_fraction_series)
bl_branch_Q_minus_series = arr(bl_branch_Q_minus_series)
bl_branch_eta_minus_series = arr(bl_branch_eta_minus_series)

for s in BL_EPS_STEPS:
    s = int(s)
    bl_exact_rate[s] = arr(bl_exact_rate[s])
    bl_linear_rate[s] = arr(bl_linear_rate[s])
    bl_quadratic_rate[s] = arr(bl_quadratic_rate[s])
    bl_rate0_valid[s] = arr(bl_rate0_valid[s])
    bl_linear_abs_L1_error_rate[s] = arr(bl_linear_abs_L1_error_rate[s])
    bl_quadratic_abs_L1_error_rate[s] = arr(bl_quadratic_abs_L1_error_rate[s])
    bl_linear_negative_rate_part[s] = arr(bl_linear_negative_rate_part[s])
    bl_quadratic_negative_rate_part[s] = arr(bl_quadratic_negative_rate_part[s])
    bl_valid_power_fraction[s] = arr(bl_valid_power_fraction[s])
    bl_branch_Q_minus_eps[s] = arr(bl_branch_Q_minus_eps[s])
    bl_branch_eta_minus_eps[s] = arr(bl_branch_eta_minus_eps[s])
    bl_branch_rem_ratio_eps[s] = arr(bl_branch_rem_ratio_eps[s])

for s in DEPTH_STEPS:
    s = int(s)
    duhamel_E_full[s] = arr(duhamel_E_full[s])
    duhamel_E_full_hom[s] = arr(duhamel_E_full_hom[s])
    duhamel_E_linear_wave[s] = arr(duhamel_E_linear_wave[s])
    duhamel_E_plus_only[s] = arr(duhamel_E_plus_only[s])
    duhamel_valid_power_fraction[s] = arr(duhamel_valid_power_fraction[s])

# Save core arrays.
np.save(out_dir / "diag_times.npy", diag_times)
np.save(out_dir / "P_total_series.npy", P_total_series)
np.save(out_dir / "prob_t.npy", prob_t)
np.save(out_dir / "prob_val.npy", prob_val)

np.save(out_dir / "det_rate_toprow.npy", det_rate_toprow_series)
np.save(out_dir / "det_rate_ghost_trace.npy", det_rate_ghost_series)
np.save(out_dir / "pauli_roof_net.npy", pauli_roof_net_series)
np.save(out_dir / "pauli_roof_local_minus.npy", pauli_roof_local_minus_series)
np.save(out_dir / "pauli_roof_local_plus.npy", pauli_roof_local_plus_series)

np.save(out_dir / "bl_rate0.npy", bl_rate0_series)
np.save(out_dir / "bl_rate_Ra.npy", bl_rate_Ra_series)
np.save(out_dir / "bl_rate_sa.npy", bl_rate_sa_series)
np.save(out_dir / "bl_rate_abs_sa_bound.npy", bl_rate_abs_sa_bound_series)
np.save(out_dir / "bl_rate_R2.npy", bl_rate_R2_series)
np.save(out_dir / "bl_rate_s2.npy", bl_rate_s2_series)
np.save(out_dir / "bl_edge_fraction.npy", bl_edge_fraction_series)
np.save(out_dir / "bl_neutral_fraction.npy", bl_neutral_fraction_series)
np.save(out_dir / "bl_branch_Q_minus.npy", bl_branch_Q_minus_series)
np.save(out_dir / "bl_branch_eta_minus.npy", bl_branch_eta_minus_series)

for s in BL_EPS_STEPS:
    s = int(s)
    np.save(out_dir / f"bl_exact_rate_epsSteps_{s}.npy", bl_exact_rate[s])
    np.save(out_dir / f"bl_linear_rate_epsSteps_{s}.npy", bl_linear_rate[s])
    np.save(out_dir / f"bl_quadratic_rate_epsSteps_{s}.npy", bl_quadratic_rate[s])
    np.save(out_dir / f"bl_rate0_valid_epsSteps_{s}.npy", bl_rate0_valid[s])
    np.save(out_dir / f"bl_linear_abs_L1_error_rate_epsSteps_{s}.npy", bl_linear_abs_L1_error_rate[s])
    np.save(out_dir / f"bl_quadratic_abs_L1_error_rate_epsSteps_{s}.npy", bl_quadratic_abs_L1_error_rate[s])
    np.save(out_dir / f"bl_linear_negative_rate_part_epsSteps_{s}.npy", bl_linear_negative_rate_part[s])
    np.save(out_dir / f"bl_quadratic_negative_rate_part_epsSteps_{s}.npy", bl_quadratic_negative_rate_part[s])
    np.save(out_dir / f"bl_valid_power_fraction_epsSteps_{s}.npy", bl_valid_power_fraction[s])
    np.save(out_dir / f"bl_branch_Q_minus_epsSteps_{s}.npy", bl_branch_Q_minus_eps[s])
    np.save(out_dir / f"bl_branch_eta_minus_epsSteps_{s}.npy", bl_branch_eta_minus_eps[s])
    np.save(out_dir / f"bl_branch_rem_ratio_epsSteps_{s}.npy", bl_branch_rem_ratio_eps[s])

for s in DEPTH_STEPS:
    s = int(s)
    np.save(out_dir / f"duhamel_E_full_rel_true_depthSteps_{s}.npy", duhamel_E_full[s])
    np.save(out_dir / f"duhamel_E_full_rel_hom_depthSteps_{s}.npy", duhamel_E_full_hom[s])
    np.save(out_dir / f"duhamel_E_linear_wave_rel_true_depthSteps_{s}.npy", duhamel_E_linear_wave[s])
    np.save(out_dir / f"duhamel_E_plus_only_rel_true_depthSteps_{s}.npy", duhamel_E_plus_only[s])
    np.save(out_dir / f"duhamel_valid_power_fraction_depthSteps_{s}.npy", duhamel_valid_power_fraction[s])

# Integrated detector budget.
Z0 = integrate_rate(bl_rate0_series)
M1_0 = integrate_time_rate(bl_rate0_series)
tau0 = M1_0 / Z0 if Z0 > 0 else np.nan

Z_Ra = integrate_rate(bl_rate_Ra_series)
M1_Ra = integrate_time_rate(bl_rate_Ra_series)
Z_sa = integrate_rate(bl_rate_sa_series)
M1_sa = integrate_time_rate(bl_rate_sa_series)

A_Ra_mean = Z_Ra / Z0 if Z0 > 0 else np.nan
B_Ra_mean = M1_Ra / Z0 if Z0 > 0 else np.nan
cov_t_Ra = B_Ra_mean - tau0 * A_Ra_mean if np.isfinite(tau0) else np.nan

A_sa_mean = Z_sa / Z0 if Z0 > 0 else np.nan
B_sa_mean = M1_sa / Z0 if Z0 > 0 else np.nan
beta_cov = B_sa_mean - tau0 * A_sa_mean if np.isfinite(tau0) else np.nan

E_R2 = integrate_rate(bl_rate_R2_series) / Z0 if Z0 > 0 else np.nan
E_s2 = integrate_rate(bl_rate_s2_series) / Z0 if Z0 > 0 else np.nan
abs_sa_bound_mean = integrate_rate(bl_rate_abs_sa_bound_series) / Z0 if Z0 > 0 else np.nan

# RMST coefficient to the requested horizon.  This is the correct coefficient
# if the paper-1 fitted quantity is a restricted mean survival time.
T_rmst_eff = min(float(RMST_T), float(T_final))
F0_T = integrate_rate_to_T(bl_rate0_series, T_rmst_eff)
M1_0_T = integrate_time_rate_to_T(bl_rate0_series, T_rmst_eff)
rmst0_T = M1_0_T + T_rmst_eff * (1.0 - F0_T)

gamma_rmst_linear = integrate_shifted_time_rate_to_T(bl_rate_sa_series, T_rmst_eff)
# Same coefficient before division by sqrt(omega): Delta mu = eps * gamma_Ra.
gamma_Ra_rmst_linear = integrate_shifted_time_rate_to_T(bl_rate_Ra_series, T_rmst_eff)

# Detector budget against total probability loss.
norm_loss_diag = float(P0_init - P_total_series[-1]) if P_total_series.size else np.nan
P_det_toprow_int = Z0
det_toprow_budget_error = P_det_toprow_int - norm_loss_diag if np.isfinite(norm_loss_diag) else np.nan
det_toprow_budget_rel_error = det_toprow_budget_error / norm_loss_diag if np.isfinite(norm_loss_diag) and abs(norm_loss_diag) > 0 else np.nan

P_det_ghost_int = integrate_rate(det_rate_ghost_series)
det_ghost_budget_error = P_det_ghost_int - norm_loss_diag if np.isfinite(norm_loss_diag) else np.nan
det_ghost_budget_rel_error = det_ghost_budget_error / norm_loss_diag if np.isfinite(norm_loss_diag) and abs(norm_loss_diag) > 0 else np.nan

P_pauli_plus_local = integrate_rate(pauli_roof_local_plus_series)
P_pauli_minus_local = integrate_rate(pauli_roof_local_minus_series)
R_pauli_local_backflow = P_pauli_minus_local / P_pauli_plus_local if P_pauli_plus_local > 0 else np.nan

# Active window for reporting instantaneous diagnostics.
active_weight = np.maximum(bl_rate0_series, 0.0)
if active_weight.size and np.nanmax(active_weight) > 0:
    active_mask = active_weight > ACTIVE_REL_THRESHOLD * np.nanmax(active_weight)
else:
    active_mask = np.zeros_like(active_weight, dtype=bool)

# Finite-epsilon exact/taylor moment summaries.
eps_summaries = {}
for s in BL_EPS_STEPS:
    s = int(s)
    eps = float(s * hz)
    denom = eps * sqrt_omega if eps > 0 and sqrt_omega > 0 else np.nan

    Z_exact = integrate_rate(bl_exact_rate[s])
    M1_exact = integrate_time_rate(bl_exact_rate[s])
    tau_exact = M1_exact / Z_exact if Z_exact > 0 else np.nan

    Z_linear = integrate_rate(bl_linear_rate[s])
    M1_linear = integrate_time_rate(bl_linear_rate[s])
    tau_linear = M1_linear / Z_linear if Z_linear > 0 else np.nan

    Z_quad = integrate_rate(bl_quadratic_rate[s])
    M1_quad = integrate_time_rate(bl_quadratic_rate[s])
    tau_quad = M1_quad / Z_quad if Z_quad > 0 else np.nan

    beta_exact = (tau_exact - tau0) / denom if np.isfinite(denom) and denom != 0 else np.nan
    beta_linear_from_tau = (tau_linear - tau0) / denom if np.isfinite(denom) and denom != 0 else np.nan
    beta_quad_from_tau = (tau_quad - tau0) / denom if np.isfinite(denom) and denom != 0 else np.nan

    F_exact_T = integrate_rate_to_T(bl_exact_rate[s], T_rmst_eff)
    M1_exact_T = integrate_time_rate_to_T(bl_exact_rate[s], T_rmst_eff)
    rmst_exact_T = M1_exact_T + T_rmst_eff * (1.0 - F_exact_T)

    F_linear_T = integrate_rate_to_T(bl_linear_rate[s], T_rmst_eff)
    M1_linear_T = integrate_time_rate_to_T(bl_linear_rate[s], T_rmst_eff)
    rmst_linear_T = M1_linear_T + T_rmst_eff * (1.0 - F_linear_T)

    gamma_exact = (rmst_exact_T - rmst0_T) / denom if np.isfinite(denom) and denom != 0 else np.nan
    gamma_linear_from_rmst = (rmst_linear_T - rmst0_T) / denom if np.isfinite(denom) and denom != 0 else np.nan

    L1_linear_int = integrate_rate(bl_linear_abs_L1_error_rate[s])
    L1_quad_int = integrate_rate(bl_quadratic_abs_L1_error_rate[s])
    L1_linear_rel = L1_linear_int / Z_exact if Z_exact > 0 else np.nan
    L1_quad_rel = L1_quad_int / Z_exact if Z_exact > 0 else np.nan

    neg_linear_int = integrate_rate(bl_linear_negative_rate_part[s])
    neg_quad_int = integrate_rate(bl_quadratic_negative_rate_part[s])

    eps_summaries[str(s)] = {
        "epsilon": eps,
        "epsilon_sqrt_omega": float(eps * sqrt_omega) if sqrt_omega > 0 else np.nan,
        "O_eps2_R2_scale": float((eps**2) * E_R2) if np.isfinite(E_R2) else np.nan,
        "O_eps2_omega_Es2_scale": float((eps**2) * omega * E_s2) if np.isfinite(E_s2) else np.nan,
        "valid_power_fraction": stats_weighted_active(bl_valid_power_fraction[s], active_weight, active_mask),
        "exact": {
            "Z_exact": float(Z_exact),
            "tau_exact": float(tau_exact),
            "beta_exact_from_tau": float(beta_exact),
            "rmst_exact_T": float(rmst_exact_T),
            "gamma_exact_from_rmst": float(gamma_exact),
        },
        "linear_taylor": {
            "Z_linear": float(Z_linear),
            "tau_linear": float(tau_linear),
            "beta_linear_from_tau": float(beta_linear_from_tau),
            "rmst_linear_T": float(rmst_linear_T),
            "gamma_linear_from_rmst": float(gamma_linear_from_rmst),
            "linear_density_L1_relative_error": float(L1_linear_rel),
            "linear_negative_mass_integral": float(neg_linear_int),
        },
        "quadratic_taylor": {
            "Z_quadratic": float(Z_quad),
            "tau_quadratic": float(tau_quad),
            "beta_quadratic_from_tau": float(beta_quad_from_tau),
            "quadratic_density_L1_relative_error": float(L1_quad_rel),
            "quadratic_negative_mass_integral": float(neg_quad_int),
        },
        "comparison_to_covariance": {
            "beta_covariance": float(beta_cov),
            "beta_exact_minus_beta_cov": float(beta_exact - beta_cov) if np.isfinite(beta_exact) and np.isfinite(beta_cov) else np.nan,
            "relative_beta_exact_minus_cov": float(abs(beta_exact - beta_cov) / (abs(beta_exact) + EPS_NUM)) if np.isfinite(beta_exact) and np.isfinite(beta_cov) else np.nan,
            "gamma_rmst_linear_direct": float(gamma_rmst_linear),
            "gamma_exact_minus_gamma_linear": float(gamma_exact - gamma_rmst_linear) if np.isfinite(gamma_exact) and np.isfinite(gamma_rmst_linear) else np.nan,
            "relative_gamma_exact_minus_linear": float(abs(gamma_exact - gamma_rmst_linear) / (abs(gamma_exact) + EPS_NUM)) if np.isfinite(gamma_exact) and np.isfinite(gamma_rmst_linear) else np.nan,
        },
        "branch_split_report_only": {
            "Q_minus_eps": stats_weighted_active(bl_branch_Q_minus_eps[s], active_weight, active_mask),
            "eta_minus_eps": stats_weighted_active(bl_branch_eta_minus_eps[s], active_weight, active_mask),
            "rem_ratio_eps": stats_weighted_active(bl_branch_rem_ratio_eps[s], active_weight, active_mask),
        },
    }

# Duhamel summaries.
duhamel_summary = {}
for s in DEPTH_STEPS:
    s = int(s)
    duhamel_summary[str(s)] = {
        "delta": float(s * hz),
        "epsilon_sqrt_omega": float(s * hz * sqrt_omega) if sqrt_omega > 0 else np.nan,
        "valid_power_fraction": stats_weighted_active(duhamel_valid_power_fraction[s], active_weight, active_mask),
        "E_full_two_branch_rel_true": stats_weighted_active(duhamel_E_full[s], active_weight, active_mask),
        "E_full_two_branch_rel_hom": stats_weighted_active(duhamel_E_full_hom[s], active_weight, active_mask),
        "E_linear_wave_rel_true": stats_weighted_active(duhamel_E_linear_wave[s], active_weight, active_mask),
        "E_plus_only_rel_true_historical": stats_weighted_active(duhamel_E_plus_only[s], active_weight, active_mask),
    }

# Optional save of extra coefficient metadata.
if SAVE_EXTRA_ARRAYS:
    beta_integrands = {
        "diag_times": diag_times,
        "rate0": bl_rate0_series,
        "rate_sa": bl_rate_sa_series,
        "rate_Ra": bl_rate_Ra_series,
        "rate_R2": bl_rate_R2_series,
        "rate_s2": bl_rate_s2_series,
    }
    np.savez(out_dir / "boundary_layer_covariance_integrands.npz", **beta_integrands)

summary = {
    "run_dir": str(out_dir.resolve()),
    "omega": float(omega),
    "sqrt_omega": float(sqrt_omega),
    "dt": float(dt),
    "T_final": float(T_final),
    "RMST_T_requested": float(RMST_T),
    "RMST_T_used": float(T_rmst_eff),
    "NxNyNz": [int(Nx), int(Ny), int(Nz)],
    "LxLyLz_nominal": [float(Lx), float(Ly), float(Lz)],
    "z_roof_pos_discrete": float(z_roof_pos),
    "hxhyhz": [float(hx), float(hy), float(hz)],
    "k_bc": float(k_bc),
    "k0": float(k0),
    "z0_init": float(z0_init),
    "sigma_z": float(sigma_z),
    "theta": float(theta),
    "phi": float(phi),
    "diagnostic_controls": {
        "DIAG_EVERY": int(DIAG_EVERY),
        "dt_diag": float(DIAG_EVERY * dt),
        "DEPTH_STEPS": [int(s) for s in DEPTH_STEPS],
        "BL_EPS_STEPS": [int(s) for s in BL_EPS_STEPS],
        "USE_FD_SYMBOL": bool(USE_FD_SYMBOL),
        "REL_POWER_CUTOFF": float(REL_POWER_CUTOFF),
        "R_BRANCH_MIN": float(R_BRANCH_MIN),
        "RDELTA_MAX": float(RDELTA_MAX),
        "EDGE_BAND": int(EDGE_BAND),
        "ACTIVE_REL_THRESHOLD": float(ACTIVE_REL_THRESHOLD),
        "top_row_is_primary_boundary_trace": True,
        "ghost_trace_is_only_consistency_check": bool(ENABLE_GHOST_CHECK),
        "Pi_minus_is_not_neglected_in_central_diagnostic": True,
        "central_density_formula": "rho_exact(eps)=W^dagger exp(-2 eps J) W; rho_linear=rho0+eps R a rho0 with R a rho0=-2 W^dagger J W",
    },
    "detector_budget_toprow_primary": {
        "integrated_detector_rate": float(P_det_toprow_int),
        "norm_loss_diag": float(norm_loss_diag),
        "budget_error": float(det_toprow_budget_error),
        "budget_rel_error": float(det_toprow_budget_rel_error),
        "P0_init": float(P0_init),
        "P_final_diag": float(P_total_series[-1]) if P_total_series.size else np.nan,
    },
    "ghost_trace_check_not_primary": {
        "enabled": bool(ENABLE_GHOST_CHECK),
        "integrated_detector_rate_ghost": float(P_det_ghost_int),
        "budget_error_ghost": float(det_ghost_budget_error),
        "budget_rel_error_ghost": float(det_ghost_budget_rel_error),
    },
    "pauli_backflow_check_not_reflection_proof": {
        "integrated_local_positive": float(P_pauli_plus_local),
        "integrated_local_negative": float(P_pauli_minus_local),
        "local_negative_over_positive": float(R_pauli_local_backflow),
    },
    "zeroth_order_measure_nu0": {
        "Z0": float(Z0),
        "tau0_conditional_mean": float(tau0),
        "F0_RMST_T": float(F0_T),
        "rmst0_T": float(rmst0_T),
        "E_R2": float(E_R2),
        "E_s2": float(E_s2),
        "E_abs_sa_bound": float(abs_sa_bound_mean),
        "edge_fraction": stats_weighted_active(bl_edge_fraction_series, active_weight, active_mask),
        "neutral_R_fraction": stats_weighted_active(bl_neutral_fraction_series, active_weight, active_mask),
        "branch_Q_minus_report_only": stats_weighted_active(bl_branch_Q_minus_series, active_weight, active_mask),
        "branch_eta_minus_report_only": stats_weighted_active(bl_branch_eta_minus_series, active_weight, active_mask),
    },
    "first_order_covariance_coefficients": {
        "definition": "beta_omega = Cov_{nu0}(t, s a_omega), s=R/sqrt(omega), computed via s a rho0 = -2 W^dagger J W / sqrt(omega)",
        "beta_covariance": float(beta_cov),
        "cov_t_Ra_equals_sqrtomega_beta": float(cov_t_Ra),
        "sqrtomega_times_beta": float(sqrt_omega * beta_cov) if np.isfinite(beta_cov) and sqrt_omega > 0 else np.nan,
        "A_sa_mean_E_sa": float(A_sa_mean),
        "B_sa_mean_E_tsa": float(B_sa_mean),
        "A_Ra_mean_E_Ra": float(A_Ra_mean),
        "B_Ra_mean_E_tRa": float(B_Ra_mean),
        "tau_bl_linear_slope_per_epsilon": float(sqrt_omega * beta_cov) if np.isfinite(beta_cov) and sqrt_omega > 0 else np.nan,
        "rmst_gamma_linear_T": float(gamma_rmst_linear),
        "rmst_gamma_Ra_linear_T_equals_sqrtomega_gamma": float(gamma_Ra_rmst_linear),
    },
    "finite_epsilon_exact_vs_taylor": eps_summaries,
    "duhamel_boundary_slab_full_two_branch": duhamel_summary,
}

with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

elapsed = time.time() - start_time
print(f"Total execution time: {elapsed:.2f} s", flush=True)
print(f"[info] Outputs saved in: {out_dir.resolve()}", flush=True)

print("\n[detector budget: top-row discrete primary]")
print(f"  int detector rate = {P_det_toprow_int:.8e}")
print(f"  norm loss         = {norm_loss_diag:.8e}")
print(f"  rel error         = {det_toprow_budget_rel_error:.8e}")

print("\n[new central coefficient]")
print(f"  tau0                       = {tau0:.8e}")
print(f"  beta_cov = Cov(t,s a)      = {beta_cov:.8e}")
print(f"  sqrt(omega)*beta_cov       = {(sqrt_omega * beta_cov) if np.isfinite(beta_cov) else np.nan:.8e}")
print(f"  RMST gamma_linear(T)       = {gamma_rmst_linear:.8e}")
print(f"  E_R2                       = {E_R2:.8e}")
print(f"  E_s2                       = {E_s2:.8e}")

print("\n[exact finite-epsilon vs covariance/taylor]")
for s in BL_EPS_STEPS:
    s = int(s)
    es = eps_summaries[str(s)]
    print(
        f"  eps={s} hz={s*hz:.6e}: "
        f"eps*sqrt(omega)={es['epsilon_sqrt_omega']:.4e}, "
        f"beta_exact={es['exact']['beta_exact_from_tau']:.6e}, "
        f"beta_cov={beta_cov:.6e}, "
        f"L1lin={es['linear_taylor']['linear_density_L1_relative_error']:.3e}, "
        f"Oeps2R2={es['O_eps2_R2_scale']:.3e}"
    )

print("\n[Duhamel full two-branch slab diagnostics]")
for s in DEPTH_STEPS:
    s = int(s)
    ds = duhamel_summary[str(s)]
    print(
        f"  depth={s} hz={s*hz:.6e}: "
        f"E_full_weighted={ds['E_full_two_branch_rel_true']['weighted_mean_active']:.3e}, "
        f"E_linear_weighted={ds['E_linear_wave_rel_true']['weighted_mean_active']:.3e}, "
        f"E_plus_only_weighted={ds['E_plus_only_rel_true_historical']['weighted_mean_active']:.3e}"
    )

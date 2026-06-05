
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-D Crank-Nicolson Schrödinger Solver (GPU-Optimized with CuPy)

This is a *reflection-check* variant:
  - Keeps your CN+GMRES field evolution and BC implementation unchanged.
  - Removes Bohmian trajectories and heavy saving.
  - Adds a windowed-FFT-in-z reflection ratio in a slab below the roof.

Reflection diagnostic:
  Pick a slab below the roof, apply a z-window, FFT in z for each (x,y),
  sum spectral power over (x,y), and form
      R_fft(t) = W_minus(t) / W_plus(t)
  where W_plus is k_z>0 power (upgoing), W_minus is k_z<0 power (downgoing).
"""

import sys, time, os, logging
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags, eye, kron, coo_matrix, csr_matrix, bmat
from cupyx.scipy.sparse.linalg import gmres, LinearOperator
from concurrent.futures import ProcessPoolExecutor


# -------------------- directories / logging --------------------
backend = "CuPy"
print(f"[info] backend: {backend}")

out_dir = Path(os.getenv("OUTDIR", ".")) / "2038(1).Spin_Gauss_DirichletABC_theta=pi6+C_Omega=1_L=20_k0=1_sigma=0.5_T=40"   #21h on A40
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

# -------------------- parameters (keep yours) --------------------
DTYPE_R, DTYPE_C = cp.float32, cp.complex64

Nx = 100
Ny = 100
Nz = 1500
ell = 10.0
omega = float(1.0)

Lx = ell / np.sqrt(omega)
Ly = ell / np.sqrt(omega)
Lz = 20.0

dt = 8e-4
T_final = 40.0
SNAPSHOT_DT = 0.1  # save snapshots every 0.05

Vxy0 = omega**2
V0z = 0.0
alpha = 0.0
k_bc = float(1.0)
# --- Gaussian packet parameters (choose these) ---
z0      = 10.0          # initial center in z (must be > a few sigma_z from z=0)
sigma_z = 0.5         # longitudinal width (in your dimensionless z-units)
k0      = float(1.0) # mean momentum along +z; adjust as needed
#-----Spinor parameter---------
theta = float(np.pi/6.0)
phi   = np.float32(0.0)

# -------------------- mesh / initial state (keep yours) --------------------
x = cp.linspace(0.0, Lx, Nx, endpoint=False, dtype=DTYPE_R)
y = cp.linspace(0.0, Ly, Ny, endpoint=False, dtype=DTYPE_R)
z = cp.linspace(0.0, Lz, Nz, endpoint=False, dtype=DTYPE_R)
Mpart = 100000


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

# -------------------- Bohmian particle positions: Q(0) ~ rho = |psi|^2 --------------------
cp.random.seed(0)
Q = cp.empty((Mpart, 3), dtype=DTYPE_R)

eps = float(min(hx, hy, hz) * 0.5)
z_det = float(cp.asnumpy(z[-1]))   # keep your on-grid detector convention


# x,y: |psi_xy|^2 ∝ exp(-omega * ((x-x0)^2+(y-y0)^2))
# => 1D marginals are Normal(mean, sigma_xy) with sigma_xy = 1/sqrt(2*omega)
sigma_xy = 1.0 / cp.sqrt(2.0 * omega)
x0 = Lx / 2.0
y0 = Ly / 2.0


# exact initialized density on the actual grid
rho0 = (cp.abs(psi_up)**2 + cp.abs(psi_down)**2).astype(cp.float64)

# exclude detector plane and boundary faces for Bohmian starts
rho0[[0, -1], :, :] = 0.0
rho0[:, [0, -1], :] = 0.0
rho0[:, :, 0] = 0.0
rho0[:, :, -1] = 0.0   # avoid starting directly on the detector plane

p = rho0.ravel()
p /= p.sum()

# sample grid cells according to the true discrete Born density
flat_idx = cp.random.choice(p.size, size=Mpart, replace=True, p=p)

ix, iy, iz = cp.unravel_index(flat_idx, rho0.shape)

# uniform jitter inside each selected cell
rx = cp.random.uniform(0.0, 1.0, Mpart)
ry = cp.random.uniform(0.0, 1.0, Mpart)
rz = cp.random.uniform(0.0, 1.0, Mpart)

Q[:, 0] = cp.clip(x[ix] + rx * hx, eps, Lx - eps)
Q[:, 1] = cp.clip(y[iy] + ry * hy, eps, Ly - eps)
Q[:, 2] = cp.clip(z[iz] + rz * hz, eps, z_det - eps)

traj = []
#vel_list = []
#grad_up_list = []
#grad_down_list = []

# Free memory: large construction-only arrays (correct for Gaussian init)
del X, Y, Z, Rho2, psi_xy, psi_z, psi_scalar, rho0
cp.get_default_memory_pool().free_all_blocks()

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


constants = {
    # grid
    "Nx": int(Nx), "Ny": int(Ny), "Nz": int(Nz),
    "Lx": float(Lx), "Ly": float(Ly), "Lz": float(Lz),
    "hx": float(hx), "hy": float(hy), "hz": float(hz),

    # time
    "dt": float(dt), "T_final": float(T_final),

    # physics
    "omega": float(omega),
    "Vxy0": float(Vxy0),
    "V0z": float(V0z),
    "alpha": float(alpha),
    "k_bc": float(k_bc),
    # spin
    "theta": float(theta),
    "phi": float(phi),
    # particles / RNG
    "Mpart": int(Mpart),
    "seed": 0,
    "eps": float(eps),
    # dtypes (store as strings)
    "DTYPE_R": "float32",
    "DTYPE_C": "complex64",
    "z0": float(z0),
    "sigma_z": float(sigma_z),
    "k0": float(k0),
    "SNAPSHOT_DT": float(SNAPSHOT_DT),
    "z_det": float(z_det),
    # init slab mask in physical z
}
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
print(f"SNAPSHOT_DT: {SNAPSHOT_DT}")
print(f"z_det: {z_det}")
print(f"k0: {k0}")
print(f"z0: {z0}")
print(f"Mpart {Mpart}")
print(f"sigma_z: {sigma_z}")

print("-------------------------------------------")



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

# Precompute reciprocals (avoid repeated division in hot loop)
_inv_hx  = 1.0 / hx
_inv_hy  = 1.0 / hy
_inv_hz  = 1.0 / hz
_inv_2hx = 1.0 / (2.0 * hx)
_inv_2hy = 1.0 / (2.0 * hy)
_inv_2hz = 1.0 / (2.0 * hz)
_inv_12hx = 1.0 / (12.0 * hx)
_inv_12hy = 1.0 / (12.0 * hy)
_inv_12hz = 1.0 / (12.0 * hz)


def grad2(u, h, ax):
    """2nd-order central + one-sided at boundaries.  Slicing only (no cp.roll)."""
    g = cp.empty_like(u)
    if ax == 0:
        g[1:-1, :, :] = (u[2:, :, :] - u[:-2, :, :]) * _inv_2hx
        g[0, :, :]    = (u[1, :, :]  - u[0, :, :])    * _inv_hx
        g[-1, :, :]   = (u[-1, :, :] - u[-2, :, :])   * _inv_hx
    elif ax == 1:
        g[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) * _inv_2hy
        g[:, 0, :]    = (u[:, 1, :]  - u[:, 0, :])    * _inv_hy
        g[:, -1, :]   = (u[:, -1, :] - u[:, -2, :])   * _inv_hy
    else:
        g[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) * _inv_2hz
        g[:, :, 0]    = (u[:, :, 1]  - u[:, :, 0])    * _inv_hz
        g[:, :, -1]   = (u[:, :, -1] - u[:, :, -2])   * _inv_hz
    return g


def grad4(u, h, ax):
    """4th-order central interior + 2nd/1st-order boundaries.  Slicing only."""
    g = cp.empty_like(u)
    if ax == 0:
        g[2:-2, :, :] = (-u[4:, :, :] + 8.0*u[3:-1, :, :]
                          - 8.0*u[1:-3, :, :] + u[:-4, :, :]) * _inv_12hx
        g[1, :, :]  = (-3.0*u[1, :, :] + 4.0*u[2, :, :] - u[3, :, :]) * _inv_2hx
        g[-2, :, :] = (u[-4, :, :] - 4.0*u[-3, :, :] + 3.0*u[-2, :, :]) * _inv_2hx
        g[0, :, :]  = (u[1, :, :]  - u[0, :, :])  * _inv_hx
        g[-1, :, :] = (u[-1, :, :] - u[-2, :, :]) * _inv_hx
    elif ax == 1:
        g[:, 2:-2, :] = (-u[:, 4:, :] + 8.0*u[:, 3:-1, :]
                          - 8.0*u[:, 1:-3, :] + u[:, :-4, :]) * _inv_12hy
        g[:, 1, :]  = (-3.0*u[:, 1, :] + 4.0*u[:, 2, :] - u[:, 3, :]) * _inv_2hy
        g[:, -2, :] = (u[:, -4, :] - 4.0*u[:, -3, :] + 3.0*u[:, -2, :]) * _inv_2hy
        g[:, 0, :]  = (u[:, 1, :]  - u[:, 0, :])  * _inv_hy
        g[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) * _inv_hy
    else:
        g[:, :, 2:-2] = (-u[:, :, 4:] + 8.0*u[:, :, 3:-1]
                          - 8.0*u[:, :, 1:-3] + u[:, :, :-4]) * _inv_12hz
        g[:, :, 1]  = (-3.0*u[:, :, 1] + 4.0*u[:, :, 2] - u[:, :, 3]) * _inv_2hz
        g[:, :, -2] = (u[:, :, -4] - 4.0*u[:, :, -3] + 3.0*u[:, :, -2]) * _inv_2hz
        g[:, :, 0]  = (u[:, :, 1]  - u[:, :, 0])  * _inv_hz
        g[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) * _inv_hz
    return g


# ================================================================
#  OPTIMISED trilinear interpolation (precomputed indices/weights)
# ================================================================

def _precompute_interp(Q, x, y, z):
    """Compute cell indices and 8 trilinear weights ONCE for a set of particles."""
    i = cp.clip(((Q[:, 0] - x[0]) * _inv_hx).astype(cp.int32), 0, Nx - 2)
    j = cp.clip(((Q[:, 1] - y[0]) * _inv_hy).astype(cp.int32), 0, Ny - 2)
    k = cp.clip(((Q[:, 2] - z[0]) * _inv_hz).astype(cp.int32), 0, Nz - 2)

    sx = (Q[:, 0] - x[i]) * _inv_hx
    sy = (Q[:, 1] - y[j]) * _inv_hy
    sz = (Q[:, 2] - z[k]) * _inv_hz

    osx = 1.0 - sx;  osy = 1.0 - sy;  osz = 1.0 - sz
    return (i, j, k,
            osx * osy * osz,   # w000
            sx  * osy * osz,   # w100
            osx * sy  * osz,   # w010
            sx  * sy  * osz,   # w110
            osx * osy * sz,    # w001
            sx  * osy * sz,    # w101
            osx * sy  * sz,    # w011
            sx  * sy  * sz)    # w111


def _interp_fast(arr, iw):
    """Trilinear lookup using precomputed (i,j,k, w000..w111) tuple."""
    i, j, k, w000, w100, w010, w110, w001, w101, w011, w111 = iw
    return (w000 * arr[i,   j,   k]   +
            w100 * arr[i+1, j,   k]   +
            w010 * arr[i,   j+1, k]   +
            w110 * arr[i+1, j+1, k]   +
            w001 * arr[i,   j,   k+1] +
            w101 * arr[i+1, j,   k+1] +
            w011 * arr[i,   j+1, k+1] +
            w111 * arr[i+1, j+1, k+1])


# keep old interp3 for step_with_cross (which calls velocity with different Q each sub-step)
def interp3(arr, Q, x, y, z):
    iw = _precompute_interp(Q, x, y, z)
    return _interp_fast(arr, iw)


# ================================================================
#  OPTIMISED velocity (fused: no redundant |ψ|², shared interp)
# ================================================================

def clip_speed(v, *, dt, hmin, cfl=0.9):
    vmax = cfl * hmin / dt
    n = cp.linalg.norm(v, axis=1) + 1e-30
    s = cp.minimum(1.0, vmax / n)
    return v * s[:, None]


def velocity(Q, psi_up, psi_down, d_up, d_down, x, y, z, hx, hy, hz, dt):
    # --- densities: compute |ψ_↑|² and |ψ_↓|² ONCE, reuse for ρ AND S_z ---
    abs2_up   = psi_up.real**2   + psi_up.imag**2     # avoids abs() + square
    abs2_down = psi_down.real**2 + psi_down.imag**2
    rho = abs2_up + abs2_down

    # convective current J
    J = [cp.imag(psi_up.conj()*d_up[i] + psi_down.conj()*d_down[i]) for i in range(3)]

    # spin density S = ψ†σψ  (reuse abs2)
    up_conj_down = psi_up.conj() * psi_down
    Sx = 2.0 * up_conj_down.real       # cp.real returns a view → fast
    Sy = 2.0 * up_conj_down.imag
    Sz = abs2_up - abs2_down            # ← no recomputation

    # curl of S  (2nd-order, slicing-based grad2)
    dSz_dy = grad2(Sz, hy, 1);  dSy_dz = grad2(Sy, hz, 2)
    dSx_dz = grad2(Sx, hz, 2);  dSz_dx = grad2(Sz, hx, 0)
    dSy_dx = grad2(Sy, hx, 0);  dSx_dy = grad2(Sx, hy, 1)
    curl_x = dSz_dy - dSy_dz
    curl_y = dSx_dz - dSz_dx
    curl_z = dSy_dx - dSx_dy

    # --- interpolate: precompute indices/weights ONCE for all 7 lookups ---
    iw = _precompute_interp(Q, x, y, z)

    rho_Q  = _interp_fast(rho, iw)
    Jx_Q   = _interp_fast(J[0], iw)
    Jy_Q   = _interp_fast(J[1], iw)
    Jz_Q   = _interp_fast(J[2], iw)
    cx_Q   = _interp_fast(curl_x, iw)
    cy_Q   = _interp_fast(curl_y, iw)
    cz_Q   = _interp_fast(curl_z, iw)

    J_Q    = cp.stack([Jx_Q, Jy_Q, Jz_Q], axis=1)
    curl_Q = cp.stack([cx_Q, cy_Q, cz_Q], axis=1)

    # dynamic density floor
    rho_max = float(cp.max(rho)) + 1e-30
    den = cp.maximum(rho_Q, cp.asarray(1e-6 * rho_max, dtype=rho_Q.dtype))

    v = (J_Q + 0.5*curl_Q) / den[:, None]
    v = cp.where(cp.isfinite(v), v, 0.0)
    return v


# ================================================================
#  Particle stepping (UNCHANGED logic)
# ================================================================

def step_with_cross(idx, dt_local, t_base):
    """One RK2 step for particles idx, with on-grid first-crossing detection."""
    if idx.size == 0:
        return
    Q_old = Q[idx]

    # your RK2 with CLIPPED velocities (simple + stable)
    v0_raw = velocity(Q_old, psi_up, psi_down, d_up, d_down, x, y, z, hx, hy, hz, dt_local)
    v0     = clip_speed(v0_raw, dt=dt_local, hmin=min(hx,hy,hz), cfl=0.8)
    Qh     = Q_old + 0.5*dt_local*v0
    vh_raw = velocity(Qh,    psi_up, psi_down, d_up, d_down, x, y, z, hx, hy, hz, dt_local)
    vh     = clip_speed(vh_raw, dt=dt_local, hmin=min(hx,hy,hz), cfl=0.8)
    Q_new  = Q_old + dt_local*vh

    # FIRST-CROSSING (linear in z over the step)
    z0 = Q_old[:,2]; z1 = Q_new[:,2]
    cross = (z0 < z_det) & (z1 >= z_det)
    if bool(cp.any(cross)):
        dz = z1[cross] - z0[cross]
        dz = cp.where(cp.abs(dz) < 1e-30, cp.sign(dz)*1e-30, dz)
        frac = cp.clip((z_det - z0[cross]) / dz, 1e-12, 1.0 - 1e-12) # fraction of the step
        # place exactly on the plane (single hit)
        Q_new[cross] = Q_old[cross] + frac[:,None]*(Q_new[cross] - Q_old[cross])
        Q_new[cross, 2] = z_det
        # timestamp for first-hit
        t_hit[idx[cross]] = t_base + frac.astype(cp.float64)*dt_local
        arrived[idx[cross]] = True

    # side/bottom reflections for everyone
    tmp = Q_new[:,0] < 0;  Q_new[tmp,0] = -Q_new[tmp,0]
    tmp = Q_new[:,0] > Lx; Q_new[tmp,0] =  2*Lx - Q_new[tmp,0]
    tmp = Q_new[:,1] < 0;  Q_new[tmp,1] = -Q_new[tmp,1]
    tmp = Q_new[:,1] > Ly; Q_new[tmp,1] =  2*Ly - Q_new[tmp,1]
    tmp = Q_new[:,2] < 0;  Q_new[tmp,2] = -Q_new[tmp,2]

    Q[idx] = Q_new


#--- define physical snapshot times, independent of dt --------------------------------------------------- 
snapshot_times = np.arange(SNAPSHOT_DT, T_final + 1e-9, SNAPSHOT_DT)

# map from time-step index n to the *nominal* snapshot time τ
snapshot_map = {
    int(round(t_snap / dt)): float(t_snap)
    for t_snap in snapshot_times
}
snapshot_steps = set(snapshot_map.keys())

print("[info] snapshot times:", snapshot_times)
print("[info] snapshot steps:", sorted(list(snapshot_steps))[:10], "...")



# ── simulation bookkeeping ──────────────────────────────────────---------------------------------------------- 
num_steps = int(round(T_final/dt))
prob_steps_cpu = set(to_cpu(cp.linspace(1,num_steps,1000,dtype=int)).tolist())
view_steps_cpu = set(to_cpu(cp.linspace(1,num_steps,10, dtype=int)).tolist())
total_probs = []
prob_times = []

# --- choose a small, fixed subset to track (no physics change) ---
K = 20
rng = np.random.RandomState(0)  # reproducible choice
sel_idx_cpu = rng.choice(Mpart, size=K, replace=False)
sel_idx = cp.asarray(sel_idx_cpu, dtype=cp.int32)
np.save(out_dir / "traj_indices.npy", sel_idx_cpu)  # so you know which particles

# preallocate storage for the selected trajectories (CPU array is fine)
traj_sel = np.empty((num_steps, K, 3), dtype=np.float32)


# ---------- detector on-grid ----------
z_det = float(cp.asnumpy(z[-1]))   # ON-GRID: last z node (not Lz)
DET_EPS = float(min(1e-6, 0.1*hz)) # tiny cushion below the plane

# first-arrival bookkeeping (GPU arrays)
arrived = cp.zeros(Mpart, dtype=cp.bool_)
t_hit   = cp.full(Mpart, cp.nan, dtype=cp.float64)
hmin = min(hx, hy, hz)    

# ── main time loop ─────────────────────────────────────────────────
print(f"[info] Starting time loop – {num_steps} steps")
start_time = time.time()
plot_pool = ProcessPoolExecutor(max_workers=2)
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
        
    # ── Gradients (OPTIMISED: slicing, no cp.roll) ─────────────────
    d_up   = [grad4(psi_up,   hx, 0), grad4(psi_up,   hy, 1), grad2(psi_up,   hz, 2)]
    d_down = [grad4(psi_down, hx, 0), grad4(psi_down, hy, 1), grad2(psi_down, hz, 2)]
 
    # decide who needs substeps: near roof or high Courant
    v_now_raw = velocity(Q, psi_up, psi_down, d_up, d_down, x, y, z, hx, hy, hz, dt)
    courant = cp.linalg.norm(v_now_raw, axis=1) * dt / hmin
    
    active = ~arrived
    cap_rate = float(cp.mean((courant[active] > 0.8).astype(cp.float32))) if bool(cp.any(active)) else 0.0
    if n % 200 == 0:
        print(f"[step {n:6d}] capped fraction = {cap_rate:.4f}", flush=True)
    
    near_roof = Q[:, 2] > (z_det - 2*hz)
    needs_sub = (courant > 0.6) | near_roof
    t_base = (n-1)*dt
    
    # split into active sets (skip already-arrived)
    active = ~arrived
    idx_sub  = cp.where(active & needs_sub)[0]
    idx_rest = cp.where(active & ~needs_sub)[0]
    
    # two half-steps for substeppers (detect crossing each half)
    if idx_sub.size > 0:
        dt2 = 0.5*dt
        step_with_cross(idx_sub, dt2, t_base)
    
        # only those *in idx_sub* that are still not arrived
        still_mask = ~arrived[idx_sub]
        idx_sub2 = idx_sub[still_mask]
        if idx_sub2.size > 0:
            step_with_cross(idx_sub2, dt2, t_base + dt2)
   
    # one full step for the rest
    if idx_rest.size > 0:
        step_with_cross(idx_rest, dt, t_base)
        
    # pin arrivals exactly on the plane (harmless if already pinned)
    Q[arrived, 2] = z_det
    # record the selected particle positions at this step
    traj_sel[n-1, :, :] = to_cpu(Q[sel_idx])

    # ── diagnostics / I-O cadence (UNCHANGED) ────────────────────────────
    if (n % 100 == 0) or (n in prob_steps_cpu):
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)*(hx*hy*hz)))
        total_probs.append(total_prob)
        prob_times.append(n*dt)
        logging.info(f"[Probability] t={n*dt:.2f}, ∫|ψ|² dV = {total_prob:.6f}")
        # save to disk (overwrite each time)
        np.save(out_dir/"prob_times.npy", np.array(prob_times))
        np.save(out_dir/"total_probs.npy", np.array(total_probs))
        
        
    
    # (2) density snapshots at fixed physical times τ = SNAPSHOT_DT, 2*SNAPSHOT_DT, ...
    if n in snapshot_steps:
        tau_snap = snapshot_map[n]  # nominal snapshot time
        rho_prob = to_cpu(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)
        # filename uses the *nominal* snapshot time with 5 decimals
        np.save(out_dir / f"rho_prob_t{tau_snap:.5f}.npy", rho_prob)


       
    # optional live-view at sparse intervals (unchanged numerics)
    if n in view_steps_cpu:
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_3d[0])**2 + cp.abs(psi_3d[1])**2)*(hx*hy*hz)))
        print(f"[Probability] t={n*dt:.2f}, Total Probability = {total_prob:.6f}", flush=True)
    
    
# ── final flush (UNCHANGED) ────────────────────────────────────────────────────
np.save(out_dir/"prob_times.npy", np.array(prob_times))
np.save(out_dir/"total_probs.npy", np.array(total_probs))
np.save(out_dir/"bohm_t_hit.npy",        cp.asnumpy(t_hit))
np.save(out_dir/"bohm_arrived_mask.npy", cp.asnumpy(arrived))
# save only the selected trajectories (compact)
np.save(out_dir / "bohmian_traj_selected.npy", traj_sel)
# also save arrival info restricted to the same selected indices
np.save(out_dir / "bohm_t_hit_selected.npy",        cp.asnumpy(t_hit[sel_idx]))
np.save(out_dir / "bohm_arrived_mask_selected.npy", cp.asnumpy(arrived[sel_idx]))
bohm_times = np.arange(1, num_steps + 1) * dt
np.save(out_dir/"bohm_times.npy", bohm_times)
elapsed = time.time()-start_time
print(f"Total execution time: {elapsed:.2f} s", flush=True)
print(f"[info] All output files are in: {out_dir.resolve()}")
print(f"[info] Log saved to {out_dir/'simulation_log.txt'}")


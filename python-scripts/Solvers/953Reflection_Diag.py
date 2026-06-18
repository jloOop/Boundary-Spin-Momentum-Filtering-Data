#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reflection and finite-window diagnostic solver for the spinor boundary response.

Repository role
---------------
This script is the representative diagnostic run for distinguishing ordinary
propagating reflection from near-roof evanescent/slab-storage behavior.  It is
connected to the reflection checks and early/late finite-window interpretation
used around the Supplemental Material discussion of finite-window bookkeeping.

It keeps the same basic CN/GMRES finite-difference evolution architecture but
adds wave-function diagnostics rather than Bohmian trajectory sampling:

    1. roof-grid Pauli current/backflow monitor,
    2. windowed longitudinal FFT in a slab below the roof,
    3. tangential Pi_+/Pi_- decomposition of the spinor-ABC boundary symbol,
    4. homogeneous two-branch / Duhamel relative-error checks,
    5. first-pass-style summary window to separate early reflected signal from
       later bottom-return contamination.

Main physical question
----------------------
Does the boundary generate a clean propagating reflected wave, or mainly a
near-roof spin--momentum-filtered sector whose later effect appears through
finite-guide memory?

Important reader note
---------------------
The diagnostic code is intentionally conservative: it does not alter the CN
field evolution inside the diagnostic loop.  Added comments explain the role of
blocks for a GitHub reader; the working numerical statements are preserved.
"""


import sys, time, os, logging, json
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags, eye, kron, coo_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import gmres, LinearOperator


# =============================================================================
# Run configuration and output directory
# =============================================================================
# This diagnostic is intended to be run as a controlled reflection/backflow test.
# OMEGA and OUTDIR may be supplied externally for small parameter sweeps.
# -------------------- directories / logging --------------------


OMEGA_ENV = os.getenv("OMEGA", None)

backend = "CuPy"
print(f"[info] backend: {backend}")

out_dir = Path(os.getenv("OUTDIR", ".")) / "3003.ReflectionDiag_SpinorABC_theta=0+C_Omega=100" 
out_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(out_dir / "simulation_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# -------------------- utilities --------------------


def to_cpu(arr):
    return cp.asnumpy(arr)


# =============================================================================
# Physical and numerical parameters
# =============================================================================
# These values define the reflection-diagnostic run.  The body below keeps the
# TDSE/CN evolution structure unchanged and attaches diagnostics to it.
# -------------------- parameters (UNCHANGED TDSE/CN part) --------------------
DTYPE_R, DTYPE_C = cp.float32, cp.complex64

Nx = 100
Ny = 100
Nz = 1500


omega = float(OMEGA_ENV) if (OMEGA_ENV is not None) else float(100.0)

ell = 10.0

Lx = ell / np.sqrt(omega)
Ly = ell / np.sqrt(omega)
Lz = 20.0

dt = 5e-4
T_final =5.2

Vxy0 = omega**2
V0z = 0.0
alpha = 0.0



# Diagnostic boundary setting.
# In this reflection script k_bc is kept exactly as in the working diagnostic
# version.  Do not silently replace it by the absorbing-production value used in
# the main confinement solver; this file tests a specific reflection setup.
beta_R = float(np.pi)
k_bc = -1j * beta_R

z0_init  = 10.0
sigma_z  = 0.5

k0       = float(np.pi)

theta = float(0.0)
phi   = np.float32(0.0)

# -------------------- mesh / initial state (UNCHANGED) --------------------
# Mesh and initial condition
x = cp.linspace(0.0, Lx, Nx, endpoint=False, dtype=DTYPE_R)
y = cp.linspace(0.0, Ly, Ny, endpoint=False, dtype=DTYPE_R)
z = cp.linspace(0.0, Lz, Nz, endpoint=False, dtype=DTYPE_R)
hx = float(x[1] - x[0])
hy = float(y[1] - y[0])
hz = float(z[1] - z[0])

X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
mid_x = Nx // 2
mid_y = Ny // 2
mid_z = Nz // 2


Rho2 = (X - Lx/2)**2 + (Y - Ly/2)**2
psi_xy = cp.exp(-omega * Rho2 / 2)

psi_z = cp.exp(-(Z - z0_init)**2 / (2 * sigma_z**2)) * cp.exp(1j * k0 * (Z - z0_init))

psi_scalar = (psi_xy * psi_z).astype(DTYPE_C)

# Define initial spin state (general: both up and down)
c_up = cp.cos(theta / 2)
c_down = cp.sin(theta / 2) * cp.exp(1j * phi)

# Apply to scalar part (no phase term as requested)
psi_up = c_up * psi_scalar
psi_down = c_down * psi_scalar

# Enforce initial Dirichlet BCs
psi_up[[0, -1], :, :] = 0
psi_up[:, [0, -1], :] = 0
psi_up[:, :, 0] = 0
psi_down[[0, -1], :, :] = 0
psi_down[:, [0, -1], :] = 0
psi_down[:, :, 0] = 0

# Normalize the wavefunction (now for full spinor)
def normalize_wavefunction(psi_up, psi_down, hx, hy, hz):
    """Normalize the full spinor wavefunction to ensure total probability = 1."""
    total_prob = cp.sum(cp.abs(psi_up)**2 + cp.abs(psi_down)**2) * (hx * hy * hz)
    norm = cp.sqrt(total_prob)
    psi_up /= norm
    psi_down /= norm
    return psi_up, psi_down

psi_up, psi_down = normalize_wavefunction(psi_up, psi_down, hx, hy, hz) # Normalize total
psi_flat = cp.concatenate((psi_up.ravel(), psi_down.ravel())).astype(DTYPE_C)



# Harmonic oscillator potential in x and y directions
V_real = 0.5 * Vxy0 * ((X - Lx/2)**2 + (Y - Ly/2)**2)

# Absorbing potential 

# ---- complex absorbing layer (smooth ramp) near the roof ----
        # strictly non-positive imaginary part
V_abs=0.0   #Now @ this *Moment there is no absorpin pot

V = V_real.astype(DTYPE_C) + V_abs            # <-- replace your old V definition
V_diag = V.ravel().astype(DTYPE_C)
V_full = cp.concatenate((V_diag, V_diag))


# ── save coordinate grids & constants once ─────────────────────────
np.save(out_dir / "x.npy", to_cpu(x))
np.save(out_dir / "y.npy", to_cpu(y))
np.save(out_dir / "z.npy", to_cpu(z))

constants = {
    "hx": hx, "hy": hy, "hz": hz,
    "Lx": Lx, "Ly": Ly, "Lz": Lz,
    "dt": dt, "T_final": T_final,
    "Nx": Nx, "Ny": Ny, "Nz": Nz,
    "omega": float(omega),
    "k0": float(k0),
    "z0_init": float(z0_init),
    "sigma_z": float(sigma_z),
    "theta": float(theta),
    "phi": float(phi),
}

np.savez(out_dir / "constants.npz", **constants)
logging.info("[info] Saved coordinate grids and constants")
# <<< END MOVED BLOCK

# frees hundreds of MB, no effect on solver or outputs
del X, Y, Z, Rho2, psi_xy, psi_z, psi_scalar


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
print(f"mid_x: {mid_x}")
print(f"mid_y: {mid_y}")
print(f"mid_z: {mid_z}")
print(f"out_dir: {out_dir}")
#print(f"mask: (Z > 0) & (Z < 1) # Width 1, centered at ~14.5")
print("-------------------------")


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
                # Modified for Dirichlet: omit left, adjust diag
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
            indptr.append(indptr[-1]) # Empty row for Dirichlet
        else:
            left = i-1
            right = i+1
            if left == 0:
                indices.extend([i, right])
                data.append(inv_d2)
                data.append(-half_inv_d2)
                indptr.append(indptr[-1] + 2)
            elif right == N-1:
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



# =============================================================================
# Sparse finite-difference Hamiltonian assembly
# =============================================================================
# Same architecture as the main solver: scalar finite-difference pieces plus the
# roof-layer spinor coupling C implementing the tangential derivative terms.
# Laplacian with BCs
L1 = L_z(Nz, hz, k_bc)
L2 = L_dirichlet(Nx, hx)
L3 = L_dirichlet(Ny, hy)
Ix = eye(Nx, dtype=DTYPE_C, format='csr')
Iy = eye(Ny, dtype=DTYPE_C, format='csr')
Iz = eye(Nz, dtype=DTYPE_C, format='csr')

Lap_scalar = (kron(L2, kron(Iy, Iz)) + kron(Ix, kron(L3, Iz)) + kron(Ix, kron(Iy, L1))).tocsr()
Lap_diag = kron(eye(2, dtype=DTYPE_C, format='csr'), Lap_scalar)

# Build coupling matrix C for spinor ABC at top boundary (Hermitian when k_bc=0)
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


# print('--------------------------------------------------------\n')
# # -----------------
# Crank-Nicolson matrices
Ntot = 2 * Nx * Ny * Nz
Id = eye(Ntot, dtype=DTYPE_C, format='csr')

Potential = diags(V_full, 0, format='csr')
A = Id + 1j*dt/2 * (Lap_spinor + Potential).tocsr()
B = Id - 1j*dt/2 * (Lap_spinor + Potential).tocsr()
inv_diag = 1.0 / A.diagonal()
M = LinearOperator(shape=A.shape, matvec=lambda x: inv_diag * x)

# # build once outside loop
A64 = A.astype(cp.complex128)
B64 = B.astype(cp.complex128)
inv_diag64 = 1.0 / A64.diagonal()
M64 = LinearOperator(shape=A64.shape, matvec=lambda x: inv_diag64 * x)



# =============================================================================
# Diagnostic controls
# =============================================================================
# DIAG_EVERY sets the sampling cadence.  The kz slab is deliberately below the
# roof and excludes the ABC row, so the FFT diagnostic targets propagating
# longitudinal content rather than the boundary trace itself.
# =============================================================================
# Spectral reflection diagnostics
# =============================================================================

DIAG_EVERY = 20          # dt_samp = DIAG_EVERY * dt = 0.01 for dt=5e-4
PROB_EVERY = 100

# Longitudinal scattering-style kz diagnostic slab.
# Choose a slab below the roof, not including the ABC row.
KZ_SLAB_ZMIN = Lz - 5.0
KZ_SLAB_ZMAX = Lz - 0.8

# Near-roof slab mass diagnostic.
NEAR_SLAB_DEPTH = 1.0

# Duhamel / Pi diagnostic depths measured from physical boundary z=Lz.
# depth_steps=1 means z=L-hz, i.e. the top stored grid plane.
DEPTH_STEPS = [1, 2, 4, 8, 16, 32, 64]

# Avoid interpreting zero kz as either incident or reflected.
KZ_ZERO_EPS_FACTOR = 0.5

# For the Pi_- growing branch, avoid numerical-noise blowup at very high R*delta.
RDELTA_MAX = 18.0

# Ignore spectral modes whose boundary amplitude is pure roundoff.
REL_POWER_CUTOFF = 1e-12
USE_FD_SYMBOL = True
EDGE_BAND = 4



# =============================================================================
# Current, derivative, and boundary-trace helpers
# =============================================================================
# These routines are diagnostic only.  They evaluate Pauli-current fluxes and
# reconstruct the roof trace needed for tangential branch analysis.
# -------------------- derivative helpers (keep your existing dz/dx/dy helpers if already defined) --------------------
def dz_slice_2nd(u, z_plane, hz):
    if z_plane == 0:
        return (u[:, :, 1] - u[:, :, 0]) / hz
    elif z_plane == u.shape[2] - 1:
        return (u[:, :, -1] - u[:, :, -2]) / hz
    else:
        return (u[:, :, z_plane+1] - u[:, :, z_plane-1]) / (2*hz)

def dx_2nd_2d(f2d, hx):
    out = cp.empty_like(f2d, dtype=cp.float64)
    f2d = f2d.astype(cp.float64, copy=False)
    out[1:-1, :] = (f2d[2:, :] - f2d[:-2, :]) / (2.0 * hx)
    out[0, :]    = (f2d[1, :] - f2d[0, :]) / hx
    out[-1, :]   = (f2d[-1, :] - f2d[-2, :]) / hx
    return out

def dy_2nd_2d(f2d, hy):
    out = cp.empty_like(f2d, dtype=cp.float64)
    f2d = f2d.astype(cp.float64, copy=False)
    out[:, 1:-1] = (f2d[:, 2:] - f2d[:, :-2]) / (2.0 * hy)
    out[:, 0]    = (f2d[:, 1] - f2d[:, 0]) / hy
    out[:, -1]   = (f2d[:, -1] - f2d[:, -2]) / hy
    return out

def plane_pauli_fluxes(psi_up, psi_down, zp):
    """
    Returns:
      Jnet_total, Jnet_conv, Jnet_spin,
      Jplus_loc_total, Jminus_loc_total,
      Jplus_loc_conv,  Jminus_loc_conv
    """
    u = psi_up[:, :, zp]
    d = psi_down[:, :, zp]

    duz = dz_slice_2nd(psi_up,   zp, hz)
    ddz = dz_slice_2nd(psi_down, zp, hz)

    j_conv = cp.imag(u.conj() * duz + d.conj() * ddz).astype(cp.float64)

    c  = u.conj() * d
    Sx = (2.0 * cp.real(c)).astype(cp.float64)
    Sy = (2.0 * cp.imag(c)).astype(cp.float64)

    dSydx = dx_2nd_2d(Sy, hx)
    dSxdy = dy_2nd_2d(Sx, hy)
    j_spin = (0.5 * (dSydx - dSxdy)).astype(cp.float64)

    jz = j_conv + j_spin

    Jnet_total = cp.sum(jz)     * (hx * hy)
    Jnet_conv  = cp.sum(j_conv) * (hx * hy)
    Jnet_spin  = cp.sum(j_spin) * (hx * hy)

    Jplus_loc_total  = cp.sum(cp.maximum(jz,     0.0)) * (hx * hy)
    Jminus_loc_total = cp.sum(cp.maximum(-jz,    0.0)) * (hx * hy)

    Jplus_loc_conv   = cp.sum(cp.maximum(j_conv, 0.0)) * (hx * hy)
    Jminus_loc_conv  = cp.sum(cp.maximum(-j_conv,0.0)) * (hx * hy)

    return (float(Jnet_total.get()), float(Jnet_conv.get()), float(Jnet_spin.get()),
            float(Jplus_loc_total.get()), float(Jminus_loc_total.get()),
            float(Jplus_loc_conv.get()),  float(Jminus_loc_conv.get()))



def dx_complex_2d(f2d, hx):
    f2d = f2d.astype(cp.complex128, copy=False)
    out = cp.empty_like(f2d, dtype=cp.complex128)
    out[1:-1, :] = (f2d[2:, :] - f2d[:-2, :]) / (2.0 * hx)
    out[0, :]    = (f2d[1, :] - f2d[0, :]) / hx
    out[-1, :]   = (f2d[-1, :] - f2d[-2, :]) / hx
    return out

def dy_complex_2d(f2d, hy):
    f2d = f2d.astype(cp.complex128, copy=False)
    out = cp.empty_like(f2d, dtype=cp.complex128)
    out[:, 1:-1] = (f2d[:, 2:] - f2d[:, :-2]) / (2.0 * hy)
    out[:, 0]    = (f2d[:, 1] - f2d[:, 0]) / hy
    out[:, -1]   = (f2d[:, -1] - f2d[:, -2]) / hy
    return out


def roof_trace_from_abc(psi_up, psi_down):
    """
    Reconstruct the ghost/boundary trace at the physical roof z=Lz
    from the stored top interior plane z=Lz-hz using the spinor ABC:

        ∂z psi_up   = i*kappa psi_up - (∂x - i∂y) psi_down
        ∂z psi_down = i*kappa psi_down + (∂x + i∂y) psi_up

    with forward ghost:
        psi(L) ≈ psi(L-hz) + hz * ∂z psi(L-hz).
    """
    u = psi_up[:, :, z_roof_idx].astype(cp.complex128, copy=False)
    d = psi_down[:, :, z_roof_idx].astype(cp.complex128, copy=False)

    dxd = dx_complex_2d(d, hx)
    dyd = dy_complex_2d(d, hy)
    dxu = dx_complex_2d(u, hx)
    dyu = dy_complex_2d(u, hy)

    Dminus_d = dxd - 1j * dyd
    Dplus_u  = dxu + 1j * dyu

    uL = u + hz * (1j * k_bc * u - Dminus_d)
    dL = d + hz * (1j * k_bc * d + Dplus_u)

    # enforce sidewall zeros on reconstructed trace
    uL[[0, -1], :] = 0
    uL[:, [0, -1]] = 0
    dL[[0, -1], :] = 0
    dL[:, [0, -1]] = 0

    return uL, dL


kx = 2.0 * np.pi * cp.fft.fftfreq(Nx, d=hx)
ky = 2.0 * np.pi * cp.fft.fftfreq(Ny, d=hy)
KX, KY = cp.meshgrid(kx, ky, indexing="ij")

if USE_FD_SYMBOL:
    XI_X = cp.sin(KX * hx) / hx
    XI_Y = cp.sin(KY * hy) / hy
else:
    XI_X = KX
    XI_Y = KY

R_TAN = cp.sqrt(XI_X**2 + XI_Y**2)
R_SAFE = cp.where(R_TAN > 0, R_TAN, 1.0)

XI_PLUS  = XI_X + 1j * XI_Y
XI_MINUS = XI_X - 1j * XI_Y



def transverse_sidewall_mass_2d(u, d, band=EDGE_BAND):
    """
    Fraction of roof-trace mass lying in a near-sidewall band.

    This is a diagnostic for whether the transverse FFT approximation is safe.
    Since the exact boundary rows are Dirichlet-zero by construction, we test a
    finite band of width `band` grid cells, not only the outermost rows.
    """
    rho = cp.abs(u)**2 + cp.abs(d)**2

    mask = cp.zeros(rho.shape, dtype=cp.bool_)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True

    edge = cp.sum(cp.where(mask, rho, 0.0)) * hx * hy
    total = cp.sum(rho) * hx * hy

    return float(edge.get()), float(total.get())




def apply_projectors(uhat, dhat):
    """
    Apply Pi_+ and Pi_- for the tangential spinor ABC matrix.

    J/R acting on (u,d):
        (J/R psi)_up   = (-i xi_- / R) d
        (J/R psi)_down = ( i xi_+ / R) u

    At R=0 we set J/R=0, so Pi_+=Pi_-=1/2 I.
    """
    jr_u = (-1j * XI_MINUS / R_SAFE) * dhat
    jr_d = ( 1j * XI_PLUS  / R_SAFE) * uhat

    jr_u = cp.where(R_TAN > 0, jr_u, 0.0)
    jr_d = cp.where(R_TAN > 0, jr_d, 0.0)

    p_u = 0.5 * (uhat + jr_u)
    p_d = 0.5 * (dhat + jr_d)

    m_u = 0.5 * (uhat - jr_u)
    m_d = 0.5 * (dhat - jr_d)

    return p_u, p_d, m_u, m_d


def spinor_spec_norm2(uhat, dhat, mask=None):
    dens = cp.abs(uhat)**2 + cp.abs(dhat)**2
    if mask is not None:
        dens = cp.where(mask, dens, 0.0)
    return cp.sum(dens)



z_cpu = np.asarray(to_cpu(z))
kz_i0 = int(np.searchsorted(z_cpu, KZ_SLAB_ZMIN, side="left"))
kz_i1 = int(np.searchsorted(z_cpu, KZ_SLAB_ZMAX, side="right"))

kz_i0 = max(1, kz_i0)
kz_i1 = min(Nz - 5, kz_i1)

if kz_i1 <= kz_i0 + 8:
    raise ValueError("kz diagnostic slab too thin. Increase KZ_SLAB_ZMAX-KZ_SLAB_ZMIN.")

N_kz_win = kz_i1 - kz_i0
z_window = cp.hanning(N_kz_win).astype(cp.float64)

kz_grid = 2.0 * np.pi * cp.fft.fftfreq(N_kz_win, d=hz)
dkz = float(2.0 * np.pi / (N_kz_win * hz))
kz_zero_eps = KZ_ZERO_EPS_FACTOR * dkz

kz_pos_mask = kz_grid >  kz_zero_eps
kz_neg_mask = kz_grid < -kz_zero_eps




# Longitudinal FFT reflection diagnostic.
# Positive kz weight is treated as upgoing/incident spectral power in the slab;
# negative kz weight is treated as downgoing/reflected spectral power.
def kz_reflection_diagnostic(psi_up, psi_down):
    """
    Windowed longitudinal FFT diagnostic in a slab below the roof.

    P_plus:  kz > 0 propagating/upgoing spectral weight
    P_minus: kz < 0 propagating/downgoing spectral weight

    This is the closest diagnostic to scattering-style reflection.
    """
    u = psi_up[:, :, kz_i0:kz_i1].astype(cp.complex128, copy=False)
    d = psi_down[:, :, kz_i0:kz_i1].astype(cp.complex128, copy=False)

    w = z_window[None, None, :]
    u_fft = cp.fft.fft(u * w, axis=2, norm="ortho")
    d_fft = cp.fft.fft(d * w, axis=2, norm="ortho")

    dens = cp.abs(u_fft)**2 + cp.abs(d_fft)**2

    P_plus  = cp.sum(dens[:, :, kz_pos_mask]) * (hx * hy * hz)
    P_minus = cp.sum(dens[:, :, kz_neg_mask]) * (hx * hy * hz)

    P_plus_f  = float(P_plus.get())
    P_minus_f = float(P_minus.get())
    R_kz = P_minus_f / P_plus_f if P_plus_f > 0 else np.nan

    # mass in the actual slab, without window
    slab_mass = cp.sum(cp.abs(u)**2 + cp.abs(d)**2) * (hx * hy * hz)

    return {
        "P_plus": P_plus_f,
        "P_minus": P_minus_f,
        "R_kz": float(R_kz),
        "slab_mass": float(slab_mass.get()),
    }



# Tangential branch diagnostic.
# This probes the spinor-ABC matrix symbol by decomposing the roof trace into
# Pi_+ and Pi_- branches and comparing true slab rows with the homogeneous
# two-branch continuation.
def tangential_branch_and_duhamel_diagnostic(psi_up, psi_down):
    """
    Computes:
      - W_plus_roof, W_minus_roof
      - Q_minus_roof = W_minus/(W_plus+W_minus)
      - for each depth step s:
          Q_minus_delta[s]
          E_full[s]  = ||true - exp(-delta C) boundary|| / ||exp(-delta C) boundary||
          E_plus[s]  = ||true - exp(-i kappa delta) exp(-R delta) Pi_+ boundary||
                       / ||plus-only||
    """
    # physical boundary trace at z=L
    # physical boundary trace at z=L
    uL, dL = roof_trace_from_abc(psi_up, psi_down)
    
    edge_mass, total_mass = transverse_sidewall_mass_2d(uL, dL)
    edge_fraction = edge_mass / total_mass if total_mass > 0 else np.nan
    
    uhat = cp.fft.fft2(uL, norm="ortho")
    dhat = cp.fft.fft2(dL, norm="ortho")

    p_u, p_d, m_u, m_d = apply_projectors(uhat, dhat)

    boundary_power = cp.abs(uhat)**2 + cp.abs(dhat)**2
    max_power = cp.max(boundary_power)
    power_mask = boundary_power > (REL_POWER_CUTOFF * max_power)

    Wp = spinor_spec_norm2(p_u, p_d, mask=power_mask)
    Wm = spinor_spec_norm2(m_u, m_d, mask=power_mask)

    Wp_f = float(Wp.get())
    Wm_f = float(Wm.get())
    Qm_roof = Wm_f / (Wp_f + Wm_f) if (Wp_f + Wm_f) > 0 else np.nan

    out = {
        "W_plus_roof": Wp_f,
        "W_minus_roof": Wm_f,
        "Q_minus_roof": float(Qm_roof),
        "edge_mass_roof": float(edge_mass),
        "total_mass_roof": float(total_mass),
        "edge_fraction_roof": float(edge_fraction),
        "depths": {}
    }

    for s in DEPTH_STEPS:
        if s < 1 or s >= Nz:
            continue

        delta = float(s * hz)
        zidx = Nz - s

        # true field at z=L-delta
        u_true = psi_up[:, :, zidx].astype(cp.complex128, copy=False)
        d_true = psi_down[:, :, zidx].astype(cp.complex128, copy=False)
        u_true_hat = cp.fft.fft2(u_true, norm="ortho")
        d_true_hat = cp.fft.fft2(d_true, norm="ortho")

        Rdelta = R_TAN * delta
        valid_mask = power_mask & (Rdelta <= RDELTA_MAX)

        phase = cp.exp(-1j * k_bc * delta)

        exp_plus  = cp.exp(-Rdelta)
        exp_minus = cp.exp(+Rdelta)

        # full homogeneous ABC extrapolation:
        # e^{-delta C} = e^{-i kappa delta}( e^{-R delta} Pi_+ + e^{+R delta} Pi_- )
        u_hom = phase * (exp_plus * p_u + exp_minus * m_u)
        d_hom = phase * (exp_plus * p_d + exp_minus * m_d)

        # plus-only approximation:
        u_plus = phase * (exp_plus * p_u)
        d_plus = phase * (exp_plus * p_d)

        diff_full = spinor_spec_norm2(u_true_hat - u_hom, d_true_hat - d_hom, mask=valid_mask)
        norm_full = spinor_spec_norm2(u_hom, d_hom, mask=valid_mask)

        diff_plus = spinor_spec_norm2(u_true_hat - u_plus, d_true_hat - d_plus, mask=valid_mask)
        norm_plus = spinor_spec_norm2(u_plus, d_plus, mask=valid_mask)

        Wp_delta = spinor_spec_norm2(exp_plus * p_u, exp_plus * p_d, mask=valid_mask)
        Wm_delta = spinor_spec_norm2(exp_minus * m_u, exp_minus * m_d, mask=valid_mask)

        Wp_delta_f = float(Wp_delta.get())
        Wm_delta_f = float(Wm_delta.get())

        E_full = float(cp.sqrt(diff_full / norm_full).get()) if float(norm_full.get()) > 0 else np.nan
        E_plus = float(cp.sqrt(diff_plus / norm_plus).get()) if float(norm_plus.get()) > 0 else np.nan
        Qm_delta = Wm_delta_f / (Wp_delta_f + Wm_delta_f) if (Wp_delta_f + Wm_delta_f) > 0 else np.nan

        out["depths"][int(s)] = {
            "delta": float(delta),
            "z_index": int(zidx),
            "z_pos": float((Lz - delta)),
            "valid_modes": int(cp.count_nonzero(valid_mask).get()),
            "W_plus_delta": Wp_delta_f,
            "W_minus_delta": Wm_delta_f,
            "Q_minus_delta": float(Qm_delta),
            "E_full": float(E_full),
            "E_plus": float(E_plus),
        }

    return out




near_slab_i0 = max(1, Nz - int(round(NEAR_SLAB_DEPTH / hz)))
near_slab_i1 = Nz

def near_roof_slab_mass(psi_up, psi_down):
    u = psi_up[:, :, near_slab_i0:near_slab_i1]
    d = psi_down[:, :, near_slab_i0:near_slab_i1]
    return float((cp.sum(cp.abs(u)**2 + cp.abs(d)**2) * (hx * hy * hz)).get())



kz_P_plus_series = []
kz_P_minus_series = []
kz_R_series = []
kz_slab_mass_series = []

near_roof_mass_series = []

Q_minus_roof_series = []
W_plus_roof_series = []
W_minus_roof_series = []
edge_mass_roof_series = []
total_mass_roof_series = []
edge_fraction_roof_series = []


E_full_by_depth = {int(s): [] for s in DEPTH_STEPS}
E_plus_by_depth = {int(s): [] for s in DEPTH_STEPS}
Q_minus_delta_by_depth = {int(s): [] for s in DEPTH_STEPS}
W_plus_delta_by_depth = {int(s): [] for s in DEPTH_STEPS}
W_minus_delta_by_depth = {int(s): [] for s in DEPTH_STEPS}
valid_modes_by_depth = {int(s): [] for s in DEPTH_STEPS}






# =============================================================================
# Time-series arrays for reflection diagnostics
# =============================================================================
# These arrays are saved as .npy files and summarized in summary.json.
# -------------------- storage --------------------
# original probability logging arrays (kept)
prob_t   = []
prob_val = []

# -------------------- roof diagnostics (NEW) --------------------
ROOF_DIAG = True

z_roof_idx = Nz - 1          # true boundary plane (Robin/ABC row in L_z)
z_roof_in  = Nz - 2          # one grid inside (recommended sanity check)


diag_times = []
P_total_series = []

roof_Jnet = []
roof_Jplus = []
roof_Jminus = []
roof_rho_int = []

roof_in_Jnet = []
roof_in_Jplus = []
roof_in_Jminus = []
roof_in_rho_int = []



#---------------------------

det_J_series = []
det_rho_int_series = []



def detector_flux_from_abc_trace(psi_up, psi_down):
    uL, dL = roof_trace_from_abc(psi_up, psi_down)
    rhoL = cp.abs(uL)**2 + cp.abs(dL)**2
    rho_int = cp.sum(rhoL) * (hx * hy)

    # Spinor RBC with real RHS is reflecting, not absorbing.
    Jdet = 0.0

    return float(Jdet), float(rho_int.get())





# =============================================================================
# Main CN/GMRES evolution loop with diagnostic sampling
# =============================================================================
# The wavefunction evolution is unchanged.  Diagnostics are evaluated only at
# DIAG_EVERY intervals to keep runtime manageable.
# -------------------- time loop (UNCHANGED evolution) --------------------
num_steps = int(round(T_final / dt))
print(f"[info] Starting time loop – {num_steps} steps")
start_time = time.time()

psi0 = psi_flat.reshape(2, Nx, Ny, Nz)
P0_init = float(to_cpu(cp.sum(cp.abs(psi0[0])**2 + cp.abs(psi0[1])**2) * (hx * hy * hz)))

del psi0
for n in range(1, num_steps + 1):

    if n % 200 == 0:
        elapsed = time.time() - start_time
        eta = (num_steps - n) * (elapsed / n) / 60.0
        print(f"[progress] Step {n}/{num_steps} (t={n*dt:.3f}) | ETA: {eta:.2f} min", flush=True)

    # CN advance (UNCHANGED)
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

    if info != 0:
        logging.warning(f"[GMRES] not converged at step {n}, info={info}")
    psi_flat = psi64.astype(cp.complex64)

    psi_3d = psi_flat.reshape(2, Nx, Ny, Nz)
    psi_up = psi_3d[0]
    psi_down = psi_3d[1]

    # enforce Dirichlet BCs (UNCHANGED)
    psi_up[[0, -1], :, :] = 0
    psi_up[:, [0, -1], :] = 0
    psi_up[:, :, 0] = 0
    psi_down[[0, -1], :, :] = 0
    psi_down[:, [0, -1], :] = 0
    psi_down[:, :, 0] = 0

    t = n * dt

    # probability monitor (original logging cadence)
    if (n % PROB_EVERY) == 0:
        total_prob = float(to_cpu(cp.sum(cp.abs(psi_up)**2 + cp.abs(psi_down)**2) * (hx * hy * hz)))
        prob_t.append(t)
        prob_val.append(total_prob)
        logging.info(f"[Probability] t={t:.3f}, ∫|ψ|² dV = {total_prob:.8f}")

    # monitor-grid diagnostics
    # spectral / roof diagnostics
    if (n % DIAG_EVERY) == 0:
        diag_times.append(t)
    
        Ptot = float(to_cpu(cp.sum(cp.abs(psi_up)**2 + cp.abs(psi_down)**2) * (hx * hy * hz)))
        P_total_series.append(Ptot)
    
        # roof flux diagnostic
        Jt, Jc, Js, JplT, JmlT, JplC, JmlC = plane_pauli_fluxes(psi_up, psi_down, z_roof_idx)
        roof_Jnet.append(Jt)
        roof_Jplus.append(max(Jt, 0.0))
        roof_Jminus.append(max(-Jt, 0.0))
    
        rho2d = cp.abs(psi_up[:, :, z_roof_idx])**2 + cp.abs(psi_down[:, :, z_roof_idx])**2
        roof_rho_int.append(float(cp.sum(rho2d).get()) * (hx * hy))
    
        # roof_in flux diagnostic
        Jt2, Jc2, Js2, JplT2, JmlT2, JplC2, JmlC2 = plane_pauli_fluxes(psi_up, psi_down, z_roof_in)
        roof_in_Jnet.append(Jt2)
        roof_in_Jplus.append(max(Jt2, 0.0))
        roof_in_Jminus.append(max(-Jt2, 0.0))
    
        rho2d_in = cp.abs(psi_up[:, :, z_roof_in])**2 + cp.abs(psi_down[:, :, z_roof_in])**2
        roof_in_rho_int.append(float(cp.sum(rho2d_in).get()) * (hx * hy))
    
        # longitudinal kz scattering diagnostic
        kz_diag = kz_reflection_diagnostic(psi_up, psi_down)
        kz_P_plus_series.append(kz_diag["P_plus"])
        kz_P_minus_series.append(kz_diag["P_minus"])
        kz_R_series.append(kz_diag["R_kz"])
        kz_slab_mass_series.append(kz_diag["slab_mass"])
        
        Jdet, rhoL_int = detector_flux_from_abc_trace(psi_up, psi_down)
        det_J_series.append(Jdet)
        det_rho_int_series.append(rhoL_int)
            
        # near-roof storage mass
        near_roof_mass_series.append(near_roof_slab_mass(psi_up, psi_down))
    
        # tangential Pi / Duhamel diagnostics
        td = tangential_branch_and_duhamel_diagnostic(psi_up, psi_down)
    
        W_plus_roof_series.append(td["W_plus_roof"])
        W_minus_roof_series.append(td["W_minus_roof"])
        Q_minus_roof_series.append(td["Q_minus_roof"])
        edge_mass_roof_series.append(td["edge_mass_roof"])
        total_mass_roof_series.append(td["total_mass_roof"])
        edge_fraction_roof_series.append(td["edge_fraction_roof"])
            
        for s in DEPTH_STEPS:
            s = int(s)
            if s in td["depths"]:
                dd = td["depths"][s]
                E_full_by_depth[s].append(dd["E_full"])
                E_plus_by_depth[s].append(dd["E_plus"])
                Q_minus_delta_by_depth[s].append(dd["Q_minus_delta"])
                W_plus_delta_by_depth[s].append(dd["W_plus_delta"])
                W_minus_delta_by_depth[s].append(dd["W_minus_delta"])
                valid_modes_by_depth[s].append(dd["valid_modes"])
            else:
                E_full_by_depth[s].append(np.nan)
                E_plus_by_depth[s].append(np.nan)
                Q_minus_delta_by_depth[s].append(np.nan)
                W_plus_delta_by_depth[s].append(np.nan)
                W_minus_delta_by_depth[s].append(np.nan)
                valid_modes_by_depth[s].append(0)



# =============================================================================
# Save diagnostics and build finite-window summary
# =============================================================================
# The summary distinguishes cumulative roof backflow, kz scattering-style ratios,
# and Pi_-/Duhamel near-roof diagnostics.
# =============================================================================
# Save spectral diagnostic outputs
# =============================================================================

diag_times = np.asarray(diag_times, dtype=np.float64)
P_total_series = np.asarray(P_total_series, dtype=np.float64)

roof_Jnet = np.asarray(roof_Jnet, dtype=np.float64)
roof_Jplus = np.asarray(roof_Jplus, dtype=np.float64)
roof_Jminus = np.asarray(roof_Jminus, dtype=np.float64)
roof_rho_int = np.asarray(roof_rho_int, dtype=np.float64)

roof_in_Jnet = np.asarray(roof_in_Jnet, dtype=np.float64)
roof_in_Jplus = np.asarray(roof_in_Jplus, dtype=np.float64)
roof_in_Jminus = np.asarray(roof_in_Jminus, dtype=np.float64)
roof_in_rho_int = np.asarray(roof_in_rho_int, dtype=np.float64)

kz_P_plus_series = np.asarray(kz_P_plus_series, dtype=np.float64)
kz_P_minus_series = np.asarray(kz_P_minus_series, dtype=np.float64)
kz_R_series = np.asarray(kz_R_series, dtype=np.float64)
kz_slab_mass_series = np.asarray(kz_slab_mass_series, dtype=np.float64)

near_roof_mass_series = np.asarray(near_roof_mass_series, dtype=np.float64)

W_plus_roof_series = np.asarray(W_plus_roof_series, dtype=np.float64)
W_minus_roof_series = np.asarray(W_minus_roof_series, dtype=np.float64)
Q_minus_roof_series = np.asarray(Q_minus_roof_series, dtype=np.float64)

edge_mass_roof_series = np.asarray(edge_mass_roof_series, dtype=np.float64)
total_mass_roof_series = np.asarray(total_mass_roof_series, dtype=np.float64)
edge_fraction_roof_series = np.asarray(edge_fraction_roof_series, dtype=np.float64)


det_J_series = np.asarray(det_J_series, dtype=np.float64)
det_rho_int_series = np.asarray(det_rho_int_series, dtype=np.float64)

np.save(out_dir / "det_J_abc_trace.npy", det_J_series)
np.save(out_dir / "det_rho_int_abc_trace.npy", det_rho_int_series)


np.save(out_dir / "diag_times.npy", diag_times)
np.save(out_dir / "P_total_series.npy", P_total_series)

np.save(out_dir / "roof_Jnet.npy", roof_Jnet)
np.save(out_dir / "roof_Jplus.npy", roof_Jplus)
np.save(out_dir / "roof_Jminus.npy", roof_Jminus)
np.save(out_dir / "roof_rho_int.npy", roof_rho_int)

np.save(out_dir / "roof_in_Jnet.npy", roof_in_Jnet)
np.save(out_dir / "roof_in_Jplus.npy", roof_in_Jplus)
np.save(out_dir / "roof_in_Jminus.npy", roof_in_Jminus)
np.save(out_dir / "roof_in_rho_int.npy", roof_in_rho_int)

np.save(out_dir / "kz_P_plus.npy", kz_P_plus_series)
np.save(out_dir / "kz_P_minus.npy", kz_P_minus_series)
np.save(out_dir / "kz_R.npy", kz_R_series)
np.save(out_dir / "kz_slab_mass.npy", kz_slab_mass_series)

np.save(out_dir / "near_roof_mass.npy", near_roof_mass_series)

np.save(out_dir / "W_plus_roof.npy", W_plus_roof_series)
np.save(out_dir / "W_minus_roof.npy", W_minus_roof_series)
np.save(out_dir / "Q_minus_roof.npy", Q_minus_roof_series)
np.save(out_dir / "edge_mass_roof.npy", edge_mass_roof_series)
np.save(out_dir / "total_mass_roof.npy", total_mass_roof_series)
np.save(out_dir / "edge_fraction_roof.npy", edge_fraction_roof_series)



for s in DEPTH_STEPS:
    s = int(s)
    np.save(out_dir / f"E_full_depthSteps_{s}.npy", np.asarray(E_full_by_depth[s], dtype=np.float64))
    np.save(out_dir / f"E_plus_depthSteps_{s}.npy", np.asarray(E_plus_by_depth[s], dtype=np.float64))
    np.save(out_dir / f"Q_minus_delta_depthSteps_{s}.npy", np.asarray(Q_minus_delta_by_depth[s], dtype=np.float64))
    np.save(out_dir / f"W_plus_delta_depthSteps_{s}.npy", np.asarray(W_plus_delta_by_depth[s], dtype=np.float64))
    np.save(out_dir / f"W_minus_delta_depthSteps_{s}.npy", np.asarray(W_minus_delta_by_depth[s], dtype=np.float64))
    np.save(out_dir / f"valid_modes_depthSteps_{s}.npy", np.asarray(valid_modes_by_depth[s], dtype=np.int64))
    
    
    
def trapz_safe(y, x):
    if len(x) < 2:
        return np.nan
    return float(np.trapezoid(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)))

P_roof_plus_int = trapz_safe(roof_Jplus, diag_times)
P_roof_minus_int = trapz_safe(roof_Jminus, diag_times)
R_roof_cum = P_roof_minus_int / P_roof_plus_int if P_roof_plus_int > 0 else np.nan

P_det_int = trapz_safe(det_J_series, diag_times)
norm_loss_diag = float(P0_init - P_total_series[-1]) if P_total_series.size else np.nan
det_budget_error = P_det_int - norm_loss_diag if np.isfinite(norm_loss_diag) else np.nan
det_budget_rel_error = (
    det_budget_error / norm_loss_diag
    if np.isfinite(norm_loss_diag) and abs(norm_loss_diag) > 0
    else np.nan
)


P_kz_plus_int = trapz_safe(kz_P_plus_series, diag_times)
P_kz_minus_int = trapz_safe(kz_P_minus_series, diag_times)
R_kz_timeint = P_kz_minus_int / P_kz_plus_int if P_kz_plus_int > 0 else np.nan

KZ_MASS_THRESHOLD = 1e-8
valid_kz_mass = kz_slab_mass_series > KZ_MASS_THRESHOLD

max_R_kz_valid = (
    float(np.nanmax(kz_R_series[valid_kz_mass]))
    if np.any(valid_kz_mass)
    else np.nan
)


# First-pass / later-return bookkeeping.
# This is not a fit model.  It is a conservative timing cut used to avoid mixing
# the first reflected signal with later bottom-return contamination.
# Approximate first-pass window based only on longitudinal group velocity.
# This is not a backflow diagnostic; it is only a conservative time window
# to avoid bottom-return contamination in the spectral kz summary.
v_est = abs(k0) if abs(k0) > 0 else abs(k_bc)

t_roof_est = (Lz - z0_init) / v_est
t_down_to_slab_min_est = t_roof_est + (Lz - KZ_SLAB_ZMIN) / v_est
t_bottom_return_to_slab_est = t_roof_est + Lz / v_est + KZ_SLAB_ZMIN / v_est

# Use midpoint between expected first downward exit and expected bottom-return re-entry.
FIRSTPASS_TMAX = 0.5 * (t_down_to_slab_min_est + t_bottom_return_to_slab_est)

firstpass_mask = diag_times <= FIRSTPASS_TMAX
valid_kz_firstpass = valid_kz_mass & firstpass_mask

P_kz_plus_first = trapz_safe(
    np.where(valid_kz_firstpass, kz_P_plus_series, 0.0),
    diag_times
)
P_kz_minus_first = trapz_safe(
    np.where(valid_kz_firstpass, kz_P_minus_series, 0.0),
    diag_times
)

R_kz_firstpass = (
    P_kz_minus_first / P_kz_plus_first
    if P_kz_plus_first > 0
    else np.nan
)

max_R_kz_firstpass = (
    float(np.nanmax(kz_R_series[valid_kz_firstpass]))
    if np.any(valid_kz_firstpass)
    else np.nan
)


summary = {
    "run_dir": str(out_dir.resolve()),
    "omega": float(omega),
    "dt": float(dt),
    "T_final": float(T_final),
    "NxNyNz": [int(Nx), int(Ny), int(Nz)],
    "LxLyLz": [float(Lx), float(Ly), float(Lz)],
    "hxhyhz": [float(hx), float(hy), float(hz)],
    "k_bc": float(k_bc),
    "k0": float(k0),
    "z0_init": float(z0_init),
    "sigma_z": float(sigma_z),
    "theta": float(theta),
    
    "diagnostic": {
        "DIAG_EVERY": int(DIAG_EVERY),
        "dt_diag": float(DIAG_EVERY * dt),
        "KZ_SLAB_ZMIN": float(KZ_SLAB_ZMIN),
        "KZ_SLAB_ZMAX": float(KZ_SLAB_ZMAX),
        "kz_slab_zmin_actual": float(z_cpu[kz_i0]),
        "kz_slab_zmax_actual": float(z_cpu[kz_i1-1]),
        "N_kz_win": int(N_kz_win),
        "kz_i0": int(kz_i0),
        "kz_i1": int(kz_i1),
        "kz_zero_eps": float(kz_zero_eps),
        "NEAR_SLAB_DEPTH": float(NEAR_SLAB_DEPTH),
        "DEPTH_STEPS": [int(s) for s in DEPTH_STEPS],
        "RDELTA_MAX": float(RDELTA_MAX),
        "REL_POWER_CUTOFF": float(REL_POWER_CUTOFF),
        "USE_FD_SYMBOL": bool(USE_FD_SYMBOL),
        "EDGE_BAND": int(EDGE_BAND),
    },
    "roof_grid_backflow": {
        "int_Jplus": float(P_roof_plus_int),
        "int_Jminus": float(P_roof_minus_int),
        "R_roof_cum": float(R_roof_cum),
    },
    
    "detector_flux_abc_trace": {
        "int_Jdet": float(P_det_int),
        "norm_loss_diag": float(norm_loss_diag),
        "det_budget_error": float(det_budget_error),
        "det_budget_rel_error": float(det_budget_rel_error),
        "P0_init": float(P0_init),
        "P_final_diag": float(P_total_series[-1]) if P_total_series.size else np.nan,
    },
    
    "kz_scattering": {
        "int_P_plus": float(P_kz_plus_int),
        "int_P_minus": float(P_kz_minus_int),
        "R_kz_timeint": float(R_kz_timeint),
        
        "max_R_kz_raw": float(np.nanmax(kz_R_series)) if kz_R_series.size else np.nan,
        "max_R_kz_mass_filtered": float(max_R_kz_valid),
        "KZ_MASS_THRESHOLD": float(KZ_MASS_THRESHOLD),
        "v_est": float(v_est),
        "t_roof_est": float(t_roof_est),
        "t_down_to_slab_min_est": float(t_down_to_slab_min_est),
        "t_bottom_return_to_slab_est": float(t_bottom_return_to_slab_est),
        "FIRSTPASS_TMAX": float(FIRSTPASS_TMAX),
        "int_P_plus_firstpass": float(P_kz_plus_first),
        "int_P_minus_firstpass": float(P_kz_minus_first),
        "R_kz_firstpass": float(R_kz_firstpass),
        "max_R_kz_firstpass": float(max_R_kz_firstpass),
    },
    "branch_roof": {
        "max_Q_minus_roof": float(np.nanmax(Q_minus_roof_series)) if Q_minus_roof_series.size else np.nan,
        "median_Q_minus_roof": float(np.nanmedian(Q_minus_roof_series)) if Q_minus_roof_series.size else np.nan,
        "max_edge_fraction_roof": float(np.nanmax(edge_fraction_roof_series)) if edge_fraction_roof_series.size else np.nan,
        "median_edge_fraction_roof": float(np.nanmedian(edge_fraction_roof_series)) if edge_fraction_roof_series.size else np.nan,
    }
}


np.save(out_dir / "prob_t.npy", np.asarray(prob_t, dtype=np.float64))
np.save(out_dir / "prob_val.npy", np.asarray(prob_val, dtype=np.float64))

summary["depth_summaries"] = {}
for s in DEPTH_STEPS:
    s = int(s)
    Ef = np.asarray(E_full_by_depth[s], dtype=np.float64)
    Ep = np.asarray(E_plus_by_depth[s], dtype=np.float64)
    Qd = np.asarray(Q_minus_delta_by_depth[s], dtype=np.float64)
    summary["depth_summaries"][str(s)] = {
        "delta": float(s * hz),
        "max_E_full": float(np.nanmax(Ef)) if Ef.size else np.nan,
        "median_E_full": float(np.nanmedian(Ef)) if Ef.size else np.nan,
        "max_E_plus": float(np.nanmax(Ep)) if Ep.size else np.nan,
        "median_E_plus": float(np.nanmedian(Ep)) if Ep.size else np.nan,
        "max_Q_minus_delta": float(np.nanmax(Qd)) if Qd.size else np.nan,
        "median_Q_minus_delta": float(np.nanmedian(Qd)) if Qd.size else np.nan,
    }

with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

elapsed = time.time() - start_time
print(f"Total execution time: {elapsed:.2f} s", flush=True)
print(f"[info] Outputs saved in: {out_dir.resolve()}", flush=True)

print("\n[roof backflow]")
print(f"  R_roof_cum = {R_roof_cum:.6e}")

print("\n[kz scattering reflection]")
print(f"  R_kz_timeint = {R_kz_timeint:.6e}")
print(f"  R_kz_firstpass = {R_kz_firstpass:.6e}")
print(f"  FIRSTPASS_TMAX = {FIRSTPASS_TMAX:.6f}")
print(f"  max R_kz raw           = {np.nanmax(kz_R_series):.6e}")
print(f"  max R_kz mass-filtered = {max_R_kz_valid:.6e}")
print("\n[Pi-/Duhamel diagnostics]")
print(f"  max Q_minus_roof = {np.nanmax(Q_minus_roof_series):.6e}")
for s in DEPTH_STEPS:
    s = int(s)
    print(
        f"  depth_steps={s}, delta={s*hz:.6g}: "
        f"median E_full={np.nanmedian(E_full_by_depth[s]):.6e}, "
        f"median E_plus={np.nanmedian(E_plus_by_depth[s]):.6e}, "
        f"median Q_minus_delta={np.nanmedian(Q_minus_delta_by_depth[s]):.6e}"
    )


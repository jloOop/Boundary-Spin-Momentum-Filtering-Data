"""Grid constants loading helpers for Bohmian trajectory plotting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrajectoryGrid:
    """Grid constants and derived detector position."""

    Nx: int
    Ny: int
    Nz: int
    hx: float
    hy: float
    hz: float
    Lx: float
    Ly: float
    Lz: float
    z_det: float


def load_trajectory_grid(base):
    """Load grid constants from ``constants.npz``."""
    constants_path = Path(base) / "constants.npz"

    with np.load(constants_path) as constants:
        Nx, Ny, Nz = int(constants["Nx"]), int(constants["Ny"]), int(constants["Nz"])
        hx, hy, hz = float(constants["hx"]), float(constants["hy"]), float(constants["hz"])
        Lx, Ly, Lz = float(constants["Lx"]), float(constants["Ly"]), float(constants["Lz"])

    return TrajectoryGrid(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        hx=hx,
        hy=hy,
        hz=hz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        z_det=Lz - hz,
    )

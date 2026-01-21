"""Small-o helpers for tensor network states."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from plum import dispatch

from vmc.models.mps import MPS
from vmc.models.peps import PEPS

__all__ = [
    "mps_site_dims",
    "peps_site_dims",
    "params_per_site",
]


def mps_site_dims(site: int, n_sites: int, bond_dim: int) -> tuple[int, int]:
    """Return (left_dim, right_dim) for an MPS site."""
    left = 1 if site == 0 else bond_dim
    right = 1 if site == n_sites - 1 else bond_dim
    return left, right


def peps_site_dims(
    row: int, col: int, n_rows: int, n_cols: int, bond_dim: int
) -> tuple[int, int, int, int]:
    """Return (up, down, left, right) dimensions for a PEPS site."""
    up = 1 if row == 0 else bond_dim
    down = 1 if row == n_rows - 1 else bond_dim
    left = 1 if col == 0 else bond_dim
    right = 1 if col == n_cols - 1 else bond_dim
    return up, down, left, right


@dispatch
def params_per_site(model: MPS) -> list[int]:
    """Compute number of parameters per physical slice for each MPS site."""
    n_sites, bond_dim = model.n_sites, model.bond_dim
    return [
        left * right
        for site in range(n_sites)
        for left, right in [mps_site_dims(site, n_sites, bond_dim)]
    ]


@dispatch
def params_per_site(model: PEPS) -> list[int]:
    """Compute number of parameters per physical slice for each PEPS site."""
    n_rows, n_cols = model.shape
    bond_dim = model.bond_dim
    return [
        up * down * left * right
        for r in range(n_rows)
        for c in range(n_cols)
        for up, down, left, right in [peps_site_dims(r, c, n_rows, n_cols, bond_dim)]
    ]

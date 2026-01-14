"""Small-o helpers for tensor network states."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

from plum import dispatch

from VMC.models.mps import MPS
from VMC.models.peps import PEPS

__all__ = [
    "params_per_site",
]


@dispatch
def params_per_site(model: MPS) -> list[int]:
    """Compute number of parameters per physical slice for each MPS site."""
    n_sites = model.n_sites
    bond_dim = model.bond_dim
    result = []
    for site in range(n_sites):
        left_dim = 1 if site == 0 else bond_dim
        right_dim = 1 if site == n_sites - 1 else bond_dim
        result.append(left_dim * right_dim)
    return result


@dispatch
def params_per_site(model: PEPS) -> list[int]:
    """Compute number of parameters per physical slice for each PEPS site."""
    n_rows, n_cols = model.shape
    bond_dim = model.bond_dim
    result = []
    for r in range(n_rows):
        for c in range(n_cols):
            up = 1 if r == 0 else bond_dim
            down = 1 if r == n_rows - 1 else bond_dim
            left = 1 if c == 0 else bond_dim
            right = 1 if c == n_cols - 1 else bond_dim
            result.append(up * down * left * right)
    return result

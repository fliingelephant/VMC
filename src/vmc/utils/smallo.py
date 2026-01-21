"""Small-o helpers for tensor network states."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from plum import dispatch

from vmc.models.mps import MPS
from vmc.models.peps import PEPS

__all__ = ["params_per_site"]


@dispatch
def params_per_site(model: MPS) -> list[int]:
    """Compute number of parameters per physical slice for each MPS site."""
    n_sites, bond_dim = model.n_sites, model.bond_dim
    return [
        left * right
        for site in range(n_sites)
        for left, right in [MPS.site_dims(site, n_sites, bond_dim)]
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
        for up, down, left, right in [
            PEPS.site_dims(r, c, n_rows, n_cols, bond_dim)
        ]
    ]

"""Small-o helpers for tensor network states.

Key concepts:
- params_per_site: Number of variational parameters in each "active slice" at a site.
- sliced_dims: Number of distinct active slices per site, determined by the sample.

For a given sample configuration, only one slice of parameters at each site
contributes to the wavefunction amplitude. The slice index is determined by:
- PEPS: physical state σ ∈ {0, ..., d-1}
- GIPEPS: combined index (σ, cfg) where cfg encodes local gauge configuration
"""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from plum import dispatch

from vmc.peps.standard.model import PEPS

__all__ = ["params_per_site", "sliced_dims"]


@dispatch
def params_per_site(model: PEPS) -> list[int]:
    """Number of parameters per active slice at each PEPS site."""
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


@dispatch
def sliced_dims(model: PEPS) -> tuple[int, ...]:
    """Number of distinct active slices per site (= phys_dim for PEPS)."""
    n_rows, n_cols = model.shape
    return (model.phys_dim,) * (n_rows * n_cols)

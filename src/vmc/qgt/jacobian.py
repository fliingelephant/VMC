"""Jacobian representations for QGT construction."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc import config  # noqa: F401

__all__ = [
    "PhysicalOrdering",
    "SiteOrdering",
    "Jacobian",
    "SlicedJacobian",
    "jacobian_mean",
]


@dataclass(frozen=True)
class PhysicalOrdering:
    """Loop over physical indices first."""

    pass


@dataclass(frozen=True)
class SiteOrdering:
    """Loop over sites first."""

    params_per_site: tuple[int, ...]


@dataclass
class Jacobian:
    """Full Jacobian O (n_samples, n_params)."""

    O: jax.Array


@dataclass
class SlicedJacobian:
    """Reduced Jacobian using the small-o trick for memory efficiency.

    For each sample, only one "active slice" of parameters contributes to the
    amplitude at each site. The slice index p[sample, site] is determined by:
    - MPS/PEPS: physical state σ ∈ {0, ..., d-1}
    - GIPEPS: combined index σ * nc + cfg_idx (gauge config encoded)

    sliced_dims[site] specifies how many distinct slices exist at each site.
    """

    o: jax.Array  # shape: (n_samples, sum(params_per_site))
    p: jax.Array  # shape: (n_samples, n_sites), active slice index per site
    sliced_dims: tuple[int, ...]  # number of slices per site
    ordering: PhysicalOrdering | SiteOrdering = PhysicalOrdering()

    @property
    def phys_dim(self) -> int:
        """For backward compatibility, return max sliced dim."""
        return max(self.sliced_dims) if self.sliced_dims else 2

    @classmethod
    def from_samples(
        cls,
        model,
        samples: jax.Array,
        ordering: PhysicalOrdering | SiteOrdering = PhysicalOrdering(),
    ):
        """Construct from model and samples."""
        from vmc.core import _value_and_grad
        from vmc.utils.vmc_utils import flatten_samples

        samples = flatten_samples(samples)
        amps, grads, p = _value_and_grad(model, samples, full_gradient=False)
        o = grads / amps[:, None]
        # Default: uniform phys_dim for all sites
        n_sites = getattr(model, "n_sites", grads.shape[1])
        sliced_dims = (model.phys_dim,) * n_sites
        return cls(o, p, sliced_dims, ordering)


@dispatch
def jacobian_mean(jac: Jacobian) -> jax.Array:
    return jnp.mean(jac.O, axis=0)


@dispatch
def jacobian_mean(jac: SlicedJacobian) -> jax.Array:
    return _sliced_mean(jac.ordering, jac.o, jac.p, jac.sliced_dims)


# TODO: possible break: this produces a vector of length n_sites * sliced_dim *
# params_per_site, but the SR centering formula from Wu 2025 Eq. 5 expects the
# mean to be applied before reconstructing the full parameter space. Verify that
# mean subtraction aligns parameters correctly when using SiteOrdering.
@dispatch
def _sliced_mean(
    ordering: SiteOrdering,
    o: jax.Array,
    p: jax.Array,
    sliced_dims: tuple[int, ...],
) -> jax.Array:
    blocks = []
    i = 0
    for site_idx, n in enumerate(ordering.params_per_site):
        sliced_dim = sliced_dims[site_idx]
        for k in range(sliced_dim):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            blocks.append(jnp.mean(ok, axis=0))
        i += n
    return jnp.concatenate(blocks, axis=0)


@dispatch
def _sliced_mean(
    ordering: PhysicalOrdering,
    o: jax.Array,
    p: jax.Array,
    sliced_dims: tuple[int, ...],
) -> jax.Array:
    max_sliced_dim = max(sliced_dims)
    blocks = []
    for k in range(max_sliced_dim):
        ok = jnp.where(p == k, o, 0)
        blocks.append(jnp.mean(ok, axis=0))
    return jnp.concatenate(blocks, axis=0)

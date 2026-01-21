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
    """Reduced Jacobian using small-o trick."""

    o: jax.Array
    p: jax.Array
    phys_dim: int
    ordering: PhysicalOrdering | SiteOrdering = PhysicalOrdering()

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
        return cls(o, p, model.phys_dim, ordering)


@dispatch
def jacobian_mean(jac: Jacobian) -> jax.Array:
    return jnp.mean(jac.O, axis=0)


@dispatch
def jacobian_mean(jac: SlicedJacobian) -> jax.Array:
    return _sliced_mean(jac.ordering, jac.o, jac.p, jac.phys_dim)


# TODO: possible break: this produces a vector of length n_sites * phys_dim *
# params_per_site, but the SR centering formula from Wu 2025 Eq. 5 expects the
# mean to be applied before reconstructing the full parameter space. Verify that
# mean subtraction aligns parameters correctly when using SiteOrdering.
@dispatch
def _sliced_mean(
    ordering: SiteOrdering,
    o: jax.Array,
    p: jax.Array,
    phys_dim: int,
) -> jax.Array:
    blocks = []
    i = 0
    for n in ordering.params_per_site:
        for k in range(phys_dim):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            blocks.append(jnp.mean(ok, axis=0))
        i += n
    return jnp.concatenate(blocks, axis=0)


@dispatch
def _sliced_mean(
    ordering: PhysicalOrdering,
    o: jax.Array,
    p: jax.Array,
    phys_dim: int,
) -> jax.Array:
    blocks = []
    for k in range(phys_dim):
        ok = jnp.where(p == k, o, 0)
        blocks.append(jnp.mean(ok, axis=0))
    return jnp.concatenate(blocks, axis=0)

"""Jacobian representations for QGT construction."""
from __future__ import annotations

from dataclasses import dataclass

import jax

from VMC import config  # noqa: F401

__all__ = ["PhysicalOrdering", "SiteOrdering", "Jacobian", "SlicedJacobian"]


@dataclass(frozen=True)
class PhysicalOrdering:
    """Loop over physical indices first."""
    pass


@dataclass(frozen=True)
class SiteOrdering:
    """Loop over sites first."""
    params_per_site: tuple[int, ...]


class Jacobian:
    """Full Jacobian O (n_samples, n_params)."""
    def __init__(self, O: jax.Array):
        self.O = O


class SlicedJacobian:
    """Reduced Jacobian using small-o trick."""
    def __init__(
        self,
        o: jax.Array,
        p: jax.Array,
        phys_dim: int,
        ordering: PhysicalOrdering | SiteOrdering = PhysicalOrdering(),
    ):
        self.o = o
        self.p = p
        self.phys_dim = phys_dim
        self.ordering = ordering

    @classmethod
    def from_samples(cls, model, samples: jax.Array, ordering=None):
        """Construct from model and samples."""
        from VMC.core import _value_and_grad_batch
        from VMC.utils.smallo import params_per_site
        from VMC.utils.vmc_utils import flatten_samples

        samples = flatten_samples(samples)
        amps, grads, p = _value_and_grad_batch(model, samples, full_gradient=False)
        o = grads / amps[:, None]
        if ordering is None:
            ordering = PhysicalOrdering()
        return cls(o, p, model.phys_dim, ordering)

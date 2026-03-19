"""Time-dependent local Hamiltonian wrappers and schedules."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from plum import dispatch

__all__ = [
    "TermCoefficientSchedule",
    "AffineSchedule",
    "CubicSchedule",
    "TimeDependentHamiltonian",
    "coeffs_at",
]


class TermCoefficientSchedule(abc.ABC):
    """Base type for schedules producing per-term coefficients."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AffineSchedule(TermCoefficientSchedule):
    """Affine coefficient schedule: ``offset + t * slope``."""

    offset: jax.Array
    slope: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", jnp.atleast_1d(self.offset))
        object.__setattr__(self, "slope", jnp.atleast_1d(self.slope))

    def tree_flatten(self):
        return (self.offset, self.slope), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        offset, slope = children
        return cls(offset=offset, slope=slope)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CubicSchedule(TermCoefficientSchedule):
    """Cubic coefficient schedule."""

    constant: jax.Array
    linear: jax.Array
    quadratic: jax.Array
    cubic: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "constant", jnp.atleast_1d(self.constant))
        object.__setattr__(self, "linear", jnp.atleast_1d(self.linear))
        object.__setattr__(self, "quadratic", jnp.atleast_1d(self.quadratic))
        object.__setattr__(self, "cubic", jnp.atleast_1d(self.cubic))

    def tree_flatten(self):
        return (self.constant, self.linear, self.quadratic, self.cubic), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        constant, linear, quadratic, cubic = children
        return cls(
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            cubic=cubic,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TimeDependentHamiltonian:
    """Wrapper combining a static Hamiltonian topology with coefficient schedule."""

    base: Any
    schedule: TermCoefficientSchedule

    def tree_flatten(self):
        return (self.base, self.schedule), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        base, schedule = children
        return cls(base=base, schedule=schedule)


@dispatch
def coeffs_at(schedule: TermCoefficientSchedule, t: float | jax.Array) -> jax.Array:
    """Return per-term coefficients at time ``t``."""
    raise NotImplementedError(f"Unsupported schedule type: {type(schedule)!r}")


@coeffs_at.dispatch
def coeffs_at(schedule: AffineSchedule, t: float | jax.Array) -> jax.Array:
    return schedule.offset + jnp.asarray(t) * schedule.slope


@coeffs_at.dispatch
def coeffs_at(schedule: CubicSchedule, t: float | jax.Array) -> jax.Array:
    t = jnp.asarray(t)
    return (
        ((schedule.cubic * t + schedule.quadratic) * t + schedule.linear) * t
        + schedule.constant
    )

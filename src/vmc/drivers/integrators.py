"""Time integrators and time units for TDVP dynamics."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import abc
import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp

from vmc.utils import _tree_add_scaled

__all__ = [
    "Integrator",
    "Euler",
    "RK4",
    "TimeUnit",
    "RealTimeUnit",
    "ImaginaryTimeUnit",
]


class Integrator(abc.ABC):
    """Abstract base class for time integrators."""

    @staticmethod
    @abc.abstractmethod
    def step(
        derivative_fn: Callable[..., tuple[Any, Any, Any]],
        state: Any,
        t: float,
        dt: float,
        carry: Any,
    ) -> tuple[Any, float, Any, Any]:
        """Perform one integration step.

        `state` is the ODE variable being integrated.
        `carry` is auxiliary evolving runtime context needed to evaluate the
        derivative, but it is not integrated with `dt`.
        """


class Euler(Integrator):
    """1st-order Euler integrator."""

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("derivative_fn",))
    def step(
        derivative_fn: Callable[..., tuple[Any, Any, Any]],
        state: Any,
        t: float,
        dt: float,
        carry: Any,
    ) -> tuple[Any, float, Any, Any]:
        derivative, carry_next, aux = derivative_fn(state, t, carry)
        return (
            _tree_add_scaled(state, derivative, dt),
            t + dt,
            carry_next,
            aux,
        )


class RK4(Integrator):
    """4th-order Runge-Kutta integrator."""

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("derivative_fn",))
    def step(
        derivative_fn: Callable[..., tuple[Any, Any, Any]],
        state: Any,
        t: float,
        dt: float,
        carry: Any,
    ) -> tuple[Any, float, Any, Any]:
        k1, carry1, _ = derivative_fn(state, t, carry)
        k2, carry2, _ = derivative_fn(
            _tree_add_scaled(state, k1, 0.5 * dt),
            t + 0.5 * dt,
            carry1,
        )

        k3, carry3, _ = derivative_fn(
            _tree_add_scaled(state, k2, 0.5 * dt),
            t + 0.5 * dt,
            carry2,
        )

        k4, carry4, aux = derivative_fn(
            _tree_add_scaled(state, k3, dt),
            t + dt,
            carry3,
        )
        incr = jax.tree.map(
            lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d)
            / jnp.asarray(6.0, dtype=a.dtype),
            k1,
            k2,
            k3,
            k4,
        )
        return (
            _tree_add_scaled(state, incr, dt),
            t + dt,
            carry4,
            aux,
        )


class TimeUnit(abc.ABC):
    """Abstract base class for time units (real/imaginary)."""

    @property
    @abc.abstractmethod
    def grad_factor(self) -> complex:
        """Multiplicative factor for the gradient."""

    @abc.abstractmethod
    def default_integrator(self) -> Integrator:
        """Default integrator for this time unit."""


class RealTimeUnit(TimeUnit):
    """Real-time propagation: i d/dt |psi> = H |psi>."""

    @property
    def grad_factor(self) -> complex:
        return -1.0j

    def default_integrator(self) -> Integrator:
        return RK4()


class ImaginaryTimeUnit(TimeUnit):
    """Imaginary-time propagation: d/dt |psi> = -H |psi>."""

    @property
    def grad_factor(self) -> complex:
        return -1.0

    def default_integrator(self) -> Integrator:
        return Euler()

"""Custom variational drivers for VMC and TDVP.

This module provides a single dynamics driver for real- and imaginary-time
propagation with pluggable integrators and preconditioners.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import abc
import os
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from netket import stats as nkstats
from flax import nnx
from tqdm.auto import tqdm

from VMC.preconditioners import SRPreconditioner
from VMC.utils.vmc_utils import local_estimate

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DynamicsDriver",
    "Integrator",
    "Euler",
    "RK4",
    "TimeUnit",
    "RealTimeUnit",
    "ImaginaryTimeUnit",
    "SRPreconditioner",
]

if "JAX_GPU_MAGMA_PATH" not in os.environ:
    _magma_path = "/usr/local/magma/lib"
    if os.path.isdir(_magma_path):
        os.environ["JAX_GPU_MAGMA_PATH"] = _magma_path


class Integrator(abc.ABC):
    """Abstract base class for time integrators."""

    @abc.abstractmethod
    def step(
        self,
        driver: "DynamicsDriver",
        params: Any,
        t: float,
        dt: float,
    ) -> Any:
        """Perform one integration step.

        Args:
            driver: The dynamics driver instance.
            params: Current parameters.
            t: Current time.
            dt: Time step.

        Returns:
            Updated parameters.
        """


class Euler(Integrator):
    """Forward Euler integrator (first-order)."""

    def step(
        self,
        driver: "DynamicsDriver",
        params: Any,
        t: float,
        dt: float,
    ) -> Any:
        k1 = driver._time_derivative(params, t, stage=0)
        return driver._tree_add_scaled(params, k1, dt)


class RK4(Integrator):
    """Classical 4th-order Runge-Kutta integrator."""

    def step(
        self,
        driver: "DynamicsDriver",
        params: Any,
        t: float,
        dt: float,
    ) -> Any:
        k1 = driver._time_derivative(params, t, stage=0)
        k2 = driver._time_derivative(
            driver._tree_add_scaled(params, k1, 0.5 * dt), t + 0.5 * dt, stage=1
        )
        k3 = driver._time_derivative(
            driver._tree_add_scaled(params, k2, 0.5 * dt), t + 0.5 * dt, stage=2
        )
        k4 = driver._time_derivative(
            driver._tree_add_scaled(params, k3, dt), t + dt, stage=3
        )
        incr = driver._tree_weighted_sum(k1, k2, k3, k4)
        return driver._tree_add_scaled(params, incr, dt)


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


class DynamicsDriver:
    """Unified dynamics driver for VMC and TDVP-style evolution."""

    def __init__(
        self,
        model,
        operator,
        *,
        sampler: Callable,
        preconditioner: SRPreconditioner,
        dt: float,
        t0: float = 0.0,
        time_unit: TimeUnit | None = None,
        integrator: Integrator | None = None,
        sampler_key: jax.Array | None = None,
    ):
        self.model = model
        self.operator = operator
        self.sampler = sampler
        self.preconditioner = preconditioner
        self.dt = float(dt)
        self.t = float(t0)
        self.step_count = 0
        self._loss_stats = None
        self.last_samples = None
        self.last_o = None
        self.last_p = None
        self._sampler_configuration = None

        if time_unit is None:
            time_unit = RealTimeUnit()
        self.time_unit = time_unit

        if integrator is None:
            integrator = time_unit.default_integrator()
        self.integrator = integrator

        if sampler_key is None:
            sampler_key = jax.random.key(0)
        self._sampler_key = sampler_key

        self.diag_shift_error: float | None = None
        self.residual_error: float | None = None
        self.solve_time: float | None = None

    def _sync_preconditioner_metrics(self) -> None:
        for name in ("diag_shift_error", "residual_error", "solve_time"):
            if hasattr(self.preconditioner, name):
                setattr(self, name, getattr(self.preconditioner, name))

    def _operator_at(self, t: float):
        if callable(self.operator) and not hasattr(self.operator, "get_conn_padded"):
            return self.operator(t)
        return self.operator

    def _time_derivative(self, params: Any, t: float, *, stage: int) -> Any:
        self._assign_params(self.model.tensors, params)
        samples, o, p, self._sampler_key, self._sampler_configuration = self.sampler(
            self.model,
            key=self._sampler_key,
            initial_configuration=self._sampler_configuration,
        )
        self.last_samples = samples
        self.last_o = o
        self.last_p = p
        op_t = self._operator_at(t)
        local_energies = local_estimate(self.model, samples, op_t)
        if stage == 0:
            self._loss_stats = nkstats.statistics(local_energies)

        updates = self.preconditioner.apply(
            self.model,
            samples,
            o,
            p,
            local_energies,
            step=self.step_count,
            grad_factor=self.time_unit.grad_factor,
            stage=stage,
        )
        if stage == 0:
            self._sync_preconditioner_metrics()
        return updates

    @staticmethod
    @jax.jit
    def _tree_add_scaled(base: Any, delta: Any, scale: float) -> Any:
        return jax.tree_util.tree_map(
            lambda x, y: jax.lax.add(
                x, jax.lax.mul(jnp.asarray(scale, dtype=y.dtype), y)
            ),
            base,
            delta,
        )

    @staticmethod
    @jax.jit
    def _tree_weighted_sum(k1: Any, k2: Any, k3: Any, k4: Any) -> Any:
        def combine(a, b, c, d):
            weighted = a + 2.0 * b + 2.0 * c + d
            return jax.lax.mul(weighted, jnp.asarray(1.0 / 6.0, dtype=weighted.dtype))

        return jax.tree_util.tree_map(combine, k1, k2, k3, k4)

    def _get_model_params(self) -> Any:
        return jax.tree_util.tree_map(jnp.asarray, self.model.tensors)

    @staticmethod
    def _assign_params(target: Any, values: Any) -> None:
        if isinstance(target, (list, tuple, nnx.List)):
            for target_item, value_item in zip(target, values):
                DynamicsDriver._assign_params(target_item, value_item)
            return
        if isinstance(target, nnx.Variable):
            target.copy_from(values)
            return
        target[...] = values

    def step(self, dt: float | None = None) -> None:
        dt_step = self.dt if dt is None else float(dt)
        params = self._get_model_params()
        params_new = self.integrator.step(self, params, self.t, dt_step)
        self._assign_params(self.model.tensors, params_new)
        self.t += dt_step
        self.step_count += 1

    def run(self, T: float, *, show_progress: bool = True) -> None:
        if T <= 0:
            return
        t_end = self.t + float(T)
        total = float(T)
        pbar = tqdm(total=total, disable=not show_progress, unit="t")
        n_steps = int(float(jnp.ceil(total / self.dt)))
        for _ in range(n_steps):
            remaining = t_end - self.t
            if remaining <= 0:
                break
            dt_step = self.dt if remaining > self.dt else remaining
            self.step(dt_step)
            if show_progress:
                pbar.update(dt_step)
        if show_progress:
            pbar.close()

    @property
    def energy(self) -> Any:
        return self._loss_stats

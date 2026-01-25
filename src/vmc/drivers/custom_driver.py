"""Custom variational drivers for VMC and TDVP.

This module provides a single dynamics driver for real- and imaginary-time
propagation with pluggable integrators and preconditioners.
"""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import abc
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from netket import stats as nkstats
from flax import nnx
from tqdm.auto import tqdm

from vmc.preconditioners import SRPreconditioner

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

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
        time_unit: TimeUnit = RealTimeUnit(),
        integrator: Integrator | None = None,
        sampler_key: jax.Array = jax.random.key(0),
    ):
        self.model = model
        self.operator = operator
        self.sampler = sampler
        self.preconditioner = preconditioner
        self.dt = float(dt)
        self.t = float(t0)
        self.step_count = 0
        self._loss_stats = None
        self._sampler_configuration = None
        self._graphdef, params, model_state = nnx.split(self.model, nnx.Param, ...)
        self._params = params.to_pure_dict()
        self._model_state = model_state.to_pure_dict()

        self.time_unit = time_unit
        self.integrator = integrator or self.time_unit.default_integrator()
        self._sampler_key = sampler_key

        self.diag_shift_error: float | None = None
        self.residual_error: float | None = None
        self.solve_time: float | None = None

    def _sync_preconditioner_metrics(self) -> None:
        self.diag_shift_error = getattr(self.preconditioner, "diag_shift_error", None)
        self.residual_error = getattr(self.preconditioner, "residual_error", None)
        self.solve_time = getattr(self.preconditioner, "solve_time", None)

    def _operator_at(self, t: float):
        if callable(self.operator) and not hasattr(self.operator, "get_conn_padded"):
            return self.operator(t)
        return self.operator

    def _time_derivative(self, params: Any, t: float, *, stage: int) -> Any:
        self.model = nnx.merge(self._graphdef, params, self._model_state)
        log_timing = logger.isEnabledFor(logging.INFO)
        if log_timing:
            t0 = time.perf_counter()
        op_t = self._operator_at(t)
        (
            samples,
            o,
            p,
            self._sampler_key,
            self._sampler_configuration,
            amp,
            local_energies,
        ) = self.sampler(
            self.model,
            op_t,
            key=self._sampler_key,
            initial_configuration=self._sampler_configuration,
        )
        if log_timing:
            jax.block_until_ready(samples)
            jax.block_until_ready(o)
            if p is not None:
                jax.block_until_ready(p)
            jax.block_until_ready(amp)
            jax.block_until_ready(local_energies)
            t1 = time.perf_counter()
            logger.info("Step %d stage %d | sampling %.3fs", self.step_count, stage, t1 - t0)
        if stage == 0:
            self._loss_stats = nkstats.statistics(local_energies)
            if log_timing:
                e = self._loss_stats
                logger.info(
                    "Step %d | E = %.6f ± %.4f [σ²=%.4f]",
                    self.step_count,
                    e.mean.real,
                    e.error_of_mean.real,
                    e.variance.real,
                )

        updates = self.preconditioner.apply(
            self.model,
            params,
            samples,
            o,
            p,
            local_energies,
            grad_factor=self.time_unit.grad_factor,
        )
        if log_timing:
            jax.block_until_ready(updates)
            t2 = time.perf_counter()
            logger.info(
                "Step %d stage %d | sr_solve %.3fs",
                self.step_count,
                stage,
                t2 - t1,
            )
        if stage == 0:
            self._sync_preconditioner_metrics()
        return updates

    @staticmethod
    @jax.jit
    def _tree_add_scaled(base: Any, delta: Any, scale: float) -> Any:
        return jax.tree_util.tree_map(
            lambda x, y: x + jnp.asarray(scale, dtype=y.dtype) * y,
            base,
            delta,
        )

    @staticmethod
    @jax.jit
    def _tree_weighted_sum(k1: Any, k2: Any, k3: Any, k4: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / jnp.asarray(6.0, dtype=a.dtype),
            k1, k2, k3, k4,
        )

    def _get_model_params(self) -> Any:
        return self._params

    def step(self, dt: float | None = None) -> None:
        dt_step = self.dt if dt is None else float(dt)
        params = self._get_model_params()
        params_new = self.integrator.step(self, params, self.t, dt_step)
        self._params = params_new
        self.model = nnx.merge(self._graphdef, self._params, self._model_state)
        self.t += dt_step
        self.step_count += 1

    def run(self, T: float, *, show_progress: bool = True) -> None:
        if T <= 0:
            return
        t_end = self.t + float(T)
        pbar = tqdm(total=float(T), disable=not show_progress, unit="t")
        while self.t < t_end:
            dt_step = min(self.dt, t_end - self.t)
            self.step(dt_step)
            pbar.update(dt_step)
        pbar.close()

    @property
    def energy(self) -> Any:
        return self._loss_stats

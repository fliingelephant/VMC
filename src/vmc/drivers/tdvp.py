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
from typing import Any

import jax
import jax.numpy as jnp
from netket import stats as nkstats
from flax import nnx
from tqdm.auto import tqdm

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.peps import build_mc_kernels
import vmc.peps.blockade.kernels  # noqa: F401  # register blockade build_mc_kernels dispatch
import vmc.peps.gi.kernels  # noqa: F401  # register GI build_mc_kernels dispatch
from vmc.preconditioners import SRPreconditioner

logger = logging.getLogger(__name__)

__all__ = [
    "TDVPDriver",
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
        driver: "TDVPDriver",
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
        driver: "TDVPDriver",
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
        driver: "TDVPDriver",
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


class TDVPDriver:
    """Unified dynamics driver for VMC and TDVP-style evolution."""

    def __init__(
        self,
        model,
        operator,
        *,
        preconditioner: SRPreconditioner,
        dt: float,
        t0: float = 0.0,
        time_unit: TimeUnit = RealTimeUnit(),
        integrator: Integrator | None = None,
        sampler_key: jax.Array = jax.random.key(0),
        n_samples: int = 1,
        n_chains: int = 1,
        full_gradient: bool = False,
    ):
        self.model = model
        self.operator = operator
        self.preconditioner = preconditioner
        self.dt = float(dt)
        self.t = float(t0)
        self.step_count = 0
        self.n_samples = int(n_samples)
        self.n_chains = int(n_chains)
        self.full_gradient = bool(full_gradient)
        self._loss_stats = None
        self._sampler_configuration = None
        self._graphdef, params, model_state = nnx.split(self.model, nnx.Param, ...)
        self._params = nnx.to_pure_dict(params)
        self._model_state = nnx.to_pure_dict(model_state)

        self.time_unit = time_unit
        self.integrator = integrator or self.time_unit.default_integrator()
        self._sampler_key = sampler_key

        self._sampler_key, init_key = jax.random.split(self._sampler_key)
        self._sampler_configuration = self.model.random_physical_configuration(
            init_key, n_samples=n_chains
        )

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
        t0 = time.perf_counter() if log_timing else 0.0

        operator = self._operator_at(t)
        _, num_chains, chain_length, total_samples = _sample_counts(
            self.n_samples,
            self.n_chains,
        )
        init_cache, transition, estimate = build_mc_kernels(
            self.model,
            operator,
            full_gradient=self.full_gradient,
        )
        mc_sampler = make_mc_sampler(transition, estimate)
        self._sampler_key, chain_key = jax.random.split(self._sampler_key)
        config_states = self._sampler_configuration.reshape(num_chains, -1)
        chain_keys = jax.random.split(chain_key, num_chains)
        tensors = [[jnp.asarray(t) for t in row] for row in self.model.tensors]
        cache = init_cache(tensors, config_states)
        (final_configurations, _, _), (samples_hist, estimates) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=chain_length,
        )
        samples = _trim_samples(samples_hist, total_samples, self.n_samples)
        o = _trim_samples(
            estimates.local_log_derivatives,
            total_samples,
            self.n_samples,
        )
        local_energies = _trim_samples(
            estimates.local_estimate,
            total_samples,
            self.n_samples,
        )
        if self.full_gradient:
            p = None
        else:
            p = _trim_samples(
                estimates.active_slice_indices,
                total_samples,
                self.n_samples,
            )
        self._sampler_configuration = final_configurations

        if log_timing:
            for arr in (samples, o, p, local_energies):
                if arr is not None:
                    jax.block_until_ready(arr)
            t1 = time.perf_counter()
            logger.info("Step %d stage %d | sampling %.3fs", self.step_count, stage, t1 - t0)

        if stage == 0:
            self._loss_stats = nkstats.statistics(local_energies)
            if log_timing:
                e = self._loss_stats
                logger.info(
                    "Step %d | E = %.6f +/- %.4f [var=%.4f]",
                    self.step_count,
                    e.mean.real,
                    e.error_of_mean.real,
                    e.variance.real,
                )

        updates = self.preconditioner.apply(
            self.model, params, samples, o, p, local_energies,
            grad_factor=self.time_unit.grad_factor,
        )

        if log_timing:
            jax.block_until_ready(updates)
            logger.info("Step %d stage %d | sr_solve %.3fs", self.step_count, stage, time.perf_counter() - t1)

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

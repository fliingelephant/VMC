"""Custom variational drivers for VMC and TDVP.

This module provides a single dynamics driver for real- and imaginary-time
propagation with pluggable integrators and preconditioners.
"""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import functools
import logging
import os
from typing import Any

import jax
from netket import stats as nkstats
from flax import nnx

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.drivers.integrators import (
    Euler,
    ImaginaryTimeUnit,
    Integrator,
    RK4,
    RealTimeUnit,
    TimeUnit,
)
from vmc.operators.local_terms import LocalHamiltonian
from vmc.peps import build_mc_kernels
import vmc.peps.blockade.kernels  # noqa: F401  # register blockade build_mc_kernels dispatch
import vmc.peps.gi.kernels  # noqa: F401  # register GI build_mc_kernels dispatch
from vmc.peps.gi.local_terms import GILocalHamiltonian
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


class TDVPDriver:
    """Unified dynamics driver for VMC and TDVP-style evolution."""

    def __init__(
        self,
        model,
        operator: LocalHamiltonian | GILocalHamiltonian,
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
        self._model = model
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
        self._graphdef, params, model_state = nnx.split(self._model, nnx.Param, ...)
        self._tensors = nnx.to_pure_dict(params)["tensors"]
        self._model_state = nnx.to_pure_dict(model_state)

        self.time_unit = time_unit
        self.integrator = integrator or self.time_unit.default_integrator()
        self._sampler_key = sampler_key

        self._sampler_key, init_key = jax.random.split(self._sampler_key)
        self._sampler_configuration = self._model.random_physical_configuration(
            init_key, n_samples=n_chains
        )
        init_cache, transition, estimate = build_mc_kernels(
            self._model,
            self.operator,
            full_gradient=self.full_gradient,
        )
        self._init_cache = init_cache
        self._mc_sampler = make_mc_sampler(transition, estimate)

        self.diag_shift_error: float | None = None
        self.residual_error: float | None = None
        self.solve_time: float | None = None

    def _sync_preconditioner_metrics(self, metrics: dict[str, Any]) -> None:
        self.diag_shift_error = metrics.get("diag_shift_error")
        self.residual_error = metrics.get("residual_error")
        self.solve_time = metrics.get("solve_time")
        self.preconditioner._metrics = metrics

    def _time_derivative(
        self,
        tensors: Any,
        t: float,
        carry: tuple[jax.Array, jax.Array],
    ) -> tuple[Any, tuple[jax.Array, jax.Array], tuple[jax.Array, dict[str, Any]]]:
        del t
        key, config_states = carry
        _, num_chains, chain_length, total_samples = _sample_counts(
            self.n_samples,
            self.n_chains,
        )
        key, chain_key = jax.random.split(key)
        chain_keys = jax.random.split(chain_key, num_chains)
        cache = self._init_cache(tensors, config_states)
        (config_states, _, _), (samples_hist, estimates) = self._mc_sampler(
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
        updates, metrics = self.preconditioner.apply(
            self._model,
            tensors,
            samples,
            o,
            p,
            local_energies,
            grad_factor=self.time_unit.grad_factor,
        )
        return updates, (key, config_states), (local_energies, metrics)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _run_n_steps(
        self,
        tensors: Any,
        t: float,
        key: jax.Array,
        config_states: jax.Array,
        n_steps: int,
    ) -> tuple[Any, float, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        tensors, t, (key, config_states), (local_energies, metrics) = self.integrator.step(
            self._time_derivative,
            tensors,
            t,
            self.dt,
            (key, config_states),
        )

        def body(
            _: int,
            carry: tuple[Any, float, jax.Array, jax.Array, jax.Array, dict[str, Any]],
        ) -> tuple[Any, float, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
            state, t_cur, key_cur, configs_cur, _, _ = carry
            state, t_next, (key_next, configs_next), (energies_next, metrics_next) = self.integrator.step(
                self._time_derivative,
                state,
                t_cur,
                self.dt,
                (key_cur, configs_cur),
            )
            return state, t_next, key_next, configs_next, energies_next, metrics_next

        return jax.lax.fori_loop(
            1,
            n_steps,
            body,
            (tensors, t, key, config_states, local_energies, metrics),
        )

    def run(self, T: float) -> None:
        duration = float(T)
        assert duration > 0.0, f"T must be positive, got {duration}"
        n_steps = int(round(duration / self.dt))
        eps = 1e-12 * max(1.0, abs(duration), abs(self.dt))
        assert abs(duration - n_steps * self.dt) <= eps, (
            f"T={duration} must be an integer multiple of dt={self.dt}"
        )
        assert n_steps > 0, (
            f"T={duration} with dt={self.dt} yields zero integration steps."
        )

        self._tensors, self.t, self._sampler_key, self._sampler_configuration, local_energies, metrics = (
            self._run_n_steps(
                self._tensors,
                self.t,
                self._sampler_key,
                self._sampler_configuration.reshape(self.n_chains, -1),
                n_steps,
            )
        )
        self._loss_stats = nkstats.statistics(local_energies)
        self._sync_preconditioner_metrics(metrics)
        self.step_count += n_steps
        if logger.isEnabledFor(logging.INFO):
            e = self._loss_stats
            logger.info(
                "Step %d | E = %.6f +/- %.4f [var=%.4f]",
                self.step_count - 1,
                e.mean.real,
                e.error_of_mean.real,
                e.variance.real,
            )

    @property
    def energy(self) -> Any:
        return self._loss_stats

    @property
    def model(self):
        return nnx.merge(
            self._graphdef,
            {"tensors": self._tensors},
            self._model_state,
        )

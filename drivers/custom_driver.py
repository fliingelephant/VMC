"""Custom variational drivers for VMC and TDVP.

This module provides custom driver implementations that extend NetKet's
variational drivers with support for custom preconditioners.

Uses ABC pattern for Integrator and PropagationType (single dispatch on self).
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import abc
import os
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from netket import stats as nkstats
from netket.driver.abstract_variational_driver import AbstractVariationalDriver
from netket.jax import tree_cast
from netket.operator import AbstractOperator
from netket.optimizer import identity_preconditioner
from netket.vqs import MCState
from tqdm.auto import tqdm

from VMC.gauge import GaugeConfig
from VMC.preconditioners import (
    DirectSolve,
    LinearSolver,
    MinSRFormulation,
    QRSolve,
    SRFormulation,
    SRPreconditioner,
    solve_cholesky,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    # VMC Drivers
    "CustomVMC",
    "CustomVMC_SR",
    "CustomVMC_QR",
    # TDVP Driver
    "CustomTDVP_SR",
    # Integrator classes (ABC pattern)
    "Integrator",
    "Euler",
    "RK4",
    # Propagation type classes (ABC pattern)
    "PropagationType",
    "RealTime",
    "ImaginaryTime",
    # Re-exports
    "GaugeConfig",
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
        driver: "CustomTDVP_SR",
        params: dict,
        t: float,
        dt: float,
    ) -> dict:
        """Perform one integration step.

        Args:
            driver: The TDVP driver instance.
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
        driver: "CustomTDVP_SR",
        params: dict,
        t: float,
        dt: float,
    ) -> dict:
        k1 = driver._time_derivative(params, t, stage=0)
        return driver._tree_add_scaled(params, k1, dt)


class RK4(Integrator):
    """Classical 4th-order Runge-Kutta integrator."""

    def step(
        self,
        driver: "CustomTDVP_SR",
        params: dict,
        t: float,
        dt: float,
    ) -> dict:
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




class PropagationType(abc.ABC):
    """Abstract base class for propagation types (real/imaginary time)."""

    @property
    @abc.abstractmethod
    def grad_factor(self) -> complex:
        """Multiplicative factor for the gradient."""

    @abc.abstractmethod
    def default_integrator(self) -> Integrator:
        """Default integrator for this propagation type."""


class RealTime(PropagationType):
    """Real-time propagation: i d/dt |psi> = H |psi>."""

    @property
    def grad_factor(self) -> complex:
        return -1.0j

    def default_integrator(self) -> Integrator:
        return RK4()


class ImaginaryTime(PropagationType):
    """Imaginary-time propagation: d/dt |psi> = -H |psi>."""

    @property
    def grad_factor(self) -> complex:
        return -1.0

    def default_integrator(self) -> Integrator:
        return Euler()




class CustomVMC(AbstractVariationalDriver):
    """Minimal VMC driver with customizable preconditioner.

    This driver computes energy gradients and applies an optional
    preconditioner (e.g., SR) before passing to the optimizer.

    Attributes:
        preconditioner: The gradient preconditioner to use.
        diag_shift_error: Last computed diagonal shift error (from preconditioner).
        residual_error: Last computed residual error (from preconditioner).
        solve_time: Last solver wall time (from preconditioner).
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Any,
        *,
        variational_state: MCState,
        preconditioner: Any = identity_preconditioner,
    ):
        """Initialize CustomVMC driver.

        Args:
            hamiltonian: The Hamiltonian operator.
            optimizer: Optimizer (e.g., SGD, Adam).
            variational_state: The variational Monte Carlo state.
            preconditioner: Gradient preconditioner (default: identity).
        """
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        self._ham = (
            hamiltonian.collect() if hasattr(hamiltonian, "collect") else hamiltonian
        )
        self.preconditioner = preconditioner or identity_preconditioner
        self.diag_shift_error: float | None = None
        self.residual_error: float | None = None
        self.solve_time: float | None = None

    def _sync_preconditioner_metrics(self) -> None:
        """Copy metrics from preconditioner to driver for logging."""
        for name in ("diag_shift_error", "residual_error", "solve_time"):
            if hasattr(self.preconditioner, name):
                setattr(self, name, getattr(self.preconditioner, name))

    def _forward_and_backward(self) -> dict:
        """Compute gradients and apply preconditioner.

        Returns:
            Parameter updates as a pytree.
        """
        self.state.reset()

        if getattr(self.preconditioner, "uses_local_energies", False):
            local_energies = self.state.local_estimators(self._ham)
            self._loss_stats = nkstats.statistics(local_energies)
            updates = self.preconditioner.apply(
                self.state,
                local_energies,
                step=self.step_count,
            )
            self._sync_preconditioner_metrics()
        else:
            self._loss_stats, grad = self.state.expect_and_grad(self._ham)
            updates = self.preconditioner(self.state, grad, self.step_count)

        return tree_cast(updates, self.state.parameters)

    @property
    def energy(self) -> Any:
        """Return the energy statistics from the last step."""
        return self._loss_stats


class CustomVMC_QR(CustomVMC):
    """VMC driver using pivoted QR decomposition for SR.

    This driver uses pivoted QR factorization to solve the SR equations,
    which provides better numerical stability for rank-deficient Jacobians.
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Any,
        *,
        variational_state: MCState,
        rcond: float | None = None,
        min_norm: bool = True,
        gauge_config: GaugeConfig | None = None,
    ):
        """Initialize CustomVMC_QR driver.

        Args:
            hamiltonian: The Hamiltonian operator.
            optimizer: Optimizer.
            variational_state: The variational state.
            rcond: Relative condition number for rank truncation.
            min_norm: If True, compute minimum-norm solution.
            gauge_config: Optional gauge configuration.
        """
        preconditioner = SRPreconditioner(
            formulation=SRFormulation(),
            strategy=QRSolve(rcond=rcond, min_norm=min_norm),
            gauge_config=gauge_config,
        )
        super().__init__(
            hamiltonian,
            optimizer,
            variational_state=variational_state,
            preconditioner=preconditioner,
        )

    def __repr__(self) -> str:
        return f"CustomVMC_QR(step_count={self.step_count}, state={self.state})"


class CustomVMC_SR(CustomVMC):
    """VMC driver using Stochastic Reconfiguration (SR/minSR).

    This driver implements the SR algorithm with optional minSR (NTK) formulation
    and various linear solvers.
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Any,
        *,
        diag_shift: float = 1e-2,
        variational_state: MCState,
        use_ntk: bool = True,
        gauge_config: GaugeConfig | None = None,
        solver: LinearSolver = solve_cholesky,
    ):
        """Initialize CustomVMC_SR driver.

        Args:
            hamiltonian: The Hamiltonian operator.
            optimizer: Optimizer.
            diag_shift: Regularization parameter for the QGT.
            variational_state: The variational state.
            use_ntk: If True, use minSR/NTK formulation (sample-space).
            gauge_config: Optional gauge configuration.
            solver: Linear solver function.
        """
        formulation = MinSRFormulation() if use_ntk else SRFormulation()
        preconditioner = SRPreconditioner(
            formulation=formulation,
            strategy=DirectSolve(solver),
            diag_shift=diag_shift,
            gauge_config=gauge_config,
        )
        super().__init__(
            hamiltonian,
            optimizer,
            variational_state=variational_state,
            preconditioner=preconditioner,
        )

    def __repr__(self) -> str:
        return f"CustomVMC_SR(step_count={self.step_count}, state={self.state})"




class CustomTDVP_SR:
    """TDVP driver with SR preconditioner for time evolution.

    This driver integrates real- or imaginary-time dynamics using
    explicit time-stepping with configurable integrators.
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        variational_state: MCState,
        *,
        dt: float,
        t0: float = 0.0,
        propagation: PropagationType | None = None,
        integrator: Integrator | None = None,
        diag_shift: float = 1e-2,
        use_ntk: bool = True,
        gauge_config: GaugeConfig | None = None,
        solver: LinearSolver = solve_cholesky,
        preconditioner: SRPreconditioner | None = None,
    ):
        """Initialize CustomTDVP_SR driver.

        Args:
            hamiltonian: The Hamiltonian operator.
            variational_state: The variational state.
            dt: Fixed time step for integration.
            t0: Initial time.
            propagation: Propagation type instance (default: RealTime).
            integrator: Time integrator instance (default: from propagation type).
            diag_shift: Regularization parameter for the QGT.
            use_ntk: If True, use minSR/NTK formulation.
            gauge_config: Optional gauge configuration.
            solver: Linear solver function.
            preconditioner: Optional pre-built preconditioner.
        """
        if dt <= 0:
            raise ValueError("dt must be a positive float.")

        # Set propagation type
        if propagation is None:
            propagation = RealTime()
        self.propagation = propagation
        self._loss_grad_factor = propagation.grad_factor

        # Set integrator (use propagation default if not provided)
        if integrator is None:
            integrator = propagation.default_integrator()
        self.integrator = integrator

        self.dt = float(dt)
        self.state = variational_state
        self._ham = (
            hamiltonian.collect() if hasattr(hamiltonian, "collect") else hamiltonian
        )
        self.t = float(t0)
        self.step_count = 0
        self._loss_stats = None

        if preconditioner is None:
            formulation = MinSRFormulation() if use_ntk else SRFormulation()
            preconditioner = SRPreconditioner(
                formulation=formulation,
                strategy=DirectSolve(solver),
                diag_shift=diag_shift,
                gauge_config=gauge_config,
            )
        self.preconditioner = preconditioner
        self.diag_shift_error: float | None = None
        self.residual_error: float | None = None
        self.solve_time: float | None = None

    def generator(self, t: float):
        """Get the Hamiltonian at time t (for time-dependent Hamiltonians)."""
        if callable(self._ham) and not isinstance(self._ham, AbstractOperator):
            return self._ham(t)
        return self._ham

    def _sync_preconditioner_metrics(self) -> None:
        """Copy metrics from preconditioner to driver for logging."""
        for name in ("diag_shift_error", "residual_error", "solve_time"):
            if hasattr(self.preconditioner, name):
                setattr(self, name, getattr(self.preconditioner, name))

    def _time_derivative(self, params: dict, t: float, *, stage: int) -> dict:
        """Compute the time derivative of parameters."""
        self.state.parameters = params
        self.state.reset()

        op_t = self.generator(t)
        local_energies = self.state.local_estimators(op_t)
        if stage == 0:
            self._loss_stats = nkstats.statistics(local_energies)

        updates = self.preconditioner.apply(
            self.state,
            local_energies,
            step=self.step_count,
            grad_factor=self._loss_grad_factor,
            stage=stage,
        )
        if stage == 0:
            self._sync_preconditioner_metrics()
        return tree_cast(updates, self.state.parameters)

    @staticmethod
    @jax.jit
    def _tree_add_scaled(base: dict, delta: dict, scale: float) -> dict:
        """Compute base + scale * delta for pytrees."""
        return jax.tree_util.tree_map(
            lambda x, y: jax.lax.add(
                x, jax.lax.mul(jnp.asarray(scale, dtype=y.dtype), y)
            ),
            base,
            delta,
        )

    @staticmethod
    @jax.jit
    def _tree_weighted_sum(k1: dict, k2: dict, k3: dict, k4: dict) -> dict:
        """Compute RK4 weighted sum: (k1 + 2*k2 + 2*k3 + k4) / 6."""

        def combine(a, b, c, d):
            weighted = a + 2.0 * b + 2.0 * c + d
            return jax.lax.mul(weighted, jnp.asarray(1.0 / 6.0, dtype=weighted.dtype))

        return jax.tree_util.tree_map(combine, k1, k2, k3, k4)

    def step(self, dt: float | None = None) -> None:
        """Perform one integration step."""
        dt_step = self.dt if dt is None else float(dt)
        params = self.state.parameters

        params_new = self.integrator.step(self, params, self.t, dt_step)

        params_new = tree_cast(params_new, self.state.parameters)
        self.state.parameters = params_new
        self.state.reset()
        self.t += dt_step
        self.step_count += 1

    def run(self, T: float, *, show_progress: bool = True, out=None) -> None:
        """Run the time evolution for a total time T."""
        if T <= 0:
            return
        _ = out
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

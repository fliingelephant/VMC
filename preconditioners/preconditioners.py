"""Preconditioners for variational Monte Carlo optimization.

This module provides SR (Stochastic Reconfiguration) preconditioners with a
two-level abstraction using plum multiple dispatch:
- SpaceFormulation types: How to build the linear system (SR vs minSR)
- SolveStrategy types: How to solve it (Direct vs QR-enhanced)
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.sparse.linalg as jspsl
from netket.jax import tree_cast
from plum import dispatch

from VMC.utils.vmc_utils import build_dense_jac, flatten_samples

if TYPE_CHECKING:
    from VMC.gauge import GaugeConfig

logger = logging.getLogger(__name__)

__all__ = [
    # Linear solvers
    "LinearSolver",
    "solve_cg",
    "solve_cholesky",
    "solve_svd",
    # Space formulations (marker types)
    "SRFormulation",
    "MinSRFormulation",
    # Solve strategies (marker types)
    "DirectSolve",
    "QRSolve",
    # Dispatched functions
    "build_system",
    "recover_updates",
    "solve",
    # Unified preconditioner
    "SRPreconditioner",
]


LinearSolver = Callable[[jax.Array, jax.Array], jax.Array]


@functools.partial(jax.jit, static_argnames=("maxiter", "tol"))
def solve_cg(
    mat: jax.Array, rhs: jax.Array, *, maxiter: int = 1000, tol: float = 1e-5
) -> jax.Array:
    """Solve linear system using conjugate gradient."""
    sol, _ = jspsl.cg(mat, rhs, maxiter=maxiter, tol=tol)
    return sol


@functools.partial(jax.jit, static_argnames=("rcond",))
def solve_svd(mat: jax.Array, rhs: jax.Array, *, rcond: float = 1e-12) -> jax.Array:
    """Solve linear system using SVD with regularization."""
    U, S, Vh = jsp.linalg.svd(mat, full_matrices=False)
    cutoff = rcond * jnp.max(S)
    S_inv = jnp.where(S > cutoff, 1.0 / S, 0.0)
    y = U.conj().T @ rhs
    return Vh.conj().T @ (S_inv * y)


@jax.jit
def solve_cholesky(mat: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve positive-definite linear system using Cholesky decomposition."""
    return jsp.linalg.solve(mat, rhs, assume_a="pos")


@dataclass(frozen=True)
class SRFormulation:
    """Parameter-space SR formulation marker.

    Solves: (O^H @ O + diag_shift * I) @ x = O^H @ dv
    Matrix size: (n_params_reduced, n_params_reduced)
    """

    pass


@dataclass(frozen=True)
class MinSRFormulation:
    """Sample-space SR formulation marker (minSR / NTK).

    Solves: (O @ O^H + diag_shift * I) @ x = dv
    Then recovers: updates = O^H @ x
    Matrix size: (n_samples, n_samples)
    """

    pass


@dispatch
def build_system(
    f: SRFormulation, O_eff: jax.Array, dv: jax.Array, diag_shift: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the linear system for parameter-space SR.

    Args:
        f: SRFormulation marker.
        O_eff: Effective Jacobian matrix (n_samples, n_params_reduced).
        dv: Gradient vector (n_samples,).
        diag_shift: Regularization parameter.

    Returns:
        Tuple of (matrix, rhs, base_matrix) where base_matrix is without diag_shift.
    """
    base_mat = O_eff.conj().T @ O_eff
    mat = base_mat + diag_shift * jnp.eye(base_mat.shape[0], dtype=base_mat.dtype)
    rhs = O_eff.conj().T @ dv
    return mat, rhs, base_mat


@dispatch
def build_system(
    f: MinSRFormulation, O_eff: jax.Array, dv: jax.Array, diag_shift: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the linear system for sample-space SR (minSR/NTK).

    Args:
        f: MinSRFormulation marker.
        O_eff: Effective Jacobian matrix (n_samples, n_params_reduced).
        dv: Gradient vector (n_samples,).
        diag_shift: Regularization parameter.

    Returns:
        Tuple of (matrix, rhs, base_matrix) where base_matrix is without diag_shift.
    """
    base_mat = O_eff @ O_eff.conj().T
    mat = base_mat + diag_shift * jnp.eye(base_mat.shape[0], dtype=base_mat.dtype)
    rhs = dv
    return mat, rhs, base_mat


@dispatch
def recover_updates(f: SRFormulation, O_eff: jax.Array, solution: jax.Array) -> jax.Array:
    """Transform solution back to parameter space for SR formulation.

    Args:
        f: SRFormulation marker.
        O_eff: Effective Jacobian matrix.
        solution: Solution from the linear solver.

    Returns:
        Updates in the reduced parameter space.
    """
    # Solution is already in parameter space
    return solution


@dispatch
def recover_updates(
    f: MinSRFormulation, O_eff: jax.Array, solution: jax.Array
) -> jax.Array:
    """Transform solution back to parameter space for minSR formulation.

    Args:
        f: MinSRFormulation marker.
        O_eff: Effective Jacobian matrix.
        solution: Solution from the linear solver.

    Returns:
        Updates in the reduced parameter space.
    """
    # Transform from sample space to parameter space
    return O_eff.conj().T @ solution


@dataclass(frozen=True)
class DirectSolve:
    """Direct solve strategy marker.

    Solves via matrix inverse with a dispatched linear solver.
    Supports: solve_cholesky, solve_svd, solve_cg, or any compatible solver.

    Attributes:
        solver: Linear solver function (default: solve_cholesky).
    """

    solver: LinearSolver = solve_cholesky


@dataclass(frozen=True)
class QRSolve:
    """QR solve strategy marker.

    Solves via pivoted QR decomposition.
    Only valid with SRFormulation.

    Attributes:
        rcond: Relative condition number for rank truncation.
        min_norm: If True, compute minimum-norm solution.
    """

    rcond: float | None = None
    min_norm: bool = True


@dispatch
def solve(
    strategy: DirectSolve,
    formulation: SRFormulation,
    O_eff: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """Solve SR equations using direct method with SRFormulation.

    Args:
        strategy: DirectSolve configuration.
        formulation: SRFormulation marker.
        O_eff: Effective Jacobian matrix.
        dv: Gradient vector.
        diag_shift: Regularization parameter.

    Returns:
        Tuple of (updates_reduced, metrics_dict).
    """
    mat, rhs, base_mat = build_system(formulation, O_eff, dv, diag_shift)
    solution = strategy.solver(mat, rhs)
    updates_red = recover_updates(formulation, O_eff, solution)

    metrics = {
        "diag_shift_error": (
            jnp.linalg.norm(mat @ solution - rhs) ** 2 / jnp.linalg.norm(rhs) ** 2
        ),
        "residual_error": (
            jnp.linalg.norm(base_mat @ solution - rhs) ** 2 / jnp.linalg.norm(rhs) ** 2
        ),
    }
    return updates_red, metrics


@dispatch
def solve(
    strategy: DirectSolve,
    formulation: MinSRFormulation,
    O_eff: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """Solve SR equations using direct method with MinSRFormulation.

    Args:
        strategy: DirectSolve configuration.
        formulation: MinSRFormulation marker.
        O_eff: Effective Jacobian matrix.
        dv: Gradient vector.
        diag_shift: Regularization parameter.

    Returns:
        Tuple of (updates_reduced, metrics_dict).
    """
    mat, rhs, base_mat = build_system(formulation, O_eff, dv, diag_shift)
    solution = strategy.solver(mat, rhs)
    updates_red = recover_updates(formulation, O_eff, solution)

    metrics = {
        "diag_shift_error": (
            jnp.linalg.norm(mat @ solution - rhs) ** 2 / jnp.linalg.norm(rhs) ** 2
        ),
        "residual_error": (
            jnp.linalg.norm(base_mat @ solution - rhs) ** 2 / jnp.linalg.norm(rhs) ** 2
        ),
    }
    return updates_red, metrics


@dispatch
def solve(
    strategy: QRSolve,
    formulation: SRFormulation,
    O_eff: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """Solve SR equations using pivoted QR with SRFormulation.

    Args:
        strategy: QRSolve configuration.
        formulation: SRFormulation marker.
        O_eff: Effective Jacobian matrix.
        dv: Gradient vector.
        diag_shift: Regularization parameter (unused in QR method).

    Returns:
        Tuple of (updates_reduced, metrics_dict).
    """
    # Pivoted QR decomposition
    q, r, piv = jax.lax.linalg.qr(
        O_eff, full_matrices=False, pivoting=True, use_magma=True
    )
    y = q.conj().T @ dv

    # Determine rank
    if strategy.rcond is None:
        r_rank = int(jnp.linalg.matrix_rank(r))
    else:
        r_rank = int(jnp.linalg.matrix_rank(r, rtol=strategy.rcond))

    n_red = r.shape[1]
    n2 = n_red - r_rank

    if r_rank == 0:
        updates_red = jnp.zeros((n_red,), dtype=O_eff.dtype)
    else:
        r11 = r[:r_rank, :r_rank]
        r12 = r[:r_rank, r_rank:]
        y1 = y[:r_rank]

        if strategy.min_norm and n2 > 0:
            r11_inv_y1 = jsp.linalg.solve_triangular(r11, y1, lower=False)
            r11_inv_r12 = jsp.linalg.solve_triangular(r11, r12, lower=False)
            lhs = r11_inv_r12.conj().T @ r11_inv_r12 + jnp.eye(n2, dtype=r12.dtype)
            rhs_minorm = r11_inv_r12.conj().T @ r11_inv_y1
            x2 = jsp.linalg.solve(lhs, rhs_minorm, assume_a="pos")
            x1 = r11_inv_y1 - r11_inv_r12 @ x2
        else:
            x1 = jsp.linalg.solve_triangular(r11, y1, lower=False)
            x2 = jnp.zeros((n2,), dtype=O_eff.dtype)

        x = jnp.concatenate([x1, x2], axis=0)
        updates_red = jnp.zeros((n_red,), dtype=O_eff.dtype).at[piv].set(x)

    # Compute metrics
    rhs_full = O_eff.conj().T @ dv
    resid = O_eff.conj().T @ (O_eff @ updates_red) - rhs_full
    numer = jnp.linalg.norm(resid) ** 2
    denom = jnp.linalg.norm(rhs_full) ** 2
    residual_error = jnp.where(denom > 0, numer / denom, jnp.inf)

    metrics = {
        "residual_error": residual_error,
        "rank_R": r_rank,
        "rcond": strategy.rcond,
    }
    return updates_red, metrics


@dispatch
def solve(
    strategy: QRSolve,
    formulation: MinSRFormulation,
    O_eff: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """QR solve with MinSRFormulation is not implemented.

    Raises:
        NotImplementedError: Always raises - use DirectSolve with MinSRFormulation.
    """
    raise NotImplementedError(
        "QR-enhanced solving for minSR formulation is not yet implemented. "
        "Use DirectSolve with MinSRFormulation, or use "
        "SRFormulation with QRSolve."
    )


# Type aliases for formulation and strategy markers
FormulationType = SRFormulation | MinSRFormulation
StrategyType = DirectSolve | QRSolve


class SRPreconditioner:
    """Unified SR preconditioner with two-level abstraction.

    Composes a formulation type (SR vs minSR) with a strategy type
    (Direct vs QR-enhanced) using plum dispatch.

    Attributes:
        formulation: Space formulation marker (SRFormulation or MinSRFormulation).
        strategy: Solving strategy marker (DirectSolve or QRSolve).
        diag_shift: Regularization parameter added to the diagonal.
        gauge_config: Optional gauge projection configuration for tensor networks.
        uses_local_energies: Flag indicating this preconditioner uses local energies.
    """

    def __init__(
        self,
        formulation: FormulationType | None = None,
        strategy: StrategyType | None = None,
        diag_shift: float = 1e-2,
        gauge_config: "GaugeConfig | None" = None,
    ):
        """Initialize SRPreconditioner.

        Args:
            formulation: Space formulation marker (default: SRFormulation()).
            strategy: Solving strategy marker (default: DirectSolve()).
            diag_shift: Regularization parameter.
            gauge_config: Optional gauge projection config for MPS/PEPS.
        """
        self.formulation = formulation if formulation is not None else SRFormulation()
        self.strategy = strategy if strategy is not None else DirectSolve()
        self.diag_shift = diag_shift
        self.gauge_config = gauge_config
        self.uses_local_energies = True

        # Runtime metrics (mutable, lazily computed)
        self._metrics: dict | None = None

    @property
    def diag_shift_error(self) -> float | None:
        """Diagonal shift error from last solve (materialized on access)."""
        if self._metrics is None or "diag_shift_error" not in self._metrics:
            return None
        return float(jax.block_until_ready(self._metrics["diag_shift_error"]))

    @property
    def residual_error(self) -> float | None:
        """Residual error from last solve (materialized on access)."""
        if self._metrics is None or "residual_error" not in self._metrics:
            return None
        return float(jax.block_until_ready(self._metrics["residual_error"]))

    def apply(
        self,
        state,
        local_energies: jax.Array,
        *,
        step: int | None = None,
        grad_factor: complex = 1.0,
        stage: int = 0,
    ) -> dict:
        """Apply SR preconditioner to compute parameter updates.

        Args:
            state: Variational state with samples and parameters.
            local_energies: Local energy estimates for each sample.
            step: Current optimization step (for logging).
            grad_factor: Multiplicative factor for the gradient.
            stage: RK stage index (only log on stage 0).

        Returns:
            Parameter updates as a pytree matching state.parameters.
        """
        # Import here to avoid circular import
        from VMC.gauge import compute_gauge_projection

        samples = flatten_samples(state.samples)
        O = build_dense_jac(
            state._apply_fun,
            state.parameters,
            state.model_state,
            samples,
        )

        # Compute gradient vector
        de = local_energies.reshape(-1) - jnp.mean(local_energies)
        dv = de / jnp.sqrt(samples.shape[0])
        dv = grad_factor * dv

        # Apply gauge removal if provided
        Q = None
        gauge_info = None
        O_eff = O
        if self.gauge_config is not None:
            Q, gauge_info = compute_gauge_projection(
                self.gauge_config,
                state.model,
                state.parameters,
                return_info=True,
            )
            O_eff = O @ Q

        # Solve using dispatch (strategy + formulation determine implementation)
        updates_red, self._metrics = solve(
            self.strategy, self.formulation, O_eff, dv, self.diag_shift
        )

        # Transform back to full parameter space
        updates_flat = Q @ updates_red if Q is not None else updates_red

        # Debug logging (guarded to skip expensive computations when not debugging)
        if logger.isEnabledFor(logging.DEBUG) and stage == 0:
            msg = (
                f"[SRPreconditioner] formulation={type(self.formulation).__name__} "
                f"strategy={type(self.strategy).__name__} shape(O)={O.shape} rank(O)={int(jnp.linalg.matrix_rank(O))} "
            )
            if "diag_shift_error" in self._metrics:
                msg += f" diag_shift_error={self.diag_shift_error:.3e}"
            if "residual_error" in self._metrics:
                msg += f" residual_error={self.residual_error:.3e}"
            if gauge_info is not None:
                msg += (
                    f" shape(O_eff)={O_eff.shape} rank(O_eff)={int(jnp.linalg.matrix_rank(O_eff))} "
                    f"rank(T)={gauge_info['rank_T']} null_rank={gauge_info['null_rank']}"
                )
            logger.debug(msg)

        # Unravel to pytree structure
        _, unravel = jax.flatten_util.ravel_pytree(state.parameters)
        updates = unravel(updates_flat)
        if jnp.isrealobj(grad_factor):
            updates = jax.tree_util.tree_map(
                lambda x, target: (
                    (2.0 * x.real).astype(target.dtype)
                    if not jnp.iscomplexobj(target)
                    else x
                ),
                updates,
                state.parameters,
            )
        return tree_cast(updates, state.parameters)

"""Preconditioners for variational Monte Carlo optimization."""
from __future__ import annotations

from VMC import config  # noqa: F401

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from netket.jax import tree_cast
from plum import dispatch

from VMC.qgt import DiagonalQGT, QGT, Jacobian, ParameterSpace, SampleSpace
from VMC.qgt.solvers import solve_cg, solve_cholesky, solve_svd
from VMC.utils.vmc_utils import build_dense_jac, flatten_samples

if TYPE_CHECKING:
    from VMC.gauge import GaugeConfig

logger = logging.getLogger(__name__)

__all__ = [
    "LinearSolver",
    "solve_cg",
    "solve_cholesky",
    "solve_svd",
    "DirectSolve",
    "QRSolve",
    "DiagonalSolve",
    "SRPreconditioner",
]

LinearSolver = Callable[[jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class DirectSolve:
    """Direct solve strategy using matrix solver."""

    solver: LinearSolver = solve_cholesky


@dataclass(frozen=True)
class QRSolve:
    """QR solve strategy (only for ParameterSpace)."""

    rcond: float | None = None
    min_norm: bool = True


@dataclass(frozen=True)
class DiagonalSolve:
    """Block-diagonal solve strategy (ParameterSpace only)."""

    solver: LinearSolver = solve_cholesky
    params_per_site: tuple[int, ...] | None = None


# --------------------------------------------------------------------------- #
# Solve dispatch
# --------------------------------------------------------------------------- #


@dispatch
def _solve_sr(
    strategy: DirectSolve,
    space: ParameterSpace,
    O: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """Direct solve in parameter space: (O†O + λI) x = O†dv."""
    qgt = QGT(Jacobian(O), space=ParameterSpace())
    S = qgt.to_dense()
    rhs = O.conj().T @ dv
    mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
    x = strategy.solver(mat, rhs)
    metrics = {
        "residual_error": jnp.linalg.norm(S @ x - rhs) ** 2 / jnp.linalg.norm(rhs) ** 2,
    }
    return x, metrics


@dispatch
def _solve_sr(
    strategy: DirectSolve,
    space: SampleSpace,
    O: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """Direct solve in sample space: (OO† + λI) y = dv, then x = O†y."""
    qgt = QGT(Jacobian(O), space=SampleSpace())
    S = qgt.to_dense()
    mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
    y = strategy.solver(mat, dv)
    x = O.conj().T @ y  # Recovery step
    metrics = {
        "residual_error": jnp.linalg.norm(S @ y - dv) ** 2 / jnp.linalg.norm(dv) ** 2,
    }
    return x, metrics


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: ParameterSpace,
    O: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """QR solve in parameter space."""
    q, r, piv = jax.lax.linalg.qr(O, full_matrices=False, pivoting=True, use_magma=True)
    y = q.conj().T @ dv

    if strategy.rcond is None:
        r_rank = int(jnp.linalg.matrix_rank(r))
    else:
        r_rank = int(jnp.linalg.matrix_rank(r, rtol=strategy.rcond))

    n_red = r.shape[1]
    n2 = n_red - r_rank

    if r_rank == 0:
        x = jnp.zeros((n_red,), dtype=O.dtype)
    else:
        r11, r12, y1 = r[:r_rank, :r_rank], r[:r_rank, r_rank:], y[:r_rank]

        if strategy.min_norm and n2 > 0:
            r11_inv_y1 = jsp.linalg.solve_triangular(r11, y1, lower=False)
            r11_inv_r12 = jsp.linalg.solve_triangular(r11, r12, lower=False)
            lhs = r11_inv_r12.conj().T @ r11_inv_r12 + jnp.eye(n2, dtype=r12.dtype)
            x2 = jsp.linalg.solve(lhs, r11_inv_r12.conj().T @ r11_inv_y1, assume_a="pos")
            x1 = r11_inv_y1 - r11_inv_r12 @ x2
        else:
            x1 = jsp.linalg.solve_triangular(r11, y1, lower=False)
            x2 = jnp.zeros((n2,), dtype=O.dtype)

        x = jnp.zeros((n_red,), dtype=O.dtype).at[piv].set(jnp.concatenate([x1, x2]))

    rhs = O.conj().T @ dv
    resid = O.conj().T @ (O @ x) - rhs
    metrics = {
        "residual_error": jnp.linalg.norm(resid) ** 2 / jnp.linalg.norm(rhs) ** 2,
        "rank": r_rank,
    }
    return x, metrics


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: SampleSpace,
    O: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    raise NotImplementedError("QRSolve not supported for SampleSpace")


@dispatch
def _solve_sr(
    strategy: DiagonalSolve,
    space: ParameterSpace,
    O: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    qgt = DiagonalQGT(Jacobian(O), space=ParameterSpace(), params_per_site=strategy.params_per_site)
    S = qgt.to_dense()
    rhs = O.conj().T @ dv
    mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
    x = jnp.zeros_like(rhs)
    i = 0
    for n in strategy.params_per_site:
        block = mat[i : i + n, i : i + n]
        x = x.at[i : i + n].set(strategy.solver(block, rhs[i : i + n]))
        i += n
    metrics = {
        "residual_error": jnp.linalg.norm(S @ x - rhs) ** 2 / jnp.linalg.norm(rhs) ** 2,
    }
    return x, metrics


@dispatch
def _solve_sr(
    strategy: DiagonalSolve,
    space: SampleSpace,
    O: jax.Array,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    raise NotImplementedError("DiagonalSolve not supported for SampleSpace")


# --------------------------------------------------------------------------- #
# SRPreconditioner
# --------------------------------------------------------------------------- #


class SRPreconditioner:
    """SR preconditioner with configurable space and solve strategy."""

    def __init__(
        self,
        space: ParameterSpace | SampleSpace | None = None,
        strategy: DirectSolve | QRSolve | DiagonalSolve | None = None,
        diag_shift: float = 1e-2,
        gauge_config: "GaugeConfig | None" = None,
    ):
        self.space = space if space is not None else ParameterSpace()
        self.strategy = strategy if strategy is not None else DirectSolve()
        self.diag_shift = diag_shift
        self.gauge_config = gauge_config
        self.uses_local_energies = True
        self._metrics: dict | None = None

    @property
    def residual_error(self) -> float | None:
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
        from VMC.gauge import compute_gauge_projection

        samples = flatten_samples(state.samples)
        O = build_dense_jac(state._apply_fun, state.parameters, state.model_state, samples)

        dv = (local_energies.reshape(-1) - jnp.mean(local_energies)) / jnp.sqrt(samples.shape[0])
        dv = grad_factor * dv

        # Gauge projection
        Q = None
        O_eff = O
        if self.gauge_config is not None:
            Q, _ = compute_gauge_projection(self.gauge_config, state.model, state.parameters, return_info=True)
            O_eff = O @ Q

        # Solve
        updates_red, self._metrics = _solve_sr(self.strategy, self.space, O_eff, dv, self.diag_shift)

        # Transform back
        updates_flat = Q @ updates_red if Q is not None else updates_red

        # Unravel
        _, unravel = jax.flatten_util.ravel_pytree(state.parameters)
        updates = unravel(updates_flat)
        if jnp.isrealobj(grad_factor):
            updates = jax.tree_util.tree_map(
                lambda x, t: (2.0 * x.real).astype(t.dtype) if not jnp.iscomplexobj(t) else x,
                updates,
                state.parameters,
            )
        return tree_cast(updates, state.parameters)

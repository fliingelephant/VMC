"""Preconditioners for variational Monte Carlo optimization."""
from __future__ import annotations

from vmc import config  # noqa: F401

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from netket.jax import tree_cast
from plum import dispatch

from vmc.qgt import QGT, Jacobian, ParameterSpace, SampleSpace, SlicedJacobian
from vmc.qgt.jacobian import SliceOrdering, SiteOrdering, jacobian_mean
from vmc.qgt.qgt import _params_per_site
from vmc.qgt.solvers import solve_cg, solve_cholesky, solve_svd
from vmc.utils.smallo import params_per_site, sliced_dims

if TYPE_CHECKING:
    from vmc.gauge import GaugeConfig

logger = logging.getLogger(__name__)

__all__ = [
    "LinearSolver",
    "solve_cg",
    "solve_cholesky",
    "solve_svd",
    "DirectSolve",
    "QRSolve",
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


# --------------------------------------------------------------------------- #
# Solve dispatch
# --------------------------------------------------------------------------- #


@dispatch
def _reorder_updates(
    ordering: SliceOrdering,
    updates_flat: jax.Array,
    pps: tuple[int, ...],
    phys_dim: int,
) -> jax.Array:
    total = sum(pps)
    perm = []
    site_offset = 0
    for n in pps:
        for k in range(phys_dim):
            base = k * total + site_offset
            perm.extend(range(base, base + n))
        site_offset += n
    return updates_flat[jnp.asarray(perm)]


@dispatch
def _reorder_updates(
    ordering: SiteOrdering,
    updates_flat: jax.Array,
    pps: tuple[int, ...],
    phys_dim: int,
) -> jax.Array:
    return updates_flat


@dispatch
def _adjoint_matvec(jac: Jacobian, v: jax.Array) -> jax.Array:
    mean = jacobian_mean(jac)
    return jac.O.conj().T @ v - mean.conj() * jnp.sum(v)


@dispatch
def _adjoint_matvec(jac: SlicedJacobian, v: jax.Array) -> jax.Array:
    from vmc.qgt.qgt import _iter_sliced_blocks

    o, p = jac.o, jac.p
    parts = [ok.conj().T @ v for ok, _ in _iter_sliced_blocks(o, p, jac.sliced_dims, jac.ordering)]
    result = jnp.concatenate(parts, axis=0)
    mean = jacobian_mean(jac)
    return result - mean.conj() * jnp.sum(v)


@dispatch
def _direct_solve(
    space: ParameterSpace,
    jac: Jacobian | SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
    solver: LinearSolver,
) -> tuple[jax.Array, dict]:
    rhs = _adjoint_matvec(jac, dv)
    qgt = QGT(jac, space=ParameterSpace())
    S = qgt.to_dense()
    mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
    x = solver(mat, rhs)
    metrics = {
        "residual_error": jnp.linalg.norm(S @ x - rhs) ** 2 / jnp.linalg.norm(rhs) ** 2,
    }
    return x, metrics


@dispatch
def _direct_solve(
    space: SampleSpace,
    jac: Jacobian | SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
    solver: LinearSolver,
) -> tuple[jax.Array, dict]:
    qgt = QGT(jac, space=SampleSpace())
    S = qgt.to_dense()
    mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
    y = solver(mat, dv)
    x = _adjoint_matvec(jac, y)
    metrics = {
        "residual_error": jnp.linalg.norm(S @ y - dv) ** 2 / jnp.linalg.norm(dv) ** 2,
    }
    return x, metrics


@dispatch
def _solve_sr(
    strategy: DirectSolve,
    space: ParameterSpace | SampleSpace,
    jac: Jacobian | SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    return _direct_solve(space, jac, dv, diag_shift, strategy.solver)


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: ParameterSpace,
    jac: Jacobian,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    """QR solve in parameter space."""
    # TODO: diag_shift not used
    mean = jacobian_mean(jac)
    o_centered = jac.O - mean[None, :]
    q, r, piv = jax.lax.linalg.qr(
        o_centered, full_matrices=False, pivoting=True, use_magma=True
    )
    y = q.conj().T @ dv

    if strategy.rcond is None:
        r_rank = int(jnp.linalg.matrix_rank(r))
    else:
        r_rank = int(jnp.linalg.matrix_rank(r, rtol=strategy.rcond))

    n_red = r.shape[1]
    n2 = n_red - r_rank

    if r_rank == 0:
        x = jnp.zeros((n_red,), dtype=jac.O.dtype)
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
            x2 = jnp.zeros((n2,), dtype=jac.O.dtype)

        x = jnp.zeros((n_red,), dtype=jac.O.dtype).at[piv].set(jnp.concatenate([x1, x2]))

    rhs = o_centered.conj().T @ dv
    resid = o_centered.conj().T @ (o_centered @ x) - rhs
    metrics = {
        "residual_error": jnp.linalg.norm(resid) ** 2 / jnp.linalg.norm(rhs) ** 2,
        "rank": r_rank,
    }
    return x, metrics


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: ParameterSpace,
    jac: SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    raise NotImplementedError("QRSolve not supported for SlicedJacobian")


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: SampleSpace,
    jac: Jacobian | SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
) -> tuple[jax.Array, dict]:
    raise NotImplementedError("QRSolve not supported for SampleSpace")


# --------------------------------------------------------------------------- #
# SRPreconditioner
# --------------------------------------------------------------------------- #


class SRPreconditioner:
    """SR preconditioner with configurable space and solve strategy."""

    def __init__(
        self,
        space: ParameterSpace | SampleSpace = ParameterSpace(),
        strategy: DirectSolve | QRSolve = DirectSolve(),
        diag_shift: float = 1e-2,
        gauge_config: "GaugeConfig | None" = None,
        ordering: SliceOrdering | SiteOrdering = SliceOrdering(),
    ):
        self.space = space
        self.strategy = strategy
        self.diag_shift = diag_shift
        self.gauge_config = gauge_config
        self.ordering = ordering
        self.uses_local_energies = True
        self._metrics: dict = {}

    @property
    def residual_error(self) -> jax.Array | None:
        """Return residual error from last solve."""
        return self._metrics.get("residual_error")

    def apply(
        self,
        model,
        params: Any,
        samples: jax.Array,
        o: jax.Array,
        p: jax.Array | None,
        local_energies: jax.Array,
        *,
        grad_factor: complex = 1.0,
    ) -> dict:
        from vmc.gauge import compute_gauge_projection

        dv = (local_energies.reshape(-1) - jnp.mean(local_energies)) / samples.shape[0]
        dv = grad_factor * dv

        params = jax.tree_util.tree_map(jnp.asarray, params)
        pps = tuple(params_per_site(model)) if p is not None else None
        sd = sliced_dims(model)
        Q = None
        if self.gauge_config is not None:
            Q, _ = compute_gauge_projection(
                self.gauge_config, model, params, return_info=True
            )
            if p is None:
                o_eff = o @ Q
            else:
                from vmc.qgt.qgt import _iter_sliced_blocks

                blocks = [ok for ok, _ in _iter_sliced_blocks(o, p, sd, self.ordering)]
                o_eff = jnp.concatenate(blocks, axis=1) @ Q
            jac = Jacobian(o_eff)
        elif p is None:
            jac = Jacobian(o)
        else:
            jac = SlicedJacobian(
                o,
                p,
                sd,
                self.ordering,
            )

        strategy = self.strategy
        updates_red, self._metrics = _solve_sr(
            strategy, self.space, jac, dv, self.diag_shift
        )

        updates_flat = Q @ updates_red if Q is not None else updates_red
        if Q is None and p is not None:
            max_sliced_dim = max(sd)
            updates_flat = _reorder_updates(
                self.ordering, updates_flat, pps, max_sliced_dim
            )
        _, unravel = ravel_pytree(params)
        updates = unravel(updates_flat)
        if jnp.isrealobj(grad_factor):
            updates = jax.tree_util.tree_map(
                lambda x, t: (2.0 * x.real).astype(t.dtype)
                if not jnp.iscomplexobj(t)
                else x,
                updates,
                params,
            )
        return tree_cast(updates, params)

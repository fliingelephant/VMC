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
from vmc.qgt.solvers import solve_cg, solve_cholesky, solve_svd

if TYPE_CHECKING:
    from vmc.gauge import GaugeConfig

logger = logging.getLogger(__name__)

__all__ = [
    "LinearSolver",
    "solve_cg",
    "solve_cholesky",
    "solve_svd",
    "DirectSolve",
    "MetricsConfig",
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


@dataclass(frozen=True)
class MetricsConfig:
    """Optional SR metrics."""

    record_FS_norm: bool = False
    record_TDVP_residual: bool = False
    record_SR_solve_residual: bool = False
    record_step_wall_time: bool = False


# --------------------------------------------------------------------------- #
# Solve dispatch
# --------------------------------------------------------------------------- #


@dispatch
def _reorder_updates(
    ordering: SliceOrdering,
    updates_flat: jax.Array,
    pps: tuple[int, ...],
    sliced_dims: tuple[int, ...],
) -> jax.Array:
    """Permute updates from k-major to site-major order.

    SliceOrdering produces the expanded Jacobian with columns ordered as:
        [k=0 all sites] [k=1 all sites] ... [k=max all sites]
    But the parameter tree expects site-major order:
        [site0 all k] [site1 all k] ... [siteN all k]

    This function builds a permutation that extracts entries in site-major order.
    For non-uniform sliced_dims, only valid (site, k) pairs are included.
    """
    total = sum(pps)
    perm = []
    site_offset = 0
    for site_idx, n in enumerate(pps):
        for k in range(sliced_dims[site_idx]):
            base = k * total + site_offset
            perm.extend(range(base, base + n))
        site_offset += n
    return updates_flat[jnp.asarray(perm)]


@dispatch
def _reorder_updates(
    ordering: SiteOrdering,
    updates_flat: jax.Array,
    pps: tuple[int, ...],
    sliced_dims: tuple[int, ...],
) -> jax.Array:
    """SiteOrdering already produces site-major order, no reordering needed."""
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
    metrics_config: MetricsConfig,
) -> tuple[jax.Array, dict]:
    rhs = _adjoint_matvec(jac, dv)
    qgt = QGT(jac, space=ParameterSpace())
    S = qgt.to_dense()
    mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
    x = solver(mat, rhs)
    metrics = {}
    shifted_residual = None
    rhs_norm = None
    if (
        metrics_config.record_TDVP_residual
        or metrics_config.record_SR_solve_residual
    ):
        shifted_residual = mat @ x - rhs
        rhs_norm = jnp.linalg.norm(rhs)
    if metrics_config.record_FS_norm:
        if shifted_residual is None:
            metrics["FS_norm_squared"] = (
                jnp.real(jnp.vdot(x, rhs))
                - diag_shift * jnp.real(jnp.vdot(x, x))
            )
        else:
            metrics["FS_norm_squared"] = jnp.real(
                jnp.vdot(x, rhs - diag_shift * x + shifted_residual)
            )
    if metrics_config.record_TDVP_residual:
        tdvp_residual = jnp.linalg.norm(
            shifted_residual - diag_shift * x
        ) / rhs_norm
        metrics["TDVP_residual"] = tdvp_residual
    if metrics_config.record_SR_solve_residual:
        metrics["SR_solve_residual"] = (
            jnp.linalg.norm(shifted_residual) / rhs_norm
        )
    return x, metrics


@dispatch
def _direct_solve(
    space: SampleSpace,
    jac: Jacobian | SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
    solver: LinearSolver,
    metrics_config: MetricsConfig,
) -> tuple[jax.Array, dict]:
    qgt = QGT(jac, space=SampleSpace())
    S = qgt.to_dense()
    mat = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)
    y = solver(mat, dv)
    x = _adjoint_matvec(jac, y)
    metrics = {}
    shifted_residual = None
    dv_norm = None
    if (
        metrics_config.record_TDVP_residual
        or metrics_config.record_SR_solve_residual
    ):
        shifted_residual = mat @ y - dv
        dv_norm = jnp.linalg.norm(dv)
    if metrics_config.record_FS_norm:
        Gy = (
            dv - diag_shift * y
            if shifted_residual is None
            else dv - diag_shift * y + shifted_residual
        )
        metrics["FS_norm_squared"] = dv.shape[0] * jnp.real(jnp.vdot(Gy, Gy))
    if metrics_config.record_TDVP_residual:
        tdvp_residual = jnp.linalg.norm(
            shifted_residual - diag_shift * y
        ) / dv_norm
        metrics["TDVP_residual"] = tdvp_residual
    if metrics_config.record_SR_solve_residual:
        metrics["SR_solve_residual"] = (
            jnp.linalg.norm(shifted_residual) / dv_norm
        )
    return x, metrics


@dispatch
def _solve_sr(
    strategy: DirectSolve,
    space: ParameterSpace | SampleSpace,
    jac: Jacobian | SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
    metrics_config: MetricsConfig,
) -> tuple[jax.Array, dict]:
    return _direct_solve(
        space,
        jac,
        dv,
        diag_shift,
        strategy.solver,
        metrics_config,
    )


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: ParameterSpace,
    jac: Jacobian,
    dv: jax.Array,
    diag_shift: float,
    metrics_config: MetricsConfig,
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
    metrics = {"rank": r_rank}
    rhs_norm = None
    if (
        metrics_config.record_FS_norm
        or metrics_config.record_TDVP_residual
        or metrics_config.record_SR_solve_residual
    ):
        rhs_norm = jnp.linalg.norm(rhs)
    if metrics_config.record_FS_norm:
        metrics["FS_norm_squared"] = jnp.real(jnp.vdot(x, rhs + resid)) / jac.O.shape[0]
    if metrics_config.record_TDVP_residual:
        tdvp_residual = jnp.linalg.norm(resid) / rhs_norm
        metrics["TDVP_residual"] = tdvp_residual
    if metrics_config.record_SR_solve_residual:
        metrics["SR_solve_residual"] = jnp.linalg.norm(resid) / rhs_norm
    return x, metrics


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: ParameterSpace,
    jac: SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
    metrics_config: MetricsConfig,
) -> tuple[jax.Array, dict]:
    raise NotImplementedError("QRSolve not supported for SlicedJacobian")


@dispatch
def _solve_sr(
    strategy: QRSolve,
    space: SampleSpace,
    jac: Jacobian | SlicedJacobian,
    dv: jax.Array,
    diag_shift: float,
    metrics_config: MetricsConfig,
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
        metrics_config: MetricsConfig = MetricsConfig(),
    ):
        self.space = space
        self.strategy = strategy
        self.diag_shift = diag_shift
        self.gauge_config = gauge_config
        self.ordering = ordering
        self.metrics_config = metrics_config
        self.uses_local_energies = True

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
    ) -> tuple[Any, dict]:
        from vmc.gauge import compute_gauge_projection

        dv = (local_energies - jnp.mean(local_energies)) / samples.shape[0]
        dv = grad_factor * dv

        params = jax.tree_util.tree_map(jnp.asarray, params)
        pps = model.params_per_site if p is not None else None
        sd = model.sliced_dims
        Q = None
        if self.gauge_config is not None:
            if isinstance(self.space, SampleSpace):
                raise NotImplementedError(
                    "Gauge removal is not supported for SampleSpace. "
                    "Use ParameterSpace or set gauge_config=None."
                )
            Q, _ = compute_gauge_projection(
                self.gauge_config, model, params, return_info=True
            )
            if p is None:
                o_eff = o @ Q
            else:
                from vmc.qgt.qgt import _iter_sliced_blocks

                site_order = SiteOrdering(pps)
                blocks = [ok for ok, _ in _iter_sliced_blocks(o, p, sd, site_order)]
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
        updates_red, metrics = _solve_sr(
            strategy,
            self.space,
            jac,
            dv,
            self.diag_shift,
            self.metrics_config,
        )

        updates_flat = Q @ updates_red if Q is not None else updates_red
        if Q is None and p is not None:
            updates_flat = _reorder_updates(
                self.ordering, updates_flat, pps, sd
            )
        _, unravel = ravel_pytree(params)
        updates = unravel(updates_flat)
        if jnp.isrealobj(grad_factor):
            if jnp.issubdtype(model.dtype, jnp.complexfloating):
                updates = jax.tree_util.tree_map(lambda x: 2.0 * x, updates)
            else:
                updates = jax.tree_util.tree_map(
                    lambda x, t: (2.0 * x.real).astype(t.dtype),
                    updates,
                    params,
                )
        return tree_cast(updates, params), metrics

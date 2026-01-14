"""Quantum Geometric Tensor."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from plum import dispatch

from VMC import config  # noqa: F401
from VMC.gauge import WeightConfig, compute_weight_projection
from VMC.preconditioners import solve_cholesky
from VMC.qgt.jacobian import Jacobian, SlicedJacobian, PhysicalOrdering, SiteOrdering

__all__ = ["QGT"]


class QGT:
    """Quantum Geometric Tensor supporting both spaces and Jacobian types."""

    def __init__(self, jac: Jacobian | SlicedJacobian, space: str):
        self.jac = jac
        self.space = space
        self.matrix = self._build_matrix()

    def to_dense(self) -> jax.Array:
        """Expand block-diagonal physical space O†O to full dense matrix."""
        if not (isinstance(self.jac, SlicedJacobian) and self.space == "physical"):
            return self.matrix
        o, p, d = self.jac.o, self.jac.p, self.jac.phys_dim
        n_s, n = o.shape
        col = d * jnp.arange(n)[None, :] + p
        O = jnp.zeros((n_s, d * n), dtype=o.dtype).at[jnp.arange(n_s)[:, None], col].set(o)
        return O.conj().T @ O

    def _build_matrix(self) -> jax.Array:
        if isinstance(self.jac, Jacobian):
            O = self.jac.O
            return O.conj().T @ O if self.space == "physical" else O @ O.conj().T
        jac = self.jac
        pps = _get_pps(jac.ordering, jac.o.shape[1])
        if self.space == "sample":
            return _build_OOdag(jac.o, jac.p, jac.phys_dim, pps)
        return _build_OdagO(jac.o, jac.p, jac.phys_dim, pps)

    def solve(self, rhs: jax.Array, diag_shift: float, samples=None, project_null: bool = True):
        """Solve (QGT + λI) @ x = rhs."""
        if project_null and self.space == "sample":
            if samples is None:
                raise ValueError("samples required for null projection")
            null = compute_weight_projection(WeightConfig(), samples)
            q = _nullspace_from_vector(null)
            T, rhs_proj = q.conj().T @ self.matrix @ q, q.conj().T @ (rhs - jnp.mean(rhs))
        else:
            q, T, rhs_proj = None, self.matrix, rhs

        x = solve_cholesky(T + diag_shift * jnp.eye(T.shape[0], dtype=T.dtype), rhs_proj)
        if q is not None:
            x = q @ x
        if self.space == "sample" and isinstance(self.jac, SlicedJacobian):
            jac = self.jac
            x = _recover(jac.o, jac.p, x, jac.phys_dim, _get_pps(jac.ordering, jac.o.shape[1]))
        return x, {}


@dispatch
def _get_pps(ordering: PhysicalOrdering, n: int) -> tuple[int, ...]:
    return (n,)

@dispatch
def _get_pps(ordering: SiteOrdering, n: int) -> tuple[int, ...]:
    return ordering.params_per_site


def _nullspace_from_vector(null: jax.Array) -> jax.Array:
    """Build orthonormal basis orthogonal to null vector."""
    null = null[:, None] / jnp.linalg.norm(null)
    q, _ = jnp.linalg.qr(null, mode="complete")
    return q[:, 1:]


@functools.partial(jax.jit, static_argnames=("d", "pps"))
def _build_OOdag(o, p, d: int, pps: tuple[int, ...]) -> jax.Array:
    """OO† = Σ_{s,k} o_sk @ o_sk†."""
    G = jnp.zeros((o.shape[0], o.shape[0]), dtype=o.dtype)
    i = 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i:i+n] == k, o[:, i:i+n], 0)
            G = G + ok @ ok.conj().T
        i += n
    return G


@functools.partial(jax.jit, static_argnames=("d", "pps"))
def _build_OdagO(o, p, d: int, pps: tuple[int, ...]) -> jax.Array:
    """O†O = Σ_{s,k} o_sk† @ o_sk (block diagonal)."""
    m = sum(d * n for n in pps)
    S = jnp.zeros((m, m), dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i:i+n] == k, o[:, i:i+n], 0)
            S = S.at[j:j+n, j:j+n].set(ok.conj().T @ ok)
            j += n
        i += n
    return S


@functools.partial(jax.jit, static_argnames=("d", "pps"))
def _recover(o, p, y, d: int, pps: tuple[int, ...]) -> jax.Array:
    """O† @ y."""
    u = jnp.zeros(sum(d * n for n in pps), dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i:i+n] == k, o[:, i:i+n], 0)
            u = u.at[j:j+n].set(ok.conj().T @ y)
            j += n
        i += n
    return u

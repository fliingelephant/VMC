"""Quantum Geometric Tensor with lazy matvec."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc import config  # noqa: F401
from vmc.qgt.jacobian import (
    Jacobian,
    PhysicalOrdering,
    SiteOrdering,
    SlicedJacobian,
    jacobian_mean,
)

__all__ = ["QGT", "ParameterSpace", "SampleSpace"]


@dataclass(frozen=True)
class ParameterSpace:
    """O†O formulation (n_params x n_params)."""

    pass


@dataclass(frozen=True)
class SampleSpace:
    """OO† formulation (n_samples x n_samples)."""

    pass


@dataclass
class QGT:
    """Lazy quantum geometric tensor supporting full and sliced Jacobians."""

    jac: Jacobian | SlicedJacobian
    space: ParameterSpace | SampleSpace = ParameterSpace()

    def __matmul__(self, v):
        """S @ v without building S explicitly."""
        return _matvec(self.jac, self.space, v)

    def to_dense(self):
        """Build explicit S matrix."""
        return _to_dense(self.jac, self.space)


@dispatch
def _params_per_site(ordering: SiteOrdering, o: jax.Array) -> tuple[int, ...]:
    return ordering.params_per_site


@dispatch
def _params_per_site(ordering: PhysicalOrdering, o: jax.Array) -> tuple[int, ...]:
    return (o.shape[1],)


def _iter_sliced_blocks(
    o: jax.Array,
    p: jax.Array,
    phys_dim: int,
    pps: tuple[int, ...],
):
    i = 0
    for n in pps:
        for k in range(phys_dim):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            yield ok, n
        i += n


def _sliced_forward_matvec(jac: SlicedJacobian, v: jax.Array) -> jax.Array:
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _params_per_site(jac.ordering, o)
    result = jnp.zeros((o.shape[0],), dtype=o.dtype)
    offset = 0
    for ok, n in _iter_sliced_blocks(o, p, d, pps):
        result = result + ok @ v[offset : offset + n]
        offset += n
    return result


def _sliced_adjoint_matvec(jac: SlicedJacobian, v: jax.Array) -> jax.Array:
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _params_per_site(jac.ordering, o)
    parts = [ok.conj().T @ v for ok, _ in _iter_sliced_blocks(o, p, d, pps)]
    return jnp.concatenate(parts, axis=0)


def _sliced_dense_blocks(
    o: jax.Array,
    p: jax.Array,
    phys_dim: int,
    pps: tuple[int, ...],
) -> jax.Array:
    blocks = [ok for ok, _ in _iter_sliced_blocks(o, p, phys_dim, pps)]
    return jnp.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------- #
# Matvec dispatch
# --------------------------------------------------------------------------- #


@dispatch
def _matvec(jac: Jacobian, space: ParameterSpace, v):
    scale = 1.0 / jac.O.shape[0]
    result = (jac.O.conj().T @ (jac.O @ v)) * scale
    mean = jacobian_mean(jac)
    result = result - mean.conj() * jnp.dot(mean, v)
    return result


@dispatch
def _matvec(jac: Jacobian, space: SampleSpace, v):
    scale = 1.0 / jac.O.shape[0]
    result = (jac.O @ (jac.O.conj().T @ v)) * scale
    mean = jacobian_mean(jac)
    ones = jnp.ones((jac.O.shape[0],), dtype=result.dtype)
    v_sum = jnp.sum(v)
    w = jac.O @ mean.conj()
    result = result - w * (v_sum * scale)
    result = result - ones * (jnp.vdot(w, v) * scale)
    result = result + ones * (jnp.vdot(mean, mean) * v_sum * scale)
    return result


@dispatch
def _matvec(jac: SlicedJacobian, space: ParameterSpace, v):
    o = jac.o
    result = _sliced_adjoint_matvec(jac, _sliced_forward_matvec(jac, v))
    scale = 1.0 / o.shape[0]
    result = result * scale
    mean = jacobian_mean(jac)
    result = result - mean.conj() * jnp.dot(mean, v)
    return result


@dispatch
def _matvec(jac: SlicedJacobian, space: SampleSpace, v):
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _params_per_site(jac.ordering, o)
    result = jnp.zeros_like(v, dtype=o.dtype)
    i = 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            result = result + ok @ (ok.conj().T @ v)
        i += n
    scale = 1.0 / o.shape[0]
    result = result * scale
    mean = jacobian_mean(jac)
    ones = jnp.ones((o.shape[0],), dtype=result.dtype)
    v_sum = jnp.sum(v)
    w = _sliced_forward_matvec(jac, mean.conj())
    result = result - w * (v_sum * scale)
    result = result - ones * (jnp.vdot(w, v) * scale)
    result = result + ones * (jnp.vdot(mean, mean) * v_sum * scale)
    return result


# --------------------------------------------------------------------------- #
# to_dense dispatch
# --------------------------------------------------------------------------- #


@dispatch
def _to_dense(jac: Jacobian, space: ParameterSpace):
    scale = 1.0 / jac.O.shape[0]
    S = (jac.O.conj().T @ jac.O) * scale
    mean = jacobian_mean(jac)
    S = S - mean.conj()[:, None] * mean[None, :]
    return S


@dispatch
def _to_dense(jac: Jacobian, space: SampleSpace):
    scale = 1.0 / jac.O.shape[0]
    G = (jac.O @ jac.O.conj().T) * scale
    mean = jacobian_mean(jac)
    ones = jnp.ones((jac.O.shape[0],), dtype=G.dtype)
    w = jac.O @ mean.conj()
    G = G - (w[:, None] * ones[None, :]) * scale
    G = G - (ones[:, None] * w.conj()[None, :]) * scale
    G = G + (jnp.vdot(mean, mean) * scale) * (
        ones[:, None] * ones[None, :]
    )
    return G


@dispatch
def _to_dense(jac: SlicedJacobian, space: ParameterSpace):
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _params_per_site(jac.ordering, o)
    O = _sliced_dense_blocks(o, p, d, pps)
    scale = 1.0 / o.shape[0]
    S = (O.conj().T @ O) * scale
    mean = jacobian_mean(jac)
    S = S - mean.conj()[:, None] * mean[None, :]
    return S


@dispatch
def _to_dense(jac: SlicedJacobian, space: SampleSpace):
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _params_per_site(jac.ordering, o)
    G = jnp.zeros((o.shape[0], o.shape[0]), dtype=o.dtype)
    i = 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            G = G + ok @ ok.conj().T
        i += n
    scale = 1.0 / o.shape[0]
    G = G * scale
    mean = jacobian_mean(jac)
    ones = jnp.ones((o.shape[0],), dtype=G.dtype)
    w = _sliced_forward_matvec(jac, mean.conj())
    G = G - (w[:, None] * ones[None, :]) * scale
    G = G - (ones[:, None] * w.conj()[None, :]) * scale
    G = G + (jnp.vdot(mean, mean) * scale) * (
        ones[:, None] * ones[None, :]
    )
    return G

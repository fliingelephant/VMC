"""Quantum Geometric Tensor with lazy matvec."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from plum import dispatch

from VMC import config  # noqa: F401
from VMC.qgt.jacobian import (
    Jacobian,
    PhysicalOrdering,
    SiteOrdering,
    SlicedJacobian,
    jacobian_mean,
)

__all__ = ["QGT", "DiagonalQGT", "ParameterSpace", "SampleSpace"]


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


@dataclass
class DiagonalQGT:
    """Block-diagonal approximation of the QGT in parameter space."""

    jac: Jacobian | SlicedJacobian
    space: ParameterSpace | SampleSpace = ParameterSpace()
    params_per_site: tuple[int, ...] | None = None

    def __matmul__(self, v):
        return _diag_space_matvec(self.space, self.jac, v, self.params_per_site)

    def to_dense(self):
        return _diag_space_to_dense(self.space, self.jac, self.params_per_site)


@dispatch
def _diag_space_matvec(
    space: ParameterSpace,
    jac: Jacobian | SlicedJacobian,
    v: jax.Array,
    params_per_site: tuple[int, ...] | None,
) -> jax.Array:
    _ = space
    return _diag_matvec(jac, v, params_per_site)


@dispatch
def _diag_space_matvec(
    space: SampleSpace,
    jac: Jacobian | SlicedJacobian,
    v: jax.Array,
    params_per_site: tuple[int, ...] | None,
) -> jax.Array:
    _ = (space, jac, v, params_per_site)
    raise NotImplementedError("DiagonalQGT only supports ParameterSpace")


@dispatch
def _diag_space_to_dense(
    space: ParameterSpace,
    jac: Jacobian | SlicedJacobian,
    params_per_site: tuple[int, ...] | None,
) -> jax.Array:
    _ = space
    return _diag_to_dense(jac, params_per_site)


@dispatch
def _diag_space_to_dense(
    space: SampleSpace,
    jac: Jacobian | SlicedJacobian,
    params_per_site: tuple[int, ...] | None,
) -> jax.Array:
    _ = (space, jac, params_per_site)
    raise NotImplementedError("DiagonalQGT only supports ParameterSpace")


@dispatch
def _params_per_site(ordering: SiteOrdering, o: jax.Array) -> tuple[int, ...]:
    return ordering.params_per_site


@dispatch
def _params_per_site(ordering: PhysicalOrdering, o: jax.Array) -> tuple[int, ...]:
    return (o.shape[1],)


@dispatch
def _diag_params_per_site(
    ordering: SiteOrdering, params_per_site: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    return ordering.params_per_site


@dispatch
def _diag_params_per_site(
    ordering: PhysicalOrdering, params_per_site: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    return params_per_site


@dispatch
def _diag_matvec(
    jac: Jacobian, v: jax.Array, params_per_site: tuple[int, ...] | None
) -> jax.Array:
    mean = jacobian_mean(jac)
    result = jnp.zeros_like(v, dtype=jac.O.dtype)
    i = 0
    for n in params_per_site:
        o_site = jac.O[:, i : i + n]
        block = (o_site.conj().T @ (o_site @ v[i : i + n])) / jac.O.shape[0]
        mean_block = mean[i : i + n]
        block = block - mean_block.conj() * jnp.dot(mean_block, v[i : i + n])
        result = result.at[i : i + n].set(block)
        i += n
    return result


@dispatch
def _diag_matvec(
    jac: SlicedJacobian, v: jax.Array, params_per_site: tuple[int, ...] | None
) -> jax.Array:
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _diag_params_per_site(jac.ordering, params_per_site)
    mean = jacobian_mean(jac)
    return _diag_sliced_matvec(jac.ordering, o, p, v, pps, d, mean)


@dispatch
def _diag_to_dense(
    jac: Jacobian, params_per_site: tuple[int, ...] | None
) -> jax.Array:
    mean = jacobian_mean(jac)
    m = jac.O.shape[1]
    S = jnp.zeros((m, m), dtype=jac.O.dtype)
    i = 0
    for n in params_per_site:
        o_site = jac.O[:, i : i + n]
        block = (o_site.conj().T @ o_site) / jac.O.shape[0]
        mean_block = mean[i : i + n]
        block = block - mean_block.conj()[:, None] * mean_block[None, :]
        S = S.at[i : i + n, i : i + n].set(block)
        i += n
    return S


@dispatch
def _diag_to_dense(
    jac: SlicedJacobian, params_per_site: tuple[int, ...] | None
) -> jax.Array:
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _diag_params_per_site(jac.ordering, params_per_site)
    m = sum(d * n for n in pps)
    mean = jacobian_mean(jac)
    return _diag_sliced_to_dense(jac.ordering, o, p, pps, d, m, mean)


@dispatch
def _diag_sliced_matvec(
    ordering: SiteOrdering,
    o: jax.Array,
    p: jax.Array,
    v: jax.Array,
    pps: tuple[int, ...],
    phys_dim: int,
    mean: jax.Array,
) -> jax.Array:
    result = jnp.zeros_like(v, dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(phys_dim):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            block = (ok.conj().T @ (ok @ v[j : j + n])) / o.shape[0]
            mean_block = mean[j : j + n]
            block = block - mean_block.conj() * jnp.dot(mean_block, v[j : j + n])
            result = result.at[j : j + n].add(block)
            j += n
        i += n
    return result


@dispatch
def _diag_sliced_matvec(
    ordering: PhysicalOrdering,
    o: jax.Array,
    p: jax.Array,
    v: jax.Array,
    pps: tuple[int, ...],
    phys_dim: int,
    mean: jax.Array,
) -> jax.Array:
    total = sum(pps)
    result = jnp.zeros_like(v, dtype=o.dtype)
    site_offset = 0
    for n in pps:
        for k in range(phys_dim):
            ok = jnp.where(
                p[:, site_offset : site_offset + n] == k,
                o[:, site_offset : site_offset + n],
                0,
            )
            base = k * total + site_offset
            block = (ok.conj().T @ (ok @ v[base : base + n])) / o.shape[0]
            mean_block = mean[base : base + n]
            block = block - mean_block.conj() * jnp.dot(mean_block, v[base : base + n])
            result = result.at[base : base + n].add(block)
        site_offset += n
    return result


@dispatch
def _diag_sliced_to_dense(
    ordering: SiteOrdering,
    o: jax.Array,
    p: jax.Array,
    pps: tuple[int, ...],
    phys_dim: int,
    size: int,
    mean: jax.Array,
) -> jax.Array:
    S = jnp.zeros((size, size), dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(phys_dim):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            block = (ok.conj().T @ ok) / o.shape[0]
            mean_block = mean[j : j + n]
            block = block - mean_block.conj()[:, None] * mean_block[None, :]
            S = S.at[j : j + n, j : j + n].set(block)
            j += n
        i += n
    return S


@dispatch
def _diag_sliced_to_dense(
    ordering: PhysicalOrdering,
    o: jax.Array,
    p: jax.Array,
    pps: tuple[int, ...],
    phys_dim: int,
    size: int,
    mean: jax.Array,
) -> jax.Array:
    S = jnp.zeros((size, size), dtype=o.dtype)
    total = sum(pps)
    site_offset = 0
    for n in pps:
        for k in range(phys_dim):
            ok = jnp.where(
                p[:, site_offset : site_offset + n] == k,
                o[:, site_offset : site_offset + n],
                0,
            )
            block = (ok.conj().T @ ok) / o.shape[0]
            base = k * total + site_offset
            mean_block = mean[base : base + n]
            block = block - mean_block.conj()[:, None] * mean_block[None, :]
            S = S.at[base : base + n, base : base + n].set(block)
        site_offset += n
    return S


def _sliced_forward_matvec(jac: SlicedJacobian, v: jax.Array) -> jax.Array:
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _params_per_site(jac.ordering, o)
    result = jnp.zeros((o.shape[0],), dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            result = result + ok @ v[j : j + n]
            j += n
        i += n
    return result


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
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = _params_per_site(jac.ordering, o)
    result = jnp.zeros_like(v, dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            result = result.at[j : j + n].add(ok.conj().T @ (ok @ v[j : j + n]))
            j += n
        i += n
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
    m = sum(d * n for n in pps)
    S = jnp.zeros((m, m), dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            S = S.at[j : j + n, j : j + n].set(ok.conj().T @ ok)
            j += n
        i += n
    scale = 1.0 / o.shape[0]
    S = S * scale
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

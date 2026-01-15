"""Quantum Geometric Tensor with lazy matvec."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from plum import dispatch

from VMC import config  # noqa: F401
from VMC.qgt.jacobian import Jacobian, SlicedJacobian, SiteOrdering

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


# --------------------------------------------------------------------------- #
# Matvec dispatch
# --------------------------------------------------------------------------- #


@dispatch
def _matvec(jac: Jacobian, space: ParameterSpace, v):
    return jac.O.conj().T @ (jac.O @ v)


@dispatch
def _matvec(jac: Jacobian, space: SampleSpace, v):
    return jac.O @ (jac.O.conj().T @ v)


@dispatch
def _matvec(jac: SlicedJacobian, space: ParameterSpace, v):
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = jac.ordering.params_per_site if isinstance(jac.ordering, SiteOrdering) else (o.shape[1],)
    result = jnp.zeros_like(v)
    i, j = 0, 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            result = result.at[j : j + n].add(ok.conj().T @ (ok @ v[j : j + n]))
            j += n
        i += n
    return result


@dispatch
def _matvec(jac: SlicedJacobian, space: SampleSpace, v):
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = jac.ordering.params_per_site if isinstance(jac.ordering, SiteOrdering) else (o.shape[1],)
    result = jnp.zeros_like(v)
    i = 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            result = result + ok @ (ok.conj().T @ v)
        i += n
    return result


# --------------------------------------------------------------------------- #
# to_dense dispatch
# --------------------------------------------------------------------------- #


@dispatch
def _to_dense(jac: Jacobian, space: ParameterSpace):
    return jac.O.conj().T @ jac.O


@dispatch
def _to_dense(jac: Jacobian, space: SampleSpace):
    return jac.O @ jac.O.conj().T


@dispatch
def _to_dense(jac: SlicedJacobian, space: ParameterSpace):
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = jac.ordering.params_per_site if isinstance(jac.ordering, SiteOrdering) else (o.shape[1],)
    m = sum(d * n for n in pps)
    S = jnp.zeros((m, m), dtype=o.dtype)
    i, j = 0, 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            S = S.at[j : j + n, j : j + n].set(ok.conj().T @ ok)
            j += n
        i += n
    return S


@dispatch
def _to_dense(jac: SlicedJacobian, space: SampleSpace):
    o, p, d = jac.o, jac.p, jac.phys_dim
    pps = jac.ordering.params_per_site if isinstance(jac.ordering, SiteOrdering) else (o.shape[1],)
    G = jnp.zeros((o.shape[0], o.shape[0]), dtype=o.dtype)
    i = 0
    for n in pps:
        for k in range(d):
            ok = jnp.where(p[:, i : i + n] == k, o[:, i : i + n], 0)
            G = G + ok @ ok.conj().T
        i += n
    return G

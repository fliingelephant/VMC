"""Linear solvers for QGT systems."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.sparse.linalg as jspsl

from VMC import config  # noqa: F401

__all__ = ["solve_cg", "solve_cholesky", "solve_svd"]


@functools.partial(jax.jit, static_argnames=("maxiter", "tol"))
def solve_cg(
    mat: jax.Array, rhs: jax.Array, *, maxiter: int = 1000, tol: float = 1e-5
) -> jax.Array:
    """Solve using conjugate gradient."""
    sol, _ = jspsl.cg(mat, rhs, maxiter=maxiter, tol=tol)
    return sol


@jax.jit
def solve_cholesky(mat: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve positive-definite system using Cholesky."""
    return jsp.linalg.cho_solve(jsp.linalg.cho_factor(mat), rhs)


@functools.partial(jax.jit, static_argnames=("rcond",))
def solve_svd(mat: jax.Array, rhs: jax.Array, *, rcond: float = 1e-12) -> jax.Array:
    """Solve using SVD with regularization."""
    U, S, Vh = jsp.linalg.svd(mat, full_matrices=False)
    cutoff = rcond * jnp.max(S)
    S_inv = jnp.where(S > cutoff, 1.0 / S, 0.0)
    return Vh.conj().T @ (S_inv * (U.conj().T @ rhs))

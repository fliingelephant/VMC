from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

__all__ = ["_qr_compactwy", "_qr_cholesky"]


def _qr_compactwy(a: jax.Array):
    r, tau = jnp.linalg.qr(a, mode="raw") # batchable geqrf by CuSOLVER
    q = _householder_wy(r.mT, tau)
    return q, jnp.triu(r.mT[: tau.shape[0]])

def _householder_wy(r: jax.Array, tau: jax.Array):
    """Build Q from Householder vectors using the compact WY representation.
    
    Q = I - Y @ T @ Y^H
    """
    m = r.shape[0]
    n = tau.shape[0]
    dtype = r.dtype
    
    Y = jnp.tril(r[:, :n], k=-1) + jnp.eye(m, n, dtype=dtype)
    YHY = Y.conj().T @ Y
    
    col_idx = jnp.arange(n)
    row_idx = jnp.arange(n)
    
    def build_T_col(T, j):
        # Only the strictly lower-triangular part contributes
        yhy_col = jnp.where(col_idx < j, YHY[:, j], 0.0)
        t_yhy = T @ yhy_col
        new_col = jnp.where(row_idx < j, -tau[j] * t_yhy, 0.0)
        new_col = new_col.at[j].set(tau[j])
        return T.at[:, j].set(new_col), None
    
    T, _ = lax.scan(build_T_col, jnp.zeros((n, n), dtype=dtype), jnp.arange(n))
    
    # Q = I - Y @ T @ Y[:n]^H
    return jnp.eye(m, n, dtype=dtype) - jnp.einsum('ik,kl,jl->ij', Y, T, Y[:n, :].conj(), optimize=True)


def _qr_cholesky(a: jax.Array):
    gram = a.conj().T @ a
    L = jnp.linalg.cholesky(gram)
    q = jax.scipy.linalg.solve_triangular(L, a.conj().T, lower=True).conj().T
    return q, L.conj().T

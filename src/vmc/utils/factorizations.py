from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["_qr_compactwy", "_qr_cholesky"]


def _qr_compactwy(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Householder QR via compact WY representation.

    Computes reduced ``(Q, R)`` on trailing matrix axes, batch-polymorphically.
    """
    r, tau = jnp.linalg.qr(a, mode="raw")  # batchable geqrf by CuSOLVER
    q = _householder_wy(r.mT, tau)
    return q, jnp.triu(r.mT[..., : tau.shape[-1], :])


def _householder_wy(r: jax.Array, tau: jax.Array) -> jax.Array:
    """Build reduced ``Q`` from geqrf reflectors in compact WY form.

    Implements ``Q = I - Y T Y^H`` on trailing matrix axes, batch-polymorphically.
    """
    m = r.shape[-2]
    k = tau.shape[-1]
    dtype = r.dtype

    Y = jnp.tril(r[..., :, :k], k=-1) + jnp.eye(m, k, dtype=dtype)
    YHY = jnp.einsum("...ki,...kj->...ij", Y.conj(), Y, optimize=True)
    strict_lower = jnp.tril(jnp.ones((k, k), dtype=dtype), k=-1)
    basis = jnp.eye(k, dtype=dtype)

    def update_column(j: int, T: jax.Array) -> jax.Array:
        mask = strict_lower[:, j]
        yhy_col = YHY[..., :, j] * mask
        t_yhy = jnp.einsum("...ab,...b->...a", T, yhy_col, optimize=True)
        tau_j = tau[..., j][..., None]
        new_col = -tau_j * t_yhy * mask + tau_j * basis[j]
        return jax.lax.dynamic_update_slice_in_dim(T, new_col[..., None], j, axis=-1)

    T = jax.lax.fori_loop(
        0,
        k,
        update_column,
        jnp.zeros(tau.shape[:-1] + (k, k), dtype=dtype),
    )

    return jnp.eye(m, k, dtype=dtype) - jnp.einsum(
        "...ik,...kl,...jl->...ij",
        Y,
        T,
        Y[..., :k, :].conj(),
        optimize=True,
    )


def _qr_cholesky(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    gram = a.mH @ a
    L = jnp.linalg.cholesky(gram)
    q = jax.scipy.linalg.solve_triangular(L, a.mH, lower=True).mH
    return q, L.mH

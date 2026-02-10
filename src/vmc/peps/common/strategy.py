"""PEPS boundary-boundary state contraction strategies."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import abc

import jax
import jax.numpy as jnp

from vmc.utils.factorizations import _qr_compactwy

__all__ = [
    "ContractionStrategy",
    "NoTruncation",
    "ZipUp",
    "DensityMatrix",
    "Variational",
    "_apply_mpo_exact",
    "_apply_mpo_zip_up",
    "_apply_mpo_density_matrix",
    "_apply_mpo_variational",
]

class ContractionStrategy(abc.ABC):
    """Abstract base class for MPO-to-boundary state contraction strategies."""

    def __init__(self, truncate_bond_dimension: int):
        self.truncate_bond_dimension = truncate_bond_dimension

    @abc.abstractmethod
    def apply(self, mps: tuple, mpo: tuple) -> tuple:
        """Apply MPO to boundary state with this strategy.

        Args:
            mps: Boundary boundary state as tuple of tensors.
            mpo: Row MPO as tuple of tensors.

        Returns:
            New boundary state tuple after applying the MPO.
        """


class NoTruncation(ContractionStrategy):
    """No truncation strategy - exact contraction."""

    def __init__(self):
        super().__init__(truncate_bond_dimension=1)

    def apply(self, mps: tuple, mpo: tuple) -> tuple:
        return _apply_mpo_exact(mps, mpo)


class ZipUp(ContractionStrategy):
    """Zip-up truncation strategy - on-the-fly SVD truncation."""

    def __init__(self, truncate_bond_dimension: int):
        super().__init__(truncate_bond_dimension=truncate_bond_dimension)

    def apply(self, mps: tuple, mpo: tuple) -> tuple:
        return _apply_mpo_zip_up(mps, mpo, self.truncate_bond_dimension)


class DensityMatrix(ContractionStrategy):
    """Density-matrix truncation strategy - TEBD-style truncation."""

    def __init__(self, truncate_bond_dimension: int):
        super().__init__(truncate_bond_dimension=truncate_bond_dimension)

    def apply(self, mps: tuple, mpo: tuple) -> tuple:
        return _apply_mpo_density_matrix(mps, mpo, self.truncate_bond_dimension)


class Variational(ContractionStrategy):
    """Variational boundary state compression - iterative sweep optimization.

    This strategy avoids direct SVD by using alternating least squares (ALS)
    to compress the boundary state-MPO product. Unlike ZipUp which uses batched SVD
    (limited to 32x32 matrices on GPU), this uses tensor contractions and
    QR decomposition which scale better under vmap.

    Reference: Liu et al. 2021, Appendix B - "Boundary-boundary state contraction scheme"
    """

    def __init__(self, truncate_bond_dimension: int, n_sweeps: int = 2):
        super().__init__(truncate_bond_dimension=truncate_bond_dimension)
        self.n_sweeps = n_sweeps

    def apply(self, mps: tuple, mpo: tuple) -> tuple:
        return _apply_mpo_variational(
            mps, mpo, self.truncate_bond_dimension, self.n_sweeps
        )


def _contract_theta(m: jax.Array, w: jax.Array, carry: jax.Array | None) -> tuple:
    """Contract boundary state tensor with MPO tensor and optional carry from previous site.

    Returns (theta, left_dim, phys_dim, Dr, wr) where theta has shape
    (left_dim, phys_dim, Dr, wr).
    """
    if carry is not None:
        theta = jnp.einsum(
            "kdl,dpr,lwpq->kqrw",
            carry,
            m,
            w,
            optimize=[(0, 1), (0, 1)],
        )
        left_dim, phys_dim, Dr, wr = theta.shape
        return theta, left_dim, phys_dim, Dr, wr
    theta = jnp.einsum("dpr,lwpq->dlqrw", m, w)
    Dl, wl, phys_dim, Dr, wr = theta.shape
    theta = theta.reshape(Dl * wl, phys_dim, Dr, wr)
    left_dim = Dl * wl
    return theta, left_dim, phys_dim, Dr, wr


def _apply_mpo_exact(mps: tuple, mpo: tuple) -> tuple:
    """Exact MPO application without truncation."""
    return tuple(
        (lambda c: c.reshape(c.shape[0] * c.shape[1], c.shape[2], c.shape[3] * c.shape[4]))(
            jnp.transpose(jnp.tensordot(m, w, axes=[[1], [2]]), (0, 2, 4, 1, 3))
        )
        for m, w in zip(mps, mpo)
    )


def _apply_mpo_zip_up(
    mps: tuple, mpo: tuple, truncate_bond_dimension: int
) -> tuple:
    """Apply MPO with on-the-fly SVD truncation (zip-up)."""
    new = []
    carry = None

    for i, (m, w) in enumerate(zip(mps, mpo)):
        theta, left_dim, phys_dim, Dr, wr = _contract_theta(m, w, carry)

        if i == len(mps) - 1:
            new.append(theta.reshape(left_dim, phys_dim, Dr * wr))
            break

        mat = theta.reshape(left_dim * phys_dim, Dr * wr)
        U, S, Vh = jnp.linalg.svd(mat, full_matrices=False)
        k = min(truncate_bond_dimension, S.shape[0])

        new.append(U[:, :k].reshape(left_dim, phys_dim, k))
        S_c = S[:k].astype(Vh.dtype)
        carry = (S_c[:, None] * Vh[:k, :]).reshape(k, Dr, wr)

    return tuple(new)


def _apply_mpo_density_matrix(
    mps: tuple, mpo: tuple, truncate_bond_dimension: int
) -> tuple:
    """Density-matrix truncation while applying an MPO to an boundary state."""
    new = []
    carry = None

    for i, (m, w) in enumerate(zip(mps, mpo)):
        theta, left_dim, phys_dim, Dr, wr = _contract_theta(m, w, carry)
        theta = theta.reshape(left_dim * phys_dim, Dr * wr)

        if i == len(mps) - 1:
            new.append(theta.reshape(left_dim, phys_dim, Dr * wr))
            break

        rho = theta.conj().T @ theta
        evals, evecs = jnp.linalg.eigh(rho)
        k = min(truncate_bond_dimension, rho.shape[0])
        vecs_k = evecs[:, jnp.argsort(evals)[::-1][:k]]

        new.append((theta @ vecs_k).reshape(left_dim, phys_dim, k))
        carry = vecs_k.conj().T.reshape(k, Dr, wr)

    return tuple(new)


def _apply_mpo_variational(
    mps: tuple, mpo: tuple, truncate_bond_dimension: int, n_sweeps: int = 2
) -> tuple:
    """Apply MPO with variational boundary state compression (iterative sweeping).

    Implements variational boundary state compression from Schollwöck (arXiv:1008.3477, Sec 4.5.2)
    and Paeckel et al. (arXiv:1901.05824, Sec 2.6.2, 2.8.2).

    Algorithm:
    1. Initialize compressed boundary state with target bond dimension Dc via QR sweep
       without materializing the full MPO-boundary state product.
    2. Iterative sweeps: at each site compute optimal tensor M'[i] = L̃ @ M[i] @ R̃
       where L̃, R̃ are overlap environments between result and the implicit target
       defined by the MPO-boundary state product.
    3. QR maintains canonical form (no truncation needed - bond dim fixed by init)

    This avoids SVD entirely, using only QR for canonical form maintenance.
    """
    dtype = mps[0].dtype
    Dc = truncate_bond_dimension

    # Step 1: Initialize compressed boundary state with target bond dimension
    # Use QR-based initialization without materializing the full MPO-boundary state product.
    result = _init_compressed_mps(mps, mpo, Dc)

    # Step 2: Iterative refinement via variational sweeping
    # Initial right environments for the first left-to-right sweep.
    right_envs = _build_right_envs(mps, mpo, result, dtype)
    for _ in range(n_sweeps):
        # Left-to-right sweep with proper overlap environments.
        # This also builds left environments for the updated result.
        result, left_envs = _variational_sweep_lr(
            mps, mpo, result, right_envs, dtype
        )
        # Right-to-left sweep with proper overlap environments.
        # This also builds right environments for the updated result.
        result, right_envs = _variational_sweep_rl(
            mps, mpo, result, left_envs, dtype
        )

    return tuple(result)


def _init_compressed_mps(mps: tuple, mpo: tuple, Dc: int) -> list[jax.Array]:
    """Initialize compressed boundary state with target bond dimension via QR.

    Creates a left-canonical boundary state by sweeping left-to-right with QR on the
    implicit MPO-boundary state product, without materializing the full product.
    Bond dimensions are capped at Dc.
    """
    n_sites = len(mps)
    result = []
    carry = None

    for i, (m, w) in enumerate(zip(mps, mpo)):
        theta, left_dim, phys_dim, Dr, wr = _contract_theta(m, w, carry)

        if i == n_sites - 1:
            result.append(theta.reshape(left_dim, phys_dim, Dr * wr))
            break

        # Reshape for QR: (left * phys, right)
        mat = theta.reshape(left_dim * phys_dim, Dr * wr)

        # QR decomposition
        Q, R = _qr_compactwy(mat)

        # Truncate to Dc columns
        k = min(Dc, Q.shape[1])
        Q_trunc = Q[:, :k]
        R_trunc = R[:k, :]

        result.append(Q_trunc.reshape(left_dim, phys_dim, k))
        carry = R_trunc.reshape(k, Dr, wr)

    return result


def _build_right_envs(
    mps: tuple, mpo: tuple, result: list, dtype: jnp.dtype
) -> list[jax.Array]:
    """Build all right overlap environments R̃[i] from right to left.

    R̃[i] contracts sites i+1: with the implicit MPO-boundary state product and result.
    Shape: R̃[i][mps_right, mpo_right, result_right]
    """
    n_sites = len(mps)
    right_envs = [None] * n_sites

    # R̃[n-1] = identity (no sites to the right of last site)
    R = jnp.ones((1, 1, 1), dtype=dtype)
    right_envs[n_sites - 1] = R

    for i in range(n_sites - 2, -1, -1):
        m = mps[i + 1]  # (Dl, pin, Dr)
        w = mpo[i + 1]  # (wl, wr, pin, pout)
        r = result[i + 1]  # (rl, pout, rr)

        # R̃_new[Dl, wl, rl] = sum_{Dr, wr, rr, pin, pout}
        #   m[Dl, pin, Dr] * w[wl, wr, pin, pout] * r*[rl, pout, rr] * R̃[Dr, wr, rr]
        R = jnp.einsum(
            "aqb,rwb,lwpq,dpr->dla",
            r.conj(),
            R,
            w,
            m,
            optimize=[(1, 0), (2, 0), (1, 0)],
        )
        right_envs[i] = R

    return right_envs


def _variational_sweep_lr(
    mps: tuple, mpo: tuple, result: list, right_envs: list, dtype: jnp.dtype
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Left-to-right variational sweep (one-site, QR only).

    At each site j:
    1. Compute optimal tensor: M'[j] = L̃ @ M[j] @ R̃[j]
    2. QR decompose: M'[j] = A[j] @ C[j]
    3. Absorb C[j] into result[j+1] for next iteration (Paeckel Sec 2.5)

    Reference: Paeckel et al. arXiv:1901.05824, Sec 2.5-2.6.2, Alg 5
    """
    n_sites = len(mps)
    new_result = list(result)  # Copy to allow absorbing C
    left_envs = [None] * n_sites

    # Left environment starts as identity
    L = jnp.ones((1, 1, 1), dtype=dtype)

    for i in range(n_sites):
        left_envs[i] = L
        m = mps[i]  # (Dl, pin, Dr)
        w = mpo[i]  # (wl, wr, pin, pout)
        R = right_envs[i]  # (Dr, wr, rr)

        # Compute optimal tensor from implicit MPO-boundary state product.
        # M'[rl, pout, rr] = sum L[Dl, wl, rl] * m[Dl, pin, Dr]
        #                     * w[wl, wr, pin, pout] * R[Dr, wr, rr]
        optimal = jnp.einsum(
            "dpr,dla,lwpq,rwb->aqb",
            m,
            L,
            w,
            R,
            optimize=[(1, 0), (2, 0), (1, 0)],
        )

        if i == n_sites - 1:
            new_result[i] = optimal
            break

        # QR decompose: optimal = A @ C (Paeckel Sec 2.5)
        left_dim, phys_dim, right_dim = optimal.shape
        mat = optimal.reshape(left_dim * phys_dim, right_dim)
        A, C = _qr_compactwy(mat)

        new_tensor = A.reshape(left_dim, phys_dim, A.shape[1])
        new_result[i] = new_tensor

        # Absorb C into next tensor: result[j+1] ← C @ result[j+1] (Paeckel Sec 2.5)
        # This maintains state equivalence during the sweep
        next_tensor = new_result[i + 1]  # (rl_next, p_next, rr_next)
        new_result[i + 1] = jnp.einsum("ab,bpr->apr", C, next_tensor)

        # Update left environment for next site using the new tensor
        L = jnp.einsum(
            "dpr,dla,lwpq,aqb->rwb",
            m,
            L,
            w,
            new_tensor.conj(),
            optimize=[(1, 0), (2, 0), (1, 0)],
        )

    return new_result, left_envs


def _variational_sweep_rl(
    mps: tuple, mpo: tuple, result: list, left_envs: list, dtype: jnp.dtype
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Right-to-left variational sweep (one-site, QR only).

    At each site j:
    1. Compute optimal tensor: M'[j] = L̃[j] @ M[j] @ R̃
    2. LQ decompose: M'[j] = C[j-1] @ B[j]
    3. Absorb C[j-1] into result[j-1] for next iteration (Paeckel Sec 2.5)

    Reference: Paeckel et al. arXiv:1901.05824, Sec 2.5-2.6.2, Alg 5
    """
    n_sites = len(mps)
    new_result = list(result)  # Copy to allow absorbing C
    right_envs = [None] * n_sites

    # Right environment starts as identity
    R = jnp.ones((1, 1, 1), dtype=dtype)

    for i in range(n_sites - 1, -1, -1):
        right_envs[i] = R
        m = mps[i]  # (Dl, pin, Dr)
        w = mpo[i]  # (wl, wr, pin, pout)
        L = left_envs[i]  # (Dl, wl, rl)

        # Compute optimal tensor from implicit MPO-boundary state product.
        # M'[rl, pout, rr] = sum L[Dl, wl, rl] * m[Dl, pin, Dr]
        #                     * w[wl, wr, pin, pout] * R[Dr, wr, rr]
        optimal = jnp.einsum(
            "dpr,dla,lwpq,rwb->aqb",
            m,
            L,
            w,
            R,
            optimize=[(1, 0), (2, 0), (1, 0)],
        )

        if i == 0:
            new_result[i] = optimal
            break

        # LQ decompose: optimal = C @ B (Paeckel Sec 2.5)
        # LQ via QR on transpose: A = C @ B, A.T = B.T @ C.T
        left_dim, phys_dim, right_dim = optimal.shape
        mat = optimal.reshape(left_dim, phys_dim * right_dim)
        Q_t, R_t = _qr_compactwy(mat.T)
        # B = Q_t.T has orthonormal rows, C = R_t.T
        B = Q_t.T  # (k, phys * right)
        C = R_t.T  # (left, k)

        new_tensor = B.reshape(B.shape[0], phys_dim, right_dim)
        new_result[i] = new_tensor

        # Absorb C into previous tensor: result[j-1] ← result[j-1] @ C (Paeckel Sec 2.5)
        # This maintains state equivalence during the sweep
        prev_tensor = new_result[i - 1]  # (rl_prev, p_prev, rr_prev)
        new_result[i - 1] = jnp.einsum("lpr,rb->lpb", prev_tensor, C)

        # Update right environment for next site using the new tensor
        R = jnp.einsum(
            "aqb,rwb,lwpq,dpr->dla",
            new_tensor.conj(),
            R,
            w,
            m,
            optimize=[(1, 0), (2, 0), (1, 0)],
        )

    return new_result, right_envs

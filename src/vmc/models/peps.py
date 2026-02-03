"""PEPS (Projected Entangled Pair States) implementation.

This module provides a PEPS implementation using boundary MPS contraction
with configurable truncation strategies via ABC pattern.
"""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import abc
import functools
import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import nnx

from plum import dispatch

from vmc.utils.utils import random_tensor, spin_to_occupancy

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = [
    "PEPS",
    # Contraction strategies (ABC pattern)
    "ContractionStrategy",
    "NoTruncation",
    "ZipUp",
    "DensityMatrix",
    "Variational",
    # Dispatched API functions
    "bottom_envs",
    "grads_and_energy",
    "sweep",
    "_metropolis_ratio",
    # Internal functions exposed for external use
    "_apply_mpo_from_below",
    "_build_row_mpo",
    "_compute_all_gradients",
    "_compute_all_row_gradients",
    "_compute_all_env_grads_and_energy",
    "_compute_right_envs",
    "_compute_right_envs_2row",
    "_compute_2site_horizontal_env",
    "_compute_single_gradient",
    "_contract_bottom",
    "_compute_row_pair_vertical_energy",
    "_forward_with_cache",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Contraction Strategy Classes (ABC pattern)
# =============================================================================


class ContractionStrategy(abc.ABC):
    """Abstract base class for MPO-to-MPS contraction strategies."""

    def __init__(self, truncate_bond_dimension: int):
        self.truncate_bond_dimension = truncate_bond_dimension

    @abc.abstractmethod
    def apply(self, mps: tuple, mpo: tuple) -> tuple:
        """Apply MPO to MPS with this strategy.

        Args:
            mps: Boundary MPS as tuple of tensors.
            mpo: Row MPO as tuple of tensors.

        Returns:
            New MPS tuple after applying the MPO.
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
    """Variational MPS compression - iterative sweep optimization.

    This strategy avoids direct SVD by using alternating least squares (ALS)
    to compress the MPS-MPO product. Unlike ZipUp which uses batched SVD
    (limited to 32x32 matrices on GPU), this uses tensor contractions and
    QR decomposition which scale better under vmap.

    Reference: Liu et al. 2021, Appendix B - "Boundary-MPS contraction scheme"
    """

    def __init__(self, truncate_bond_dimension: int, n_sweeps: int = 2):
        super().__init__(truncate_bond_dimension=truncate_bond_dimension)
        self.n_sweeps = n_sweeps

    def apply(self, mps: tuple, mpo: tuple) -> tuple:
        return _apply_mpo_variational(
            mps, mpo, self.truncate_bond_dimension, self.n_sweeps
        )


def _contract_theta(m: jax.Array, w: jax.Array, carry: jax.Array | None) -> tuple:
    """Contract MPS tensor with MPO tensor and optional carry from previous site.

    Returns (theta, left_dim, phys_dim, Dr, wr) where theta has shape
    (left_dim, phys_dim, Dr, wr).
    """
    if carry is not None:
        tmp = jnp.einsum("kdl,dpr->prkl", carry, m)
        theta = jnp.einsum("prkl,lwpq->kqrw", tmp, w)
        left_dim, phys_dim, Dr, wr = theta.shape
        return theta, left_dim, phys_dim, Dr, wr
    theta = jnp.einsum("dpr,lwpq->dlqrw", m, w)
    Dl, wl, phys_dim, Dr, wr = theta.shape
    theta = theta.reshape(Dl * wl, phys_dim, Dr, wr)
    left_dim = Dl * wl
    return theta, left_dim, phys_dim, Dr, wr


@jax.jit
def _apply_mpo_exact(mps: tuple, mpo: tuple) -> tuple:
    """Exact MPO application without truncation."""
    return tuple(
        (lambda c: c.reshape(c.shape[0] * c.shape[1], c.shape[2], c.shape[3] * c.shape[4]))(
            jnp.transpose(jnp.tensordot(m, w, axes=[[1], [2]]), (0, 2, 4, 1, 3))
        )
        for m, w in zip(mps, mpo)
    )


@functools.partial(jax.jit, static_argnums=(2,))
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


@functools.partial(jax.jit, static_argnums=(2,))
def _apply_mpo_density_matrix(
    mps: tuple, mpo: tuple, truncate_bond_dimension: int
) -> tuple:
    """Density-matrix truncation while applying an MPO to an MPS."""
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


@functools.partial(jax.jit, static_argnums=(2, 3))
def _apply_mpo_variational(
    mps: tuple, mpo: tuple, truncate_bond_dimension: int, n_sweeps: int = 2
) -> tuple:
    """Apply MPO with variational MPS compression (iterative sweeping).

    Implements variational MPS compression from Schollwöck (arXiv:1008.3477, Sec 4.5.2)
    and Paeckel et al. (arXiv:1901.05824, Sec 2.6.2, 2.8.2).

    Algorithm:
    1. Initialize compressed MPS with target bond dimension Dc via QR sweep
       without materializing the full MPO-MPS product.
    2. Iterative sweeps: at each site compute optimal tensor M'[i] = L̃ @ M[i] @ R̃
       where L̃, R̃ are overlap environments between result and the implicit target
       defined by the MPO-MPS product.
    3. QR maintains canonical form (no truncation needed - bond dim fixed by init)

    This avoids SVD entirely, using only QR for canonical form maintenance.
    """
    n_sites = len(mps)
    dtype = mps[0].dtype
    Dc = truncate_bond_dimension

    # Step 1: Initialize compressed MPS with target bond dimension
    # Use QR-based initialization without materializing the full MPO-MPS product.
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
    """Initialize compressed MPS with target bond dimension via QR.

    Creates a left-canonical MPS by sweeping left-to-right with QR on the
    implicit MPO-MPS product, without materializing the full product.
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
        Q, R = jax.lax.linalg.qr(mat, full_matrices=False)

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

    R̃[i] contracts sites i+1: with the implicit MPO-MPS product and result.
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

        # Compute optimal tensor from implicit MPO-MPS product.
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
        A, C = jax.lax.linalg.qr(mat, full_matrices=False)

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

        # Compute optimal tensor from implicit MPO-MPS product.
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
        Q_t, R_t = jax.lax.linalg.qr(mat.T, full_matrices=False)
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


def _build_row_mpo(tensors, row_indices, row, n_cols):
    """Build row-MPO for PEPS contraction."""
    return tuple(
        jnp.transpose(jnp.asarray(tensors[row][col])[row_indices[col]], (2, 3, 0, 1))
        for col in range(n_cols)
    )


def _contract_bottom(mps):
    """Contract bottom boundary of MPS to get scalar amplitude."""
    state = jnp.array([1.0], dtype=mps[0].dtype)
    for site in mps:
        state = jnp.tensordot(state, site[:, 0, :], axes=[[0], [0]])
    return state.squeeze()


def _forward_with_cache(
    tensors: Any,
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> tuple[jax.Array, list[tuple]]:
    """Forward pass that caches all intermediate boundary MPSs."""
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype

    top_envs = []
    boundary = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    top_envs.append(boundary)

    for row in range(n_rows):
        mpo = _build_row_mpo(tensors, spins[row], row, n_cols)
        boundary = strategy.apply(boundary, mpo)
        top_envs.append(boundary)

    return _contract_bottom(boundary), top_envs


def _apply_mpo_from_below(
    bottom_mps: tuple,
    mpo: tuple,
    strategy: ContractionStrategy,
) -> tuple:
    """Apply MPO to boundary MPS from below (for backward sweep)."""
    return strategy.apply(bottom_mps, tuple(jnp.transpose(w, (0, 1, 3, 2)) for w in mpo))


def _compute_right_envs_2row(
    top_env: tuple,
    mpo_row0: tuple,
    mpo_row1: tuple,
    bottom_env: tuple,
    dtype,
) -> list[jax.Array]:
    """Compute right environments for 2-row contractions using direct einsum."""
    n_cols = len(mpo_row0)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        # Direct einsum: top @ mpo0 @ mpo1 @ bot @ right_env -> new_right_env
        # top: (a, u, b), mpo0: (l, r, u, v), mpo1: (x, y, v, w), bot: (e, w, f)
        # right_env: (b, r, y, f) -> output: (a, l, x, e)
        right_envs[c] = jnp.einsum(
            "aub,lruv,xyvw,ewf,bryf->alxe",
            top_env[c + 1], mpo_row0[c + 1], mpo_row1[c + 1], bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 4), (0, 3), (0, 2), (0, 1)],
        )
    return right_envs


def _compute_row_pair_vertical_energy(
    top_mps: tuple,
    bottom_mps: tuple,
    mpo_row0: tuple,
    mpo_row1: tuple,
    tensors_row0: list[jax.Array],
    tensors_row1: list[jax.Array],
    spins_row0: jax.Array,
    spins_row1: jax.Array,
    terms_row: list[list],
    amp: jax.Array,
    phys_dim: int,
    *,
    right_envs_2row: list[jax.Array] | None = None,
) -> jax.Array:
    """Compute vertical 2-site energy contributions for a row pair."""
    if not any(terms_row):
        return jnp.zeros((), dtype=amp.dtype)
    n_cols = len(mpo_row0)
    dtype = mpo_row0[0].dtype
    if right_envs_2row is None:
        right_envs_2row = _compute_right_envs_2row(
            top_mps, mpo_row0, mpo_row1, bottom_mps, dtype
        )
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
    energy = jnp.zeros((), dtype=amp.dtype)
    for c in range(n_cols):
        col_terms = terms_row[c]
        if col_terms:
            # Direct einsum: left_env @ top @ tensor0 @ tensor1 @ bot @ right_env -> (p, q)
            # tensor0: (p, u, v, l, r), tensor1: (q, v, w, m, n)
            amps_edge = jnp.einsum(
                "almg,aub,puvlr,qvwmn,gwf,brnf->pq",
                left_env, top_mps[c], tensors_row0[c], tensors_row1[c], bottom_mps[c], right_envs_2row[c],
                optimize=[(0, 1), (2, 3), (0, 2), (1, 2), (0, 1)],
            )
            spin0 = spins_row0[c]
            spin1 = spins_row1[c]
            col_idx = spin0 * phys_dim + spin1
            amps_flat = amps_edge.reshape(-1)
            for term in col_terms:
                energy = energy + jnp.dot(term.op[:, col_idx], amps_flat) / amp
        # Direct einsum for left_env_2row update
        left_env = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf->bryf",
            left_env, top_mps[c], mpo_row0[c], mpo_row1[c], bottom_mps[c],
            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
        )
    return energy


def _compute_right_envs(
    top_env: tuple,
    mpo_row: tuple,
    bottom_env: tuple,
    dtype,
) -> list[jax.Array]:
    """Compute right environments using direct einsum (no transfers)."""
    n_cols = len(mpo_row)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        # Direct einsum: top @ mpo @ bot @ right_env -> new_right_env
        # top: (a, u, b), mpo: (c, d, u, v), bot: (e, v, f), right_env: (b, d, f) -> (a, c, e)
        right_envs[c] = jnp.einsum(
            "aub,cduv,evf,bdf->ace",
            top_env[c + 1], mpo_row[c + 1], bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 3), (0, 2), (0, 1)],
        )
    return right_envs


def _compute_all_row_gradients(
    top_mps: tuple,
    bottom_mps: tuple,
    mpo: tuple,
) -> list[jax.Array]:
    """Compute gradients for all tensors in a row using direct einsum."""
    n_cols = len(mpo)
    dtype = mpo[0].dtype
    right_envs = _compute_right_envs(top_mps, mpo, bottom_mps, dtype)

    env_grads = []
    left_env = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols):
        env_grads.append(
            _compute_single_gradient(left_env, right_envs[c], top_mps[c], bottom_mps[c])
        )
        # Direct einsum for left_env update: left_env @ top @ mpo @ bot -> new_left_env
        left_env = jnp.einsum(
            "ace,aub,cduv,evf->bdf",
            left_env, top_mps[c], mpo[c], bottom_mps[c],
            optimize=[(0, 1), (0, 2), (0, 1)],
        )
    return env_grads


def _compute_all_env_grads_and_energy(
    tensors: Any,
    spins: jax.Array,
    amp: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    bottom_envs: list[tuple],
    *,
    diagonal_terms: list,
    one_site_terms: list[list[list]],
    horizontal_terms: list[list[list]],
    vertical_terms: list[list[list]],
    collect_grads: bool = True,
) -> tuple[list[list[jax.Array]], jax.Array]:
    """Compute gradients and local energy for a PEPS sample."""
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype
    phys_dim = int(jnp.asarray(tensors[0][0]).shape[0])

    env_grads = (
        [[None for _ in range(n_cols)] for _ in range(n_rows)]
        if collect_grads
        else []
    )
    energy = jnp.zeros((), dtype=amp.dtype)
    for term in diagonal_terms:
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + spins[row, col]
        energy = energy + term.diag[idx]

    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    mpo = _build_row_mpo(tensors, spins[0], 0, n_cols)
    for row in range(n_rows):
        bottom_env = bottom_envs[row]
        right_envs = _compute_right_envs(top_env, mpo, bottom_env, dtype)
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        for c in range(n_cols):
            site_terms = one_site_terms[row][c]
            need_env_grad = collect_grads or site_terms
            if need_env_grad:
                env_grad = _compute_single_gradient(
                    left_env, right_envs[c], top_env[c], bottom_env[c]
                )
                if collect_grads:
                    env_grads[row][c] = env_grad
                if site_terms:
                    amps_site = jnp.einsum("pudlr,udlr->p", tensors[row][c], env_grad)
                    spin_idx = spins[row, c]
                    for term in site_terms:
                        energy = energy + jnp.dot(term.op[:, spin_idx], amps_site) / amp
            if c < n_cols - 1:
                edge_terms = horizontal_terms[row][c]
                if edge_terms:
                    env_2site = _compute_2site_horizontal_env(
                        left_env,
                        right_envs[c + 1],
                        top_env[c],
                        bottom_env[c],
                        top_env[c + 1],
                        bottom_env[c + 1],
                    )
                    amps_edge = jnp.einsum(
                        "pudlr,qverx,udlvex->pq",
                        tensors[row][c],
                        tensors[row][c + 1],
                        env_2site,
                        optimize=[(0, 2), (0, 1)],
                    )
                    spin0 = spins[row, c]
                    spin1 = spins[row, c + 1]
                    col_idx = spin0 * phys_dim + spin1
                    amps_flat = amps_edge.reshape(-1)
                    for term in edge_terms:
                        energy = energy + jnp.dot(term.op[:, col_idx], amps_flat) / amp
            # Direct einsum for left_env update
            left_env = jnp.einsum(
                "ace,aub,cduv,evf->bdf",
                left_env, top_env[c], mpo[c], bottom_env[c],
                optimize=[(0, 1), (0, 2), (0, 1)],
            )
        if row < n_rows - 1:
            mpo_next = _build_row_mpo(tensors, spins[row + 1], row + 1, n_cols)
            energy = energy + _compute_row_pair_vertical_energy(
                top_env,
                bottom_envs[row + 1],
                mpo,
                mpo_next,
                tensors[row],
                tensors[row + 1],
                spins[row],
                spins[row + 1],
                vertical_terms[row],
                amp,
                phys_dim,
            )
        top_env = strategy.apply(top_env, mpo)
        if row < n_rows - 1:
            mpo = mpo_next

    return env_grads, energy


def _compute_2site_horizontal_env(
    left_env: jax.Array,
    right_env: jax.Array,
    top0: jax.Array,
    bot0: jax.Array,
    top1: jax.Array,
    bot1: jax.Array,
) -> jax.Array:
    """Compute 2-site environment for horizontal edge (c, c+1).

    Index conventions:
        left_env: (tL, mL, bL) - top/mpo/bottom left bonds
        right_env: (tR, mR, bR) - top/mpo/bottom right bonds
        top0/top1: (left, up, right) - boundary MPS
        bot0/bot1: (left, down, right) - boundary MPS

    Returns tensor with shape (up0, down0, mL, up1, down1, mR).
    """
    # Contract left side: left_env (a,c,e) @ top0 (a,u,b) @ bot0 (e,d,f) -> (c,u,b,d,f)
    tmp_left = jnp.einsum("ace,aub,edf->cubdf", left_env, top0, bot0, optimize=[(0, 1), (0, 1)])
    # Contract right side: top1 (b,v,g) @ right_env (g,h,i) @ bot1 (f,w,i) -> (b,v,h,f,w)
    tmp_right = jnp.einsum("bvg,ghi,fwi->bvhfw", top1, right_env, bot1, optimize=[(0, 1), (0, 1)])
    # Contract left and right: (c,u,b,d,f) @ (b,v,h,f,w) -> (c,u,d,v,h,w)
    env = jnp.einsum("cubdf,bvhfw->cudvhw", tmp_left, tmp_right, optimize=[(0, 1)])
    # Transpose to (up0, down0, mL, up1, down1, mR)
    return jnp.transpose(env, (1, 2, 0, 3, 5, 4))


def _compute_single_gradient(
    left_env: jax.Array,
    right_env: jax.Array,
    top_tensor: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Compute gradient for a single tensor given left/right environments.

    Returns gradient tensor with shape (up, down, mL, mR).
    """
    grad = jnp.einsum(
        "ace,aub,evf,bdf->cuvd", left_env, top_tensor, bot_tensor, right_env,
        optimize=[(0, 1), (0, 1), (0, 1)],
    )
    return jnp.transpose(grad, (1, 2, 0, 3))


def _compute_all_gradients(
    tensors: Any,
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    top_envs: list[tuple],
    *,
    cache_bottom_envs: bool = False,
    row_mpos: list[tuple] | None = None,
) -> list[list[jax.Array]] | tuple[list[list[jax.Array]], list[tuple]]:
    """Compute gradients for all PEPS tensors using cached top environments."""
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype

    grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    bottom_envs_cached = [None] * n_rows if cache_bottom_envs else None
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    for row in range(n_rows - 1, -1, -1):
        if cache_bottom_envs:
            bottom_envs_cached[row] = bottom_env
        mpo = row_mpos[row] if row_mpos else _build_row_mpo(tensors, spins[row], row, n_cols)
        grads[row] = _compute_all_row_gradients(top_envs[row], bottom_env, mpo)
        bottom_env = _apply_mpo_from_below(bottom_env, mpo, strategy)

    return (grads, bottom_envs_cached) if cache_bottom_envs else grads


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _peps_apply(
    tensors: Any,
    sample: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> jax.Array:
    """Compute PEPS amplitude with custom VJP for efficient gradients."""
    spins = spin_to_occupancy(sample).reshape(shape)
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(shape[1])
    )
    for row in range(shape[0]):
        mpo = _build_row_mpo(tensors, spins[row], row, shape[1])
        boundary = strategy.apply(boundary, mpo)
    return _contract_bottom(boundary)


def _peps_apply_fwd(tensors, sample, shape, strategy):
    spins = spin_to_occupancy(sample).reshape(shape)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, strategy)
    return amp, (tensors, spins, top_envs)


def _peps_apply_bwd(shape, strategy, residuals, g):
    tensors, spins, top_envs = residuals
    n_rows, n_cols = shape
    env_grads = _compute_all_gradients(tensors, spins, shape, strategy, top_envs)
    grad_leaves = []
    for r in range(n_rows):
        for c in range(n_cols):
            grad_full = jnp.zeros_like(jnp.asarray(tensors[r][c]))
            grad_leaves.append(grad_full.at[spins[r, c]].set(g * env_grads[r][c]))
    return (
        jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tensors), grad_leaves),
        None,
    )


_peps_apply.defvjp(_peps_apply_fwd, _peps_apply_bwd)


# =============================================================================
# Dispatched API Functions (for unified sampler)
# =============================================================================


@dispatch
def bottom_envs(model: "PEPS", sample: jax.Array) -> list[tuple]:
    """Compute bottom boundary environments for PEPS.

    Args:
        model: PEPS model
        sample: flat sample array (occupancy indices), reshaped to (n_rows, n_cols)

    Returns:
        List of bottom environments, one per row.
    """
    indices = sample.reshape(model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    n_rows, n_cols = model.shape
    dtype = tensors[0][0].dtype
    envs = [None] * n_rows
    env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        envs[row] = env
        mpo = _build_row_mpo(tensors, indices[row], row, n_cols)
        env = _apply_mpo_from_below(env, mpo, model.strategy)
    return envs


@dispatch
def grads_and_energy(
    model: "PEPS",
    sample: jax.Array,
    amp: jax.Array,
    operator: Any,
    envs: list[tuple],
) -> tuple[list[list[jax.Array]], jax.Array]:
    """Compute environment gradients and local energy for PEPS.

    Args:
        model: PEPS model
        sample: flat sample array (occupancy indices)
        amp: amplitude for this configuration
        operator: LocalHamiltonian with terms
        envs: bottom environments (computed via bottom_envs())

    Returns:
        (env_grads, energy)
    """
    from vmc.operators.local_terms import bucket_terms

    indices = sample.reshape(model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    diagonal_terms, one_site_terms, horizontal_terms, vertical_terms, _ = bucket_terms(
        operator.terms, model.shape
    )
    return _compute_all_env_grads_and_energy(
        tensors,
        indices,
        amp,
        model.shape,
        model.strategy,
        envs,
        diagonal_terms=diagonal_terms,
        one_site_terms=one_site_terms,
        horizontal_terms=horizontal_terms,
        vertical_terms=vertical_terms,
        collect_grads=True,
    )


def _metropolis_ratio(weight_cur: jax.Array, weight_flip: jax.Array) -> jax.Array:
    """Compute Metropolis acceptance ratio with proper handling of zero weights."""
    return jnp.where(
        weight_cur > 0.0,
        weight_flip / weight_cur,
        jnp.where(weight_flip > 0.0, jnp.inf, 0.0),
    )


@dispatch
def sweep(
    model: "PEPS",
    sample: jax.Array,
    key: jax.Array,
    envs: list[tuple],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single Metropolis sweep for PEPS.

    Args:
        model: PEPS model
        sample: flat sample array (occupancy indices)
        key: PRNG key
        envs: bottom environments from previous iteration

    Returns:
        (new_sample, key, amp) - flat sample array after sweep
    """
    indices = sample.reshape(model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    n_rows, n_cols = model.shape
    dtype = tensors[0][0].dtype

    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows):
        bottom_env = envs[row]
        mpo_row = _build_row_mpo(tensors, indices[row], row, n_cols)
        right_envs = _compute_right_envs(top_env, mpo_row, bottom_env, dtype)
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        updated_row = []
        for col in range(n_cols):
            site_tensor = tensors[row][col]
            phys_dim = int(site_tensor.shape[0])
            cur_idx = indices[row, col]
            if phys_dim == 1:
                flip_idx = cur_idx
            elif phys_dim == 2:
                flip_idx = 1 - cur_idx
            else:
                key, flip_key = jax.random.split(key)
                delta = jax.random.randint(flip_key, (), 1, phys_dim, dtype=jnp.int32)
                flip_idx = (cur_idx + delta) % phys_dim
            mpo_flip = jnp.transpose(site_tensor[flip_idx], (2, 3, 0, 1))
            mpo_cur = mpo_row[col]
            # Direct einsum for amplitude computation (no transfer intermediate)
            amp_cur = jnp.einsum(
                "ace,aub,cduv,evf,bdf->",
                left_env, top_env[col], mpo_cur, bottom_env[col], right_envs[col],
                optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
            )
            amp_flip = jnp.einsum(
                "ace,aub,cduv,evf,bdf->",
                left_env, top_env[col], mpo_flip, bottom_env[col], right_envs[col],
                optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
            )
            weight_cur = jnp.abs(amp_cur) ** 2
            weight_flip = jnp.abs(amp_flip) ** 2
            ratio = _metropolis_ratio(weight_cur, weight_flip)

            key, accept_key = jax.random.split(key)
            accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)
            new_idx = jnp.where(accept, flip_idx, cur_idx)
            indices = indices.at[row, col].set(new_idx)

            mpo_sel = jnp.where(accept, mpo_flip, mpo_cur)
            updated_row.append(mpo_sel)
            # Direct einsum for left_env update
            left_env = jnp.einsum(
                "ace,aub,cduv,evf->bdf",
                left_env, top_env[col], mpo_sel, bottom_env[col],
                optimize=[(0, 1), (0, 2), (0, 1)],
            )

        top_env = model.strategy.apply(top_env, tuple(updated_row))

    amp = _contract_bottom(top_env)
    return indices.reshape(-1), key, amp


class PEPS(nnx.Module):
    """Open-boundary PEPS on a rectangular grid contracted with a boundary MPS.

    Each site tensor has shape (phys_dim, up, down, left, right) with boundary
    bonds set to dimension 1. Truncation behavior is controlled by the
    contraction strategy (for example, Variational(truncate_bond_dimension=...)).
    The default strategy is Variational(truncate_bond_dimension=bond_dim**2).
    """

    tensors: list[list[nnx.Param]] = nnx.data()

    @staticmethod
    def site_dims(
        row: int, col: int, n_rows: int, n_cols: int, bond_dim: int
    ) -> tuple[int, int, int, int]:
        """Return (up, down, left, right) dimensions for a PEPS site."""
        up = 1 if row == 0 else bond_dim
        down = 1 if row == n_rows - 1 else bond_dim
        left = 1 if col == 0 else bond_dim
        right = 1 if col == n_cols - 1 else bond_dim
        return up, down, left, right

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        shape: tuple[int, int],
        bond_dim: int,
        phys_dim: int = 2,
        contraction_strategy: ContractionStrategy | None = None,
        dtype: "DTypeLike" = jnp.complex128,
    ):
        """Initialize PEPS with random tensors.

        Args:
            rngs: Flax NNX random key generator.
            shape: Grid shape (n_rows, n_cols).
            bond_dim: Virtual bond dimension.
            phys_dim: Physical dimension (default 2 for spins).
            contraction_strategy: Contraction strategy instance (default:
                Variational with truncate_bond_dimension=bond_dim**2).
            dtype: Data type for tensors (default: complex128).
        """
        self.shape = (int(shape[0]), int(shape[1]))
        self.bond_dim = int(bond_dim)
        self.phys_dim = int(phys_dim)
        self.dtype = jnp.dtype(dtype)
        if contraction_strategy is None:
            contraction_strategy = Variational(
                truncate_bond_dimension=self.bond_dim * self.bond_dim
            )
        self.strategy = contraction_strategy

        n_rows, n_cols = self.shape
        self.tensors = [
            [
                nnx.Param(
                    random_tensor(
                        rngs,
                        (
                            self.phys_dim,
                            *self.site_dims(r, c, n_rows, n_cols, self.bond_dim),
                        ),
                        self.dtype,
                    ),
                    dtype=self.dtype,
                )
                for c in range(n_cols)
            ]
            for r in range(n_rows)
        ]

    apply = staticmethod(_peps_apply)

    @staticmethod
    def unflatten_sample(sample: jax.Array, shape: tuple[int, int]) -> jax.Array:
        """Unflatten sample to (n_rows, n_cols) indices."""
        return sample.reshape(shape)

    @nnx.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute log-amplitudes for input spin configurations.

        Args:
            x: Spin configurations with shape (batch, n_sites).

        Returns:
            Log-amplitudes with shape (batch,).
        """
        amps = jax.vmap(
            lambda s: self.apply(
                self.tensors, s, self.shape, self.strategy
            )
        )(x)
        return jnp.log(amps)

    def random_physical_configuration(
        self, key: jax.Array, n_samples: int = 1
    ) -> jax.Array:
        return jax.random.randint(
            key,
            (n_samples, self.shape[0], self.shape[1]),
            0,
            self.phys_dim,
            dtype=jnp.int32,
        )

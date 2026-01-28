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
    # Internal functions exposed for external use
    "_apply_mpo_from_below",
    "_build_row_mpo",
    "_compute_all_gradients",
    "_compute_all_row_gradients",
    "_compute_all_env_grads_and_energy",
    "_compute_right_envs",
    "_compute_2site_horizontal_env",
    "_compute_single_gradient",
    "_contract_bottom",
    "_contract_column_transfer",
    "_contract_left_partial",
    "_contract_right_partial",
    "_compute_row_pair_vertical_amps",
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


def _contract_theta(m: jax.Array, w: jax.Array, carry: jax.Array | None) -> tuple:
    """Contract MPS tensor with MPO tensor and optional carry from previous site.

    Returns (theta, left_dim, phys_dim, Dr, wr) where theta has shape
    (left_dim, phys_dim, Dr, wr).
    """
    theta = jnp.tensordot(m, w, axes=[[1], [2]])  # (Dl, Dr, wl, wr, down)
    theta = jnp.transpose(theta, (0, 2, 4, 1, 3))  # (Dl, wl, down, Dr, wr)
    if carry is not None:
        theta = jnp.tensordot(carry, theta, axes=[[1, 2], [0, 1]])
    if theta.ndim == 5:
        Dl, wl, down, Dr, wr = theta.shape
        theta = theta.reshape(Dl * wl, down, Dr, wr)
    left_dim, phys_dim, Dr, wr = theta.shape
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
        svd_eps = jnp.array(1e-7, dtype=mat.dtype)
        mat = mat + svd_eps * jnp.eye(mat.shape[0], mat.shape[1], dtype=mat.dtype)
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


def _contract_column_transfer(
    top_tensor: jax.Array,
    mpo_tensor: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Contract column into transfer tensor with shape (tL, tR, mL, mR, bL, bR)."""
    top_mpo = jnp.einsum("aub,cduv->abcdv", top_tensor, mpo_tensor)
    return jnp.einsum("abcdv,evf->abcdef", top_mpo, bot_tensor)


def _contract_left_partial(left_env: jax.Array, transfer: jax.Array) -> jax.Array:
    """Contract left environment (tL, mL, bL) with transfer -> (tR, mR, bR)."""
    return jnp.einsum("ace,abcdef->bdf", left_env, transfer)


def _contract_right_partial(transfer: jax.Array, right_env: jax.Array) -> jax.Array:
    """Contract transfer with right environment (tR, mR, bR) -> (tL, mL, bL)."""
    return jnp.einsum("abcdef,bdf->ace", transfer, right_env)


def _contract_column_transfer_2row(
    top_tensor: jax.Array,
    mpo0: jax.Array,
    mpo1: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Contract two-row column transfer with shape (tL, tR, m0L, m0R, m1L, m1R, bL, bR)."""
    tmp = jnp.einsum("aub,lruv->alrbv", top_tensor, mpo0)
    tmp = jnp.einsum("alrbv,xyvw->alrbxyw", tmp, mpo1)
    tmp = jnp.einsum("alrbxyw,ewf->alrbxyef", tmp, bot_tensor)
    return jnp.transpose(tmp, (0, 3, 1, 2, 4, 5, 6, 7))


def _contract_left_partial_2row(
    left_env: jax.Array,
    transfer: jax.Array,
) -> jax.Array:
    """Contract left environment (tL, m0L, m1L, bL) with transfer."""
    return jnp.einsum("aceg,abcdefgh->bdfh", left_env, transfer)


def _contract_right_partial_2row(
    transfer: jax.Array,
    right_env: jax.Array,
) -> jax.Array:
    """Contract transfer with right environment (tR, m0R, m1R, bR)."""
    return jnp.einsum("abcdefgh,bdfh->aceg", transfer, right_env)


def _contract_column_transfer_2row_open(
    top_tensor: jax.Array,
    tensor0: jax.Array,
    tensor1: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Contract two rows leaving physical indices open."""
    tmp = jnp.einsum("aub,puvlr->apbvlr", top_tensor, tensor0)
    tmp = jnp.einsum("apbvlr,qvdmn->apbqlrdmn", tmp, tensor1)
    tmp = jnp.einsum("apbqlrdmn,edf->apbqlrmnef", tmp, bot_tensor)
    return jnp.transpose(tmp, (0, 2, 4, 5, 6, 7, 8, 9, 1, 3))


def _compute_right_envs_2row(
    transfers: list[jax.Array],
    dtype,
) -> list[jax.Array]:
    """Compute right environments for 2-row contractions (backward pass)."""
    n_cols = len(transfers)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        right_envs[c] = _contract_right_partial_2row(transfers[c + 1], right_envs[c + 1])
    return right_envs


def _compute_row_pair_vertical_amps(
    top_mps: tuple,
    bottom_mps: tuple,
    mpo0: tuple,
    mpo1: tuple,
    tensors_row0: list[jax.Array],
    tensors_row1: list[jax.Array],
) -> list[jax.Array]:
    """Compute vertical 2-site amplitudes between adjacent rows."""
    n_cols = len(mpo0)
    dtype = mpo0[0].dtype
    transfers = [
        _contract_column_transfer_2row(top_mps[c], mpo0[c], mpo1[c], bottom_mps[c])
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs_2row(transfers, dtype)

    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
    amps_v2site = []
    for c in range(n_cols):
        transfer_open = _contract_column_transfer_2row_open(
            top_mps[c], tensors_row0[c], tensors_row1[c], bottom_mps[c]
        )
        amps_v2site.append(
            jnp.einsum(
                "aceg,abcdefghpq,bdfh->pq",
                left_env,
                transfer_open,
                right_envs[c],
            )
        )
        left_env = _contract_left_partial_2row(left_env, transfers[c])
    return amps_v2site


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
) -> jax.Array:
    """Compute vertical 2-site energy contributions for a row pair."""
    if not any(terms_row):
        return jnp.zeros((), dtype=amp.dtype)
    n_cols = len(mpo_row0)
    dtype = mpo_row0[0].dtype
    transfers = [
        _contract_column_transfer_2row(
            top_mps[c], mpo_row0[c], mpo_row1[c], bottom_mps[c]
        )
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs_2row(transfers, dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
    energy = jnp.zeros((), dtype=amp.dtype)
    for c in range(n_cols):
        col_terms = terms_row[c]
        if col_terms:
            transfer_open = _contract_column_transfer_2row_open(
                top_mps[c], tensors_row0[c], tensors_row1[c], bottom_mps[c]
            )
            amps_edge = jnp.einsum(
                "aceg,abcdefghpq,bdfh->pq",
                left_env,
                transfer_open,
                right_envs[c],
            )
            spin0 = spins_row0[c]
            spin1 = spins_row1[c]
            col_idx = spin0 * phys_dim + spin1
            amps_flat = amps_edge.reshape(-1)
            for term in col_terms:
                energy = energy + jnp.dot(term.op[:, col_idx], amps_flat) / amp
        left_env = _contract_left_partial_2row(left_env, transfers[c])
    return energy


def _compute_right_envs(
    transfers: list[jax.Array],
    dtype,
) -> list[jax.Array]:
    """Compute right environments from column transfers (backward pass)."""
    n_cols = len(transfers)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        right_envs[c] = _contract_right_partial(transfers[c + 1], right_envs[c + 1])
    return right_envs


def _compute_all_row_gradients(
    top_mps: tuple,
    bottom_mps: tuple,
    mpo: tuple,
) -> list[jax.Array]:
    """Compute gradients for all tensors in a row using environment contraction."""
    n_cols = len(mpo)
    dtype = mpo[0].dtype
    transfers = [
        _contract_column_transfer(top_mps[c], mpo[c], bottom_mps[c])
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs(transfers, dtype)

    env_grads = []
    left_env = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols):
        env_grads.append(
            _compute_single_gradient(left_env, right_envs[c], top_mps[c], bottom_mps[c])
        )
        left_env = _contract_left_partial(left_env, transfers[c])
    return env_grads


def _compute_all_env_grads_and_energy(
    tensors: Any,
    spins: jax.Array,
    amp: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    *,
    diagonal_terms: list,
    one_site_terms: list[list[list]],
    horizontal_terms: list[list[list]],
    vertical_terms: list[list[list]],
    collect_grads: bool = True,
) -> tuple[list[list[jax.Array]], jax.Array, list[tuple]]:
    """Compute gradients and local energy for a PEPS sample."""
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype
    phys_dim = int(jnp.asarray(tensors[0][0]).shape[0])

    env_grads = (
        [[None for _ in range(n_cols)] for _ in range(n_rows)]
        if collect_grads
        else []
    )
    bottom_envs = [None for _ in range(n_rows)]
    energy = jnp.zeros((), dtype=amp.dtype)
    for term in diagonal_terms:
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + spins[row, col]
        energy = energy + term.diag[idx]

    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        bottom_envs[row] = bottom_env
        mpo = _build_row_mpo(tensors, spins[row], row, n_cols)
        bottom_env = _apply_mpo_from_below(bottom_env, mpo, strategy)

    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    mpo = _build_row_mpo(tensors, spins[0], 0, n_cols)
    for row in range(n_rows):
        bottom_env = bottom_envs[row]
        transfers = [
            _contract_column_transfer(top_env[c], mpo[c], bottom_env[c])
            for c in range(n_cols)
        ]
        right_envs = _compute_right_envs(transfers, dtype)
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        for c in range(n_cols):
            env_grad = _compute_single_gradient(
                left_env, right_envs[c], top_env[c], bottom_env[c]
            )
            if collect_grads:
                env_grads[row][c] = env_grad
            site_terms = one_site_terms[row][c]
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
                    )
                    spin0 = spins[row, c]
                    spin1 = spins[row, c + 1]
                    col_idx = spin0 * phys_dim + spin1
                    amps_flat = amps_edge.reshape(-1)
                    for term in edge_terms:
                        energy = energy + jnp.dot(term.op[:, col_idx], amps_flat) / amp
            left_env = _contract_left_partial(left_env, transfers[c])
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

    return env_grads, energy, bottom_envs


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
    # Contract left side: left_env with top0 and bot0
    # left_env (a,c,e) @ top0 (a,u,b) -> (c,e,u,b) = (mL, bL, up0, t01)
    tmp_left = jnp.einsum("ace,aub->ceub", left_env, top0)
    # (c,e,u,b) @ bot0 (e,d,f) -> (c,u,b,d,f) = (mL, up0, t01, down0, b01)
    tmp_left = jnp.einsum("ceub,edf->cubdf", tmp_left, bot0)

    # Contract right side: top1 and bot1 with right_env
    # top1 (b,v,g) @ right_env (g,h,i) -> (b,v,h,i) = (t01, up1, mR, bR)
    tmp_right = jnp.einsum("bvg,ghi->bvhi", top1, right_env)
    # (b,v,h,i) @ bot1 (f,w,i) -> (b,v,h,f,w) = (t01, up1, mR, b01, down1)
    tmp_right = jnp.einsum("bvhi,fwi->bvhfw", tmp_right, bot1)

    # Contract left and right: connect t01 and b01
    # (c,u,b,d,f) @ (b,v,h,f,w) -> (c,u,d,v,h,w) = (mL, up0, down0, up1, mR, down1)
    env = jnp.einsum("cubdf,bvhfw->cudvhw", tmp_left, tmp_right)
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
    tmp_top = jnp.einsum("ace,aub->ceub", left_env, top_tensor)
    tmp_bot = jnp.einsum("evf,bdf->ebdv", bot_tensor, right_env)
    grad = jnp.einsum("ceub,ebdv->cuvd", tmp_top, tmp_bot)
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


class PEPS(nnx.Module):
    """Open-boundary PEPS on a rectangular grid contracted with a boundary MPS.

    Each site tensor has shape (phys_dim, up, down, left, right) with boundary
    bonds set to dimension 1. Truncation behavior is controlled by the
    contraction strategy (for example, ZipUp(truncate_bond_dimension=...)).
    The default strategy is ZipUp(truncate_bond_dimension=bond_dim**2).
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
            contraction_strategy: Contraction strategy instance (default: ZipUp
                with truncate_bond_dimension=bond_dim**2).
            dtype: Data type for tensors (default: complex128).
        """
        self.shape = (int(shape[0]), int(shape[1]))
        self.bond_dim = int(bond_dim)
        self.phys_dim = int(phys_dim)
        self.dtype = jnp.dtype(dtype)
        if contraction_strategy is None:
            contraction_strategy = ZipUp(
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

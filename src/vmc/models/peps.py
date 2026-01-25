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
    "_compute_single_gradient",
    "_contract_bottom",
    "_contract_column_transfer",
    "_contract_left_partial",
    "_contract_right_partial",
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

    left_envs = [jnp.ones((1, 1, 1), dtype=dtype)]
    for c in range(n_cols - 1):
        left_envs.append(_contract_left_partial(left_envs[-1], transfers[c]))

    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        right_envs[c] = _contract_right_partial(transfers[c + 1], right_envs[c + 1])

    return [
        _compute_single_gradient(
            left_envs[c], right_envs[c], top_mps[c], bottom_mps[c]
        )
        for c in range(n_cols)
    ]


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
        if row_mpos is None:
            mpo = _build_row_mpo(tensors, spins[row], row, n_cols)
        else:
            mpo = row_mpos[row]
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

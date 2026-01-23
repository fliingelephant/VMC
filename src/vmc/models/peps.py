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


@jax.jit
def _apply_mpo_exact(mps: tuple, mpo: tuple) -> tuple:
    """Exact MPO application without truncation."""
    new = []
    for m, w in zip(mps, mpo):
        contracted = jnp.tensordot(m, w, axes=[[1], [2]])
        contracted = jnp.transpose(contracted, (0, 2, 4, 1, 3))
        left, wl, phys, right, wr = contracted.shape
        new.append(contracted.reshape(left * wl, phys, right * wr))
    return tuple(new)


@functools.partial(jax.jit, static_argnums=(2,))
def _apply_mpo_zip_up(
    mps: tuple, mpo: tuple, truncate_bond_dimension: int
) -> tuple:
    """Apply MPO with on-the-fly SVD truncation (zip-up).

    This avoids the large intermediate bond growth of the two-stage
    apply-then-truncate approach and keeps the right bond of each
    intermediate MPS bounded by `truncate_bond_dimension`.
    """
    new = []
    carry = None  # shape (bond, Dr_prev, wr_prev) propagated to the right

    for i, (m, w) in enumerate(zip(mps, mpo)):
        # Contract physical/up legs.
        theta = jnp.tensordot(m, w, axes=[[1], [2]])  # (Dl, Dr, wl, wr, down)
        theta = jnp.transpose(theta, (0, 2, 4, 1, 3))  # (Dl, wl, down, Dr, wr)

        if carry is not None:
            # Merge previous transfer matrix into the left legs (Dr_prev, wr_prev).
            theta = jnp.tensordot(carry, theta, axes=[[1, 2], [0, 1]])
            # theta shape -> (bond_prev, down, Dr, wr)

        if theta.ndim == 5:
            Dl, wl, down, Dr, wr = theta.shape
            theta = theta.reshape(Dl * wl, down, Dr, wr)
        # theta now has shape (left_dim, phys_dim=down, Dr, wr)
        left_dim, phys_dim, Dr, wr = theta.shape

        if i == len(mps) - 1:
            # Last site: no further truncation needed; keep right leg explicit.
            new.append(theta.reshape(left_dim, phys_dim, Dr * wr))
            carry = None
            break

        # SVD-based truncation on reshaped matrix (left*phys, right).
        mat = theta.reshape(left_dim * phys_dim, Dr * wr)
        # Jitter avoids exact rank-deficiency that can blow up SVD derivatives.
        svd_eps = jnp.array(1e-7, dtype=mat.dtype)
        mat = mat + svd_eps * jnp.eye(mat.shape[0], mat.shape[1], dtype=mat.dtype)
        U, S, Vh = jnp.linalg.svd(mat, full_matrices=False)
        k = min(truncate_bond_dimension, S.shape[0])
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]

        site = U.reshape(left_dim, phys_dim, k)
        new.append(site)

        # Keep truncation algebra in the complex dtype to avoid complex->real
        # casts in autodiff (which can trigger warnings/NaNs).
        S_c = S.astype(Vh.dtype)
        transfer = S_c[:, None] * Vh  # (k, Dr*wr)
        carry = transfer.reshape(k, Dr, wr)

    return tuple(new)


@functools.partial(jax.jit, static_argnums=(2,))
def _apply_mpo_density_matrix(
    mps: tuple, mpo: tuple, truncate_bond_dimension: int
) -> tuple:
    """Density-matrix truncation while applying an MPO to an MPS.

    Uses the right reduced density matrix to select the dominant
    eigenvectors (TEBD-style).
    """
    new = []
    carry = None  # shape (bond, Dr_prev, wr_prev)

    for i, (m, w) in enumerate(zip(mps, mpo)):
        theta = jnp.tensordot(m, w, axes=[[1], [2]])
        theta = jnp.transpose(theta, (0, 2, 4, 1, 3))  # (Dl, wl, down, Dr, wr)

        if carry is not None:
            theta = jnp.tensordot(carry, theta, axes=[[1, 2], [0, 1]])

        if theta.ndim == 5:
            Dl, wl, down, Dr, wr = theta.shape
            theta = theta.reshape(Dl * wl, down, Dr, wr)

        left_dim, phys_dim, Dr, wr = theta.shape
        theta = theta.reshape(left_dim * phys_dim, Dr * wr)

        if i == len(mps) - 1:
            new.append(theta.reshape(left_dim, phys_dim, Dr * wr))
            carry = None
            break

        rho = theta.conj().T @ theta  # (Dr*wr, Dr*wr)
        evals, evecs = jnp.linalg.eigh(rho)
        order = jnp.argsort(evals)[::-1]
        k = min(truncate_bond_dimension, rho.shape[0])
        vecs_k = evecs[:, order[:k]]

        theta_projected = theta @ vecs_k  # (left_dim*phys, k)
        site = theta_projected.reshape(left_dim, phys_dim, k)
        new.append(site)

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
    """Forward pass that caches all intermediate boundary MPSs.

    Args:
        tensors: Nested list of PEPS site tensors.
        spins: Physical indices array with shape (n_rows, n_cols).
        shape: Grid shape (n_rows, n_cols).
        strategy: Contraction strategy instance.

    Returns:
        Tuple of (amplitude, top_envs) where top_envs[r] is the boundary MPS
        after contracting rows 0..r-1.
    """
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype

    # Cache boundary MPSs after each row
    top_envs = []

    # Initial boundary (before row 0)
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
    """Apply MPO to boundary MPS from below (for backward sweep).

    When contracting from below, the MPO's 'down' leg connects to the boundary's
    physical dimension, and the 'up' leg becomes the new physical dimension.

    Args:
        bottom_mps: Current bottom boundary MPS.
        mpo: Row MPO with tensors of shape (left, right, up, down).
        strategy: Contraction strategy instance.

    Returns:
        Updated bottom boundary MPS.
    """
    # Transpose MPO tensors to swap up/down: (left, right, up, down) -> (left, right, down, up)
    mpo_transposed = tuple(jnp.transpose(w, (0, 1, 3, 2)) for w in mpo)
    return strategy.apply(bottom_mps, mpo_transposed)


def _contract_column_transfer(
    top_tensor: jax.Array,
    mpo_tensor: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Contract a single column (top MPS tensor + MPO tensor + bottom MPS tensor)
    into a transfer tensor.

    Args:
        top_tensor: (tL, tD, tR) - top boundary MPS tensor, tD == up bond of mpo
        mpo_tensor: (mL, mR, up, down) - row MPO tensor
        bot_tensor: (bL, bD, bR) - bottom boundary MPS tensor, bD == down bond of mpo

    Returns:
        Transfer tensor with shape (tL, tR, mL, mR, bL, bR) after contracting
        over the physical dimensions (up, down).
    """
    # Contract top with mpo over physical/up bond, then with bottom over down bond
    return jnp.einsum(
        "abcdv,evf->abcdef",
        jnp.einsum("aub,cduv->abcdv", top_tensor, mpo_tensor),
        bot_tensor,
    )


def _contract_left_partial(
    left_env: jax.Array,
    transfer: jax.Array,
) -> jax.Array:
    """Contract left environment with a transfer tensor.

    Args:
        left_env: (tL, mL, bL) - left environment tensor
        transfer: (tL, tR, mL, mR, bL, bR) - column transfer tensor

    Returns:
        New left environment with shape (tR, mR, bR).
    """
    return jnp.einsum("ace,abcdef->bdf", left_env, transfer)


def _contract_right_partial(
    transfer: jax.Array,
    right_env: jax.Array,
) -> jax.Array:
    """Contract transfer tensor with right environment.

    Args:
        transfer: (tL, tR, mL, mR, bL, bR) - column transfer tensor
        right_env: (tR, mR, bR) - right environment tensor

    Returns:
        New right environment with shape (tL, mL, bL).
    """
    return jnp.einsum("abcdef,bdf->ace", transfer, right_env)


def _compute_all_row_gradients(
    top_mps: tuple,
    bottom_mps: tuple,
    mpo: tuple,
) -> list[jax.Array]:
    """Compute gradients for all tensors in a row using environment contraction.

    Args:
        top_mps: Top boundary MPS after contracting rows above.
        bottom_mps: Bottom boundary MPS after contracting rows below.
        mpo: Row MPO (with physical indices selected).

    Returns:
        List of gradient tensors, one per column.
    """
    n_cols = len(mpo)
    dtype = mpo[0].dtype

    transfers = [
        _contract_column_transfer(top_mps[c], mpo[c], bottom_mps[c])
        for c in range(n_cols)
    ]

    # Compute left environments (partial contractions from left)
    left_envs = [jnp.ones((1, 1, 1), dtype=dtype)]
    for c in range(n_cols - 1):
        left_envs.append(_contract_left_partial(left_envs[-1], transfers[c]))

    # Compute right environments (partial contractions from right)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        right_envs[c] = _contract_right_partial(transfers[c + 1], right_envs[c + 1])

    return [
        _compute_single_gradient(
            left_envs[c], right_envs[c], top_mps[c], bottom_mps[c], mpo[c].shape
        )
        for c in range(n_cols)
    ]


def _compute_single_gradient(
    left_env: jax.Array,
    right_env: jax.Array,
    top_tensor: jax.Array,
    bot_tensor: jax.Array,
    mpo_shape: tuple,
) -> jax.Array:
    """Compute gradient for a single tensor given left/right environments.

    Args:
        left_env: (tL, mL, bL) - left environment
        right_env: (tR, mR, bR) - right environment
        top_tensor: (tL, up, tR) - top boundary MPS tensor
        bot_tensor: (bL, down, bR) - bottom boundary MPS tensor
        mpo_shape: (mL, mR, up, down) - shape of the MPO tensor

    Returns:
        Gradient tensor with shape (up, down, mL, mR).
    """
    # TODO: optimize tensor network contraction order for better performance
    # Contract left_env with top_tensor over tL
    tmp1 = jnp.einsum("ace,aub->ceub", left_env, top_tensor)
    # Contract with bot_tensor over bL
    tmp2 = jnp.einsum("ceub,evf->cubvf", tmp1, bot_tensor)
    # Contract with right_env over tR and bR, transpose to (up, down, left, right)
    return jnp.transpose(jnp.einsum("cubvf,bdf->cuvd", tmp2, right_env), (1, 2, 0, 3))


def _compute_all_gradients(
    tensors: Any,
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    top_envs: list[tuple],
    *,
    cache_bottom_envs: bool = False,
) -> list[list[jax.Array]] | tuple[list[list[jax.Array]], list[tuple]]:
    """Compute gradients for all PEPS tensors using cached top environments.

    Args:
        tensors: Nested list of PEPS site tensors.
        spins: Physical indices array with shape (n_rows, n_cols).
        shape: Grid shape (n_rows, n_cols).
        strategy: Contraction strategy instance.
        top_envs: Cached top boundary MPSs from forward pass.

    Returns:
        Nested list of gradient tensors matching the structure of tensors.
        If ``cache_bottom_envs`` is True, also returns the bottom environments
        for each row (before updating with that row's MPO).
    """
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype

    # Initialize gradient storage
    grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    # Optional cache for callers that reuse bottom environments (e.g. samplers).
    bottom_envs_cached = [None] * n_rows if cache_bottom_envs else None

    # Initialize bottom environment (trivial for below the last row)
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    # Sweep from bottom to top
    for row in range(n_rows - 1, -1, -1):
        top_env = top_envs[row]
        if cache_bottom_envs:
            bottom_envs_cached[row] = bottom_env
        mpo = _build_row_mpo(tensors, spins[row], row, n_cols)

        # Compute gradients for all tensors in this row
        row_grads = _compute_all_row_gradients(top_env, bottom_env, mpo)
        grads[row] = row_grads

        # Update bottom_env by contracting this row from below
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

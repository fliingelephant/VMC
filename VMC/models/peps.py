"""PEPS (Projected Entangled Pair States) implementation.

This module provides a PEPS implementation using boundary MPS contraction
with configurable truncation strategies via ABC pattern.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import abc
import functools
import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import nnx
from netket.utils import timing

from VMC.utils.utils import spin_to_occupancy

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = [
    "SimplePEPS",
    # Contraction strategies (ABC pattern)
    "ContractionStrategy",
    "NoTruncation",
    "ZipUp",
    "DensityMatrix",
    # Amplitude functions
    "make_peps_amplitude",
    "peps_amplitude",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Contraction Strategy Classes (ABC pattern)
# =============================================================================


class ContractionStrategy(abc.ABC):
    """Abstract base class for MPO-to-MPS contraction strategies."""

    def __init__(self, truncate_bond_dimension: int):
        if truncate_bond_dimension <= 0:
            raise ValueError("truncate_bond_dimension must be positive.")
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


def _build_row_mpo_common(tensors, row_indices, row, n_cols):
    """Shared row-MPO construction for static and class helpers."""
    mpo = []
    for col in range(n_cols):
        site = jnp.asarray(tensors[row][col])[row_indices[col]]
        mpo.append(jnp.transpose(site, (2, 3, 0, 1)))  # (left, right, up, down)
    return tuple(mpo)


def _build_row_mpo_static(tensors, row_indices, row, n_cols):
    """Static version of _build_row_mpo for use in custom_vjp."""
    return _build_row_mpo_common(tensors, row_indices, row, n_cols)


def _contract_bottom_common(mps):
    """Shared boundary contraction for static and class helpers."""
    state = jnp.array([1.0], dtype=mps[0].dtype)
    for site in mps:
        state = jnp.tensordot(state, site[:, 0, :], axes=[[0], [0]])
    return state.squeeze()


def _contract_bottom_static(mps):
    """Static version of _contract_bottom for use in custom_vjp."""
    return _contract_bottom_common(mps)


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
        mpo = _build_row_mpo_static(tensors, spins[row], row, n_cols)
        boundary = strategy.apply(boundary, mpo)
        top_envs.append(boundary)

    # Contract final boundary to get amplitude
    amp = _contract_bottom_static(boundary)
    return amp, top_envs


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
    # Contract top with mpo over physical/up bond (index u)
    top_mpo = jnp.einsum("aub,cduv->abcdv", top_tensor, mpo_tensor)
    # Contract with bottom over down bond (index v)
    transfer = jnp.einsum("abcdv,evf->abcdef", top_mpo, bot_tensor)
    return transfer


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

    # Compute all transfer tensors
    transfers = []
    for c in range(n_cols):
        transfer = _contract_column_transfer(top_mps[c], mpo[c], bottom_mps[c])
        transfers.append(transfer)

    # Compute left environments (partial contractions from left)
    left_envs = [jnp.ones((1, 1, 1), dtype=dtype)]
    env = left_envs[0]
    for c in range(n_cols - 1):
        env = _contract_left_partial(env, transfers[c])
        left_envs.append(env)

    # Compute right environments (partial contractions from right)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
    env = right_envs[n_cols - 1]
    for c in range(n_cols - 2, -1, -1):
        env = _contract_right_partial(transfers[c + 1], env)
        right_envs[c] = env

    # Compute gradient for each tensor
    grads = []
    for c in range(n_cols):
        grad = _compute_single_gradient(
            left_envs[c],
            right_envs[c],
            top_mps[c],
            bottom_mps[c],
            mpo[c].shape,
        )
        grads.append(grad)

    return grads


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
    # Contract left_env with top_tensor over tL
    tmp1 = jnp.einsum("ace,aub->ceub", left_env, top_tensor)
    # Contract with bot_tensor over bL
    tmp2 = jnp.einsum("ceub,evf->cubvf", tmp1, bot_tensor)
    # Contract with right_env over tR and bR
    grad_mpo = jnp.einsum("cubvf,bdf->cuvd", tmp2, right_env)
    # Transpose to (up, down, left, right)
    grad = jnp.transpose(grad_mpo, (1, 2, 0, 3))
    return grad


def _compute_all_gradients(
    tensors: Any,
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    top_envs: list[tuple],
) -> list[list[jax.Array]]:
    """Compute gradients for all PEPS tensors using cached top environments.

    Args:
        tensors: Nested list of PEPS site tensors.
        spins: Physical indices array with shape (n_rows, n_cols).
        shape: Grid shape (n_rows, n_cols).
        strategy: Contraction strategy instance.
        top_envs: Cached top boundary MPSs from forward pass.

    Returns:
        Nested list of gradient tensors matching the structure of tensors.
    """
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype

    # Initialize gradient storage
    grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    # Initialize bottom environment (trivial for below the last row)
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    # Sweep from bottom to top
    for row in range(n_rows - 1, -1, -1):
        top_env = top_envs[row]
        mpo = _build_row_mpo_static(tensors, spins[row], row, n_cols)

        # Compute gradients for all tensors in this row
        row_grads = _compute_all_row_gradients(top_env, bottom_env, mpo)
        grads[row] = row_grads

        # Update bottom_env by contracting this row from below
        bottom_env = _apply_mpo_from_below(bottom_env, mpo, strategy)

    return grads


def make_peps_amplitude(
    shape: tuple[int, int],
    strategy: ContractionStrategy,
):
    """Create a PEPS amplitude function with custom VJP for the given configuration.

    The returned function has signature `(tensors, sample) -> amplitude` and
    implements a custom VJP computing gradients via environment contraction.

    Args:
        shape: Grid shape (n_rows, n_cols).
        strategy: Contraction strategy instance.

    Returns:
        A function `(tensors, sample) -> amplitude` with a custom VJP.
    """
    if not isinstance(strategy, ContractionStrategy):
        raise TypeError(
            "strategy must be a ContractionStrategy instance, "
            f"got {type(strategy)}"
        )

    @jax.custom_vjp
    def amplitude_fn(tensors: Any, sample: jax.Array) -> jax.Array:
        """Compute PEPS amplitude with custom VJP."""
        return SimplePEPS._single_amplitude(tensors, sample, shape, strategy)

    def amplitude_fwd(tensors: Any, sample: jax.Array) -> tuple[jax.Array, tuple]:
        """Forward pass returning residuals."""
        spins = spin_to_occupancy(sample).reshape(shape)
        amp, top_envs = _forward_with_cache(tensors, spins, shape, strategy)
        residuals = (tensors, spins, top_envs)
        return amp, residuals

    def amplitude_bwd(residuals: tuple, g: jax.Array) -> tuple:
        """Backward pass computing gradients via environment contraction."""
        tensors, spins, top_envs = residuals
        n_rows, n_cols = shape

        # Compute all environment-based gradients
        env_grads = _compute_all_gradients(
            tensors, spins, shape, strategy, top_envs
        )

        # Build gradients matching the pytree structure of `tensors`.
        grad_leaves: list[jax.Array] = []
        for r in range(n_rows):
            for c in range(n_cols):
                full_tensor = jnp.asarray(tensors[r][c])
                grad_full = jnp.zeros_like(full_tensor)
                phys_idx = spins[r, c]
                grad_full = grad_full.at[phys_idx].set(g * env_grads[r][c])
                grad_leaves.append(grad_full)

        treedef = jax.tree_util.tree_structure(tensors)
        grad_tensors = jax.tree_util.tree_unflatten(treedef, grad_leaves)

        # Return gradients: (grad_tensors, None for sample)
        return (grad_tensors, None)

    amplitude_fn.defvjp(amplitude_fwd, amplitude_bwd)
    return amplitude_fn


def peps_amplitude(
    tensors: Any,
    sample: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> jax.Array:
    """Convenience wrapper around `make_peps_amplitude`.

    For JAX transforms (`grad`, `jacrev`, `vmap`) prefer calling
    `make_peps_amplitude(shape, strategy)` once and reusing it.
    """
    amp_fn = make_peps_amplitude(shape, strategy)
    return amp_fn(tensors, sample)


class SimplePEPS(nnx.Module):
    """Open-boundary PEPS on a rectangular grid contracted with a boundary MPS.

    Each site tensor has shape (phys_dim, up, down, left, right) with boundary
    bonds set to dimension 1. Truncation behavior is controlled by the
    contraction strategy (for example, ZipUp(truncate_bond_dimension=...)).
    The default strategy is ZipUp(truncate_bond_dimension=bond_dim**2).
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        shape: tuple[int, int],
        bond_dim: int,
        contraction_strategy: ContractionStrategy | None = None,
        dtype: "DTypeLike" = jnp.complex128,
    ):
        """Initialize PEPS with random tensors.

        Args:
            rngs: Flax NNX random key generator.
            shape: Grid shape (n_rows, n_cols).
            bond_dim: Virtual bond dimension.
            contraction_strategy: Contraction strategy instance (default: ZipUp
                with truncate_bond_dimension=bond_dim**2).
            dtype: Data type for tensors (default: complex128).
        """
        self.shape = (int(shape[0]), int(shape[1]))
        self.bond_dim = int(bond_dim)
        self.dtype = jnp.dtype(dtype)
        if contraction_strategy is None:
            contraction_strategy = ZipUp(
                truncate_bond_dimension=self.bond_dim * self.bond_dim
            )
        if not isinstance(contraction_strategy, ContractionStrategy):
            raise TypeError(
                "contraction_strategy must be a ContractionStrategy instance, "
                f"got {type(contraction_strategy)}"
            )
        self.strategy = contraction_strategy

        # Determine real dtype for random initialization
        is_complex = jnp.issubdtype(self.dtype, jnp.complexfloating)
        if is_complex:
            real_dtype = jnp.real(jnp.zeros((), dtype=self.dtype)).dtype
            complex_unit = jnp.array(1j, dtype=self.dtype)
        else:
            real_dtype = self.dtype
            complex_unit = None

        rows = []
        n_rows, n_cols = self.shape
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                up = 1 if r == 0 else self.bond_dim
                down = 1 if r == n_rows - 1 else self.bond_dim
                left = 1 if c == 0 else self.bond_dim
                right = 1 if c == n_cols - 1 else self.bond_dim
                shape_rc = (2, up, down, left, right)

                if is_complex:
                    key_re, key_im = rngs.params(), rngs.params()
                    tensor_val = 1/2 * (
                        jax.random.uniform(key_re, shape_rc, dtype=real_dtype)
                        + complex_unit
                        * jax.random.uniform(key_im, shape_rc, dtype=real_dtype)
                    )
                else:
                    tensor_val = jax.random.uniform(
                        rngs.params(),
                        shape_rc,
                        dtype=real_dtype,
                    )

                row.append(nnx.Param(tensor_val, dtype=self.dtype))
            rows.append(nnx.List(row))
        self.tensors = nnx.List(rows)

    @staticmethod
    def _build_row_mpo(tensors, row_indices, row, n_cols):
        return _build_row_mpo_common(tensors, row_indices, row, n_cols)

    # Keep static methods for backwards compatibility with test code
    _apply_mpo = staticmethod(_apply_mpo_exact)
    _apply_mpo_zip_up = staticmethod(_apply_mpo_zip_up)
    _apply_mpo_density_matrix = staticmethod(_apply_mpo_density_matrix)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(1,))
    def _truncate_mps(mps, truncate_bond_dimension):
        mps = list(mps)
        for i in range(len(mps) - 1):
            left, phys, right = mps[i].shape
            mat = mps[i].reshape(left * phys, right)
            svd_eps = jnp.array(1e-7, dtype=mat.dtype)
            mat = mat + svd_eps * jnp.eye(mat.shape[0], mat.shape[1], dtype=mat.dtype)
            u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
            k = min(truncate_bond_dimension, s.shape[0])
            u = u[:, :k]
            s = s[:k]
            vh = vh[:k, :]
            mps[i] = u.reshape(left, phys, k)
            s_mat = s.astype(vh.dtype)[:, None] * vh  # keep complex dtype for grads
            mps[i + 1] = jnp.tensordot(s_mat, mps[i + 1], axes=[[1], [0]])
        return tuple(mps)

    @staticmethod
    @jax.jit
    def _contract_bottom(mps):
        return _contract_bottom_common(mps)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("shape", "strategy"))
    def _single_amplitude(
        tensors,
        sample: jax.Array,
        shape: tuple[int, int],
        strategy: ContractionStrategy,
    ) -> jax.Array:
        """Compute a single PEPS amplitude for a spin configuration.

        Args:
            tensors: Nested list of PEPS site tensors.
            sample: Spin configuration with shape (n_sites,).
            shape: Grid shape (rows, cols).
            strategy: Contraction strategy instance.

        Returns:
            Complex amplitude scalar.
        """
        spins = spin_to_occupancy(sample).reshape(shape)
        boundary = tuple(
            jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
            for _ in range(shape[1])
        )
        for row in range(shape[0]):
            mpo = SimplePEPS._build_row_mpo(tensors, spins[row], row, shape[1])
            boundary = strategy.apply(boundary, mpo)
        return SimplePEPS._contract_bottom(boundary)

    @nnx.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute log-amplitudes for input spin configurations.

        Args:
            x: Spin configurations with shape (batch, n_sites).

        Returns:
            Log-amplitudes with shape (batch,).
        """
        amps = jax.vmap(
            lambda s: self._single_amplitude(
                self.tensors, s, self.shape, self.strategy
            )
        )(x)
        return jnp.log(amps)



def test_gradient_correctness():
    """Test that custom VJP produces correct gradients."""
    logger.info("\n%s", "=" * 60)
    logger.info("Testing custom VJP gradient correctness")
    logger.info("=" * 60)

    # Small PEPS for testing
    shape = (2, 2)
    bond_dim = 2

    rngs = nnx.Rngs(42)
    model = SimplePEPS(rngs=rngs, shape=shape, bond_dim=bond_dim)
    tensors = model.tensors

    sample = jnp.array([1, -1, -1, 1], dtype=jnp.int32)

    def tensors_to_flat(tensors):
        leaves = []
        for row in tensors:
            for tensor in row:
                leaves.append(jnp.asarray(tensor).ravel())
        return jnp.concatenate(leaves)

    def flat_to_tensors(flat, tensors_template):
        result = []
        offset = 0
        for row in tensors_template:
            row_result = []
            for tensor in row:
                t = jnp.asarray(tensor)
                size = t.size
                row_result.append(flat[offset : offset + size].reshape(t.shape))
                offset += size
            result.append(row_result)
        return result

    # Test 1: Compare custom VJP gradient with no-truncation autodiff
    logger.info("\n1. Comparing custom VJP (no truncation) with autodiff...")

    amp_fn = make_peps_amplitude(shape, NoTruncation())

    def amplitude_custom_vjp(flat_params):
        tensors_nested = flat_to_tensors(flat_params, tensors)
        return amp_fn(tensors_nested, sample)

    def amplitude_autodiff(flat_params):
        tensors_nested = flat_to_tensors(flat_params, tensors)
        return SimplePEPS._single_amplitude(
            tensors_nested, sample, shape, NoTruncation()
        )

    flat_params = tensors_to_flat(tensors)

    grad_custom = jax.grad(amplitude_custom_vjp, holomorphic=True)(flat_params)
    grad_autodiff = jax.grad(amplitude_autodiff, holomorphic=True)(flat_params)

    rel_error = jnp.linalg.norm(grad_custom - grad_autodiff) / (
        jnp.linalg.norm(grad_autodiff) + 1e-10
    )
    logger.info("   Relative gradient error: %.6e", float(rel_error))

    if rel_error < 1e-5:
        logger.info("   PASS: Gradients match!")
    else:
        logger.warning("   FAIL: Gradients differ significantly!")

    # Test 2: Test that the amplitude values match
    logger.info("\n2. Comparing amplitude values...")
    amp_custom = amplitude_custom_vjp(flat_params)
    amp_autodiff = amplitude_autodiff(flat_params)
    amp_error = jnp.abs(amp_custom - amp_autodiff)
    logger.info("   Custom VJP amplitude: %s", amp_custom)
    logger.info("   Autodiff amplitude:   %s", amp_autodiff)
    logger.info("   Absolute error: %.6e", float(amp_error))

    # Test 3: Test with zip-up truncation strategy
    logger.info("\n3. Testing custom VJP with zip-up truncation...")
    truncate_bond_dimension_test = 4
    amp_fn_zipup = make_peps_amplitude(
        shape, ZipUp(truncate_bond_dimension=truncate_bond_dimension_test)
    )
    grad_zipup = None

    def amplitude_zipup_custom(flat_params):
        tensors_nested = flat_to_tensors(flat_params, tensors)
        return amp_fn_zipup(tensors_nested, sample)

    try:
        grad_zipup = jax.grad(amplitude_zipup_custom, holomorphic=True)(flat_params)
        amp_zipup = amplitude_zipup_custom(flat_params)
        logger.info("   Zip-up amplitude: %s", amp_zipup)
        logger.info(
            "   Zip-up gradient norm: %.6e", float(jnp.linalg.norm(grad_zipup))
        )
        logger.info("   PASS: Zip-up truncation works with custom VJP!")
    except Exception as exc:
        logger.warning("   FAIL: Zip-up error: %s", exc)

    # Test 4: Test with density matrix truncation strategy
    logger.info("\n4. Testing custom VJP with density matrix truncation...")
    amp_fn_dm = make_peps_amplitude(
        shape, DensityMatrix(truncate_bond_dimension=truncate_bond_dimension_test)
    )
    grad_dm = None

    def amplitude_dm_custom(flat_params):
        tensors_nested = flat_to_tensors(flat_params, tensors)
        return amp_fn_dm(tensors_nested, sample)

    try:
        grad_dm = jax.grad(amplitude_dm_custom, holomorphic=True)(flat_params)
        amp_dm = amplitude_dm_custom(flat_params)
        logger.info("   Density matrix amplitude: %s", amp_dm)
        logger.info(
            "   Density matrix gradient norm: %.6e", float(jnp.linalg.norm(grad_dm))
        )
        logger.info("   PASS: Density matrix truncation works with custom VJP!")
    except Exception as exc:
        logger.warning("   FAIL: Density matrix error: %s", exc)

    # Test 5: Compare truncation custom VJP gradients to exact no-truncation autodiff
    logger.info(
        "\n5. Comparing truncation gradients to no-truncation autodiff..."
    )
    for name, grad in (
        ("ZipUp", grad_zipup),
        ("DensityMatrix", grad_dm),
    ):
        if grad is None:
            logger.warning("   %s: skipped (gradient not available)", name)
            continue
        trunc_rel_error = jnp.linalg.norm(grad - grad_autodiff) / (
            jnp.linalg.norm(grad_autodiff) + 1e-10
        )
        logger.info(
            "   %s vs no-truncation rel diff: %.6e",
            name,
            float(trunc_rel_error),
        )

    logger.info("\n%s", "=" * 60)
    logger.info("Custom VJP tests complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    shape = (4, 4)
    bond_dim = 8
    truncate_bond_dimension = bond_dim * bond_dim

    sample = jnp.array(
        [1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=jnp.int32
    )

    rngs = nnx.Rngs(1)
    model = SimplePEPS(rngs=rngs, shape=shape, bond_dim=bond_dim)
    tensors = model.tensors

    def timed(name, fn):
        with timing.timed_scope(name=name, force=True) as t:
            out = fn()
            out = t.block_until_ready(out)
        return out, t.total

    amp_no_trunc, t_no = timed(
        "no_trunc",
        lambda: SimplePEPS._single_amplitude(
            tensors, sample, shape, NoTruncation()
        ),
    )
    amp_zip, t_zip = timed(
        "zip_up",
        lambda: SimplePEPS._single_amplitude(
            tensors,
            sample,
            shape,
            ZipUp(truncate_bond_dimension=truncate_bond_dimension),
        ),
    )
    amp_dm, t_dm = timed(
        "density_matrix",
        lambda: SimplePEPS._single_amplitude(
            tensors,
            sample,
            shape,
            DensityMatrix(truncate_bond_dimension=truncate_bond_dimension),
        ),
    )

    # Explicit apply-then-truncate path.
    spins = spin_to_occupancy(sample).reshape(shape)
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(shape[1])
    )

    def apply_then_truncate():
        boundary_local = boundary
        for row in range(shape[0]):
            mpo = SimplePEPS._build_row_mpo(tensors, spins[row], row, shape[1])
            boundary_local = SimplePEPS._apply_mpo(boundary_local, mpo)
            boundary_local = SimplePEPS._truncate_mps(
                boundary_local, truncate_bond_dimension
            )
        return SimplePEPS._contract_bottom(boundary_local)

    amp_apply_trunc, t_apply = timed("apply_truncate", apply_then_truncate)

    def _fmt(z):
        return f"{float(jnp.real(z)):+.6f}{float(jnp.imag(z)):+.6f}j"

    logger.info(
        "=== PEPS contraction consistency check (4x4, bond_dim=8, "
        "truncate_bond_dimension=D^2) ==="
    )
    logger.info("sample spins: %s", sample)
    logger.info("amp no trunc   : %s  (t=%.3es)", _fmt(amp_no_trunc), t_no)
    logger.info("amp zip-up     : %s  (t=%.3es)", _fmt(amp_zip), t_zip)
    logger.info("amp dens-mat   : %s  (t=%.3es)", _fmt(amp_dm), t_dm)
    logger.info("amp apply+trunc: %s  (t=%.3es)", _fmt(amp_apply_trunc), t_apply)
    logger.info("|zip - no|     : %.3e", float(jnp.abs(amp_zip - amp_no_trunc)))
    logger.info("|dm  - no|     : %.3e", float(jnp.abs(amp_dm - amp_no_trunc)))
    logger.info("|apply - no|   : %.3e", float(jnp.abs(amp_apply_trunc - amp_no_trunc)))

    # Run gradient correctness tests
    test_gradient_correctness()

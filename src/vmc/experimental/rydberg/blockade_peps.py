"""Blockade PEPS implementation for Rydberg atom simulation.

This implements a blockade-constrained PEPS using a directed gauge-canonical
formulation. The nearest-neighbor blockade constraint (n_i * n_j = 0) is
enforced by only parameterizing valid sector configurations.

Following the directed formulation:
- Outgoing sectors (kR, kD) = site occupation n
- Incoming sectors (kL, kU) must satisfy: if n=1, then kL=kU=0 (blockade)

Valid configurations at a bulk site:
| n   | kL  | kU  | kR  | kD  | cfg_idx |
| 0   | 0   | 0   | 0   | 0   | 0       |
| 0   | 0   | 1   | 0   | 0   | 1       |
| 0   | 1   | 0   | 0   | 0   | 2       |
| 0   | 1   | 1   | 0   | 0   | 3       |
| 1   | 0   | 0   | 1   | 1   | 0       |
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.models.peps import (
    ContractionStrategy,
    Variational,
    _apply_mpo_from_below,
    _contract_bottom,
    _compute_right_envs_2row,
    _metropolis_ratio,
    bottom_envs,
    grads_and_energy,
    sweep,
)
from vmc.operators.local_terms import DiagonalTerm, OneSiteTerm, bucket_terms
from vmc.utils.utils import random_tensor


@dataclass(frozen=True)
class BlockadePEPSConfig:
    """Configuration for Blockade PEPS."""

    shape: tuple[int, int]
    D0: int  # Degeneracy for sector k=0
    D1: int  # Degeneracy for sector k=1
    phys_dim: int = 2  # Must be 2 for blockade
    dtype: Any = jnp.complex128

    def __post_init__(self):
        if self.phys_dim != 2:
            raise ValueError("BlockadePEPS requires phys_dim=2")

    @property
    def Dmax(self) -> int:
        return max(self.D0, self.D1)


class BlockadePEPS(nnx.Module):
    """Blockade-constrained PEPS with Nc-sliced tensors (no masking).

    Tensor storage: (2, nc, Dmax, Dmax, Dmax, Dmax) where:
    - nc = 2^num_incoming (1, 2, or 4 depending on boundary)
    - Dmax = max(D0, D1) - blocks padded to uniform shape
    """

    tensors: list[list[nnx.Param]] = nnx.data()

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        config: BlockadePEPSConfig,
        contraction_strategy: ContractionStrategy | None = None,
    ) -> None:
        self.config = config
        self.shape = config.shape
        self.D0 = int(config.D0)
        self.D1 = int(config.D1)
        self.Dmax = config.Dmax
        self.phys_dim = int(config.phys_dim)
        self.dtype = config.dtype

        if contraction_strategy is None:
            contraction_strategy = Variational(
                truncate_bond_dimension=self.Dmax * self.Dmax
            )
        self.strategy = contraction_strategy

        n_rows, n_cols = self.shape
        tensors: list[list[nnx.Param]] = []
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                # Compute nc (number of valid configurations for n=0)
                # n=0: 2^num_incoming configs, n=1: always 1 config
                num_incoming = int(r > 0) + int(c > 0)
                nc = 2**num_incoming

                # Boundary-aware bond dims (using Dmax for uniform shape)
                mu_u = self.Dmax if r > 0 else 1
                mu_d = self.Dmax if r < n_rows - 1 else 1
                mu_l = self.Dmax if c > 0 else 1
                mu_r = self.Dmax if c < n_cols - 1 else 1

                tensor_val = random_tensor(
                    rngs,
                    (self.phys_dim, nc, mu_u, mu_d, mu_l, mu_r),
                    self.dtype,
                )
                row.append(nnx.Param(tensor_val, dtype=self.dtype))
            tensors.append(row)
        self.tensors = tensors

    @staticmethod
    def flatten_sample(config: jax.Array) -> jax.Array:
        """Flatten configuration to 1D array."""
        return config.reshape(-1)

    @staticmethod
    def unflatten_sample(sample: jax.Array, shape: tuple[int, int]) -> jax.Array:
        """Unflatten sample to (n_rows, n_cols) configuration."""
        return sample.reshape(shape)

    @staticmethod
    def apply(
        tensors: list[list[jax.Array]],
        sample: jax.Array,
        shape: tuple[int, int],
        config: BlockadePEPSConfig,
        strategy: ContractionStrategy,
    ) -> jax.Array:
        """Compute PEPS amplitude for a given configuration."""
        n_config = BlockadePEPS.unflatten_sample(sample, shape)
        eff_tensors = assemble_tensors(tensors, n_config, config)
        return _peps_apply_occupancy(eff_tensors, n_config, shape, strategy)

    def random_physical_configuration(
        self,
        key: jax.Array,
        n_samples: int = 1,
    ) -> jax.Array:
        """Generate random valid configurations (independent sets)."""
        keys = jax.random.split(key, n_samples)
        return jax.vmap(
            lambda k: random_independent_set(k, self.shape)
        )(keys)


# =============================================================================
# Tensor Assembly
# =============================================================================


def assemble_tensors(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
) -> list[list[jax.Array]]:
    """Assemble effective tensors for all sites given configuration."""
    n_rows, n_cols = peps_config.shape
    eff = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append(_assemble_site(tensors, config, peps_config, r, c))
        eff.append(row)
    return eff


def _assemble_site(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    r: int,
    c: int,
) -> jax.Array:
    """Assemble site tensor based on current configuration.

    For n=0: cfg_idx = kL * stride + kU (up to 4 configs)
    For n=1: cfg_idx = 0 (only valid config)
    """
    n_rows, n_cols = peps_config.shape
    n = config[r, c]
    kL = config[r, c - 1] if c > 0 else 0
    kU = config[r - 1, c] if r > 0 else 0

    # Compute cfg_idx
    stride = 2 if r > 0 else 1
    cfg_idx_n0 = kL * stride + kU
    cfg_idx = jnp.where(n == 0, cfg_idx_n0, 0)

    return tensors[r][c][:, cfg_idx, :, :, :, :]


def _assemble_site_with_n(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    r: int,
    c: int,
    n_new: jax.Array,
) -> jax.Array:
    """Assemble site tensor with overridden n value."""
    n_rows, n_cols = peps_config.shape
    kL = config[r, c - 1] if c > 0 else 0
    kU = config[r - 1, c] if r > 0 else 0

    stride = 2 if r > 0 else 1
    cfg_idx_n0 = kL * stride + kU
    cfg_idx = jnp.where(n_new == 0, cfg_idx_n0, 0)

    return tensors[r][c][:, cfg_idx, :, :, :, :]


def _assemble_site_with_kU(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    r: int,
    c: int,
    kU_new: jax.Array,
) -> jax.Array:
    """Assemble site tensor with overridden kU (from flipped neighbor above)."""
    n_rows, n_cols = peps_config.shape
    n = config[r, c]
    kL = config[r, c - 1] if c > 0 else 0

    stride = 2 if r > 0 else 1
    cfg_idx_n0 = kL * stride + kU_new
    cfg_idx = jnp.where(n == 0, cfg_idx_n0, 0)

    return tensors[r][c][:, cfg_idx, :, :, :, :]


def _assemble_site_with_kL(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    r: int,
    c: int,
    kL_new: jax.Array,
) -> jax.Array:
    """Assemble site tensor with overridden kL (from flipped neighbor to left)."""
    n_rows, n_cols = peps_config.shape
    n = config[r, c]
    kU = config[r - 1, c] if r > 0 else 0

    stride = 2 if r > 0 else 1
    cfg_idx_n0 = kL_new * stride + kU
    cfg_idx = jnp.where(n == 0, cfg_idx_n0, 0)

    return tensors[r][c][:, cfg_idx, :, :, :, :]


# =============================================================================
# Configuration utilities
# =============================================================================


@functools.partial(jax.jit, static_argnames=("n_rows", "n_cols", "r", "c"))
def can_flip_to_one(
    config: jax.Array, n_rows: int, n_cols: int, r: int, c: int
) -> jax.Array:
    """Check if flipping to n=1 at (r,c) violates blockade.

    Returns True if flip is allowed (no neighbors have n=1).
    """
    blocked = jnp.zeros((), dtype=jnp.bool_)
    if c > 0:
        blocked = blocked | (config[r, c - 1] == 1)
    if c < n_cols - 1:
        blocked = blocked | (config[r, c + 1] == 1)
    if r > 0:
        blocked = blocked | (config[r - 1, c] == 1)
    if r < n_rows - 1:
        blocked = blocked | (config[r + 1, c] == 1)
    return ~blocked


@functools.partial(jax.jit, static_argnames=("shape",))
def random_independent_set(key: jax.Array, shape: tuple[int, int]) -> jax.Array:
    """Generate a random valid independent set configuration.

    Uses a greedy sequential approach to ensure validity.
    """
    n_rows, n_cols = shape
    config = jnp.zeros(shape, dtype=jnp.int32)

    # Use fori_loop with static unrolling via helper functions
    def process_site(r: int, c: int, carry):
        key, config = carry
        key, flip_key = jax.random.split(key)
        can_flip = can_flip_to_one(config, n_rows, n_cols, r, c)
        do_flip = jax.random.bernoulli(flip_key) & can_flip
        config = config.at[r, c].set(jnp.where(do_flip, 1, 0))
        return key, config

    # Unroll the loop since we need static r, c values
    for r in range(n_rows):
        for c in range(n_cols):
            key, config = process_site(r, c, (key, config))

    return config


# =============================================================================
# PEPS contraction
# =============================================================================


def _build_row_mpo(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    row: int,
) -> tuple:
    """Build row-MPO for PEPS contraction."""
    n_cols = peps_config.shape[1]
    return tuple(
        jnp.transpose(
            _assemble_site(tensors, config, peps_config, row, c)[config[row, c]],
            (2, 3, 0, 1),
        )
        for c in range(n_cols)
    )


def _peps_apply_occupancy(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> jax.Array:
    """Compute PEPS amplitude for occupancy configuration."""
    n_rows, n_cols = shape
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(n_cols)
    )
    for row in range(n_rows):
        mpo = tuple(
            jnp.transpose(tensors[row][c][config[row, c]], (2, 3, 0, 1))
            for c in range(n_cols)
        )
        boundary = strategy.apply(boundary, mpo)
    return _contract_bottom(boundary)


# =============================================================================
# Dispatched API Functions
# =============================================================================


@bottom_envs.dispatch
def bottom_envs(model: BlockadePEPS, sample: jax.Array) -> list[tuple]:
    """Compute bottom boundary environments for BlockadePEPS.

    For 2-row sweep, bottom_envs[r] covers rows r+1 onwards.
    """
    config = BlockadePEPS.unflatten_sample(sample, model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    return _compute_bottom_envs(tensors, config, model.config, model.strategy)


def _compute_bottom_envs(
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    strategy: ContractionStrategy,
) -> list[tuple]:
    """Compute bottom boundary environments."""
    n_rows, n_cols = peps_config.shape
    dtype = tensors[0][0].dtype
    envs = [None] * n_rows
    env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    for row in range(n_rows - 1, -1, -1):
        envs[row] = env
        mpo = _build_row_mpo(tensors, config, peps_config, row)
        env = _apply_mpo_from_below(env, mpo, strategy)

    return envs


@grads_and_energy.dispatch
def grads_and_energy(
    model: BlockadePEPS,
    sample: jax.Array,
    amp: jax.Array,
    operator: Any,
    envs: list[tuple],
) -> tuple[list[list[jax.Array]], jax.Array]:
    """Compute environment gradients and local energy for BlockadePEPS.

    Diagonal terms: computed directly from configuration (no tensor operations)
    X terms: use 2-row sweep with tensor contractions
    """
    config = BlockadePEPS.unflatten_sample(sample, model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    peps_config = model.config
    strategy = model.strategy
    n_rows, n_cols = peps_config.shape
    dtype = tensors[0][0].dtype
    phys_dim = peps_config.phys_dim

    (
        diagonal_terms,
        one_site_terms,
        horizontal_terms,
        vertical_terms,
        plaquette_terms,
    ) = bucket_terms(operator.terms, peps_config.shape)

    env_grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    # 1. Diagonal energy - NO tensor operations!
    energy = jnp.zeros((), dtype=amp.dtype)
    for term in diagonal_terms:
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + config[row, col]
        energy = energy + term.diag[idx]

    # 2. Gradients and X term energy via row sweep
    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    mpo = _build_row_mpo(tensors, config, peps_config, 0)

    for row in range(n_rows):
        bottom_env = envs[row]
        right_envs = _compute_right_envs_1row(top_env, mpo, bottom_env, dtype)
        left_env = jnp.ones((1, 1, 1), dtype=dtype)

        eff_row = [
            _assemble_site(tensors, config, peps_config, row, c)
            for c in range(n_cols)
        ]

        for c in range(n_cols):
            # Compute gradient
            env_grad = _compute_single_gradient(
                left_env, right_envs[c], top_env[c], bottom_env[c]
            )
            env_grads[row][c] = env_grad

            # X term energy (sigma^x flips n)
            site_terms = one_site_terms[row][c]
            if site_terms:
                amps_site = jnp.einsum("pudlr,udlr->p", eff_row[c], env_grad)
                spin_idx = config[row, c]
                for term in site_terms:
                    energy = energy + jnp.dot(term.op[:, spin_idx], amps_site) / amp

            # Update left_env
            left_env = jnp.einsum(
                "ace,aub,cduv,evf->bdf",
                left_env, top_env[c], mpo[c], bottom_env[c],
                optimize=[(0, 1), (0, 2), (0, 1)],
            )

        top_env = strategy.apply(top_env, mpo)
        if row < n_rows - 1:
            mpo = _build_row_mpo(tensors, config, peps_config, row + 1)

    return env_grads, energy


def _compute_right_envs_1row(
    top_env: tuple,
    mpo_row: tuple,
    bottom_env: tuple,
    dtype,
) -> list[jax.Array]:
    """Compute right environments for 1-row contractions."""
    n_cols = len(mpo_row)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        right_envs[c] = jnp.einsum(
            "aub,cduv,evf,bdf->ace",
            top_env[c + 1], mpo_row[c + 1], bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 3), (0, 2), (0, 1)],
        )
    return right_envs


def _compute_single_gradient(
    left_env: jax.Array,
    right_env: jax.Array,
    top_tensor: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Compute gradient for a single tensor given left/right environments."""
    grad = jnp.einsum(
        "ace,aub,evf,bdf->cuvd", left_env, top_tensor, bot_tensor, right_env,
        optimize=[(0, 1), (0, 1), (0, 1)],
    )
    return jnp.transpose(grad, (1, 2, 0, 3))


@sweep.dispatch
def sweep(
    model: BlockadePEPS,
    sample: jax.Array,
    key: jax.Array,
    envs: list[tuple],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """2-row Metropolis sweep for BlockadePEPS.

    Uses overlapping row pairs (0,1), (1,2), ... with 2-column explicit window.
    """
    config = BlockadePEPS.unflatten_sample(sample, model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    peps_config = model.config
    strategy = model.strategy
    n_rows, n_cols = peps_config.shape
    dtype = tensors[0][0].dtype

    # Process row pairs
    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    if n_rows == 1:
        # Single row: standard 1-row sweep
        key, config = _sweep_single_row(
            key, tensors, config, peps_config, 0, top_env, envs[0]
        )
        mpo = _build_row_mpo(tensors, config, peps_config, 0)
        top_env = strategy.apply(top_env, mpo)
    else:
        # Multi-row: 2-row sweep over overlapping pairs
        # This sweeps rows 0, 1, ..., n_rows-2 (each row r is swept in pair (r, r+1))
        for r in range(n_rows - 1):
            bottom_env_pair = envs[r + 1] if r + 1 < n_rows else tuple(
                jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols)
            )
            key, config = _sweep_row_pair(
                key, tensors, config, peps_config, r, top_env, bottom_env_pair, strategy
            )
            # Update top_env with row r
            mpo_r = _build_row_mpo(tensors, config, peps_config, r)
            top_env = strategy.apply(top_env, mpo_r)

        # Sweep the last row (n_rows-1) with single-row sweep
        # This row wasn't swept in the pair loop above
        bottom_env_last = tuple(
            jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols)
        )
        key, config = _sweep_single_row(
            key, tensors, config, peps_config, n_rows - 1, top_env, bottom_env_last
        )

        # Contract final row to get amplitude
        mpo_last = _build_row_mpo(tensors, config, peps_config, n_rows - 1)
        top_env = strategy.apply(top_env, mpo_last)

    amp = _contract_bottom(top_env)
    return BlockadePEPS.flatten_sample(config), key, amp


def _sweep_single_row(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    r: int,
    top_env: tuple,
    bottom_env: tuple,
) -> tuple[jax.Array, jax.Array]:
    """Sweep a single row using 2-column window to handle kL dependency.

    Like 2-row sweep uses right_envs[c+2], single-row uses right_envs[c+1]
    (covering columns c+2 onwards) and explicitly includes column c+1 in window.
    """
    n_rows, n_cols = peps_config.shape
    dtype = tensors[0][0].dtype

    mpo = _build_row_mpo(tensors, config, peps_config, r)
    # right_envs[c] covers columns c+1, c+2, ... So right_envs[c+1] covers c+2 onwards
    right_envs = _compute_right_envs_1row(top_env, mpo, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1), dtype=dtype)

    for c in range(n_cols):
        n_cur = config[r, c]
        n_flip = 1 - n_cur

        # Check blockade constraint for flip to 1
        can_flip = jnp.where(
            n_flip == 1,
            can_flip_to_one(config, n_rows, n_cols, r, c),
            jnp.ones((), dtype=jnp.bool_),
        )

        mpo_c = mpo[c]

        if c + 1 < n_cols:
            # 2-column window: explicitly include c and c+1
            # right_envs[c+1] covers columns c+2 onwards (doesn't include c+1)
            mpo_c1 = mpo[c + 1]
            right_env = right_envs[c + 1] if c + 1 < n_cols - 1 else jnp.ones((1, 1, 1), dtype=dtype)

            # Current amplitude
            # Indices: left_env(ace), top[c](aub), mpo_c(cduv), bottom[c](evf),
            #          top[c+1](bpg), mpo_c1(dhpq), bottom[c+1](fqi), right_env(ghi)
            amp_cur = jnp.einsum(
                "ace,aub,cduv,evf,bpg,dhpq,fqi,ghi->",
                left_env, top_env[c], mpo_c, bottom_env[c],
                top_env[c + 1], mpo_c1, bottom_env[c + 1], right_env,
                optimize=[(0, 1), (0, 4), (0, 3), (0, 2), (2, 3), (0, 2), (0, 1)],
            )

            # Flipped tensors: (r,c) changes, (r,c+1) changes due to kL
            eff_flip = _assemble_site_with_n(tensors, config, peps_config, r, c, n_flip)
            eff_c1_flip = _assemble_site_with_kL(tensors, config, peps_config, r, c + 1, n_flip)
            mpo_c_flip = jnp.transpose(eff_flip[n_flip], (2, 3, 0, 1))
            mpo_c1_flip = jnp.transpose(eff_c1_flip[config[r, c + 1]], (2, 3, 0, 1))

            amp_flip = jnp.einsum(
                "ace,aub,cduv,evf,bpg,dhpq,fqi,ghi->",
                left_env, top_env[c], mpo_c_flip, bottom_env[c],
                top_env[c + 1], mpo_c1_flip, bottom_env[c + 1], right_env,
                optimize=[(0, 1), (0, 4), (0, 3), (0, 2), (2, 3), (0, 2), (0, 1)],
            )
        else:
            # Last column: single column window (no c+1 to worry about)
            right_env = jnp.ones((1, 1, 1), dtype=dtype)
            amp_cur = jnp.einsum(
                "ace,aub,cduv,evf,bdf->",
                left_env, top_env[c], mpo_c, bottom_env[c], right_env,
                optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
            )

            eff_flip = _assemble_site_with_n(tensors, config, peps_config, r, c, n_flip)
            mpo_c_flip = jnp.transpose(eff_flip[n_flip], (2, 3, 0, 1))
            amp_flip = jnp.einsum(
                "ace,aub,cduv,evf,bdf->",
                left_env, top_env[c], mpo_c_flip, bottom_env[c], right_env,
                optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
            )

        amp_flip = jnp.where(can_flip, amp_flip, jnp.zeros_like(amp_flip))

        # Metropolis
        ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_flip) ** 2)
        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

        config = config.at[r, c].set(jnp.where(accept, n_flip, n_cur))

        # Update mpo[c] for left_env update
        mpo_c_sel = jnp.where(accept, mpo_c_flip, mpo_c)
        mpo = tuple(mpo_c_sel if i == c else mpo[i] for i in range(n_cols))

        # Also update mpo[c+1] if we have a next column
        if c + 1 < n_cols:
            mpo_c1_sel = jnp.where(accept, mpo_c1_flip, mpo_c1)
            mpo = tuple(mpo_c1_sel if i == c + 1 else mpo[i] for i in range(n_cols))

        # Update left_env
        left_env = jnp.einsum(
            "ace,aub,cduv,evf->bdf",
            left_env, top_env[c], mpo[c], bottom_env[c],
            optimize=[(0, 1), (0, 2), (0, 1)],
        )

    return key, config


def _sweep_row_pair(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    config: jax.Array,
    peps_config: BlockadePEPSConfig,
    r: int,
    top_env: tuple,
    bottom_env: tuple,
    strategy: ContractionStrategy,
) -> tuple[jax.Array, jax.Array]:
    """Sweep row pair (r, r+1) using 2-column explicit window.

    Key insight: just track configuration n and assemble tensors on-demand.
    right_envs_2row[c+2] is always valid (doesn't include columns c or c+1).
    """
    n_rows, n_cols = peps_config.shape
    dtype = tensors[0][0].dtype

    # Build initial row MPOs
    mpo0 = _build_row_mpo(tensors, config, peps_config, r)
    mpo1 = _build_row_mpo(tensors, config, peps_config, r + 1)

    # Compute 2-row right envs ONCE at start (valid throughout sweep)
    right_envs = _compute_right_envs_2row(top_env, mpo0, mpo1, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)

    for c in range(n_cols):
        n_cur = config[r, c]
        n_flip = 1 - n_cur

        # Check blockade constraint
        can_flip = jnp.where(
            n_flip == 1,
            can_flip_to_one(config, n_rows, n_cols, r, c),
            jnp.ones((), dtype=jnp.bool_),
        )

        # Assemble current window tensors (on-demand from config)
        mpo0_c = mpo0[c]
        mpo1_c = mpo1[c]

        if c + 1 < n_cols:
            # 2-column window
            mpo0_c1 = mpo0[c + 1]
            mpo1_c1 = mpo1[c + 1]
            right_env = right_envs[c + 1] if c + 2 >= n_cols else right_envs[c + 1]

            # Current amplitude
            amp_cur = _contract_2row_2col(
                left_env, top_env, mpo0_c, mpo1_c, mpo0_c1, mpo1_c1,
                bottom_env, right_envs[c + 1], c
            )

            # Flipped tensors: 3 change (r,c), (r+1,c), (r,c+1)
            eff0_c_flip = _assemble_site_with_n(tensors, config, peps_config, r, c, n_flip)
            eff1_c_flip = _assemble_site_with_kU(tensors, config, peps_config, r + 1, c, n_flip)
            eff0_c1_flip = _assemble_site_with_kL(tensors, config, peps_config, r, c + 1, n_flip)

            mpo0_c_flip = jnp.transpose(eff0_c_flip[n_flip], (2, 3, 0, 1))
            mpo1_c_flip = jnp.transpose(eff1_c_flip[config[r + 1, c]], (2, 3, 0, 1))
            mpo0_c1_flip = jnp.transpose(eff0_c1_flip[config[r, c + 1]], (2, 3, 0, 1))

            amp_flip = _contract_2row_2col(
                left_env, top_env, mpo0_c_flip, mpo1_c_flip, mpo0_c1_flip, mpo1_c1,
                bottom_env, right_envs[c + 1], c
            )
        else:
            # Last column: 1-column window in 2-row context
            right_env_1col = jnp.ones((1, 1, 1, 1), dtype=dtype)
            amp_cur = _contract_2row_1col(
                left_env, top_env[c], mpo0_c, mpo1_c, bottom_env[c], right_env_1col
            )

            eff0_c_flip = _assemble_site_with_n(tensors, config, peps_config, r, c, n_flip)
            eff1_c_flip = _assemble_site_with_kU(tensors, config, peps_config, r + 1, c, n_flip)
            mpo0_c_flip = jnp.transpose(eff0_c_flip[n_flip], (2, 3, 0, 1))
            mpo1_c_flip = jnp.transpose(eff1_c_flip[config[r + 1, c]], (2, 3, 0, 1))

            amp_flip = _contract_2row_1col(
                left_env, top_env[c], mpo0_c_flip, mpo1_c_flip, bottom_env[c], right_env_1col
            )

        amp_flip = jnp.where(can_flip, amp_flip, jnp.zeros_like(amp_flip))

        # Metropolis
        ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_flip) ** 2)
        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

        # Update config and MPOs
        config = config.at[r, c].set(jnp.where(accept, n_flip, n_cur))

        mpo0_c_new = jnp.where(accept, mpo0_c_flip, mpo0_c)
        mpo1_c_new = jnp.where(accept, mpo1_c_flip, mpo1_c)

        # Update mpo0 and mpo1 lists
        mpo0 = tuple(mpo0_c_new if i == c else mpo0[i] for i in range(n_cols))
        mpo1 = tuple(mpo1_c_new if i == c else mpo1[i] for i in range(n_cols))

        # Also update c+1 MPOs if accept (kL changed)
        if c + 1 < n_cols:
            mpo0_c1_new = jnp.where(accept, mpo0_c1_flip, mpo0_c1)
            mpo0 = tuple(mpo0_c1_new if i == c + 1 else mpo0[i] for i in range(n_cols))

        # Update left_env (only includes column c)
        left_env = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf->bryf",
            left_env, top_env[c], mpo0[c], mpo1[c], bottom_env[c],
            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
        )

    return key, config


def _contract_2row_2col(
    left_env: jax.Array,
    top_env: tuple,
    mpo0_c: jax.Array,
    mpo1_c: jax.Array,
    mpo0_c1: jax.Array,
    mpo1_c1: jax.Array,
    bottom_env: tuple,
    right_env: jax.Array,
    c: int,
) -> jax.Array:
    """Contract 2-row, 2-column window for amplitude."""
    return jnp.einsum(
        "alxe,aub,lruv,xyvw,ewf,bgc,rsgh,ythi,fij,cstj->",
        left_env, top_env[c], mpo0_c, mpo1_c, bottom_env[c],
        top_env[c + 1], mpo0_c1, mpo1_c1, bottom_env[c + 1], right_env,
        optimize=[(1, 5), (3, 6), (1, 2), (1, 2), (0, 2), (2, 4), (1, 3), (0, 2), (0, 1)],
    )


def _contract_2row_1col(
    left_env: jax.Array,
    top: jax.Array,
    mpo0: jax.Array,
    mpo1: jax.Array,
    bottom: jax.Array,
    right_env: jax.Array,
) -> jax.Array:
    """Contract 2-row, 1-column for amplitude (last column)."""
    return jnp.einsum(
        "alxe,aub,lruv,xyvw,ewf,bryf->",
        left_env, top, mpo0, mpo1, bottom, right_env,
        optimize=[(0, 1), (0, 4), (1, 2), (1, 2), (0, 1)],
    )


# =============================================================================
# Dispatches for smallo helpers
# =============================================================================
from vmc.utils.smallo import params_per_site, sliced_dims


@params_per_site.dispatch
def _(model: BlockadePEPS) -> list[int]:
    """Number of parameters per active slice at each BlockadePEPS site."""
    n_rows, n_cols = model.shape
    return [
        int(jnp.asarray(model.tensors[r][c])[0, 0].size)
        for r in range(n_rows)
        for c in range(n_cols)
    ]


@sliced_dims.dispatch
def _(model: BlockadePEPS) -> tuple[int, ...]:
    """Number of distinct active slices per site (= phys_dim * nc)."""
    n_rows, n_cols = model.shape
    return tuple(
        model.phys_dim * jnp.asarray(model.tensors[r][c]).shape[1]
        for r in range(n_rows)
        for c in range(n_cols)
    )

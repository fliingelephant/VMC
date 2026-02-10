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
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.peps.blockade.compat import blockade_apply
from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _compute_right_envs,
    _contract_bottom,
    _metropolis_ratio,
)
from vmc.peps.common.energy import _compute_right_envs_2row, _compute_single_gradient
from vmc.peps.common.strategy import ContractionStrategy, Variational
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
    mask_per_charge: jax.Array | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.phys_dim != 2:
            raise ValueError("BlockadePEPS requires phys_dim=2")
        dmax = max(self.D0, self.D1)
        if self.D0 == dmax and self.D1 == dmax:
            object.__setattr__(self, "mask_per_charge", None)
            return
        deg = jnp.asarray((self.D0, self.D1), dtype=jnp.int32)
        mask = jnp.arange(dmax) < deg[:, None]
        object.__setattr__(self, "mask_per_charge", mask)

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

    apply = staticmethod(blockade_apply)

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


def _assemble_site(
    tensors: list[list[jax.Array]],
    peps_config: BlockadePEPSConfig,
    r: int,
    c: int,
    n: jax.Array,
    kL: jax.Array,
    kU: jax.Array,
) -> jax.Array:
    """Assemble site tensor based on current configuration.

    For n=0: cfg_idx = kL * stride + kU (up to 4 configs)
    For n=1: cfg_idx = 0 (only valid config)
    """
    # Compute cfg_idx
    stride = 2 if r > 0 else 1
    cfg_idx_n0 = kL * stride + kU
    cfg_idx = jnp.where(n == 0, cfg_idx_n0, 0)

    tensor = tensors[r][c][:, cfg_idx, :, :, :, :]
    mask_per_charge = peps_config.mask_per_charge
    if mask_per_charge is None:
        return tensor
    mask_u = mask_per_charge[kU][: tensor.shape[1]]
    tensor = tensor * mask_u[None, :, None, None, None]
    mask_d = mask_per_charge[n][: tensor.shape[2]]
    tensor = tensor * mask_d[None, None, :, None, None]
    mask_l = mask_per_charge[kL][: tensor.shape[3]]
    tensor = tensor * mask_l[None, None, None, :, None]
    mask_r = mask_per_charge[n][: tensor.shape[4]]
    return tensor * mask_r[None, None, None, None, :]


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
            _assemble_site(
                tensors,
                peps_config,
                row,
                c,
                config[row, c],
                config[row, c - 1] if c > 0 else 0,
                config[row - 1, c] if row > 0 else 0,
            )[config[row, c]],
            (2, 3, 0, 1),
        )
        for c in range(n_cols)
    )


def estimate(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    amp: jax.Array,
    operator: Any,
    shape: tuple[int, int],
    peps_config: BlockadePEPSConfig,
    strategy: ContractionStrategy,
    top_envs: list[tuple],
) -> tuple[list[list[jax.Array]], jax.Array, list[tuple]]:
    """Compute environment gradients and local energy for BlockadePEPS.

    Diagonal terms: computed directly from configuration (no tensor operations)
    X terms: use 2-row sweep with tensor contractions
    """
    config = BlockadePEPS.unflatten_sample(sample, shape)
    n_rows, n_cols = peps_config.shape
    dtype = tensors[0][0].dtype
    phys_dim = peps_config.phys_dim
    bottom_envs_cache = [None] * n_rows

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

    # 2. Gradients and X term energy with incremental bottom-env construction
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        bottom_envs_cache[row] = bottom_env
        top_env = top_envs[row]
        eff_row = [
            _assemble_site(
                tensors,
                peps_config,
                row,
                c,
                config[row, c],
                config[row, c - 1] if c > 0 else 0,
                config[row - 1, c] if row > 0 else 0,
            )
            for c in range(n_cols)
        ]
        mpo = tuple(
            jnp.transpose(eff_row[c][config[row, c]], (2, 3, 0, 1))
            for c in range(n_cols)
        )
        right_envs = _compute_right_envs(top_env, mpo, bottom_env, dtype)
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        # Boundary-aware branching keeps static shapes: 2-row window if row+1 exists.
        row_has_x = any(one_site_terms[row][c] for c in range(n_cols))
        mpo_next = None
        if row_has_x and row < n_rows - 1:
            bottom_env_pair = bottom_envs_cache[row + 1]
            eff_row_next = [
                _assemble_site(
                    tensors,
                    peps_config,
                    row + 1,
                    c,
                    config[row + 1, c],
                    config[row + 1, c - 1] if c > 0 else 0,
                    config[row, c],
                )
                for c in range(n_cols)
            ]
            mpo_next = tuple(
                jnp.transpose(eff_row_next[c][config[row + 1, c]], (2, 3, 0, 1))
                for c in range(n_cols)
            )
            right_envs_2row = _compute_right_envs_2row(
                top_env, mpo, mpo_next, bottom_env_pair, dtype
            )
            left_env_2row = jnp.ones((1, 1, 1, 1), dtype=dtype)

        for c in range(n_cols):
            # Compute gradient
            env_grad = _compute_single_gradient(
                left_env, right_envs[c], top_env[c], bottom_env[c]
            )
            env_grads[row][c] = env_grad

            # X term energy (sigma^x flips n)
            site_terms = one_site_terms[row][c]
            if site_terms:
                n_cur = config[row, c]
                n_flip = 1 - n_cur
                can_flip = jnp.where(
                    n_flip == 1,
                    can_flip_to_one(config, n_rows, n_cols, row, c),
                    jnp.ones((), dtype=jnp.bool_),
                )
                if row < n_rows - 1:
                    # 2-row contraction needs row+1 tensors (kU depends on flip).
                    if c + 1 < n_cols:
                        # 2-col window: include c+1 because kL depends on flip.
                        right_env = right_envs_2row[c + 1]

                        def _compute_flip(_):
                            eff0_c_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row,
                                c,
                                n_flip,
                                config[row, c - 1] if c > 0 else 0,
                                config[row - 1, c] if row > 0 else 0,
                            )
                            eff1_c_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row + 1,
                                c,
                                config[row + 1, c],
                                config[row + 1, c - 1] if c > 0 else 0,
                                n_flip,
                            )
                            eff0_c1_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row,
                                c + 1,
                                config[row, c + 1],
                                n_flip,
                                config[row - 1, c + 1] if row > 0 else 0,
                            )
                            mpo0_c_flip = jnp.transpose(eff0_c_flip[n_flip], (2, 3, 0, 1))
                            mpo1_c_flip = jnp.transpose(
                                eff1_c_flip[config[row + 1, c]], (2, 3, 0, 1)
                            )
                            mpo0_c1_flip = jnp.transpose(
                                eff0_c1_flip[config[row, c + 1]], (2, 3, 0, 1)
                            )
                            return _contract_2row_2col(
                                left_env_2row,
                                top_env,
                                mpo0_c_flip,
                                mpo1_c_flip,
                                mpo0_c1_flip,
                                mpo_next[c + 1],
                                bottom_env_pair,
                                right_env,
                                c,
                            )

                        amp_flip = jax.lax.cond(
                            can_flip,
                            _compute_flip,
                            lambda _: jnp.zeros((), dtype=amp.dtype),
                            operand=None,
                        )
                    else:
                        # Last column: no c+1, use 1-col window in 2-row shape.
                        def _compute_flip(_):
                            eff0_c_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row,
                                c,
                                n_flip,
                                config[row, c - 1] if c > 0 else 0,
                                config[row - 1, c] if row > 0 else 0,
                            )
                            eff1_c_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row + 1,
                                c,
                                config[row + 1, c],
                                config[row + 1, c - 1] if c > 0 else 0,
                                n_flip,
                            )
                            mpo0_c_flip = jnp.transpose(eff0_c_flip[n_flip], (2, 3, 0, 1))
                            mpo1_c_flip = jnp.transpose(
                                eff1_c_flip[config[row + 1, c]], (2, 3, 0, 1)
                            )
                            right_env_1col = jnp.ones((1, 1, 1, 1), dtype=dtype)
                            return _contract_2row_1col(
                                left_env_2row,
                                top_env[c],
                                mpo0_c_flip,
                                mpo1_c_flip,
                                bottom_env_pair[c],
                                right_env_1col,
                            )

                        amp_flip = jax.lax.cond(
                            can_flip,
                            _compute_flip,
                            lambda _: jnp.zeros((), dtype=amp.dtype),
                            operand=None,
                        )
                else:
                    # Last row: 1-row contraction (no row+1).
                    if c + 1 < n_cols:
                        # 2-col window: include c+1 because kL depends on flip.
                        right_env = (
                            right_envs[c + 1]
                            if c + 1 < n_cols - 1
                            else jnp.ones((1, 1, 1), dtype=dtype)
                        )

                        def _compute_flip(_):
                            eff_c_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row,
                                c,
                                n_flip,
                                config[row, c - 1] if c > 0 else 0,
                                config[row - 1, c] if row > 0 else 0,
                            )
                            eff_c1_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row,
                                c + 1,
                                config[row, c + 1],
                                n_flip,
                                config[row - 1, c + 1] if row > 0 else 0,
                            )
                            mpo_c_flip = jnp.transpose(eff_c_flip[n_flip], (2, 3, 0, 1))
                            mpo_c1_flip = jnp.transpose(
                                eff_c1_flip[config[row, c + 1]], (2, 3, 0, 1)
                            )
                            return jnp.einsum(
                                "ace,aub,cduv,evf,bpg,dhpq,fqi,ghi->",
                                left_env,
                                top_env[c],
                                mpo_c_flip,
                                bottom_env[c],
                                top_env[c + 1],
                                mpo_c1_flip,
                                bottom_env[c + 1],
                                right_env,
                                optimize=[(0, 1), (0, 4), (0, 3), (0, 2), (2, 3), (0, 2), (0, 1)],
                            )

                        amp_flip = jax.lax.cond(
                            can_flip,
                            _compute_flip,
                            lambda _: jnp.zeros((), dtype=amp.dtype),
                            operand=None,
                        )
                    else:
                        # Last column: 1-col window in 1-row shape.
                        def _compute_flip(_):
                            eff_c_flip = _assemble_site(
                                tensors,
                                peps_config,
                                row,
                                c,
                                n_flip,
                                config[row, c - 1] if c > 0 else 0,
                                config[row - 1, c] if row > 0 else 0,
                            )
                            mpo_c_flip = jnp.transpose(eff_c_flip[n_flip], (2, 3, 0, 1))
                            right_env = jnp.ones((1, 1, 1), dtype=dtype)
                            return jnp.einsum(
                                "ace,aub,cduv,evf,bdf->",
                                left_env,
                                top_env[c],
                                mpo_c_flip,
                                bottom_env[c],
                                right_env,
                                optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
                            )

                        amp_flip = jax.lax.cond(
                            can_flip,
                            _compute_flip,
                            lambda _: jnp.zeros((), dtype=amp.dtype),
                            operand=None,
                        )

                for term in site_terms:
                    energy = energy + term.op[n_flip, n_cur] * amp_flip / amp

            # Update left_env
            left_env = jnp.einsum(
                "ace,aub,cduv,evf->bdf",
                left_env, top_env[c], mpo[c], bottom_env[c],
                optimize=[(0, 1), (0, 2), (0, 1)],
            )
            if row_has_x and row < n_rows - 1:
                left_env_2row = jnp.einsum(
                    "alxe,aub,lruv,xyvw,ewf->bryf",
                    left_env_2row, top_env[c], mpo[c], mpo_next[c], bottom_env_pair[c],
                    optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
                )

        bottom_env = _apply_mpo_from_below(bottom_env, mpo, strategy)

    return env_grads, energy, bottom_envs_cache

def transition(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    key: jax.Array,
    envs: list[tuple],
    shape: tuple[int, int],
    peps_config: BlockadePEPSConfig,
    strategy: ContractionStrategy,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple]:
    """2-row Metropolis sweep for BlockadePEPS.

    Uses overlapping row pairs (0,1), (1,2), ... with 2-column explicit window.
    """
    config = BlockadePEPS.unflatten_sample(sample, shape)
    n_rows, n_cols = peps_config.shape
    dtype = tensors[0][0].dtype

    # Process row pairs
    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    top_envs_cache = [None] * n_rows

    if n_rows == 1:
        top_envs_cache[0] = top_env
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
            top_envs_cache[r] = top_env
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
        top_envs_cache[n_rows - 1] = top_env
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
    return BlockadePEPS.flatten_sample(config), key, amp, tuple(top_envs_cache)


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
    right_envs = _compute_right_envs(top_env, mpo, bottom_env, dtype)
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

            def _compute_flip(_):
                eff_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r,
                    c,
                    n_flip,
                    config[r, c - 1] if c > 0 else 0,
                    config[r - 1, c] if r > 0 else 0,
                )
                eff_c1_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r,
                    c + 1,
                    config[r, c + 1],
                    n_flip,
                    config[r - 1, c + 1] if r > 0 else 0,
                )
                mpo_c_flip = jnp.transpose(eff_flip[n_flip], (2, 3, 0, 1))
                mpo_c1_flip = jnp.transpose(
                    eff_c1_flip[config[r, c + 1]], (2, 3, 0, 1)
                )
                amp_flip = jnp.einsum(
                    "ace,aub,cduv,evf,bpg,dhpq,fqi,ghi->",
                    left_env, top_env[c], mpo_c_flip, bottom_env[c],
                    top_env[c + 1], mpo_c1_flip, bottom_env[c + 1], right_env,
                    optimize=[(0, 1), (0, 4), (0, 3), (0, 2), (2, 3), (0, 2), (0, 1)],
                )
                return amp_flip, mpo_c_flip, mpo_c1_flip

            def _no_flip(_):
                return jnp.zeros((), dtype=amp_cur.dtype), mpo_c, mpo_c1

            amp_flip, mpo_c_flip, mpo_c1_flip = jax.lax.cond(
                can_flip, _compute_flip, _no_flip, operand=None
            )
        else:
            # Last column: single column window (no c+1 to worry about)
            right_env = jnp.ones((1, 1, 1), dtype=dtype)
            amp_cur = jnp.einsum(
                "ace,aub,cduv,evf,bdf->",
                left_env, top_env[c], mpo_c, bottom_env[c], right_env,
                optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
            )

            def _compute_flip(_):
                eff_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r,
                    c,
                    n_flip,
                    config[r, c - 1] if c > 0 else 0,
                    config[r - 1, c] if r > 0 else 0,
                )
                mpo_c_flip = jnp.transpose(eff_flip[n_flip], (2, 3, 0, 1))
                amp_flip = jnp.einsum(
                    "ace,aub,cduv,evf,bdf->",
                    left_env, top_env[c], mpo_c_flip, bottom_env[c], right_env,
                    optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
                )
                return amp_flip, mpo_c_flip

            def _no_flip(_):
                return jnp.zeros((), dtype=amp_cur.dtype), mpo_c

            amp_flip, mpo_c_flip = jax.lax.cond(
                can_flip, _compute_flip, _no_flip, operand=None
            )

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

            # Current amplitude
            amp_cur = _contract_2row_2col(
                left_env, top_env, mpo0_c, mpo1_c, mpo0_c1, mpo1_c1,
                bottom_env, right_envs[c + 1], c
            )

            def _compute_flip(_):
                eff0_c_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r,
                    c,
                    n_flip,
                    config[r, c - 1] if c > 0 else 0,
                    config[r - 1, c] if r > 0 else 0,
                )
                eff1_c_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r + 1,
                    c,
                    config[r + 1, c],
                    config[r + 1, c - 1] if c > 0 else 0,
                    n_flip,
                )
                eff0_c1_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r,
                    c + 1,
                    config[r, c + 1],
                    n_flip,
                    config[r - 1, c + 1] if r > 0 else 0,
                )
                mpo0_c_flip = jnp.transpose(eff0_c_flip[n_flip], (2, 3, 0, 1))
                mpo1_c_flip = jnp.transpose(
                    eff1_c_flip[config[r + 1, c]], (2, 3, 0, 1)
                )
                mpo0_c1_flip = jnp.transpose(
                    eff0_c1_flip[config[r, c + 1]], (2, 3, 0, 1)
                )
                amp_flip = _contract_2row_2col(
                    left_env, top_env, mpo0_c_flip, mpo1_c_flip, mpo0_c1_flip, mpo1_c1,
                    bottom_env, right_envs[c + 1], c
                )
                return amp_flip, mpo0_c_flip, mpo1_c_flip, mpo0_c1_flip

            def _no_flip(_):
                amp_zero = jnp.zeros((), dtype=amp_cur.dtype)
                return amp_zero, mpo0_c, mpo1_c, mpo0_c1

            amp_flip, mpo0_c_flip, mpo1_c_flip, mpo0_c1_flip = jax.lax.cond(
                can_flip, _compute_flip, _no_flip, operand=None
            )
        else:
            # Last column: 1-column window in 2-row context
            right_env_1col = jnp.ones((1, 1, 1, 1), dtype=dtype)
            amp_cur = _contract_2row_1col(
                left_env, top_env[c], mpo0_c, mpo1_c, bottom_env[c], right_env_1col
            )

            def _compute_flip(_):
                eff0_c_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r,
                    c,
                    n_flip,
                    config[r, c - 1] if c > 0 else 0,
                    config[r - 1, c] if r > 0 else 0,
                )
                eff1_c_flip = _assemble_site(
                    tensors,
                    peps_config,
                    r + 1,
                    c,
                    config[r + 1, c],
                    config[r + 1, c - 1] if c > 0 else 0,
                    n_flip,
                )
                mpo0_c_flip = jnp.transpose(eff0_c_flip[n_flip], (2, 3, 0, 1))
                mpo1_c_flip = jnp.transpose(
                    eff1_c_flip[config[r + 1, c]], (2, 3, 0, 1)
                )
                amp_flip = _contract_2row_1col(
                    left_env, top_env[c], mpo0_c_flip, mpo1_c_flip, bottom_env[c], right_env_1col
                )
                return amp_flip, mpo0_c_flip, mpo1_c_flip

            def _no_flip(_):
                amp_zero = jnp.zeros((), dtype=amp_cur.dtype)
                return amp_zero, mpo0_c, mpo1_c

            amp_flip, mpo0_c_flip, mpo1_c_flip = jax.lax.cond(
                can_flip, _compute_flip, _no_flip, operand=None
            )

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

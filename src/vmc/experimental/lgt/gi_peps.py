"""Gauge-invariant PEPS (experimental, ZN matter + gauge).

This implements a gauge-invariant PEPS in a gauge-canonical-form style: gauge
degrees of freedom are represented by link configurations in the Monte Carlo
sample, and the PEPS tensors only store the matter (vertex) variational
parameters.

Following Wu & Liu (2025), each physical configuration selects a single charge
sector. We avoid a redundant "mask + slice" parameterization by storing only
the feasible local link configurations in an Nc axis and selecting entries by
slicing.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.models.peps import (
    _apply_mpo_from_below,
    _build_row_mpo,
    _compute_all_gradients,
    _compute_2site_horizontal_env,
    _compute_right_envs,
    _compute_right_envs_2row,
    _compute_row_pair_vertical_energy,
    _compute_single_gradient,
    _contract_bottom,
    _contract_column_transfer,
    _contract_column_transfer_2row,
    _contract_left_partial,
    _contract_left_partial_2row,
    _forward_with_cache,
    _metropolis_ratio,
    bottom_envs,
    grads_and_energy,
    sweep,
)
from vmc.operators.local_terms import bucket_terms
from vmc.utils.utils import random_tensor


@dataclass(frozen=True)
class GIPEPSConfig:
    """Configuration for GIPEPS."""

    shape: tuple[int, int]
    N: int
    phys_dim: int
    Qx: int
    degeneracy_per_charge: tuple[int, ...]
    charge_of_site: tuple[int, ...]
    dtype: Any = jnp.complex128

    @property
    def dmax(self) -> int:
        return int(max(self.degeneracy_per_charge))


class GIPEPS(nnx.Module):
    """Gauge-invariant PEPS with Nc-sliced tensors (no masking)."""

    tensors: list[list[nnx.Param]] = nnx.data()

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        config: GIPEPSConfig,
        contraction_strategy: Any,
    ) -> None:
        self.config = config
        self.shape = config.shape
        self.N = int(config.N)
        self.phys_dim = int(config.phys_dim)
        self.Qx = int(config.Qx)
        self.degeneracy_per_charge = tuple(int(d) for d in config.degeneracy_per_charge)
        self.charge_of_site = tuple(int(c) % self.N for c in config.charge_of_site)
        self.charge_to_indices, self.charge_deg = _build_charge_index_map(
            self.charge_of_site, self.N
        )
        # NOTE: charge_deg counts physical-state multiplicity per charge; it is
        # unrelated to degeneracy_per_charge (virtual bond-sector dimension).
        # TODO: rename charge_deg to avoid confusion with virtual degeneracy.
        if self.phys_dim > 1 and bool(jnp.any(self.charge_deg <= 0)):
            raise ValueError("charge_of_site must include all charges 0..N-1.")
        self.dmax = config.dmax
        self.dtype = config.dtype
        self.strategy = contraction_strategy

        n_rows, n_cols = self.shape
        tensors: list[list[nnx.Param]] = []
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                # Compute nc (number of gauge-invariant configurations)
                num_links = int(r > 0) + int(r < n_rows - 1) + int(c > 0) + int(c < n_cols - 1)
                nc = int(config.N ** max(num_links - 1, 0))
                # Compute boundary-aware bond dims
                mu_u = config.dmax if r > 0 else 1
                mu_d = config.dmax if r < n_rows - 1 else 1
                mu_l = config.dmax if c > 0 else 1
                mu_r = config.dmax if c < n_cols - 1 else 1
                tensor_val = random_tensor(
                    rngs,
                    (self.phys_dim, nc, mu_u, mu_d, mu_l, mu_r),
                    self.dtype,
                )
                row.append(nnx.Param(tensor_val, dtype=self.dtype))
            tensors.append(row)
        self.tensors = tensors

    @staticmethod
    def flatten_sample(
        sites: jax.Array,
        h_links: jax.Array,
        v_links: jax.Array,
    ) -> jax.Array:
        return jnp.concatenate(
            [sites.reshape(-1), h_links.reshape(-1), v_links.reshape(-1)], axis=0
        )

    @staticmethod
    def unflatten_sample(
        sample: jax.Array, shape: tuple[int, int]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        n_rows, n_cols = shape
        num_sites = n_rows * n_cols
        num_h = n_rows * (n_cols - 1)
        sites = sample[:num_sites].reshape((n_rows, n_cols))
        h_flat = sample[num_sites : num_sites + num_h]
        v_flat = sample[num_sites + num_h :]
        h_links = h_flat.reshape((n_rows, n_cols - 1))
        v_links = v_flat.reshape((n_rows - 1, n_cols))
        return sites, h_links, v_links

    @staticmethod
    def apply(
        tensors: list[list[jax.Array]],
        sample: jax.Array,
        shape: tuple[int, int],
        config: GIPEPSConfig,
        strategy: Any,
    ) -> jax.Array:
        sites, h_links, v_links = GIPEPS.unflatten_sample(sample, shape)
        eff_tensors = assemble_tensors(tensors, h_links, v_links, config)
        spins = sites.reshape(-1)
        return _peps_apply_occupancy(eff_tensors, spins, shape, strategy)

    def random_physical_configuration(
        self,
        key: jax.Array,
        n_samples: int = 1,
    ) -> jax.Array:
        n_rows, n_cols = self.shape
        keys = jax.random.split(key, n_samples)
        return jax.vmap(
            lambda k: self._single_physical_configuration(k, n_rows, n_cols)
        )(keys)

    @functools.partial(jax.jit, static_argnames=("n_rows", "n_cols"))
    def _single_physical_configuration(
        self, key: jax.Array, n_rows: int, n_cols: int
    ) -> jax.Array:
        h_links = jnp.zeros((n_rows, n_cols - 1), dtype=jnp.int32)
        v_links = jnp.zeros((n_rows - 1, n_cols), dtype=jnp.int32)
        key, site_key = jax.random.split(key)
        if n_rows > 1 and n_cols > 1:
            deltas = jax.random.randint(
                key, (n_rows - 1, n_cols - 1), 0, self.N, dtype=jnp.int32
            )
            h_links = h_links.at[: n_rows - 1, :].add(deltas)
            h_links = h_links.at[1:, :].add(-deltas)
            v_links = v_links.at[:, : n_cols - 1].add(-deltas)
            v_links = v_links.at[:, 1:].add(deltas)
            h_links = h_links % self.N
            v_links = v_links % self.N
        nl = jnp.pad(h_links, ((0, 0), (1, 0)), constant_values=0)
        nr = jnp.pad(h_links, ((0, 0), (0, 1)), constant_values=0)
        nu = jnp.pad(v_links, ((1, 0), (0, 0)), constant_values=0)
        nd = jnp.pad(v_links, ((0, 1), (0, 0)), constant_values=0)
        div = (nl + nu - nr - nd) % self.N
        charge = (self.Qx - div) % self.N
        keys = jax.random.split(site_key, n_rows * n_cols).reshape((n_rows, n_cols))
        sites = jax.vmap(
            lambda row_keys, row_charge: jax.vmap(
                _sample_site_index_for_charge, in_axes=(0, 0, None, None)
            )(row_keys, row_charge, self.charge_to_indices, self.charge_deg)
        )(keys, charge)
        return GIPEPS.flatten_sample(sites, h_links, v_links)


# ------------------------- helpers -------------------------


def _build_charge_index_map(
    charge_of_site: tuple[int, ...],
    N: int,
) -> tuple[jax.Array, jax.Array]:
    charges = jnp.asarray(charge_of_site, dtype=jnp.int32) % N
    charge_deg = jnp.bincount(charges, length=N)
    max_deg = int(jnp.max(charge_deg))
    padded = jnp.full((N, max_deg), -1, dtype=jnp.int32)

    def fill_charge(c, buf):
        idx = jnp.where(charges == c, size=max_deg, fill_value=-1)[0].astype(jnp.int32)
        return buf.at[c].set(idx)

    charge_to_indices = jax.lax.fori_loop(0, N, lambda c, buf: fill_charge(c, buf), padded)
    return charge_to_indices, charge_deg


def _sample_site_index_for_charge(
    key: jax.Array,
    charge: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> jax.Array:
    count = charge_deg[charge]
    k = jnp.floor(jax.random.uniform(key) * count).astype(jnp.int32)
    return charge_to_indices[charge, k]


@functools.partial(jax.jit, static_argnames=("direction",))
def _link_value_or_zero(
    h_links: jax.Array,
    v_links: jax.Array,
    r: int,
    c: int,
    *,
    direction: str,
) -> jax.Array:
    if direction == "left":
        return jax.lax.cond(
            c > 0,
            lambda _: h_links[r, c - 1],
            lambda _: jnp.zeros((), dtype=h_links.dtype),
            operand=None,
        )
    if direction == "right":
        return jax.lax.cond(
            c < h_links.shape[1],
            lambda _: h_links[r, c],
            lambda _: jnp.zeros((), dtype=h_links.dtype),
            operand=None,
        )
    if direction == "up":
        return jax.lax.cond(
            r > 0,
            lambda _: v_links[r - 1, c],
            lambda _: jnp.zeros((), dtype=v_links.dtype),
            operand=None,
        )
    if direction == "down":
        return jax.lax.cond(
            r < v_links.shape[0],
            lambda _: v_links[r, c],
            lambda _: jnp.zeros((), dtype=v_links.dtype),
            operand=None,
        )
    raise ValueError(f"Unknown direction: {direction}")


def assemble_tensors(
    tensors: list[list[jax.Array]],
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
) -> list[list[jax.Array]]:
    n_rows, n_cols = config.shape
    eff = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append(_assemble_site(tensors, h_links, v_links, config, r, c))
        eff.append(row)
    return eff


def _assemble_site(
    tensors: list[list[jax.Array]],
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    r: int,
    c: int,
) -> jax.Array:
    k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
    k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
    k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
    k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
    cfg_idx = _site_cfg_index(
        config, k_l=k_l, k_u=k_u, k_r=k_r, k_d=k_d, r=r, c=c
    )
    return tensors[r][c][:, cfg_idx, :, :, :, :]


def _site_cfg_index(
    config: GIPEPSConfig,
    *,
    k_l: jax.Array,
    k_u: jax.Array,
    k_r: jax.Array,
    k_d: jax.Array,
    r: int,
    c: int,
) -> jax.Array:
    """Map local link charges to a config index (Nc axis).

    For a physical configuration, Gauss law fixes one adjacent link value given
    the other links and the matter charge. The Nc axis stores only the feasible
    configurations (one per choice of the independent link charges).
    """
    n_rows, n_cols = config.shape
    active = {
        "left": c > 0,
        "right": c < n_cols - 1,
        "up": r > 0,
        "down": r < n_rows - 1,
    }
    dependent = None
    for direction in ("right", "down", "up", "left"):
        if active[direction]:
            dependent = direction
            break

    cfg_idx = jnp.zeros((), dtype=jnp.int32)
    for direction in ("left", "up", "down", "right"):
        if not active[direction] or direction == dependent:
            continue
        k = {"left": k_l, "up": k_u, "down": k_d, "right": k_r}[direction]
        cfg_idx = cfg_idx * jnp.asarray(config.N, dtype=jnp.int32) + k.astype(jnp.int32)
    return cfg_idx


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _peps_apply_occupancy(
    tensors: Any,
    sample: jax.Array,
    shape: tuple[int, int],
    strategy: Any,
) -> jax.Array:
    # TODO: refactor to use this as default _peps_apply (drop Netket compatibility).
    spins = sample.reshape(shape)
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(shape[1])
    )
    for row in range(shape[0]):
        mpo = _build_row_mpo(tensors, spins[row], row, shape[1])
        boundary = strategy.apply(boundary, mpo)
    return _contract_bottom(boundary)


def _peps_apply_occupancy_fwd(tensors, sample, shape, strategy):
    spins = sample.reshape(shape)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, strategy)
    return amp, (tensors, spins, top_envs)


def _peps_apply_occupancy_bwd(shape, strategy, residuals, g):
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


_peps_apply_occupancy.defvjp(_peps_apply_occupancy_fwd, _peps_apply_occupancy_bwd)


# =============================================================================
# GI-PEPS specific helpers
# =============================================================================


def _build_row_mpo_gi(
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    row: int,
    n_cols: int,
) -> tuple:
    """Build row-MPO for GI-PEPS contraction."""
    return tuple(
        jnp.transpose(
            _assemble_site(tensors, h_links, v_links, config, row, c)[sites[row, c]],
            (2, 3, 0, 1),
        )
        for c in range(n_cols)
    )


def _local_pair_amp(
    left_env: jax.Array,
    transfer0: jax.Array,
    transfer1: jax.Array,
    right_env: jax.Array,
) -> jax.Array:
    """Compute amplitude for a 2-column pair in 2-row contraction."""
    tmp = _contract_left_partial_2row(left_env, transfer0)
    tmp = _contract_left_partial_2row(tmp, transfer1)
    return jnp.einsum("aceg,aceg->", tmp, right_env)


def _plaquette_flip(
    h_links: jax.Array,
    v_links: jax.Array,
    r: int,
    c: int,
    *,
    delta: int,
    N: int,
) -> tuple[jax.Array, jax.Array]:
    """Flip plaquette at (r, c) by delta."""
    n = jnp.asarray(N, dtype=h_links.dtype)
    h_links = h_links.at[r, c].set((h_links[r, c] + delta) % n)
    h_links = h_links.at[r + 1, c].set((h_links[r + 1, c] - delta) % n)
    v_links = v_links.at[r, c].set((v_links[r, c] - delta) % n)
    v_links = v_links.at[r, c + 1].set((v_links[r, c + 1] + delta) % n)
    return h_links, v_links


def _updated_transfers_for_plaquette(
    tensors: list[list[jax.Array]],
    row_mpo0: tuple,
    row_mpo1: tuple,
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
) -> tuple[jax.Array, jax.Array, tuple, tuple]:
    """Update transfers for plaquette at (r, c) after link flip."""
    eff00 = _assemble_site(tensors, h_links, v_links, config, r, c)
    eff01 = _assemble_site(tensors, h_links, v_links, config, r, c + 1)
    eff10 = _assemble_site(tensors, h_links, v_links, config, r + 1, c)
    eff11 = _assemble_site(tensors, h_links, v_links, config, r + 1, c + 1)

    def update_mpo(mpo, col, eff, phys_idx):
        mpo_list = list(mpo)
        mpo_list[col] = jnp.transpose(eff[phys_idx], (2, 3, 0, 1))
        return tuple(mpo_list)

    row_mpo0 = update_mpo(row_mpo0, c, eff00, sites[r, c])
    row_mpo0 = update_mpo(row_mpo0, c + 1, eff01, sites[r, c + 1])
    row_mpo1 = update_mpo(row_mpo1, c, eff10, sites[r + 1, c])
    row_mpo1 = update_mpo(row_mpo1, c + 1, eff11, sites[r + 1, c + 1])

    transfer0 = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    transfer1 = _contract_column_transfer_2row(
        top_env[c + 1], row_mpo0[c + 1], row_mpo1[c + 1], bottom_env[c + 1]
    )
    return transfer0, transfer1, row_mpo0, row_mpo1


# =============================================================================
# Dispatched API Functions for GIPEPS
# =============================================================================


@bottom_envs.dispatch
def bottom_envs(model: GIPEPS, sample: jax.Array) -> list[tuple]:
    """Compute bottom boundary environments for GI-PEPS."""
    sites, h_links, v_links = GIPEPS.unflatten_sample(sample, model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    return _compute_bottom_envs(
        tensors, sites, h_links, v_links, model.config, model.strategy
    )


@grads_and_energy.dispatch
def grads_and_energy(
    model: GIPEPS,
    sample: jax.Array,
    amp: jax.Array,
    operator: Any,
    envs: list[tuple],
) -> tuple[list[list[jax.Array]], jax.Array]:
    """Compute environment gradients and local energy for GI-PEPS."""
    from vmc.experimental.lgt.gi_local_terms import LinkDiagonalTerm

    sites, h_links, v_links = GIPEPS.unflatten_sample(sample, model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    config = model.config
    strategy = model.strategy
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    phys_dim = config.phys_dim

    (
        diagonal_terms,
        one_site_terms,
        horizontal_terms,
        vertical_terms,
        plaquette_terms,
    ) = bucket_terms(operator.terms, config.shape)

    env_grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    bottom_envs_list = envs  # Use passed-in envs

    # Compute diagonal energy
    energy = jnp.zeros((), dtype=amp.dtype)
    for term in diagonal_terms:
        if isinstance(term, LinkDiagonalTerm):
            energy = energy + term.energy(h_links, v_links)
            continue
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + sites[row, col]
        energy = energy + term.diag[idx]

    # Main row iteration
    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    row_mpo = _build_row_mpo_gi(tensors, sites, h_links, v_links, config, 0, n_cols)
    for row in range(n_rows):
        bottom_env = bottom_envs_list[row]

        # Check what terms exist for this row
        row_has_one_site = any(one_site_terms[row][c] for c in range(n_cols))
        row_has_horizontal = any(horizontal_terms[row][c] for c in range(n_cols - 1))
        row_has_vertical = row < n_rows - 1 and any(
            vertical_terms[row][c] for c in range(n_cols)
        )
        row_has_plaquette = row < n_rows - 1 and any(
            plaquette_terms[row][c] for c in range(n_cols - 1)
        )

        # 1-row transfers and envs (always needed for gradients)
        transfers = [
            _contract_column_transfer(top_env[c], row_mpo[c], bottom_env[c])
            for c in range(n_cols)
        ]
        right_envs = _compute_right_envs(transfers, dtype)

        # Assemble effective tensors if needed
        eff_row = None
        eff_row_next = None
        if row_has_one_site or row_has_horizontal or row_has_vertical:
            eff_row = [
                _assemble_site(tensors, h_links, v_links, config, row, c)
                for c in range(n_cols)
            ]
        if row_has_vertical:
            eff_row_next = [
                _assemble_site(tensors, h_links, v_links, config, row + 1, c)
                for c in range(n_cols)
            ]

        # Column iteration for gradients + 1-row terms
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        for c in range(n_cols):
            # Gradient
            env_grad = _compute_single_gradient(
                left_env, right_envs[c], top_env[c], bottom_env[c]
            )
            env_grads[row][c] = env_grad

            # Single-site energy
            site_terms = one_site_terms[row][c]
            if site_terms:
                amps_site = jnp.einsum("pudlr,udlr->p", eff_row[c], env_grad)
                spin_idx = sites[row, c]
                for term in site_terms:
                    energy = energy + jnp.dot(term.op[:, spin_idx], amps_site) / amp

            # Horizontal energy
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
                        eff_row[c],
                        eff_row[c + 1],
                        env_2site,
                    )
                    spin0 = sites[row, c]
                    spin1 = sites[row, c + 1]
                    col_idx = spin0 * phys_dim + spin1
                    amps_flat = amps_edge.reshape(-1)
                    for term in edge_terms:
                        energy = energy + jnp.dot(term.op[:, col_idx], amps_flat) / amp

            left_env = _contract_left_partial(left_env, transfers[c])

        # 2-row terms (vertical + plaquette)
        if row < n_rows - 1:
            row_mpo_next = _build_row_mpo_gi(
                tensors, sites, h_links, v_links, config, row + 1, n_cols
            )
            if row_has_vertical or row_has_plaquette:
                bottom_env_next = bottom_envs_list[row + 1]
                transfers_2row = [
                    _contract_column_transfer_2row(
                        top_env[c], row_mpo[c], row_mpo_next[c], bottom_env_next[c]
                    )
                    for c in range(n_cols)
                ]
                right_envs_2row = _compute_right_envs_2row(
                    transfers_2row, transfers_2row[0].dtype
                )

                # Vertical energy
                if row_has_vertical:
                    energy = energy + _compute_row_pair_vertical_energy(
                        top_env,
                        bottom_env_next,
                        row_mpo,
                        row_mpo_next,
                        eff_row,
                        eff_row_next,
                        sites[row],
                        sites[row + 1],
                        vertical_terms[row],
                        amp,
                        phys_dim,
                        transfers_2row=transfers_2row,
                        right_envs_2row=right_envs_2row,
                    )

                # Plaquette energy
                if row_has_plaquette:
                    left_env_2row = jnp.ones((1, 1, 1, 1), dtype=transfers_2row[0].dtype)
                    for c in range(n_cols - 1):
                        plaquette_here = plaquette_terms[row][c]
                        if not plaquette_here:
                            left_env_2row = _contract_left_partial_2row(
                                left_env_2row, transfers_2row[c]
                            )
                            continue
                        amp_cur = _local_pair_amp(
                            left_env_2row,
                            transfers_2row[c],
                            transfers_2row[c + 1],
                            right_envs_2row[c + 1],
                        )
                        h_plus, v_plus = _plaquette_flip(
                            h_links, v_links, row, c, delta=1, N=config.N
                        )
                        trans0p, trans1p, _, _ = _updated_transfers_for_plaquette(
                            tensors,
                            row_mpo,
                            row_mpo_next,
                            h_plus,
                            v_plus,
                            sites,
                            config,
                            top_env,
                            bottom_env_next,
                            row,
                            c,
                        )
                        amp_plus = _local_pair_amp(
                            left_env_2row,
                            trans0p,
                            trans1p,
                            right_envs_2row[c + 1],
                        )
                        h_minus, v_minus = _plaquette_flip(
                            h_links, v_links, row, c, delta=-1, N=config.N
                        )
                        trans0m, trans1m, _, _ = _updated_transfers_for_plaquette(
                            tensors,
                            row_mpo,
                            row_mpo_next,
                            h_minus,
                            v_minus,
                            sites,
                            config,
                            top_env,
                            bottom_env_next,
                            row,
                            c,
                        )
                        amp_minus = _local_pair_amp(
                            left_env_2row,
                            trans0m,
                            trans1m,
                            right_envs_2row[c + 1],
                        )
                        if len(plaquette_here) == 1:
                            coeff = plaquette_here[0].coeff
                        else:
                            coeff = jnp.sum(
                                jnp.asarray([term.coeff for term in plaquette_here])
                            )
                        energy = energy + coeff * (amp_plus + amp_minus) / amp_cur
                        left_env_2row = _contract_left_partial_2row(
                            left_env_2row, transfers_2row[c]
                        )

        top_env = strategy.apply(top_env, row_mpo)
        if row < n_rows - 1:
            row_mpo = row_mpo_next

    return env_grads, energy


# =============================================================================
# GIPEPS Sweep helpers
# =============================================================================


def _update_row_mpo_for_site(
    row_mpo: tuple, col: int, tensor: jax.Array, phys_index: jax.Array
) -> tuple:
    """Update a single site in the row MPO."""
    row_list = list(row_mpo)
    row_list[col] = jnp.transpose(tensor[phys_index], (2, 3, 0, 1))
    return tuple(row_list)


def _updated_transfer_for_column(
    tensors: list[list[jax.Array]],
    row_mpo0: tuple,
    row_mpo1: tuple,
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
) -> tuple[jax.Array, tuple, tuple]:
    """Update 2-row transfer for a single column after link flip."""
    eff0 = _assemble_site(tensors, h_links, v_links, config, r, c)
    eff1 = _assemble_site(tensors, h_links, v_links, config, r + 1, c)
    row_mpo0 = _update_row_mpo_for_site(row_mpo0, c, eff0, sites[r, c])
    row_mpo1 = _update_row_mpo_for_site(row_mpo1, c, eff1, sites[r + 1, c])
    transfer = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    return transfer, row_mpo0, row_mpo1


def _plaquette_sweep_row_pair(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    row_mpo0: tuple,
    row_mpo1: tuple,
    r: int,
) -> tuple[jax.Array, tuple, tuple, jax.Array, jax.Array]:
    """Sweep plaquettes in a row pair."""
    n_cols = config.shape[1]
    transfers = [
        _contract_column_transfer_2row(
            top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
        )
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs_2row(transfers, transfers[0].dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=transfers[0].dtype)

    for c in range(n_cols - 1):
        key, subkey = jax.random.split(key)
        delta = jnp.where(jax.random.bernoulli(subkey), jnp.int32(1), jnp.int32(-1))

        left_partial = _contract_left_partial_2row(left_env, transfers[c])
        tmp = _contract_left_partial_2row(left_partial, transfers[c + 1])
        amp_cur = jnp.einsum("aceg,aceg->", tmp, right_envs[c + 1])
        h_prop, v_prop = _plaquette_flip(h_links, v_links, r, c, delta=delta, N=config.N)
        trans0, trans1, row_mpo0_prop, row_mpo1_prop = _updated_transfers_for_plaquette(
            tensors,
            row_mpo0,
            row_mpo1,
            h_prop,
            v_prop,
            sites,
            config,
            top_env,
            bottom_env,
            r,
            c,
        )
        left_partial_prop = _contract_left_partial_2row(left_env, trans0)
        tmp_prop = _contract_left_partial_2row(left_partial_prop, trans1)
        amp_prop = jnp.einsum("aceg,aceg->", tmp_prop, right_envs[c + 1])
        ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)

        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

        def accept_branch(_):
            updated_transfers = list(transfers)
            updated_transfers[c] = trans0
            updated_transfers[c + 1] = trans1
            return h_prop, v_prop, row_mpo0_prop, row_mpo1_prop, updated_transfers, left_partial_prop

        def reject_branch(_):
            return h_links, v_links, row_mpo0, row_mpo1, transfers, left_partial

        h_links, v_links, row_mpo0, row_mpo1, transfers, left_env = jax.lax.cond(
            accept, accept_branch, reject_branch, operand=None
        )

    return key, row_mpo0, row_mpo1, h_links, v_links


def _horizontal_link_sweep_row(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    row_mpo: tuple,
    r: int,
    charge_of_site: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> tuple[jax.Array, tuple, jax.Array, jax.Array]:
    """Sweep horizontal links in a single row."""
    n_cols = config.shape[1]
    n = jnp.asarray(config.N, dtype=jnp.int32)
    dtype = row_mpo[0].dtype
    transfers = [
        _contract_column_transfer(top_env[c], row_mpo[c], bottom_env[c])
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs(transfers, dtype)
    left_env = jnp.ones((1, 1, 1), dtype=dtype)

    for c in range(n_cols - 1):
        key, subkey = jax.random.split(key)
        delta = jnp.where(jax.random.bernoulli(subkey), jnp.int32(1), jnp.int32(-1))

        left_partial = _contract_left_partial(left_env, transfers[c])
        amp_cur = jnp.einsum(
            "ace,abcdef,bdf->", left_partial, transfers[c + 1], right_envs[c + 1]
        )

        h_prop = h_links.at[r, c].set((h_links[r, c] + delta) % n)
        q_left = charge_of_site[sites[r, c]]
        q_right = charge_of_site[sites[r, c + 1]]
        q_left_new = (q_left + delta) % n
        q_right_new = (q_right - delta) % n
        key, site_key = jax.random.split(key)
        key_left, key_right = jax.random.split(site_key)
        site_left = _sample_site_index_for_charge(
            key_left, q_left_new, charge_to_indices, charge_deg
        )
        site_right = _sample_site_index_for_charge(
            key_right, q_right_new, charge_to_indices, charge_deg
        )
        sites_prop = sites.at[r, c].set(site_left)
        sites_prop = sites_prop.at[r, c + 1].set(site_right)

        eff0 = _assemble_site(tensors, h_prop, v_links, config, r, c)
        eff1 = _assemble_site(tensors, h_prop, v_links, config, r, c + 1)
        mpo0 = jnp.transpose(eff0[sites_prop[r, c]], (2, 3, 0, 1))
        mpo1 = jnp.transpose(eff1[sites_prop[r, c + 1]], (2, 3, 0, 1))
        trans0 = _contract_column_transfer(top_env[c], mpo0, bottom_env[c])
        trans1 = _contract_column_transfer(top_env[c + 1], mpo1, bottom_env[c + 1])
        left_partial_prop = _contract_left_partial(left_env, trans0)
        amp_prop = jnp.einsum(
            "ace,abcdef,bdf->", left_partial_prop, trans1, right_envs[c + 1]
        )
        ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)
        prop_ratio = (charge_deg[q_left_new] * charge_deg[q_right_new]) / (
            charge_deg[q_left] * charge_deg[q_right]
        )
        ratio = ratio * prop_ratio
        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

        def accept_branch(_):
            updated_transfers = list(transfers)
            updated_transfers[c] = trans0
            updated_transfers[c + 1] = trans1
            row_mpo_next = _update_row_mpo_for_site(row_mpo, c, eff0, sites_prop[r, c])
            row_mpo_next = _update_row_mpo_for_site(
                row_mpo_next, c + 1, eff1, sites_prop[r, c + 1]
            )
            return h_prop, sites_prop, row_mpo_next, updated_transfers, left_partial_prop

        def reject_branch(_):
            return h_links, sites, row_mpo, transfers, left_partial

        h_links, sites, row_mpo, transfers, left_env = jax.lax.cond(
            accept, accept_branch, reject_branch, operand=None
        )

    return key, row_mpo, sites, h_links


def _vertical_link_sweep_row_pair(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    row_mpo0: tuple,
    row_mpo1: tuple,
    r: int,
    charge_of_site: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> tuple[jax.Array, tuple, tuple, jax.Array, jax.Array]:
    """Sweep vertical links in a row pair."""
    n_cols = config.shape[1]
    n = jnp.asarray(config.N, dtype=jnp.int32)
    transfers = [
        _contract_column_transfer_2row(
            top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
        )
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs_2row(transfers, transfers[0].dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=transfers[0].dtype)

    for c in range(n_cols):
        key, subkey = jax.random.split(key)
        delta = jnp.where(jax.random.bernoulli(subkey), jnp.int32(1), jnp.int32(-1))

        left_partial = _contract_left_partial_2row(left_env, transfers[c])
        amp_cur = jnp.einsum("aceg,aceg->", left_partial, right_envs[c])
        v_prop = v_links.at[r, c].set((v_links[r, c] + delta) % n)
        q_top = charge_of_site[sites[r, c]]
        q_bottom = charge_of_site[sites[r + 1, c]]
        q_top_new = (q_top + delta) % n
        q_bottom_new = (q_bottom - delta) % n
        key, site_key = jax.random.split(key)
        key_top, key_bottom = jax.random.split(site_key)
        site_top = _sample_site_index_for_charge(
            key_top, q_top_new, charge_to_indices, charge_deg
        )
        site_bottom = _sample_site_index_for_charge(
            key_bottom, q_bottom_new, charge_to_indices, charge_deg
        )
        sites_prop = sites.at[r, c].set(site_top)
        sites_prop = sites_prop.at[r + 1, c].set(site_bottom)
        trans, row_mpo0_prop, row_mpo1_prop = _updated_transfer_for_column(
            tensors,
            row_mpo0,
            row_mpo1,
            h_links,
            v_prop,
            sites_prop,
            config,
            top_env,
            bottom_env,
            r,
            c,
        )
        left_partial_prop = _contract_left_partial_2row(left_env, trans)
        amp_prop = jnp.einsum("aceg,aceg->", left_partial_prop, right_envs[c])
        ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)
        prop_ratio = (charge_deg[q_top_new] * charge_deg[q_bottom_new]) / (
            charge_deg[q_top] * charge_deg[q_bottom]
        )
        ratio = ratio * prop_ratio
        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

        def accept_branch(_):
            updated_transfers = list(transfers)
            updated_transfers[c] = trans
            return (
                v_prop,
                sites_prop,
                row_mpo0_prop,
                row_mpo1_prop,
                updated_transfers,
                left_partial_prop,
            )

        def reject_branch(_):
            return v_links, sites, row_mpo0, row_mpo1, transfers, left_partial

        v_links, sites, row_mpo0, row_mpo1, transfers, left_env = jax.lax.cond(
            accept, accept_branch, reject_branch, operand=None
        )

    return key, row_mpo0, row_mpo1, sites, v_links


def _compute_bottom_envs(
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    strategy: Any,
) -> list[tuple]:
    """Compute bottom boundary environments (internal helper for sweep)."""
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    envs = [None] * n_rows
    env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        envs[row] = env
        row_mpo = _build_row_mpo_gi(tensors, sites, h_links, v_links, config, row, n_cols)
        env = _apply_mpo_from_below(env, row_mpo, strategy)
    return envs


@sweep.dispatch
def sweep(
    model: GIPEPS,
    sample: jax.Array,
    key: jax.Array,
    envs: list[tuple],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Combined plaquette + link sweeps for GI-PEPS."""
    sites, h_links, v_links = GIPEPS.unflatten_sample(sample, model.shape)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    config = model.config
    strategy = model.strategy
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype

    charge_of_site = jnp.asarray(model.charge_of_site, dtype=jnp.int32)
    charge_to_indices = model.charge_to_indices
    charge_deg = model.charge_deg

    # 1. Plaquette sweep over the full lattice (row pairs)
    top_env_plaquettes = None
    if n_rows > 1:
        top_env_plaquettes = tuple(
            jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols)
        )
        row_mpo0 = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, 0, n_cols
        )
        row_mpo1 = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, 1, n_cols
        )
        for r in range(n_rows - 1):
            key, row_mpo0, row_mpo1, h_links, v_links = _plaquette_sweep_row_pair(
                key,
                tensors,
                sites,
                h_links,
                v_links,
                config,
                top_env_plaquettes,
                envs[r + 1],
                row_mpo0,
                row_mpo1,
                r,
            )
            top_env_plaquettes = strategy.apply(top_env_plaquettes, row_mpo0)
            if r + 2 < n_rows:
                row_mpo0 = row_mpo1
                row_mpo1 = _build_row_mpo_gi(
                    tensors, sites, h_links, v_links, config, r + 2, n_cols
                )
        top_env_plaquettes = strategy.apply(top_env_plaquettes, row_mpo1)

    # For pure gauge (phys_dim == 1), no link/matter sweeps needed
    if config.phys_dim == 1:
        if top_env_plaquettes is None:
            top_env_plaquettes = tuple(
                jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols)
            )
            for row in range(n_rows):
                row_mpo = _build_row_mpo_gi(
                    tensors, sites, h_links, v_links, config, row, n_cols
                )
                top_env_plaquettes = strategy.apply(top_env_plaquettes, row_mpo)
        amp = _contract_bottom(top_env_plaquettes)
        return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp

    # 2. Horizontal link sweeps for all rows
    # Recompute bottom_envs after plaquette changes
    bottom_envs_h = _compute_bottom_envs(
        tensors, sites, h_links, v_links, config, strategy
    )

    top_env_h = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows):
        row_mpo = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, row, n_cols
        )
        key, row_mpo, sites, h_links = _horizontal_link_sweep_row(
            key,
            tensors,
            sites,
            h_links,
            v_links,
            config,
            top_env_h,
            bottom_envs_h[row],
            row_mpo,
            row,
            charge_of_site,
            charge_to_indices,
            charge_deg,
        )
        top_env_h = strategy.apply(top_env_h, row_mpo)

    if n_rows == 1:
        amp = _contract_bottom(top_env_h)
        return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp

    # 3. Vertical link sweeps for all row pairs
    # Recompute bottom_envs after horizontal changes
    bottom_envs_v = _compute_bottom_envs(
        tensors, sites, h_links, v_links, config, strategy
    )

    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    row_mpo0 = _build_row_mpo_gi(tensors, sites, h_links, v_links, config, 0, n_cols)
    row_mpo1 = _build_row_mpo_gi(tensors, sites, h_links, v_links, config, 1, n_cols)
    for r in range(n_rows - 1):
        key, row_mpo0, row_mpo1, sites, v_links = _vertical_link_sweep_row_pair(
            key,
            tensors,
            sites,
            h_links,
            v_links,
            config,
            top_env,
            bottom_envs_v[r + 1],
            row_mpo0,
            row_mpo1,
            r,
            charge_of_site,
            charge_to_indices,
            charge_deg,
        )
        top_env = strategy.apply(top_env, row_mpo0)
        if r + 2 < n_rows:
            row_mpo0 = row_mpo1
            row_mpo1 = _build_row_mpo_gi(
                tensors, sites, h_links, v_links, config, r + 2, n_cols
            )
    top_env = strategy.apply(top_env, row_mpo1)

    amp = _contract_bottom(top_env)
    return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp

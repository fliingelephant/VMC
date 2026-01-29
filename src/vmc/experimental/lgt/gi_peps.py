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
    _build_row_mpo,
    _compute_all_gradients,
    _contract_bottom,
    _forward_with_cache,
)
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

        tensors: list[list[nnx.Param]] = []
        for r in range(self.shape[0]):
            row = []
            for c in range(self.shape[1]):
                nc = _site_nc(self.config, r, c)
                mu_u, mu_d, mu_l, mu_r = _site_mu_dims(self.config, r, c)
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


def _site_mu_dims(config: GIPEPSConfig, r: int, c: int) -> tuple[int, int, int, int]:
    n_rows, n_cols = config.shape
    mu_u = config.dmax if r > 0 else 1
    mu_d = config.dmax if r < n_rows - 1 else 1
    mu_l = config.dmax if c > 0 else 1
    mu_r = config.dmax if c < n_cols - 1 else 1
    return mu_u, mu_d, mu_l, mu_r


def _site_nc(config: GIPEPSConfig, r: int, c: int) -> int:
    n_rows, n_cols = config.shape
    num_links = int(r > 0) + int(r < n_rows - 1) + int(c > 0) + int(c < n_cols - 1)
    return int(config.N ** max(num_links - 1, 0))

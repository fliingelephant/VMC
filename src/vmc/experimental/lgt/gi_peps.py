"""Gauge-invariant PEPS (experimental, ZN matter + gauge)."""
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
    """Gauge-invariant PEPS with charge-axis tensors."""

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
                ku_dim, mu_u_dim, kd_dim, md_dim, kl_dim, ml_dim, kr_dim, mr_dim = _site_dims(
                    self.config, r, c
                )
                shape = (
                    self.phys_dim,
                    ku_dim,
                    mu_u_dim,
                    kd_dim,
                    md_dim,
                    kl_dim,
                    ml_dim,
                    kr_dim,
                    mr_dim,
                )
                tensor_val = random_tensor(rngs, shape, self.dtype)
                tensor_val = _apply_gauss_mask(
                    tensor_val,
                    self.N,
                    self.Qx,
                    self.degeneracy_per_charge,
                    self.charge_of_site,
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
        idx = jnp.where(charges == c, size=max_deg, fill_value=-1)[0]
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


@functools.partial(
    jax.jit,
    static_argnames=("N", "Qx", "degeneracy_per_charge", "charge_of_site"),
)
def _apply_gauss_mask(
    tensor: jax.Array,
    N: int,
    Qx: int,
    degeneracy_per_charge: tuple[int, ...],
    charge_of_site: tuple[int, ...],
) -> jax.Array:
    phys_dim, ku_dim, mu_u_dim, kd_dim, md_dim, kl_dim, ml_dim, kr_dim, mr_dim = tensor.shape
    dmax = int(max(degeneracy_per_charge))

    def k_vals(dim: int) -> jax.Array:
        return jnp.arange(dim, dtype=jnp.int32) if dim > 1 else jnp.zeros((1,), jnp.int32)

    ku_vals, kd_vals, kl_vals, kr_vals = k_vals(ku_dim), k_vals(kd_dim), k_vals(kl_dim), k_vals(kr_dim)
    ku, kd, kl, kr = jnp.meshgrid(ku_vals, kd_vals, kl_vals, kr_vals, indexing="ij")
    charge = jnp.asarray(charge_of_site, dtype=jnp.int32) % N
    gauss = (kl + ku - kr - kd + charge[:, None, None, None, None] - Qx) % N
    base = (gauss == 0).astype(tensor.dtype)

    mu_mask = jnp.asarray([jnp.arange(dmax) < d for d in degeneracy_per_charge], dtype=tensor.dtype)

    def get_mu_mask(k_vals: jax.Array, mu_dim: int, k_dim: int) -> jax.Array:
        return jnp.ones((k_dim, 1), dtype=tensor.dtype) if mu_dim == 1 else mu_mask[k_vals]

    mu_u = get_mu_mask(ku_vals, mu_u_dim, ku_dim)[None, :, :, None, None, None, None, None, None]
    mu_d = get_mu_mask(kd_vals, md_dim, kd_dim)[None, None, None, :, :, None, None, None, None]
    mu_l = get_mu_mask(kl_vals, ml_dim, kl_dim)[None, None, None, None, None, :, :, None, None]
    mu_r = get_mu_mask(kr_vals, mr_dim, kr_dim)[None, None, None, None, None, None, None, :, :]

    return tensor * base[:, :, None, :, None, :, None, :, None] * mu_u * mu_d * mu_l * mu_r


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


@functools.partial(jax.jit, static_argnames=("config",))
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
            k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
            k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
            k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
            k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
            row.append(tensors[r][c][:, k_u, :, k_d, :, k_l, :, k_r, :])
        eff.append(row)
    return eff


def _assemble_site(
    tensors: list[list[jax.Array]],
    h_links: jax.Array,
    v_links: jax.Array,
    r: int,
    c: int,
) -> jax.Array:
    k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
    k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
    k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
    k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
    return tensors[r][c][:, k_u, :, k_d, :, k_l, :, k_r, :]


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


def _site_dims(config: GIPEPSConfig, r: int, c: int) -> tuple[int, int, int, int, int, int, int, int]:
    n_rows, n_cols = config.shape
    ku_dim = config.N if r > 0 else 1
    kd_dim = config.N if r < n_rows - 1 else 1
    kl_dim = config.N if c > 0 else 1
    kr_dim = config.N if c < n_cols - 1 else 1
    mu_u_dim = config.dmax if r > 0 else 1
    md_dim = config.dmax if r < n_rows - 1 else 1
    ml_dim = config.dmax if c > 0 else 1
    mr_dim = config.dmax if c < n_cols - 1 else 1
    return ku_dim, mu_u_dim, kd_dim, md_dim, kl_dim, ml_dim, kr_dim, mr_dim

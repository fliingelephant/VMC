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

from dataclasses import dataclass, field
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _compute_right_envs,
    _contract_2row_1col,
    _contract_2row_2col,
    _contract_bottom,
)
from vmc.peps.common.energy import (
    RowEnvs,
    TwoRowEnvs,
    _eval_term,
    _compute_right_envs_2row,
    _compute_single_gradient,
    _update_left_env_1row,
    _update_left_env_2row,
)
from vmc.peps.gi.compat import gi_apply
from vmc.operators.local_terms import (
    BucketedOperators,
    PlaquetteOperator,
    VerticalTwoSiteOperator,
    support_span,
)
from vmc.peps.gi.local_terms import (
    HorizontalHiggsLinkTerm,
    HorizontalMatterHoppingTerm,
    LinkDiagonalTerm,
    VerticalHiggsLinkTerm,
    VerticalMatterHoppingTerm,
)
from vmc.utils.utils import random_tensor, _hastings_ratio, _metropolis_hastings_accept


@dataclass(frozen=True)
class GIPEPSConfig:
    """Configuration for GIPEPS."""

    shape: tuple[int, int]
    N: int
    phys_dim: int
    Qx: Any
    degeneracy_per_charge: tuple[int, ...]
    charge_of_site: tuple[int, ...]
    particle_number: int | None = None
    dtype: Any = jnp.complex128
    conserve_particle_number: bool = field(default=True, kw_only=True)
    mask_per_charge: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        qx = np.asarray(self.Qx, dtype=np.int32) % self.N
        if qx.ndim == 0:
            qx = np.full(self.shape, qx, dtype=np.int32)
        elif tuple(qx.shape) != self.shape:
            raise ValueError(f"Qx must be a scalar or have shape {self.shape}.")
        object.__setattr__(
            self,
            "Qx",
            tuple(tuple(int(q) for q in row) for row in qx.tolist()),
        )

        if self.is_binary_occupancy_matter:
            if self.conserve_particle_number:
                if self.particle_number is None:
                    raise ValueError(
                        "Binary Z2 matter with number-conserving updates requires particle_number."
                    )
                n_sites = self.shape[0] * self.shape[1]
                if self.particle_number < 0 or self.particle_number > n_sites:
                    raise ValueError("particle_number must satisfy 0 <= particle_number <= n_sites.")
                required_parity = int(np.sum(qx) % 2)
                if self.particle_number % 2 != required_parity:
                    raise ValueError(
                        "particle_number parity must match sum(Qx) mod 2 for Z2 open-boundary GIPEPS."
                    )
            elif self.particle_number is not None:
                raise ValueError(
                    "Binary Z2 matter with parity-only updates must not set particle_number."
                )
        elif self.particle_number is not None:
            raise NotImplementedError(
                "Fixed particle number is currently supported only for Z2 hard-core matter."
            )

        dmax = int(max(self.degeneracy_per_charge))
        if all(d == dmax for d in self.degeneracy_per_charge):
            object.__setattr__(self, "mask_per_charge", None)
            return
        mask = tuple(
            tuple(i < d for i in range(dmax))
            for d in self.degeneracy_per_charge
        )
        object.__setattr__(self, "mask_per_charge", mask)

    @property
    def dmax(self) -> int:
        return int(max(self.degeneracy_per_charge))

    @property
    def is_binary_occupancy_matter(self) -> bool:
        return (
            self.N == 2
            and self.phys_dim == 2
            and self.charge_of_site == (0, 1)
        )


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
        self.Qx = config.Qx
        self.particle_number = (
            None if config.particle_number is None else int(config.particle_number)
        )
        self.degeneracy_per_charge = tuple(int(d) for d in config.degeneracy_per_charge)
        self.charge_of_site = tuple(int(c) % self.N for c in config.charge_of_site)
        self.charge_to_indices, self.charge_deg = _build_charge_index_map(
            self.charge_of_site, self.N
        )
        # NOTE: charge_deg counts physical-state multiplicity per charge; it is
        # unrelated to degeneracy_per_charge (virtual bond-sector dimension).
        # TODO: rename charge_deg to avoid confusion with virtual degeneracy.
        if self.phys_dim > 1 and any(d <= 0 for d in self.charge_deg):
            raise ValueError("charge_of_site must include all charges 0..N-1.")
        self.dmax = config.dmax
        self.dtype = config.dtype
        self.strategy = contraction_strategy

        n_rows, n_cols = self.shape
        tensors: list[list[nnx.Param]] = []
        params_per_site: list[int] = []
        sliced_dims: list[int] = []
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
                params_per_site.append(mu_u * mu_d * mu_l * mu_r)
                sliced_dims.append(self.phys_dim * nc)
            tensors.append(row)
        self.tensors = tensors
        self.params_per_site = tuple(params_per_site)
        self.sliced_dims = tuple(sliced_dims)

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

    apply = staticmethod(gi_apply)
    eval_span = staticmethod(support_span)
    def random_physical_configuration(
        self,
        key: jax.Array,
        n_samples: int = 1,
    ) -> jax.Array:
        keys = jax.random.split(key, n_samples)
        if self.config.is_binary_occupancy_matter and self.config.conserve_particle_number:
            return jax.vmap(
                lambda k: _single_z2_hardcore_configuration_with_particles(
                    k,
                    self.Qx,
                    self.particle_number,
                )
            )(keys)
        if self.config.is_binary_occupancy_matter:
            return jax.vmap(
                lambda k: _single_z2_hardcore_configuration_with_parity(
                    k,
                    self.Qx,
                )
            )(keys)
        return jax.vmap(
            lambda k: _single_physical_configuration(
                k,
                self.shape[0],
                self.shape[1],
                self.N,
                self.Qx,
                self.charge_to_indices,
                self.charge_deg,
            )
        )(keys)


# ------------------------- helpers -------------------------


def _build_charge_index_map(
    charge_of_site: tuple[int, ...],
    N: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    charges = tuple(int(charge) % N for charge in charge_of_site)
    charge_deg = tuple(sum(charge == c for charge in charges) for c in range(N))
    max_deg = max(charge_deg, default=0)
    charge_to_indices = tuple(
        tuple(
            [idx for idx, charge in enumerate(charges) if charge == c]
            + [-1] * (max_deg - charge_deg[c])
        )
        for c in range(N)
    )
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


def _random_plaquette_background(
    key: jax.Array,
    n_rows: int,
    n_cols: int,
    N: int,
) -> tuple[jax.Array, jax.Array]:
    h_links = jnp.zeros((n_rows, n_cols - 1), dtype=jnp.int32)
    v_links = jnp.zeros((n_rows - 1, n_cols), dtype=jnp.int32)
    if n_rows <= 1 or n_cols <= 1:
        return h_links, v_links
    deltas = jax.random.randint(
        key, (n_rows - 1, n_cols - 1), 0, N, dtype=jnp.int32
    )
    h_links = h_links.at[: n_rows - 1, :].add(deltas)
    h_links = h_links.at[1:, :].add(-deltas)
    v_links = v_links.at[:, : n_cols - 1].add(deltas)
    v_links = v_links.at[:, 1:].add(-deltas)
    return h_links % N, v_links % N


def _flip_path_masks(
    shape: tuple[int, int],
    start_flat: jax.Array,
    end_flat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    n_rows, n_cols = shape
    r0, c0 = start_flat // n_cols, start_flat % n_cols
    r1, c1 = end_flat // n_cols, end_flat % n_cols
    c_min, c_max = jnp.minimum(c0, c1), jnp.maximum(c0, c1)
    r_min, r_max = jnp.minimum(r0, r1), jnp.maximum(r0, r1)
    h_mask = (
        (jnp.arange(n_rows)[:, None] == r0)
        & (jnp.arange(n_cols - 1)[None, :] >= c_min)
        & (jnp.arange(n_cols - 1)[None, :] < c_max)
    )
    v_mask = (
        (jnp.arange(n_rows - 1)[:, None] >= r_min)
        & (jnp.arange(n_rows - 1)[:, None] < r_max)
        & (jnp.arange(n_cols)[None, :] == c1)
    )
    return h_mask.astype(jnp.int32), v_mask.astype(jnp.int32)


def _single_z2_hardcore_configuration_with_particles(
    key: jax.Array,
    Qx: Any,
    particle_number: int,
) -> jax.Array:
    Qx = jnp.asarray(Qx, dtype=jnp.int32)
    n_rows, n_cols = Qx.shape
    n_sites = n_rows * n_cols
    n_pairs_max = (n_sites + 1) // 2
    key_sites, key_bg, key_pair = jax.random.split(key, 3)
    permutation = jax.random.permutation(key_sites, n_sites)
    occupied = permutation[:particle_number]
    sites = jnp.zeros((n_sites,), dtype=jnp.int32).at[occupied].set(1).reshape((n_rows, n_cols))
    h_links, v_links = _random_plaquette_background(key_bg, n_rows, n_cols, 2)
    h_links, v_links = _repair_z2_gauss_law(key_pair, h_links, v_links, sites, Qx)
    return GIPEPS.flatten_sample(sites, h_links, v_links)


def _single_z2_hardcore_configuration_with_parity(
    key: jax.Array,
    Qx: Any,
) -> jax.Array:
    Qx = jnp.asarray(Qx, dtype=jnp.int32)
    n_rows, n_cols = Qx.shape
    n_sites = n_rows * n_cols
    key_sites, key_bg, key_pair = jax.random.split(key, 3)
    sites_flat = jax.random.randint(key_sites, (n_sites - 1,), 0, 2, dtype=jnp.int32)
    required_parity = jnp.sum(Qx) % 2
    last_site = (required_parity - jnp.sum(sites_flat)) % 2
    sites = jnp.concatenate([sites_flat, last_site[None]], axis=0).reshape((n_rows, n_cols))
    h_links, v_links = _random_plaquette_background(key_bg, n_rows, n_cols, 2)
    h_links, v_links = _repair_z2_gauss_law(key_pair, h_links, v_links, sites, Qx)
    return GIPEPS.flatten_sample(sites, h_links, v_links)


def _repair_z2_gauss_law(
    key: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    Qx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    n_rows, n_cols = Qx.shape
    n_sites = n_rows * n_cols
    n_pairs_max = (n_sites + 1) // 2
    defects = (Qx - sites) % 2
    defect_flat = jnp.where(
        defects.reshape(-1) == 1,
        size=2 * n_pairs_max,
        fill_value=-1,
    )[0]
    priorities = jax.random.uniform(key, (2 * n_pairs_max,))
    order = jnp.argsort(jnp.where(defect_flat >= 0, priorities, 2.0))
    pairs = defect_flat[order].reshape(n_pairs_max, 2)

    def flip_pair(carry, pair):
        h_cur, v_cur = carry
        valid = jnp.all(pair >= 0)

        def apply_pair(_):
            h_mask, v_mask = _flip_path_masks((n_rows, n_cols), pair[0], pair[1])
            return (
                jnp.bitwise_xor(h_cur, h_mask),
                jnp.bitwise_xor(v_cur, v_mask),
            )

        return jax.lax.cond(valid, apply_pair, lambda _: (h_cur, v_cur), operand=None), None

    return jax.lax.scan(flip_pair, (h_links, v_links), pairs)[0]


@functools.partial(
    jax.jit,
    static_argnames=("n_rows", "n_cols", "N", "Qx", "charge_to_indices", "charge_deg"),
)
def _single_physical_configuration(
    key: jax.Array,
    n_rows: int,
    n_cols: int,
    N: int,
    Qx: Any,
    charge_to_indices: Any,
    charge_deg: Any,
) -> jax.Array:
    Qx = jnp.asarray(Qx, dtype=jnp.int32)
    charge_to_indices = jnp.asarray(charge_to_indices, dtype=jnp.int32)
    charge_deg = jnp.asarray(charge_deg, dtype=jnp.int32)
    field_key, site_key = jax.random.split(key)
    h_links, v_links = _random_plaquette_background(field_key, n_rows, n_cols, N)
    nl = jnp.pad(h_links, ((0, 0), (1, 0)), constant_values=0)
    nr = jnp.pad(h_links, ((0, 0), (0, 1)), constant_values=0)
    nu = jnp.pad(v_links, ((1, 0), (0, 0)), constant_values=0)
    nd = jnp.pad(v_links, ((0, 1), (0, 0)), constant_values=0)
    div = (nl + nd - nu - nr) % N
    charge = (Qx - div) % N
    keys = jax.random.split(site_key, n_rows * n_cols).reshape((n_rows, n_cols))
    sites = jax.vmap(
        lambda row_keys, row_charge: jax.vmap(
            _sample_site_index_for_charge, in_axes=(0, 0, None, None)
        )(row_keys, row_charge, charge_to_indices, charge_deg)
    )(keys, charge)
    return GIPEPS.flatten_sample(sites, h_links, v_links)


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
        if h_links.shape[1] == 0:
            return jnp.zeros((), dtype=h_links.dtype)
        return jax.lax.cond(
            c > 0,
            lambda _: h_links[r, c - 1],
            lambda _: jnp.zeros((), dtype=h_links.dtype),
            operand=None,
        )
    if direction == "right":
        if h_links.shape[1] == 0:
            return jnp.zeros((), dtype=h_links.dtype)
        return jax.lax.cond(
            c < h_links.shape[1],
            lambda _: h_links[r, c],
            lambda _: jnp.zeros((), dtype=h_links.dtype),
            operand=None,
        )
    if direction == "up":
        if v_links.shape[0] == 0:
            return jnp.zeros((), dtype=v_links.dtype)
        return jax.lax.cond(
            r > 0,
            lambda _: v_links[r - 1, c],
            lambda _: jnp.zeros((), dtype=v_links.dtype),
            operand=None,
        )
    if direction == "down":
        if v_links.shape[0] == 0:
            return jnp.zeros((), dtype=v_links.dtype)
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
    mask_per_charge: jax.Array | None = None,
) -> list[list[jax.Array]]:
    n_rows, n_cols = config.shape
    eff = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append(_assemble_site(tensors, h_links, v_links, config, r, c, mask_per_charge))
        eff.append(row)
    return eff


def _assemble_site(
    tensors: list[list[jax.Array]],
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    r: int,
    c: int,
    mask_per_charge: jax.Array | None = None,
) -> jax.Array:
    k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
    k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
    k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
    k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
    cfg_idx = _site_cfg_index(
        config, k_l=k_l, k_u=k_u, k_r=k_r, k_d=k_d, r=r, c=c
    )
    tensor = tensors[r][c][:, cfg_idx, :, :, :, :]
    if mask_per_charge is None:
        if config.mask_per_charge is None:
            return tensor
        mask_per_charge = jnp.asarray(config.mask_per_charge, dtype=tensor.dtype)
    if mask_per_charge is None:
        return tensor
    mask_u = mask_per_charge[k_u][: tensor.shape[1]]
    mask_d = mask_per_charge[k_d][: tensor.shape[2]]
    mask_l = mask_per_charge[k_l][: tensor.shape[3]]
    mask_r = mask_per_charge[k_r][: tensor.shape[4]]
    return (
        tensor
        * mask_u[None, :, None, None, None]
        * mask_d[None, None, :, None, None]
        * mask_l[None, None, None, :, None]
        * mask_r[None, None, None, None, :]
    )


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
    mask_per_charge: jax.Array | None = None,
) -> tuple:
    """Build row-MPO for GI-PEPS contraction."""
    return tuple(
        jnp.transpose(
            _assemble_site(
                tensors,
                h_links,
                v_links,
                config,
                row,
                c,
                mask_per_charge,
            )[sites[row, c]],
            (2, 3, 0, 1),
        )
        for c in range(n_cols)
    )


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
    v_links = v_links.at[r, c].set((v_links[r, c] + delta) % n)
    v_links = v_links.at[r, c + 1].set((v_links[r, c + 1] - delta) % n)
    return h_links, v_links


def _horizontal_hardcore_hop(
    sites: jax.Array,
    h_links: jax.Array,
    r: int,
    c: int,
) -> tuple[jax.Array, jax.Array]:
    allowed = sites[r, c] != sites[r, c + 1]
    left = jnp.where(allowed, sites[r, c + 1], sites[r, c])
    right = jnp.where(allowed, sites[r, c], sites[r, c + 1])
    sites_prop = sites.at[r, c].set(left)
    sites_prop = sites_prop.at[r, c + 1].set(right)
    h_prop = h_links.at[r, c].set(
        jnp.where(allowed, 1 - h_links[r, c], h_links[r, c])
    )
    return sites_prop, h_prop


def _horizontal_higgs_link_flip(
    sites: jax.Array,
    h_links: jax.Array,
    r: int,
    c: int,
) -> tuple[jax.Array, jax.Array]:
    sites_prop = sites.at[r, c].set(1 - sites[r, c])
    sites_prop = sites_prop.at[r, c + 1].set(1 - sites[r, c + 1])
    h_prop = h_links.at[r, c].set(1 - h_links[r, c])
    return sites_prop, h_prop


def _vertical_hardcore_hop(
    sites: jax.Array,
    v_links: jax.Array,
    r: int,
    c: int,
) -> tuple[jax.Array, jax.Array]:
    allowed = sites[r, c] != sites[r + 1, c]
    top = jnp.where(allowed, sites[r + 1, c], sites[r, c])
    bottom = jnp.where(allowed, sites[r, c], sites[r + 1, c])
    sites_prop = sites.at[r, c].set(top)
    sites_prop = sites_prop.at[r + 1, c].set(bottom)
    v_prop = v_links.at[r, c].set(
        jnp.where(allowed, 1 - v_links[r, c], v_links[r, c])
    )
    return sites_prop, v_prop


def _vertical_higgs_link_flip(
    sites: jax.Array,
    v_links: jax.Array,
    r: int,
    c: int,
) -> tuple[jax.Array, jax.Array]:
    sites_prop = sites.at[r, c].set(1 - sites[r, c])
    sites_prop = sites_prop.at[r + 1, c].set(1 - sites[r + 1, c])
    v_prop = v_links.at[r, c].set(1 - v_links[r, c])
    return sites_prop, v_prop


def _horizontal_link_transition_amplitude(
    envs: RowEnvs,
    tensors: Any,
    sites_prop: jax.Array,
    h_prop: jax.Array,
    row: int,
    col: int,
) -> jax.Array:
    eff0 = _assemble_site(
        tensors,
        h_prop,
        envs.config.v_links,
        envs.config.peps_config,
        row,
        col,
        envs.config.mask_per_charge,
    )
    eff1 = _assemble_site(
        tensors,
        h_prop,
        envs.config.v_links,
        envs.config.peps_config,
        row,
        col + 1,
        envs.config.mask_per_charge,
    )
    mpo0 = jnp.transpose(eff0[sites_prop[row, col]], (2, 3, 0, 1))
    mpo1 = jnp.transpose(eff1[sites_prop[row, col + 1]], (2, 3, 0, 1))
    return jnp.einsum(
        "ace,aub,cduv,evf,bgh,digw,fwj,hij->",
        envs.left_env,
        envs.top_env[col],
        mpo0,
        envs.bottom_env[col],
        envs.top_env[col + 1],
        mpo1,
        envs.bottom_env[col + 1],
        envs.right_envs[col + 1],
        optimize=[(0, 1), (0, 6), (0, 5), (0, 3), (1, 2), (1, 2), (0, 1)],
    )


def _vertical_link_transition_amplitude(
    envs: GITwoRowEnvs,
    tensors: Any,
    sites_prop: jax.Array,
    v_prop: jax.Array,
    row: int,
    col: int,
) -> jax.Array:
    eff0 = _assemble_site(
        tensors,
        envs.h_links,
        v_prop,
        envs.config,
        row,
        col,
        envs.mask_per_charge,
    )
    eff1 = _assemble_site(
        tensors,
        envs.h_links,
        v_prop,
        envs.config,
        row + 1,
        col,
        envs.mask_per_charge,
    )
    mpo0 = jnp.transpose(eff0[sites_prop[row, col]], (2, 3, 0, 1))
    mpo1 = jnp.transpose(eff1[sites_prop[row + 1, col]], (2, 3, 0, 1))
    return _contract_2row_1col(
        envs.left_env,
        envs.top_env[col],
        mpo0,
        mpo1,
        envs.bottom_env_next[col],
        envs.right_envs[col],
    )


# =============================================================================
# Runtime Internals for GIPEPS Kernels
# =============================================================================


class GITwoRowEnvs(NamedTuple):
    """2-row environment context for GI dr=2 evaluation (Vertical + Plaquette)."""
    left_env: jax.Array
    right_envs: list
    top_env: tuple
    bottom_env_next: tuple
    row_tensors: list
    row_tensors_next: list
    h_links: jax.Array
    v_links: jax.Array
    config: Any
    mask_per_charge: jax.Array | None = None


class GIRowEnvConfig(NamedTuple):
    """Row-local GI context for dr=1 evaluation."""

    h_links: jax.Array
    v_links: jax.Array
    peps_config: Any
    mask_per_charge: jax.Array | None = None


@_eval_term.dispatch
def _eval_term(
    term: VerticalTwoSiteOperator,
    envs: GITwoRowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    amps = jnp.einsum(
        "almg,aub,puvlr,qvwmn,gwf,brnf->pq",
        envs.left_env,
        envs.top_env[col],
        envs.row_tensors[col],
        envs.row_tensors_next[col],
        envs.bottom_env_next[col],
        envs.right_envs[col],
        optimize=[(0, 1), (2, 3), (0, 2), (1, 2), (0, 1)],
    )
    s0, s1 = spins[row, col], spins[row + 1, col]
    return jnp.dot(term.op[:, s0 * phys_dim + s1], amps.reshape(-1))


@_eval_term.dispatch
def _eval_term(
    term: HorizontalMatterHoppingTerm,
    envs: RowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    del phys_dim
    allowed = spins[row, col] != spins[row, col + 1]

    def eval_hop(_):
        sites_prop, h_prop = _horizontal_hardcore_hop(spins, envs.config.h_links, row, col)
        return _horizontal_link_transition_amplitude(
            envs,
            tensors,
            sites_prop,
            h_prop,
            row,
            col,
        )

    return jax.lax.cond(
        allowed,
        eval_hop,
        lambda _: jnp.zeros((), dtype=envs.left_env.dtype),
        operand=None,
    )


@_eval_term.dispatch
def _eval_term(
    term: HorizontalHiggsLinkTerm,
    envs: RowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    del phys_dim
    sites_prop, h_prop = _horizontal_higgs_link_flip(
        spins, envs.config.h_links, row, col
    )
    return _horizontal_link_transition_amplitude(
        envs,
        tensors,
        sites_prop,
        h_prop,
        row,
        col,
    )


@_eval_term.dispatch
def _eval_term(
    term: PlaquetteOperator,
    envs: GITwoRowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    def _flip_amp(delta):
        h_f, v_f = _plaquette_flip(envs.h_links, envs.v_links, row, col, delta=delta, N=envs.config.N)
        mpos = []
        for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
            eff = _assemble_site(
                tensors,
                h_f,
                v_f,
                envs.config,
                row + dr,
                col + dc,
                envs.mask_per_charge,
            )
            mpos.append(jnp.transpose(eff[spins[row + dr, col + dc]], (2, 3, 0, 1)))
        return _contract_2row_2col(
            envs.left_env, envs.top_env, mpos[0], mpos[2],
            mpos[1], mpos[3], envs.bottom_env_next, envs.right_envs[col + 1], col,
        )
    return _flip_amp(1) + _flip_amp(-1)


@_eval_term.dispatch
def _eval_term(
    term: VerticalMatterHoppingTerm,
    envs: GITwoRowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    del phys_dim
    allowed = spins[row, col] != spins[row + 1, col]

    def eval_hop(_):
        sites_prop, v_prop = _vertical_hardcore_hop(spins, envs.v_links, row, col)
        return _vertical_link_transition_amplitude(
            envs,
            tensors,
            sites_prop,
            v_prop,
            row,
            col,
        )

    return jax.lax.cond(
        allowed,
        eval_hop,
        lambda _: jnp.zeros((), dtype=envs.left_env.dtype),
        operand=None,
    )


@_eval_term.dispatch
def _eval_term(
    term: VerticalHiggsLinkTerm,
    envs: GITwoRowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    del phys_dim
    sites_prop, v_prop = _vertical_higgs_link_flip(spins, envs.v_links, row, col)
    return _vertical_link_transition_amplitude(
        envs,
        tensors,
        sites_prop,
        v_prop,
        row,
        col,
    )


def estimate(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    amp: jax.Array,
    config: GIPEPSConfig,
    strategy: Any,
    top_envs: list[tuple],
    mask_per_charge: jax.Array | None = None,
    *,
    terms: BucketedOperators,
    coeffs: jax.Array | None = None,
) -> tuple[list[list[jax.Array]], jax.Array, list[tuple]]:
    """Compute environment gradients and local energy for GI-PEPS."""
    sites, h_links, v_links = GIPEPS.unflatten_sample(sample, config.shape)
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    phys_dim = config.phys_dim
    bottom_envs_cache = [None] * n_rows

    env_grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    dr1_direct_terms = (
        HorizontalMatterHoppingTerm,
        HorizontalHiggsLinkTerm,
    )
    dr2_direct_terms = (
        PlaquetteOperator,
        VerticalMatterHoppingTerm,
        VerticalHiggsLinkTerm,
    )

    # Compute diagonal energy
    energies = jnp.zeros(len(terms), dtype=amp.dtype)
    for term, contributions in terms.diagonal:
        if isinstance(term, LinkDiagonalTerm):
            diag_val = term.energy(h_links, v_links)
        else:
            idx = jnp.asarray(0, dtype=jnp.int32)
            for row, col in term.sites:
                idx = idx * phys_dim + sites[row, col]
            diag_val = term.diag[idx]
        for op_idx, coeff_idx in contributions:
            coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
            energies = energies.at[op_idx].add(coeff * diag_val)

    # Main row iteration
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    empty_cols = tuple(() for _ in range(n_cols))
    next_row_mpo = None
    for row in range(n_rows - 1, -1, -1):
        bottom_envs_cache[row] = bottom_env
        top_env = top_envs[row]
        row_mpo = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, row, n_cols, mask_per_charge
        )
        row_passes = terms.rows[row]
        if not any(dr == 1 for dr, _ in row_passes):
            row_passes = ((1, empty_cols),) + row_passes

        eff_row = None

        def _eval_dr1(
            energies_acc: jax.Array,
            eff_row_acc: list[jax.Array] | None,
            col_terms: tuple,
        ) -> tuple[jax.Array, list[jax.Array] | None]:
            right_envs = _compute_right_envs(top_env, row_mpo, bottom_env, dtype)
            for terms_at_col in col_terms:
                if any(
                    not isinstance(term, dr1_direct_terms)
                    for term, _contribs in terms_at_col
                ):
                    eff_row_acc = [
                        _assemble_site(
                            tensors, h_links, v_links, config, row, c, mask_per_charge
                        )
                        for c in range(n_cols)
                    ]
                    break
            row_env_config = GIRowEnvConfig(h_links, v_links, config, mask_per_charge)
            left_env = jnp.ones((1, 1, 1), dtype=dtype)
            for col in range(n_cols):
                env_grad = _compute_single_gradient(
                    left_env, right_envs[col], top_env[col], bottom_env[col]
                )
                env_grads[row][col] = env_grad
                envs = RowEnvs(
                    left_env,
                    right_envs,
                    top_env,
                    bottom_env,
                    env_grad,
                    eff_row_acc,
                    config=row_env_config,
                )
                for term, contributions in col_terms[col]:
                    val = _eval_term(
                        term, envs, tensors, row, col, sites, phys_dim,
                    ) / amp
                    for op_idx, coeff_idx in contributions:
                        coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                        energies_acc = energies_acc.at[op_idx].add(coeff * val)
                left_env = _update_left_env_1row(
                    left_env, top_env[col], row_mpo[col], bottom_env[col]
                )
            return energies_acc, eff_row_acc

        def _eval_dr2(
            energies_acc: jax.Array,
            eff_row_acc: list[jax.Array] | None,
            col_terms: tuple,
        ) -> tuple[jax.Array, list[jax.Array] | None]:
            if row >= n_rows - 1:
                return energies_acc, eff_row_acc
            if next_row_mpo is None:
                raise NotImplementedError("Missing next-row MPO for GI dr=2 evaluation.")
            row_mpo_next = next_row_mpo
            bottom_env_next = bottom_envs_cache[row + 1]
            right_envs_2row = _compute_right_envs_2row(
                top_env, row_mpo, row_mpo_next, bottom_env_next, dtype
            )
            eff_row_next = None
            for terms_at_col in col_terms:
                if any(
                    not isinstance(term, dr2_direct_terms)
                    for term, _contribs in terms_at_col
                ):
                    if eff_row_acc is None:
                        eff_row_acc = [
                            _assemble_site(
                                tensors, h_links, v_links, config, row, c, mask_per_charge
                            )
                            for c in range(n_cols)
                        ]
                    eff_row_next = [
                        _assemble_site(
                            tensors, h_links, v_links, config, row + 1, c, mask_per_charge
                        )
                        for c in range(n_cols)
                    ]
                    break
            left_env_2row = jnp.ones((1, 1, 1, 1), dtype=dtype)
            for col in range(n_cols):
                envs = GITwoRowEnvs(
                    left_env_2row, right_envs_2row, top_env, bottom_env_next,
                    eff_row_acc, eff_row_next, h_links, v_links, config, mask_per_charge,
                )
                for term, contributions in col_terms[col]:
                    val = _eval_term(
                        term, envs, tensors, row, col, sites, phys_dim,
                    ) / amp
                    for op_idx, coeff_idx in contributions:
                        coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                        energies_acc = energies_acc.at[op_idx].add(coeff * val)
                left_env_2row = _update_left_env_2row(
                    left_env_2row,
                    top_env[col],
                    row_mpo[col],
                    row_mpo_next[col],
                    bottom_env_next[col],
                )
            return energies_acc, eff_row_acc

        for dr, col_terms in row_passes:
            if dr == 1:
                energies, eff_row = _eval_dr1(energies, eff_row, col_terms)
                continue
            if dr == 2:
                energies, eff_row = _eval_dr2(energies, eff_row, col_terms)
                continue
            raise NotImplementedError(
                f"GI transition evaluation for dr={dr} is not implemented."
            )

        bottom_env = _apply_mpo_from_below(bottom_env, row_mpo, strategy)
        next_row_mpo = row_mpo

    return env_grads, energies, bottom_envs_cache


# =============================================================================
# GIPEPS Sweep helpers
# =============================================================================


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
    """Sweep plaquettes in a row pair using direct einsum."""
    n_cols = config.shape[1]
    dtype = row_mpo0[0].dtype
    right_envs = _compute_right_envs_2row(top_env, row_mpo0, row_mpo1, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)

    for c in range(n_cols - 1):
        key, subkey = jax.random.split(key)
        delta = jax.random.randint(subkey, (), 1, config.N, dtype=jnp.int32)

        amp_cur = _contract_2row_2col(
            left_env, top_env, row_mpo0[c], row_mpo1[c],
            row_mpo0[c + 1], row_mpo1[c + 1], bottom_env, right_envs[c + 1], c,
        )
        # Compute proposed configuration
        h_prop, v_prop = _plaquette_flip(h_links, v_links, r, c, delta=delta, N=config.N)
        eff00 = _assemble_site(tensors, h_prop, v_prop, config, r, c)
        eff01 = _assemble_site(tensors, h_prop, v_prop, config, r, c + 1)
        eff10 = _assemble_site(tensors, h_prop, v_prop, config, r + 1, c)
        eff11 = _assemble_site(tensors, h_prop, v_prop, config, r + 1, c + 1)
        mpo00_prop = jnp.transpose(eff00[sites[r, c]], (2, 3, 0, 1))
        mpo01_prop = jnp.transpose(eff01[sites[r, c + 1]], (2, 3, 0, 1))
        mpo10_prop = jnp.transpose(eff10[sites[r + 1, c]], (2, 3, 0, 1))
        mpo11_prop = jnp.transpose(eff11[sites[r + 1, c + 1]], (2, 3, 0, 1))
        amp_prop = _contract_2row_2col(
            left_env, top_env, mpo00_prop, mpo10_prop,
            mpo01_prop, mpo11_prop, bottom_env, right_envs[c + 1], c,
        )
        key, accept = _metropolis_hastings_accept(
            key, jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2
        )

        # Update MPOs for accepted proposals
        row_mpo0_list = list(row_mpo0)
        row_mpo1_list = list(row_mpo1)
        row_mpo0_list[c] = jnp.where(accept, mpo00_prop, row_mpo0[c])
        row_mpo0_list[c + 1] = jnp.where(accept, mpo01_prop, row_mpo0[c + 1])
        row_mpo1_list[c] = jnp.where(accept, mpo10_prop, row_mpo1[c])
        row_mpo1_list[c + 1] = jnp.where(accept, mpo11_prop, row_mpo1[c + 1])
        row_mpo0 = tuple(row_mpo0_list)
        row_mpo1 = tuple(row_mpo1_list)
        h_links = jnp.where(accept, h_prop, h_links)
        v_links = jnp.where(accept, v_prop, v_links)

        left_env = _update_left_env_2row(left_env, top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c])

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
    mask_per_charge: jax.Array | None,
    r: int,
    charge_of_site: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> tuple[jax.Array, tuple, jax.Array, jax.Array]:
    """Sweep horizontal links in a single row using direct einsum."""
    n_cols = config.shape[1]
    n = jnp.asarray(config.N, dtype=jnp.int32)
    dtype = row_mpo[0].dtype
    right_envs = _compute_right_envs(top_env, row_mpo, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1), dtype=dtype)

    for c in range(n_cols - 1):
        key, subkey = jax.random.split(key)
        delta = jax.random.randint(subkey, (), 1, config.N, dtype=jnp.int32)

        # Direct einsum for 2-site amplitude
        # Convention: left_env (a,c,e), top[c] (a,u,b), mpo[c] (c,d,u,v), bot[c] (e,v,f)
        # top[c+1] (b,g,h), mpo[c+1] (d,i,g,w), bot[c+1] (f,w,j), right_env (h,i,j)
        amp_cur = jnp.einsum(
            "ace,aub,cduv,evf,bgh,digw,fwj,hij->",
            left_env, top_env[c], row_mpo[c], bottom_env[c],
            top_env[c + 1], row_mpo[c + 1], bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 1), (0, 6), (0, 5), (0, 3), (1, 2), (1, 2), (0, 1)],
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
        proposal_ratio = _hastings_ratio(
            forward_prob=1.0 / (charge_deg[q_left_new] * charge_deg[q_right_new]),
            backward_prob=1.0 / (charge_deg[q_left] * charge_deg[q_right]),
        )

        eff0 = _assemble_site(
            tensors, h_prop, v_links, config, r, c, mask_per_charge
        )
        eff1 = _assemble_site(
            tensors, h_prop, v_links, config, r, c + 1, mask_per_charge
        )
        mpo0 = jnp.transpose(eff0[sites_prop[r, c]], (2, 3, 0, 1))
        mpo1 = jnp.transpose(eff1[sites_prop[r, c + 1]], (2, 3, 0, 1))
        # Direct einsum for proposed amplitude
        amp_prop = jnp.einsum(
            "ace,aub,cduv,evf,bgh,digw,fwj,hij->",
            left_env, top_env[c], mpo0, bottom_env[c],
            top_env[c + 1], mpo1, bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 1), (0, 6), (0, 5), (0, 3), (1, 2), (1, 2), (0, 1)],
        )
        key, accept = _metropolis_hastings_accept(
            key,
            jnp.abs(amp_cur) ** 2,
            jnp.abs(amp_prop) ** 2,
            proposal_ratio=proposal_ratio,
        )

        # Update row_mpo, h_links, sites based on accept
        row_mpo_list = list(row_mpo)
        row_mpo_list[c] = jnp.where(accept, mpo0, row_mpo[c])
        row_mpo_list[c + 1] = jnp.where(accept, mpo1, row_mpo[c + 1])
        row_mpo = tuple(row_mpo_list)
        h_links = jnp.where(accept, h_prop, h_links)
        sites = jnp.where(accept, sites_prop, sites)

        left_env = _update_left_env_1row(left_env, top_env[c], row_mpo[c], bottom_env[c])

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
    mask_per_charge: jax.Array | None,
    r: int,
    charge_of_site: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> tuple[jax.Array, tuple, tuple, jax.Array, jax.Array]:
    """Sweep vertical links in a row pair using direct einsum."""
    n_cols = config.shape[1]
    n = jnp.asarray(config.N, dtype=jnp.int32)
    dtype = row_mpo0[0].dtype
    right_envs = _compute_right_envs_2row(top_env, row_mpo0, row_mpo1, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)

    for c in range(n_cols):
        key, subkey = jax.random.split(key)
        delta = jax.random.randint(subkey, (), 1, config.N, dtype=jnp.int32)

        amp_cur = _contract_2row_1col(
            left_env, top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c], right_envs[c],
        )
        v_prop = v_links.at[r, c].set((v_links[r, c] + delta) % n)
        q_top = charge_of_site[sites[r, c]]
        q_bottom = charge_of_site[sites[r + 1, c]]
        q_top_new = (q_top - delta) % n
        q_bottom_new = (q_bottom + delta) % n
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
        proposal_ratio = _hastings_ratio(
            forward_prob=1.0 / (charge_deg[q_top_new] * charge_deg[q_bottom_new]),
            backward_prob=1.0 / (charge_deg[q_top] * charge_deg[q_bottom]),
        )
        eff0 = _assemble_site(
            tensors, h_links, v_prop, config, r, c, mask_per_charge
        )
        eff1 = _assemble_site(
            tensors, h_links, v_prop, config, r + 1, c, mask_per_charge
        )
        mpo0_prop = jnp.transpose(eff0[sites_prop[r, c]], (2, 3, 0, 1))
        mpo1_prop = jnp.transpose(eff1[sites_prop[r + 1, c]], (2, 3, 0, 1))
        amp_prop = _contract_2row_1col(
            left_env, top_env[c], mpo0_prop, mpo1_prop, bottom_env[c], right_envs[c],
        )
        key, accept = _metropolis_hastings_accept(
            key,
            jnp.abs(amp_cur) ** 2,
            jnp.abs(amp_prop) ** 2,
            proposal_ratio=proposal_ratio,
        )

        # Update row_mpo, v_links, sites based on accept
        row_mpo0_list = list(row_mpo0)
        row_mpo1_list = list(row_mpo1)
        row_mpo0_list[c] = jnp.where(accept, mpo0_prop, row_mpo0[c])
        row_mpo1_list[c] = jnp.where(accept, mpo1_prop, row_mpo1[c])
        row_mpo0 = tuple(row_mpo0_list)
        row_mpo1 = tuple(row_mpo1_list)
        v_links = jnp.where(accept, v_prop, v_links)
        sites = jnp.where(accept, sites_prop, sites)

        left_env = _update_left_env_2row(left_env, top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c])

    return key, row_mpo0, row_mpo1, sites, v_links


def _horizontal_hardcore_hop_sweep_row(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    row_mpo: tuple,
    mask_per_charge: jax.Array | None,
    r: int,
) -> tuple[jax.Array, tuple, jax.Array, jax.Array]:
    """Sweep number-conserving Z2 hard-core hops on a row."""
    n_cols = config.shape[1]
    if n_cols <= 1:
        return key, row_mpo, sites, h_links
    dtype = row_mpo[0].dtype
    right_envs = _compute_right_envs(top_env, row_mpo, bottom_env, dtype)
    mpo = list(row_mpo)
    left_env = jnp.ones((1, 1, 1), dtype=dtype)
    amp_cur = jnp.einsum(
        "ace,aub,cduv,evf,bgh,digw,fwj,hij->",
        left_env, top_env[0], mpo[0], bottom_env[0],
        top_env[1], mpo[1], bottom_env[1], right_envs[1],
        optimize=[(0, 1), (0, 6), (0, 5), (0, 3), (1, 2), (1, 2), (0, 1)],
    )

    for c in range(n_cols - 1):
        allowed = sites[r, c] != sites[r, c + 1]
        mpo_c = mpo[c]
        mpo_cp1 = mpo[c + 1]
        right_env = right_envs[c + 1]

        def keep(_):
            return (
                key,
                amp_cur,
                _update_left_env_1row(left_env, top_env[c], mpo_c, bottom_env[c]),
                mpo_c,
                mpo_cp1,
                sites,
                h_links,
            )

        def propose(_):
            sites_prop, h_prop = _horizontal_hardcore_hop(sites, h_links, r, c)
            eff0 = _assemble_site(
                tensors, h_prop, v_links, config, r, c, mask_per_charge
            )
            eff1 = _assemble_site(
                tensors, h_prop, v_links, config, r, c + 1, mask_per_charge
            )
            mpo0 = jnp.transpose(eff0[sites_prop[r, c]], (2, 3, 0, 1))
            mpo1 = jnp.transpose(eff1[sites_prop[r, c + 1]], (2, 3, 0, 1))
            prefix_prop = _update_left_env_1row(
                left_env, top_env[c], mpo0, bottom_env[c]
            )
            amp_prop = jnp.einsum(
                "bdf,bgh,digw,fwj,hij->",
                prefix_prop, top_env[c + 1], mpo1, bottom_env[c + 1], right_env,
                optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
            )
            key_next, accept = _metropolis_hastings_accept(
                key,
                jnp.abs(amp_cur) ** 2,
                jnp.abs(amp_prop) ** 2,
            )
            return jax.lax.cond(
                accept,
                lambda _: (
                    key_next,
                    amp_prop,
                    prefix_prop,
                    mpo0,
                    mpo1,
                    sites_prop,
                    h_prop,
                ),
                lambda _: (
                    key_next,
                    amp_cur,
                    _update_left_env_1row(left_env, top_env[c], mpo_c, bottom_env[c]),
                    mpo_c,
                    mpo_cp1,
                    sites,
                    h_links,
                ),
                operand=None,
            )

        key, amp_cur, left_env, mpo[c], mpo[c + 1], sites, h_links = jax.lax.cond(
            allowed,
            propose,
            keep,
            operand=None,
        )

    return key, tuple(mpo), sites, h_links


def _vertical_hardcore_hop_sweep_row_pair(
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
    mask_per_charge: jax.Array | None,
    r: int,
) -> tuple[jax.Array, tuple, tuple, jax.Array, jax.Array]:
    """Sweep number-conserving Z2 hard-core hops on a row pair."""
    n_cols = config.shape[1]
    dtype = row_mpo0[0].dtype
    right_envs = _compute_right_envs_2row(top_env, row_mpo0, row_mpo1, bottom_env, dtype)
    mpo0 = list(row_mpo0)
    mpo1 = list(row_mpo1)
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
    amp_cur = _contract_2row_1col(
        left_env, top_env[0], mpo0[0], mpo1[0], bottom_env[0], right_envs[0],
    )

    for c in range(n_cols):
        allowed = sites[r, c] != sites[r + 1, c]
        mpo0_c = mpo0[c]
        mpo1_c = mpo1[c]
        right_env = right_envs[c]

        def keep(_):
            return (
                key,
                amp_cur,
                _update_left_env_2row(left_env, top_env[c], mpo0_c, mpo1_c, bottom_env[c]),
                mpo0_c,
                mpo1_c,
                sites,
                v_links,
            )

        def propose(_):
            sites_prop, v_prop = _vertical_hardcore_hop(sites, v_links, r, c)
            eff0 = _assemble_site(
                tensors, h_links, v_prop, config, r, c, mask_per_charge
            )
            eff1 = _assemble_site(
                tensors, h_links, v_prop, config, r + 1, c, mask_per_charge
            )
            mpo0_prop = jnp.transpose(eff0[sites_prop[r, c]], (2, 3, 0, 1))
            mpo1_prop = jnp.transpose(eff1[sites_prop[r + 1, c]], (2, 3, 0, 1))
            prefix_prop = _update_left_env_2row(
                left_env, top_env[c], mpo0_prop, mpo1_prop, bottom_env[c]
            )
            amp_prop = jnp.einsum(
                "bryf,bryf->",
                prefix_prop,
                right_env,
                optimize=[(0, 1)],
            )
            key_next, accept = _metropolis_hastings_accept(
                key,
                jnp.abs(amp_cur) ** 2,
                jnp.abs(amp_prop) ** 2,
            )
            return jax.lax.cond(
                accept,
                lambda _: (
                    key_next,
                    amp_prop,
                    prefix_prop,
                    mpo0_prop,
                    mpo1_prop,
                    sites_prop,
                    v_prop,
                ),
                lambda _: (
                    key_next,
                    amp_cur,
                    _update_left_env_2row(left_env, top_env[c], mpo0_c, mpo1_c, bottom_env[c]),
                    mpo0_c,
                    mpo1_c,
                    sites,
                    v_links,
                ),
                operand=None,
            )

        key, amp_cur, left_env, mpo0[c], mpo1[c], sites, v_links = jax.lax.cond(
            allowed,
            propose,
            keep,
            operand=None,
        )

    return key, tuple(mpo0), tuple(mpo1), sites, v_links


def _compute_bottom_envs(
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    strategy: Any,
    mask_per_charge: jax.Array | None = None,
) -> list[tuple]:
    """Compute bottom boundary environments (internal helper for sweep)."""
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    envs = [None] * n_rows
    env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        envs[row] = env
        row_mpo = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, row, n_cols, mask_per_charge
        )
        env = _apply_mpo_from_below(env, row_mpo, strategy)
    return envs


def transition(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    key: jax.Array,
    envs: list[tuple],
    shape: tuple[int, int],
    config: GIPEPSConfig,
    strategy: Any,
    mask_per_charge: jax.Array | None,
    charge_of_site: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple]:
    """Combined plaquette + link sweeps for GI-PEPS."""
    sites, h_links, v_links = GIPEPS.unflatten_sample(sample, shape)
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    top_envs_cache = [None] * n_rows
    number_conserving = (
        config.is_binary_occupancy_matter and config.conserve_particle_number
    )

    # 1. Plaquette sweep over the full lattice (row pairs)
    top_env_plaquettes = None
    if n_rows > 1:
        top_env_plaquettes = tuple(
            jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols)
        )
        row_mpo0 = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, 0, n_cols, mask_per_charge
        )
        row_mpo1 = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, 1, n_cols, mask_per_charge
        )
        for r in range(n_rows - 1):
            if config.phys_dim == 1:
                top_envs_cache[r] = top_env_plaquettes
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
                    tensors, sites, h_links, v_links, config, r + 2, n_cols, mask_per_charge
                )
        if config.phys_dim == 1:
            top_envs_cache[n_rows - 1] = top_env_plaquettes
        top_env_plaquettes = strategy.apply(top_env_plaquettes, row_mpo1)

    # For pure gauge (phys_dim == 1), no link/matter sweeps needed
    if config.phys_dim == 1:
        if top_env_plaquettes is None:
            top_env_plaquettes = tuple(
                jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols)
            )
            for row in range(n_rows):
                top_envs_cache[row] = top_env_plaquettes
                row_mpo = _build_row_mpo_gi(
                    tensors, sites, h_links, v_links, config, row, n_cols, mask_per_charge
                )
                top_env_plaquettes = strategy.apply(top_env_plaquettes, row_mpo)
        amp = _contract_bottom(top_env_plaquettes)
        return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp, tuple(top_envs_cache)

    # 2. Horizontal link sweeps for all rows
    # Recompute bottom_envs after plaquette changes
    bottom_envs_h = _compute_bottom_envs(
        tensors, sites, h_links, v_links, config, strategy, mask_per_charge
    )

    top_env_h = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows):
        if n_rows == 1:
            top_envs_cache[row] = top_env_h
        row_mpo = _build_row_mpo_gi(
            tensors, sites, h_links, v_links, config, row, n_cols, mask_per_charge
        )
        if number_conserving:
            key, row_mpo, sites, h_links = _horizontal_hardcore_hop_sweep_row(
                key,
                tensors,
                sites,
                h_links,
                v_links,
                config,
                top_env_h,
                bottom_envs_h[row],
                row_mpo,
                mask_per_charge,
                row,
            )
        else:
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
                mask_per_charge,
                row,
                charge_of_site,
                charge_to_indices,
                charge_deg,
            )
        top_env_h = strategy.apply(top_env_h, row_mpo)

    if n_rows == 1:
        amp = _contract_bottom(top_env_h)
        return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp, tuple(top_envs_cache)

    # 3. Vertical link sweeps for all row pairs
    # Recompute bottom_envs after horizontal changes
    bottom_envs_v = _compute_bottom_envs(
        tensors, sites, h_links, v_links, config, strategy, mask_per_charge
    )

    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    row_mpo0 = _build_row_mpo_gi(
        tensors, sites, h_links, v_links, config, 0, n_cols, mask_per_charge
    )
    row_mpo1 = _build_row_mpo_gi(
        tensors, sites, h_links, v_links, config, 1, n_cols, mask_per_charge
    )
    for r in range(n_rows - 1):
        top_envs_cache[r] = top_env
        if number_conserving:
            key, row_mpo0, row_mpo1, sites, v_links = _vertical_hardcore_hop_sweep_row_pair(
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
                mask_per_charge,
                r,
            )
        else:
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
                mask_per_charge,
                r,
                charge_of_site,
                charge_to_indices,
                charge_deg,
            )
        top_env = strategy.apply(top_env, row_mpo0)
        if r + 2 < n_rows:
            row_mpo0 = row_mpo1
            row_mpo1 = _build_row_mpo_gi(
                tensors, sites, h_links, v_links, config, r + 2, n_cols, mask_per_charge
            )
    top_envs_cache[n_rows - 1] = top_env
    top_env = strategy.apply(top_env, row_mpo1)

    amp = _contract_bottom(top_env)
    return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp, tuple(top_envs_cache)

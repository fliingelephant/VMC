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
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _compute_right_envs,
    _contract_bottom,
)
from vmc.peps.common.energy import (
    _compute_2site_horizontal_env,
    _compute_right_envs_2row,
    _compute_row_pair_vertical_energy,
    _compute_single_gradient,
)
from vmc.peps.gi.compat import gi_apply
from vmc.operators.local_terms import (
    BucketedOperators,
    bucket_operators,
)
from vmc.utils.utils import random_tensor, _hastings_ratio, _metropolis_hastings_accept


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
    mask_per_charge: jax.Array | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dmax = int(max(self.degeneracy_per_charge))
        if all(d == dmax for d in self.degeneracy_per_charge):
            object.__setattr__(self, "mask_per_charge", None)
            return
        deg = jnp.asarray(self.degeneracy_per_charge, dtype=jnp.int32)
        mask = jnp.arange(dmax) < deg[:, None]
        object.__setattr__(self, "mask_per_charge", mask)

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

    apply = staticmethod(gi_apply)
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
            v_links = v_links.at[:, : n_cols - 1].add(deltas)
            v_links = v_links.at[:, 1:].add(-deltas)
            h_links = h_links % self.N
            v_links = v_links % self.N
        nl = jnp.pad(h_links, ((0, 0), (1, 0)), constant_values=0)
        nr = jnp.pad(h_links, ((0, 0), (0, 1)), constant_values=0)
        nu = jnp.pad(v_links, ((1, 0), (0, 0)), constant_values=0)
        nd = jnp.pad(v_links, ((0, 1), (0, 0)), constant_values=0)
        div = (nl + nd - nu - nr) % self.N
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
    tensor = tensors[r][c][:, cfg_idx, :, :, :, :]
    mask_per_charge = config.mask_per_charge
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
) -> tuple:
    """Build row-MPO for GI-PEPS contraction."""
    return tuple(
        jnp.transpose(
            _assemble_site(tensors, h_links, v_links, config, row, c)[sites[row, c]],
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


# =============================================================================
# Runtime Internals for GIPEPS Kernels
# =============================================================================


def estimate(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    amp: jax.Array,
    operator: Any,
    shape: tuple[int, int],
    config: GIPEPSConfig,
    strategy: Any,
    top_envs: list[tuple],
    *,
    terms: BucketedOperators | None = None,
) -> tuple[list[list[jax.Array]], jax.Array, list[tuple]]:
    """Compute environment gradients and local energy for GI-PEPS."""
    from vmc.peps.gi.local_terms import LinkDiagonalTerm

    sites, h_links, v_links = GIPEPS.unflatten_sample(sample, shape)
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    phys_dim = config.phys_dim
    bottom_envs_cache = [None] * n_rows

    if terms is None:
        terms = bucket_operators(operator.terms, config.shape)
    diagonal_terms = terms.diagonal
    span_11_terms = terms.span_11
    span_12_terms = terms.span_12
    span_21_terms = terms.span_21
    span_22_terms = terms.span_22

    env_grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    # Compute diagonal energy
    energy = jnp.zeros((), dtype=amp.dtype)
    for _, term in diagonal_terms:
        if isinstance(term, LinkDiagonalTerm):
            energy = energy + term.energy(h_links, v_links)
            continue
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + sites[row, col]
        energy = energy + term.diag[idx]

    # Main row iteration
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        bottom_envs_cache[row] = bottom_env
        top_env = top_envs[row]
        row_mpo = _build_row_mpo_gi(tensors, sites, h_links, v_links, config, row, n_cols)

        site_row_terms = span_11_terms[row]
        horizontal_row_terms = span_12_terms[row]
        vertical_row_terms = span_21_terms[row] if row < n_rows - 1 else ()
        plaquette_row_terms = span_22_terms[row] if row < n_rows - 1 else ()
        row_has_one_site = any(site_row_terms)
        row_has_horizontal = any(horizontal_row_terms)
        row_has_vertical = row < n_rows - 1 and any(vertical_row_terms)
        row_has_plaquette = row < n_rows - 1 and any(plaquette_row_terms)

        # 1-row right envs (always needed for gradients)
        right_envs = _compute_right_envs(top_env, row_mpo, bottom_env, dtype)

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

            site_terms = site_row_terms[c]
            horizontal_terms = horizontal_row_terms[c] if c < n_cols - 1 else ()

            amps_site = None
            if site_terms:
                amps_site = jnp.einsum("pudlr,udlr->p", eff_row[c], env_grad)
            amps_edge = None
            if horizontal_terms:
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
                    optimize=[(0, 2), (0, 1)],
                )
            for _, term in site_terms:
                spin_idx = sites[row, c]
                energy = energy + jnp.dot(term.op[:, spin_idx], amps_site) / amp
            for _, term in horizontal_terms:
                spin0 = sites[row, c]
                spin1 = sites[row, c + 1]
                col_idx = spin0 * phys_dim + spin1
                amps_flat = amps_edge.reshape(-1)
                energy = energy + jnp.dot(term.op[:, col_idx], amps_flat) / amp

            # Direct einsum for left_env update
            left_env = jnp.einsum(
                "ace,aub,cduv,evf->bdf",
                left_env, top_env[c], row_mpo[c], bottom_env[c],
                optimize=[(0, 1), (0, 2), (0, 1)],
            )

        # 2-row terms (vertical + plaquette)
        if row < n_rows - 1:
            row_mpo_next = _build_row_mpo_gi(
                tensors, sites, h_links, v_links, config, row + 1, n_cols
            )
            if row_has_vertical or row_has_plaquette:
                bottom_env_next = bottom_envs_cache[row + 1]
                right_envs_2row = _compute_right_envs_2row(
                    top_env, row_mpo, row_mpo_next, bottom_env_next, dtype
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
                        vertical_row_terms,
                        amp,
                        phys_dim,
                        right_envs_2row=right_envs_2row,
                    )

                # Plaquette energy
                if row_has_plaquette:
                    left_env_2row = jnp.ones((1, 1, 1, 1), dtype=dtype)
                    for c in range(n_cols - 1):
                        plaquette_here = plaquette_row_terms[c]
                        if not plaquette_here:
                            # Direct einsum for left_env_2row update
                            left_env_2row = jnp.einsum(
                                "alxe,aub,lruv,xyvw,ewf->bryf",
                                left_env_2row, top_env[c], row_mpo[c], row_mpo_next[c], bottom_env_next[c],
                                optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
                            )
                            continue
                        # Use global amp to normalize plaquette contributions (saves one contraction).
                        # Compute amplitude for +delta flip
                        h_plus, v_plus = _plaquette_flip(
                            h_links, v_links, row, c, delta=1, N=config.N
                        )
                        eff00p = _assemble_site(tensors, h_plus, v_plus, config, row, c)
                        eff01p = _assemble_site(tensors, h_plus, v_plus, config, row, c + 1)
                        eff10p = _assemble_site(tensors, h_plus, v_plus, config, row + 1, c)
                        eff11p = _assemble_site(tensors, h_plus, v_plus, config, row + 1, c + 1)
                        mpo00p = jnp.transpose(eff00p[sites[row, c]], (2, 3, 0, 1))
                        mpo01p = jnp.transpose(eff01p[sites[row, c + 1]], (2, 3, 0, 1))
                        mpo10p = jnp.transpose(eff10p[sites[row + 1, c]], (2, 3, 0, 1))
                        mpo11p = jnp.transpose(eff11p[sites[row + 1, c + 1]], (2, 3, 0, 1))
                        amp_plus = jnp.einsum(
                            "alxe,aub,lruv,xyvw,ewf,bgc,rsgh,ythi,fij,cstj->",
                            left_env_2row, top_env[c], mpo00p, mpo10p, bottom_env_next[c],
                            top_env[c + 1], mpo01p, mpo11p, bottom_env_next[c + 1],
                            right_envs_2row[c + 1],
                            optimize=[(1, 5), (3, 6), (1, 2), (1, 2), (0, 2), (2, 4), (1, 3), (0, 2), (0, 1)],
                        )
                        # Compute amplitude for -delta flip
                        h_minus, v_minus = _plaquette_flip(
                            h_links, v_links, row, c, delta=-1, N=config.N
                        )
                        eff00m = _assemble_site(tensors, h_minus, v_minus, config, row, c)
                        eff01m = _assemble_site(tensors, h_minus, v_minus, config, row, c + 1)
                        eff10m = _assemble_site(tensors, h_minus, v_minus, config, row + 1, c)
                        eff11m = _assemble_site(tensors, h_minus, v_minus, config, row + 1, c + 1)
                        mpo00m = jnp.transpose(eff00m[sites[row, c]], (2, 3, 0, 1))
                        mpo01m = jnp.transpose(eff01m[sites[row, c + 1]], (2, 3, 0, 1))
                        mpo10m = jnp.transpose(eff10m[sites[row + 1, c]], (2, 3, 0, 1))
                        mpo11m = jnp.transpose(eff11m[sites[row + 1, c + 1]], (2, 3, 0, 1))
                        amp_minus = jnp.einsum(
                            "alxe,aub,lruv,xyvw,ewf,bgc,rsgh,ythi,fij,cstj->",
                            left_env_2row, top_env[c], mpo00m, mpo10m, bottom_env_next[c],
                            top_env[c + 1], mpo01m, mpo11m, bottom_env_next[c + 1],
                            right_envs_2row[c + 1],
                            optimize=[(1, 5), (3, 6), (1, 2), (1, 2), (0, 2), (2, 4), (1, 3), (0, 2), (0, 1)],
                        )
                        if len(plaquette_here) == 1:
                            coeff = plaquette_here[0][1].coeff
                        else:
                            coeff = jnp.sum(
                                jnp.asarray([term.coeff for _, term in plaquette_here])
                            )
                        energy = energy + coeff * (amp_plus + amp_minus) / amp
                        # Direct einsum for left_env_2row update
                        left_env_2row = jnp.einsum(
                            "alxe,aub,lruv,xyvw,ewf->bryf",
                            left_env_2row, top_env[c], row_mpo[c], row_mpo_next[c], bottom_env_next[c],
                            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
                        )

        bottom_env = _apply_mpo_from_below(bottom_env, row_mpo, strategy)

    return env_grads, energy, bottom_envs_cache


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

        # Direct einsum for 2-column amplitude
        # Convention: left_env (a,l,x,e), top[c] (a,u,b), mpo0[c] (l,r,u,v), mpo1[c] (x,y,v,w), bot[c] (e,w,f)
        # top[c+1] (b,g,c), mpo0[c+1] (r,s,g,h), mpo1[c+1] (y,t,h,i), bot[c+1] (f,i,j), right_env (c,s,t,j)
        amp_cur = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf,bgc,rsgh,ythi,fij,cstj->",
            left_env, top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c],
            top_env[c + 1], row_mpo0[c + 1], row_mpo1[c + 1], bottom_env[c + 1],
            right_envs[c + 1],
            optimize=[(1, 5), (3, 6), (1, 2), (1, 2), (0, 2), (2, 4), (1, 3), (0, 2), (0, 1)],
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
        amp_prop = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf,bgc,rsgh,ythi,fij,cstj->",
            left_env, top_env[c], mpo00_prop, mpo10_prop, bottom_env[c],
            top_env[c + 1], mpo01_prop, mpo11_prop, bottom_env[c + 1],
            right_envs[c + 1],
            optimize=[(1, 5), (3, 6), (1, 2), (1, 2), (0, 2), (2, 4), (1, 3), (0, 2), (0, 1)],
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

        # Direct einsum for left_env update (use current MPOs after accept/reject)
        left_env = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf->bryf",
            left_env, top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c],
            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
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

        eff0 = _assemble_site(tensors, h_prop, v_links, config, r, c)
        eff1 = _assemble_site(tensors, h_prop, v_links, config, r, c + 1)
        mpo0 = jnp.transpose(eff0[sites_prop[r, c]], (2, 3, 0, 1))
        mpo1 = jnp.transpose(eff1[sites_prop[r, c + 1]], (2, 3, 0, 1))
        # Direct einsum for proposed amplitude
        amp_prop = jnp.einsum(
            "ace,aub,cduv,evf,bgh,digw,fwj,hij->",
            left_env, top_env[c], mpo0, bottom_env[c],
            top_env[c + 1], mpo1, bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 1), (0, 6), (0, 5), (0, 3), (1, 2), (1, 2), (0, 1)],
        )
        proposal_ratio = _hastings_ratio(
            forward_prob=1.0 / (charge_deg[q_left_new] * charge_deg[q_right_new]),
            backward_prob=1.0 / (charge_deg[q_left] * charge_deg[q_right]),
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

        # Direct einsum for left_env update
        left_env = jnp.einsum(
            "ace,aub,cduv,evf->bdf",
            left_env, top_env[c], row_mpo[c], bottom_env[c],
            optimize=[(0, 1), (0, 2), (0, 1)],
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
    """Sweep vertical links in a row pair using direct einsum."""
    n_cols = config.shape[1]
    n = jnp.asarray(config.N, dtype=jnp.int32)
    dtype = row_mpo0[0].dtype
    right_envs = _compute_right_envs_2row(top_env, row_mpo0, row_mpo1, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)

    for c in range(n_cols):
        key, subkey = jax.random.split(key)
        delta = jax.random.randint(subkey, (), 1, config.N, dtype=jnp.int32)

        # Direct einsum for amplitude (single column in 2-row)
        amp_cur = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf,bryf->",
            left_env, top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c], right_envs[c],
            optimize=[(0, 1), (0, 4), (1, 2), (1, 2), (0, 1)],
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
        eff0 = _assemble_site(tensors, h_links, v_prop, config, r, c)
        eff1 = _assemble_site(tensors, h_links, v_prop, config, r + 1, c)
        mpo0_prop = jnp.transpose(eff0[sites_prop[r, c]], (2, 3, 0, 1))
        mpo1_prop = jnp.transpose(eff1[sites_prop[r + 1, c]], (2, 3, 0, 1))
        # Direct einsum for proposed amplitude
        amp_prop = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf,bryf->",
            left_env, top_env[c], mpo0_prop, mpo1_prop, bottom_env[c], right_envs[c],
            optimize=[(0, 1), (0, 4), (1, 2), (1, 2), (0, 1)],
        )
        proposal_ratio = _hastings_ratio(
            forward_prob=1.0 / (charge_deg[q_top_new] * charge_deg[q_bottom_new]),
            backward_prob=1.0 / (charge_deg[q_top] * charge_deg[q_bottom]),
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

        # Direct einsum for left_env update
        left_env = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf->bryf",
            left_env, top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c],
            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
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


def transition(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    key: jax.Array,
    envs: list[tuple],
    shape: tuple[int, int],
    config: GIPEPSConfig,
    strategy: Any,
    charge_of_site: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple]:
    """Combined plaquette + link sweeps for GI-PEPS."""
    sites, h_links, v_links = GIPEPS.unflatten_sample(sample, shape)
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    top_envs_cache = [None] * n_rows

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
                    tensors, sites, h_links, v_links, config, r + 2, n_cols
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
                    tensors, sites, h_links, v_links, config, row, n_cols
                )
                top_env_plaquettes = strategy.apply(top_env_plaquettes, row_mpo)
        amp = _contract_bottom(top_env_plaquettes)
        return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp, tuple(top_envs_cache)

    # 2. Horizontal link sweeps for all rows
    # Recompute bottom_envs after plaquette changes
    bottom_envs_h = _compute_bottom_envs(
        tensors, sites, h_links, v_links, config, strategy
    )

    top_env_h = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows):
        if n_rows == 1:
            top_envs_cache[row] = top_env_h
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
        return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp, tuple(top_envs_cache)

    # 3. Vertical link sweeps for all row pairs
    # Recompute bottom_envs after horizontal changes
    bottom_envs_v = _compute_bottom_envs(
        tensors, sites, h_links, v_links, config, strategy
    )

    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    row_mpo0 = _build_row_mpo_gi(tensors, sites, h_links, v_links, config, 0, n_cols)
    row_mpo1 = _build_row_mpo_gi(tensors, sites, h_links, v_links, config, 1, n_cols)
    for r in range(n_rows - 1):
        top_envs_cache[r] = top_env
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
    top_envs_cache[n_rows - 1] = top_env
    top_env = strategy.apply(top_env, row_mpo1)

    amp = _contract_bottom(top_env)
    return GIPEPS.flatten_sample(sites, h_links, v_links), key, amp, tuple(top_envs_cache)


# --------------------------------------------------------------------------- #
# Dispatches for small-o helpers
# --------------------------------------------------------------------------- #
from vmc.utils.smallo import params_per_site, sliced_dims


@params_per_site.dispatch
def _(model: GIPEPS) -> list[int]:
    """Number of parameters per active slice at each GIPEPS site.

    For GIPEPS, the active slice is determined by (σ, cfg) where cfg encodes
    the local gauge configuration. Each slice has shape [bond_dims...].
    """
    n_rows, n_cols = model.shape
    return [
        int(jnp.asarray(model.tensors[r][c])[0, 0].size)
        for r in range(n_rows)
        for c in range(n_cols)
    ]


@sliced_dims.dispatch
def _(model: GIPEPS) -> tuple[int, ...]:
    """Number of distinct active slices per site (= phys_dim * nc for GIPEPS).

    Unlike standard PEPS where only σ ∈ {0,...,d-1} selects the slice,
    GIPEPS has a combined index: slice_idx = σ * nc + cfg_idx, where cfg_idx
    encodes the local gauge configuration satisfying Gauss law.
    """
    n_rows, n_cols = model.shape
    return tuple(
        model.phys_dim * jnp.asarray(model.tensors[r][c]).shape[1]
        for r in range(n_rows)
        for c in range(n_cols)
    )

"""Generic sampled-block non-Abelian gauge-invariant PEPS model."""

from __future__ import annotations

import vmc.config  # noqa: F401 - JAX config must be imported first

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.operators.local_terms import support_span
from vmc.peps.common.strategy import ContractionStrategy, Variational
from vmc.peps.non_abelian_gi.builders import build_pure_gauge_tables
from vmc.peps.non_abelian_gi.contraction import (
    active_block_ids_from_fields,
    flatten_matter_sample,
    flatten_sample,
    non_abelian_gi_apply,
    unflatten_matter_sample,
    unflatten_sample,
    unflatten_spin_network_sample,
)
from vmc.peps.non_abelian_gi.factors import (
    build_hopping_factor_tables,
    build_plaquette_factor_tables,
)
from vmc.peps.non_abelian_gi.initial import (
    ConservedMatter,
    Vacuum,
    build_init_tables,
    sample_initial,
)
from vmc.utils.utils import random_tensor

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = ["NonAbelianGIPEPS", "NonAbelianGIPEPSConfig"]


@dataclass(frozen=True)
class NonAbelianGIPEPSConfig:
    """Configuration for a sampled-block non-Abelian GI-PEPS."""

    shape: tuple[int, int]
    gauge_group: Any
    D: int
    chi: int
    target_charge: Any = 0
    phys_dim: int = 1
    matter_irreps: tuple[int, ...] = (0,)
    matter_numbers: tuple[int, ...] = (0,)
    particle_number: int = 0
    dtype: "DTypeLike" = jnp.complex128

    def __post_init__(self) -> None:
        n_rows, n_cols = self.shape
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("shape must have positive dimensions.")
        if self.D <= 0:
            raise ValueError("D must be positive.")
        if self.chi <= 0:
            raise ValueError("chi must be positive.")
        if self.phys_dim <= 0:
            raise ValueError("phys_dim must be positive.")
        if len(self.matter_irreps) != self.phys_dim:
            raise ValueError("matter_irreps must have length phys_dim.")
        if len(self.matter_numbers) != self.phys_dim:
            raise ValueError("matter_numbers must have length phys_dim.")
        if any(number < 0 for number in self.matter_numbers):
            raise ValueError("matter_numbers must be non-negative.")
        if self.particle_number < 0:
            raise ValueError("particle_number must be non-negative.")
        max_particles = n_rows * n_cols * max(self.matter_numbers)
        if self.particle_number > max_particles:
            raise ValueError("particle_number exceeds the lattice matter capacity.")
        if self.phys_dim == 1 and (
            self.matter_irreps != (0,)
            or self.matter_numbers != (0,)
            or self.particle_number != 0
        ):
            raise ValueError(
                "Pure-gauge configurations must use the singlet matter basis."
            )
        if (
            self.matter_irreps == (0, 1)
            and self.matter_numbers == (0, 1)
            and self.particle_number % 2
        ):
            raise ValueError(
                "Singlet/fundamental SU(2) matter requires even particle_number."
            )


class NonAbelianGIPEPS(nnx.Module):
    """Non-Abelian GI-PEPS with one sampled allowed block per vertex."""

    tensors: list[list[nnx.Param]] = nnx.data()

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        config: NonAbelianGIPEPSConfig,
        contraction_strategy: ContractionStrategy | None = None,
    ) -> None:
        self.config = config
        self.shape = (int(config.shape[0]), int(config.shape[1]))
        self.gauge_group = config.gauge_group
        self.D = int(config.D)
        self.chi = int(config.chi)
        self.phys_dim = int(config.phys_dim)
        self.matter_irreps = tuple(int(irrep) for irrep in config.matter_irreps)
        self.matter_numbers = tuple(int(number) for number in config.matter_numbers)
        self.particle_number = int(config.particle_number)
        self.dtype = jnp.dtype(config.dtype)

        self.tables = build_pure_gauge_tables(
            self.gauge_group,
            shape=self.shape,
            target_charge=config.target_charge,
            matter_irreps=self.matter_irreps,
            matter_numbers=self.matter_numbers,
        )
        self.plaquette_factor_tables = build_plaquette_factor_tables(
            self.gauge_group,
            self.tables,
        )
        self.hopping_factor_tables = (
            build_hopping_factor_tables(self.gauge_group, self.tables)
            if self.phys_dim > 1
            else None
        )
        n_rows, n_cols = self.shape
        self.matter_spec = (
            Vacuum()
            if self.phys_dim == 1
            else ConservedMatter(
                irreps=self.matter_irreps,
                numbers=self.matter_numbers,
                particle_number=self.particle_number,
            )
        )
        self.init_tables = build_init_tables(self.tables, self.matter_spec)
        if contraction_strategy is None:
            contraction_strategy = Variational(truncate_bond_dimension=self.chi)
        self.strategy = contraction_strategy

        self.params_per_site = tuple(
            up * down * left * right
            for r in range(n_rows)
            for c in range(n_cols)
            for up, down, left, right in [self._site_dims(r, c, n_rows, n_cols)]
        )
        self.sliced_dims = tuple(
            self.tables.n_blocks(r, c) for r in range(n_rows) for c in range(n_cols)
        )
        self.tensors = [
            [
                nnx.Param(
                    self._initial_site_tensor(rngs, r, c),
                    dtype=self.dtype,
                )
                for c in range(n_cols)
            ]
            for r in range(n_rows)
        ]

    apply = staticmethod(non_abelian_gi_apply)
    eval_span = staticmethod(support_span)
    flatten_sample = staticmethod(flatten_sample)
    flatten_matter_sample = staticmethod(flatten_matter_sample)
    unflatten_sample = staticmethod(unflatten_sample)
    unflatten_matter_sample = staticmethod(unflatten_matter_sample)
    unflatten_spin_network_sample = staticmethod(unflatten_spin_network_sample)

    def _site_dims(
        self,
        r: int,
        c: int,
        n_rows: int,
        n_cols: int,
    ) -> tuple[int, int, int, int]:
        """Return reduced tensor dims in ``(up, down, left, right)`` order."""
        return (
            self.D if r > 0 else 1,
            self.D if r < n_rows - 1 else 1,
            self.D if c > 0 else 1,
            self.D if c < n_cols - 1 else 1,
        )

    def _initial_site_tensor(
        self,
        rngs: nnx.Rngs,
        r: int,
        c: int,
    ) -> jax.Array:
        n_rows, n_cols = self.shape
        return random_tensor(
            rngs,
            (self.tables.n_blocks(r, c), *self._site_dims(r, c, n_rows, n_cols)),
            self.dtype,
        ) / jnp.sqrt(self.params_per_site[r * n_cols + c])

    def all_zero_sample(self) -> jax.Array:
        n_rows, n_cols = self.shape
        h = jnp.zeros((n_rows, n_cols - 1), dtype=jnp.int32)
        v = jnp.zeros((n_rows - 1, n_cols), dtype=jnp.int32)
        iotas = jnp.zeros(self.shape, dtype=jnp.int32)
        if isinstance(self.matter_spec, Vacuum):
            return self.flatten_sample(h, v, iotas)
        return sample_initial(jax.random.PRNGKey(0), self.matter_spec, self.init_tables)

    def random_physical_configuration(
        self,
        key: jax.Array,
        n_samples: int = 1,
    ) -> jax.Array:
        keys = jax.random.split(key, n_samples)
        return jax.vmap(
            lambda k: sample_initial(k, self.matter_spec, self.init_tables)
        )(keys)

    def active_block_ids(self, sample: jax.Array) -> jax.Array:
        """Return active vertex block ids for a sampled spin-network state."""
        matter, h_links, v_links, iotas = self.unflatten_spin_network_sample(
            sample,
            self.shape,
        )
        n_irreps = self.tables.block_id_lookup.shape[3]
        if bool(
            jnp.any(matter < 0)
            | jnp.any(matter >= self.phys_dim)
            | jnp.any(h_links < 0)
            | jnp.any(h_links >= n_irreps)
            | jnp.any(v_links < 0)
            | jnp.any(v_links >= n_irreps)
            | jnp.any(iotas < 0)
            | jnp.any(iotas >= self.tables.max_iotas)
        ):
            raise ValueError("Invalid non-Abelian local block in sample.")
        block_ids = self._active_block_ids_from_fields(matter, h_links, v_links, iotas)
        if bool(jnp.any(block_ids < 0)):
            raise ValueError("Invalid non-Abelian local block in sample.")
        return block_ids

    def _active_block_ids_from_fields(
        self,
        matter: jax.Array,
        h_links: jax.Array,
        v_links: jax.Array,
        iotas: jax.Array,
    ) -> jax.Array:
        return active_block_ids_from_fields(
            self.tables.block_id_lookup,
            matter,
            h_links,
            v_links,
            iotas,
            self.shape,
        )

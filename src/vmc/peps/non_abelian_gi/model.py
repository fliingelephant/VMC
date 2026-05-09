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
from vmc.peps.non_abelian_gi.builders import (
    build_horizontal_hopping_matrix_tables,
    build_plaquette_link_transitions,
    build_plaquette_matrix_tables,
    build_pure_gauge_tables,
    build_vertical_hopping_matrix_tables,
)
from vmc.peps.non_abelian_gi.contraction import (
    active_block_ids_from_fields,
    non_abelian_gi_apply,
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
        self.plaquette_link_transitions = build_plaquette_link_transitions(
            self.gauge_group
        )
        self.plaquette_matrix_tables = build_plaquette_matrix_tables(
            self.gauge_group,
            self.tables,
        )
        n_rows, n_cols = self.shape
        if self.phys_dim > 1:
            self.horizontal_hopping_matrix_tables = (
                build_horizontal_hopping_matrix_tables(
                    self.gauge_group,
                    self.tables,
                )
            )
            self.vertical_hopping_matrix_tables = build_vertical_hopping_matrix_tables(
                self.gauge_group,
                self.tables,
            )
        else:
            self.horizontal_hopping_matrix_tables = tuple(() for _ in range(n_rows))
            self.vertical_hopping_matrix_tables = tuple(() for _ in range(n_rows - 1))
        self.random_init_sweeps = int(
            getattr(self.gauge_group, "random_init_sweeps", 1)
        )
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

    @staticmethod
    def flatten_sample(
        h_links: jax.Array,
        v_links: jax.Array,
        iotas: jax.Array,
    ) -> jax.Array:
        return jnp.concatenate(
            [
                h_links.reshape(-1),
                v_links.reshape(-1),
                iotas.reshape(-1),
            ],
            axis=0,
        ).astype(jnp.int32)

    @staticmethod
    def flatten_matter_sample(
        matter: jax.Array,
        h_links: jax.Array,
        v_links: jax.Array,
        iotas: jax.Array,
    ) -> jax.Array:
        return jnp.concatenate(
            [
                matter.reshape(-1),
                h_links.reshape(-1),
                v_links.reshape(-1),
                iotas.reshape(-1),
            ],
            axis=0,
        ).astype(jnp.int32)

    @staticmethod
    def unflatten_sample(
        sample: jax.Array,
        shape: tuple[int, int],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        n_rows, n_cols = shape
        num_h = n_rows * (n_cols - 1)
        num_v = (n_rows - 1) * n_cols
        h_links = sample[:num_h].reshape((n_rows, n_cols - 1))
        v_links = sample[num_h : num_h + num_v].reshape((n_rows - 1, n_cols))
        iotas = sample[num_h + num_v :].reshape(shape)
        return h_links, v_links, iotas

    @staticmethod
    def unflatten_matter_sample(
        sample: jax.Array,
        shape: tuple[int, int],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        n_rows, n_cols = shape
        num_sites = n_rows * n_cols
        num_h = n_rows * (n_cols - 1)
        num_v = (n_rows - 1) * n_cols
        matter = sample[:num_sites].reshape(shape)
        offset = num_sites
        h_links = sample[offset : offset + num_h].reshape((n_rows, n_cols - 1))
        offset += num_h
        v_links = sample[offset : offset + num_v].reshape((n_rows - 1, n_cols))
        iotas = sample[offset + num_v :].reshape(shape)
        return matter, h_links, v_links, iotas

    @staticmethod
    def unflatten_spin_network_sample(
        sample: jax.Array,
        shape: tuple[int, int],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        n_rows, n_cols = shape
        pure_size = n_rows * (n_cols - 1) + (n_rows - 1) * n_cols + n_rows * n_cols
        if sample.size == pure_size:
            h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_sample(sample, shape)
            matter = jnp.zeros(shape, dtype=sample.dtype)
            return matter, h_links, v_links, iotas
        return NonAbelianGIPEPS.unflatten_matter_sample(sample, shape)

    def all_zero_sample(self) -> jax.Array:
        n_rows, n_cols = self.shape
        if self.phys_dim > 1:
            matter = self._particle_number_matter(
                jnp.arange(n_rows * n_cols, dtype=jnp.int32)
            )
            h_links, v_links = self._matter_string_links(matter)
            iotas = jnp.zeros(self.shape, dtype=jnp.int32)
            return self.flatten_matter_sample(matter, h_links, v_links, iotas)
        return self.flatten_sample(
            jnp.zeros((n_rows, n_cols - 1), dtype=jnp.int32),
            jnp.zeros((n_rows - 1, n_cols), dtype=jnp.int32),
            jnp.zeros(self.shape, dtype=jnp.int32),
        )

    def random_physical_configuration(
        self,
        key: jax.Array,
        n_samples: int = 1,
    ) -> jax.Array:
        if self.phys_dim > 1:
            keys = jax.random.split(key, n_samples)
            return jax.vmap(self._single_random_matter_configuration)(keys)
        sample = self.all_zero_sample()
        n_rows, n_cols = self.shape
        if n_rows < 2 or n_cols < 2 or self.random_init_sweeps <= 0:
            return jnp.broadcast_to(sample, (n_samples, sample.size))
        keys = jax.random.split(key, n_samples)
        return jax.vmap(self._single_random_physical_configuration)(keys)

    def _single_random_physical_configuration(self, key: jax.Array) -> jax.Array:
        h_links, v_links, iotas = self.unflatten_sample(
            self.all_zero_sample(), self.shape
        )
        for _ in range(self.random_init_sweeps):
            for row in range(self.shape[0] - 1):
                for col in range(self.shape[1] - 1):
                    key, update_key = jax.random.split(key)
                    h_links, v_links, iotas = self._random_plaquette_update(
                        update_key,
                        h_links,
                        v_links,
                        iotas,
                        row=row,
                        col=col,
                    )
        return self.flatten_sample(h_links, v_links, iotas)

    def _single_random_matter_configuration(self, key: jax.Array) -> jax.Array:
        n_sites = self.shape[0] * self.shape[1]
        permutation = jax.random.permutation(key, n_sites)
        matter = self._particle_number_matter(permutation)
        h_links, v_links = self._matter_string_links(matter)
        iotas = jnp.zeros(self.shape, dtype=jnp.int32)
        return self.flatten_matter_sample(matter, h_links, v_links, iotas)

    def _particle_number_matter(self, site_order: jax.Array) -> jax.Array:
        if self.matter_numbers != (0, 1):
            raise NotImplementedError(
                "Fixed particle-number initialization currently requires "
                "matter_numbers=(0, 1)."
            )
        matter = jnp.zeros((self.shape[0] * self.shape[1],), dtype=jnp.int32)
        matter = matter.at[site_order[: self.particle_number]].set(1)
        return matter.reshape(self.shape)

    def _matter_string_links(self, matter: jax.Array) -> tuple[jax.Array, jax.Array]:
        if self.matter_irreps != (0, 1):
            raise NotImplementedError(
                "Matter initialization currently supports singlet/fundamental SU(2) matter."
            )
        n_rows, n_cols = self.shape
        h_links = jnp.zeros((n_rows, n_cols - 1), dtype=jnp.int32)
        v_links = jnp.zeros((n_rows - 1, n_cols), dtype=jnp.int32)
        occupied = jnp.where(matter.reshape(-1) == 1, size=self.particle_number)[0]
        for pair_idx in range(0, self.particle_number, 2):
            start = occupied[pair_idx]
            end = occupied[pair_idx + 1]
            h_mask, v_mask = _path_masks(self.shape, start, end)
            h_links = (h_links + h_mask) % 2
            v_links = (v_links + v_mask) % 2
        return h_links, v_links

    def _random_plaquette_update(
        self,
        key: jax.Array,
        h_links: jax.Array,
        v_links: jax.Array,
        iotas: jax.Array,
        *,
        row: int,
        col: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        active_block_ids = self._active_block_ids_from_fields(
            jnp.zeros(self.shape, dtype=jnp.int32), h_links, v_links, iotas
        )
        input_blocks = (
            active_block_ids[row, col],
            active_block_ids[row, col + 1],
            active_block_ids[row + 1, col],
            active_block_ids[row + 1, col + 1],
        )
        table = self.plaquette_matrix_tables[row][col]
        if table.max_count == 0:
            return h_links, v_links, iotas
        start = table.starts[input_blocks]
        count = table.counts[input_blocks]
        arange = jnp.arange(table.max_count)
        valid = arange < count
        safe_indices = jnp.where(valid, start + arange, 0)
        weights = jnp.where(valid, table.proposal_weights[safe_indices], 0.0)
        norm = table.proposal_norms[input_blocks]
        can_update = norm > 0.0
        apply_key, outcome_key = jax.random.split(key)
        safe_norm = jnp.where(can_update, norm, 1.0)
        threshold = jax.random.uniform(outcome_key, dtype=weights.dtype) * safe_norm
        out_idx = jnp.minimum(
            jnp.sum(jnp.cumsum(weights) < threshold),
            weights.shape[0] - 1,
        ).astype(jnp.int32)
        output_blocks = table.output_block_ids[
            jnp.where(can_update, start + out_idx, 0)
        ]
        output_links = jnp.stack(
            [
                self.tables.j_r_by_block[row, col, output_blocks[0]],
                self.tables.j_d_by_block[row, col + 1, output_blocks[1]],
                self.tables.j_r_by_block[row + 1, col, output_blocks[2]],
                self.tables.j_d_by_block[row, col, output_blocks[0]],
            ]
        )
        output_iotas = jnp.stack(
            [
                self.tables.iota_by_block[row, col, output_blocks[0]],
                self.tables.iota_by_block[row, col + 1, output_blocks[1]],
                self.tables.iota_by_block[row + 1, col, output_blocks[2]],
                self.tables.iota_by_block[row + 1, col + 1, output_blocks[3]],
            ]
        )
        do_update = jax.random.bernoulli(apply_key) & can_update

        h_candidate = h_links.at[row, col].set(output_links[0])
        h_candidate = h_candidate.at[row + 1, col].set(output_links[2])
        v_candidate = v_links.at[row, col + 1].set(output_links[1])
        v_candidate = v_candidate.at[row, col].set(output_links[3])
        iota_candidate = iotas
        for (dr, dc), output_iota in zip(
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            output_iotas,
            strict=True,
        ):
            iota_candidate = iota_candidate.at[row + dr, col + dc].set(output_iota)
        return (
            jnp.where(do_update, h_candidate, h_links),
            jnp.where(do_update, v_candidate, v_links),
            jnp.where(do_update, iota_candidate, iotas),
        )

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


def _path_masks(
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

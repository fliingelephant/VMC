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
    build_plaquette_link_transitions,
    build_plaquette_matrix_tables,
    build_pure_gauge_tables,
)
from vmc.peps.non_abelian_gi.contraction import non_abelian_gi_apply
from vmc.utils.utils import random_tensor

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = ["NonAbelianGIPEPS", "NonAbelianGIPEPSConfig"]


@dataclass(frozen=True)
class NonAbelianGIPEPSConfig:
    """Configuration for a pure-gauge non-Abelian GI-PEPS."""

    shape: tuple[int, int]
    gauge_group: Any
    D: int
    chi: int
    target_charge: Any = 0
    dtype: "DTypeLike" = jnp.complex128

    def __post_init__(self) -> None:
        n_rows, n_cols = self.shape
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("shape must have positive dimensions.")
        if self.D <= 0:
            raise ValueError("D must be positive.")
        if self.chi <= 0:
            raise ValueError("chi must be positive.")


class NonAbelianGIPEPS(nnx.Module):
    """Pure-gauge non-Abelian GI-PEPS with one sampled block per vertex."""

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
        self.dtype = jnp.dtype(config.dtype)

        self.tables = build_pure_gauge_tables(
            self.gauge_group,
            shape=self.shape,
            target_charge=config.target_charge,
        )
        self.plaquette_link_transitions = build_plaquette_link_transitions(
            self.gauge_group
        )
        self.plaquette_matrix_tables = build_plaquette_matrix_tables(
            self.gauge_group,
            self.tables,
        )
        self.random_init_sweeps = int(getattr(self.gauge_group, "random_init_sweeps", 1))
        self.n_irreps = int(self.tables.block_id_lookup.shape[2])
        if contraction_strategy is None:
            contraction_strategy = Variational(truncate_bond_dimension=self.chi)
        self.strategy = contraction_strategy

        n_rows, n_cols = self.shape
        self.params_per_site = tuple(
            up * down * left * right
            for r in range(n_rows)
            for c in range(n_cols)
            for up, down, left, right in [
                self._site_dims(r, c, n_rows, n_cols)
            ]
        )
        self.sliced_dims = tuple(
            self.tables.n_blocks(r, c)
            for r in range(n_rows)
            for c in range(n_cols)
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

    def all_zero_sample(self) -> jax.Array:
        n_rows, n_cols = self.shape
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
        sample = self.all_zero_sample()
        n_rows, n_cols = self.shape
        if n_rows < 2 or n_cols < 2 or self.random_init_sweeps <= 0:
            return jnp.broadcast_to(sample, (n_samples, sample.size))
        keys = jax.random.split(key, n_samples)
        return jax.vmap(self._single_random_physical_configuration)(keys)

    def _single_random_physical_configuration(self, key: jax.Array) -> jax.Array:
        h_links, v_links, iotas = self.unflatten_sample(self.all_zero_sample(), self.shape)
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
        active_block_ids = self._active_block_ids_from_links(h_links, v_links, iotas)
        input_blocks = (
            active_block_ids[row, col],
            active_block_ids[row, col + 1],
            active_block_ids[row + 1, col],
            active_block_ids[row + 1, col + 1],
        )
        table = self.plaquette_matrix_tables[row][col]
        weights = table.proposal_weights[input_blocks]
        norm = table.proposal_norms[input_blocks]
        can_update = norm > 0.0
        apply_key, outcome_key = jax.random.split(key)
        safe_norm = jnp.where(can_update, norm, 1.0)
        threshold = jax.random.uniform(outcome_key, dtype=weights.dtype) * safe_norm
        out_idx = jnp.minimum(
            jnp.sum(jnp.cumsum(weights) < threshold),
            weights.shape[0] - 1,
        ).astype(jnp.int32)
        slot = input_blocks + (out_idx,)
        output_links = table.output_links[slot]
        output_iotas = table.output_iotas[slot]
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
        """Return active vertex block ids for a pure-gauge sample."""
        h_links, v_links, iotas = self.unflatten_sample(sample, self.shape)
        if bool(
            jnp.any(h_links < 0)
            | jnp.any(h_links >= self.n_irreps)
            | jnp.any(v_links < 0)
            | jnp.any(v_links >= self.n_irreps)
            | jnp.any(iotas < 0)
            | jnp.any(iotas >= self.tables.max_iotas)
        ):
            raise ValueError("Invalid non-Abelian local block in sample.")
        block_ids = self._active_block_ids_from_links(h_links, v_links, iotas)
        if bool(jnp.any(block_ids < 0)):
            raise ValueError("Invalid non-Abelian local block in sample.")
        return block_ids

    def _active_block_ids_unchecked(self, sample: jax.Array) -> jax.Array:
        h_links, v_links, iotas = self.unflatten_sample(sample, self.shape)
        return self._active_block_ids_from_links(h_links, v_links, iotas)

    def _active_block_ids_from_links(
        self,
        h_links: jax.Array,
        v_links: jax.Array,
        iotas: jax.Array,
    ) -> jax.Array:
        lookup = self.tables.block_id_lookup
        n_rows, n_cols = self.shape
        rows = []
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                row.append(
                    lookup[
                        r,
                        c,
                        h_links[r, c - 1] if c > 0 else 0,
                        v_links[r - 1, c] if r > 0 else 0,
                        h_links[r, c] if c < n_cols - 1 else 0,
                        v_links[r, c] if r < n_rows - 1 else 0,
                        iotas[r, c],
                    ]
                )
            rows.append(jnp.stack(row))
        return jnp.stack(rows)

"""Static sampled spin-network metadata shared by non-Abelian GI-PEPS backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PureGaugeTables:
    """Static vertex-block metadata.

    The concrete group backend owns the meaning of irrep labels and block
    objects. Generic kernels only require integer block lookup tables.
    """

    group: Any
    shape: tuple[int, int]
    phys_dim: int
    matter_irreps: tuple[int, ...]
    matter_numbers: tuple[int, ...]
    blocks: tuple[tuple[tuple[Any, ...], ...], ...]
    _block_ids: tuple[tuple[dict[tuple[int, int, int, int, int, int], int], ...], ...]
    max_iotas: int
    block_id_lookup: jax.Array
    matter_state_by_block: jax.Array
    j_l_by_block: jax.Array
    j_u_by_block: jax.Array
    j_r_by_block: jax.Array
    j_d_by_block: jax.Array
    iota_by_block: jax.Array

    @classmethod
    def from_blocks(
        cls,
        *,
        group: Any,
        shape: tuple[int, int],
        matter_irreps: tuple[int, ...],
        matter_numbers: tuple[int, ...],
        blocks: tuple[tuple[tuple[Any, ...], ...], ...],
    ) -> "PureGaugeTables":
        n_rows, n_cols = shape
        phys_dim = len(matter_irreps)
        max_iotas = max(
            block.iota + 1
            for row in blocks
            for site_blocks in row
            for block in site_blocks
        )
        n_irreps = len(group.irreps())
        max_blocks = max(len(site_blocks) for row in blocks for site_blocks in row)
        block_id_lookup = np.full(
            (
                n_rows,
                n_cols,
                phys_dim,
                n_irreps,
                n_irreps,
                n_irreps,
                n_irreps,
                max_iotas,
            ),
            -1,
            dtype=np.int32,
        )
        matter_state_by_block = np.full(
            (n_rows, n_cols, max_blocks), -1, dtype=np.int32
        )
        j_l_by_block = np.full((n_rows, n_cols, max_blocks), -1, dtype=np.int32)
        j_u_by_block = np.full((n_rows, n_cols, max_blocks), -1, dtype=np.int32)
        j_r_by_block = np.full((n_rows, n_cols, max_blocks), -1, dtype=np.int32)
        j_d_by_block = np.full((n_rows, n_cols, max_blocks), -1, dtype=np.int32)
        iota_by_block = np.full((n_rows, n_cols, max_blocks), -1, dtype=np.int32)
        lookup_rows = []
        for r, row in enumerate(blocks):
            lookup_row = []
            for c, site_blocks in enumerate(row):
                lookup = {}
                for block_id, block in enumerate(site_blocks):
                    matter_state = block.matter_state
                    key = (
                        matter_state,
                        block.j_l,
                        block.j_u,
                        block.j_r,
                        block.j_d,
                        block.iota,
                    )
                    lookup[key] = block_id
                    block_id_lookup[(r, c, *key)] = block_id
                    matter_state_by_block[r, c, block_id] = matter_state
                    j_l_by_block[r, c, block_id] = block.j_l
                    j_u_by_block[r, c, block_id] = block.j_u
                    j_r_by_block[r, c, block_id] = block.j_r
                    j_d_by_block[r, c, block_id] = block.j_d
                    iota_by_block[r, c, block_id] = block.iota
                lookup_row.append(lookup)
            lookup_rows.append(tuple(lookup_row))
        return cls(
            group=group,
            shape=shape,
            phys_dim=phys_dim,
            matter_irreps=tuple(int(irrep) for irrep in matter_irreps),
            matter_numbers=tuple(int(number) for number in matter_numbers),
            blocks=blocks,
            _block_ids=tuple(lookup_rows),
            max_iotas=max_iotas,
            block_id_lookup=jnp.asarray(block_id_lookup),
            matter_state_by_block=jnp.asarray(matter_state_by_block),
            j_l_by_block=jnp.asarray(j_l_by_block),
            j_u_by_block=jnp.asarray(j_u_by_block),
            j_r_by_block=jnp.asarray(j_r_by_block),
            j_d_by_block=jnp.asarray(j_d_by_block),
            iota_by_block=jnp.asarray(iota_by_block),
        )

    def active_legs(self, r: int, c: int) -> tuple[bool, bool, bool, bool]:
        """Return active ``(left, up, right, down)`` legs at a site."""
        self._validate_site(r, c)
        n_rows, n_cols = self.shape
        return (c > 0, r > 0, c < n_cols - 1, r < n_rows - 1)

    def n_blocks(self, r: int, c: int) -> int:
        """Return the number of valid vertex blocks at a site."""
        self._validate_site(r, c)
        return len(self.blocks[r][c])

    def block_id(
        self,
        r: int,
        c: int,
        left: int,
        up: int,
        right: int,
        down: int,
        iota: int,
        matter_state: int = 0,
    ) -> int:
        """Return the flat block id for a sampled local spin-network state."""
        self._validate_site(r, c)
        key = (matter_state, left, up, right, down, iota)
        if key not in self._block_ids[r][c]:
            raise ValueError(f"No vertex block for site {(r, c)} and key {key}.")
        return self._block_ids[r][c][key]

    def _validate_site(self, r: int, c: int) -> None:
        n_rows, n_cols = self.shape
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise IndexError(f"Site {(r, c)} is outside shape {self.shape}.")


@dataclass(frozen=True)
class FusionOutputs:
    """Static ``irrep x operator-irrep -> output irreps`` fusion table."""

    outputs: jax.Array  # (n_irreps, max_outputs) int32, -1 padded
    counts: jax.Array  # (n_irreps,) int32


def pack_fusion_outputs(
    n_irreps: int,
    outputs_by_irrep: Mapping[int, Sequence[int]],
) -> FusionOutputs:
    max_outputs = max((len(v) for v in outputs_by_irrep.values()), default=0)
    outputs = np.full((n_irreps, max_outputs), -1, dtype=np.int32)
    counts = np.zeros((n_irreps,), dtype=np.int32)
    for irrep, outs in outputs_by_irrep.items():
        counts[irrep] = len(outs)
        outputs[irrep, : len(outs)] = outs
    return FusionOutputs(outputs=jnp.asarray(outputs), counts=jnp.asarray(counts))


@dataclass(frozen=True)
class VertexFactorTable:
    """Per-(site, role, orientation) Schur factors of a string operator.

    Key = ``(input block, new sectors of the touched legs)``; the key arity is
    the number of touched legs (two for plaquette corners, one for hopping
    endpoints). ``out_blocks[start:start+count]`` lists the candidate output
    blocks for one key and ``factors`` their scalar reduced matrix elements,
    so ``<out|O|in> = kappa * prod_x factors_x``. Everything is O(n_blocks)
    per vertex; no array is indexed by more than one block axis.
    """

    group_starts: jax.Array  # (n_blocks, n_irreps[, n_irreps]) int32
    group_counts: jax.Array  # (n_blocks, n_irreps[, n_irreps]) int32
    max_candidates: int
    out_blocks: jax.Array  # (total,) int32
    factors: jax.Array  # (total,)
    w2_sums: jax.Array  # (n_blocks, n_irreps[, n_irreps]) float64

    @classmethod
    def from_rows(
        cls,
        n_blocks: int,
        n_irreps: int,
        key_arity: int,
        rows: Mapping[tuple[int, ...], Sequence[tuple[int, complex]]],
        *,
        factor_dtype: np.dtype | type,
    ) -> "VertexFactorTable":
        key_shape = (n_blocks,) + (n_irreps,) * key_arity
        starts = np.zeros(key_shape, dtype=np.int32)
        counts = np.zeros(key_shape, dtype=np.int32)
        w2_sums = np.zeros(key_shape, dtype=np.float64)
        max_candidates = max((len(v) for v in rows.values()), default=0)
        total = sum(len(v) for v in rows.values())
        out_blocks = np.full((total,), -1, dtype=np.int32)
        factors = np.zeros((total,), dtype=factor_dtype)
        cursor = 0
        for key, candidates in rows.items():
            starts[key] = cursor
            counts[key] = len(candidates)
            for out_block, factor in candidates:
                out_blocks[cursor] = out_block
                factors[cursor] = factor
                cursor += 1
            w2_sums[key] = np.sum(
                np.abs(factors[cursor - len(candidates) : cursor]) ** 2
            )
        return cls(
            group_starts=jnp.asarray(starts),
            group_counts=jnp.asarray(counts),
            max_candidates=max_candidates,
            out_blocks=jnp.asarray(out_blocks),
            factors=jnp.asarray(factors),
            w2_sums=jnp.asarray(w2_sums),
        )

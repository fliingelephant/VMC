"""Static sampled spin-network metadata shared by non-Abelian GI-PEPS backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class PureGaugeTables:
    """Static pure-gauge vertex-block metadata.

    The concrete group backend owns the meaning of irrep labels and block
    objects. Generic kernels only require integer block lookup tables.
    """

    group: Any
    shape: tuple[int, int]
    blocks: tuple[tuple[tuple[Any, ...], ...], ...]
    _block_ids: tuple[tuple[dict[tuple[int, int, int, int, int], int], ...], ...]
    max_iotas: int
    block_id_lookup: jax.Array

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
    ) -> int:
        """Return the flat block id for a sampled local spin-network state."""
        self._validate_site(r, c)
        key = (left, up, right, down, iota)
        if key not in self._block_ids[r][c]:
            raise ValueError(f"No vertex block for site {(r, c)} and key {key}.")
        return self._block_ids[r][c][key]

    def _validate_site(self, r: int, c: int) -> None:
        n_rows, n_cols = self.shape
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise IndexError(f"Site {(r, c)} is outside shape {self.shape}.")


@dataclass(frozen=True)
class PlaquetteLinkTransitions:
    """Dense static plaquette-link topology table.

    Link order is ``(top, right, bottom, left)``. The table records candidate
    output link irreps from acting with the representation carried by the
    plaquette operator. Magnetic matrix elements are intentionally not stored
    here.
    """

    output_links: jax.Array
    counts: jax.Array
    max_outputs: int

    def outputs(
        self,
        top: int,
        right: int,
        bottom: int,
        left: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return valid output link tuples for one input plaquette."""
        count = int(self.counts[top, right, bottom, left])
        links = self.output_links[top, right, bottom, left, :count]
        return tuple(tuple(int(value) for value in row) for row in links)


@dataclass(frozen=True)
class PlaquetteMatrixTable:
    """Static plaquette matrix elements indexed by four corner block ids."""

    output_links: jax.Array
    output_iotas: jax.Array
    output_block_ids: jax.Array
    matrix_elements: jax.Array
    proposal_weights: jax.Array
    proposal_norms: jax.Array
    counts: jax.Array
    max_outputs: int

    def find_outcome(
        self,
        input_blocks: tuple[int, int, int, int],
        output_blocks: tuple[int, int, int, int],
    ) -> int:
        """Return the outcome slot for ``input_blocks -> output_blocks``."""
        count = int(self.counts[input_blocks])
        for out_idx in range(count):
            candidate = self.output_block_ids[input_blocks + (out_idx,)]
            if tuple(int(x) for x in candidate) == output_blocks:
                return out_idx
        return -1

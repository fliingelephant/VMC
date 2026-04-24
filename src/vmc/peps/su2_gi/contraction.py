"""Pure-gauge SU(2) GI-PEPS contraction helpers."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["build_row_mpo"]


def build_row_mpo(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    tables: Any,
    *,
    row: int,
) -> tuple[jax.Array, ...]:
    """Build one row MPO from sampled SU(2) vertex blocks."""
    from vmc.peps.su2_gi.model import SU2GIPEPS

    h_links, v_links, iotas = SU2GIPEPS.unflatten_sample(sample, shape)
    n_rows, n_cols = shape
    lookup = tables.block_id_lookup
    return tuple(
        jnp.transpose(
            tensors[row][c][
                lookup[
                    row,
                    c,
                    h_links[row, c - 1] if c > 0 else 0,
                    v_links[row - 1, c] if row > 0 else 0,
                    h_links[row, c] if c < n_cols - 1 else 0,
                    v_links[row, c] if row < n_rows - 1 else 0,
                    iotas[row, c],
                ]
            ],
            (2, 3, 0, 1),
        )
        for c in range(n_cols)
    )

"""Sampled-block contraction helpers for non-Abelian GI-PEPS."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vmc.peps.common.contraction import _contract_bottom
from vmc.peps.non_abelian_gi.tables import PureGaugeTables

__all__ = ["build_row_mpo", "non_abelian_gi_apply"]


def build_row_mpo(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    tables: PureGaugeTables,
    *,
    row: int,
) -> tuple[jax.Array, ...]:
    """Build one row MPO by selecting one sampled vertex block per site."""
    from vmc.peps.non_abelian_gi.model import NonAbelianGIPEPS

    h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_sample(sample, shape)
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


def non_abelian_gi_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    tables: PureGaugeTables,
    strategy: object,
) -> jax.Array:
    """Compute a pure-gauge non-Abelian GI-PEPS amplitude."""
    n_cols = shape[1]
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(n_cols)
    )
    for row in range(shape[0]):
        boundary = strategy.apply(
            boundary,
            build_row_mpo(tensors, sample, shape, tables, row=row),
        )
    return _contract_bottom(boundary)

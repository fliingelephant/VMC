"""Sampled-block contraction helpers for non-Abelian GI-PEPS."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vmc.peps.common.contraction import _contract_bottom
from vmc.peps.non_abelian_gi.tables import PureGaugeTables

__all__ = [
    "active_block_ids_from_fields",
    "build_row_mpo",
    "non_abelian_gi_apply",
]


def active_block_ids_from_fields(
    block_id_lookup: jax.Array,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    shape: tuple[int, int],
) -> jax.Array:
    n_rows, n_cols = shape
    h_padded = jnp.pad(h_links, ((0, 0), (1, 1)))
    v_padded = jnp.pad(v_links, ((1, 1), (0, 0)))
    r_idx = jnp.arange(n_rows)[:, None]
    c_idx = jnp.arange(n_cols)[None, :]
    return block_id_lookup[
        r_idx,
        c_idx,
        matter,
        h_padded[:, :-1],
        v_padded[:-1, :],
        h_padded[:, 1:],
        v_padded[1:, :],
        iotas,
    ]


def build_row_mpo(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    block_id_lookup: jax.Array,
    *,
    row: int,
) -> tuple[jax.Array, ...]:
    """Build one row MPO by selecting one sampled vertex block per site."""
    from vmc.peps.non_abelian_gi.model import NonAbelianGIPEPS

    matter, h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_spin_network_sample(
        sample,
        shape,
    )
    active_block_ids = active_block_ids_from_fields(
        block_id_lookup,
        matter,
        h_links,
        v_links,
        iotas,
        shape,
    )
    n_cols = shape[1]
    return tuple(
        jnp.transpose(
            tensors[row][c][active_block_ids[row, c]],
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
            build_row_mpo(
                tensors,
                sample,
                shape,
                tables.block_id_lookup,
                row=row,
            ),
        )
    return _contract_bottom(boundary)

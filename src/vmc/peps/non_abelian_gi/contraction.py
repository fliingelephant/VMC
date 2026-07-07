"""Sample codec and sampled-block contraction for non-Abelian GI-PEPS."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vmc.peps.common.contraction import _contract_bottom
from vmc.peps.non_abelian_gi.tables import PureGaugeTables

__all__ = [
    "active_block_ids_from_fields",
    "active_block_ids_from_sample",
    "build_row_mpo",
    "build_row_mpo_from_blocks",
    "flatten_like_sample",
    "flatten_matter_sample",
    "flatten_sample",
    "non_abelian_gi_apply",
    "unflatten_matter_sample",
    "unflatten_sample",
    "unflatten_spin_network_sample",
]


def flatten_sample(
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
) -> jax.Array:
    return jnp.concatenate(
        [h_links.reshape(-1), v_links.reshape(-1), iotas.reshape(-1)],
        axis=0,
    ).astype(jnp.int32)


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


def flatten_like_sample(
    sample: jax.Array,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
) -> jax.Array:
    """Re-flatten spin-network fields with the layout of ``sample``."""
    if sample.size == h_links.size + v_links.size + iotas.size:
        return flatten_sample(h_links, v_links, iotas)
    return flatten_matter_sample(matter, h_links, v_links, iotas)


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


def unflatten_spin_network_sample(
    sample: jax.Array,
    shape: tuple[int, int],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_rows, n_cols = shape
    pure_size = n_rows * (n_cols - 1) + (n_rows - 1) * n_cols + n_rows * n_cols
    if sample.size == pure_size:
        h_links, v_links, iotas = unflatten_sample(sample, shape)
        return jnp.zeros(shape, dtype=sample.dtype), h_links, v_links, iotas
    return unflatten_matter_sample(sample, shape)


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


def active_block_ids_from_sample(
    block_id_lookup: jax.Array,
    sample: jax.Array,
    shape: tuple[int, int],
) -> jax.Array:
    return active_block_ids_from_fields(
        block_id_lookup,
        *unflatten_spin_network_sample(sample, shape),
        shape,
    )


def build_row_mpo(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    block_id_lookup: jax.Array,
    *,
    row: int,
) -> tuple[jax.Array, ...]:
    """Build one row MPO by selecting one sampled vertex block per site."""
    return build_row_mpo_from_blocks(
        tensors,
        active_block_ids_from_sample(block_id_lookup, sample, shape),
        row=row,
    )


def build_row_mpo_from_blocks(
    tensors: list[list[jax.Array]],
    active_block_ids: jax.Array,
    *,
    row: int,
) -> tuple[jax.Array, ...]:
    n_cols = active_block_ids.shape[1]
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
    active_block_ids = active_block_ids_from_sample(
        tables.block_id_lookup,
        sample,
        shape,
    )
    n_cols = shape[1]
    boundary = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(n_cols)
    )
    for row in range(shape[0]):
        boundary = strategy.apply(
            boundary,
            build_row_mpo_from_blocks(
                tensors,
                active_block_ids,
                row=row,
            ),
        )
    return _contract_bottom(boundary)

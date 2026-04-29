"""Generic kernel dispatch for sampled-block non-Abelian GI-PEPS."""
from __future__ import annotations

from itertools import product
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.operators.local_terms import DiagonalOperator, TransitionOperator, merge_operators
from vmc.peps.common.block_sparse import build_eval_schedule
from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _compute_right_envs,
    _contract_2row_1col,
    _contract_2row_2col,
    _contract_bottom,
)
from vmc.peps.common.energy import (
    _compute_all_row_gradients,
    _compute_right_envs_2row,
    _update_left_env_1row,
    _update_left_env_2row,
)
from vmc.peps.non_abelian_gi.contraction import build_row_mpo_from_lookup
from vmc.peps.non_abelian_gi.local_terms import (
    HorizontalLinkCasimirTerm,
    HorizontalMatterHoppingTerm,
    MatterNumberTerm,
    PlaquetteTerm,
    VerticalLinkCasimirTerm,
    VerticalMatterHoppingTerm,
    link_casimir_energy,
    matter_number_energy,
)
from vmc.peps.non_abelian_gi.model import NonAbelianGIPEPS
from vmc.peps.non_abelian_gi.tables import (
    HoppingMatrixTable,
    PlaquetteLinkTransitions,
    PlaquetteMatrixTable,
    PureGaugeTables,
)
from vmc.peps.standard.kernels import Cache, Context, LocalEstimates, build_mc_kernels
from vmc.utils.utils import _hastings_ratio, _metropolis_hastings_accept

__all__ = ["build_mc_kernels"]


class SpinNetworkTwoRowEnvs(NamedTuple):
    """Two-row contraction context for plaquette transition terms."""

    left_env: jax.Array
    right_envs: list[jax.Array]
    top_env: tuple
    bottom_env_next: tuple
    active_block_ids: jax.Array
    matrix_tables: tuple[tuple[PlaquetteMatrixTable, ...], ...]
    vertical_hopping_tables: tuple[tuple[HoppingMatrixTable, ...], ...]


class SpinNetworkRowEnvs(NamedTuple):
    """One-row contraction context for horizontal matter-hopping terms."""

    left_env: jax.Array
    right_envs: list[jax.Array]
    top_env: tuple
    bottom_env: tuple
    active_block_ids: jax.Array
    horizontal_hopping_tables: tuple[tuple[HoppingMatrixTable, ...], ...]


def _plaquette_candidate_samples(
    sample: jax.Array,
    *,
    row: int,
    col: int,
    shape: tuple[int, int],
    tables: PureGaugeTables,
    link_transitions: PlaquetteLinkTransitions,
) -> tuple[jax.Array, jax.Array]:
    """Build padded plaquette-output samples and a validity mask."""
    if link_transitions.max_outputs == 0:
        return (
            jnp.empty((0, sample.size), dtype=sample.dtype),
            jnp.zeros((0,), dtype=jnp.bool_),
        )

    matter, h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_spin_network_sample(
        sample,
        shape,
    )
    link_outputs = link_transitions.output_links
    lookup = tables.block_id_lookup
    plaquette_outputs = link_outputs[
        h_links[row, col],
        v_links[row, col + 1],
        h_links[row + 1, col],
        v_links[row, col],
    ]

    candidates = []
    valid = []
    corner_offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
    for link_out_idx in range(link_transitions.max_outputs):
        out_top, out_right, out_bottom, out_left = plaquette_outputs[link_out_idx]
        links_valid = out_top >= 0
        h_candidate = h_links.at[row, col].set(out_top)
        h_candidate = h_candidate.at[row + 1, col].set(out_bottom)
        v_candidate = v_links.at[row, col].set(out_left)
        v_candidate = v_candidate.at[row, col + 1].set(out_right)
        for candidate_iotas in product(range(tables.max_iotas), repeat=4):
            iota_candidate = iotas
            for (dr, dc), iota in zip(corner_offsets, candidate_iotas, strict=True):
                iota_candidate = iota_candidate.at[row + dr, col + dc].set(iota)
            block_valid = links_valid
            for dr, dc in corner_offsets:
                block_valid = block_valid & (
                    _site_block_id(
                        lookup,
                        matter,
                        h_candidate,
                        v_candidate,
                        iota_candidate,
                        shape,
                        row + dr,
                        col + dc,
                    )
                    >= 0
                )
            candidates.append(
                _flatten_like_sample(
                    sample,
                    matter,
                    h_candidate,
                    v_candidate,
                    iota_candidate,
                )
            )
            valid.append(block_valid)
    return jnp.stack(candidates), jnp.stack(valid)


def _site_block_id(
    lookup: jax.Array,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    shape: tuple[int, int],
    row: int,
    col: int,
) -> jax.Array:
    n_rows, n_cols = shape
    return lookup[
        row,
        col,
        matter[row, col],
        h_links[row, col - 1] if col > 0 else 0,
        v_links[row - 1, col] if row > 0 else 0,
        h_links[row, col] if col < n_cols - 1 else 0,
        v_links[row, col] if row < n_rows - 1 else 0,
        iotas[row, col],
    ]


def _active_block_ids(
    lookup: jax.Array,
    sample: jax.Array,
    shape: tuple[int, int],
) -> jax.Array:
    matter, h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_spin_network_sample(
        sample,
        shape,
    )
    n_rows, n_cols = shape
    return jnp.stack(
        [
            jnp.stack(
                [
                    _site_block_id(
                        lookup,
                        matter,
                        h_links,
                        v_links,
                        iotas,
                        shape,
                        row,
                        col,
                    )
                    for col in range(n_cols)
                ]
            )
            for row in range(n_rows)
        ]
    )


def _flatten_like_sample(
    sample: jax.Array,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
) -> jax.Array:
    pure_size = h_links.size + v_links.size + iotas.size
    if sample.size == pure_size:
        return NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)
    return NonAbelianGIPEPS.flatten_matter_sample(matter, h_links, v_links, iotas)


@dispatch
def _diagonal_energy(
    term: DiagonalOperator,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    del matter, h_links, v_links
    raise NotImplementedError(f"Unsupported non-Abelian diagonal term: {type(term)!r}.")


@_diagonal_energy.dispatch
def _diagonal_energy(
    term: HorizontalLinkCasimirTerm,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    del matter
    return link_casimir_energy(term, h_links, v_links)


@_diagonal_energy.dispatch
def _diagonal_energy(
    term: VerticalLinkCasimirTerm,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    del matter
    return link_casimir_energy(term, h_links, v_links)


@_diagonal_energy.dispatch
def _diagonal_energy(
    term: MatterNumberTerm,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    del h_links, v_links
    return matter_number_energy(term, matter)


@dispatch
def _transition_energy(
    term: TransitionOperator,
    envs: SpinNetworkTwoRowEnvs,
    tensors: Any,
) -> jax.Array:
    del envs, tensors
    raise NotImplementedError(f"Unsupported non-Abelian transition term: {type(term)!r}.")


@_transition_energy.dispatch
def _transition_energy(
    term: PlaquetteTerm,
    envs: SpinNetworkTwoRowEnvs,
    tensors: Any,
) -> jax.Array:
    row, col = term.row, term.col
    table = envs.matrix_tables[row][col]
    input_blocks = (
        envs.active_block_ids[row, col],
        envs.active_block_ids[row, col + 1],
        envs.active_block_ids[row + 1, col],
        envs.active_block_ids[row + 1, col + 1],
    )
    count = table.counts[input_blocks]
    start = table.starts[input_blocks]
    total = jnp.zeros((), dtype=tensors[row][col].dtype)
    for out_idx in range(table.max_count):
        valid = out_idx < count
        flat_idx = jnp.where(valid, start + out_idx, 0)
        output_block_ids = table.output_block_ids[flat_idx]
        safe_block_ids = jnp.where(valid, output_block_ids, jnp.zeros_like(output_block_ids))
        mpo_tl = _block_mpo(tensors[row][col], safe_block_ids[0])
        mpo_tr = _block_mpo(tensors[row][col + 1], safe_block_ids[1])
        mpo_bl = _block_mpo(tensors[row + 1][col], safe_block_ids[2])
        mpo_br = _block_mpo(tensors[row + 1][col + 1], safe_block_ids[3])
        amp = _contract_2row_2col(
            envs.left_env,
            envs.top_env,
            mpo_tl,
            mpo_bl,
            mpo_tr,
            mpo_br,
            envs.bottom_env_next,
            envs.right_envs[col + 1],
            col,
        )
        total = total + jnp.where(valid, table.matrix_elements[flat_idx] * amp, 0.0)
    return total


@_transition_energy.dispatch
def _transition_energy(
    term: HorizontalMatterHoppingTerm,
    envs: SpinNetworkRowEnvs,
    tensors: Any,
) -> jax.Array:
    row, col = term.row, term.col
    table = envs.horizontal_hopping_tables[row][col]
    input_blocks = (
        envs.active_block_ids[row, col],
        envs.active_block_ids[row, col + 1],
    )
    count = table.counts[input_blocks]
    start = table.starts[input_blocks]
    total = jnp.zeros((), dtype=tensors[row][col].dtype)
    for out_idx in range(table.max_count):
        valid = out_idx < count
        flat_idx = jnp.where(valid, start + out_idx, 0)
        output_block_ids = table.output_block_ids[flat_idx]
        safe_block_ids = jnp.where(valid, output_block_ids, jnp.zeros_like(output_block_ids))
        mpo_left = _block_mpo(tensors[row][col], safe_block_ids[0])
        mpo_right = _block_mpo(tensors[row][col + 1], safe_block_ids[1])
        amp = _contract_1row_2col(
            envs.left_env,
            envs.top_env,
            mpo_left,
            mpo_right,
            envs.bottom_env,
            envs.right_envs[col + 1],
            col,
        )
        total = total + jnp.where(valid, table.matrix_elements[flat_idx] * amp, 0.0)
    return total


@_transition_energy.dispatch
def _transition_energy(
    term: VerticalMatterHoppingTerm,
    envs: SpinNetworkTwoRowEnvs,
    tensors: Any,
) -> jax.Array:
    row, col = term.row, term.col
    table = envs.vertical_hopping_tables[row][col]
    input_blocks = (
        envs.active_block_ids[row, col],
        envs.active_block_ids[row + 1, col],
    )
    count = table.counts[input_blocks]
    start = table.starts[input_blocks]
    total = jnp.zeros((), dtype=tensors[row][col].dtype)
    for out_idx in range(table.max_count):
        valid = out_idx < count
        flat_idx = jnp.where(valid, start + out_idx, 0)
        output_block_ids = table.output_block_ids[flat_idx]
        safe_block_ids = jnp.where(valid, output_block_ids, jnp.zeros_like(output_block_ids))
        mpo_top = _block_mpo(tensors[row][col], safe_block_ids[0])
        mpo_bottom = _block_mpo(tensors[row + 1][col], safe_block_ids[1])
        amp = _contract_2row_1col(
            envs.left_env,
            envs.top_env[col],
            mpo_top,
            mpo_bottom,
            envs.bottom_env_next[col],
            envs.right_envs[col],
        )
        total = total + jnp.where(valid, table.matrix_elements[flat_idx] * amp, 0.0)
    return total


def _block_mpo(site_tensor: jax.Array, block_id: jax.Array) -> jax.Array:
    return jnp.transpose(site_tensor[block_id], (2, 3, 0, 1))


def _contract_1row_2col(
    left_env: jax.Array,
    top_env: tuple,
    mpo_left: jax.Array,
    mpo_right: jax.Array,
    bottom_env: tuple,
    right_env: jax.Array,
    col: int,
) -> jax.Array:
    return jnp.einsum(
        "ace,aub,cduv,evf,bgh,digw,fwj,hij->",
        left_env,
        top_env[col],
        mpo_left,
        bottom_env[col],
        top_env[col + 1],
        mpo_right,
        bottom_env[col + 1],
        right_env,
        optimize=[(0, 1), (0, 6), (0, 5), (0, 3), (1, 2), (1, 2), (0, 1)],
    )


def _contract_1row_1col(
    left_env: jax.Array,
    top: jax.Array,
    mpo: jax.Array,
    bottom: jax.Array,
    right_env: jax.Array,
) -> jax.Array:
    return jnp.einsum(
        "ace,aub,cduv,evf,bdf->",
        left_env,
        top,
        mpo,
        bottom,
        right_env,
        optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
    )


def _sample_table_outcome(
    key: jax.Array,
    weights: jax.Array,
    norm: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    can_propose = norm > 0.0
    key, sample_key = jax.random.split(key)
    safe_norm = jnp.where(can_propose, norm, 1.0)
    threshold = jax.random.uniform(sample_key, dtype=weights.dtype) * safe_norm
    out_idx = jnp.sum(jnp.cumsum(weights) < threshold)
    out_idx = jnp.minimum(out_idx, weights.shape[0] - 1).astype(jnp.int32)
    return key, out_idx, can_propose


def _proposal_ratio(
    table: PlaquetteMatrixTable | HoppingMatrixTable,
    input_blocks: tuple[jax.Array, ...],
    output_blocks: jax.Array,
    out_idx: jax.Array,
    can_propose: jax.Array,
) -> jax.Array:
    forward_norm = table.proposal_norms[input_blocks]
    forward_start = table.starts[input_blocks]
    forward_idx = jnp.where(can_propose, forward_start + out_idx, 0)
    forward_weight = table.proposal_weights[forward_idx]
    output_key = tuple(output_blocks[idx] for idx in range(len(input_blocks)))
    reverse_norm = table.proposal_norms[output_key]
    reverse_start = table.starts[output_key]
    reverse_count = table.counts[output_key]
    input_vec = jnp.stack(input_blocks).astype(table.output_block_ids.dtype)
    reverse_weight = jnp.zeros((), dtype=table.proposal_weights.dtype)
    for reverse_idx in range(table.max_count):
        valid = reverse_idx < reverse_count
        flat_idx = jnp.where(valid, reverse_start + reverse_idx, 0)
        reverse_outputs = table.output_block_ids[flat_idx]
        reverse_matches = jnp.all(reverse_outputs == input_vec)
        reverse_weight = reverse_weight + jnp.where(
            valid & reverse_matches,
            table.proposal_weights[flat_idx],
            0.0,
        )
    forward_prob = jnp.where(forward_norm > 0.0, forward_weight / forward_norm, 0.0)
    reverse_prob = jnp.where(reverse_norm > 0.0, reverse_weight / reverse_norm, 0.0)
    return jnp.where(
        can_propose,
        _hastings_ratio(forward_prob, reverse_prob),
        1.0,
    )


def _iota_candidate_blocks(
    block_id_lookup: jax.Array,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    shape: tuple[int, int],
    *,
    row: int,
    col: int,
) -> jax.Array:
    n_rows, n_cols = shape
    return jnp.stack(
        [
            block_id_lookup[
                row,
                col,
                matter[row, col],
                h_links[row, col - 1] if col > 0 else 0,
                v_links[row - 1, col] if row > 0 else 0,
                h_links[row, col] if col < n_cols - 1 else 0,
                v_links[row, col] if row < n_rows - 1 else 0,
                iota,
            ]
            for iota in range(block_id_lookup.shape[-1])
        ]
    )


def _iota_heatbath_sweep_row(
    key: jax.Array,
    tensors: Any,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo: tuple,
    top_env: tuple,
    bottom_env: tuple,
    block_id_lookup: jax.Array,
    shape: tuple[int, int],
    *,
    row: int,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple]:
    dtype = tensors[row][0].dtype
    right_envs = _compute_right_envs(top_env, row_mpo, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1), dtype=dtype)
    row_mpo_list = list(row_mpo)
    for col in range(len(row_mpo)):
        (
            key,
            iotas,
            active_block_ids,
            row_mpo_list,
        ) = _iota_heatbath_sweep_site(
            key,
            tensors,
            matter,
            h_links,
            v_links,
            iotas,
            active_block_ids,
            row_mpo_list,
            left_env,
            top_env,
            bottom_env,
            right_envs,
            block_id_lookup,
            shape,
            row=row,
            col=col,
        )
        left_env = _update_left_env_1row(
            left_env,
            top_env[col],
            row_mpo_list[col],
            bottom_env[col],
        )
    return key, iotas, active_block_ids, tuple(row_mpo_list)


def _iota_heatbath_sweep_site(
    key: jax.Array,
    tensors: Any,
    matter: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo: list[jax.Array],
    left_env: jax.Array,
    top_env: tuple,
    bottom_env: tuple,
    right_envs: list[jax.Array],
    block_id_lookup: jax.Array,
    shape: tuple[int, int],
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array, jax.Array, list]:
    current_block = active_block_ids[row, col]
    candidate_blocks = _iota_candidate_blocks(
        block_id_lookup,
        matter,
        h_links,
        v_links,
        shape,
        row=row,
        col=col,
    )
    valid = candidate_blocks >= 0
    safe_blocks = jnp.where(valid, candidate_blocks, current_block)
    weights = []
    for candidate_block in safe_blocks:
        candidate_mpo = _block_mpo(tensors[row][col], candidate_block)
        amp = _contract_1row_1col(
            left_env,
            top_env[col],
            candidate_mpo,
            bottom_env[col],
            right_envs[col],
        )
        weights.append(jnp.abs(amp) ** 2)
    weights = jnp.where(valid, jnp.stack(weights), 0.0)
    key, iota_idx, can_sample = _sample_table_outcome(key, weights, jnp.sum(weights))
    selected_block = candidate_blocks[iota_idx]
    safe_selected_block = jnp.where(can_sample, selected_block, current_block)
    selected_mpo = _block_mpo(tensors[row][col], safe_selected_block)

    row_mpo[col] = jnp.where(can_sample, selected_mpo, row_mpo[col])
    iotas = iotas.at[row, col].set(
        jnp.where(can_sample, iota_idx.astype(iotas.dtype), iotas[row, col])
    )
    active_block_ids = active_block_ids.at[row, col].set(safe_selected_block)
    return key, iotas, active_block_ids, row_mpo


def _table_row_weights(
    table: PlaquetteMatrixTable | HoppingMatrixTable,
    input_blocks: tuple[jax.Array, ...],
) -> jax.Array:
    start = table.starts[input_blocks]
    count = table.counts[input_blocks]
    return jnp.stack(
        [
            jnp.where(
                out_idx < count,
                table.proposal_weights[jnp.where(out_idx < count, start + out_idx, 0)],
                0.0,
            )
            for out_idx in range(table.max_count)
        ]
    )


def _plaquette_output_links_and_iotas(
    output_blocks: jax.Array,
    j_r_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.stack(
            [
                j_r_by_block[row, col, output_blocks[0]],
                j_d_by_block[row, col + 1, output_blocks[1]],
                j_r_by_block[row + 1, col, output_blocks[2]],
                j_d_by_block[row, col, output_blocks[0]],
            ]
        ),
        jnp.stack(
            [
                iota_by_block[row, col, output_blocks[0]],
                iota_by_block[row, col + 1, output_blocks[1]],
                iota_by_block[row + 1, col, output_blocks[2]],
                iota_by_block[row + 1, col + 1, output_blocks[3]],
            ]
        ),
    )


def _horizontal_hopping_output_fields(
    output_blocks: jax.Array,
    matter_state_by_block: jax.Array,
    j_r_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.stack(
            [
                matter_state_by_block[row, col, output_blocks[0]],
                matter_state_by_block[row, col + 1, output_blocks[1]],
            ]
        ),
        j_r_by_block[row, col, output_blocks[0]],
        jnp.stack(
            [
                iota_by_block[row, col, output_blocks[0]],
                iota_by_block[row, col + 1, output_blocks[1]],
            ]
        ),
    )


def _vertical_hopping_output_fields(
    output_blocks: jax.Array,
    matter_state_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.stack(
            [
                matter_state_by_block[row, col, output_blocks[0]],
                matter_state_by_block[row + 1, col, output_blocks[1]],
            ]
        ),
        j_d_by_block[row, col, output_blocks[0]],
        jnp.stack(
            [
                iota_by_block[row, col, output_blocks[0]],
                iota_by_block[row + 1, col, output_blocks[1]],
            ]
        ),
    )


def _plaquette_sweep_row_pair(
    key: jax.Array,
    tensors: Any,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo0: tuple,
    row_mpo1: tuple,
    top_env: tuple,
    bottom_env_next: tuple,
    matrix_tables: tuple[tuple[PlaquetteMatrixTable, ...], ...],
    j_r_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple, tuple]:
    dtype = tensors[row][0].dtype
    right_envs = _compute_right_envs_2row(
        top_env,
        row_mpo0,
        row_mpo1,
        bottom_env_next,
        dtype,
    )
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
    row_mpo0_list = list(row_mpo0)
    row_mpo1_list = list(row_mpo1)
    for col in range(len(row_mpo0) - 1):
        (
            key,
            h_links,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0_list,
            row_mpo1_list,
        ) = _plaquette_sweep_site(
            key,
            tensors,
            h_links,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0_list,
            row_mpo1_list,
            left_env,
            top_env,
            bottom_env_next,
            right_envs,
            matrix_tables[row][col],
            j_r_by_block,
            j_d_by_block,
            iota_by_block,
            row=row,
            col=col,
        )
        left_env = _update_left_env_2row(
            left_env,
            top_env[col],
            row_mpo0_list[col],
            row_mpo1_list[col],
            bottom_env_next[col],
        )
    return (
        key,
        h_links,
        v_links,
        iotas,
        active_block_ids,
        tuple(row_mpo0_list),
        tuple(row_mpo1_list),
    )


def _horizontal_hopping_sweep_row(
    key: jax.Array,
    tensors: Any,
    matter: jax.Array,
    h_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo: tuple,
    top_env: tuple,
    bottom_env: tuple,
    hopping_tables: tuple[tuple[HoppingMatrixTable, ...], ...],
    matter_state_by_block: jax.Array,
    j_r_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple]:
    dtype = tensors[row][0].dtype
    right_envs = _compute_right_envs(top_env, row_mpo, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1), dtype=dtype)
    row_mpo_list = list(row_mpo)
    for col in range(len(row_mpo) - 1):
        (
            key,
            matter,
            h_links,
            iotas,
            active_block_ids,
            row_mpo_list,
        ) = _horizontal_hopping_sweep_site(
            key,
            tensors,
            matter,
            h_links,
            iotas,
            active_block_ids,
            row_mpo_list,
            left_env,
            top_env,
            bottom_env,
            right_envs,
            hopping_tables[row][col],
            matter_state_by_block,
            j_r_by_block,
            iota_by_block,
            row=row,
            col=col,
        )
        left_env = _update_left_env_1row(
            left_env,
            top_env[col],
            row_mpo_list[col],
            bottom_env[col],
        )
    return key, matter, h_links, iotas, active_block_ids, tuple(row_mpo_list)


def _horizontal_hopping_sweep_site(
    key: jax.Array,
    tensors: Any,
    matter: jax.Array,
    h_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo: list[jax.Array],
    left_env: jax.Array,
    top_env: tuple,
    bottom_env: tuple,
    right_envs: list[jax.Array],
    table: HoppingMatrixTable,
    matter_state_by_block: jax.Array,
    j_r_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, list]:
    if table.max_count == 0:
        return key, matter, h_links, iotas, active_block_ids, row_mpo
    input_blocks = (active_block_ids[row, col], active_block_ids[row, col + 1])
    weights = _table_row_weights(table, input_blocks)
    key, out_idx, can_propose = _sample_table_outcome(
        key,
        weights,
        table.proposal_norms[input_blocks],
    )
    start = table.starts[input_blocks]
    flat_idx = jnp.where(can_propose, start + out_idx, 0)
    output_blocks = table.output_block_ids[flat_idx]
    output_matter, output_link, output_iotas = _horizontal_hopping_output_fields(
        output_blocks,
        matter_state_by_block,
        j_r_by_block,
        iota_by_block,
        row=row,
        col=col,
    )
    input_vec = jnp.stack(input_blocks).astype(output_blocks.dtype)
    safe_blocks = jnp.where(can_propose, output_blocks, input_vec)

    mpo_left = _block_mpo(tensors[row][col], safe_blocks[0])
    mpo_right = _block_mpo(tensors[row][col + 1], safe_blocks[1])
    amp_current = _contract_1row_2col(
        left_env,
        top_env,
        row_mpo[col],
        row_mpo[col + 1],
        bottom_env,
        right_envs[col + 1],
        col,
    )
    amp_proposed = _contract_1row_2col(
        left_env,
        top_env,
        mpo_left,
        mpo_right,
        bottom_env,
        right_envs[col + 1],
        col,
    )
    key, accept = _metropolis_hastings_accept(
        key,
        jnp.abs(amp_current) ** 2,
        jnp.abs(amp_proposed) ** 2,
        proposal_ratio=_proposal_ratio(table, input_blocks, safe_blocks, out_idx, can_propose),
    )
    accept = accept & can_propose

    row_mpo[col] = jnp.where(accept, mpo_left, row_mpo[col])
    row_mpo[col + 1] = jnp.where(accept, mpo_right, row_mpo[col + 1])
    matter_candidate = matter.at[row, col].set(output_matter[0])
    matter_candidate = matter_candidate.at[row, col + 1].set(output_matter[1])
    h_candidate = h_links.at[row, col].set(output_link)
    iota_candidate = iotas.at[row, col].set(output_iotas[0])
    iota_candidate = iota_candidate.at[row, col + 1].set(output_iotas[1])
    block_candidate = active_block_ids.at[row, col].set(output_blocks[0])
    block_candidate = block_candidate.at[row, col + 1].set(output_blocks[1])
    return (
        key,
        jnp.where(accept, matter_candidate, matter),
        jnp.where(accept, h_candidate, h_links),
        jnp.where(accept, iota_candidate, iotas),
        jnp.where(accept, block_candidate, active_block_ids),
        row_mpo,
    )


def _vertical_hopping_sweep_row_pair(
    key: jax.Array,
    tensors: Any,
    matter: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo0: tuple,
    row_mpo1: tuple,
    top_env: tuple,
    bottom_env_next: tuple,
    hopping_tables: tuple[tuple[HoppingMatrixTable, ...], ...],
    matter_state_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple, tuple]:
    dtype = tensors[row][0].dtype
    right_envs = _compute_right_envs_2row(
        top_env,
        row_mpo0,
        row_mpo1,
        bottom_env_next,
        dtype,
    )
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
    row_mpo0_list = list(row_mpo0)
    row_mpo1_list = list(row_mpo1)
    for col in range(len(row_mpo0)):
        (
            key,
            matter,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0_list,
            row_mpo1_list,
        ) = _vertical_hopping_sweep_site(
            key,
            tensors,
            matter,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0_list,
            row_mpo1_list,
            left_env,
            top_env,
            bottom_env_next,
            right_envs,
            hopping_tables[row][col],
            matter_state_by_block,
            j_d_by_block,
            iota_by_block,
            row=row,
            col=col,
        )
        left_env = _update_left_env_2row(
            left_env,
            top_env[col],
            row_mpo0_list[col],
            row_mpo1_list[col],
            bottom_env_next[col],
        )
    return key, matter, v_links, iotas, active_block_ids, tuple(row_mpo0_list), tuple(row_mpo1_list)


def _vertical_hopping_sweep_site(
    key: jax.Array,
    tensors: Any,
    matter: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo0: list[jax.Array],
    row_mpo1: list[jax.Array],
    left_env: jax.Array,
    top_env: tuple,
    bottom_env_next: tuple,
    right_envs: list[jax.Array],
    table: HoppingMatrixTable,
    matter_state_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, list, list]:
    if table.max_count == 0:
        return key, matter, v_links, iotas, active_block_ids, row_mpo0, row_mpo1
    input_blocks = (active_block_ids[row, col], active_block_ids[row + 1, col])
    weights = _table_row_weights(table, input_blocks)
    key, out_idx, can_propose = _sample_table_outcome(
        key,
        weights,
        table.proposal_norms[input_blocks],
    )
    start = table.starts[input_blocks]
    flat_idx = jnp.where(can_propose, start + out_idx, 0)
    output_blocks = table.output_block_ids[flat_idx]
    output_matter, output_link, output_iotas = _vertical_hopping_output_fields(
        output_blocks,
        matter_state_by_block,
        j_d_by_block,
        iota_by_block,
        row=row,
        col=col,
    )
    input_vec = jnp.stack(input_blocks).astype(output_blocks.dtype)
    safe_blocks = jnp.where(can_propose, output_blocks, input_vec)

    mpo_top = _block_mpo(tensors[row][col], safe_blocks[0])
    mpo_bottom = _block_mpo(tensors[row + 1][col], safe_blocks[1])
    amp_current = _contract_2row_1col(
        left_env,
        top_env[col],
        row_mpo0[col],
        row_mpo1[col],
        bottom_env_next[col],
        right_envs[col],
    )
    amp_proposed = _contract_2row_1col(
        left_env,
        top_env[col],
        mpo_top,
        mpo_bottom,
        bottom_env_next[col],
        right_envs[col],
    )
    key, accept = _metropolis_hastings_accept(
        key,
        jnp.abs(amp_current) ** 2,
        jnp.abs(amp_proposed) ** 2,
        proposal_ratio=_proposal_ratio(table, input_blocks, safe_blocks, out_idx, can_propose),
    )
    accept = accept & can_propose

    row_mpo0[col] = jnp.where(accept, mpo_top, row_mpo0[col])
    row_mpo1[col] = jnp.where(accept, mpo_bottom, row_mpo1[col])
    matter_candidate = matter.at[row, col].set(output_matter[0])
    matter_candidate = matter_candidate.at[row + 1, col].set(output_matter[1])
    v_candidate = v_links.at[row, col].set(output_link)
    iota_candidate = iotas.at[row, col].set(output_iotas[0])
    iota_candidate = iota_candidate.at[row + 1, col].set(output_iotas[1])
    block_candidate = active_block_ids.at[row, col].set(output_blocks[0])
    block_candidate = block_candidate.at[row + 1, col].set(output_blocks[1])
    return (
        key,
        jnp.where(accept, matter_candidate, matter),
        jnp.where(accept, v_candidate, v_links),
        jnp.where(accept, iota_candidate, iotas),
        jnp.where(accept, block_candidate, active_block_ids),
        row_mpo0,
        row_mpo1,
    )


def _plaquette_sweep_site(
    key: jax.Array,
    tensors: Any,
    h_links: jax.Array,
    v_links: jax.Array,
    iotas: jax.Array,
    active_block_ids: jax.Array,
    row_mpo0: list[jax.Array],
    row_mpo1: list[jax.Array],
    left_env: jax.Array,
    top_env: tuple,
    bottom_env_next: tuple,
    right_envs: list[jax.Array],
    table: PlaquetteMatrixTable,
    j_r_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, list, list]:
    if table.max_count == 0:
        return key, h_links, v_links, iotas, active_block_ids, row_mpo0, row_mpo1
    input_blocks = (
        active_block_ids[row, col],
        active_block_ids[row, col + 1],
        active_block_ids[row + 1, col],
        active_block_ids[row + 1, col + 1],
    )
    weights = _table_row_weights(table, input_blocks)
    key, out_idx, can_propose = _sample_table_outcome(
        key,
        weights,
        table.proposal_norms[input_blocks],
    )
    start = table.starts[input_blocks]
    flat_idx = jnp.where(can_propose, start + out_idx, 0)
    output_blocks = table.output_block_ids[flat_idx]
    output_links, output_iotas = _plaquette_output_links_and_iotas(
        output_blocks,
        j_r_by_block,
        j_d_by_block,
        iota_by_block,
        row=row,
        col=col,
    )
    input_vec = jnp.stack(input_blocks).astype(output_blocks.dtype)
    safe_blocks = jnp.where(can_propose, output_blocks, input_vec)

    mpo_tl = _block_mpo(tensors[row][col], safe_blocks[0])
    mpo_tr = _block_mpo(tensors[row][col + 1], safe_blocks[1])
    mpo_bl = _block_mpo(tensors[row + 1][col], safe_blocks[2])
    mpo_br = _block_mpo(tensors[row + 1][col + 1], safe_blocks[3])
    amp_current = _contract_2row_2col(
        left_env,
        top_env,
        row_mpo0[col],
        row_mpo1[col],
        row_mpo0[col + 1],
        row_mpo1[col + 1],
        bottom_env_next,
        right_envs[col + 1],
        col,
    )
    amp_proposed = _contract_2row_2col(
        left_env,
        top_env,
        mpo_tl,
        mpo_bl,
        mpo_tr,
        mpo_br,
        bottom_env_next,
        right_envs[col + 1],
        col,
    )
    key, accept = _metropolis_hastings_accept(
        key,
        jnp.abs(amp_current) ** 2,
        jnp.abs(amp_proposed) ** 2,
        proposal_ratio=_proposal_ratio(table, input_blocks, safe_blocks, out_idx, can_propose),
    )
    accept = accept & can_propose

    row_mpo0[col] = jnp.where(accept, mpo_tl, row_mpo0[col])
    row_mpo0[col + 1] = jnp.where(accept, mpo_tr, row_mpo0[col + 1])
    row_mpo1[col] = jnp.where(accept, mpo_bl, row_mpo1[col])
    row_mpo1[col + 1] = jnp.where(accept, mpo_br, row_mpo1[col + 1])

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
    block_candidate = active_block_ids
    for (dr, dc), output_block in zip(
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        output_blocks,
        strict=True,
    ):
        block_candidate = block_candidate.at[row + dr, col + dc].set(output_block)
    h_links = jnp.where(accept, h_candidate, h_links)
    v_links = jnp.where(accept, v_candidate, v_links)
    iotas = jnp.where(accept, iota_candidate, iotas)
    active_block_ids = jnp.where(accept, block_candidate, active_block_ids)
    return key, h_links, v_links, iotas, active_block_ids, row_mpo0, row_mpo1


@build_mc_kernels.dispatch
def build_mc_kernels(
    model: NonAbelianGIPEPS,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Any, Any, Any]:
    """Build init_cache/transition/estimate kernels for sampled-block GI-PEPS."""
    shape = model.shape
    n_rows, n_cols = shape
    strategy = model.strategy
    tables = model.tables
    block_id_lookup = tables.block_id_lookup
    matter_state_by_block = tables.matter_state_by_block
    j_r_by_block = tables.j_r_by_block
    j_d_by_block = tables.j_d_by_block
    iota_by_block = tables.iota_by_block
    matrix_tables = model.plaquette_matrix_tables
    horizontal_hopping_tables = model.horizontal_hopping_matrix_tables
    vertical_hopping_tables = model.vertical_hopping_matrix_tables
    all_operators = (operator,) + observables
    bucketed_terms, coeff_structure = merge_operators(
        all_operators,
        shape,
        eval_span=type(model).eval_span,
    )
    has_time_dep = any(s is not None for s in coeff_structure.schedules)
    static_coeffs = None if has_time_dep else coeff_structure.build_coeffs()
    eval_schedule = build_eval_schedule(bucketed_terms, type(model).eval_span)
    has_matter_hopping_terms = any(
        isinstance(term, (HorizontalMatterHoppingTerm, VerticalMatterHoppingTerm))
        for row_passes in bucketed_terms.rows
        for _dr, cols in row_passes
        for col_terms in cols
        for term, _contributions in col_terms
    )
    if has_matter_hopping_terms and model.phys_dim == 1:
        raise ValueError("Matter hopping terms require phys_dim > 1.")

    def build_bottom_envs(tensors: Any, sample: jax.Array) -> tuple:
        dtype = tensors[0][0].dtype
        envs = [None] * n_rows
        env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows - 1, -1, -1):
            envs[row] = env
            env = _apply_mpo_from_below(
                env,
                build_row_mpo_from_lookup(
                    tensors,
                    sample,
                    shape,
                    block_id_lookup,
                    row=row,
                ),
                strategy,
            )
        return tuple(envs)

    def init_cache(
        tensors: Any,
        samples: jax.Array,
        t: float | jax.Array | None = None,
    ) -> Cache:
        dynamic_coeffs = None if not has_time_dep else coeff_structure.build_coeffs(t)
        return Cache(
            bottom_envs=jax.vmap(lambda s: build_bottom_envs(tensors, s))(samples),
            coeffs=(
                None
                if dynamic_coeffs is None
                else jnp.broadcast_to(
                    dynamic_coeffs,
                    (samples.shape[0], dynamic_coeffs.shape[0]),
                )
            ),
        )

    def build_row_mpos(tensors: Any, sample: jax.Array) -> list[tuple[jax.Array, ...]]:
        return [
            build_row_mpo_from_lookup(
                tensors,
                sample,
                shape,
                block_id_lookup,
                row=row,
            )
            for row in range(n_rows)
        ]

    def context_from_row_mpos(
        row_mpos: list[tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, tuple]:
        top_envs = [None] * n_rows
        top_env = tuple(
            jnp.ones((1, 1, 1), dtype=row_mpos[0][0].dtype)
            for _ in range(n_cols)
        )
        for row in range(n_rows):
            top_envs[row] = top_env
            top_env = strategy.apply(top_env, row_mpos[row])
        return _contract_bottom(top_env), tuple(top_envs)

    def finish_transition(
        tensors: Any,
        sample: jax.Array,
        key: jax.Array,
        cache: Cache,
        matter: jax.Array,
        h_links: jax.Array,
        v_links: jax.Array,
        iotas: jax.Array,
        active_block_ids: jax.Array,
        row_mpos: list[tuple[jax.Array, ...]] | None = None,
    ) -> tuple[jax.Array, jax.Array, Context]:
        final_sample = _flatten_like_sample(sample, matter, h_links, v_links, iotas)
        if row_mpos is None:
            row_mpos = build_row_mpos(tensors, final_sample)
        if tables.max_iotas > 1:
            bottom_envs_iota = build_bottom_envs(tensors, final_sample)
            top_envs = [None] * n_rows
            top_env = tuple(
                jnp.ones((1, 1, 1), dtype=row_mpos[0][0].dtype)
                for _ in range(n_cols)
            )
            for row in range(n_rows):
                top_envs[row] = top_env
                (
                    key,
                    iotas,
                    active_block_ids,
                    row_mpos[row],
                ) = _iota_heatbath_sweep_row(
                    key,
                    tensors,
                    matter,
                    h_links,
                    v_links,
                    iotas,
                    active_block_ids,
                    row_mpos[row],
                    top_env,
                    bottom_envs_iota[row],
                    block_id_lookup,
                    shape,
                    row=row,
                )
                top_env = strategy.apply(top_env, row_mpos[row])
            final_sample = _flatten_like_sample(
                sample,
                matter,
                h_links,
                v_links,
                iotas,
            )
            return (
                final_sample,
                key,
                Context(
                    amp=_contract_bottom(top_env),
                    top_envs=tuple(top_envs),
                    coeffs=cache.coeffs,
                ),
            )

        amp, top_envs = context_from_row_mpos(row_mpos)
        return (
            final_sample,
            key,
            Context(amp=amp, top_envs=top_envs, coeffs=cache.coeffs),
        )

    def transition(
        tensors: Any,
        sample: jax.Array,
        key: jax.Array,
        cache: Cache,
    ) -> tuple[jax.Array, jax.Array, Context]:
        matter, h_links, v_links, iotas = (
            NonAbelianGIPEPS.unflatten_spin_network_sample(sample, shape)
        )
        active_block_ids = _active_block_ids(block_id_lookup, sample, shape)
        dtype = tensors[0][0].dtype

        if n_rows > 1 and n_cols > 1:
            row_mpos = build_row_mpos(tensors, sample)
            top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
            for row in range(n_rows - 1):
                (
                    key,
                    h_links,
                    v_links,
                    iotas,
                    active_block_ids,
                    row_mpos[row],
                    row_mpos[row + 1],
                ) = _plaquette_sweep_row_pair(
                    key,
                    tensors,
                    h_links,
                    v_links,
                    iotas,
                    active_block_ids,
                    row_mpos[row],
                    row_mpos[row + 1],
                    top_env,
                    cache.bottom_envs[row + 1],
                    matrix_tables,
                    j_r_by_block,
                    j_d_by_block,
                    iota_by_block,
                    row=row,
                )
                top_env = strategy.apply(top_env, row_mpos[row])
            if model.phys_dim == 1:
                return finish_transition(
                    tensors,
                    sample,
                    key,
                    cache,
                    matter,
                    h_links,
                    v_links,
                    iotas,
                    active_block_ids,
                    row_mpos,
                )

        if model.phys_dim == 1:
            return finish_transition(
                tensors,
                sample,
                key,
                cache,
                matter,
                h_links,
                v_links,
                iotas,
                active_block_ids,
            )

        sample_after_plaquettes = _flatten_like_sample(
            sample,
            matter,
            h_links,
            v_links,
            iotas,
        )
        row_mpos = build_row_mpos(tensors, sample_after_plaquettes)
        bottom_envs_h = build_bottom_envs(tensors, sample_after_plaquettes)
        top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows):
            (
                key,
                matter,
                h_links,
                iotas,
                active_block_ids,
                row_mpos[row],
            ) = _horizontal_hopping_sweep_row(
                key,
                tensors,
                matter,
                h_links,
                iotas,
                active_block_ids,
                row_mpos[row],
                top_env,
                bottom_envs_h[row],
                horizontal_hopping_tables,
                matter_state_by_block,
                j_r_by_block,
                iota_by_block,
                row=row,
            )
            top_env = strategy.apply(top_env, row_mpos[row])
        if n_rows == 1:
            return finish_transition(
                tensors,
                sample,
                key,
                cache,
                matter,
                h_links,
                v_links,
                iotas,
                active_block_ids,
                row_mpos,
            )

        sample_after_horizontal = _flatten_like_sample(
            sample,
            matter,
            h_links,
            v_links,
            iotas,
        )
        row_mpos = build_row_mpos(tensors, sample_after_horizontal)
        bottom_envs_v = build_bottom_envs(tensors, sample_after_horizontal)
        top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows - 1):
            (
                key,
                matter,
                v_links,
                iotas,
                active_block_ids,
                row_mpos[row],
                row_mpos[row + 1],
            ) = _vertical_hopping_sweep_row_pair(
                key,
                tensors,
                matter,
                v_links,
                iotas,
                active_block_ids,
                row_mpos[row],
                row_mpos[row + 1],
                top_env,
                bottom_envs_v[row + 1],
                vertical_hopping_tables,
                matter_state_by_block,
                j_d_by_block,
                iota_by_block,
                row=row,
            )
            top_env = strategy.apply(top_env, row_mpos[row])
        return finish_transition(
            tensors,
            sample,
            key,
            cache,
            matter,
            h_links,
            v_links,
            iotas,
            active_block_ids,
            row_mpos,
        )

    def estimate(
        tensors: Any,
        sample: jax.Array,
        context: Context,
    ) -> tuple[Cache, LocalEstimates]:
        matter, h_links, v_links, _iotas = (
            NonAbelianGIPEPS.unflatten_spin_network_sample(sample, shape)
        )
        bottom_envs = build_bottom_envs(tensors, sample)
        active_block_ids = _active_block_ids(block_id_lookup, sample, shape)
        row_mpos = tuple(
            build_row_mpo_from_lookup(
                tensors,
                sample,
                shape,
                block_id_lookup,
                row=r,
            )
            for r in range(n_rows)
        )
        grad_parts = []
        p_parts = [] if not full_gradient else None
        for r in range(n_rows):
            env_grads = _compute_all_row_gradients(
                context.top_envs[r],
                bottom_envs[r],
                row_mpos[r],
            )
            for c in range(n_cols):
                env_grad = env_grads[c]
                if full_gradient:
                    grad_full = jnp.zeros_like(jnp.asarray(tensors[r][c]))
                    grad_parts.append(
                        grad_full.at[active_block_ids[r, c]].set(env_grad).reshape(-1)
                    )
                    continue
                grad_parts.append(env_grad.reshape(-1))
                p_parts.append(
                    jnp.full(
                        (env_grad.size,),
                        active_block_ids[r, c],
                        dtype=jnp.int16,
                    )
                )

        coeffs = static_coeffs if context.coeffs is None else context.coeffs
        local_estimates = jnp.zeros(len(bucketed_terms), dtype=context.amp.dtype)
        for term, contributions in bucketed_terms.diagonal:
            term_energy = _diagonal_energy(term, matter, h_links, v_links)
            for op_idx, coeff_idx in contributions:
                coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                local_estimates = local_estimates.at[op_idx].add(coeff * term_energy)
        for r, row_passes in enumerate(eval_schedule.rows):
            for row_pass in row_passes:
                if row_pass.dr == 1:
                    top_env = context.top_envs[r]
                    bottom_env = bottom_envs[r]
                    right_envs = _compute_right_envs(
                        top_env,
                        row_mpos[r],
                        bottom_env,
                        tensors[r][0].dtype,
                    )
                    left_env = jnp.ones((1, 1, 1), dtype=tensors[r][0].dtype)
                    for c in range(n_cols):
                        envs = SpinNetworkRowEnvs(
                            left_env,
                            right_envs,
                            top_env,
                            bottom_env,
                            active_block_ids,
                            horizontal_hopping_tables,
                        )
                        for column in row_pass.columns[c]:
                            for term, contributions in column.terms:
                                term_energy = (
                                    _transition_energy(term, envs, tensors)
                                    / context.amp
                                )
                                for op_idx, coeff_idx in contributions:
                                    coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                                    local_estimates = local_estimates.at[op_idx].add(
                                        coeff * term_energy
                                    )
                        left_env = _update_left_env_1row(
                            left_env,
                            top_env[c],
                            row_mpos[r][c],
                            bottom_env[c],
                        )
                    continue
                if row_pass.dr != 2:
                    if any(row_pass.columns):
                        raise NotImplementedError(
                            "Non-Abelian transition evaluation supports only dr=1 and dr=2 terms."
                        )
                    continue
                if r >= n_rows - 1:
                    continue
                top_env = context.top_envs[r]
                bottom_env_next = bottom_envs[r + 1]
                right_envs = _compute_right_envs_2row(
                    top_env,
                    row_mpos[r],
                    row_mpos[r + 1],
                    bottom_env_next,
                    tensors[r][0].dtype,
                )
                left_env = jnp.ones((1, 1, 1, 1), dtype=tensors[r][0].dtype)
                for c in range(n_cols):
                    envs = SpinNetworkTwoRowEnvs(
                        left_env,
                        right_envs,
                        top_env,
                        bottom_env_next,
                        active_block_ids,
                        matrix_tables,
                        vertical_hopping_tables,
                    )
                    for column in row_pass.columns[c]:
                        for term, contributions in column.terms:
                            term_energy = (
                                _transition_energy(term, envs, tensors) / context.amp
                            )
                            for op_idx, coeff_idx in contributions:
                                coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                                local_estimates = local_estimates.at[op_idx].add(
                                    coeff * term_energy
                                )
                    left_env = _update_left_env_2row(
                        left_env,
                        top_env[c],
                        row_mpos[r][c],
                        row_mpos[r + 1][c],
                        bottom_env_next[c],
                    )
        active_slice_indices = None if full_gradient else jnp.concatenate(p_parts)
        return Cache(bottom_envs=bottom_envs, coeffs=context.coeffs), LocalEstimates(
            local_log_derivatives=jnp.concatenate(grad_parts) / context.amp,
            local_estimate=local_estimates,
            active_slice_indices=active_slice_indices,
            amp=context.amp,
        )

    return init_cache, transition, estimate

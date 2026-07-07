"""Generic kernel dispatch for sampled-block non-Abelian GI-PEPS."""

from __future__ import annotations

from itertools import product
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from plum import dispatch

from vmc.operators.local_terms import (
    DiagonalOperator,
    TransitionOperator,
    merge_operators,
)
from vmc.peps.common.block_sparse import build_eval_schedule
from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _compute_right_envs,
    _contract_1row_1col,
    _contract_1row_2col,
    _contract_2row_1col,
    _contract_2row_2col,
    _contract_bottom,
)
from vmc.peps.common.energy import (
    _compute_right_envs_2row,
    _compute_single_gradient,
    _update_left_env_1row,
    _update_left_env_2row,
)
from vmc.peps.non_abelian_gi.contraction import (
    active_block_ids_from_fields,
    active_block_ids_from_sample,
    build_row_mpo_from_blocks,
    flatten_like_sample,
    unflatten_spin_network_sample,
)
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
from vmc.peps.non_abelian_gi.factors import (
    HOPPING_KEY_LEGS,
    HOPPING_LEG_FWD,
    PLAQUETTE_KEY_LEGS,
    PLAQUETTE_LEG_FWD,
    HoppingFactorTables,
    PlaquetteFactorTables,
)
from vmc.peps.non_abelian_gi.model import NonAbelianGIPEPS
from vmc.peps.common.kernels import (
    Cache,
    Context,
    LocalEstimates,
    _assemble_log_derivatives,
    _broadcast_coeffs,
    build_mc_kernels,
)
from vmc.utils.utils import _metropolis_hastings_accept

__all__ = ["build_mc_kernels"]


class SpinNetworkTwoRowEnvs(NamedTuple):
    """Two-row contraction context for plaquette transition terms."""

    left_env: jax.Array
    right_envs: list[jax.Array]
    top_env: tuple
    bottom_env_next: tuple
    active_block_ids: jax.Array
    plaquette_tables: PlaquetteFactorTables
    hopping_tables: HoppingFactorTables | None
    j_r_by_block: jax.Array
    j_d_by_block: jax.Array


class SpinNetworkRowEnvs(NamedTuple):
    """One-row contraction context for horizontal matter-hopping terms."""

    left_env: jax.Array
    right_envs: list[jax.Array]
    top_env: tuple
    bottom_env: tuple
    active_block_ids: jax.Array
    hopping_tables: HoppingFactorTables | None
    j_r_by_block: jax.Array


def _window_combos(
    fuse_op,
    fuse_rev,
    vertex_tables,
    key_legs,
    leg_fwd,
    blocks,
    links,
    *,
    with_candidates,
):
    """Enumerate the (orientation x fusion-combo) outcomes of one window.

    ``vertex_tables`` holds one ``(fwd, bwd)`` VertexFactorTable pair per
    vertex, ``key_legs`` the moved-leg positions forming each vertex key and
    ``leg_fwd`` whether a moved leg fuses forward under the fwd orientation.
    Returns stacked ``legs (n, n_legs)`` and ``w2 (n,)`` (zero where
    fusion-invalid), plus masked ``factors``/``blocks`` ``(n, n_vertices, K)``
    when ``with_candidates``.
    """
    max_k = max(t.max_candidates for pair in vertex_tables for t in pair)
    slots = jnp.arange(max_k, dtype=jnp.int32)
    legs_parts, w2_parts, fac_parts, blk_parts = [], [], [], []
    for orient in range(2):
        maps = tuple(fuse_op if fwd == (orient == 0) else fuse_rev for fwd in leg_fwd)
        ks = np.array(
            list(product(*(range(m.outputs.shape[1]) for m in maps))),
            dtype=np.int32,
        ).reshape(-1, len(maps))
        valid = jnp.ones((ks.shape[0],), dtype=bool)
        for i, (m, link) in enumerate(zip(maps, links)):
            valid = valid & (ks[:, i] < m.counts[link])
        legs = jnp.where(
            valid[:, None],
            jnp.stack(
                [
                    m.outputs[link, ks[:, i]]
                    for i, (m, link) in enumerate(zip(maps, links))
                ],
                axis=1,
            ),
            0,
        )
        w2 = valid.astype(jnp.float64)
        facs, blks = [], []
        for vertex, (pair, key) in enumerate(zip(vertex_tables, key_legs)):
            table = pair[orient]
            idx = (blocks[vertex],) + tuple(legs[:, p] for p in key)
            w2 = w2 * table.w2_sums[idx]
            if not with_candidates:
                continue
            if table.max_candidates:
                count = jnp.where(valid, table.group_counts[idx], 0)
                ok = slots[None, :] < count[:, None]
                flat = jnp.where(ok, table.group_starts[idx][:, None] + slots, 0)
                facs.append(jnp.where(ok, table.factors[flat], 0.0))
                blks.append(jnp.where(ok, table.out_blocks[flat], 0))
            else:
                facs.append(jnp.zeros((ks.shape[0], max_k), dtype=jnp.complex128))
                blks.append(jnp.zeros((ks.shape[0], max_k), dtype=jnp.int32))
        legs_parts.append(legs)
        w2_parts.append(w2)
        if with_candidates:
            fac_parts.append(jnp.stack(facs, axis=1))
            blk_parts.append(jnp.stack(blks, axis=1))
    legs = jnp.concatenate(legs_parts, axis=0)
    w2 = jnp.concatenate(w2_parts, axis=0)
    if not with_candidates:
        return legs, w2
    return (
        legs,
        w2,
        jnp.concatenate(fac_parts, axis=0),
        jnp.concatenate(blk_parts, axis=0),
    )


def _plaquette_geometry(
    plaquette_tables, active_block_ids, j_r_by_block, j_d_by_block, row, col
):
    """Corner tables, in-blocks and current links of one plaquette window."""
    corner_tables = (
        plaquette_tables.tl[row][col],
        plaquette_tables.tr[row][col + 1],
        plaquette_tables.bl[row + 1][col],
        plaquette_tables.br[row + 1][col + 1],
    )
    blocks = (
        active_block_ids[row, col],
        active_block_ids[row, col + 1],
        active_block_ids[row + 1, col],
        active_block_ids[row + 1, col + 1],
    )
    links = (
        j_r_by_block[row, col, blocks[0]],
        j_d_by_block[row, col + 1, blocks[1]],
        j_r_by_block[row + 1, col, blocks[2]],
        j_d_by_block[row, col, blocks[0]],
    )
    return corner_tables, blocks, links


def _folded_mpo(site_tensor, factors, blocks):
    """lambda-weighted candidate MPO: ``sum_i factors[i] T[blocks[i]]``."""
    return jnp.einsum("k,kabcd->cdab", factors, site_tensor[blocks], optimize=True)


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
    raise NotImplementedError(
        f"Unsupported non-Abelian transition term: {type(term)!r}."
    )


@_transition_energy.dispatch
def _transition_energy(
    term: PlaquetteTerm,
    envs: SpinNetworkTwoRowEnvs,
    tensors: Any,
) -> jax.Array:
    row, col = term.row, term.col
    tabs = envs.plaquette_tables
    corner_tables, blocks, links = _plaquette_geometry(
        tabs, envs.active_block_ids, envs.j_r_by_block, envs.j_d_by_block, row, col
    )
    sites = (
        tensors[row][col],
        tensors[row][col + 1],
        tensors[row + 1][col],
        tensors[row + 1][col + 1],
    )
    total = jnp.zeros((), dtype=sites[0].dtype)
    if max(t.max_candidates for pair in corner_tables for t in pair) == 0:
        return total
    _, _, factors, out_blocks = _window_combos(
        tabs.fuse_op,
        tabs.fuse_rev,
        corner_tables,
        PLAQUETTE_KEY_LEGS,
        PLAQUETTE_LEG_FWD,
        blocks,
        links,
        with_candidates=True,
    )
    for combo in range(factors.shape[0]):
        mpos = [
            _folded_mpo(sites[v], factors[combo, v], out_blocks[combo, v])
            for v in range(4)
        ]
        total = total + _contract_2row_2col(
            envs.left_env,
            envs.top_env,
            mpos[0],
            mpos[2],
            mpos[1],
            mpos[3],
            envs.bottom_env_next,
            envs.right_envs[col + 1],
            col,
        )
    return total


@_transition_energy.dispatch
def _transition_energy(
    term: HorizontalMatterHoppingTerm,
    envs: SpinNetworkRowEnvs,
    tensors: Any,
) -> jax.Array:
    row, col = term.row, term.col
    hop = envs.hopping_tables
    endpoint_tables = (hop.h_src[row][col], hop.h_tgt[row][col + 1])
    total = jnp.zeros((), dtype=tensors[row][col].dtype)
    if max(t.max_candidates for pair in endpoint_tables for t in pair) == 0:
        return total
    blocks = (
        envs.active_block_ids[row, col],
        envs.active_block_ids[row, col + 1],
    )
    _, _, factors, out_blocks = _window_combos(
        hop.fuse_op,
        hop.fuse_rev,
        endpoint_tables,
        HOPPING_KEY_LEGS,
        HOPPING_LEG_FWD,
        blocks,
        (envs.j_r_by_block[row, col, blocks[0]],),
        with_candidates=True,
    )
    for combo in range(factors.shape[0]):
        total = total + _contract_1row_2col(
            envs.left_env,
            envs.top_env[col],
            _folded_mpo(tensors[row][col], factors[combo, 0], out_blocks[combo, 0]),
            envs.bottom_env[col],
            envs.top_env[col + 1],
            _folded_mpo(tensors[row][col + 1], factors[combo, 1], out_blocks[combo, 1]),
            envs.bottom_env[col + 1],
            envs.right_envs[col + 1],
        )
    return total


@_transition_energy.dispatch
def _transition_energy(
    term: VerticalMatterHoppingTerm,
    envs: SpinNetworkTwoRowEnvs,
    tensors: Any,
) -> jax.Array:
    row, col = term.row, term.col
    hop = envs.hopping_tables
    endpoint_tables = (hop.v_src[row][col], hop.v_tgt[row + 1][col])
    total = jnp.zeros((), dtype=tensors[row][col].dtype)
    if max(t.max_candidates for pair in endpoint_tables for t in pair) == 0:
        return total
    blocks = (
        envs.active_block_ids[row, col],
        envs.active_block_ids[row + 1, col],
    )
    _, _, factors, out_blocks = _window_combos(
        hop.fuse_op,
        hop.fuse_rev,
        endpoint_tables,
        HOPPING_KEY_LEGS,
        HOPPING_LEG_FWD,
        blocks,
        (envs.j_d_by_block[row, col, blocks[0]],),
        with_candidates=True,
    )
    for combo in range(factors.shape[0]):
        total = total + _contract_2row_1col(
            envs.left_env,
            envs.top_env[col],
            _folded_mpo(tensors[row][col], factors[combo, 0], out_blocks[combo, 0]),
            _folded_mpo(tensors[row + 1][col], factors[combo, 1], out_blocks[combo, 1]),
            envs.bottom_env_next[col],
            envs.right_envs[col],
        )
    return total


def _block_mpo(site_tensor: jax.Array, block_id: jax.Array) -> jax.Array:
    return jnp.transpose(site_tensor[block_id], (2, 3, 0, 1))


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


def _sample_window_outcome(
    key: jax.Array,
    fuse_op,
    fuse_rev,
    vertex_tables,
    key_legs,
    leg_fwd,
    blocks,
    links,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Ancestrally sample one window outcome.

    Combo (orientation + new legs) is drawn proportional to the product of
    per-vertex ``w2_sums`` and each vertex candidate proportional to
    ``|lambda|^2``, so the realized density is ``prod|lambda|^2 / Z(in)``,
    which is symmetric under conjugation — the Hastings ratio reduces to
    ``Z(in)/Z(out)``.  Returns ``(key, out_blocks, new_legs, can_propose,
    z_in)``.
    """
    legs, w2, factors, cand_blocks = _window_combos(
        fuse_op,
        fuse_rev,
        vertex_tables,
        key_legs,
        leg_fwd,
        blocks,
        links,
        with_candidates=True,
    )
    z_in = jnp.sum(w2)
    key, combo_idx, can_propose = _sample_table_outcome(key, w2, z_in)
    outs = []
    for vertex in range(len(vertex_tables)):
        weights = jnp.abs(factors[combo_idx, vertex]) ** 2
        key, cand_idx, _ = _sample_table_outcome(key, weights, jnp.sum(weights))
        outs.append(cand_blocks[combo_idx, vertex, cand_idx])
    return key, jnp.stack(outs), legs[combo_idx], can_propose, z_in


def _window_proposal_ratio(
    z_in: jax.Array,
    z_out: jax.Array,
    can_propose: jax.Array,
) -> jax.Array:
    return jnp.where(can_propose, z_in / jnp.where(z_out > 0.0, z_out, 1.0), 1.0)


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
    return block_id_lookup[
        row,
        col,
        matter[row, col],
        h_links[row, col - 1] if col > 0 else 0,
        v_links[row - 1, col] if row > 0 else 0,
        h_links[row, col] if col < n_cols - 1 else 0,
        v_links[row, col] if row < n_rows - 1 else 0,
        :,
    ]


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
    env_grad = _compute_single_gradient(
        left_env,
        right_envs[col],
        top_env[col],
        bottom_env[col],
    )
    candidate_mpos = jnp.transpose(tensors[row][col][safe_blocks], (0, 3, 4, 1, 2))
    amps = jnp.einsum("ncduv,uvcd->n", candidate_mpos, env_grad)
    weights = jnp.where(valid, jnp.abs(amps) ** 2, 0.0)
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
    amp_cur: jax.Array | None,
    plaquette_tables: PlaquetteFactorTables,
    j_r_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    tuple,
    tuple,
    jax.Array | None,
]:
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
    if amp_cur is None and len(row_mpo0) > 1:
        amp_cur = _contract_2row_2col(
            left_env,
            top_env,
            row_mpo0_list[0],
            row_mpo1_list[0],
            row_mpo0_list[1],
            row_mpo1_list[1],
            bottom_env_next,
            right_envs[1],
            0,
        )
    for col in range(len(row_mpo0) - 1):
        (
            key,
            h_links,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0_list,
            row_mpo1_list,
            left_env,
            amp_cur,
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
            amp_cur,
            plaquette_tables,
            j_r_by_block,
            j_d_by_block,
            iota_by_block,
            row=row,
            col=col,
        )
    return (
        key,
        h_links,
        v_links,
        iotas,
        active_block_ids,
        tuple(row_mpo0_list),
        tuple(row_mpo1_list),
        amp_cur,
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
    amp_cur: jax.Array | None,
    hopping_tables: HoppingFactorTables,
    matter_state_by_block: jax.Array,
    j_r_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple, jax.Array | None
]:
    dtype = tensors[row][0].dtype
    right_envs = _compute_right_envs(top_env, row_mpo, bottom_env, dtype)
    left_env = jnp.ones((1, 1, 1), dtype=dtype)
    row_mpo_list = list(row_mpo)
    if amp_cur is None and len(row_mpo) > 1:
        amp_cur = _contract_1row_2col(
            left_env,
            top_env[0],
            row_mpo_list[0],
            bottom_env[0],
            top_env[1],
            row_mpo_list[1],
            bottom_env[1],
            right_envs[1],
        )
    for col in range(len(row_mpo) - 1):
        (
            key,
            matter,
            h_links,
            iotas,
            active_block_ids,
            row_mpo_list,
            left_env,
            amp_cur,
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
            amp_cur,
            hopping_tables,
            matter_state_by_block,
            j_r_by_block,
            iota_by_block,
            row=row,
            col=col,
        )
    return key, matter, h_links, iotas, active_block_ids, tuple(row_mpo_list), amp_cur


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
    amp_cur: jax.Array,
    hopping_tables: HoppingFactorTables,
    matter_state_by_block: jax.Array,
    j_r_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, list, jax.Array, jax.Array
]:
    endpoint_tables = (
        hopping_tables.h_src[row][col],
        hopping_tables.h_tgt[row][col + 1],
    )
    if max(t.max_candidates for pair in endpoint_tables for t in pair) == 0:
        left_env = _update_left_env_1row(
            left_env,
            top_env[col],
            row_mpo[col],
            bottom_env[col],
        )
        return key, matter, h_links, iotas, active_block_ids, row_mpo, left_env, amp_cur
    input_blocks = (active_block_ids[row, col], active_block_ids[row, col + 1])
    key, output_blocks, output_legs, can_propose, z_in = _sample_window_outcome(
        key,
        hopping_tables.fuse_op,
        hopping_tables.fuse_rev,
        endpoint_tables,
        HOPPING_KEY_LEGS,
        HOPPING_LEG_FWD,
        input_blocks,
        (j_r_by_block[row, col, input_blocks[0]],),
    )
    output_link = output_legs[0]
    output_matter = jnp.stack(
        [
            matter_state_by_block[row, col, output_blocks[0]],
            matter_state_by_block[row, col + 1, output_blocks[1]],
        ]
    )
    output_iotas = jnp.stack(
        [
            iota_by_block[row, col, output_blocks[0]],
            iota_by_block[row, col + 1, output_blocks[1]],
        ]
    )
    input_vec = jnp.stack(input_blocks).astype(output_blocks.dtype)
    safe_blocks = jnp.where(can_propose, output_blocks, input_vec)
    _, w2_reverse = _window_combos(
        hopping_tables.fuse_op,
        hopping_tables.fuse_rev,
        endpoint_tables,
        HOPPING_KEY_LEGS,
        HOPPING_LEG_FWD,
        (safe_blocks[0], safe_blocks[1]),
        (output_link,),
        with_candidates=False,
    )

    mpo_left = _block_mpo(tensors[row][col], safe_blocks[0])
    mpo_right = _block_mpo(tensors[row][col + 1], safe_blocks[1])
    prefix_current = _update_left_env_1row(
        left_env,
        top_env[col],
        row_mpo[col],
        bottom_env[col],
    )
    prefix_proposed = _update_left_env_1row(
        left_env,
        top_env[col],
        mpo_left,
        bottom_env[col],
    )
    amp_proposed = _contract_1row_1col(
        prefix_proposed,
        top_env[col + 1],
        mpo_right,
        bottom_env[col + 1],
        right_envs[col + 1],
    )
    key, accept = _metropolis_hastings_accept(
        key,
        jnp.abs(amp_cur) ** 2,
        jnp.abs(amp_proposed) ** 2,
        proposal_ratio=_window_proposal_ratio(z_in, jnp.sum(w2_reverse), can_propose),
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
        jnp.where(accept, prefix_proposed, prefix_current),
        jnp.where(accept, amp_proposed, amp_cur),
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
    amp_cur: jax.Array | None,
    hopping_tables: HoppingFactorTables,
    matter_state_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    tuple,
    tuple,
    jax.Array | None,
]:
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
    if amp_cur is None and row_mpo0:
        amp_cur = _contract_2row_1col(
            left_env,
            top_env[0],
            row_mpo0_list[0],
            row_mpo1_list[0],
            bottom_env_next[0],
            right_envs[0],
        )
    for col in range(len(row_mpo0)):
        (
            key,
            matter,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0_list,
            row_mpo1_list,
            left_env,
            amp_cur,
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
            amp_cur,
            hopping_tables,
            matter_state_by_block,
            j_d_by_block,
            iota_by_block,
            row=row,
            col=col,
        )
    return (
        key,
        matter,
        v_links,
        iotas,
        active_block_ids,
        tuple(row_mpo0_list),
        tuple(row_mpo1_list),
        amp_cur,
    )


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
    amp_cur: jax.Array,
    hopping_tables: HoppingFactorTables,
    matter_state_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    list,
    list,
    jax.Array,
    jax.Array,
]:
    endpoint_tables = (
        hopping_tables.v_src[row][col],
        hopping_tables.v_tgt[row + 1][col],
    )
    if max(t.max_candidates for pair in endpoint_tables for t in pair) == 0:
        left_env = _update_left_env_2row(
            left_env,
            top_env[col],
            row_mpo0[col],
            row_mpo1[col],
            bottom_env_next[col],
        )
        return (
            key,
            matter,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0,
            row_mpo1,
            left_env,
            amp_cur,
        )
    input_blocks = (active_block_ids[row, col], active_block_ids[row + 1, col])
    key, output_blocks, output_legs, can_propose, z_in = _sample_window_outcome(
        key,
        hopping_tables.fuse_op,
        hopping_tables.fuse_rev,
        endpoint_tables,
        HOPPING_KEY_LEGS,
        HOPPING_LEG_FWD,
        input_blocks,
        (j_d_by_block[row, col, input_blocks[0]],),
    )
    output_link = output_legs[0]
    output_matter = jnp.stack(
        [
            matter_state_by_block[row, col, output_blocks[0]],
            matter_state_by_block[row + 1, col, output_blocks[1]],
        ]
    )
    output_iotas = jnp.stack(
        [
            iota_by_block[row, col, output_blocks[0]],
            iota_by_block[row + 1, col, output_blocks[1]],
        ]
    )
    input_vec = jnp.stack(input_blocks).astype(output_blocks.dtype)
    safe_blocks = jnp.where(can_propose, output_blocks, input_vec)
    _, w2_reverse = _window_combos(
        hopping_tables.fuse_op,
        hopping_tables.fuse_rev,
        endpoint_tables,
        HOPPING_KEY_LEGS,
        HOPPING_LEG_FWD,
        (safe_blocks[0], safe_blocks[1]),
        (output_link,),
        with_candidates=False,
    )

    mpo_top = _block_mpo(tensors[row][col], safe_blocks[0])
    mpo_bottom = _block_mpo(tensors[row + 1][col], safe_blocks[1])
    prefix_current = _update_left_env_2row(
        left_env,
        top_env[col],
        row_mpo0[col],
        row_mpo1[col],
        bottom_env_next[col],
    )
    prefix_proposed = _update_left_env_2row(
        left_env,
        top_env[col],
        mpo_top,
        mpo_bottom,
        bottom_env_next[col],
    )
    amp_proposed = jnp.einsum(
        "bryf,bryf->",
        prefix_proposed,
        right_envs[col],
        optimize=[(0, 1)],
    )
    key, accept = _metropolis_hastings_accept(
        key,
        jnp.abs(amp_cur) ** 2,
        jnp.abs(amp_proposed) ** 2,
        proposal_ratio=_window_proposal_ratio(z_in, jnp.sum(w2_reverse), can_propose),
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
        jnp.where(accept, prefix_proposed, prefix_current),
        jnp.where(accept, amp_proposed, amp_cur),
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
    amp_cur: jax.Array,
    plaquette_tables: PlaquetteFactorTables,
    j_r_by_block: jax.Array,
    j_d_by_block: jax.Array,
    iota_by_block: jax.Array,
    *,
    row: int,
    col: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    list,
    list,
    jax.Array,
    jax.Array,
]:
    corner_tables, input_blocks, links = _plaquette_geometry(
        plaquette_tables, active_block_ids, j_r_by_block, j_d_by_block, row, col
    )
    if max(t.max_candidates for pair in corner_tables for t in pair) == 0:
        left_env = _update_left_env_2row(
            left_env,
            top_env[col],
            row_mpo0[col],
            row_mpo1[col],
            bottom_env_next[col],
        )
        return (
            key,
            h_links,
            v_links,
            iotas,
            active_block_ids,
            row_mpo0,
            row_mpo1,
            left_env,
            amp_cur,
        )
    key, output_blocks, output_links, can_propose, z_in = _sample_window_outcome(
        key,
        plaquette_tables.fuse_op,
        plaquette_tables.fuse_rev,
        corner_tables,
        PLAQUETTE_KEY_LEGS,
        PLAQUETTE_LEG_FWD,
        input_blocks,
        links,
    )
    output_iotas = jnp.stack(
        [
            iota_by_block[row, col, output_blocks[0]],
            iota_by_block[row, col + 1, output_blocks[1]],
            iota_by_block[row + 1, col, output_blocks[2]],
            iota_by_block[row + 1, col + 1, output_blocks[3]],
        ]
    )
    input_vec = jnp.stack(input_blocks).astype(output_blocks.dtype)
    safe_blocks = jnp.where(can_propose, output_blocks, input_vec)
    _, w2_reverse = _window_combos(
        plaquette_tables.fuse_op,
        plaquette_tables.fuse_rev,
        corner_tables,
        PLAQUETTE_KEY_LEGS,
        PLAQUETTE_LEG_FWD,
        tuple(safe_blocks[v] for v in range(4)),
        tuple(output_links[i] for i in range(4)),
        with_candidates=False,
    )

    mpo_tl = _block_mpo(tensors[row][col], safe_blocks[0])
    mpo_tr = _block_mpo(tensors[row][col + 1], safe_blocks[1])
    mpo_bl = _block_mpo(tensors[row + 1][col], safe_blocks[2])
    mpo_br = _block_mpo(tensors[row + 1][col + 1], safe_blocks[3])
    prefix_current = _update_left_env_2row(
        left_env,
        top_env[col],
        row_mpo0[col],
        row_mpo1[col],
        bottom_env_next[col],
    )
    prefix_proposed = _update_left_env_2row(
        left_env,
        top_env[col],
        mpo_tl,
        mpo_bl,
        bottom_env_next[col],
    )
    amp_proposed = _contract_2row_1col(
        prefix_proposed,
        top_env[col + 1],
        mpo_tr,
        mpo_br,
        bottom_env_next[col + 1],
        right_envs[col + 1],
    )
    key, accept = _metropolis_hastings_accept(
        key,
        jnp.abs(amp_cur) ** 2,
        jnp.abs(amp_proposed) ** 2,
        proposal_ratio=_window_proposal_ratio(z_in, jnp.sum(w2_reverse), can_propose),
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
    return (
        key,
        h_links,
        v_links,
        iotas,
        active_block_ids,
        row_mpo0,
        row_mpo1,
        jnp.where(accept, prefix_proposed, prefix_current),
        jnp.where(accept, amp_proposed, amp_cur),
    )


@build_mc_kernels.dispatch
def build_mc_kernels(
    model: NonAbelianGIPEPS,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Any, Any, Any]:
    """Build init_cache/transition/estimate kernels for sampled-block GI-PEPS."""
    config = model.config
    shape = tuple(int(x) for x in config.shape)
    n_rows, n_cols = shape
    strategy = model.strategy
    tables = model.tables
    phys_dim = int(config.phys_dim)
    max_iotas = int(tables.max_iotas)
    block_id_lookup = tables.block_id_lookup
    matter_state_by_block = tables.matter_state_by_block
    j_r_by_block = tables.j_r_by_block
    j_d_by_block = tables.j_d_by_block
    iota_by_block = tables.iota_by_block
    plaquette_tables = model.plaquette_factor_tables
    hopping_tables = model.hopping_factor_tables
    total_active_params = int(sum(model.params_per_site))
    params_per_site = jnp.asarray(model.params_per_site, dtype=jnp.int32)
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
    if has_matter_hopping_terms and phys_dim == 1:
        raise ValueError("Matter hopping terms require phys_dim > 1.")

    def build_row_mpos(
        tensors: Any,
        active_block_ids: jax.Array,
    ) -> list[tuple[jax.Array, ...]]:
        return [
            build_row_mpo_from_blocks(tensors, active_block_ids, row=row)
            for row in range(n_rows)
        ]

    def build_bottom_envs_from_row_mpos(row_mpos: list[tuple[jax.Array, ...]]) -> tuple:
        dtype = row_mpos[0][0].dtype
        envs = [None] * n_rows
        env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows - 1, -1, -1):
            envs[row] = env
            env = _apply_mpo_from_below(env, row_mpos[row], strategy)
        return tuple(envs)

    def build_bottom_envs(tensors: Any, sample: jax.Array) -> tuple:
        return build_bottom_envs_from_row_mpos(
            build_row_mpos(
                tensors,
                active_block_ids_from_sample(block_id_lookup, sample, shape),
            )
        )

    def init_cache(
        tensors: Any,
        samples: jax.Array,
        t: float | jax.Array | None = None,
    ) -> Cache:
        return Cache(
            bottom_envs=jax.vmap(lambda s: build_bottom_envs(tensors, s))(samples),
            coeffs=_broadcast_coeffs(
                None if not has_time_dep else coeff_structure.build_coeffs(t),
                samples.shape[0],
            ),
        )

    def context_from_row_mpos(
        row_mpos: list[tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, tuple]:
        top_envs = [None] * n_rows
        top_env = tuple(
            jnp.ones((1, 1, 1), dtype=row_mpos[0][0].dtype) for _ in range(n_cols)
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
        row_mpos: list[tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, jax.Array, Context]:
        final_sample = flatten_like_sample(sample, matter, h_links, v_links, iotas)
        if max_iotas > 1:
            bottom_envs_iota = build_bottom_envs_from_row_mpos(row_mpos)
            top_envs = [None] * n_rows
            top_env = tuple(
                jnp.ones((1, 1, 1), dtype=row_mpos[0][0].dtype) for _ in range(n_cols)
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
            final_sample = flatten_like_sample(
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
        matter, h_links, v_links, iotas = unflatten_spin_network_sample(sample, shape)
        active_block_ids = active_block_ids_from_fields(
            block_id_lookup,
            matter,
            h_links,
            v_links,
            iotas,
            shape,
        )
        dtype = tensors[0][0].dtype
        row_mpos = build_row_mpos(tensors, active_block_ids)

        if n_rows > 1 and n_cols > 1:
            top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
            amp_cur = None
            for row in range(n_rows - 1):
                (
                    key,
                    h_links,
                    v_links,
                    iotas,
                    active_block_ids,
                    row_mpos[row],
                    row_mpos[row + 1],
                    amp_cur,
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
                    amp_cur,
                    plaquette_tables,
                    j_r_by_block,
                    j_d_by_block,
                    iota_by_block,
                    row=row,
                )
                top_env = strategy.apply(top_env, row_mpos[row])
            if phys_dim == 1:
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

        if phys_dim == 1:
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

        bottom_envs_h = build_bottom_envs_from_row_mpos(row_mpos)
        top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        amp_cur = None
        for row in range(n_rows):
            (
                key,
                matter,
                h_links,
                iotas,
                active_block_ids,
                row_mpos[row],
                amp_cur,
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
                amp_cur,
                hopping_tables,
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

        bottom_envs_v = build_bottom_envs_from_row_mpos(row_mpos)
        top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        amp_cur = None
        for row in range(n_rows - 1):
            (
                key,
                matter,
                v_links,
                iotas,
                active_block_ids,
                row_mpos[row],
                row_mpos[row + 1],
                amp_cur,
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
                amp_cur,
                hopping_tables,
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
        matter, h_links, v_links, iotas = unflatten_spin_network_sample(sample, shape)
        active_block_ids = active_block_ids_from_fields(
            block_id_lookup,
            matter,
            h_links,
            v_links,
            iotas,
            shape,
        )
        row_mpos = build_row_mpos(tensors, active_block_ids)
        coeffs = static_coeffs if context.coeffs is None else context.coeffs
        local_estimates = jnp.zeros(len(bucketed_terms), dtype=context.amp.dtype)
        for term, contributions in bucketed_terms.diagonal:
            term_energy = _diagonal_energy(term, matter, h_links, v_links)
            for op_idx, coeff_idx in contributions:
                coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                local_estimates = local_estimates.at[op_idx].add(coeff * term_energy)

        bottom_envs = [None] * n_rows
        env_grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        bottom_env = tuple(
            jnp.ones((1, 1, 1), dtype=tensors[0][0].dtype) for _ in range(n_cols)
        )
        next_row_mpo = None
        empty_columns = tuple(() for _ in range(n_cols))
        for r in range(n_rows - 1, -1, -1):
            bottom_envs[r] = bottom_env
            top_env = context.top_envs[r]
            row_mpo = row_mpos[r]
            dr1_columns = empty_columns
            other_row_passes = []
            for row_pass in eval_schedule.rows[r]:
                if row_pass.dr == 1:
                    dr1_columns = row_pass.columns
                    continue
                other_row_passes.append(row_pass)

            right_envs = _compute_right_envs(
                top_env,
                row_mpo,
                bottom_env,
                tensors[r][0].dtype,
            )
            left_env = jnp.ones((1, 1, 1), dtype=tensors[r][0].dtype)
            for c in range(n_cols):
                env_grad = _compute_single_gradient(
                    left_env,
                    right_envs[c],
                    top_env[c],
                    bottom_env[c],
                )
                env_grads[r][c] = env_grad
                if dr1_columns[c]:
                    envs = SpinNetworkRowEnvs(
                        left_env,
                        right_envs,
                        top_env,
                        bottom_env,
                        active_block_ids,
                        hopping_tables,
                        j_r_by_block,
                    )
                    for column in dr1_columns[c]:
                        for term, contributions in column.terms:
                            term_energy = (
                                _transition_energy(term, envs, tensors) / context.amp
                            )
                            for op_idx, coeff_idx in contributions:
                                coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                                local_estimates = local_estimates.at[op_idx].add(
                                    coeff * term_energy
                                )
                left_env = _update_left_env_1row(
                    left_env,
                    top_env[c],
                    row_mpo[c],
                    bottom_env[c],
                )

            for row_pass in other_row_passes:
                if row_pass.dr != 2:
                    if any(row_pass.columns):
                        raise NotImplementedError(
                            "Non-Abelian transition evaluation supports only dr=1 and dr=2 terms."
                        )
                    continue
                if r >= n_rows - 1:
                    continue
                if next_row_mpo is None:
                    raise NotImplementedError(
                        "Missing next-row MPO for non-Abelian dr=2 evaluation."
                    )
                bottom_env_next = bottom_envs[r + 1]
                right_envs_2row = _compute_right_envs_2row(
                    top_env,
                    row_mpo,
                    next_row_mpo,
                    bottom_env_next,
                    tensors[r][0].dtype,
                )
                left_env_2row = jnp.ones((1, 1, 1, 1), dtype=tensors[r][0].dtype)
                for c in range(n_cols):
                    envs = SpinNetworkTwoRowEnvs(
                        left_env_2row,
                        right_envs_2row,
                        top_env,
                        bottom_env_next,
                        active_block_ids,
                        plaquette_tables,
                        hopping_tables,
                        j_r_by_block,
                        j_d_by_block,
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
                    left_env_2row = _update_left_env_2row(
                        left_env_2row,
                        top_env[c],
                        row_mpo[c],
                        next_row_mpo[c],
                        bottom_env_next[c],
                    )

            bottom_env = _apply_mpo_from_below(bottom_env, row_mpo, strategy)
            next_row_mpo = row_mpo

        local_log_derivatives, active_slice_indices = _assemble_log_derivatives(
            tensors,
            params_per_site,
            total_active_params,
            shape,
            env_grads,
            active_block_ids.reshape(-1),
            context.amp,
            full_gradient=full_gradient,
        )
        return Cache(
            bottom_envs=tuple(bottom_envs), coeffs=context.coeffs
        ), LocalEstimates(
            local_log_derivatives=local_log_derivatives,
            local_estimate=local_estimates,
            active_slice_indices=active_slice_indices,
            amp=context.amp,
        )

    return init_cache, transition, estimate

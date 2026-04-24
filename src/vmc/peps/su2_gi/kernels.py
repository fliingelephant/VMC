"""SU(2) GI-PEPS kernel dispatch extension for the generic MC sampler."""
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
    _contract_2row_2col,
    _contract_bottom,
)
from vmc.peps.common.energy import (
    _compute_all_row_gradients,
    _compute_right_envs_2row,
    _update_left_env_2row,
)
from vmc.peps.standard.kernels import Cache, Context, LocalEstimates, build_mc_kernels
from vmc.peps.su2_gi.contraction import build_row_mpo
from vmc.peps.su2_gi.local_terms import (
    HorizontalLinkCasimirTerm,
    PlaquetteSU2Term,
    VerticalLinkCasimirTerm,
    link_casimir_energy,
)
from vmc.peps.su2_gi.group import (
    PlaquetteLinkTransitions,
    PlaquetteMatrixTable,
    PureGaugeTables,
)
from vmc.peps.su2_gi.model import SU2GIPEPS
from vmc.utils.utils import _hastings_ratio, _metropolis_hastings_accept

__all__ = ["build_mc_kernels"]


class SU2TwoRowEnvs(NamedTuple):
    """Two-row contraction context for SU(2) transition terms."""

    left_env: jax.Array
    right_envs: list[jax.Array]
    top_env: tuple
    bottom_env_next: tuple
    active_block_ids: jax.Array
    matrix_tables: tuple[tuple[PlaquetteMatrixTable, ...], ...]


def _has_transition_terms(rows: tuple) -> bool:
    return any(
        col_terms
        for row_passes in rows
        for _dr, cols in row_passes
        for col_terms in cols
    )


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

    h_links, v_links, iotas = SU2GIPEPS.unflatten_sample(sample, shape)
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
                SU2GIPEPS.flatten_sample(
                    h_candidate,
                    v_candidate,
                    iota_candidate,
                )
            )
            valid.append(block_valid)
    return jnp.stack(candidates), jnp.stack(valid)


def _site_block_id(
    lookup: jax.Array,
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
        h_links[row, col - 1] if col > 0 else 0,
        v_links[row - 1, col] if row > 0 else 0,
        h_links[row, col] if col < n_cols - 1 else 0,
        v_links[row, col] if row < n_rows - 1 else 0,
        iotas[row, col],
    ]


@dispatch
def _diagonal_energy(
    term: DiagonalOperator,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    del h_links, v_links
    raise NotImplementedError(f"Unsupported SU(2) diagonal term: {type(term)!r}.")


@_diagonal_energy.dispatch
def _diagonal_energy(
    term: HorizontalLinkCasimirTerm,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    return link_casimir_energy(term, h_links, v_links)


@_diagonal_energy.dispatch
def _diagonal_energy(
    term: VerticalLinkCasimirTerm,
    h_links: jax.Array,
    v_links: jax.Array,
) -> jax.Array:
    return link_casimir_energy(term, h_links, v_links)


@dispatch
def _transition_energy(
    term: TransitionOperator,
    envs: SU2TwoRowEnvs,
    tensors: Any,
) -> jax.Array:
    del envs, tensors
    raise NotImplementedError(f"Unsupported SU(2) transition term: {type(term)!r}.")


@_transition_energy.dispatch
def _transition_energy(
    term: PlaquetteSU2Term,
    envs: SU2TwoRowEnvs,
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
    total = jnp.zeros((), dtype=tensors[row][col].dtype)
    for out_idx in range(table.max_outputs):
        valid = out_idx < count
        slot = input_blocks + (out_idx,)
        output_block_ids = table.output_block_ids[slot]
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
        total = total + jnp.where(valid, table.matrix_elements[slot] * amp, 0.0)
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


def _proposal_ratio(
    table: PlaquetteMatrixTable,
    input_blocks: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    output_blocks: jax.Array,
    out_idx: jax.Array,
    can_propose: jax.Array,
) -> jax.Array:
    forward_norm = table.proposal_norms[input_blocks]
    forward_weight = table.proposal_weights[input_blocks + (out_idx,)]
    output_key = tuple(output_blocks[idx] for idx in range(4))
    reverse_norm = table.proposal_norms[output_key]
    reverse_outputs = table.output_block_ids[output_key]
    reverse_weights = table.proposal_weights[output_key]
    input_vec = jnp.stack(input_blocks).astype(reverse_outputs.dtype)
    reverse_matches = jnp.all(reverse_outputs == input_vec, axis=-1)
    reverse_weight = jnp.sum(jnp.where(reverse_matches, reverse_weights, 0.0))
    forward_prob = jnp.where(forward_norm > 0.0, forward_weight / forward_norm, 0.0)
    reverse_prob = jnp.where(reverse_norm > 0.0, reverse_weight / reverse_norm, 0.0)
    return jnp.where(
        can_propose,
        _hastings_ratio(forward_prob, reverse_prob),
        1.0,
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
    col_terms: tuple,
    matrix_tables: tuple[tuple[PlaquetteMatrixTable, ...], ...],
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
        if col_terms[col]:
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
    *,
    row: int,
    col: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, list, list]:
    if table.max_outputs == 0:
        return key, h_links, v_links, iotas, active_block_ids, row_mpo0, row_mpo1
    input_blocks = (
        active_block_ids[row, col],
        active_block_ids[row, col + 1],
        active_block_ids[row + 1, col],
        active_block_ids[row + 1, col + 1],
    )
    weights = table.proposal_weights[input_blocks]
    key, out_idx, can_propose = _sample_table_outcome(
        key,
        weights,
        table.proposal_norms[input_blocks],
    )
    slot = input_blocks + (out_idx,)
    output_blocks = table.output_block_ids[slot]
    output_links = table.output_links[slot]
    output_iotas = table.output_iotas[slot]
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
    model: SU2GIPEPS,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Any, Any, Any]:
    """Build pure-gauge SU(2) init_cache/transition/estimate kernels."""
    shape = model.shape
    n_rows, n_cols = shape
    strategy = model.strategy
    tables = model.tables
    all_operators = (operator,) + observables
    bucketed_terms, coeff_structure = merge_operators(
        all_operators,
        shape,
        eval_span=type(model).eval_span,
    )
    has_time_dep = any(s is not None for s in coeff_structure.schedules)
    static_coeffs = None if has_time_dep else coeff_structure.build_coeffs()
    eval_schedule = build_eval_schedule(bucketed_terms, type(model).eval_span)
    has_transition_terms = _has_transition_terms(bucketed_terms.rows)
    empty_cols = tuple(() for _ in range(n_cols))
    transition_cols_by_row = tuple(
        dict(row_passes).get(2, empty_cols)
        for row_passes in bucketed_terms.rows
    )
    unsupported_transition_spans = tuple(
        dr
        for row_passes in bucketed_terms.rows
        for dr, cols in row_passes
        if dr != 2 and any(cols)
    )
    if unsupported_transition_spans:
        raise NotImplementedError(
            "SU(2) transition sweep supports only dr=2 plaquette terms."
        )

    def build_bottom_envs(tensors: Any, sample: jax.Array) -> tuple:
        dtype = tensors[0][0].dtype
        envs = [None] * n_rows
        env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows - 1, -1, -1):
            envs[row] = env
            env = _apply_mpo_from_below(
                env,
                build_row_mpo(tensors, sample, shape, tables, row=row),
                strategy,
            )
        return tuple(envs)

    def build_top_envs_and_amp(tensors: Any, sample: jax.Array) -> tuple[jax.Array, tuple]:
        dtype = tensors[0][0].dtype
        top_envs = [None] * n_rows
        env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows):
            top_envs[row] = env
            env = strategy.apply(
                env,
                build_row_mpo(tensors, sample, shape, tables, row=row),
            )
        return _contract_bottom(env), tuple(top_envs)

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

    def transition(
        tensors: Any,
        sample: jax.Array,
        key: jax.Array,
        cache: Cache,
    ) -> tuple[jax.Array, jax.Array, Context]:
        if has_transition_terms:
            h_links, v_links, iotas = SU2GIPEPS.unflatten_sample(sample, shape)
            active_block_ids = model._active_block_ids_unchecked(sample)
            row_mpos = [
                build_row_mpo(tensors, sample, shape, tables, row=row)
                for row in range(n_rows)
            ]
            top_envs = [None] * n_rows
            top_env = tuple(jnp.ones((1, 1, 1), dtype=tensors[0][0].dtype) for _ in range(n_cols))
            for row in range(n_rows):
                top_envs[row] = top_env
                col_terms = transition_cols_by_row[row]
                if row < n_rows - 1 and any(col_terms):
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
                        col_terms,
                        model.plaquette_matrix_tables,
                        row=row,
                    )
                top_env = strategy.apply(top_env, row_mpos[row])
            return (
                SU2GIPEPS.flatten_sample(h_links, v_links, iotas),
                key,
                Context(
                    amp=_contract_bottom(top_env),
                    top_envs=tuple(top_envs),
                    coeffs=cache.coeffs,
                ),
            )
        amp, top_envs = build_top_envs_and_amp(tensors, sample)
        return sample, key, Context(amp=amp, top_envs=top_envs, coeffs=cache.coeffs)

    def estimate(
        tensors: Any,
        sample: jax.Array,
        context: Context,
    ) -> tuple[Cache, LocalEstimates]:
        h_links, v_links, _iotas = SU2GIPEPS.unflatten_sample(sample, shape)
        bottom_envs = build_bottom_envs(tensors, sample)
        active_block_ids = model._active_block_ids_unchecked(sample)
        row_mpos = tuple(
            build_row_mpo(tensors, sample, shape, tables, row=r)
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
            term_energy = _diagonal_energy(term, h_links, v_links)
            for op_idx, coeff_idx in contributions:
                coeff = 1.0 if coeffs is None else coeffs[coeff_idx]
                local_estimates = local_estimates.at[op_idx].add(coeff * term_energy)
        for r, row_passes in enumerate(eval_schedule.rows):
            for row_pass in row_passes:
                if row_pass.dr != 2:
                    if any(row_pass.columns):
                        raise NotImplementedError(
                            "SU(2) transition evaluation supports only dr=2 terms."
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
                    envs = SU2TwoRowEnvs(
                        left_env,
                        right_envs,
                        top_env,
                        bottom_env_next,
                        active_block_ids,
                        model.plaquette_matrix_tables,
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

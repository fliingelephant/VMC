"""Truncated SU(2) gauge-group backend for non-Abelian GI-PEPS."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math

import jax
import jax.numpy as jnp

import vmc.peps.non_abelian_gi.builders as builders
from vmc.peps.non_abelian_gi.tables import (
    HoppingMatrixTable,
    PlaquetteLinkTransitions,
    PlaquetteMatrixTable,
    PureGaugeTables,
)


@dataclass(frozen=True)
class SU2:
    """Truncated SU(2) irreps using integer labels ``j_twice = 2j``."""

    j_max_twice: int

    def __post_init__(self) -> None:
        if not isinstance(self.j_max_twice, int):
            raise ValueError("j_max_twice must be an integer.")
        if self.j_max_twice < 0:
            raise ValueError("j_max_twice must be non-negative.")

    @property
    def random_init_sweeps(self) -> int:
        return self.j_max_twice

    def irreps(self) -> tuple[int, ...]:
        """Return link irrep labels inside the Hilbert-space truncation."""
        return tuple(range(self.j_max_twice + 1))

    def dim(self, j_twice: int) -> int:
        """Return ``2j + 1`` for an allowed truncated irrep."""
        self._validate_link_irrep(j_twice)
        return j_twice + 1

    def casimir(self, j_twice: int) -> float:
        """Return ``j(j + 1)``."""
        self._validate_link_irrep(j_twice)
        return 0.25 * j_twice * (j_twice + 2)

    def fuse(self, a_twice: int, b_twice: int) -> tuple[int, ...]:
        """Return truncated SU(2) fusion outputs."""
        self._validate_link_irrep(a_twice)
        self._validate_link_irrep(b_twice)
        return tuple(
            c_twice
            for c_twice in _fuse_untruncated(a_twice, b_twice)
            if c_twice <= self.j_max_twice
        )

    def fundamental_link_outputs(self, j_twice: int) -> tuple[int, ...]:
        """Return link irreps appearing in ``j tensor 1/2`` within truncation."""
        self._validate_link_irrep(j_twice)
        return tuple(
            out_twice
            for out_twice in _fuse_untruncated(j_twice, 1)
            if out_twice <= self.j_max_twice
        )

    def tensor_product(self, a_twice: int, b_twice: int) -> tuple[int, ...]:
        """Return untruncated SU(2) tensor-product outputs.

        Vertex-internal fusion channels are not link Hilbert-space sectors, so
        they are not cut off by ``j_max_twice``.
        """
        self._validate_link_irrep(a_twice)
        self._validate_link_irrep(b_twice)
        return _fuse_untruncated(a_twice, b_twice)

    def _validate_link_irrep(self, j_twice: int) -> None:
        if not isinstance(j_twice, int):
            raise ValueError("SU(2) irrep labels must be integers.")
        if j_twice < 0:
            raise ValueError("Expected a valid SU(2) irrep label.")
        if j_twice > self.j_max_twice:
            raise ValueError("SU(2) irrep label must be within truncation.")


@dataclass(frozen=True, order=True)
class VertexBlock:
    """One vertex block in the sampled spin-network basis."""

    j_l: int
    j_u: int
    j_r: int
    j_d: int
    iota: int
    internal_irreps: tuple[int, ...]
    matter_state: int = 0
    matter_irrep: int = 0
    matter_number: int = 0


def _fuse_untruncated(a_twice: int, b_twice: int) -> tuple[int, ...]:
    return tuple(range(abs(a_twice - b_twice), a_twice + b_twice + 1, 2))


def clebsch_gordan(
    j1_twice: int,
    m1_twice: int,
    j2_twice: int,
    m2_twice: int,
    j_out_twice: int,
    m_out_twice: int,
) -> float:
    """Return a Condon-Shortley SU(2) Clebsch-Gordan coefficient."""
    if not (
        _is_valid_magnetic_label(j1_twice, m1_twice)
        and _is_valid_magnetic_label(j2_twice, m2_twice)
        and _is_valid_magnetic_label(j_out_twice, m_out_twice)
    ):
        return 0.0
    if m_out_twice != m1_twice + m2_twice:
        return 0.0
    if j_out_twice not in _fuse_untruncated(j1_twice, j2_twice):
        return 0.0

    triangle_prefactor = (
        (j_out_twice + 1)
        * math.factorial(_half_sum(j_out_twice, j1_twice, -j2_twice))
        * math.factorial(_half_sum(j_out_twice, -j1_twice, j2_twice))
        * math.factorial(_half_sum(j1_twice, j2_twice, -j_out_twice))
        / math.factorial(_half_sum(j1_twice, j2_twice, j_out_twice) + 1)
    )
    magnetic_prefactor = (
        math.factorial(_half_sum(j_out_twice, m_out_twice))
        * math.factorial(_half_sum(j_out_twice, -m_out_twice))
        * math.factorial(_half_sum(j1_twice, -m1_twice))
        * math.factorial(_half_sum(j1_twice, m1_twice))
        * math.factorial(_half_sum(j2_twice, -m2_twice))
        * math.factorial(_half_sum(j2_twice, m2_twice))
    )

    a = _half_sum(j1_twice, j2_twice, -j_out_twice)
    b = _half_sum(j1_twice, -m1_twice)
    c = _half_sum(j2_twice, m2_twice)
    d = _half_sum(j_out_twice, -j2_twice, m1_twice)
    e = _half_sum(j_out_twice, -j1_twice, -m2_twice)
    total = 0.0
    for k in range(max(0, -d, -e), min(a, b, c) + 1):
        sign = -1.0 if k % 2 else 1.0
        denom = (
            math.factorial(k)
            * math.factorial(a - k)
            * math.factorial(b - k)
            * math.factorial(c - k)
            * math.factorial(d + k)
            * math.factorial(e + k)
        )
        total += sign / denom
    return float(jnp.sqrt(triangle_prefactor * magnetic_prefactor) * total)


@cache
def vertex_intertwiner_tensor(block: VertexBlock) -> jnp.ndarray:
    """Return the canonical oriented singlet intertwiner tensor.

    Lattice links are oriented right/down. At a vertex, the left/up legs are
    incoming dual legs and the right/down legs are outgoing legs. Matter blocks
    carry one additional matter magnetic axis at the end.
    """
    if len(block.internal_irreps) == 2 and block.matter_irrep == 0:
        return _pure_vertex_intertwiner_tensor(block)
    if len(block.internal_irreps) == 3:
        return _matter_vertex_intertwiner_tensor(block)
    raise ValueError("Unexpected SU(2) vertex-block fusion path.")


def _pure_vertex_intertwiner_tensor(block: VertexBlock) -> jnp.ndarray:
    j_mid, j_pair = block.internal_irreps
    tensor = jnp.zeros(
        (
            block.j_l + 1,
            block.j_u + 1,
            block.j_r + 1,
            block.j_d + 1,
        ),
        dtype=jnp.float64,
    )
    for l_idx, m_l in enumerate(_magnetic_labels(block.j_l)):
        for u_idx, m_u in enumerate(_magnetic_labels(block.j_u)):
            for r_idx, m_r in enumerate(_magnetic_labels(block.j_r)):
                for d_idx, m_d in enumerate(_magnetic_labels(block.j_d)):
                    value = 0.0
                    for m_mid in _magnetic_labels(j_mid):
                        for m_pair in _magnetic_labels(j_pair):
                            value += (
                                clebsch_gordan(
                                    block.j_l, m_l, block.j_u, m_u, j_mid, m_mid
                                )
                                * clebsch_gordan(
                                    block.j_r, m_r, block.j_d, m_d, j_pair, m_pair
                                )
                                * clebsch_gordan(j_mid, m_mid, j_pair, m_pair, 0, 0)
                            )
                    tensor = tensor.at[l_idx, u_idx, r_idx, d_idx].set(value)
    return jnp.einsum(
        "la,ub,abrd->lurd",
        _dual_metric(block.j_l),
        _dual_metric(block.j_u),
        tensor,
        optimize=True,
    )


def _matter_vertex_intertwiner_tensor(block: VertexBlock) -> jnp.ndarray:
    j_mid, j_pair, j_gauge = block.internal_irreps
    tensor = jnp.zeros(
        (
            block.j_l + 1,
            block.j_u + 1,
            block.j_r + 1,
            block.j_d + 1,
            block.matter_irrep + 1,
        ),
        dtype=jnp.float64,
    )
    for l_idx, m_l in enumerate(_magnetic_labels(block.j_l)):
        for u_idx, m_u in enumerate(_magnetic_labels(block.j_u)):
            for r_idx, m_r in enumerate(_magnetic_labels(block.j_r)):
                for d_idx, m_d in enumerate(_magnetic_labels(block.j_d)):
                    for q_idx, m_q in enumerate(_magnetic_labels(block.matter_irrep)):
                        value = 0.0
                        for m_mid in _magnetic_labels(j_mid):
                            for m_pair in _magnetic_labels(j_pair):
                                for m_gauge in _magnetic_labels(j_gauge):
                                    value += (
                                        clebsch_gordan(
                                            block.j_l, m_l, block.j_u, m_u, j_mid, m_mid
                                        )
                                        * clebsch_gordan(
                                            block.j_r,
                                            m_r,
                                            block.j_d,
                                            m_d,
                                            j_pair,
                                            m_pair,
                                        )
                                        * clebsch_gordan(
                                            j_mid,
                                            m_mid,
                                            j_pair,
                                            m_pair,
                                            j_gauge,
                                            m_gauge,
                                        )
                                        * clebsch_gordan(
                                            j_gauge,
                                            m_gauge,
                                            block.matter_irrep,
                                            m_q,
                                            0,
                                            0,
                                        )
                                    )
                        tensor = tensor.at[l_idx, u_idx, r_idx, d_idx, q_idx].set(value)
    return jnp.einsum(
        "la,ub,abrdq->lurdq",
        _dual_metric(block.j_l),
        _dual_metric(block.j_u),
        tensor,
        optimize=True,
    )


def _magnetic_labels(j_twice: int) -> tuple[int, ...]:
    return tuple(range(-j_twice, j_twice + 1, 2))


@cache
def _dual_metric(j_twice: int) -> jax.Array:
    metric = jnp.zeros((j_twice + 1, j_twice + 1), dtype=jnp.float64)
    labels = _magnetic_labels(j_twice)
    prefactor = jnp.sqrt(j_twice + 1)
    for row, m_row in enumerate(labels):
        for col, m_col in enumerate(labels):
            metric = metric.at[row, col].set(
                prefactor * clebsch_gordan(j_twice, m_row, j_twice, m_col, 0, 0)
            )
    return metric


def _is_valid_magnetic_label(j_twice: int, m_twice: int) -> bool:
    return (
        isinstance(j_twice, int)
        and isinstance(m_twice, int)
        and j_twice >= 0
        and abs(m_twice) <= j_twice
        and (j_twice - m_twice) % 2 == 0
    )


def _half_sum(*twice_values: int) -> int:
    total = sum(twice_values)
    if total % 2 != 0:
        raise ValueError("Expected an integer half-sum.")
    return total // 2


@builders.build_plaquette_link_transitions.dispatch
def build_plaquette_link_transitions(group: SU2) -> PlaquetteLinkTransitions:
    """Build static plaquette link-output candidates from fundamental fusion."""
    n_irreps = len(group.irreps())
    outputs_by_input: dict[tuple[int, int, int, int], tuple[tuple[int, ...], ...]] = {}
    max_outputs = 0
    for j_top in group.irreps():
        for j_right in group.irreps():
            for j_bottom in group.irreps():
                for j_left in group.irreps():
                    outputs = tuple(
                        (out_top, out_right, out_bottom, out_left)
                        for out_top in group.fundamental_link_outputs(j_top)
                        for out_right in group.fundamental_link_outputs(j_right)
                        for out_bottom in group.fundamental_link_outputs(j_bottom)
                        for out_left in group.fundamental_link_outputs(j_left)
                    )
                    key = (j_top, j_right, j_bottom, j_left)
                    outputs_by_input[key] = outputs
                    max_outputs = max(max_outputs, len(outputs))
    output_links = jnp.full(
        (n_irreps, n_irreps, n_irreps, n_irreps, max_outputs, 4),
        -1,
        dtype=jnp.int32,
    )
    counts = jnp.zeros((n_irreps, n_irreps, n_irreps, n_irreps), dtype=jnp.int32)
    for key, outputs in outputs_by_input.items():
        counts = counts.at[key].set(len(outputs))
        for out_idx, output in enumerate(outputs):
            output_links = output_links.at[key + (out_idx,)].set(
                jnp.asarray(output, dtype=jnp.int32)
            )
    return PlaquetteLinkTransitions(
        output_links=output_links,
        counts=counts,
        max_outputs=max_outputs,
    )


@builders.build_plaquette_matrix_table.dispatch
def build_plaquette_matrix_table(
    group: SU2,
    tables: PureGaugeTables,
    *,
    row: int,
    col: int,
) -> PlaquetteMatrixTable:
    """Build static plaquette matrix elements for one plaquette."""
    n_rows, n_cols = tables.shape
    if not (0 <= row < n_rows - 1 and 0 <= col < n_cols - 1):
        raise IndexError(f"Plaquette {(row, col)} is outside shape {tables.shape}.")

    link_transitions = build_plaquette_link_transitions(group)
    site_coords = ((row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1))
    block_counts = tuple(tables.n_blocks(r, c) for r, c in site_coords)
    outcomes_by_input: dict[
        tuple[int, int, int, int],
        list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int], float]],
    ] = {}
    max_outputs = 0
    for tl_id in range(block_counts[0]):
        for tr_id in range(block_counts[1]):
            for bl_id in range(block_counts[2]):
                for br_id in range(block_counts[3]):
                    input_ids = (tl_id, tr_id, bl_id, br_id)
                    input_blocks = _plaquette_blocks(tables, site_coords, input_ids)
                    if not _plaquette_input_consistent(input_blocks):
                        outcomes_by_input[input_ids] = []
                        continue
                    outcomes = _plaquette_output_outcomes(
                        tables,
                        site_coords,
                        input_blocks,
                        link_transitions,
                    )
                    outcomes_by_input[input_ids] = outcomes
                    max_outputs = max(max_outputs, len(outcomes))

    starts = jnp.zeros(block_counts, dtype=jnp.int32)
    counts = jnp.zeros(block_counts, dtype=jnp.int32)
    total_outputs = sum(len(outcomes) for outcomes in outcomes_by_input.values())
    output_block_ids = jnp.full((total_outputs, 4), -1, dtype=jnp.int32)
    matrix_elements = jnp.zeros((total_outputs,), dtype=jnp.float64)
    cursor = 0
    for input_ids, outcomes in outcomes_by_input.items():
        starts = starts.at[input_ids].set(cursor)
        counts = counts.at[input_ids].set(len(outcomes))
        for out_idx, (links, iotas, block_ids, matrix_element) in enumerate(outcomes):
            del links, iotas
            output_block_ids = output_block_ids.at[cursor + out_idx].set(
                jnp.asarray(block_ids, dtype=jnp.int32)
            )
            matrix_elements = matrix_elements.at[cursor + out_idx].set(matrix_element)
        cursor += len(outcomes)
    proposal_weights = jnp.abs(matrix_elements) ** 2
    proposal_norms = jnp.zeros(block_counts, dtype=proposal_weights.dtype)
    for input_ids, outcomes in outcomes_by_input.items():
        start = int(starts[input_ids])
        proposal_norms = proposal_norms.at[input_ids].set(
            jnp.sum(proposal_weights[start : start + len(outcomes)])
        )
    return PlaquetteMatrixTable(
        starts=starts,
        counts=counts,
        max_count=max_outputs,
        output_block_ids=output_block_ids,
        matrix_elements=matrix_elements,
        proposal_weights=proposal_weights,
        proposal_norms=proposal_norms,
    )


@builders.build_plaquette_matrix_tables.dispatch
def build_plaquette_matrix_tables(
    group: SU2,
    tables: PureGaugeTables,
) -> tuple[tuple[PlaquetteMatrixTable, ...], ...]:
    """Build static plaquette matrix tables for all plaquettes."""
    n_rows, n_cols = tables.shape
    return tuple(
        tuple(
            build_plaquette_matrix_table(group, tables, row=row, col=col)
            for col in range(n_cols - 1)
        )
        for row in range(n_rows - 1)
    )


def _empty_hopping_matrix_table(block_counts: tuple[int, int]) -> HoppingMatrixTable:
    return HoppingMatrixTable(
        starts=jnp.zeros(block_counts, dtype=jnp.int32),
        counts=jnp.zeros(block_counts, dtype=jnp.int32),
        max_count=0,
        output_block_ids=jnp.full((0, 2), -1, dtype=jnp.int32),
        matrix_elements=jnp.zeros((0,), dtype=jnp.float64),
        proposal_weights=jnp.zeros((0,), dtype=jnp.float64),
        proposal_norms=jnp.zeros(block_counts, dtype=jnp.float64),
    )


@builders.build_horizontal_hopping_matrix_table.dispatch
def build_horizontal_hopping_matrix_table(
    group: SU2,
    tables: PureGaugeTables,
    *,
    row: int,
    col: int,
) -> HoppingMatrixTable:
    n_rows, n_cols = tables.shape
    if not (0 <= row < n_rows and 0 <= col < n_cols - 1):
        raise IndexError(f"Horizontal link {(row, col)} is outside shape {tables.shape}.")
    site_coords = ((row, col), (row, col + 1))
    return _build_hopping_matrix_table(
        group,
        tables,
        site_coords,
        orientation="h",
    )


@builders.build_horizontal_hopping_matrix_tables.dispatch
def build_horizontal_hopping_matrix_tables(
    group: SU2,
    tables: PureGaugeTables,
) -> tuple[tuple[HoppingMatrixTable, ...], ...]:
    n_rows, n_cols = tables.shape
    return tuple(
        tuple(
            build_horizontal_hopping_matrix_table(group, tables, row=row, col=col)
            for col in range(n_cols - 1)
        )
        for row in range(n_rows)
    )


@builders.build_vertical_hopping_matrix_table.dispatch
def build_vertical_hopping_matrix_table(
    group: SU2,
    tables: PureGaugeTables,
    *,
    row: int,
    col: int,
) -> HoppingMatrixTable:
    n_rows, n_cols = tables.shape
    if not (0 <= row < n_rows - 1 and 0 <= col < n_cols):
        raise IndexError(f"Vertical link {(row, col)} is outside shape {tables.shape}.")
    site_coords = ((row, col), (row + 1, col))
    return _build_hopping_matrix_table(
        group,
        tables,
        site_coords,
        orientation="v",
    )


@builders.build_vertical_hopping_matrix_tables.dispatch
def build_vertical_hopping_matrix_tables(
    group: SU2,
    tables: PureGaugeTables,
) -> tuple[tuple[HoppingMatrixTable, ...], ...]:
    n_rows, n_cols = tables.shape
    return tuple(
        tuple(
            build_vertical_hopping_matrix_table(group, tables, row=row, col=col)
            for col in range(n_cols)
        )
        for row in range(n_rows - 1)
    )


def _build_hopping_matrix_table(
    group: SU2,
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], tuple[int, int]],
    *,
    orientation: str,
) -> HoppingMatrixTable:
    if tables.phys_dim == 1:
        block_counts = tuple(tables.n_blocks(r, c) for r, c in site_coords)
        return _empty_hopping_matrix_table(block_counts)
    block_counts = tuple(tables.n_blocks(r, c) for r, c in site_coords)
    outcomes_by_input: dict[
        tuple[int, int],
        list[tuple[tuple[int, int], float]],
    ] = {}
    max_outputs = 0
    for first_id in range(block_counts[0]):
        for second_id in range(block_counts[1]):
            input_ids = (first_id, second_id)
            input_blocks = _link_blocks(tables, site_coords, input_ids)
            if not _hopping_input_consistent(input_blocks, orientation=orientation):
                outcomes_by_input[input_ids] = []
                continue
            outcomes = _hopping_output_outcomes(
                group,
                tables,
                site_coords,
                input_blocks,
                orientation=orientation,
            )
            outcomes_by_input[input_ids] = outcomes
            max_outputs = max(max_outputs, len(outcomes))

    starts = jnp.zeros(block_counts, dtype=jnp.int32)
    counts = jnp.zeros(block_counts, dtype=jnp.int32)
    total_outputs = sum(len(outcomes) for outcomes in outcomes_by_input.values())
    output_block_ids = jnp.full((total_outputs, 2), -1, dtype=jnp.int32)
    matrix_elements = jnp.zeros((total_outputs,), dtype=jnp.float64)
    cursor = 0
    for input_ids, outcomes in outcomes_by_input.items():
        starts = starts.at[input_ids].set(cursor)
        counts = counts.at[input_ids].set(len(outcomes))
        for out_idx, (block_ids, matrix_element) in enumerate(outcomes):
            output_block_ids = output_block_ids.at[cursor + out_idx].set(
                jnp.asarray(block_ids, dtype=jnp.int32)
            )
            matrix_elements = matrix_elements.at[cursor + out_idx].set(matrix_element)
        cursor += len(outcomes)
    proposal_weights = jnp.abs(matrix_elements) ** 2
    proposal_norms = jnp.zeros(block_counts, dtype=proposal_weights.dtype)
    for input_ids, outcomes in outcomes_by_input.items():
        start = int(starts[input_ids])
        proposal_norms = proposal_norms.at[input_ids].set(
            jnp.sum(proposal_weights[start : start + len(outcomes)])
        )
    return HoppingMatrixTable(
        starts=starts,
        counts=counts,
        max_count=max_outputs,
        output_block_ids=output_block_ids,
        matrix_elements=matrix_elements,
        proposal_weights=proposal_weights,
        proposal_norms=proposal_norms,
    )


def _link_blocks(
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], tuple[int, int]],
    block_ids: tuple[int, int],
) -> tuple[VertexBlock, VertexBlock]:
    return tuple(
        tables.blocks[r][c][block_id]
        for (r, c), block_id in zip(site_coords, block_ids, strict=True)
    )


def _hopping_input_consistent(
    blocks: tuple[VertexBlock, VertexBlock],
    *,
    orientation: str,
) -> bool:
    first, second = blocks
    if orientation == "h":
        return first.j_r == second.j_l
    return first.j_d == second.j_u


def _hopping_output_outcomes(
    group: SU2,
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], tuple[int, int]],
    input_blocks: tuple[VertexBlock, VertexBlock],
    *,
    orientation: str,
) -> list[tuple[tuple[int, int], float]]:
    first, second = input_blocks
    input_link = first.j_r if orientation == "h" else first.j_d
    outcomes = []
    for output_link in group.fundamental_link_outputs(input_link):
        for output_states in _number_conserving_matter_state_pairs(tables, input_blocks):
            output_block_ids = _hopping_output_block_ids(
                tables,
                site_coords,
                input_blocks,
                output_link,
                output_states,
                orientation=orientation,
            )
            for block_ids in _product_tuples(output_block_ids):
                output_blocks = _link_blocks(tables, site_coords, block_ids)
                matrix_element = _hopping_matrix_element(
                    input_blocks,
                    output_blocks,
                    orientation=orientation,
                )
                if matrix_element == 0.0:
                    continue
                outcomes.append((block_ids, matrix_element))
    return outcomes


def _number_conserving_matter_state_pairs(
    tables: PureGaugeTables,
    input_blocks: tuple[VertexBlock, VertexBlock],
) -> tuple[tuple[int, int], ...]:
    first, second = input_blocks
    input_number = first.matter_number + second.matter_number
    return tuple(
        (left_state, right_state)
        for left_state, left_number in enumerate(tables.matter_numbers)
        for right_state, right_number in enumerate(tables.matter_numbers)
        if left_number + right_number == input_number
        and (left_state, right_state)
        != (first.matter_state, second.matter_state)
    )


def _hopping_output_block_ids(
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], tuple[int, int]],
    input_blocks: tuple[VertexBlock, VertexBlock],
    output_link: int,
    output_states: tuple[int, int],
    *,
    orientation: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    first, second = input_blocks
    if orientation == "h":
        keys = (
            (output_states[0], first.j_l, first.j_u, output_link, first.j_d),
            (output_states[1], output_link, second.j_u, second.j_r, second.j_d),
        )
    else:
        keys = (
            (output_states[0], first.j_l, first.j_u, first.j_r, output_link),
            (output_states[1], second.j_l, output_link, second.j_r, second.j_d),
        )
    return tuple(
        _site_block_ids_for_links(tables, r, c, key)
        for (r, c), key in zip(site_coords, keys, strict=True)
    )


@cache
def _hopping_matrix_element(
    input_blocks: tuple[VertexBlock, VertexBlock],
    output_blocks: tuple[VertexBlock, VertexBlock],
    *,
    orientation: str,
) -> float:
    forward = _oriented_hopping_matrix_element(
        input_blocks,
        output_blocks,
        orientation=orientation,
    )
    backward = _oriented_hopping_matrix_element(
        output_blocks,
        input_blocks,
        orientation=orientation,
    )
    return float(forward + backward)


@cache
def _oriented_hopping_matrix_element(
    input_blocks: tuple[VertexBlock, VertexBlock],
    output_blocks: tuple[VertexBlock, VertexBlock],
    *,
    orientation: str,
) -> float:
    first_in, second_in = input_blocks
    first_out, second_out = output_blocks
    if orientation == "h":
        first_overlap = _vertex_overlap(first_in, first_out, affected_axes=(2, 4))
        second_overlap = _vertex_overlap(second_in, second_out, affected_axes=(0, 4))
        link = _link_fundamental_tensor(first_in.j_r, first_out.j_r)
    else:
        first_overlap = _vertex_overlap(first_in, first_out, affected_axes=(3, 4))
        second_overlap = _vertex_overlap(second_in, second_out, affected_axes=(1, 4))
        link = _link_fundamental_tensor(first_in.j_d, first_out.j_d)
    second_entries = _nonzero_entries(second_overlap)
    total = 0.0
    for (p_out, m_first_out, p_in, _m_first_in), first_value in _nonzero_entries(
        first_overlap
    ):
        for (
            p_link_out,
            q_link_out,
            p_link_in,
            q_link_in,
            a_src,
            a_tgt,
        ), link_value in _nonzero_entries(link):
            if p_link_out != p_out or p_link_in != p_in or a_src != m_first_out:
                continue
            prefix = first_value * link_value
            for (
                q_out,
                _m_second_out,
                q_in,
                m_second_in,
            ), second_value in second_entries:
                if (
                    q_link_out == q_out
                    and q_link_in == q_in
                    and a_tgt == m_second_in
                ):
                    total += prefix * second_value
    return total


def _plaquette_blocks(
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], ...],
    block_ids: tuple[int, int, int, int],
) -> tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock]:
    return tuple(
        tables.blocks[r][c][block_id]
        for (r, c), block_id in zip(site_coords, block_ids, strict=True)
    )


def _plaquette_input_consistent(
    blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
) -> bool:
    tl, tr, bl, br = blocks
    return (
        tl.j_r == tr.j_l
        and bl.j_r == br.j_l
        and tl.j_d == bl.j_u
        and tr.j_d == br.j_u
    )


def _plaquette_output_outcomes(
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], ...],
    input_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
    link_transitions: PlaquetteLinkTransitions,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int], float]]:
    tl, tr, bl, br = input_blocks
    input_links = (tl.j_r, tr.j_d, bl.j_r, tl.j_d)
    outcomes = []
    for output_links in link_transitions.outputs(*input_links):
        output_block_ids = _plaquette_output_block_ids(
            tables,
            site_coords,
            input_blocks,
            output_links,
        )
        for block_ids in _product_tuples(output_block_ids):
            output_blocks = _plaquette_blocks(tables, site_coords, block_ids)
            matrix_element = _plaquette_matrix_element(input_blocks, output_blocks)
            if matrix_element == 0.0:
                continue
            outcomes.append(
                (
                    output_links,
                    tuple(block.iota for block in output_blocks),
                    block_ids,
                    matrix_element,
                )
            )
    return outcomes


def _plaquette_output_block_ids(
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], ...],
    input_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
    output_links: tuple[int, int, int, int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    tl, tr, bl, br = input_blocks
    top, right, bottom, left = output_links
    keys = (
        (tl.matter_state, tl.j_l, tl.j_u, top, left),
        (tr.matter_state, top, tr.j_u, tr.j_r, right),
        (bl.matter_state, bl.j_l, left, bottom, bl.j_d),
        (br.matter_state, bottom, right, br.j_r, br.j_d),
    )
    return tuple(
        _site_block_ids_for_links(tables, r, c, key)
        for (r, c), key in zip(site_coords, keys, strict=True)
    )


def _site_block_ids_for_links(
    tables: PureGaugeTables,
    row: int,
    col: int,
    links: tuple[int, int, int, int, int],
) -> tuple[int, ...]:
    matter_state, j_l, j_u, j_r, j_d = links
    return tuple(
        block_id
        for block_id, block in enumerate(tables.blocks[row][col])
        if (
            block.matter_state,
            block.j_l,
            block.j_u,
            block.j_r,
            block.j_d,
        )
        == (matter_state, j_l, j_u, j_r, j_d)
    )


def _product_tuples(
    values: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if any(len(part) == 0 for part in values):
        return ()
    out = [()]
    for part in values:
        out = [prefix + (value,) for prefix in out for value in part]
    return tuple(out)


@cache
def _plaquette_matrix_element(
    input_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
    output_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
) -> float:
    forward = _oriented_plaquette_matrix_element(input_blocks, output_blocks)
    backward = _oriented_plaquette_matrix_element(output_blocks, input_blocks)
    return float(forward + backward)


@cache
def _oriented_plaquette_matrix_element(
    input_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
    output_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
) -> float:
    tl_in, tr_in, bl_in, br_in = input_blocks
    tl_out, tr_out, bl_out, br_out = output_blocks
    tl_overlap = _vertex_overlap(tl_in, tl_out, affected_axes=(2, 3))
    tr_overlap = _vertex_overlap(tr_in, tr_out, affected_axes=(0, 3))
    bl_overlap = _vertex_overlap(bl_in, bl_out, affected_axes=(1, 2))
    br_overlap = _vertex_overlap(br_in, br_out, affected_axes=(0, 1))
    top_link = _link_fundamental_tensor(tl_in.j_r, tl_out.j_r)
    right_link = _link_fundamental_tensor(tr_in.j_d, tr_out.j_d)
    bottom_link = _link_fundamental_tensor(bl_in.j_r, bl_out.j_r)
    left_link = _link_fundamental_tensor(tl_in.j_d, tl_out.j_d)
    dual = _dual_metric(1)
    top_entries = _nonzero_entries(top_link)
    right_entries = _nonzero_entries(right_link)
    bottom_entries = _nonzero_entries(bottom_link)
    left_entries = _nonzero_entries(left_link)
    tr_entries = _nonzero_entries(tr_overlap)
    br_entries = _nonzero_entries(br_overlap)
    bl_entries = _nonzero_entries(bl_overlap)

    total = 0.0
    for (p, left_out, p_in, left_in), tl_value in _nonzero_entries(tl_overlap):
        for (p_link, q, p_link_in, q_in, a, b), top_value in top_entries:
            if p_link != p or p_link_in != p_in:
                continue
            for (q_vertex, right_out, q_vertex_in, right_in), tr_value in tr_entries:
                if q_vertex != q or q_vertex_in != q_in:
                    continue
                for (right_src, v, right_src_in, v_in, c, d), right_value in right_entries:
                    if right_src != right_out or right_src_in != right_in or c != b:
                        continue
                    prefix = tl_value * top_value * tr_value * right_value
                    for (u, v_vertex, u_in, v_vertex_in), br_value in br_entries:
                        if v_vertex != v or v_vertex_in != v_in:
                            continue
                        for (bottom_src, u_link, bottom_src_in, u_link_in, e, f), bottom_value in bottom_entries:
                            if u_link != u or u_link_in != u_in:
                                continue
                            br_connector = float(dual[d, f])
                            if br_connector == 0.0:
                                continue
                            prefix_bottom = prefix * br_value * bottom_value * br_connector
                            for (left_tgt, bottom_tgt, left_tgt_in, bottom_tgt_in), bl_value in bl_entries:
                                if (
                                    bottom_tgt != bottom_src
                                    or bottom_tgt_in != bottom_src_in
                                ):
                                    continue
                                for (left_src, left_tgt_link, left_src_in, left_tgt_link_in, h, i), left_value in left_entries:
                                    if (
                                        left_src != left_out
                                        or left_src_in != left_in
                                        or left_tgt_link != left_tgt
                                        or left_tgt_link_in != left_tgt_in
                                        or i != a
                                    ):
                                        continue
                                    bl_connector = float(dual[h, e])
                                    if bl_connector == 0.0:
                                        continue
                                    total += (
                                        prefix_bottom
                                        * bl_value
                                        * left_value
                                        * bl_connector
                                    )
    return total


@cache
def _vertex_overlap(
    input_block: VertexBlock,
    output_block: VertexBlock,
    *,
    affected_axes: tuple[int, int],
) -> jax.Array:
    input_tensor = vertex_intertwiner_tensor(input_block)
    output_tensor = vertex_intertwiner_tensor(output_block)
    if 4 in affected_axes and input_tensor.ndim == 4:
        input_tensor = input_tensor[..., None]
    if 4 in affected_axes and output_tensor.ndim == 4:
        output_tensor = output_tensor[..., None]
    external_axes = tuple(
        axis for axis in range(output_tensor.ndim) if axis not in affected_axes
    )
    out_dims = tuple(output_tensor.shape[axis] for axis in affected_axes)
    in_dims = tuple(input_tensor.shape[axis] for axis in affected_axes)
    ext_dim = math.prod(output_tensor.shape[axis] for axis in external_axes)
    output_flat = jnp.transpose(output_tensor, affected_axes + external_axes).reshape(
        math.prod(out_dims),
        ext_dim,
    )
    input_flat = jnp.transpose(input_tensor, affected_axes + external_axes).reshape(
        math.prod(in_dims),
        ext_dim,
    )
    return (output_flat @ input_flat.T).reshape(*out_dims, *in_dims)


@cache
def _link_fundamental_tensor(j_input: int, j_output: int) -> jax.Array:
    tensor = jnp.zeros(
        (j_output + 1, j_output + 1, j_input + 1, j_input + 1, 2, 2),
        dtype=jnp.float64,
    )
    prefactor = jnp.sqrt((j_input + 1) / (j_output + 1))
    for out_src_idx, m_out_src in enumerate(_magnetic_labels(j_output)):
        for out_tgt_idx, m_out_tgt in enumerate(_magnetic_labels(j_output)):
            for in_src_idx, m_in_src in enumerate(_magnetic_labels(j_input)):
                for in_tgt_idx, m_in_tgt in enumerate(_magnetic_labels(j_input)):
                    for a_src_idx, a_src in enumerate(_magnetic_labels(1)):
                        for a_tgt_idx, a_tgt in enumerate(_magnetic_labels(1)):
                            value = prefactor * clebsch_gordan(
                                1,
                                a_src,
                                j_input,
                                m_in_src,
                                j_output,
                                m_out_src,
                            ) * clebsch_gordan(
                                1,
                                a_tgt,
                                j_input,
                                m_in_tgt,
                                j_output,
                                m_out_tgt,
                            )
                            tensor = tensor.at[
                                out_src_idx,
                                out_tgt_idx,
                                in_src_idx,
                                in_tgt_idx,
                                a_src_idx,
                                a_tgt_idx,
                            ].set(value)
    return tensor


def _nonzero_entries(array: jax.Array) -> tuple[tuple[tuple[int, ...], float], ...]:
    entries = []
    for index in _array_indices(array.shape):
        value = float(array[index])
        if value != 0.0:
            entries.append((index, value))
    return tuple(entries)


def _array_indices(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    indices = [()]
    for size in shape:
        indices = [prefix + (idx,) for prefix in indices for idx in range(size)]
    return tuple(indices)


def _pure_gauge_intertwiner_paths(
    group: SU2,
    *,
    j_l: int,
    j_u: int,
    j_r: int,
    j_d: int,
    target_twice: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (j_m, j_n)
        for j_m in group.tensor_product(j_l, j_u)
        for j_n in group.tensor_product(j_r, j_d)
        if target_twice in _fuse_untruncated(j_m, j_n)
    )


def _matter_intertwiner_paths(
    group: SU2,
    *,
    j_l: int,
    j_u: int,
    j_r: int,
    j_d: int,
    matter_irrep: int,
    target_twice: int,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (j_m, j_n, j_gauge)
        for j_m in group.tensor_product(j_l, j_u)
        for j_n in group.tensor_product(j_r, j_d)
        for j_gauge in _fuse_untruncated(j_m, j_n)
        if target_twice in _fuse_untruncated(j_gauge, matter_irrep)
    )


def build_pure_gauge_vertex_blocks(
    group: SU2,
    *,
    active_legs: tuple[bool, bool, bool, bool],
    target_twice: int = 0,
    matter_irreps: tuple[int, ...] = (0,),
    matter_numbers: tuple[int, ...] = (0,),
) -> tuple[VertexBlock, ...]:
    """Enumerate valid vertex blocks for one site.

    ``active_legs`` is ordered as ``(left, up, right, down)``. Inactive
    boundary legs are fixed to the singlet irrep ``j_twice = 0``.
    """
    if len(active_legs) != 4:
        raise ValueError("active_legs must be ordered as (left, up, right, down).")
    if target_twice < 0:
        raise ValueError("target_twice must be non-negative.")
    if len(matter_irreps) != len(matter_numbers):
        raise ValueError("matter_irreps and matter_numbers must have equal length.")

    link_irreps = group.irreps()
    choices = tuple(link_irreps if active else (0,) for active in active_legs)
    blocks: list[VertexBlock] = []
    for matter_state, (matter_irrep, matter_number) in enumerate(
        zip(matter_irreps, matter_numbers, strict=True)
    ):
        if matter_irrep < 0:
            raise ValueError("matter_irreps must be non-negative SU(2) labels.")
        for j_l in choices[0]:
            for j_u in choices[1]:
                for j_r in choices[2]:
                    for j_d in choices[3]:
                        if matter_irrep == 0:
                            paths = _pure_gauge_intertwiner_paths(
                                group,
                                j_l=j_l,
                                j_u=j_u,
                                j_r=j_r,
                                j_d=j_d,
                                target_twice=target_twice,
                            )
                        else:
                            paths = _matter_intertwiner_paths(
                                group,
                                j_l=j_l,
                                j_u=j_u,
                                j_r=j_r,
                                j_d=j_d,
                                matter_irrep=matter_irrep,
                                target_twice=target_twice,
                            )
                        for iota, internal_irreps in enumerate(paths):
                            blocks.append(
                                VertexBlock(
                                    j_l=j_l,
                                    j_u=j_u,
                                    j_r=j_r,
                                    j_d=j_d,
                                    iota=iota,
                                    internal_irreps=internal_irreps,
                                    matter_state=matter_state,
                                    matter_irrep=matter_irrep,
                                    matter_number=matter_number,
                                )
                            )
    return tuple(sorted(blocks))


@builders.build_pure_gauge_tables.dispatch
def build_pure_gauge_tables(
    group: SU2,
    *,
    shape: tuple[int, int],
    target_charge: int = 0,
    matter_irreps: tuple[int, ...] = (0,),
    matter_numbers: tuple[int, ...] = (0,),
) -> PureGaugeTables:
    """Build boundary-aware vertex block tables."""
    target_twice = int(target_charge)
    n_rows, n_cols = shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("shape must have positive dimensions.")
    if len(matter_irreps) != len(matter_numbers):
        raise ValueError("matter_irreps and matter_numbers must have equal length.")

    rows: list[tuple[tuple[VertexBlock, ...], ...]] = []
    lookup_rows: list[tuple[dict[tuple[int, int, int, int, int, int], int], ...]] = []
    for r in range(n_rows):
        row: list[tuple[VertexBlock, ...]] = []
        lookup_row: list[dict[tuple[int, int, int, int, int], int]] = []
        for c in range(n_cols):
            active_legs = (c > 0, r > 0, c < n_cols - 1, r < n_rows - 1)
            blocks = build_pure_gauge_vertex_blocks(
                group,
                active_legs=active_legs,
                target_twice=target_twice,
                matter_irreps=matter_irreps,
                matter_numbers=matter_numbers,
            )
            row.append(blocks)
            lookup_row.append(
                {
                    (
                        block.matter_state,
                        block.j_l,
                        block.j_u,
                        block.j_r,
                        block.j_d,
                        block.iota,
                    ): block_id
                    for block_id, block in enumerate(blocks)
                }
            )
        rows.append(tuple(row))
        lookup_rows.append(tuple(lookup_row))
    blocks_by_site = tuple(rows)
    max_iotas = max(
        block.iota + 1
        for row in blocks_by_site
        for blocks in row
        for block in blocks
    )
    n_irreps = len(group.irreps())
    phys_dim = len(matter_irreps)
    max_blocks = max(len(blocks) for row in blocks_by_site for blocks in row)
    block_id_lookup = jnp.full(
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
        dtype=jnp.int32,
    )
    matter_state_by_block = jnp.full((n_rows, n_cols, max_blocks), -1, dtype=jnp.int32)
    j_l_by_block = jnp.full((n_rows, n_cols, max_blocks), -1, dtype=jnp.int32)
    j_u_by_block = jnp.full((n_rows, n_cols, max_blocks), -1, dtype=jnp.int32)
    j_r_by_block = jnp.full((n_rows, n_cols, max_blocks), -1, dtype=jnp.int32)
    j_d_by_block = jnp.full((n_rows, n_cols, max_blocks), -1, dtype=jnp.int32)
    iota_by_block = jnp.full((n_rows, n_cols, max_blocks), -1, dtype=jnp.int32)
    for r, row in enumerate(blocks_by_site):
        for c, blocks in enumerate(row):
            for block_id, block in enumerate(blocks):
                block_id_lookup = block_id_lookup.at[
                    r,
                    c,
                    block.matter_state,
                    block.j_l,
                    block.j_u,
                    block.j_r,
                    block.j_d,
                    block.iota,
                ].set(block_id)
                matter_state_by_block = matter_state_by_block.at[r, c, block_id].set(
                    block.matter_state
                )
                j_l_by_block = j_l_by_block.at[r, c, block_id].set(block.j_l)
                j_u_by_block = j_u_by_block.at[r, c, block_id].set(block.j_u)
                j_r_by_block = j_r_by_block.at[r, c, block_id].set(block.j_r)
                j_d_by_block = j_d_by_block.at[r, c, block_id].set(block.j_d)
                iota_by_block = iota_by_block.at[r, c, block_id].set(block.iota)
    return PureGaugeTables(
        group=group,
        shape=shape,
        phys_dim=phys_dim,
        matter_irreps=tuple(int(irrep) for irrep in matter_irreps),
        matter_numbers=tuple(int(number) for number in matter_numbers),
        blocks=blocks_by_site,
        _block_ids=tuple(lookup_rows),
        max_iotas=max_iotas,
        block_id_lookup=block_id_lookup,
        matter_state_by_block=matter_state_by_block,
        j_l_by_block=j_l_by_block,
        j_u_by_block=j_u_by_block,
        j_r_by_block=j_r_by_block,
        j_d_by_block=j_d_by_block,
        iota_by_block=iota_by_block,
    )

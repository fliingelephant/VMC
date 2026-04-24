"""Fundamental-truncated SU(3) gauge-group backend for non-Abelian GI-PEPS."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import product
import math

import jax
import jax.numpy as jnp

import vmc.peps.non_abelian_gi.builders as builders
from vmc.peps.non_abelian_gi.tables import (
    PlaquetteLinkTransitions,
    PlaquetteMatrixTable,
    PureGaugeTables,
)


@dataclass(frozen=True)
class SU3:
    """SU(3) irreps ``(p, q)`` truncated to ``p + q <= max_weight_sum``.

    The current backend implements exact magnetic tensors for the fundamental
    truncation ``max_weight_sum=1``: singlet, fundamental, and antifundamental.
    """

    max_weight_sum: int

    def __post_init__(self) -> None:
        if not isinstance(self.max_weight_sum, int):
            raise ValueError("max_weight_sum must be an integer.")
        if self.max_weight_sum != 1:
            raise NotImplementedError(
                "SU(3) backend currently supports max_weight_sum=1."
            )

    @property
    def fundamental(self) -> int:
        return self.label((1, 0))

    @property
    def antifundamental(self) -> int:
        return self.label((0, 1))

    @property
    def random_init_sweeps(self) -> int:
        return self.max_weight_sum

    def irreps(self) -> tuple[int, ...]:
        return tuple(range(len(_IRREP_WEIGHTS)))

    def highest_weight(self, irrep: int) -> tuple[int, int]:
        self._validate_irrep(irrep)
        return _IRREP_WEIGHTS[irrep]

    def label(self, highest_weight: tuple[int, int]) -> int:
        if highest_weight not in _IRREP_LABELS:
            raise ValueError(f"SU(3) irrep {highest_weight} is outside truncation.")
        return _IRREP_LABELS[highest_weight]

    def dual(self, irrep: int) -> int:
        p, q = self.highest_weight(irrep)
        return self.label((q, p))

    def dim(self, irrep: int) -> int:
        p, q = self.highest_weight(irrep)
        return (p + 1) * (q + 1) * (p + q + 2) // 2

    def casimir(self, irrep: int) -> float:
        p, q = self.highest_weight(irrep)
        return (p * p + q * q + p * q + 3 * p + 3 * q) / 3.0

    def fuse(self, irrep: int, operator_irrep: int) -> tuple[int, ...]:
        self._validate_irrep(irrep)
        self._validate_irrep(operator_irrep)
        return _fuse_labels(irrep, operator_irrep)

    def _validate_irrep(self, irrep: int) -> None:
        if not isinstance(irrep, int):
            raise ValueError("SU(3) irrep labels must be integers.")
        if irrep not in self.irreps():
            raise ValueError("SU(3) irrep label is outside truncation.")


def _fuse_labels(irrep: int, operator_irrep: int) -> tuple[int, ...]:
    if irrep == 0:
        return (operator_irrep,)
    if operator_irrep == 0:
        return (irrep,)
    weight = _IRREP_WEIGHTS[irrep]
    if operator_irrep == _IRREP_LABELS[(1, 0)]:
        return _filter_truncated(_tensor_with_fundamental(weight))
    if operator_irrep == _IRREP_LABELS[(0, 1)]:
        return _filter_truncated(_tensor_with_antifundamental(weight))
    raise NotImplementedError(
        "Only fundamental and antifundamental SU(3) fusions are implemented."
    )


@dataclass(frozen=True, order=True)
class VertexBlock:
    """One pure-gauge SU(3) vertex block in the sampled spin-network basis."""

    j_l: int
    j_u: int
    j_r: int
    j_d: int
    iota: int
    internal_irreps: tuple[int, ...]


_IRREP_WEIGHTS = ((0, 0), (1, 0), (0, 1))
_IRREP_LABELS = {weight: label for label, weight in enumerate(_IRREP_WEIGHTS)}


def _tensor_with_fundamental(weight: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    p, q = weight
    outputs = [(p + 1, q)]
    if p > 0:
        outputs.append((p - 1, q + 1))
    if q > 0:
        outputs.append((p, q - 1))
    return tuple(outputs)


def _tensor_with_antifundamental(weight: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    p, q = weight
    outputs = [(p, q + 1)]
    if q > 0:
        outputs.append((p + 1, q - 1))
    if p > 0:
        outputs.append((p - 1, q))
    return tuple(outputs)


def _filter_truncated(weights: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    labels = []
    for weight in weights:
        if weight in _IRREP_LABELS:
            labels.append(_IRREP_LABELS[weight])
    return tuple(labels)


@cache
def _fundamental_generators() -> tuple[jax.Array, ...]:
    z = 0.0 + 0.0j
    one = 1.0 + 0.0j
    imag = 0.0 + 1.0j
    matrices = (
        ((z, one, z), (one, z, z), (z, z, z)),
        ((z, -imag, z), (imag, z, z), (z, z, z)),
        ((one, z, z), (z, -one, z), (z, z, z)),
        ((z, z, one), (z, z, z), (one, z, z)),
        ((z, z, -imag), (z, z, z), (imag, z, z)),
        ((z, z, z), (z, z, one), (z, one, z)),
        ((z, z, z), (z, z, -imag), (z, imag, z)),
        (
            (one / math.sqrt(3), z, z),
            (z, one / math.sqrt(3), z),
            (z, z, -2 / math.sqrt(3)),
        ),
    )
    return tuple(0.5 * jnp.asarray(matrix, dtype=jnp.complex128) for matrix in matrices)


@cache
def _generators(irrep: int) -> tuple[jax.Array, ...]:
    if irrep == 0:
        return tuple(jnp.zeros((1, 1), dtype=jnp.complex128) for _ in range(8))
    fundamental = _fundamental_generators()
    if irrep == 1:
        return fundamental
    if irrep == 2:
        return tuple(-generator.conj() for generator in fundamental)
    raise ValueError("SU(3) generator requested outside fundamental truncation.")


def _canonical_tensor(tensor: jax.Array) -> jax.Array:
    flat = tensor.reshape(-1)
    for value in flat:
        magnitude = float(jnp.abs(value))
        if magnitude > 1e-10:
            return tensor / (value / magnitude)
    return tensor


def _kron_all(matrices: tuple[jax.Array, ...]) -> jax.Array:
    out = jnp.ones((1, 1), dtype=jnp.complex128)
    for matrix in matrices:
        out = jnp.kron(out, matrix)
    return out


@cache
def _invariant_basis(labels: tuple[int, ...]) -> tuple[jax.Array, ...]:
    dims = tuple(_IRREP_DIMS[label] for label in labels)
    total_dim = math.prod(dims)
    generators_by_leg = tuple(_generators(label) for label in labels)
    constraints = []
    for generator_idx in range(8):
        total = jnp.zeros((total_dim, total_dim), dtype=jnp.complex128)
        for leg_idx, leg_generators in enumerate(generators_by_leg):
            factors = tuple(
                leg_generators[generator_idx]
                if idx == leg_idx
                else jnp.eye(dims[idx], dtype=jnp.complex128)
                for idx in range(len(labels))
            )
            total = total + _kron_all(factors)
        constraints.append(total)
    constraint = jnp.concatenate(constraints, axis=0)
    _u, singular_values, vh = jnp.linalg.svd(constraint, full_matrices=True)
    rank = int(jnp.sum(singular_values > 1e-10))
    return tuple(_canonical_tensor(vector.reshape(dims)) for vector in vh[rank:])


@cache
def vertex_intertwiner_tensor(block: VertexBlock) -> jax.Array:
    labels = (_dual_label(block.j_l), _dual_label(block.j_u), block.j_r, block.j_d)
    return _invariant_basis(labels)[block.iota]


@cache
def _coupling_tensor(
    output_irrep: int,
    operator_irrep: int,
    input_irrep: int,
) -> jax.Array:
    labels = (_dual_label(output_irrep), operator_irrep, input_irrep)
    basis = _invariant_basis(labels)
    if len(basis) != 1:
        raise ValueError(
            f"Expected a unique SU(3) coupling for {operator_irrep} x {input_irrep} -> {output_irrep}."
        )
    return basis[0]


@cache
def _link_operator_tensor(
    input_irrep: int,
    output_irrep: int,
    operator_irrep: int,
) -> jax.Array:
    in_dim = _IRREP_DIMS[input_irrep]
    out_dim = _IRREP_DIMS[output_irrep]
    op_dim = _IRREP_DIMS[operator_irrep]
    if output_irrep not in _fuse_labels(input_irrep, operator_irrep):
        return jnp.zeros(
            (out_dim, out_dim, in_dim, in_dim, op_dim, op_dim),
            dtype=jnp.complex128,
        )
    coupling = _coupling_tensor(output_irrep, operator_irrep, input_irrep)
    prefactor = jnp.sqrt(in_dim / out_dim)
    return prefactor * jnp.einsum(
        "oai,pbj->opijab",
        coupling,
        coupling.conj(),
        optimize=True,
    )


def _dual_label(irrep: int) -> int:
    if irrep == 1:
        return 2
    if irrep == 2:
        return 1
    return 0


_IRREP_DIMS = tuple((p + 1) * (q + 1) * (p + q + 2) // 2 for p, q in _IRREP_WEIGHTS)


@builders.build_plaquette_link_transitions.dispatch
def build_plaquette_link_transitions(group: SU3) -> PlaquetteLinkTransitions:
    """Build static plaquette link-output candidates for ``U_square + h.c.``."""
    n_irreps = len(group.irreps())
    outputs_by_input: dict[tuple[int, int, int, int], tuple[tuple[int, ...], ...]] = {}
    max_outputs = 0
    for top in group.irreps():
        for right in group.irreps():
            for bottom in group.irreps():
                for left in group.irreps():
                    forward = tuple(
                        (out_top, out_right, out_bottom, out_left)
                        for out_top in group.fuse(top, group.fundamental)
                        for out_right in group.fuse(right, group.fundamental)
                        for out_bottom in group.fuse(bottom, group.antifundamental)
                        for out_left in group.fuse(left, group.antifundamental)
                    )
                    backward = tuple(
                        (out_top, out_right, out_bottom, out_left)
                        for out_top in group.fuse(top, group.antifundamental)
                        for out_right in group.fuse(right, group.antifundamental)
                        for out_bottom in group.fuse(bottom, group.fundamental)
                        for out_left in group.fuse(left, group.fundamental)
                    )
                    outputs = tuple(sorted(set(forward + backward)))
                    key = (top, right, bottom, left)
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
    return PlaquetteLinkTransitions(output_links, counts, max_outputs)


@builders.build_plaquette_matrix_table.dispatch
def build_plaquette_matrix_table(
    group: SU3,
    tables: PureGaugeTables,
    *,
    row: int,
    col: int,
) -> PlaquetteMatrixTable:
    n_rows, n_cols = tables.shape
    if not (0 <= row < n_rows - 1 and 0 <= col < n_cols - 1):
        raise IndexError(f"Plaquette {(row, col)} is outside shape {tables.shape}.")
    link_transitions = build_plaquette_link_transitions(group)
    site_coords = ((row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1))
    block_counts = tuple(tables.n_blocks(r, c) for r, c in site_coords)
    outcomes_by_input = {}
    max_outputs = 0
    for input_ids in product(*(range(count) for count in block_counts)):
        input_blocks = _plaquette_blocks(tables, site_coords, input_ids)
        if not _plaquette_input_consistent(input_blocks):
            outcomes_by_input[input_ids] = []
            continue
        outcomes = _plaquette_output_outcomes(group, tables, site_coords, input_blocks, link_transitions)
        outcomes_by_input[input_ids] = outcomes
        max_outputs = max(max_outputs, len(outcomes))
    output_shape = (*block_counts, max_outputs)
    output_links = jnp.full((*output_shape, 4), -1, dtype=jnp.int32)
    output_iotas = jnp.full((*output_shape, 4), -1, dtype=jnp.int32)
    output_block_ids = jnp.full((*output_shape, 4), -1, dtype=jnp.int32)
    matrix_elements = jnp.zeros(output_shape, dtype=jnp.complex128)
    counts = jnp.zeros(block_counts, dtype=jnp.int32)
    for input_ids, outcomes in outcomes_by_input.items():
        counts = counts.at[input_ids].set(len(outcomes))
        for out_idx, (links, iotas, block_ids, matrix_element) in enumerate(outcomes):
            slot = input_ids + (out_idx,)
            output_links = output_links.at[slot].set(jnp.asarray(links, dtype=jnp.int32))
            output_iotas = output_iotas.at[slot].set(jnp.asarray(iotas, dtype=jnp.int32))
            output_block_ids = output_block_ids.at[slot].set(jnp.asarray(block_ids, dtype=jnp.int32))
            matrix_elements = matrix_elements.at[slot].set(matrix_element)
    proposal_weights = jnp.abs(matrix_elements) ** 2
    proposal_norms = jnp.sum(proposal_weights, axis=-1)
    return PlaquetteMatrixTable(
        output_links,
        output_iotas,
        output_block_ids,
        matrix_elements,
        proposal_weights,
        proposal_norms,
        counts,
        max_outputs,
    )


@builders.build_plaquette_matrix_tables.dispatch
def build_plaquette_matrix_tables(
    group: SU3,
    tables: PureGaugeTables,
) -> tuple[tuple[PlaquetteMatrixTable, ...], ...]:
    n_rows, n_cols = tables.shape
    return tuple(
        tuple(
            build_plaquette_matrix_table(group, tables, row=row, col=col)
            for col in range(n_cols - 1)
        )
        for row in range(n_rows - 1)
    )


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
    group: SU3,
    tables: PureGaugeTables,
    site_coords: tuple[tuple[int, int], ...],
    input_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
    link_transitions: PlaquetteLinkTransitions,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int], complex]]:
    tl, tr, bl, _br = input_blocks
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
            matrix_element = _plaquette_matrix_element(group, input_blocks, output_blocks)
            if abs(matrix_element) <= 1e-12:
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
        (tl.j_l, tl.j_u, top, left),
        (top, tr.j_u, tr.j_r, right),
        (bl.j_l, left, bottom, bl.j_d),
        (bottom, right, br.j_r, br.j_d),
    )
    return tuple(
        _site_block_ids_for_links(tables, r, c, key)
        for (r, c), key in zip(site_coords, keys, strict=True)
    )


def _site_block_ids_for_links(
    tables: PureGaugeTables,
    row: int,
    col: int,
    links: tuple[int, int, int, int],
) -> tuple[int, ...]:
    j_l, j_u, j_r, j_d = links
    return tuple(
        block_id
        for block_id, block in enumerate(tables.blocks[row][col])
        if (block.j_l, block.j_u, block.j_r, block.j_d) == (j_l, j_u, j_r, j_d)
    )


def _product_tuples(values: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    if any(len(part) == 0 for part in values):
        return ()
    out = [()]
    for part in values:
        out = [prefix + (value,) for prefix in out for value in part]
    return tuple(out)


@cache
def _plaquette_matrix_element(
    group: SU3,
    input_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
    output_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
) -> complex:
    forward = _oriented_plaquette_matrix_element(group, input_blocks, output_blocks)
    backward = _oriented_plaquette_matrix_element(group, output_blocks, input_blocks)
    return complex(forward + backward)


@cache
def _oriented_plaquette_matrix_element(
    group: SU3,
    input_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
    output_blocks: tuple[VertexBlock, VertexBlock, VertexBlock, VertexBlock],
) -> complex:
    tl_in, tr_in, bl_in, br_in = input_blocks
    tl_out, tr_out, bl_out, br_out = output_blocks
    tl_overlap = _vertex_overlap(tl_in, tl_out, affected_axes=(2, 3))
    tr_overlap = _vertex_overlap(tr_in, tr_out, affected_axes=(0, 3))
    bl_overlap = _vertex_overlap(bl_in, bl_out, affected_axes=(1, 2))
    br_overlap = _vertex_overlap(br_in, br_out, affected_axes=(0, 1))
    top_link = _link_operator_tensor(tl_in.j_r, tl_out.j_r, group.fundamental)
    right_link = _link_operator_tensor(tr_in.j_d, tr_out.j_d, group.fundamental)
    bottom_link = _link_operator_tensor(bl_in.j_r, bl_out.j_r, group.antifundamental)
    left_link = _link_operator_tensor(tl_in.j_d, tl_out.j_d, group.antifundamental)
    return complex(
        jnp.einsum(
            "ABab,ACacxy,CDcd,DFdfyz,EFef,HEhewz,GHgh,BGbgwx->",
            tl_overlap,
            top_link,
            tr_overlap,
            right_link,
            br_overlap,
            bottom_link,
            bl_overlap,
            left_link,
            optimize=True,
        )
    )


@cache
def _vertex_overlap(
    input_block: VertexBlock,
    output_block: VertexBlock,
    *,
    affected_axes: tuple[int, int],
) -> jax.Array:
    input_tensor = vertex_intertwiner_tensor(input_block)
    output_tensor = vertex_intertwiner_tensor(output_block)
    external_axes = tuple(axis for axis in range(4) if axis not in affected_axes)
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
    return (output_flat.conj() @ input_flat.T).reshape(*out_dims, *in_dims)


def build_pure_gauge_vertex_blocks(
    group: SU3,
    *,
    active_legs: tuple[bool, bool, bool, bool],
    target_charge: int = 0,
) -> tuple[VertexBlock, ...]:
    if target_charge != 0:
        raise NotImplementedError("SU(3) backend currently supports singlet target_charge=0.")
    if len(active_legs) != 4:
        raise ValueError("active_legs must be ordered as (left, up, right, down).")
    choices = tuple(group.irreps() if active else (0,) for active in active_legs)
    blocks: list[VertexBlock] = []
    for j_l, j_u, j_r, j_d in product(*choices):
        labels = (group.dual(j_l), group.dual(j_u), j_r, j_d)
        for iota, _tensor in enumerate(_invariant_basis(labels)):
            blocks.append(
                VertexBlock(
                    j_l=j_l,
                    j_u=j_u,
                    j_r=j_r,
                    j_d=j_d,
                    iota=iota,
                    internal_irreps=(),
                )
            )
    return tuple(sorted(blocks))


@builders.build_pure_gauge_tables.dispatch
def build_pure_gauge_tables(
    group: SU3,
    *,
    shape: tuple[int, int],
    target_charge: int = 0,
) -> PureGaugeTables:
    n_rows, n_cols = shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("shape must have positive dimensions.")
    rows: list[tuple[tuple[VertexBlock, ...], ...]] = []
    lookup_rows: list[tuple[dict[tuple[int, int, int, int, int], int], ...]] = []
    for row_idx in range(n_rows):
        row = []
        lookup_row = []
        for col_idx in range(n_cols):
            active_legs = (
                col_idx > 0,
                row_idx > 0,
                col_idx < n_cols - 1,
                row_idx < n_rows - 1,
            )
            blocks = build_pure_gauge_vertex_blocks(
                group,
                active_legs=active_legs,
                target_charge=target_charge,
            )
            row.append(blocks)
            lookup_row.append(
                {
                    (block.j_l, block.j_u, block.j_r, block.j_d, block.iota): block_id
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
    block_id_lookup = jnp.full(
        (n_rows, n_cols, n_irreps, n_irreps, n_irreps, n_irreps, max_iotas),
        -1,
        dtype=jnp.int32,
    )
    for row_idx, row in enumerate(blocks_by_site):
        for col_idx, blocks in enumerate(row):
            for block_id, block in enumerate(blocks):
                block_id_lookup = block_id_lookup.at[
                    row_idx,
                    col_idx,
                    block.j_l,
                    block.j_u,
                    block.j_r,
                    block.j_d,
                    block.iota,
                ].set(block_id)
    return PureGaugeTables(
        group=group,
        shape=shape,
        blocks=blocks_by_site,
        _block_ids=tuple(lookup_rows),
        max_iotas=max_iotas,
        block_id_lookup=block_id_lookup,
    )

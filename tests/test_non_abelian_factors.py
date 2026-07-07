"""Validation gate for the canonical non-Abelian factor tables.

Anchors are external to the table machinery:

- Plaquette vacuum<->loop matrix elements are Haar-measure ground truths:
  the loop basis state saturates |<loop|tr U_plaq|vac>| = ||tr U_plaq|vac>||,
  giving |ME| = 1 exactly for SU(2) (any j_max) and SU(3); the codebase basis
  phase makes it -1 for SU(2) and +1 for SU(3).
- Hopping values come from the exact Gauss-sector construction on the full
  matter-Fock x link space (quark law V at both sublattices, dual-metric
  matter slot): the meson-hop reduced element is exactly 1, realized in the
  tables as lambda_src * lambda_tgt = (-1)(-1), with the forward orientation
  creating at src / annihilating at tgt and the backward its conjugate.
- Building any table runs the ring == prod(lambda) split asserts internally,
  so successful construction is itself part of the gate.
"""

import numpy as np
import jax

from nonabelian_exact import decode_rows
from vmc.gauge_groups import su2 as su2_module
from vmc.gauge_groups import SU2, SU3
from vmc.peps.non_abelian_gi.builders import build_pure_gauge_tables
from vmc.peps.non_abelian_gi.factors import (
    _oriented_hopping_matrix_element,
    _oriented_plaquette_matrix_element,
    build_hopping_factor_tables,
    build_plaquette_factor_tables,
    fundamental_irrep,
)

jax.config.update("jax_enable_x64", True)


def _blk(tables, r, c, *legs, matter_state=0):
    return tables.blocks[r][c][
        tables.block_id(r, c, *legs, 0, matter_state=matter_state)
    ]


def _vacuum_and_loop(tables, loop_legs):
    vac = tuple(
        _blk(tables, r, c, 0, 0, 0, 0) for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    loop = tuple(
        _blk(tables, r, c, *legs)
        for (r, c), legs in zip(((0, 0), (0, 1), (1, 0), (1, 1)), loop_legs)
    )
    return vac, loop


SU2_LOOP_LEGS = ((0, 0, 1, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 0, 0))
SU3_LOOP_LEGS = ((0, 0, 1, 2), (1, 0, 0, 1), (0, 2, 2, 0), (2, 1, 0, 0))


def test_su2_plaquette_haar_anchor():
    for j_max_twice in (1, 2):
        group = SU2(j_max_twice)
        tables = build_pure_gauge_tables(group, shape=(2, 2))
        build_plaquette_factor_tables(group, tables)  # split asserts
        vac, loop = _vacuum_and_loop(tables, SU2_LOOP_LEGS)
        op = fundamental_irrep(group)
        fwd = _oriented_plaquette_matrix_element(group, vac, loop, op)
        bwd = _oriented_plaquette_matrix_element(group, loop, vac, op)
        np.testing.assert_allclose(fwd, -1.0, atol=1e-12)
        np.testing.assert_allclose(np.conj(bwd), fwd, atol=1e-12)


def test_su3_plaquette_haar_anchor():
    group = SU3(1)
    tables = build_pure_gauge_tables(group, shape=(2, 2))
    build_plaquette_factor_tables(group, tables)  # split asserts
    vac, loop = _vacuum_and_loop(tables, SU3_LOOP_LEGS)
    op = fundamental_irrep(group)
    fwd = _oriented_plaquette_matrix_element(group, vac, loop, op)
    np.testing.assert_allclose(fwd, 1.0, atol=1e-12)


def test_su2_matter_vertices_are_gauss_invariant():
    """Vertex tensors intertwine V at incoming/matter slots, conj(V) at
    outgoing slots — the transformation laws fixed by u -> V_src u V_tgt^dag
    and invariance of psi^dag U psi (exact Gauss-sector construction)."""
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    V = np.array(
        [[q[0] + 1j * q[1], q[2] + 1j * q[3]], [-q[2] + 1j * q[3], q[0] - 1j * q[1]]]
    )
    rot = {0: np.ones((1, 1)), 1: V}
    dual_rot = {0: np.ones((1, 1)), 1: V.conj()}
    group = SU2(1)
    tables = build_pure_gauge_tables(
        group, shape=(2, 3), matter_irreps=(0, 1), matter_numbers=(0, 1)
    )
    for row in tables.blocks:
        for site_blocks in row:
            for block in site_blocks:
                t = np.asarray(su2_module.vertex_intertwiner_tensor(block))
                if t.ndim == 4:
                    t = t[..., None]
                rotated = np.einsum(
                    "lurdq,lL,uU,rR,dD,qQ->LURDQ",
                    t,
                    rot[block.j_l].T,
                    rot[block.j_u].T,
                    dual_rot[block.j_r].T,
                    dual_rot[block.j_d].T,
                    rot[block.matter_irrep].T,
                    optimize=True,
                )
                np.testing.assert_allclose(rotated, t, atol=1e-12)


def test_su2_hopping_exact_anchor():
    group = SU2(1)
    tables = build_pure_gauge_tables(
        group, shape=(2, 2), matter_irreps=(0, 1), matter_numbers=(0, 1)
    )
    hop = build_hopping_factor_tables(group, tables)  # split asserts

    # Meson hop (0,0)->(0,1) along the top link: exact Gauss-sector ME = 1,
    # realized in the backward (annihilate-at-src) orientation.
    ins = (
        _blk(tables, 0, 0, 0, 0, 0, 1, matter_state=1),
        _blk(tables, 0, 1, 0, 0, 0, 0),
    )
    outs = (
        _blk(tables, 0, 0, 0, 0, 1, 1),
        _blk(tables, 0, 1, 1, 0, 0, 0, matter_state=1),
    )
    bwd = _oriented_hopping_matrix_element(group, outs, ins, 1, horizontal=True)
    np.testing.assert_allclose(bwd, 1.0, atol=1e-12)

    # Occupancy filtering: forward rows create at src (m0 -> m1) and
    # annihilate at tgt (m1 -> m0); backward rows are the conjugates.
    for grid, is_src in (
        (hop.h_src, True),
        (hop.h_tgt, False),
        (hop.v_src, True),
        (hop.v_tgt, False),
    ):
        for r, row in enumerate(grid):
            for c, entry in enumerate(row):
                if entry is None:
                    continue
                fwd_t, bwd_t = entry
                blocks = tables.blocks[r][c]
                for table, shift in (
                    (fwd_t, 1 if is_src else -1),
                    (bwd_t, -1 if is_src else 1),
                ):
                    assert int(np.asarray(table.group_counts).sum()) == 2, (
                        "j_max=1/2 hop rows per endpoint"
                    )
                    for b_in in range(table.group_starts.shape[0]):
                        for new in range(table.group_starts.shape[1]):
                            for b_out, lam in decode_rows(table, b_in, (new,)):
                                assert (
                                    blocks[b_out].matter_number
                                    - blocks[b_in].matter_number
                                ) == shift
                                np.testing.assert_allclose(abs(lam), 1.0, atol=1e-12)


def test_hopping_hermitian_pairing():
    group = SU2(1)
    tables = build_pure_gauge_tables(
        group, shape=(2, 2), matter_irreps=(0, 1), matter_numbers=(0, 1)
    )
    hop = build_hopping_factor_tables(group, tables)
    for grid in (hop.h_src, hop.h_tgt, hop.v_src, hop.v_tgt):
        for r, row in enumerate(grid):
            for c, entry in enumerate(row):
                if entry is None:
                    continue
                fwd_t, bwd_t = entry

                def rows(table):
                    return {
                        (b_in, b_out): lam
                        for b_in in range(table.group_starts.shape[0])
                        for new in range(table.group_starts.shape[1])
                        for b_out, lam in decode_rows(table, b_in, (new,))
                    }

                fwd_rows, bwd_rows = rows(fwd_t), rows(bwd_t)
                assert set(fwd_rows) == {(b, a) for a, b in bwd_rows}
                for (a, b), lam in fwd_rows.items():
                    np.testing.assert_allclose(
                        bwd_rows[(b, a)], np.conj(lam), atol=1e-12
                    )

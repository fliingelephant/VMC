import pytest
import jax
import jax.numpy as jnp

from vmc.gauge_groups import SU2
from vmc.gauge_groups.su2 import (
    build_pure_gauge_vertex_blocks,
    clebsch_gordan,
    vertex_intertwiner_tensor,
)
from vmc.peps.non_abelian_gi import build_pure_gauge_tables


def test_su2_irreps_and_casimir_use_twice_j_labels():
    group = SU2(j_max_twice=2)

    assert group.irreps() == (0, 1, 2)
    assert group.dim(0) == 1
    assert group.dim(1) == 2
    assert group.dim(2) == 3
    assert group.casimir(0) == pytest.approx(0.0)
    assert group.casimir(1) == pytest.approx(0.75)
    assert group.casimir(2) == pytest.approx(2.0)


def test_su2_fusion_rules_respect_truncation():
    group = SU2(j_max_twice=2)

    assert group.fuse(0, 1) == (1,)
    assert group.fuse(1, 1) == (0, 2)
    assert group.fuse(1, 2) == (1,)
    assert group.fuse(2, 2) == (0, 2)
    assert group.fuse(1, 3) == (2,)
    assert SU2(j_max_twice=0).fuse(0, 1) == ()


def test_clebsch_gordan_matches_known_condon_shortley_values():
    assert clebsch_gordan(1, 1, 1, 1, 2, 2) == pytest.approx(1.0)
    assert clebsch_gordan(1, 1, 1, -1, 2, 0) == pytest.approx(1 / jnp.sqrt(2))
    assert clebsch_gordan(1, -1, 1, 1, 2, 0) == pytest.approx(1 / jnp.sqrt(2))
    assert clebsch_gordan(1, 1, 1, -1, 0, 0) == pytest.approx(1 / jnp.sqrt(2))
    assert clebsch_gordan(1, -1, 1, 1, 0, 0) == pytest.approx(-1 / jnp.sqrt(2))


def test_clebsch_gordan_blocks_are_orthonormal():
    columns = []
    coupled = [(0, 0), (2, -2), (2, 0), (2, 2)]
    for j_out, m_out in coupled:
        columns.append(
            [
                clebsch_gordan(1, m_l, 1, m_r, j_out, m_out)
                for m_l in (-1, 1)
                for m_r in (-1, 1)
            ]
        )
    matrix = jnp.asarray(columns)

    assert jnp.allclose(matrix @ matrix.T, jnp.eye(4))


def test_vertex_intertwiner_tensors_are_orthonormal_for_four_spin_halves():
    group = SU2(j_max_twice=1)
    blocks = [
        block
        for block in build_pure_gauge_vertex_blocks(
            group,
            active_legs=(True, True, True, True),
        )
        if block.j_l == block.j_u == block.j_r == block.j_d == 1
    ]

    tensors = [vertex_intertwiner_tensor(block) for block in blocks]
    overlaps = jnp.asarray(
        [[jnp.vdot(left, right) for right in tensors] for left in tensors]
    )

    assert jnp.allclose(overlaps, jnp.eye(2))


def test_su2_rejects_invalid_twice_j_labels():
    group = SU2(j_max_twice=2)

    with pytest.raises(ValueError, match="valid SU\\(2\\) irrep"):
        group.dim(-1)
    with pytest.raises(ValueError, match="within truncation"):
        group.fuse(3, 1)
    with pytest.raises(ValueError, match="valid SU\\(2\\) irrep"):
        group.fuse(1, -1)
    with pytest.raises(ValueError, match="integer"):
        SU2(j_max_twice=1.5)


def test_pure_gauge_jmax_half_vertex_block_counts_match_hand_table():
    group = SU2(j_max_twice=1)

    corner = build_pure_gauge_vertex_blocks(
        group,
        active_legs=(False, False, True, True),
    )
    edge = build_pure_gauge_vertex_blocks(
        group,
        active_legs=(True, False, True, True),
    )
    bulk = build_pure_gauge_vertex_blocks(
        group,
        active_legs=(True, True, True, True),
    )

    assert len(corner) == 2
    assert len(edge) == 4
    assert len(bulk) == 9


def test_four_spin_half_bulk_tuple_has_two_intertwiners():
    group = SU2(j_max_twice=1)
    blocks = build_pure_gauge_vertex_blocks(
        group,
        active_legs=(True, True, True, True),
    )

    half_blocks = [
        block
        for block in blocks
        if block.j_l == block.j_u == block.j_r == block.j_d == 1
    ]

    assert [block.iota for block in half_blocks] == [0, 1]
    assert [block.internal_irreps for block in half_blocks] == [(0, 0), (2, 2)]


def test_pure_gauge_tables_are_boundary_aware():
    tables = build_pure_gauge_tables(SU2(j_max_twice=1), shape=(3, 3))

    assert tables.n_blocks(0, 0) == 2
    assert tables.n_blocks(0, 1) == 4
    assert tables.n_blocks(1, 1) == 9
    assert tables.active_legs(0, 0) == (False, False, True, True)
    assert tables.active_legs(0, 1) == (True, False, True, True)
    assert tables.active_legs(1, 1) == (True, True, True, True)


def test_pure_gauge_block_lookup_roundtrip():
    tables = build_pure_gauge_tables(SU2(j_max_twice=1), shape=(3, 3))

    for r in range(3):
        for c in range(3):
            for block_id, block in enumerate(tables.blocks[r][c]):
                assert (
                    tables.block_id(
                        r, c, block.j_l, block.j_u, block.j_r, block.j_d, block.iota
                    )
                    == block_id
                )


def test_pure_gauge_block_lookup_rejects_invalid_local_state():
    tables = build_pure_gauge_tables(SU2(j_max_twice=1), shape=(3, 3))

    with pytest.raises(ValueError, match="No vertex block"):
        tables.block_id(1, 1, 1, 0, 0, 0, 0)
    with pytest.raises(IndexError, match="outside shape"):
        tables.block_id(3, 0, 0, 0, 0, 0, 0)


def test_pure_gauge_block_lookup_array_roundtrip():
    tables = build_pure_gauge_tables(SU2(j_max_twice=1), shape=(3, 3))

    assert tables.max_iotas == 2
    assert tables.block_id_lookup.shape == (3, 3, 1, 2, 2, 2, 2, 2)
    assert tables.block_id_lookup[1, 1, 0, 1, 0, 0, 0, 0] == -1
    for r in range(3):
        for c in range(3):
            for block_id, block in enumerate(tables.blocks[r][c]):
                assert (
                    tables.block_id_lookup[
                        r,
                        c,
                        0,
                        block.j_l,
                        block.j_u,
                        block.j_r,
                        block.j_d,
                        block.iota,
                    ]
                    == block_id
                )


def test_su2_matter_tables_store_only_allowed_matter_blocks():
    tables = build_pure_gauge_tables(
        SU2(j_max_twice=1),
        shape=(1, 2),
        matter_irreps=(0, 1),
        matter_numbers=(0, 1),
    )

    left_vacuum = tables.block_id(0, 0, 0, 0, 0, 0, 0, matter_state=0)
    left_occupied = tables.block_id(0, 0, 0, 0, 1, 0, 0, matter_state=1)
    right_vacuum = tables.block_id(0, 1, 0, 0, 0, 0, 0, matter_state=0)
    right_occupied = tables.block_id(0, 1, 1, 0, 0, 0, 0, matter_state=1)

    assert tables.phys_dim == 2
    assert tables.matter_irreps == (0, 1)
    assert tables.matter_numbers == (0, 1)
    assert tables.block_id_lookup.shape == (1, 2, 2, 2, 2, 2, 2, 1)
    assert tables.n_blocks(0, 0) == 2
    assert tables.n_blocks(0, 1) == 2
    assert tables.block_id_lookup[0, 0, 1, 0, 0, 1, 0, 0] == left_occupied
    assert tables.block_id_lookup[0, 1, 1, 1, 0, 0, 0, 0] == right_occupied
    assert int(tables.matter_state_by_block[0, 0, left_vacuum]) == 0
    assert int(tables.matter_state_by_block[0, 0, left_occupied]) == 1
    assert int(tables.matter_state_by_block[0, 1, right_vacuum]) == 0
    assert int(tables.matter_state_by_block[0, 1, right_occupied]) == 1

    with pytest.raises(ValueError, match="No vertex block"):
        tables.block_id(0, 0, 0, 0, 0, 0, 0, matter_state=1)


def test_static_numeric_tables_are_jax_arrays():
    tables = build_pure_gauge_tables(SU2(j_max_twice=1), shape=(2, 2))

    assert isinstance(tables.block_id_lookup, jax.Array)
    assert isinstance(tables.j_r_by_block, jax.Array)
    assert isinstance(tables.iota_by_block, jax.Array)

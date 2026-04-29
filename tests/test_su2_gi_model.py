import jax.numpy as jnp
from flax import nnx

from vmc.peps.common.strategy import NoTruncation
from vmc.peps.non_abelian_gi import (
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    build_row_mpo,
)
from vmc.gauge_groups import SU2
from vmc.utils.utils import random_tensor


def test_su2_gipeps_is_exported_from_public_peps_surface():
    from vmc.peps import NonAbelianGIPEPS as ExportedNonAbelianGIPEPS
    from vmc.peps import NonAbelianGIPEPSConfig as ExportedNonAbelianGIPEPSConfig

    assert ExportedNonAbelianGIPEPS is NonAbelianGIPEPS
    assert ExportedNonAbelianGIPEPSConfig is NonAbelianGIPEPSConfig


def test_su2_gipeps_uses_generic_non_abelian_sampled_block_contract():
    from vmc.peps.non_abelian_gi import (
        NonAbelianGIPEPS,
        PlaquetteMatrixTable,
        PureGaugeTables,
    )

    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )

    assert isinstance(model, NonAbelianGIPEPS)
    assert isinstance(model.tables, PureGaugeTables)
    assert isinstance(model.plaquette_matrix_tables[0][0], PlaquetteMatrixTable)
    assert model.horizontal_hopping_matrix_tables == ((), ())
    assert model.vertical_hopping_matrix_tables == ((),)


def test_su2_kernel_dispatch_is_registered_from_public_peps_surface():
    from vmc.operators.local_terms import LocalHamiltonian
    from vmc.peps import build_link_casimir_terms as exported_link_terms
    from vmc.peps import build_mc_kernels

    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(1, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=exported_link_terms(model.shape, model.gauge_group),
    )

    init_cache, transition, estimate = build_mc_kernels(model, operator)

    assert callable(init_cache)
    assert callable(transition)
    assert callable(estimate)


def test_su2_gipeps_initializes_boundary_aware_tensor_blocks():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(3, 3),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )

    assert model.tensors[0][0].get_value().shape == (2, 1, 2, 1, 2)
    assert model.tensors[0][1].get_value().shape == (4, 1, 2, 2, 2)
    assert model.tensors[1][1].get_value().shape == (9, 2, 2, 2, 2)
    assert model.params_per_site == (4, 8, 4, 8, 16, 8, 4, 8, 4)
    assert model.sliced_dims == (2, 4, 2, 4, 9, 4, 2, 4, 2)


def test_su2_gipeps_initial_tensors_are_unbiased_random_blocks():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )
    expected = random_tensor(
        nnx.Rngs(0),
        model.tensors[0][0].get_value().shape,
        model.dtype,
    ) / jnp.sqrt(model.params_per_site[0])

    assert jnp.array_equal(model.tensors[0][0].get_value(), expected)


def test_su2_gipeps_closes_over_plaquette_link_transition_metadata():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )

    assert model.plaquette_link_transitions.outputs(0, 0, 0, 0) == ((1, 1, 1, 1),)


def test_su2_gipeps_closes_over_plaquette_matrix_metadata():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )

    matrix_table = model.plaquette_matrix_tables[0][0]
    output_blocks = matrix_table.output_block_ids[
        matrix_table.flat_index((0, 0, 0, 0), 0)
    ]
    output_links = [
        model.tables.j_r_by_block[0, 0, output_blocks[0]],
        model.tables.j_d_by_block[0, 1, output_blocks[1]],
        model.tables.j_r_by_block[1, 0, output_blocks[2]],
        model.tables.j_d_by_block[0, 0, output_blocks[0]],
    ]

    assert matrix_table.counts[0, 0, 0, 0] == 1
    assert [int(value) for value in output_links] == [1, 1, 1, 1]


def test_su2_gipeps_flatten_unflatten_roundtrip():
    shape = (2, 3)
    h_links = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    v_links = jnp.array([[1, 0, 1]], dtype=jnp.int32)
    iotas = jnp.array([[0, 0, 0], [0, 0, 0]], dtype=jnp.int32)

    flat = NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)
    h_next, v_next, iotas_next = NonAbelianGIPEPS.unflatten_sample(flat, shape)

    assert jnp.array_equal(h_next, h_links)
    assert jnp.array_equal(v_next, v_links)
    assert jnp.array_equal(iotas_next, iotas)


def test_su2_gipeps_all_zero_sample_has_valid_shape_and_dtype():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(3, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )

    sample = model.all_zero_sample()
    h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_sample(sample, model.shape)

    assert sample.dtype == jnp.int32
    assert h_links.shape == (3, 1)
    assert v_links.shape == (2, 2)
    assert iotas.shape == (3, 2)
    assert jnp.array_equal(sample, jnp.zeros_like(sample))


def test_su2_gipeps_matter_uses_allowed_blocks_not_dense_physical_axis():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(1, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
    )

    samples = model.random_physical_configuration(
        jnp.array([0, 1], dtype=jnp.uint32),
        n_samples=4,
    )
    matter, h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_matter_sample(
        samples[0],
        model.shape,
    )

    assert model.tensors[0][0].get_value().shape == (2, 1, 1, 1, 2)
    assert model.tensors[0][1].get_value().shape == (2, 1, 1, 2, 1)
    assert model.sliced_dims == (2, 2)
    assert samples.shape == (4, 5)
    assert jnp.array_equal(matter, jnp.ones(model.shape, dtype=jnp.int32))
    assert jnp.array_equal(h_links, jnp.ones((1, 1), dtype=jnp.int32))
    assert v_links.shape == (0, 2)
    assert jnp.array_equal(iotas, jnp.zeros(model.shape, dtype=jnp.int32))
    for sample in samples:
        matter, _h_links, _v_links, _iotas = NonAbelianGIPEPS.unflatten_matter_sample(
            sample,
            model.shape,
        )
        assert int(jnp.sum(matter)) == model.particle_number
        model.active_block_ids(sample)


def test_su2_gipeps_random_physical_configuration_returns_valid_batch():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )

    samples = model.random_physical_configuration(
        jnp.array([0, 1], dtype=jnp.uint32),
        n_samples=16,
    )

    assert samples.shape == (16, model.all_zero_sample().size)
    assert jnp.unique(samples, axis=0).shape[0] > 1
    for sample in samples:
        model.active_block_ids(sample)


def test_su2_gipeps_active_block_ids_from_sample():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )
    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros((2, 2), dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)

    assert jnp.array_equal(
        model.active_block_ids(sample),
        jnp.array(
            [
                [
                    model.tables.block_id(0, 0, 0, 0, 1, 1, 0),
                    model.tables.block_id(0, 1, 1, 0, 0, 1, 0),
                ],
                [
                    model.tables.block_id(1, 0, 0, 1, 1, 0, 0),
                    model.tables.block_id(1, 1, 1, 1, 0, 0, 0),
                ],
            ],
            dtype=jnp.int32,
        ),
    )


def test_su2_gipeps_active_block_ids_reject_invalid_iota():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
    )
    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.ones((2, 2), dtype=jnp.int32)

    try:
        model.active_block_ids(NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas))
    except ValueError as exc:
        assert "Invalid non-Abelian local block" in str(exc)
    else:
        raise AssertionError("Expected invalid iota to be rejected.")


def test_su2_gipeps_row_mpo_selects_active_blocks_in_common_axis_order():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
        contraction_strategy=NoTruncation(),
    )
    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros((2, 2), dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)
    tensors = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]

    row_mpo = build_row_mpo(
        tensors, sample, model.shape, model.tables.block_id_lookup, row=0
    )
    block_id = model.tables.block_id(0, 0, 0, 0, 1, 1, 0)

    assert row_mpo[0].shape == (1, 2, 1, 2)
    assert row_mpo[1].shape == (2, 1, 1, 2)
    assert jnp.array_equal(
        row_mpo[0],
        jnp.transpose(tensors[0][0][block_id], (2, 3, 0, 1)),
    )


def test_su2_gipeps_apply_contracts_one_site_selected_block():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(1, 1),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
        ),
        contraction_strategy=NoTruncation(),
    )
    tensors = [[jnp.asarray([[[[[3.0 + 2.0j]]]]], dtype=jnp.complex128)]]

    assert (
        model.apply(
            tensors,
            model.all_zero_sample(),
            model.shape,
            model.tables,
            model.strategy,
        )
        == 3.0 + 2.0j
    )

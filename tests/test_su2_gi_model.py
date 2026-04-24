import jax.numpy as jnp
from flax import nnx

from vmc.peps.common.strategy import NoTruncation
from vmc.peps.su2_gi.compat import build_row_mpo
from vmc.peps.su2_gi.model import SU2GIPEPS, SU2GIPEPSConfig
from vmc.utils.utils import random_tensor


def test_su2_gipeps_is_exported_from_public_peps_surface():
    from vmc.peps import SU2GIPEPS as ExportedSU2GIPEPS
    from vmc.peps import SU2GIPEPSConfig as ExportedSU2GIPEPSConfig

    assert ExportedSU2GIPEPS is SU2GIPEPS
    assert ExportedSU2GIPEPSConfig is SU2GIPEPSConfig


def test_su2_kernel_dispatch_is_registered_from_public_peps_surface():
    from vmc.operators.local_terms import LocalHamiltonian
    from vmc.peps import build_link_casimir_terms as exported_link_terms
    from vmc.peps import build_mc_kernels

    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(1, 2), j_max_twice=1, D=2, chi=4),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=exported_link_terms(model.shape, model.group),
    )

    init_cache, transition, estimate = build_mc_kernels(model, operator)

    assert callable(init_cache)
    assert callable(transition)
    assert callable(estimate)


def test_su2_gipeps_initializes_boundary_aware_tensor_blocks():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(3, 3), j_max_twice=1, D=2, chi=4),
    )

    assert model.tensors[0][0].get_value().shape == (2, 1, 2, 1, 2)
    assert model.tensors[0][1].get_value().shape == (4, 1, 2, 2, 2)
    assert model.tensors[1][1].get_value().shape == (9, 2, 2, 2, 2)
    assert model.params_per_site == (4, 8, 4, 8, 16, 8, 4, 8, 4)
    assert model.sliced_dims == (2, 4, 2, 4, 9, 4, 2, 4, 2)


def test_su2_gipeps_initial_tensors_are_unbiased_random_blocks():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(2, 2), j_max_twice=1, D=2, chi=4),
    )
    expected = random_tensor(
        nnx.Rngs(0),
        model.tensors[0][0].get_value().shape,
        model.dtype,
    ) / jnp.sqrt(model.params_per_site[0])

    assert jnp.array_equal(model.tensors[0][0].get_value(), expected)


def test_su2_gipeps_closes_over_plaquette_link_transition_metadata():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(2, 2), j_max_twice=1, D=2, chi=4),
    )

    assert model.plaquette_link_transitions.outputs(0, 0, 0, 0) == ((1, 1, 1, 1),)


def test_su2_gipeps_closes_over_plaquette_matrix_metadata():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(2, 2), j_max_twice=1, D=2, chi=4),
    )

    matrix_table = model.plaquette_matrix_tables[0][0]

    assert matrix_table.counts[0, 0, 0, 0] == 1
    assert matrix_table.output_links[0, 0, 0, 0, 0].tolist() == [1, 1, 1, 1]


def test_su2_gipeps_flatten_unflatten_roundtrip():
    shape = (2, 3)
    h_links = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    v_links = jnp.array([[1, 0, 1]], dtype=jnp.int32)
    iotas = jnp.array([[0, 0, 0], [0, 0, 0]], dtype=jnp.int32)

    flat = SU2GIPEPS.flatten_sample(h_links, v_links, iotas)
    h_next, v_next, iotas_next = SU2GIPEPS.unflatten_sample(flat, shape)

    assert jnp.array_equal(h_next, h_links)
    assert jnp.array_equal(v_next, v_links)
    assert jnp.array_equal(iotas_next, iotas)


def test_su2_gipeps_all_zero_sample_has_valid_shape_and_dtype():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(3, 2), j_max_twice=1, D=2, chi=4),
    )

    sample = model.all_zero_sample()
    h_links, v_links, iotas = SU2GIPEPS.unflatten_sample(sample, model.shape)

    assert sample.dtype == jnp.int32
    assert h_links.shape == (3, 1)
    assert v_links.shape == (2, 2)
    assert iotas.shape == (3, 2)
    assert jnp.array_equal(sample, jnp.zeros_like(sample))


def test_su2_gipeps_random_physical_configuration_returns_valid_batch():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(2, 2), j_max_twice=1, D=2, chi=4),
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
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(2, 2), j_max_twice=1, D=2, chi=4),
    )
    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros((2, 2), dtype=jnp.int32)
    sample = SU2GIPEPS.flatten_sample(h_links, v_links, iotas)

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
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(2, 2), j_max_twice=1, D=2, chi=4),
    )
    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.ones((2, 2), dtype=jnp.int32)

    try:
        model.active_block_ids(SU2GIPEPS.flatten_sample(h_links, v_links, iotas))
    except ValueError as exc:
        assert "Invalid SU(2) local block" in str(exc)
    else:
        raise AssertionError("Expected invalid iota to be rejected.")


def test_su2_gipeps_row_mpo_selects_active_blocks_in_common_axis_order():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros((2, 2), dtype=jnp.int32)
    sample = SU2GIPEPS.flatten_sample(h_links, v_links, iotas)
    tensors = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]

    row_mpo = build_row_mpo(tensors, sample, model.shape, model.tables, row=0)
    block_id = model.tables.block_id(0, 0, 0, 0, 1, 1, 0)

    assert row_mpo[0].shape == (1, 2, 1, 2)
    assert row_mpo[1].shape == (2, 1, 1, 2)
    assert jnp.array_equal(
        row_mpo[0],
        jnp.transpose(tensors[0][0][block_id], (2, 3, 0, 1)),
    )


def test_su2_gipeps_apply_contracts_one_site_selected_block():
    model = SU2GIPEPS(
        rngs=nnx.Rngs(0),
        config=SU2GIPEPSConfig(shape=(1, 1), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    tensors = [[jnp.asarray([[[[[3.0 + 2.0j]]]]], dtype=jnp.complex128)]]

    assert model.apply(
        tensors,
        model.all_zero_sample(),
        model.shape,
        model.tables,
        model.strategy,
    ) == 3.0 + 2.0j

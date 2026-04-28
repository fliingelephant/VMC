import itertools

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.gauge_groups import SU3
from vmc.operators.local_terms import LocalHamiltonian
from vmc.peps.non_abelian_gi import (
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    PlaquetteTerm,
    build_link_casimir_terms,
    build_plaquette_link_transitions,
    build_plaquette_matrix_table,
    build_pure_gauge_tables,
)
from vmc.peps.standard.kernels import build_mc_kernels
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_svd


def _plaquette_candidate_sample(
    model: NonAbelianGIPEPS,
    sample: jax.Array,
    out_idx: int,
    *,
    row: int = 0,
    col: int = 0,
) -> jax.Array:
    h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_sample(sample, model.shape)
    active_blocks = model.active_block_ids(sample)
    input_blocks = (
        int(active_blocks[row, col]),
        int(active_blocks[row, col + 1]),
        int(active_blocks[row + 1, col]),
        int(active_blocks[row + 1, col + 1]),
    )
    matrix_table = model.plaquette_matrix_tables[row][col]
    output_blocks = matrix_table.output_block_ids[
        matrix_table.flat_index(input_blocks, out_idx)
    ]
    links = jnp.stack(
        [
            model.tables.j_r_by_block[row, col, output_blocks[0]],
            model.tables.j_d_by_block[row, col + 1, output_blocks[1]],
            model.tables.j_r_by_block[row + 1, col, output_blocks[2]],
            model.tables.j_d_by_block[row, col, output_blocks[0]],
        ]
    )
    output_iotas = jnp.stack(
        [
            model.tables.iota_by_block[row, col, output_blocks[0]],
            model.tables.iota_by_block[row, col + 1, output_blocks[1]],
            model.tables.iota_by_block[row + 1, col, output_blocks[2]],
            model.tables.iota_by_block[row + 1, col + 1, output_blocks[3]],
        ]
    )
    h_links = h_links.at[row, col].set(links[0])
    h_links = h_links.at[row + 1, col].set(links[2])
    v_links = v_links.at[row, col + 1].set(links[1])
    v_links = v_links.at[row, col].set(links[3])
    for (dr, dc), iota in zip(
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        output_iotas,
        strict=True,
    ):
        iotas = iotas.at[row + dr, col + dc].set(iota)
    return NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)


def _valid_samples(model: NonAbelianGIPEPS) -> tuple[jax.Array, ...]:
    n_rows, n_cols = model.shape
    link_irreps = model.gauge_group.irreps()
    samples = []
    for h_values in itertools.product(link_irreps, repeat=n_rows * (n_cols - 1)):
        h_links = jnp.asarray(h_values, dtype=jnp.int32).reshape(
            (n_rows, n_cols - 1)
        )
        for v_values in itertools.product(link_irreps, repeat=(n_rows - 1) * n_cols):
            v_links = jnp.asarray(v_values, dtype=jnp.int32).reshape(
                (n_rows - 1, n_cols)
            )
            iota_choices = []
            for row in range(n_rows):
                for col in range(n_cols):
                    choices = []
                    for iota in range(model.tables.max_iotas):
                        block_id = model.tables.block_id_lookup[
                            row,
                            col,
                            0,
                            h_links[row, col - 1] if col > 0 else 0,
                            v_links[row - 1, col] if row > 0 else 0,
                            h_links[row, col] if col < n_cols - 1 else 0,
                            v_links[row, col] if row < n_rows - 1 else 0,
                            iota,
                        ]
                        if int(block_id) >= 0:
                            choices.append(iota)
                    if not choices:
                        break
                    iota_choices.append(tuple(choices))
                else:
                    continue
                break
            if len(iota_choices) != n_rows * n_cols:
                continue
            for iotas in itertools.product(*iota_choices):
                samples.append(
                    NonAbelianGIPEPS.flatten_sample(
                        h_links,
                        v_links,
                        jnp.asarray(iotas, dtype=jnp.int32).reshape(model.shape),
                    )
                )
    return tuple(samples)


def _exact_pure_gauge_hamiltonian(
    model: NonAbelianGIPEPS,
    *,
    electric_coeff: float,
    plaquette_coeff: float,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    samples = _valid_samples(model)
    sample_keys = {tuple(sample.tolist()): idx for idx, sample in enumerate(samples)}
    hamiltonian = jnp.zeros((len(samples), len(samples)), dtype=jnp.complex128)
    for source_idx, sample in enumerate(samples):
        h_links, v_links, _iotas = NonAbelianGIPEPS.unflatten_sample(
            sample,
            model.shape,
        )
        electric = sum(
            electric_coeff * model.gauge_group.casimir(int(link))
            for link in (*h_links.reshape(-1), *v_links.reshape(-1))
        )
        hamiltonian = hamiltonian.at[source_idx, source_idx].set(electric)
        active_blocks = model.active_block_ids(sample)
        for row in range(model.shape[0] - 1):
            for col in range(model.shape[1] - 1):
                table = model.plaquette_matrix_tables[row][col]
                input_blocks = (
                    int(active_blocks[row, col]),
                    int(active_blocks[row, col + 1]),
                    int(active_blocks[row + 1, col]),
                    int(active_blocks[row + 1, col + 1]),
                )
                for out_idx in range(int(table.counts[input_blocks])):
                    candidate = _plaquette_candidate_sample(
                        model,
                        sample,
                        out_idx,
                        row=row,
                        col=col,
                    )
                    target_idx = sample_keys[tuple(candidate.tolist())]
                    hamiltonian = hamiltonian.at[target_idx, source_idx].add(
                        plaquette_coeff
                        * table.matrix_elements[
                            table.flat_index(input_blocks, out_idx)
                        ]
                    )
    return samples, hamiltonian


def test_su3_fundamental_truncation_metadata():
    group = SU3(max_weight_sum=1)

    assert group.irreps() == (0, 1, 2)
    assert group.highest_weight(0) == (0, 0)
    assert group.highest_weight(1) == (1, 0)
    assert group.highest_weight(2) == (0, 1)
    assert group.dual(1) == 2
    assert group.dual(2) == 1
    assert group.dim(0) == 1
    assert group.dim(1) == 3
    assert group.dim(2) == 3
    assert group.casimir(1) == pytest.approx(4.0 / 3.0)
    assert group.fuse(1, group.fundamental) == (2,)
    assert group.fuse(2, group.fundamental) == (0,)
    assert group.fuse(1, group.antifundamental) == (0,)
    assert group.fuse(2, group.antifundamental) == (1,)


def test_su3_builds_valid_pure_gauge_tables_for_generic_gipeps():
    group = SU3(max_weight_sum=1)
    tables = build_pure_gauge_tables(group, shape=(2, 2))

    assert tables.n_blocks(0, 0) == 3
    assert tables.block_id(0, 0, 0, 0, 0, 0, 0) == 0
    assert tables.block_id(0, 0, 0, 0, 1, 2, 0) >= 0
    assert tables.block_id(0, 0, 0, 0, 2, 1, 0) >= 0


def test_su3_plaquette_table_connects_vacuum_to_fundamental_loop_and_is_hermitian():
    group = SU3(max_weight_sum=1)
    tables = build_pure_gauge_tables(group, shape=(2, 2))
    transitions = build_plaquette_link_transitions(group)
    matrix_table = build_plaquette_matrix_table(group, tables, row=0, col=0)
    vacuum_blocks = (0, 0, 0, 0)

    assert (1, 1, 2, 2) in transitions.outputs(0, 0, 0, 0)
    assert int(matrix_table.counts[vacuum_blocks]) > 0
    found_loop = False
    for out_idx in range(int(matrix_table.counts[vacuum_blocks])):
        output_blocks = matrix_table.output_block_ids[
            matrix_table.flat_index(vacuum_blocks, out_idx)
        ]
        output_links = (
            int(tables.j_r_by_block[0, 0, output_blocks[0]]),
            int(tables.j_d_by_block[0, 1, output_blocks[1]]),
            int(tables.j_r_by_block[1, 0, output_blocks[2]]),
            int(tables.j_d_by_block[0, 0, output_blocks[0]]),
        )
        if output_links == (
            1,
            1,
            2,
            2,
        ):
            found_loop = True
            output_blocks = tuple(int(x) for x in output_blocks)
            reverse = matrix_table.find_outcome(output_blocks, vacuum_blocks)
            assert reverse >= 0
            assert matrix_table.matrix_elements[
                matrix_table.flat_index(vacuum_blocks, out_idx)
            ] == pytest.approx(
                matrix_table.matrix_elements[
                    matrix_table.flat_index(output_blocks, reverse)
                ]
            )
    assert found_loop


def test_su3_generic_gipeps_runs_diagonal_and_plaquette_kernels():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU3(max_weight_sum=1),
            D=2,
            chi=4,
        ),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(*build_link_casimir_terms(model.shape, model.gauge_group), PlaquetteTerm(row=0, col=0)),
    )
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    samples = model.random_physical_configuration(jax.random.PRNGKey(0), n_samples=4)
    tensors = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]

    cache = init_cache(tensors, samples)
    sample_next, _key_next, context = jax.jit(transition)(
        tensors,
        samples[0],
        jax.random.PRNGKey(1),
        jax.tree_util.tree_map(lambda x: x[0], cache),
    )
    _cache_next, estimates = jax.jit(estimate)(tensors, sample_next, context)

    model.active_block_ids(sample_next)
    assert estimates.local_estimate.shape == (1,)
    assert estimates.active_slice_indices.shape == (sum(model.params_per_site),)


def test_su3_2x2_ed_ground_energy_baseline():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU3(max_weight_sum=1),
            D=1,
            chi=1,
        ),
    )
    samples, hamiltonian = _exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=0.7,
        plaquette_coeff=-1.2,
    )
    eigenvalues, _eigenvectors = jnp.linalg.eigh(hamiltonian)

    assert len(samples) == 3
    assert jnp.isclose(eigenvalues[0], -0.04238395386264012)


def test_su3_2x2_imaginary_time_optimization_approaches_ed_energy():
    electric_coeff = 0.7
    plaquette_coeff = -1.2
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(4),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU3(max_weight_sum=1),
            D=2,
            chi=4,
        ),
    )
    samples, hamiltonian = _exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=electric_coeff,
        plaquette_coeff=plaquette_coeff,
    )
    eigenvalues, _eigenvectors = jnp.linalg.eigh(hamiltonian)
    electric_terms = build_link_casimir_terms(model.shape, model.gauge_group)
    driver = TDVPDriver(
        model,
        LocalHamiltonian(
            shape=model.shape,
            terms=(*electric_terms, PlaquetteTerm(row=0, col=0)),
            coeffs=(jnp.asarray(electric_coeff),) * len(electric_terms)
            + (jnp.asarray(plaquette_coeff),),
        ),
        preconditioner=SRPreconditioner(
            strategy=DirectSolve(solver=solve_svd),
            diag_shift=1e-3,
        ),
        dt=0.02,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(1),
        n_samples=512,
        n_chains=64,
        full_gradient=True,
    )
    assert jnp.unique(driver._sampler_configuration, axis=0).shape[0] > 1

    driver.run(60 * driver.dt)

    assert len(samples) == 3
    assert driver.step_count == 60
    assert driver.energy.error_of_mean.real < 5e-4
    assert abs(driver.energy.mean.imag) < 5e-4
    assert abs(driver.energy.mean.real - eigenvalues[0]) < 5e-4

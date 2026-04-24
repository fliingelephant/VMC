import itertools

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from vmc.core import make_mc_sampler
from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.peps.common.contraction import _contract_bottom
from vmc.peps.non_abelian_gi.kernels import _plaquette_candidate_samples
from vmc.operators.local_terms import LocalHamiltonian
from vmc.operators.time_dependent import AffineSchedule, TimeDependentHamiltonian
from vmc.peps.common.strategy import NoTruncation
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_svd
from vmc.peps.standard.kernels import Cache, Context, build_mc_kernels
from vmc.peps.non_abelian_gi import (
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    PlaquetteTerm,
    build_link_casimir_terms,
    build_row_mpo,
)
from vmc.gauge_groups import SU2
from vmc.qgt import (
    Jacobian,
    ParameterSpace,
    QGT,
    SlicedJacobian,
    SiteOrdering,
    SliceOrdering,
)
from vmc.qgt.qgt import _sliced_dense_blocks


def _su2_config(
    *,
    shape: tuple[int, int],
    j_max_twice: int,
    D: int,
    chi: int,
) -> NonAbelianGIPEPSConfig:
    return NonAbelianGIPEPSConfig(
        shape=shape,
        gauge_group=SU2(j_max_twice=j_max_twice),
        D=D,
        chi=chi,
    )


def _single_cache(cache: Cache) -> Cache:
    return jax.tree_util.tree_map(lambda x: x[0], cache)


def _context_for_sample(
    model: NonAbelianGIPEPS,
    tensors: list[list[jax.Array]],
    sample: jax.Array,
) -> Context:
    top_envs = []
    top_env = tuple(
        jnp.ones((1, 1, 1), dtype=jnp.asarray(tensors[0][0]).dtype)
        for _ in range(model.shape[1])
    )
    for row in range(model.shape[0]):
        top_envs.append(top_env)
        top_env = model.strategy.apply(
            top_env,
            build_row_mpo(tensors, sample, model.shape, model.tables, row=row),
        )
    return Context(amp=_contract_bottom(top_env), top_envs=tuple(top_envs))


def _active_blocks(model: NonAbelianGIPEPS, sample: jax.Array) -> tuple[int, ...]:
    return tuple(int(value) for value in model.active_block_ids(sample).reshape(-1))


def _weighted_block_tensors(model: NonAbelianGIPEPS) -> list[list[jax.Array]]:
    return [
        [
            jnp.ones_like(jnp.asarray(tensor))
            * jnp.arange(1, tensor.shape[0] + 1, dtype=jnp.complex128)[
                :, None, None, None, None
            ]
            for tensor in row
        ]
        for row in model.tensors
    ]


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
    links = matrix_table.output_links[input_blocks + (out_idx,)]
    output_iotas = matrix_table.output_iotas[input_blocks + (out_idx,)]
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
    link_irreps = range(model.gauge_group.j_max_twice + 1)
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


def _plaquette_transition_probability(
    model: NonAbelianGIPEPS,
    tensors: list[list[jax.Array]],
    source: jax.Array,
    target: jax.Array,
) -> jax.Array:
    table = model.plaquette_matrix_tables[0][0]
    source_blocks = _active_blocks(model, source)
    target_blocks = _active_blocks(model, target)
    source_weight = jnp.abs(
        model.apply(tensors, source, model.shape, model.tables, model.strategy)
    ) ** 2
    target_weight = jnp.abs(
        model.apply(tensors, target, model.shape, model.tables, model.strategy)
    ) ** 2
    total = jnp.asarray(0.0, dtype=source_weight.dtype)
    for out_idx in range(int(table.counts[source_blocks])):
        candidate = _plaquette_candidate_sample(model, source, out_idx)
        if not bool(jnp.array_equal(candidate, target)):
            continue
        forward_norm = table.proposal_norms[source_blocks]
        forward_prob = table.proposal_weights[source_blocks + (out_idx,)] / forward_norm
        reverse_norm = table.proposal_norms[target_blocks]
        reverse_outputs = table.output_block_ids[target_blocks]
        source_vec = jnp.asarray(source_blocks, dtype=reverse_outputs.dtype)
        reverse_matches = jnp.all(reverse_outputs == source_vec, axis=-1)
        reverse_weight = jnp.sum(
            jnp.where(reverse_matches, table.proposal_weights[target_blocks], 0.0)
        )
        reverse_prob = reverse_weight / reverse_norm
        accept_prob = jnp.minimum(
            1.0,
            target_weight * reverse_prob / (source_weight * forward_prob),
        )
        total = total + forward_prob * accept_prob
    return total


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
        h_links, v_links, _iotas = NonAbelianGIPEPS.unflatten_sample(sample, model.shape)
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
                        * table.matrix_elements[input_blocks + (out_idx,)]
                    )
    return samples, hamiltonian


def _loop_state_tensors_from_amplitudes(
    model: NonAbelianGIPEPS,
    amplitudes: jax.Array,
) -> list[list[jax.Array]]:
    root_amplitudes = jnp.exp(0.25 * jnp.log(amplitudes.astype(jnp.complex128)))
    return [
        [
            jnp.asarray(tensor)
            .at[:, 0, 0, 0, 0]
            .set(root_amplitudes[: tensor.shape[0]])
            for tensor in row
        ]
        for row in model.tensors
    ]


def test_su2_diagonal_only_kernels_estimate_link_casimir_energy():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=build_link_casimir_terms(model.shape, model.gauge_group),
    )
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    h_links = jnp.array([[1], [0]], dtype=jnp.int32)
    v_links = jnp.array([[1, 0]], dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)
    tensors = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    _cache_next, estimates = estimate(tensors, sample_next, context)

    assert jnp.array_equal(sample_next, sample)
    assert jnp.array_equal(estimates.local_estimate, jnp.asarray([1.5]))
    assert estimates.local_log_derivatives.shape == (sum(model.params_per_site),)
    assert estimates.active_slice_indices.shape == (sum(model.params_per_site),)
    assert estimates.amp == context.amp


def test_su2_diagonal_only_kernels_apply_static_coefficients():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(1, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=build_link_casimir_terms(model.shape, model.gauge_group),
        coeffs=(jnp.asarray(2.0),),
    )
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    sample = NonAbelianGIPEPS.flatten_sample(
        jnp.array([[1]], dtype=jnp.int32),
        jnp.zeros((0, 2), dtype=jnp.int32),
        jnp.zeros(model.shape, dtype=jnp.int32),
    )
    tensors = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    _cache_next, estimates = estimate(tensors, sample_next, context)

    assert jnp.array_equal(estimates.local_estimate, jnp.asarray([1.5]))


def test_su2_diagonal_only_kernels_apply_time_dependent_coefficients():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(1, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = TimeDependentHamiltonian(
        base=LocalHamiltonian(
            shape=model.shape,
            terms=build_link_casimir_terms(model.shape, model.gauge_group),
        ),
        schedule=AffineSchedule(offset=jnp.asarray([1.0]), slope=jnp.asarray([1.0])),
    )
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    sample = NonAbelianGIPEPS.flatten_sample(
        jnp.array([[1]], dtype=jnp.int32),
        jnp.zeros((0, 2), dtype=jnp.int32),
        jnp.zeros(model.shape, dtype=jnp.int32),
    )
    tensors = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]

    cache = init_cache(tensors, jnp.stack([sample]), t=2.0)
    sample_next, _key_next, context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    _cache_next, estimates = estimate(tensors, sample_next, context)

    assert jnp.array_equal(estimates.local_estimate, jnp.asarray([2.25]))


def test_su2_kernel_estimates_plaquette_term_from_static_matrix_table():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(PlaquetteTerm(row=0, col=0),),
    )

    _init_cache, _transition, estimate = build_mc_kernels(model, operator)
    sample = model.all_zero_sample()
    tensors = [[jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors]

    context = _context_for_sample(model, tensors, sample)
    _cache_next, estimates = jax.jit(estimate)(tensors, sample, context)

    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    candidate = NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)
    matrix_element = model.plaquette_matrix_tables[0][0].matrix_elements[0, 0, 0, 0, 0]
    expected = (
        matrix_element
        * model.apply(tensors, candidate, model.shape, model.tables, model.strategy)
        / model.apply(tensors, sample, model.shape, model.tables, model.strategy)
    )

    assert jnp.allclose(estimates.local_estimate, jnp.asarray([expected]))


def test_su2_kernel_sums_all_static_plaquette_outcomes():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=2, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(PlaquetteTerm(row=0, col=0),),
    )
    _init_cache, _transition, estimate = build_mc_kernels(model, operator)
    sample = NonAbelianGIPEPS.flatten_sample(
        jnp.array([[1], [1]], dtype=jnp.int32),
        jnp.array([[1, 1]], dtype=jnp.int32),
        jnp.zeros(model.shape, dtype=jnp.int32),
    )
    tensors = [[jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors]

    context = _context_for_sample(model, tensors, sample)
    _cache_next, estimates = estimate(tensors, sample, context)

    input_blocks = tuple(int(value) for value in model.active_block_ids(sample).reshape(-1))
    matrix_table = model.plaquette_matrix_tables[0][0]
    amp = model.apply(tensors, sample, model.shape, model.tables, model.strategy)
    expected = jnp.asarray(0.0, dtype=amp.dtype)
    for out_idx in range(int(matrix_table.counts[input_blocks])):
        candidate = _plaquette_candidate_sample(model, sample, out_idx)
        expected = expected + matrix_table.matrix_elements[input_blocks + (out_idx,)] * (
            model.apply(tensors, candidate, model.shape, model.tables, model.strategy)
            / amp
        )

    assert int(matrix_table.counts[input_blocks]) == 2
    assert jnp.allclose(estimates.local_estimate, jnp.asarray([expected]))


def test_su2_transition_sweeps_static_plaquette_proposals():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(PlaquetteTerm(row=0, col=0),),
    )
    init_cache, transition, _estimate = build_mc_kernels(model, operator)
    sample = model.all_zero_sample()
    tensors = [[jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors]

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, context = jax.jit(transition)(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    expected = _plaquette_candidate_sample(model, sample, out_idx=0)

    assert jnp.array_equal(sample_next, expected)
    assert jnp.allclose(
        context.amp,
        model.apply(tensors, expected, model.shape, model.tables, model.strategy),
    )


def test_su2_plaquette_term_is_noop_when_truncation_has_no_output():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=0, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(PlaquetteTerm(row=0, col=0),),
    )
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    sample = model.all_zero_sample()
    tensors = [[jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors]

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    _cache_next, estimates = estimate(tensors, sample_next, context)

    assert model.plaquette_matrix_tables[0][0].max_outputs == 0
    assert jnp.array_equal(sample_next, sample)
    assert jnp.array_equal(estimates.local_estimate, jnp.asarray([0.0]))


def test_su2_sliced_gradients_reconstruct_full_gradients():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(shape=model.shape, terms=())
    sample = _plaquette_candidate_sample(model, model.all_zero_sample(), out_idx=0)
    tensors = [[jnp.asarray(tensor) for tensor in row] for row in model.tensors]
    context = _context_for_sample(model, tensors, sample)

    _init_cache, _transition, estimate_full = build_mc_kernels(
        model,
        operator,
        full_gradient=True,
    )
    _cache_full, estimates_full = estimate_full(tensors, sample, context)
    _init_cache, _transition, estimate_sliced = build_mc_kernels(
        model,
        operator,
        full_gradient=False,
    )
    _cache_sliced, estimates_sliced = estimate_sliced(tensors, sample, context)

    jac_slice = SlicedJacobian(
        estimates_sliced.local_log_derivatives[None, :],
        estimates_sliced.active_slice_indices[None, :],
        model.sliced_dims,
        SliceOrdering(),
    )
    dense_sliced = _sliced_dense_blocks(jac_slice)
    perm = []
    total = sum(model.params_per_site)
    site_offset = 0
    for site_idx, n_params in enumerate(model.params_per_site):
        for block_id in range(model.sliced_dims[site_idx]):
            base = block_id * total + site_offset
            perm.extend(range(base, base + n_params))
        site_offset += n_params

    assert estimates_full.active_slice_indices is None
    assert jnp.allclose(
        dense_sliced[:, jnp.asarray(perm)][0],
        estimates_full.local_log_derivatives,
    )


def test_su2_plaquette_transition_satisfies_exact_detailed_balance_on_2x2():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=2, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    tensors = _weighted_block_tensors(model)
    samples = _valid_samples(model)
    weights = [
        jnp.abs(model.apply(tensors, sample, model.shape, model.tables, model.strategy))
        ** 2
        for sample in samples
    ]

    for source_idx, source in enumerate(samples):
        for target_idx, target in enumerate(samples):
            if source_idx == target_idx:
                continue
            lhs = weights[source_idx] * _plaquette_transition_probability(
                model,
                tensors,
                source,
                target,
            )
            rhs = weights[target_idx] * _plaquette_transition_probability(
                model,
                tensors,
                target,
                source,
            )
            assert jnp.allclose(lhs, rhs)


def test_su2_plaquette_transition_graph_is_connected_on_2x2():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=2, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    samples = _valid_samples(model)
    sample_keys = {tuple(sample.tolist()): idx for idx, sample in enumerate(samples)}
    neighbors = {idx: set() for idx in range(len(samples))}
    for idx, sample in enumerate(samples):
        table = model.plaquette_matrix_tables[0][0]
        input_blocks = _active_blocks(model, sample)
        for out_idx in range(int(table.counts[input_blocks])):
            candidate = _plaquette_candidate_sample(model, sample, out_idx)
            neighbors[idx].add(sample_keys[tuple(candidate.tolist())])

    visited = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in neighbors[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)

    assert visited == set(range(len(samples)))


def test_su2_sliced_qgt_matches_full_qgt_on_2x2():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(shape=model.shape, terms=())
    tensors = _weighted_block_tensors(model)
    samples = _valid_samples(model)
    _init_cache, _transition, estimate_full = build_mc_kernels(
        model,
        operator,
        full_gradient=True,
    )
    _init_cache, _transition, estimate_sliced = build_mc_kernels(
        model,
        operator,
        full_gradient=False,
    )
    full_grads = []
    sliced_grads = []
    active_slices = []
    for sample in samples:
        context = _context_for_sample(model, tensors, sample)
        _cache, full_estimates = estimate_full(tensors, sample, context)
        _cache, sliced_estimates = estimate_sliced(tensors, sample, context)
        full_grads.append(full_estimates.local_log_derivatives)
        sliced_grads.append(sliced_estimates.local_log_derivatives)
        active_slices.append(sliced_estimates.active_slice_indices)

    full_jacobian = Jacobian(jnp.stack(full_grads))
    sliced_jacobian = SlicedJacobian(
        jnp.stack(sliced_grads),
        jnp.stack(active_slices),
        model.sliced_dims,
        SiteOrdering(model.params_per_site),
    )

    assert jnp.allclose(
        QGT(sliced_jacobian, space=ParameterSpace()).to_dense(),
        QGT(full_jacobian, space=ParameterSpace()).to_dense(),
    )


def test_su2_local_energy_matches_exact_2x2_hamiltonian_matrix():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=2, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    electric_terms = build_link_casimir_terms(model.shape, model.gauge_group)
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(*electric_terms, PlaquetteTerm(row=0, col=0)),
        coeffs=(jnp.asarray(0.7),) * len(electric_terms) + (jnp.asarray(-1.2),),
    )
    tensors = _weighted_block_tensors(model)
    samples, hamiltonian = _exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=0.7,
        plaquette_coeff=-1.2,
    )
    psi = jnp.asarray(
        [
            model.apply(tensors, sample, model.shape, model.tables, model.strategy)
            for sample in samples
        ]
    )

    _init_cache, _transition, estimate = build_mc_kernels(model, operator)
    for source_idx, sample in enumerate(samples):
        context = _context_for_sample(model, tensors, sample)
        _cache, estimates = estimate(tensors, sample, context)
        expected = (hamiltonian @ psi)[source_idx] / psi[source_idx]
        assert jnp.allclose(estimates.local_estimate[0], expected)


def test_su2_2x2_ground_state_local_energy_matches_ed():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=2, D=1, chi=1),
        contraction_strategy=NoTruncation(),
    )
    samples, hamiltonian = _exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=0.7,
        plaquette_coeff=-1.2,
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    tensors = _loop_state_tensors_from_amplitudes(model, ground_state)
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(
            *build_link_casimir_terms(model.shape, model.gauge_group),
            PlaquetteTerm(row=0, col=0),
        ),
        coeffs=(jnp.asarray(0.7),) * 4 + (jnp.asarray(-1.2),),
    )
    _init_cache, _transition, estimate = build_mc_kernels(model, operator)

    for sample in samples:
        context = _context_for_sample(model, tensors, sample)
        _cache, estimates = estimate(tensors, sample, context)
        assert jnp.allclose(estimates.local_estimate[0], ground_energy)


def test_su2_local_energy_matches_exact_3x3_hamiltonian_matrix():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(3, 3), j_max_twice=1, D=1, chi=1),
        contraction_strategy=NoTruncation(),
    )
    electric_terms = build_link_casimir_terms(model.shape, model.gauge_group)
    plaquette_terms = tuple(
        PlaquetteTerm(row=row, col=col)
        for row in range(model.shape[0] - 1)
        for col in range(model.shape[1] - 1)
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(*electric_terms, *plaquette_terms),
        coeffs=(jnp.asarray(0.7),) * len(electric_terms)
        + (jnp.asarray(-1.2),) * len(plaquette_terms),
    )
    tensors = _weighted_block_tensors(model)
    samples, hamiltonian = _exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=0.7,
        plaquette_coeff=-1.2,
    )
    psi = jnp.asarray(
        [
            model.apply(tensors, sample, model.shape, model.tables, model.strategy)
            for sample in samples
        ]
    )

    _init_cache, _transition, estimate = build_mc_kernels(model, operator)
    assert len(samples) == 18
    for source_idx, sample in enumerate(samples):
        context = _context_for_sample(model, tensors, sample)
        _cache, estimates = estimate(tensors, sample, context)
        expected = (hamiltonian @ psi)[source_idx] / psi[source_idx]
        assert jnp.allclose(estimates.local_estimate[0], expected)


def test_su2_3x3_ed_ground_energy_baseline():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(3, 3), j_max_twice=1, D=1, chi=1),
        contraction_strategy=NoTruncation(),
    )
    samples, hamiltonian = _exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=0.7,
        plaquette_coeff=-1.2,
    )
    eigenvalues, _eigenvectors = jnp.linalg.eigh(hamiltonian)

    assert len(samples) == 18
    assert jnp.isclose(eigenvalues[0], -1.8353375761358925)


@pytest.mark.slow
def test_su2_3x3_imaginary_time_optimization_approaches_ed_energy():
    electric_coeff = 0.7
    plaquette_coeff = -1.2
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(3),
        config=_su2_config(shape=(3, 3), j_max_twice=1, D=2, chi=4),
    )
    samples, hamiltonian = _exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=electric_coeff,
        plaquette_coeff=plaquette_coeff,
    )
    eigenvalues, _eigenvectors = jnp.linalg.eigh(hamiltonian)
    electric_terms = build_link_casimir_terms(model.shape, model.gauge_group)
    plaquette_terms = tuple(
        PlaquetteTerm(row=row, col=col)
        for row in range(model.shape[0] - 1)
        for col in range(model.shape[1] - 1)
    )
    driver = TDVPDriver(
        model,
        LocalHamiltonian(
            shape=model.shape,
            terms=(*electric_terms, *plaquette_terms),
            coeffs=(jnp.asarray(electric_coeff),) * len(electric_terms)
            + (jnp.asarray(plaquette_coeff),) * len(plaquette_terms),
        ),
        preconditioner=SRPreconditioner(
            strategy=DirectSolve(solver=solve_svd),
            diag_shift=1e-3,
        ),
        dt=0.02,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(0),
        n_samples=256,
        n_chains=32,
        full_gradient=False,
    )

    driver.run(80 * driver.dt)

    assert len(samples) == 18
    assert driver.step_count == 80
    assert jnp.unique(driver._sampler_configuration, axis=0).shape[0] > 1
    assert driver.energy.error_of_mean.real < 5e-3
    assert abs(driver.energy.mean.real - eigenvalues[0]) < 5e-3


def test_su2_kernels_run_through_generic_mc_sampler():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(*build_link_casimir_terms(model.shape, model.gauge_group), PlaquetteTerm(row=0, col=0)),
    )
    tensors = _weighted_block_tensors(model)
    samples = model.random_physical_configuration(jax.random.PRNGKey(0), n_samples=2)
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    mc_sampler = make_mc_sampler(transition, estimate)
    cache = init_cache(tensors, samples)
    (_samples_next, _keys_next, _cache_next), (sample_history, estimates) = mc_sampler(
        tensors,
        samples,
        jax.random.split(jax.random.PRNGKey(1), 2),
        cache,
        n_steps=2,
    )

    assert sample_history.shape == (2, 2, model.all_zero_sample().size)
    assert estimates.local_estimate.shape == (2, 2, 1)
    for sample in sample_history.reshape((-1, model.all_zero_sample().size)):
        model.active_block_ids(sample)


def test_plaquette_candidate_samples_preserve_gauss_law_for_jmax_half_vacuum():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )

    candidates, valid = _plaquette_candidate_samples(
        model.all_zero_sample(),
        row=0,
        col=0,
        shape=model.shape,
        tables=model.tables,
        link_transitions=model.plaquette_link_transitions,
    )

    h_links = jnp.array([[1], [1]], dtype=jnp.int32)
    v_links = jnp.array([[1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    expected = NonAbelianGIPEPS.flatten_sample(h_links, v_links, iotas)

    assert valid.shape == (1,)
    assert jnp.array_equal(candidates[valid], jnp.stack([expected]))


def test_plaquette_candidate_samples_mask_invalid_intertwiner_combinations():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(3, 3), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )

    candidates, valid = _plaquette_candidate_samples(
        model.all_zero_sample(),
        row=1,
        col=1,
        shape=model.shape,
        tables=model.tables,
        link_transitions=model.plaquette_link_transitions,
    )

    assert candidates.shape == (16, model.all_zero_sample().size)
    assert int(jnp.sum(valid)) == 1
    for candidate in candidates[valid]:
        model.active_block_ids(candidate)

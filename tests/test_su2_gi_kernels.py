from types import FunctionType

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from vmc.core import make_mc_sampler
from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.peps.common.contraction import _contract_bottom
from vmc.peps.common import contraction as common_contraction
from vmc.peps.common import energy as common_energy
from nonabelian_exact import (
    exact_pure_gauge_hamiltonian,
    hopping_outcomes,
    plaquette_outcomes,
    valid_samples,
)
from vmc.peps.non_abelian_gi import kernels as su2_kernels
from vmc.operators.local_terms import LocalHamiltonian
from vmc.operators.time_dependent import AffineSchedule, TimeDependentHamiltonian
from vmc.peps.common.strategy import NoTruncation
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_svd
from vmc.peps.standard.kernels import Cache, Context, build_mc_kernels
from vmc.peps.non_abelian_gi import (
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    HorizontalMatterHoppingTerm,
    PlaquetteTerm,
    build_link_casimir_terms,
    build_matter_number_terms,
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
            build_row_mpo(
                tensors, sample, model.shape, model.tables.block_id_lookup, row=row
            ),
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


def _recursive_closure_ids(fn: FunctionType) -> set[int]:
    ids = set()
    seen_functions = set()

    def visit(value):
        ids.add(id(value))
        if not isinstance(value, FunctionType) or id(value) in seen_functions:
            return
        seen_functions.add(id(value))
        if value.__closure__ is None:
            return
        for cell in value.__closure__:
            visit(cell.cell_contents)

    visit(fn)
    return ids


def test_su2_kernels_do_not_close_model_or_table_objects():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
        contraction_strategy=NoTruncation(),
    )

    kernels = build_mc_kernels(
        model,
        LocalHamiltonian(shape=model.shape, terms=()),
    )
    closure_ids = set().union(*(_recursive_closure_ids(kernel) for kernel in kernels))

    assert id(model) not in closure_ids
    assert id(model.tables) not in closure_ids


def test_common_contract_1row_1col_matches_one_site_environment_dot():
    key = jax.random.PRNGKey(0)
    left_env = jax.random.normal(key, (2, 3, 2), dtype=jnp.float64)
    top = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 2), dtype=jnp.float64)
    mpo = jax.random.normal(jax.random.PRNGKey(2), (3, 4, 5, 6), dtype=jnp.float64)
    bottom = jax.random.normal(jax.random.PRNGKey(3), (2, 6, 2), dtype=jnp.float64)
    right_env = jax.random.normal(jax.random.PRNGKey(4), (2, 4, 2), dtype=jnp.float64)

    env_grad = common_energy._compute_single_gradient(
        left_env,
        right_env,
        top,
        bottom,
    )

    assert jnp.allclose(
        common_contraction._contract_1row_1col(
            left_env,
            top,
            mpo,
            bottom,
            right_env,
        ),
        jnp.einsum("cduv,uvcd->", mpo, env_grad),
    )


def _plaquette_transition_probability(
    model: NonAbelianGIPEPS,
    tensors: list[list[jax.Array]],
    source: jax.Array,
    target: jax.Array,
) -> jax.Array:
    source_weight = (
        jnp.abs(model.apply(tensors, source, model.shape, model.tables, model.strategy))
        ** 2
    )
    target_weight = (
        jnp.abs(model.apply(tensors, target, model.shape, model.tables, model.strategy))
        ** 2
    )
    outcomes = plaquette_outcomes(model, source)
    z_in = sum(abs(me) ** 2 for _, me in outcomes)
    forward_prob = (
        sum(
            abs(me) ** 2
            for candidate, me in outcomes
            if bool(jnp.array_equal(candidate, target))
        )
        / z_in
    )
    if forward_prob == 0.0:
        return jnp.asarray(0.0, dtype=source_weight.dtype)
    z_out = sum(abs(me) ** 2 for _, me in plaquette_outcomes(model, target))
    accept_prob = jnp.minimum(
        1.0,
        (target_weight / source_weight) * (z_in / z_out),
    )
    return forward_prob * accept_prob


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


def test_su2_matter_number_kernel_uses_sampled_matter_field():
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
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=build_matter_number_terms(model.shape, model.matter_numbers),
    )
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    sample = model.random_physical_configuration(jax.random.PRNGKey(0), n_samples=1)[0]
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(1),
        _single_cache(cache),
    )
    _cache_next, estimates = estimate(tensors, sample_next, context)

    assert jnp.array_equal(sample_next, sample)
    assert jnp.array_equal(estimates.local_estimate, jnp.asarray([2.0]))
    assert estimates.active_slice_indices.shape == (sum(model.params_per_site),)


def test_su2_transition_sweeps_plaquettes_before_matter_bonds(monkeypatch):
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=build_matter_number_terms(model.shape, model.matter_numbers),
    )
    calls = []

    def plaquette_sweep(
        key,
        tensors,
        h_links,
        v_links,
        iotas,
        active_block_ids,
        row_mpo0,
        row_mpo1,
        *args,
        row,
        **kwargs,
    ):
        del tensors, args, kwargs
        calls.append(("plaquette", row))
        return key, h_links, v_links, iotas, active_block_ids, row_mpo0, row_mpo1, None

    def horizontal_sweep(
        key,
        tensors,
        matter,
        h_links,
        iotas,
        active_block_ids,
        row_mpo,
        *args,
        row,
        **kwargs,
    ):
        del tensors, args, kwargs
        calls.append(("horizontal", row))
        return key, matter, h_links, iotas, active_block_ids, row_mpo, None

    def vertical_sweep(
        key,
        tensors,
        matter,
        v_links,
        iotas,
        active_block_ids,
        row_mpo0,
        row_mpo1,
        *args,
        row,
        **kwargs,
    ):
        del tensors, args, kwargs
        calls.append(("vertical", row))
        return key, matter, v_links, iotas, active_block_ids, row_mpo0, row_mpo1, None

    monkeypatch.setattr(
        "vmc.peps.non_abelian_gi.kernels._plaquette_sweep_row_pair",
        plaquette_sweep,
    )
    monkeypatch.setattr(
        "vmc.peps.non_abelian_gi.kernels._horizontal_hopping_sweep_row",
        horizontal_sweep,
    )
    monkeypatch.setattr(
        "vmc.peps.non_abelian_gi.kernels._vertical_hopping_sweep_row_pair",
        vertical_sweep,
    )
    init_cache, transition, _estimate = build_mc_kernels(model, operator)
    sample = model.all_zero_sample()
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]

    cache = init_cache(tensors, jnp.stack([sample]))
    transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )

    assert calls == [
        ("plaquette", 0),
        ("horizontal", 0),
        ("horizontal", 1),
        ("vertical", 0),
    ]


def test_su2_horizontal_matter_hopping_kernel_uses_sparse_connected_table():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(1, 3),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(HorizontalMatterHoppingTerm(row=0, col=1),),
    )
    init_cache, transition, estimate = build_mc_kernels(model, operator)
    matter = jnp.asarray([[1, 1, 0]], dtype=jnp.int32)
    h_links = jnp.asarray([[1, 0]], dtype=jnp.int32)
    v_links = jnp.zeros((0, 3), dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_matter_sample(matter, h_links, v_links, iotas)
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]
    outcomes = hopping_outcomes(model, sample, row=0, col=1, horizontal=True)
    assert len(outcomes) == 1
    candidate, matrix_element = outcomes[0]

    context = _context_for_sample(model, tensors, sample)
    _cache_next, estimates = estimate(tensors, sample, context)
    expected = matrix_element * (
        model.apply(tensors, candidate, model.shape, model.tables, model.strategy)
        / model.apply(tensors, sample, model.shape, model.tables, model.strategy)
    )

    assert jnp.allclose(estimates.local_estimate, jnp.asarray([expected]))

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, transition_context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    assert jnp.array_equal(sample_next, candidate)
    assert jnp.allclose(
        transition_context.amp,
        model.apply(tensors, candidate, model.shape, model.tables, model.strategy),
    )


def test_su2_estimate_reuses_dr1_right_envs_for_gradients_and_terms(monkeypatch):
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(1, 3),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(
        shape=model.shape,
        terms=(HorizontalMatterHoppingTerm(row=0, col=1),),
    )
    _init_cache, _transition, estimate = build_mc_kernels(model, operator)
    matter = jnp.asarray([[1, 1, 0]], dtype=jnp.int32)
    h_links = jnp.asarray([[1, 0]], dtype=jnp.int32)
    v_links = jnp.zeros((0, 3), dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_matter_sample(matter, h_links, v_links, iotas)
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]
    context = _context_for_sample(model, tensors, sample)
    calls = 0
    original_common = common_energy._compute_right_envs
    original_su2 = su2_kernels._compute_right_envs

    def count_common(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_common(*args, **kwargs)

    def count_su2(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_su2(*args, **kwargs)

    monkeypatch.setattr(common_energy, "_compute_right_envs", count_common)
    monkeypatch.setattr(su2_kernels, "_compute_right_envs", count_su2)

    estimate(tensors, sample, context)

    assert calls == 1


def test_su2_iota_heatbath_uses_one_site_environment(monkeypatch):
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 3),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
        contraction_strategy=NoTruncation(),
    )
    matter = jnp.asarray([[1, 1, 0], [0, 0, 0]], dtype=jnp.int32)
    h_links = jnp.asarray([[1, 1], [0, 1]], dtype=jnp.int32)
    v_links = jnp.asarray([[0, 1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_matter_sample(matter, h_links, v_links, iotas)
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]
    init_cache, transition, _estimate = build_mc_kernels(
        model,
        LocalHamiltonian(shape=model.shape, terms=()),
    )
    scalar_calls = 0
    gradient_calls = 0
    original_scalar = su2_kernels._contract_1row_1col
    original_gradient = su2_kernels._compute_single_gradient

    def count_scalar(*args, **kwargs):
        nonlocal scalar_calls
        scalar_calls += 1
        return original_scalar(*args, **kwargs)

    def count_gradient(*args, **kwargs):
        nonlocal gradient_calls
        gradient_calls += 1
        return original_gradient(*args, **kwargs)

    def skip_horizontal(
        key,
        tensors,
        matter,
        h_links,
        iotas,
        active_block_ids,
        row_mpo,
        *args,
        **kwargs,
    ):
        del tensors, args, kwargs
        return key, matter, h_links, iotas, active_block_ids, row_mpo, None

    monkeypatch.setattr(su2_kernels, "_contract_1row_1col", count_scalar)
    monkeypatch.setattr(su2_kernels, "_compute_single_gradient", count_gradient)
    monkeypatch.setattr(
        su2_kernels,
        "_horizontal_hopping_sweep_row",
        skip_horizontal,
    )

    transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(init_cache(tensors, jnp.stack([sample]))),
    )

    assert scalar_calls == 0
    assert gradient_calls == model.shape[0] * model.shape[1]


def test_su2_horizontal_transition_carries_current_amplitude(monkeypatch):
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(1, 3),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
        contraction_strategy=NoTruncation(),
    )
    init_cache, transition, _estimate = build_mc_kernels(
        model,
        LocalHamiltonian(shape=model.shape, terms=()),
    )
    matter = jnp.asarray([[1, 1, 0]], dtype=jnp.int32)
    h_links = jnp.asarray([[1, 0]], dtype=jnp.int32)
    v_links = jnp.zeros((0, 3), dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_matter_sample(matter, h_links, v_links, iotas)
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]
    two_col_calls = 0
    original = su2_kernels._contract_1row_2col

    def count_two_col(*args, **kwargs):
        nonlocal two_col_calls
        two_col_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(su2_kernels, "_contract_1row_2col", count_two_col)

    transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(init_cache(tensors, jnp.stack([sample]))),
    )

    assert two_col_calls == 1


def test_su2_kernel_estimates_plaquette_term_from_factor_tables():
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
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]

    context = _context_for_sample(model, tensors, sample)
    _cache_next, estimates = jax.jit(estimate)(tensors, sample, context)

    amp = model.apply(tensors, sample, model.shape, model.tables, model.strategy)
    outcomes = plaquette_outcomes(model, sample)
    assert len(outcomes) == 2  # the vacuum -> loop flip, once per orientation
    expected = (
        sum(
            me
            * model.apply(tensors, candidate, model.shape, model.tables, model.strategy)
            for candidate, me in outcomes
        )
        / amp
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
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]

    context = _context_for_sample(model, tensors, sample)
    _cache_next, estimates = estimate(tensors, sample, context)

    amp = model.apply(tensors, sample, model.shape, model.tables, model.strategy)
    outcomes = plaquette_outcomes(model, sample)
    expected = jnp.asarray(0.0, dtype=amp.dtype)
    for candidate, me in outcomes:
        expected = expected + me * (
            model.apply(tensors, candidate, model.shape, model.tables, model.strategy)
            / amp
        )

    targets = {tuple(candidate.tolist()) for candidate, _ in outcomes}
    assert len(targets) == 2  # lower to vacuum or raise to the j=1 loop
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
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, context = jax.jit(transition)(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    expected = plaquette_outcomes(model, sample)[0][0]

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
    tensors = [
        [jnp.ones_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]

    cache = init_cache(tensors, jnp.stack([sample]))
    sample_next, _key_next, context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        _single_cache(cache),
    )
    _cache_next, estimates = estimate(tensors, sample_next, context)

    tabs = model.plaquette_factor_tables
    corner_tables = (tabs.tl[0][0], tabs.tr[0][1], tabs.bl[1][0], tabs.br[1][1])
    assert max(t.max_candidates for pair in corner_tables for t in pair) == 0
    assert jnp.array_equal(sample_next, sample)
    assert jnp.array_equal(estimates.local_estimate, jnp.asarray([0.0]))


def test_su2_sliced_gradients_reconstruct_full_gradients():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        contraction_strategy=NoTruncation(),
    )
    operator = LocalHamiltonian(shape=model.shape, terms=())
    sample = plaquette_outcomes(model, model.all_zero_sample())[0][0]
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
    samples = valid_samples(model)
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
    samples = valid_samples(model)
    sample_keys = {tuple(sample.tolist()): idx for idx, sample in enumerate(samples)}
    neighbors = {idx: set() for idx in range(len(samples))}
    for idx, sample in enumerate(samples):
        for candidate, _me in plaquette_outcomes(model, sample):
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
    samples = valid_samples(model)
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
    samples, hamiltonian = exact_pure_gauge_hamiltonian(
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
    samples, hamiltonian = exact_pure_gauge_hamiltonian(
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
    samples, hamiltonian = exact_pure_gauge_hamiltonian(
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
    samples, hamiltonian = exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=0.7,
        plaquette_coeff=-1.2,
    )
    eigenvalues, _eigenvectors = jnp.linalg.eigh(hamiltonian)

    assert len(samples) == 18
    assert jnp.isclose(eigenvalues[0], -4.8234564111911995)


@pytest.mark.slow
def test_su2_3x3_imaginary_time_optimization_approaches_ed_energy():
    electric_coeff = 0.7
    plaquette_coeff = -1.2
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(3),
        config=_su2_config(shape=(3, 3), j_max_twice=1, D=2, chi=4),
    )
    samples, hamiltonian = exact_pure_gauge_hamiltonian(
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
        terms=(
            *build_link_casimir_terms(model.shape, model.gauge_group),
            PlaquetteTerm(row=0, col=0),
        ),
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


def test_su2_transition_heatbaths_single_site_iota_at_fixed_links():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 3),
            gauge_group=SU2(j_max_twice=1),
            D=2,
            chi=4,
            phys_dim=2,
            matter_irreps=(0, 1),
            matter_numbers=(0, 1),
            particle_number=2,
        ),
        contraction_strategy=NoTruncation(),
    )
    matter = jnp.asarray([[1, 1, 0], [0, 0, 0]], dtype=jnp.int32)
    h_links = jnp.asarray([[1, 1], [0, 1]], dtype=jnp.int32)
    v_links = jnp.asarray([[0, 1, 1]], dtype=jnp.int32)
    iotas = jnp.zeros(model.shape, dtype=jnp.int32)
    sample = NonAbelianGIPEPS.flatten_matter_sample(matter, h_links, v_links, iotas)
    active_blocks = model.active_block_ids(sample)
    iota1_block = model.tables.block_id_lookup[0, 1, 1, 1, 0, 1, 1, 1]
    assert active_blocks[0, 1] != iota1_block

    tensors = [
        [jnp.zeros_like(jnp.asarray(tensor)) for tensor in row] for row in model.tensors
    ]
    for row in range(model.shape[0]):
        for col in range(model.shape[1]):
            block = iota1_block if (row, col) == (0, 1) else active_blocks[row, col]
            tensors[row][col] = (
                tensors[row][col].at[block].set(jnp.ones_like(tensors[row][col][block]))
            )

    init_cache, transition, _estimate = build_mc_kernels(
        model,
        LocalHamiltonian(shape=model.shape, terms=()),
    )
    cache = _single_cache(init_cache(tensors, jnp.stack([sample])))
    sample_next, _key_next, context = transition(
        tensors,
        sample,
        jax.random.PRNGKey(0),
        cache,
    )
    matter_next, h_next, v_next, iotas_next = NonAbelianGIPEPS.unflatten_matter_sample(
        sample_next,
        model.shape,
    )

    assert jnp.array_equal(matter_next, matter)
    assert jnp.array_equal(h_next, h_links)
    assert jnp.array_equal(v_next, v_links)
    assert iotas_next[0, 1] == 1
    assert jnp.count_nonzero(iotas_next) == 1
    assert jnp.abs(context.amp) > 0

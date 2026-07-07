import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
from vmc.gauge_groups import SU3
from vmc.operators.local_terms import LocalHamiltonian
from nonabelian_exact import exact_pure_gauge_hamiltonian, plaquette_outcomes
from vmc.peps.non_abelian_gi import (
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    PlaquetteTerm,
    build_link_casimir_terms,
    build_pure_gauge_tables,
)
from vmc.peps.standard.kernels import build_mc_kernels
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_svd


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


def test_su3_plaquette_connects_vacuum_to_fundamental_loop_and_is_hermitian():
    model = NonAbelianGIPEPS(
        rngs=nnx.Rngs(0),
        config=NonAbelianGIPEPSConfig(
            shape=(2, 2),
            gauge_group=SU3(max_weight_sum=1),
            D=1,
            chi=1,
        ),
    )
    vacuum = model.all_zero_sample()
    loop = NonAbelianGIPEPS.flatten_sample(
        jnp.asarray([[1], [2]], dtype=jnp.int32),
        jnp.asarray([[2, 1]], dtype=jnp.int32),
        jnp.zeros(model.shape, dtype=jnp.int32),
    )

    def elements(source):
        acc = {}
        for candidate, me in plaquette_outcomes(model, source):
            key = tuple(candidate.tolist())
            acc[key] = acc.get(key, 0j) + me
        return acc

    forward = elements(vacuum)[tuple(loop.tolist())]
    assert forward == pytest.approx(1.0)
    assert elements(loop)[tuple(vacuum.tolist())] == pytest.approx(forward.conjugate())


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
        terms=(
            *build_link_casimir_terms(model.shape, model.gauge_group),
            PlaquetteTerm(row=0, col=0),
        ),
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
    samples, hamiltonian = exact_pure_gauge_hamiltonian(
        model,
        electric_coeff=0.7,
        plaquette_coeff=-1.2,
    )
    eigenvalues, _eigenvectors = jnp.linalg.eigh(hamiltonian)

    assert len(samples) == 3
    assert jnp.isclose(eigenvalues[0], -0.8509840232358125)


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
    samples, hamiltonian = exact_pure_gauge_hamiltonian(
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

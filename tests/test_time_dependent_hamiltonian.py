"""Time-dependent Hamiltonian integration tests."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.core import make_mc_sampler
from vmc.drivers import TDVPDriver
from vmc.operators.local_terms import DiagonalOperator, LocalHamiltonian
from vmc.operators.time_dependent import (
    AffineSchedule,
    TimeDependentHamiltonian,
)
from vmc.peps import (
    GIPEPS,
    GIPEPSConfig,
    NoTruncation,
    PEPS,
    BlockadePEPS,
    BlockadePEPSConfig,
    build_mc_kernels,
)
from vmc.peps.gi import GILocalHamiltonian


class _ZeroPreconditioner:
    """Minimal preconditioner stub for driver plumbing tests."""

    def apply(
        self,
        model,
        params,
        samples,
        o,
        p,
        local_energies,
        *,
        grad_factor,
    ):
        _ = (model, samples, o, p, local_energies, grad_factor)
        return jax.tree_util.tree_map(jnp.zeros_like, params), {}


def _diag_one_hamiltonian(shape: tuple[int, int]) -> LocalHamiltonian:
    return LocalHamiltonian(
        shape=shape,
        terms=(
            DiagonalOperator(
                sites=((0, 0),),
                diag=jnp.asarray([1.0, 1.0], dtype=jnp.complex128),
            ),
        ),
    )


def _diag_one_gi_hamiltonian(
    shape: tuple[int, int],
    value: complex | float,
) -> GILocalHamiltonian:
    return GILocalHamiltonian(
        shape=shape,
        terms=(
            DiagonalOperator(
                sites=((0, 0),),
                diag=jnp.asarray([value], dtype=jnp.complex128),
            ),
        ),
    )


class TimeDependentHamiltonianTest(unittest.TestCase):
    def test_standard_kernel_scales_local_energy(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((1, 1)),
            schedule=AffineSchedule(
                offset=jnp.asarray([2.5], dtype=jnp.float64),
                slope=jnp.asarray([0.0], dtype=jnp.float64),
            ),
        )
        init_cache, transition, estimate = build_mc_kernels(
            model,
            operator,
            full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)
        cache = init_cache(tensors, config_states, t=0.0)
        chain_keys = jax.random.split(jax.random.key(1), 1)
        mc_sampler = make_mc_sampler(transition, estimate)
        (_, _, _), (_, estimates) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=1,
        )
        self.assertAlmostEqual(
            float(estimates.local_estimate[0, 0, 0].real),
            2.5,
            places=12,
        )

    def test_standard_time_dependent_requires_explicit_time(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((1, 1)),
            schedule=AffineSchedule(offset=2.5, slope=0.0),
        )
        init_cache, _, _ = build_mc_kernels(
            model,
            operator,
            full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "non-None time"):
            init_cache(tensors, config_states)

    def test_tdvp_uses_time_dependent_coeffs(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((1, 1)),
            schedule=AffineSchedule(
                offset=jnp.asarray([1.0], dtype=jnp.float64),
                slope=jnp.asarray([3.0], dtype=jnp.float64),
            ),
        )
        driver = TDVPDriver(
            model,
            operator,
            preconditioner=_ZeroPreconditioner(),
            dt=0.1,
            n_samples=1,
            n_chains=1,
        )
        params = driver._tensors
        key = driver._sampler_key
        config_states = driver._sampler_configuration.reshape(driver.n_chains, -1)
        _, carry_next, (local_0, _) = driver._time_derivative(
            params,
            0.0,
            (key, config_states),
        )
        _, _, (local_2, _) = driver._time_derivative(
            params,
            2.0,
            carry_next,
        )
        self.assertAlmostEqual(float(jnp.mean(local_0[:, 0]).real), 1.0, places=12)
        self.assertAlmostEqual(float(jnp.mean(local_2[:, 0]).real), 7.0, places=12)

    def test_scalar_schedule_works(self) -> None:
        """AffineSchedule with scalar offset/slope should work for a 1-term Hamiltonian."""
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((1, 1)),
            schedule=AffineSchedule(offset=2.5, slope=0.0),
        )
        init_cache, transition, estimate = build_mc_kernels(
            model, operator, full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)
        cache = init_cache(tensors, config_states, t=0.0)
        chain_keys = jax.random.split(jax.random.key(1), 1)
        mc_sampler = make_mc_sampler(transition, estimate)
        (_, _, _), (_, estimates) = mc_sampler(
            tensors, config_states, chain_keys, cache, n_steps=1,
        )
        self.assertAlmostEqual(
            float(estimates.local_estimate[0, 0, 0].real), 2.5, places=12,
        )

    def test_gi_time_dependent_scales_local_energy(self) -> None:
        shape = (2, 2)
        model = GIPEPS(
            rngs=nnx.Rngs(0),
            config=GIPEPSConfig(
                shape=shape,
                N=2,
                phys_dim=1,
                Qx=0,
                degeneracy_per_charge=(1, 1),
                charge_of_site=(0,),
            ),
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_gi_hamiltonian(shape, 1.0),
            schedule=AffineSchedule(
                offset=jnp.asarray([2.5], dtype=jnp.float64),
                slope=jnp.asarray([0.0], dtype=jnp.float64),
            ),
        )
        init_cache, transition, estimate = build_mc_kernels(
            model,
            operator,
            full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = model.random_physical_configuration(jax.random.key(3), n_samples=1)
        cache = init_cache(tensors, config_states, t=0.0)
        chain_keys = jax.random.split(jax.random.key(1), 1)
        mc_sampler = make_mc_sampler(transition, estimate)
        (_, _, _), (_, estimates) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=1,
        )
        self.assertAlmostEqual(
            float(estimates.local_estimate[0, 0, 0].real),
            2.5,
            places=12,
        )

    def test_gi_multi_operator_matches_individual_with_time_dependent_observable(self) -> None:
        shape = (2, 2)
        model = GIPEPS(
            rngs=nnx.Rngs(0),
            config=GIPEPSConfig(
                shape=shape,
                N=2,
                phys_dim=1,
                Qx=0,
                degeneracy_per_charge=(1, 1),
                charge_of_site=(0,),
            ),
            contraction_strategy=NoTruncation(),
        )
        hamiltonian = _diag_one_gi_hamiltonian(shape, 1.5)
        observable = TimeDependentHamiltonian(
            base=_diag_one_gi_hamiltonian(shape, 2.0),
            schedule=AffineSchedule(
                offset=jnp.asarray([3.0], dtype=jnp.float64),
                slope=jnp.asarray([0.0], dtype=jnp.float64),
            ),
        )
        init_cache, transition, estimate_merged = build_mc_kernels(
            model,
            hamiltonian,
            observables=(observable,),
            full_gradient=True,
        )
        _, _, estimate_h = build_mc_kernels(model, hamiltonian, full_gradient=True)
        init_cache_obs, _, estimate_obs = build_mc_kernels(
            model,
            observable,
            full_gradient=True,
        )

        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = model.random_physical_configuration(jax.random.key(4), n_samples=1)
        cache = init_cache(tensors, config_states, t=0.0)
        cache_obs = init_cache_obs(tensors, config_states, t=0.0)
        cache_i = jax.tree.map(lambda x: x[0], cache)
        cache_obs_i = jax.tree.map(lambda x: x[0], cache_obs)
        sample_next, _, context = transition(
            tensors,
            config_states[0],
            jax.random.key(2),
            cache_i,
        )
        _, merged = estimate_merged(tensors, sample_next, context)
        _, ref_h = estimate_h(tensors, sample_next, context._replace(coeffs=None))
        _, ref_obs = estimate_obs(
            tensors,
            sample_next,
            context._replace(coeffs=cache_obs_i.coeffs),
        )

        self.assertEqual(merged.local_estimate.shape, (2,))
        self.assertAlmostEqual(
            float(jnp.abs(merged.local_estimate[0] - ref_h.local_estimate[0])),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(jnp.abs(merged.local_estimate[1] - ref_obs.local_estimate[0])),
            0.0,
            places=12,
        )

    def test_blockade_time_dependent_scales_local_energy(self) -> None:
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=BlockadePEPSConfig(shape=(1, 1), D0=1, D1=1),
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((1, 1)),
            schedule=AffineSchedule(
                offset=jnp.asarray([2.5], dtype=jnp.float64),
                slope=jnp.asarray([0.0], dtype=jnp.float64),
            ),
        )
        init_cache, transition, estimate = build_mc_kernels(
            model, operator, full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)
        cache = init_cache(tensors, config_states, t=0.0)
        chain_keys = jax.random.split(jax.random.key(1), 1)
        mc_sampler = make_mc_sampler(transition, estimate)
        (_, _, _), (_, estimates) = mc_sampler(
            tensors, config_states, chain_keys, cache, n_steps=1,
        )
        self.assertAlmostEqual(
            float(estimates.local_estimate[0, 0, 0].real), 2.5, places=12,
        )

    def test_blockade_time_dependent_requires_explicit_time(self) -> None:
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=BlockadePEPSConfig(shape=(1, 1), D0=1, D1=1),
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((1, 1)),
            schedule=AffineSchedule(offset=2.5, slope=0.0),
        )
        init_cache, _, _ = build_mc_kernels(
            model,
            operator,
            full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "non-None time"):
            init_cache(tensors, config_states)


if __name__ == "__main__":
    unittest.main()

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
    CubicSchedule,
    TimeDependentHamiltonian,
)
from vmc.peps import (
    GIPEPS,
    GIPEPSConfig,
    NoTruncation,
    PEPS,
    BlockadePEPS,
    BlockadePEPSConfig,
    Variational,
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

    def test_cubic_schedule_works(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        operator = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((1, 1)),
            schedule=CubicSchedule(
                constant=1.0,
                linear=2.0,
                quadratic=3.0,
                cubic=4.0,
            ),
        )
        init_cache, transition, estimate = build_mc_kernels(
            model,
            operator,
            full_gradient=True,
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        config_states = jnp.zeros((1, 1), dtype=jnp.int32)
        cache = init_cache(tensors, config_states, t=2.0)
        chain_keys = jax.random.split(jax.random.key(1), 1)
        mc_sampler = make_mc_sampler(transition, estimate)
        (_, _, _), (_, estimates) = mc_sampler(
            tensors, config_states, chain_keys, cache, n_steps=1,
        )
        self.assertAlmostEqual(
            float(estimates.local_estimate[0, 0, 0].real), 49.0, places=12,
        )

    def test_gi_tdvp_uses_time_dependent_hamiltonian_and_observable(self) -> None:
        shape = (3, 3)
        model = GIPEPS(
            rngs=nnx.Rngs(0),
            config=GIPEPSConfig(
                shape=shape,
                N=2,
                phys_dim=1,
                Qx=0,
                degeneracy_per_charge=(4, 4),
                charge_of_site=(0,),
            ),
            contraction_strategy=Variational(16, n_sweeps=2),
        )
        hamiltonian = TimeDependentHamiltonian(
            base=_diag_one_gi_hamiltonian(shape, 1.0),
            schedule=AffineSchedule(
                offset=jnp.asarray([1.0], dtype=jnp.float64),
                slope=jnp.asarray([3.0], dtype=jnp.float64),
            ),
        )
        observable = TimeDependentHamiltonian(
            base=_diag_one_gi_hamiltonian(shape, 1.0),
            schedule=AffineSchedule(
                offset=jnp.asarray([5.0], dtype=jnp.float64),
                slope=jnp.asarray([-2.0], dtype=jnp.float64),
            ),
        )
        driver = TDVPDriver(
            model,
            hamiltonian,
            observables=(observable,),
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
        self.assertAlmostEqual(float(jnp.mean(local_0[:, 1]).real), 5.0, places=12)
        self.assertAlmostEqual(float(jnp.mean(local_2[:, 1]).real), 1.0, places=12)

    def test_blockade_tdvp_uses_time_dependent_hamiltonian_and_observable(self) -> None:
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=BlockadePEPSConfig(shape=(3, 3), D0=4, D1=4),
            contraction_strategy=Variational(16, n_sweeps=2),
        )
        hamiltonian = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((3, 3)),
            schedule=AffineSchedule(
                offset=jnp.asarray([1.0], dtype=jnp.float64),
                slope=jnp.asarray([3.0], dtype=jnp.float64),
            ),
        )
        observable = TimeDependentHamiltonian(
            base=_diag_one_hamiltonian((3, 3)),
            schedule=AffineSchedule(
                offset=jnp.asarray([5.0], dtype=jnp.float64),
                slope=jnp.asarray([-2.0], dtype=jnp.float64),
            ),
        )
        driver = TDVPDriver(
            model,
            hamiltonian,
            observables=(observable,),
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
        self.assertAlmostEqual(float(jnp.mean(local_0[:, 1]).real), 5.0, places=12)
        self.assertAlmostEqual(float(jnp.mean(local_2[:, 1]).real), 1.0, places=12)

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

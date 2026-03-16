"""TDVP driver kernel caching checks."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers import TDVPDriver
import vmc.drivers.tdvp as tdvp_module
from vmc.gauge import GaugeConfig
from vmc.operators import (
    AffineSchedule,
    DiagonalOperator,
    LocalHamiltonian,
    TimeDependentHamiltonian,
)
from vmc.peps import BlockadePEPS, BlockadePEPSConfig, NoTruncation, PEPS
from vmc.peps.gi import GILocalHamiltonian, GIPEPS, GIPEPSConfig
from vmc.peps.gi.local_terms import build_electric_terms
from vmc.preconditioners import (
    DirectSolve,
    SRPreconditioner,
    solve_cholesky,
    solve_svd,
)


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


def _diag_hamiltonian(shape: tuple[int, int], value: float) -> LocalHamiltonian:
    return LocalHamiltonian(
        shape=shape,
        terms=(
            DiagonalOperator(
                sites=((0, 0),),
                diag=jnp.asarray([value, value], dtype=jnp.complex128),
            ),
        ),
    )


class TDVPKernelCacheTest(unittest.TestCase):
    def test_gauge_removal_accepts_driver_tensor_dict(self) -> None:
        for full_gradient in (False, True):
            with self.subTest(full_gradient=full_gradient):
                model = PEPS(
                    rngs=nnx.Rngs(0),
                    shape=(1, 1),
                    bond_dim=1,
                    contraction_strategy=NoTruncation(),
                )
                driver = TDVPDriver(
                    model,
                    _diag_hamiltonian((1, 1), 1.0),
                    preconditioner=SRPreconditioner(
                        strategy=DirectSolve(solver=solve_svd),
                        diag_shift=1e-8,
                        gauge_config=GaugeConfig(),
                    ),
                    dt=0.1,
                    n_samples=2,
                    n_chains=2,
                    full_gradient=full_gradient,
                )
                driver.run(driver.dt)
                self.assertEqual(driver.step_count, 1)
                self.assertAlmostEqual(driver.t, 0.1, places=12)

    def test_static_operator_reuses_kernels(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        with patch(
            "vmc.drivers.tdvp.build_mc_kernels",
            wraps=tdvp_module.build_mc_kernels,
        ) as mocked_build:
            driver = TDVPDriver(
                model,
                LocalHamiltonian(shape=(1, 1), terms=()),
                preconditioner=_ZeroPreconditioner(),
                dt=0.1,
                n_samples=1,
                n_chains=1,
            )
            params = driver._tensors
            key = driver._sampler_key
            config_states = driver._sampler_configuration.reshape(driver.n_chains, -1)
            _, (key, config_states), _ = driver._time_derivative(
                params,
                0.0,
                (key, config_states),
            )
            driver._time_derivative(
                params,
                0.0,
                (key, config_states),
            )
            self.assertEqual(mocked_build.call_count, 1)

    def test_run_chunked_k5(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        driver = TDVPDriver(
            model,
            LocalHamiltonian(shape=(1, 1), terms=()),
            preconditioner=_ZeroPreconditioner(),
            dt=0.1,
            n_samples=1,
            n_chains=1,
        )
        k = 5
        for _ in range(2):
            driver.run(k * driver.dt)
        self.assertEqual(driver.step_count, 10)
        self.assertAlmostEqual(driver.t, 1.0, places=12)

    def test_run_records_observable_stats_per_chunk(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        driver = TDVPDriver(
            model,
            _diag_hamiltonian((1, 1), 2.0),
            observables=(_diag_hamiltonian((1, 1), 5.0),),
            preconditioner=_ZeroPreconditioner(),
            dt=0.1,
            n_samples=1,
            n_chains=1,
        )
        driver.run(driver.dt)
        self.assertIsNotNone(driver.energy)
        self.assertEqual(len(driver.observable_stats), 1)
        self.assertAlmostEqual(float(driver.energy.mean.real), 2.0, places=12)
        self.assertAlmostEqual(
            float(driver.observable_stats[0].mean.real), 5.0, places=12,
        )

    def test_rk4_run_logs_first_stage_time_dependent_energy(self) -> None:
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=(1, 1),
            bond_dim=1,
            contraction_strategy=NoTruncation(),
        )
        driver = TDVPDriver(
            model,
            TimeDependentHamiltonian(
                base=_diag_hamiltonian((1, 1), 1.0),
                schedule=AffineSchedule(offset=1.0, slope=3.0),
            ),
            preconditioner=_ZeroPreconditioner(),
            dt=0.1,
            n_samples=1,
            n_chains=1,
        )
        driver.run(driver.dt)
        self.assertIsNotNone(driver.energy)
        self.assertAlmostEqual(float(driver.energy.mean.real), 1.0, places=12)

    def test_gi_fixed_step_sr_runs_multiple_steps_with_sliced_gradients(self) -> None:
        shape = (2, 2)
        model = GIPEPS(
            rngs=nnx.Rngs(0),
            config=GIPEPSConfig(
                shape=shape,
                N=2,
                phys_dim=1,
                Qx=0,
                degeneracy_per_charge=(2, 2),
                charge_of_site=(0,),
            ),
            contraction_strategy=NoTruncation(),
        )
        driver = TDVPDriver(
            model,
            GILocalHamiltonian(
                shape=shape,
                terms=build_electric_terms(shape, coeff=0.1, N=2),
            ),
            preconditioner=SRPreconditioner(
                strategy=DirectSolve(solver=solve_cholesky),
                diag_shift=1e-8,
            ),
            dt=0.1,
            n_samples=4,
            n_chains=2,
            full_gradient=False,
        )
        driver.run(2 * driver.dt)
        self.assertEqual(driver.step_count, 2)
        self.assertAlmostEqual(driver.t, 0.2, places=12)

    def test_blockade_fixed_step_sr_runs_multiple_steps_with_sliced_gradients(self) -> None:
        model = BlockadePEPS(
            rngs=nnx.Rngs(0),
            config=BlockadePEPSConfig(shape=(2, 2), D0=2, D1=2),
            contraction_strategy=NoTruncation(),
        )
        driver = TDVPDriver(
            model,
            LocalHamiltonian(shape=(2, 2), terms=()),
            preconditioner=SRPreconditioner(
                strategy=DirectSolve(solver=solve_cholesky),
                diag_shift=1e-8,
            ),
            dt=0.1,
            n_samples=4,
            n_chains=2,
            full_gradient=False,
        )
        driver.run(2 * driver.dt)
        self.assertEqual(driver.step_count, 2)
        self.assertAlmostEqual(driver.t, 0.2, places=12)

if __name__ == "__main__":
    unittest.main()

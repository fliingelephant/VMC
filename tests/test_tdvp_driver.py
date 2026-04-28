"""TDVP driver kernel caching checks."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver
import vmc.drivers.tdvp as tdvp_module
from vmc.gauge import GaugeConfig
from vmc.operators import (
    AffineSchedule,
    DiagonalOperator,
    LocalHamiltonian,
    TimeDependentHamiltonian,
)
from vmc.gauge_groups import SU2
from vmc.peps import BlockadePEPS, BlockadePEPSConfig, NoTruncation, PEPS
from vmc.peps import (
    NonAbelianGIPEPS,
    NonAbelianGIPEPSConfig,
    PlaquetteTerm,
    build_link_casimir_terms,
)
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


def _su2_2x2_loop_sample(model: NonAbelianGIPEPS, j_twice: int) -> jax.Array:
    return NonAbelianGIPEPS.flatten_sample(
        jnp.asarray([[j_twice], [j_twice]], dtype=jnp.int32),
        jnp.asarray([[j_twice, j_twice]], dtype=jnp.int32),
        jnp.zeros(model.shape, dtype=jnp.int32),
    )


def _su2_2x2_hamiltonian(
    model: NonAbelianGIPEPS,
    *,
    electric_coeff: float,
    plaquette_coeff: float,
) -> jax.Array:
    samples = tuple(_su2_2x2_loop_sample(model, j) for j in model.gauge_group.irreps())
    sample_keys = {tuple(sample.tolist()): idx for idx, sample in enumerate(samples)}
    hamiltonian = jnp.zeros((len(samples), len(samples)), dtype=jnp.complex128)
    table = model.plaquette_matrix_tables[0][0]
    for source_idx, sample in enumerate(samples):
        h_links, v_links, _iotas = NonAbelianGIPEPS.unflatten_sample(sample, model.shape)
        electric = sum(
            electric_coeff * model.gauge_group.casimir(int(link))
            for link in (*h_links.reshape(-1), *v_links.reshape(-1))
        )
        hamiltonian = hamiltonian.at[source_idx, source_idx].set(electric)
        input_blocks = tuple(int(value) for value in model.active_block_ids(sample).reshape(-1))
        for out_idx in range(int(table.counts[input_blocks])):
            output_blocks = table.output_block_ids[
                table.flat_index(input_blocks, out_idx)
            ]
            links = jnp.stack(
                [
                    model.tables.j_r_by_block[0, 0, output_blocks[0]],
                    model.tables.j_d_by_block[0, 1, output_blocks[1]],
                    model.tables.j_r_by_block[1, 0, output_blocks[2]],
                    model.tables.j_d_by_block[0, 0, output_blocks[0]],
                ]
            )
            candidate = NonAbelianGIPEPS.flatten_sample(
                jnp.asarray([[links[0]], [links[2]]], dtype=jnp.int32),
                jnp.asarray([[links[3], links[1]]], dtype=jnp.int32),
                jnp.zeros(model.shape, dtype=jnp.int32),
            )
            target_idx = sample_keys[tuple(candidate.tolist())]
            hamiltonian = hamiltonian.at[target_idx, source_idx].add(
                plaquette_coeff
                * table.matrix_elements[table.flat_index(input_blocks, out_idx)]
            )
    return hamiltonian


def _set_su2_2x2_loop_amplitudes(model: NonAbelianGIPEPS, amplitudes: jax.Array) -> None:
    root_amplitudes = jnp.exp(0.25 * jnp.log(amplitudes.astype(jnp.complex128)))
    for row in range(model.shape[0]):
        for col in range(model.shape[1]):
            tensor = jnp.asarray(model.tensors[row][col])
            model.tensors[row][col][...] = tensor.at[:, 0, 0, 0, 0].set(
                root_amplitudes[: tensor.shape[0]]
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
                terms=build_electric_terms(shape, N=2),
                coeffs=(jnp.asarray(0.1),) * (shape[0] * (shape[1] - 1) + (shape[0] - 1) * shape[1]),
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

    def test_su2_driver_reports_ed_ground_energy_for_exact_2x2_state(self) -> None:
        electric_coeff = 0.7
        plaquette_coeff = -1.2
        model = NonAbelianGIPEPS(
            rngs=nnx.Rngs(0),
            config=_su2_config(shape=(2, 2), j_max_twice=2, D=1, chi=1),
            contraction_strategy=NoTruncation(),
        )
        hamiltonian_matrix = _su2_2x2_hamiltonian(
            model,
            electric_coeff=electric_coeff,
            plaquette_coeff=plaquette_coeff,
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian_matrix)
        _set_su2_2x2_loop_amplitudes(model, eigenvectors[:, 0])
        link_terms = build_link_casimir_terms(model.shape, model.gauge_group)
        driver = TDVPDriver(
            model,
            LocalHamiltonian(
                shape=model.shape,
                terms=(*link_terms, PlaquetteTerm(row=0, col=0)),
                coeffs=(jnp.asarray(electric_coeff),) * len(link_terms)
                + (jnp.asarray(plaquette_coeff),),
            ),
            preconditioner=_ZeroPreconditioner(),
            dt=0.1,
            n_samples=4,
            n_chains=2,
            full_gradient=False,
        )
        driver.run(driver.dt)

        self.assertEqual(driver.step_count, 1)
        self.assertAlmostEqual(
            float(driver.energy.mean.real),
            float(eigenvalues[0]),
            places=12,
        )

    def test_su2_imaginary_time_optimization_approaches_2x2_ed_energy(self) -> None:
        electric_coeff = 0.7
        plaquette_coeff = -1.2
        model = NonAbelianGIPEPS(
            rngs=nnx.Rngs(2),
            config=_su2_config(shape=(2, 2), j_max_twice=1, D=2, chi=4),
        )
        eigenvalues, _eigenvectors = jnp.linalg.eigh(
            _su2_2x2_hamiltonian(
                model,
                electric_coeff=electric_coeff,
                plaquette_coeff=plaquette_coeff,
            )
        )
        link_terms = build_link_casimir_terms(model.shape, model.gauge_group)
        driver = TDVPDriver(
            model,
            LocalHamiltonian(
                shape=model.shape,
                terms=(*link_terms, PlaquetteTerm(row=0, col=0)),
                coeffs=(jnp.asarray(electric_coeff),) * len(link_terms)
                + (jnp.asarray(plaquette_coeff),),
            ),
            preconditioner=SRPreconditioner(
                strategy=DirectSolve(solver=solve_svd),
                diag_shift=1e-3,
            ),
            dt=0.03,
            time_unit=ImaginaryTimeUnit(),
            sampler_key=jax.random.key(0),
            n_samples=128,
            n_chains=16,
            full_gradient=True,
        )
        self.assertGreater(
            int(jnp.unique(driver._sampler_configuration, axis=0).shape[0]),
            1,
        )

        driver.run(20 * driver.dt)

        self.assertEqual(driver.step_count, 20)
        self.assertGreater(
            int(jnp.unique(driver._sampler_configuration, axis=0).shape[0]),
            1,
        )
        self.assertLess(float(driver.energy.error_of_mean.real), 2e-3)
        self.assertLess(
            abs(float(driver.energy.mean.real) - float(eigenvalues[0])),
            2e-3,
        )

if __name__ == "__main__":
    unittest.main()

"""Full-sum benchmarks for MPS and PEPS samplers."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx
from netket import stats as nkstats

from vmc.core import _value
from vmc.models.mps import MPS
from vmc.models.peps import PEPS
from vmc.operators.local_terms import (
    HorizontalTwoSiteTerm,
    LocalHamiltonian,
    VerticalTwoSiteTerm,
)
from vmc.samplers.sequential import sequential_sample
from vmc.utils.vmc_utils import local_estimate


class FullSumBenchmarkTest(unittest.TestCase):
    SEED = 0
    N_CHAINS = 8
    BURN_IN = 10
    ENERGY_SIGMA_MULT = 5.0
    N_SAMPLES_MPS = 65536
    N_SAMPLES_PEPS = 16384

    def _assert_close(self, approx, exact, *, label: str):
        err = float(approx.error_of_mean) + float(exact.error_of_mean)
        diff = abs(complex(approx.mean) - complex(exact.mean))
        self.assertLess(diff, self.ENERGY_SIGMA_MULT * (err + 1e-12), msg=f"{label} diff={diff} err={err}")

    def test_mps_fullsum_matches(self):
        n_sites = 14
        bond_dim = 4
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        hamiltonian = nk.operator.Heisenberg(hi, nk.graph.Chain(length=n_sites), dtype=jnp.complex128)
        model = MPS(rngs=nnx.Rngs(self.SEED), n_sites=hi.size, bond_dim=bond_dim)

        sampler = nk.sampler.MetropolisLocal(hi, n_chains=self.N_CHAINS, sweep_size=int(hi.size), reset_chains=False)
        mc_state = nk.vqs.MCState(
            sampler, model, n_samples=self.N_SAMPLES_MPS,
            n_discard_per_chain=self.BURN_IN, seed=self.SEED,
        )
        fs_state = nk.vqs.FullSumState(hi, model)

        samples = sequential_sample(
            model,
            n_samples=self.N_SAMPLES_MPS,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            key=jax.random.key(self.SEED),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(self.SEED + 1), n_samples=self.N_CHAINS
            ),
        )
        amps = _value(model, samples)
        local_energies = local_estimate(model, samples, hamiltonian, amps)
        chain_length = self.N_SAMPLES_MPS // self.N_CHAINS
        ours_stats = nkstats.statistics(local_energies.reshape(chain_length, self.N_CHAINS).T)
        mc_stats = mc_state.expect(hamiltonian)
        fs_stats = fs_state.expect(hamiltonian)

        self._assert_close(ours_stats, fs_stats, label="mps ours")
        self._assert_close(mc_stats, fs_stats, label="mps netket")

    def test_peps_fullsum_matches(self):
        shape = (3, 4)
        bond_dim = 3
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        hamiltonian = nk.operator.Heisenberg(hi, nk.graph.Grid(extent=shape, pbc=False), dtype=jnp.complex128)
        sz_sz = jnp.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=jnp.complex128,
        )
        exchange = jnp.array(
            [
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=jnp.complex128,
        )
        bond_op = sz_sz - exchange
        horizontal_terms = []
        vertical_terms = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                if c + 1 < shape[1]:
                    horizontal_terms.append(HorizontalTwoSiteTerm(r, c, bond_op))
                if r + 1 < shape[0]:
                    vertical_terms.append(VerticalTwoSiteTerm(r, c, bond_op))
        local_operator = LocalHamiltonian(
            shape=shape,
            terms=tuple(horizontal_terms + vertical_terms),
        )
        model = PEPS(rngs=nnx.Rngs(self.SEED), shape=shape, bond_dim=bond_dim)

        sampler = nk.sampler.MetropolisLocal(hi, n_chains=self.N_CHAINS, sweep_size=int(hi.size), reset_chains=False)
        mc_state = nk.vqs.MCState(
            sampler, model, n_samples=self.N_SAMPLES_PEPS,
            n_discard_per_chain=self.BURN_IN, seed=self.SEED,
        )
        fs_state = nk.vqs.FullSumState(hi, model)

        samples = sequential_sample(
            model,
            n_samples=self.N_SAMPLES_PEPS,
            n_chains=self.N_CHAINS,
            burn_in=self.BURN_IN,
            key=jax.random.key(self.SEED),
            initial_configuration=model.random_physical_configuration(
                jax.random.key(self.SEED + 1), n_samples=self.N_CHAINS
            ),
        )
        amps = _value(model, samples)
        local_energies = local_estimate(model, samples, local_operator, amps)
        chain_length = self.N_SAMPLES_PEPS // self.N_CHAINS
        ours_stats = nkstats.statistics(local_energies.reshape(chain_length, self.N_CHAINS).T)
        mc_stats = mc_state.expect(hamiltonian)
        fs_stats = fs_state.expect(hamiltonian)

        self._assert_close(ours_stats, fs_stats, label="peps ours")
        self._assert_close(mc_stats, fs_stats, label="peps netket")


if __name__ == "__main__":
    unittest.main()

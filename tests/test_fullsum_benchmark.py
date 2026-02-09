"""Full-sum benchmark for PEPS sampler/energy path."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx
from netket import stats as nkstats

from vmc.core import _sample_counts, _trim_samples, make_mc_sampler
from vmc.operators.local_terms import (
    HorizontalTwoSiteTerm,
    LocalHamiltonian,
    VerticalTwoSiteTerm,
)
from vmc.peps import PEPS, build_mc_kernels
from vmc.peps.standard.compat import _value, local_estimate


class FullSumBenchmarkTest(unittest.TestCase):
    SEED = 0
    N_CHAINS = 4
    BURN_IN = 5
    ENERGY_SIGMA_MULT = 5.0
    N_SAMPLES_PEPS = 2048

    def _assert_close(self, approx, exact, *, label: str):
        err = float(approx.error_of_mean) + float(exact.error_of_mean)
        diff = abs(complex(approx.mean) - complex(exact.mean))
        self.assertLess(
            diff,
            self.ENERGY_SIGMA_MULT * (err + 1e-12),
            msg=f"{label} diff={diff} err={err}",
        )

    def test_peps_fullsum_matches(self):
        shape = (3, 4)
        bond_dim = 3
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        hamiltonian = nk.operator.Heisenberg(
            hi,
            nk.graph.Grid(extent=shape, pbc=False),
            dtype=jnp.complex128,
        )
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

        sampler = nk.sampler.MetropolisLocal(
            hi,
            n_chains=self.N_CHAINS,
            sweep_size=int(hi.size),
            reset_chains=False,
        )
        mc_state = nk.vqs.MCState(
            sampler,
            model,
            n_samples=self.N_SAMPLES_PEPS,
            n_discard_per_chain=self.BURN_IN,
            seed=self.SEED,
        )
        fs_state = nk.vqs.FullSumState(hi, model)

        n_samples, num_chains, chain_length, total_samples = _sample_counts(
            self.N_SAMPLES_PEPS,
            self.N_CHAINS,
        )
        key = jax.random.key(self.SEED)
        key, init_key = jax.random.split(key)
        config_states = model.random_physical_configuration(
            init_key, n_samples=num_chains
        ).reshape(num_chains, -1)
        chain_keys = jax.random.split(key, num_chains)
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
        init_cache, transition, estimate = build_mc_kernels(
            model,
            local_operator,
            full_gradient=False,
        )
        cache = init_cache(tensors, config_states)
        mc_sampler = make_mc_sampler(transition, estimate)
        (_, _, _), (samples_hist, _) = mc_sampler(
            tensors,
            config_states,
            chain_keys,
            cache,
            n_steps=chain_length,
        )
        samples = _trim_samples(samples_hist, total_samples, n_samples)
        amps = _value(model, samples)
        local_energies = local_estimate(model, samples, local_operator, amps)
        ours_stats = nkstats.statistics(
            local_energies.reshape(chain_length, self.N_CHAINS).T
        )
        mc_stats = mc_state.expect(hamiltonian)
        fs_stats = fs_state.expect(hamiltonian)

        self._assert_close(ours_stats, fs_stats, label="peps ours")
        self._assert_close(mc_stats, fs_stats, label="peps netket")


if __name__ == "__main__":
    unittest.main()

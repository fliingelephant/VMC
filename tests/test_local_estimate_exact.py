"""Exact local energy checks for PEPS."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.operators.local_terms import (
    HorizontalTwoSiteOperator,
    LocalHamiltonian,
    VerticalTwoSiteOperator,
)
from vmc.peps import NoTruncation, PEPS, build_mc_kernels
from vmc.peps.standard.compat import _value
from vmc.utils.utils import spin_to_occupancy
from vmc.utils.vmc_utils import local_estimate


class LocalEstimateExactTest(unittest.TestCase):
    def test_peps_local_estimate_matches_dense(self) -> None:
        shape = (2, 3)
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Grid(extent=shape, pbc=False)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
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
                    horizontal_terms.append(HorizontalTwoSiteOperator(r, c, bond_op))
                if r + 1 < shape[0]:
                    vertical_terms.append(VerticalTwoSiteOperator(r, c, bond_op))
        local_operator = LocalHamiltonian(
            shape=shape,
            terms=tuple(horizontal_terms + vertical_terms),
        )
        model = PEPS(
            rngs=nnx.Rngs(1),
            shape=shape,
            bond_dim=3,
            contraction_strategy=NoTruncation(),
        )

        samples_spin = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        samples = spin_to_occupancy(samples_spin)
        amps = _value(model, samples)
        local = local_estimate(model, samples, local_operator, amps)
        h_dense = jnp.asarray(hamiltonian.to_dense(), dtype=amps.dtype)
        expected = (h_dense @ amps) / amps

        mask = jnp.abs(amps) > 1e-12
        max_diff = jnp.max(jnp.abs(local[mask] - expected[mask]))
        self.assertLess(float(max_diff), 1e-9)

    def test_peps_local_estimate_operator_types_match(self) -> None:
        """LocalHamiltonian and NetKet operators match for PEPS."""
        shape = (2, 2)
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Grid(extent=shape, pbc=False)
        netket_hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
        )

        sz_sz = jnp.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=jnp.complex128,
        )
        exchange = jnp.array(
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
            dtype=jnp.complex128,
        )
        bond_op = sz_sz - exchange
        terms = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                if c + 1 < shape[1]:
                    terms.append(HorizontalTwoSiteOperator(r, c, bond_op))
                if r + 1 < shape[0]:
                    terms.append(VerticalTwoSiteOperator(r, c, bond_op))
        local_hamiltonian = LocalHamiltonian(shape=shape, terms=tuple(terms))

        model = PEPS(
            rngs=nnx.Rngs(2),
            shape=shape,
            bond_dim=2,
            contraction_strategy=NoTruncation(),
        )

        samples_spin = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        samples = spin_to_occupancy(samples_spin)
        amps = _value(model, samples)
        local_fast = local_estimate(model, samples, local_hamiltonian, amps)
        local_slow = local_estimate(model, samples, netket_hamiltonian, amps)

        mask = jnp.abs(amps) > 1e-12
        max_diff = jnp.max(jnp.abs(local_fast[mask] - local_slow[mask]))
        self.assertLess(float(max_diff), 1e-10)

    def test_multi_operator_evaluation(self) -> None:
        """Multiple operators evaluated via build_mc_kernels match individual evaluations."""
        shape = (3, 3)
        n_sites = shape[0] * shape[1]

        sz_sz = jnp.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=jnp.complex128,
        )
        exchange = jnp.array(
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
            dtype=jnp.complex128,
        )

        # Hamiltonian: full Heisenberg on all bonds
        heisenberg = sz_sz - exchange
        ham_terms = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                if c + 1 < shape[1]:
                    ham_terms.append(HorizontalTwoSiteOperator(r, c, heisenberg))
                if r + 1 < shape[0]:
                    ham_terms.append(VerticalTwoSiteOperator(r, c, heisenberg))
        hamiltonian = LocalHamiltonian(shape=shape, terms=tuple(ham_terms))

        # Observable 1: Sz⊗Sz correlator on horizontal bonds only
        sz_obs = LocalHamiltonian(shape=shape, terms=tuple(
            HorizontalTwoSiteOperator(r, c, sz_sz)
            for r in range(shape[0]) for c in range(shape[1] - 1)
        ))

        # Observable 2: exchange on vertical bonds only
        ex_obs = LocalHamiltonian(shape=shape, terms=tuple(
            VerticalTwoSiteOperator(r, c, exchange)
            for r in range(shape[0] - 1) for c in range(shape[1])
        ))

        model = PEPS(
            rngs=nnx.Rngs(42),
            shape=shape,
            bond_dim=4,
            contraction_strategy=NoTruncation(),
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        # Merged kernels: Hamiltonian + two observables
        init_cache, transition, estimate_merged = build_mc_kernels(
            model, hamiltonian, observables=(sz_obs, ex_obs),
        )
        # Individual kernels for reference
        _, _, estimate_ham = build_mc_kernels(model, hamiltonian)
        _, _, estimate_sz = build_mc_kernels(model, sz_obs)
        _, _, estimate_ex = build_mc_kernels(model, ex_obs)

        n_test = 16
        key = jax.random.PRNGKey(0)
        key, init_key = jax.random.split(key)
        samples = model.random_physical_configuration(init_key, n_samples=n_test)
        cache = init_cache(tensors, samples)

        tested = 0
        for i in range(n_test):
            cache_i = jax.tree.map(lambda x: x[i], cache)
            key, subkey = jax.random.split(key)
            sample_next, _, ctx = transition(tensors, samples[i], subkey, cache_i)

            _, merged = estimate_merged(tensors, sample_next, ctx)
            _, ref_ham = estimate_ham(tensors, sample_next, ctx)
            _, ref_sz = estimate_sz(tensors, sample_next, ctx)
            _, ref_ex = estimate_ex(tensors, sample_next, ctx)

            self.assertEqual(merged.local_estimate.shape, (3,))
            self.assertAlmostEqual(
                float(jnp.abs(merged.local_estimate[0] - ref_ham.local_estimate[0])),
                0.0, places=9,
            )
            self.assertAlmostEqual(
                float(jnp.abs(merged.local_estimate[1] - ref_sz.local_estimate[0])),
                0.0, places=9,
            )
            self.assertAlmostEqual(
                float(jnp.abs(merged.local_estimate[2] - ref_ex.local_estimate[0])),
                0.0, places=9,
            )
            tested += 1
        self.assertGreater(tested, 1)


if __name__ == "__main__":
    unittest.main()

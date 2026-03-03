"""Exact local energy checks for PEPS."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.operators.local_terms import (
    HorizontalTwoSiteOperator,
    LocalHamiltonian,
    VerticalTwoSiteOperator,
    merge_operators,
)
from vmc.peps import NoTruncation, PEPS
from vmc.peps.common.contraction import _forward_with_cache
from vmc.peps.common.energy import _compute_all_env_grads_and_energy
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
        """Multiple operators evaluated in one backward pass match individual evaluations."""
        shape = (2, 2)
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)

        sz_sz = jnp.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=jnp.complex128,
        )
        exchange = jnp.array(
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
            dtype=jnp.complex128,
        )
        bond_op = sz_sz - exchange

        # Full Hamiltonian: all bonds
        h_terms = []
        v_terms = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                if c + 1 < shape[1]:
                    h_terms.append(HorizontalTwoSiteOperator(r, c, bond_op))
                if r + 1 < shape[0]:
                    v_terms.append(VerticalTwoSiteOperator(r, c, bond_op))
        full_ham = LocalHamiltonian(shape=shape, terms=tuple(h_terms + v_terms))

        # Subregion: just horizontal bonds
        h_only = LocalHamiltonian(shape=shape, terms=tuple(h_terms))
        # Subregion: just vertical bonds
        v_only = LocalHamiltonian(shape=shape, terms=tuple(v_terms))

        model = PEPS(
            rngs=nnx.Rngs(42),
            shape=shape,
            bond_dim=2,
            contraction_strategy=NoTruncation(),
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        # Merge all three operators
        merged_terms, coeff_struct = merge_operators(
            (full_ham, h_only, v_only), shape, eval_span=type(model).eval_span,
        )

        samples_spin = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        samples = spin_to_occupancy(samples_spin)
        amps = _value(model, samples)

        # Compute individual local energies
        e_full = local_estimate(model, samples, full_ham, amps)
        e_h = local_estimate(model, samples, h_only, amps)
        e_v = local_estimate(model, samples, v_only, amps)

        # Compute multi-operator local energies using merged backward pass
        for i in range(samples.shape[0]):
            sample = samples[i]
            amp = amps[i]
            if jnp.abs(amp) < 1e-12:
                continue
            spins = sample.reshape(shape)
            _, top_envs = _forward_with_cache(tensors, spins, shape, model.strategy)
            _, energies_vec, _ = _compute_all_env_grads_and_energy(
                tensors,
                spins,
                amp,
                shape,
                model.strategy,
                top_envs,
                terms=merged_terms,
                collect_grads=False,
            )
            self.assertEqual(energies_vec.shape, (3,))
            self.assertAlmostEqual(
                float(jnp.abs(energies_vec[0] - e_full[i])), 0.0, places=9
            )
            self.assertAlmostEqual(
                float(jnp.abs(energies_vec[1] - e_h[i])), 0.0, places=9
            )
            self.assertAlmostEqual(
                float(jnp.abs(energies_vec[2] - e_v[i])), 0.0, places=9
            )
            break  # One sample is enough to verify


if __name__ == "__main__":
    unittest.main()

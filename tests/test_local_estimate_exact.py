"""Exact local energy checks for MPS and PEPS."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.core import _value
from vmc.models.mps import MPS
from vmc.models.peps import NoTruncation, PEPS
from vmc.operators.local_terms import (
    HorizontalTwoSiteTerm,
    LocalHamiltonian,
    VerticalTwoSiteTerm,
)
from vmc.utils.utils import spin_to_occupancy
from vmc.utils.vmc_utils import local_estimate


class LocalEstimateExactTest(unittest.TestCase):
    def test_mps_local_estimate_matches_dense(self) -> None:
        n_sites = 8
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Chain(length=n_sites, pbc=True)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
        )
        model = MPS(rngs=nnx.Rngs(0), n_sites=n_sites, bond_dim=3)

        samples_spin = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        samples = spin_to_occupancy(samples_spin)
        amps = _value(model, samples)
        local = local_estimate(model, samples, hamiltonian, amps)
        h_dense = jnp.asarray(hamiltonian.to_dense(), dtype=amps.dtype)
        expected = (h_dense @ amps) / amps

        mask = jnp.abs(amps) > 1e-12
        max_diff = jnp.max(jnp.abs(local[mask] - expected[mask]))
        self.assertLess(float(max_diff), 1e-10)

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
                    horizontal_terms.append(HorizontalTwoSiteTerm(r, c, bond_op))
                if r + 1 < shape[0]:
                    vertical_terms.append(VerticalTwoSiteTerm(r, c, bond_op))
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
        """Test that LocalHamiltonian and NetKet operator give same results for PEPS."""
        shape = (2, 2)  # Small lattice for efficiency (generic dispatch is slow)
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Grid(extent=shape, pbc=False)
        netket_hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
        )
        # Build equivalent LocalHamiltonian
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
                    terms.append(HorizontalTwoSiteTerm(r, c, bond_op))
                if r + 1 < shape[0]:
                    terms.append(VerticalTwoSiteTerm(r, c, bond_op))
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

        # LocalHamiltonian dispatch (fast, with environment reuse)
        local_fast = local_estimate(model, samples, local_hamiltonian, amps)
        # NetKet operator dispatch (slow, no environment reuse)
        local_slow = local_estimate(model, samples, netket_hamiltonian, amps)

        mask = jnp.abs(amps) > 1e-12
        max_diff = jnp.max(jnp.abs(local_fast[mask] - local_slow[mask]))
        self.assertLess(float(max_diff), 1e-10)


if __name__ == "__main__":
    unittest.main()

"""Exact local energy checks for MPS and PEPS."""
from __future__ import annotations

import unittest

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax.numpy as jnp
import netket as nk
from flax import nnx

from VMC.core import _value
from VMC.models.mps import MPS
from VMC.models.peps import NoTruncation, PEPS
from VMC.utils.vmc_utils import local_estimate


class LocalEstimateExactTest(unittest.TestCase):
    def test_mps_local_estimate_matches_dense(self) -> None:
        n_sites = 8
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Chain(length=n_sites, pbc=True)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
        )
        model = MPS(rngs=nnx.Rngs(0), n_sites=n_sites, bond_dim=3)

        samples = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        local = local_estimate(model, samples, hamiltonian)
        amps = _value(model, samples)
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
        model = PEPS(
            rngs=nnx.Rngs(1),
            shape=shape,
            bond_dim=3,
            contraction_strategy=NoTruncation(),
        )

        samples = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        local = local_estimate(model, samples, hamiltonian)
        amps = _value(model, samples)
        h_dense = jnp.asarray(hamiltonian.to_dense(), dtype=amps.dtype)
        expected = (h_dense @ amps) / amps

        mask = jnp.abs(amps) > 1e-12
        max_diff = jnp.max(jnp.abs(local[mask] - expected[mask]))
        self.assertLess(float(max_diff), 1e-9)


if __name__ == "__main__":
    unittest.main()

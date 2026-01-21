"""SR preconditioner correctness against dense QGT solve."""
from __future__ import annotations

import unittest

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from VMC.core import _value_and_grad
from VMC.models.mps import MPS
from VMC.preconditioners import SRPreconditioner
from VMC.qgt import Jacobian, ParameterSpace, QGT
from VMC.samplers.sequential import sequential_sample
from VMC.utils.vmc_utils import local_estimate


class SRPreconditionerDenseTest(unittest.TestCase):
    def test_sr_matches_dense_solve(self) -> None:
        n_sites = 8
        bond_dim = 4
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        graph = nk.graph.Chain(length=n_sites, pbc=True)
        hamiltonian = nk.operator.Heisenberg(
            hi, graph, dtype=jnp.complex128
        )
        model = MPS(rngs=nnx.Rngs(0), n_sites=n_sites, bond_dim=bond_dim)

        key = jax.random.key(0)
        samples = sequential_sample(
            model,
            n_samples=512,
            n_chains=8,
            burn_in=30,
            key=key,
        )
        amps, grads_sliced, p = _value_and_grad(
            model, samples, full_gradient=False
        )
        amps_full, grads_full, _ = _value_and_grad(
            model, samples, full_gradient=True
        )

        local_energies = local_estimate(model, samples, hamiltonian)

        preconditioner = SRPreconditioner(diag_shift=1e-3)
        updates = preconditioner.apply(
            model,
            samples,
            grads_sliced / amps[:, None],
            p,
            local_energies,
            grad_factor=1.0,
        )
        updates_flat, _ = jax.flatten_util.ravel_pytree(
            jax.tree_util.tree_map(jnp.asarray, updates)
        )

        O_full = grads_full / amps_full[:, None]
        dv = (local_energies.reshape(-1) - jnp.mean(local_energies))
        dv = dv / samples.shape[0]
        mean = jnp.mean(O_full, axis=0)
        rhs = O_full.conj().T @ dv - mean.conj() * jnp.sum(dv)
        S = QGT(Jacobian(O_full), space=ParameterSpace()).to_dense()
        mat = S + 1e-3 * jnp.eye(S.shape[0], dtype=S.dtype)
        expected = jnp.linalg.solve(mat, rhs)

        rel_err = float(
            jnp.linalg.norm(updates_flat - expected)
            / jnp.linalg.norm(expected)
        )
        self.assertLess(rel_err, 1e-8)


if __name__ == "__main__":
    unittest.main()

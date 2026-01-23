"""QGT parity checks against NetKet for tensor-network models."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.core import _value_and_grad
from vmc.models.mps import MPS
from vmc.models.peps import NoTruncation, PEPS
from vmc.qgt import ParameterSpace, QGT, SlicedJacobian, SiteOrdering
from vmc.utils.smallo import params_per_site
from vmc.utils.vmc_utils import flatten_samples


class QGTNetKetParityTest(unittest.TestCase):
    SEED = 0
    N_CHAINS = 16
    BURN_IN = 20
    N_SAMPLES_MPS = 4096
    N_SAMPLES_PEPS = 1024

    def _compare_qgt(self, model, hilbert, n_samples: int):
        sampler = nk.sampler.MetropolisLocal(
            hilbert, n_chains=self.N_CHAINS, sweep_size=int(hilbert.size), reset_chains=False,
        )
        vstate = nk.vqs.MCState(
            sampler, model, n_samples=n_samples, n_discard_per_chain=self.BURN_IN, seed=self.SEED,
        )
        vstate.sample()
        samples = flatten_samples(jnp.asarray(vstate.samples))

        qgt_nk = nk.optimizer.qgt.QGTJacobianDense(diag_shift=0.0)
        S_nk = qgt_nk(vstate, holomorphic=True).to_dense()

        amps, grads, p = _value_and_grad(model, samples, full_gradient=False)
        o = grads / amps[:, None]
        ordering = SiteOrdering(tuple(params_per_site(model)))
        jac = SlicedJacobian(o, p, model.phys_dim, ordering)
        S_ours = QGT(jac, space=ParameterSpace()).to_dense()

        err = float(jnp.linalg.norm(S_ours - S_nk) / jnp.linalg.norm(S_nk))
        self.assertLess(err, 1e-8)

    def test_mps_qgt_matches_netket(self):
        n_sites = 12
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        model = MPS(rngs=nnx.Rngs(self.SEED), n_sites=n_sites, bond_dim=6)
        self._compare_qgt(model, hi, n_samples=self.N_SAMPLES_MPS)

    def test_peps_qgt_matches_netket(self):
        shape = (3, 3)
        hi = nk.hilbert.Spin(s=1 / 2, N=shape[0] * shape[1])
        model = PEPS(rngs=nnx.Rngs(self.SEED), shape=shape, bond_dim=4, contraction_strategy=NoTruncation())
        self._compare_qgt(model, hi, n_samples=self.N_SAMPLES_PEPS)


if __name__ == "__main__":
    unittest.main()

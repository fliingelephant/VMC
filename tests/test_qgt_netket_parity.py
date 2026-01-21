"""QGT parity checks against NetKet for tensor-network models."""
from __future__ import annotations

import functools
import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.core import _value_and_grad
from vmc.models.mps import MPS
from vmc.models.peps import NoTruncation, PEPS
from vmc.qgt import ParameterSpace, QGT, SlicedJacobian, SiteOrdering
from vmc.utils.smallo import params_per_site
from vmc.utils.vmc_utils import flatten_samples, model_params


def _mps_apply_fun(variables, x, **kwargs):
    del kwargs
    tensors = variables["params"]["tensors"]
    samples = x if x.ndim == 2 else x[None, :]
    amps = MPS._batch_amplitudes(tensors, samples)
    log_amps = jnp.log(amps)
    return log_amps if x.ndim == 2 else log_amps[0]


def _peps_apply_fun(shape, strategy, variables, x, **kwargs):
    del kwargs
    tensors = variables["params"]["tensors"]
    samples = x if x.ndim == 2 else x[None, :]
    amps = jax.vmap(
        lambda s: PEPS._single_amplitude(tensors, s, shape, strategy)
    )(samples)
    log_amps = jnp.log(amps)
    return log_amps if x.ndim == 2 else log_amps[0]


class QGTNetKetParityTest(unittest.TestCase):
    SEED = 0
    N_CHAINS = 16
    BURN_IN = 20
    N_SAMPLES_MPS = 4096
    N_SAMPLES_PEPS = 1024

    def _compare_qgt(self, model, hilbert, apply_fun, params, n_samples: int):
        sampler = nk.sampler.MetropolisLocal(
            hilbert,
            n_chains=self.N_CHAINS,
            sweep_size=int(hilbert.size),
            reset_chains=False,
        )
        vstate = nk.vqs.MCState(
            sampler,
            model=None,
            n_samples=n_samples,
            n_discard_per_chain=self.BURN_IN,
            variables={"params": params},
            apply_fun=apply_fun,
            sampler_seed=self.SEED,
        )
        vstate.sample()
        samples = flatten_samples(jnp.asarray(vstate.samples))

        qgt_nk = nk.optimizer.qgt.QGTJacobianDense(diag_shift=0.0)
        S_nk = qgt_nk(vstate, holomorphic=True).to_dense()

        amps, grads, p = _value_and_grad(model, samples, full_gradient=False)
        o = grads / amps[:, None]
        ordering = SiteOrdering(tuple(params_per_site(model)))
        if hasattr(model, "phys_dim"):
            phys_dim = int(model.phys_dim)
        else:
            phys_dim = int(jnp.asarray(model.tensors[0][0]).shape[0])
        jac = SlicedJacobian(o, p, phys_dim, ordering)
        S_ours = QGT(jac, space=ParameterSpace()).to_dense()

        err = float(
            jnp.linalg.norm(S_ours - S_nk) / jnp.linalg.norm(S_nk)
        )
        self.assertLess(err, 1e-8)

    def test_mps_qgt_matches_netket(self):
        n_sites = 12
        bond_dim = 6
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        model = MPS(rngs=nnx.Rngs(self.SEED), n_sites=n_sites, bond_dim=bond_dim)
        params = model_params(model)
        self._compare_qgt(
            model,
            hi,
            _mps_apply_fun,
            params,
            n_samples=self.N_SAMPLES_MPS,
        )

    def test_peps_qgt_matches_netket(self):
        shape = (3, 3)
        bond_dim = 4
        n_sites = shape[0] * shape[1]
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        model = PEPS(
            rngs=nnx.Rngs(self.SEED),
            shape=shape,
            bond_dim=bond_dim,
            contraction_strategy=NoTruncation(),
        )
        params = model_params(model)
        apply_fun = functools.partial(
            _peps_apply_fun, shape, model.strategy
        )
        self._compare_qgt(
            model,
            hi,
            apply_fun,
            params,
            n_samples=self.N_SAMPLES_PEPS,
        )


if __name__ == "__main__":
    unittest.main()

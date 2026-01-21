"""NetKet compatibility checks for QGT and SR adapters."""
from __future__ import annotations

import unittest

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from VMC.models.mps import MPS
from VMC.qgt import Jacobian, ParameterSpace, QGT
from VMC.qgt.netket_compat import DenseSR
from VMC.utils.vmc_utils import build_dense_jac, flatten_samples, get_apply_fun, model_params


def _mps_apply_fun(variables, x, **kwargs):
    del kwargs
    tensors = variables["params"]["tensors"]
    samples = x if x.ndim == 2 else x[None, :]
    amps = MPS._batch_amplitudes(tensors, samples)
    log_amps = jnp.log(amps)
    return log_amps if x.ndim == 2 else log_amps[0]


class NetKetCompatTest(unittest.TestCase):
    SEED = 0
    N_SAMPLES = 1024
    N_CHAINS = 16
    BURN_IN = 30

    def test_qgt_operator_dense_parity(self) -> None:
        n_sites = 10
        model = MPS(rngs=nnx.Rngs(self.SEED), n_sites=n_sites, bond_dim=4)
        params = model_params(model)
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        sampler = nk.sampler.MetropolisLocal(
            hi,
            n_chains=self.N_CHAINS,
            sweep_size=int(hi.size),
            reset_chains=False,
        )
        vstate = nk.vqs.MCState(
            sampler,
            model=None,
            n_samples=self.N_SAMPLES,
            n_discard_per_chain=self.BURN_IN,
            variables={"params": params},
            apply_fun=_mps_apply_fun,
            sampler_seed=self.SEED,
        )
        vstate.sample()

        diag_shift = 1e-2
        sr = DenseSR(diag_shift=diag_shift, holomorphic=True)
        qgt_op = sr.lhs_constructor(vstate)
        S_dense = qgt_op.to_dense()

        samples = flatten_samples(jnp.asarray(vstate.samples))
        apply_fun, params_ref, model_state, _ = get_apply_fun(vstate)
        O = build_dense_jac(
            apply_fun, params_ref, model_state, samples, holomorphic=True
        )
        S_expected = QGT(Jacobian(O), space=ParameterSpace()).to_dense()
        S_expected = S_expected + diag_shift * jnp.eye(
            S_expected.shape[0], dtype=S_expected.dtype
        )
        err = float(
            jnp.linalg.norm(S_dense - S_expected) / jnp.linalg.norm(S_expected)
        )
        self.assertLess(err, 1e-10)

        key = jax.random.key(self.SEED)
        leaves = jax.tree_util.tree_leaves(params_ref)
        keys = jax.random.split(key, len(leaves))
        keys_tree = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(params_ref), keys
        )
        vtree = jax.tree_util.tree_map(
            lambda k, x: jax.random.normal(k, x.shape, x.dtype),
            keys_tree,
            params_ref,
        )
        flat, unravel = jax.flatten_util.ravel_pytree(vtree)
        mv_expected = unravel(S_expected @ flat)
        mv_actual = qgt_op @ vtree
        diff = float(
            jnp.linalg.norm(
                jax.flatten_util.ravel_pytree(mv_actual)[0]
                - jax.flatten_util.ravel_pytree(mv_expected)[0]
            )
        )
        self.assertLess(diff, 1e-8)


if __name__ == "__main__":
    unittest.main()

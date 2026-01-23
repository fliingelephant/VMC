"""Tests for VMC utility helpers."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.models.mps import MPS
from vmc.utils.vmc_utils import batched_eval, build_dense_jac


class VMCUtilsTest(unittest.TestCase):
    def test_batched_eval_matches_direct(self) -> None:
        key = jax.random.key(0)
        samples = jax.random.normal(key, (33, 5))

        def eval_fn(x):
            return jnp.sum(x**2, axis=-1)

        direct = eval_fn(samples)
        batched = batched_eval(eval_fn, samples, batch_size=8)
        diff = float(jnp.max(jnp.abs(direct - batched)))
        self.assertLess(diff, 1e-12)

    def test_build_dense_jac_matches_manual(self) -> None:
        n_sites = 8
        model = MPS(rngs=nnx.Rngs(0), n_sites=n_sites, bond_dim=3)
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=1, sweep_size=n_sites)
        vstate = nk.vqs.MCState(sampler, model, n_samples=32, seed=0)

        samples = jnp.asarray(jax.random.bernoulli(jax.random.key(1), 0.5, (32, n_sites)) * 2 - 1, dtype=jnp.int32)
        jac_auto = build_dense_jac(vstate._apply_fun, vstate.parameters, vstate.model_state, samples, holomorphic=True)

        jac_fun = jax.jacrev(
            lambda p, x: vstate._apply_fun({"params": p, **vstate.model_state}, x), holomorphic=True,
        )
        jac_tree = jax.vmap(jac_fun, in_axes=(None, 0))(vstate.parameters, samples)
        jac_tree = jax.tree_util.tree_map(lambda x: x - jnp.mean(x, axis=0, keepdims=True), jac_tree)
        leaves = [leaf.reshape(samples.shape[0], -1) for leaf in jax.tree_util.tree_leaves(jac_tree)]
        jac_manual = jnp.concatenate(leaves, axis=1)

        diff = float(jnp.max(jnp.abs(jac_auto - jac_manual)))
        self.assertLess(diff, 1e-12)


if __name__ == "__main__":
    unittest.main()

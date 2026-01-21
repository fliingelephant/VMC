"""Tests for VMC utility helpers."""
from __future__ import annotations

import unittest

from VMC import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.models.mps import MPS
from VMC.utils.vmc_utils import batched_eval, build_dense_jac, model_params


def _mps_apply_fun(variables, x, **kwargs):
    del kwargs
    tensors = variables["params"]["tensors"]
    samples = x if x.ndim == 2 else x[None, :]
    amps = MPS._batch_amplitudes(tensors, samples)
    log_amps = jnp.log(amps)
    return log_amps if x.ndim == 2 else log_amps[0]


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
        model = MPS(rngs=nnx.Rngs(0), n_sites=8, bond_dim=3)
        params = model_params(model)
        samples = jnp.asarray(
            jax.random.bernoulli(jax.random.key(1), 0.5, (32, 8)) * 2 - 1,
            dtype=jnp.int32,
        )
        jac_auto = build_dense_jac(
            _mps_apply_fun, params, {}, samples, holomorphic=True
        )

        jac_fun = jax.jacrev(
            lambda p, x: _mps_apply_fun({"params": p}, x),
            holomorphic=True,
        )
        jac_tree = jax.vmap(jac_fun, in_axes=(None, 0))(params, samples)
        jac_tree = jax.tree_util.tree_map(
            lambda x: (x - jnp.mean(x, axis=0, keepdims=True))
            / jnp.sqrt(samples.shape[0]),
            jac_tree,
        )
        leaves = [
            leaf.reshape(samples.shape[0], -1)
            for leaf in jax.tree_util.tree_leaves(jac_tree)
        ]
        jac_manual = jnp.concatenate(leaves, axis=1)

        diff = float(jnp.max(jnp.abs(jac_auto - jac_manual)))
        self.assertLess(diff, 1e-12)


if __name__ == "__main__":
    unittest.main()

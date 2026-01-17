"""PEPS evaluation consistency checks."""
from __future__ import annotations

import unittest

from VMC import config  # noqa: F401

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.core import _value_and_grad_batch
from VMC.models.peps import NoTruncation, PEPS, make_peps_amplitude
from VMC.utils.utils import spin_to_occupancy


class PEPSEvalTest(unittest.TestCase):
    def test_peps_all_configs_gradients(self):
        shape = (3, 3)
        bond_dim = 4
        model = PEPS(
            rngs=nnx.Rngs(0),
            shape=shape,
            bond_dim=bond_dim,
            contraction_strategy=NoTruncation(),
        )
        n_sites = shape[0] * shape[1]

        configs = jnp.arange(2 ** n_sites, dtype=jnp.int32)
        site_ids = jnp.arange(n_sites, dtype=jnp.int32)
        bits = (configs[:, None] >> site_ids) & 1
        samples = (2 * bits - 1).astype(jnp.int32)

        amps, grads_sliced, _ = _value_and_grad_batch(
            model, samples, full_gradient=False
        )
        tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]

        def expand_sliced(sample, grad_sliced):
            spins = spin_to_occupancy(sample).reshape(shape)
            parts = []
            offset = 0
            for r in range(shape[0]):
                for c in range(shape[1]):
                    up = 1 if r == 0 else bond_dim
                    down = 1 if r == shape[0] - 1 else bond_dim
                    left = 1 if c == 0 else bond_dim
                    right = 1 if c == shape[1] - 1 else bond_dim
                    size_per_phys = up * down * left * right
                    grad_site = grad_sliced[offset : offset + size_per_phys]
                    offset += size_per_phys
                    full_site = jnp.zeros(
                        (2, size_per_phys), dtype=grad_sliced.dtype
                    )
                    full_site = full_site.at[spins[r, c]].set(grad_site)
                    parts.append(full_site.reshape(-1))
            return jnp.concatenate(parts)

        grads_full = jax.vmap(expand_sliced)(samples, grads_sliced)

        def amp_fn(tensors, sample):
            return PEPS._single_amplitude(tensors, sample, shape, model.strategy)

        amps_ref = jax.vmap(amp_fn, in_axes=(None, 0))(tensors, samples)

        jac_fun = jax.jacrev(amp_fn, holomorphic=True)
        jac_tree = jax.vmap(jac_fun, in_axes=(None, 0))(tensors, samples)
        jac_leaves = [
            leaf.reshape(samples.shape[0], -1)
            for leaf in jax.tree_util.tree_leaves(jac_tree)
        ]
        jac = jnp.concatenate(jac_leaves, axis=1)

        amp_custom = make_peps_amplitude(shape, model.strategy)
        jac_custom_fun = jax.jacrev(amp_custom, holomorphic=True)
        jac_custom_tree = jax.vmap(jac_custom_fun, in_axes=(None, 0))(
            tensors, samples
        )
        jac_custom_leaves = [
            leaf.reshape(samples.shape[0], -1)
            for leaf in jax.tree_util.tree_leaves(jac_custom_tree)
        ]
        jac_custom = jnp.concatenate(jac_custom_leaves, axis=1)

        max_amp_diff = jnp.max(jnp.abs(amps - amps_ref))
        max_grad_diff = jnp.max(jnp.abs(grads_full - jac))
        max_custom_diff = jnp.max(jnp.abs(jac_custom - jac))
        self.assertLess(max_amp_diff, 1e-12)
        self.assertLess(max_grad_diff, 1e-12)
        self.assertLess(max_custom_diff, 1e-12)


if __name__ == "__main__":
    unittest.main()

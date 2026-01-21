"""Unit tests for gauge projection utilities."""
from __future__ import annotations

import unittest

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.gauge import GaugeConfig, compute_gauge_projection
from vmc.gauge.gauge import _flatten_tensor_list, _null_vectors_for_bond
from vmc.core import _value_and_grad
from vmc.models.mps import MPS
from vmc.models.peps import PEPS
from vmc.qgt import Jacobian, ParameterSpace, QGT


class GaugeProjectionTest(unittest.TestCase):
    def test_mps_projection_orthonormal(self) -> None:
        model = MPS(rngs=nnx.Rngs(0), n_sites=8, bond_dim=4)
        params = {"tensors": [jnp.asarray(t) for t in model.tensors]}
        cfg = GaugeConfig(include_global_scale=True)
        Q, info = compute_gauge_projection(cfg, model, params, return_info=True)

        qhq = Q.conj().T @ Q
        eye = jnp.eye(qhq.shape[0], dtype=qhq.dtype)
        err = float(jnp.linalg.norm(qhq - eye))
        self.assertLess(err, 1e-10)
        self.assertGreaterEqual(int(info["n_keep"]), 1)

    def test_mps_projection_nullspace(self) -> None:
        model = MPS(rngs=nnx.Rngs(1), n_sites=7, bond_dim=3)
        tensors = [jnp.asarray(t) for t in model.tensors]
        params = {"tensors": tensors}
        cfg = GaugeConfig(include_global_scale=True)
        Q = compute_gauge_projection(cfg, model, params)

        null_blocks = []
        for bond in range(len(tensors) - 1):
            null_block = _null_vectors_for_bond(tensors, bond)
            null_blocks.append(null_block.T)
        flat_tensors = _flatten_tensor_list(tensors)
        null_blocks.append(flat_tensors[:, None])
        T = jnp.concatenate(null_blocks, axis=1)

        proj = Q.conj().T @ T
        err = float(jnp.linalg.norm(proj))
        self.assertLess(err, 1e-10)

    def test_gauge_vectors_null_qim(self) -> None:
        n_sites = 10
        bond_dim = 4
        model = MPS(rngs=nnx.Rngs(2), n_sites=n_sites, bond_dim=bond_dim)
        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        amps, grads, _ = _value_and_grad(model, states, full_gradient=True)
        mask = jnp.abs(amps) > 1e-12
        o = grads[mask] / amps[mask, None]
        qgt = QGT(Jacobian(o), space=ParameterSpace())
        S = qgt.to_dense()

        tensors = [jnp.asarray(t) for t in model.tensors]
        null_blocks = []
        for bond in range(len(tensors) - 1):
            null_blocks.append(_null_vectors_for_bond(tensors, bond).T)
        flat_tensors = _flatten_tensor_list(tensors)
        null_blocks.append(flat_tensors[:, None])
        T = jnp.concatenate(null_blocks, axis=1)

        residual = S @ T
        max_norm = float(jnp.max(jnp.linalg.norm(residual, axis=0)))
        ref_norm = float(jnp.linalg.norm(S))
        self.assertLess(max_norm / (ref_norm + 1e-12), 1e-8)

    def test_gauge_projection_conditions_qim(self) -> None:
        n_sites = 10
        bond_dim = 4
        model = MPS(rngs=nnx.Rngs(3), n_sites=n_sites, bond_dim=bond_dim)
        tensors = [jnp.asarray(t) for t in model.tensors]
        params = {"tensors": tensors}
        cfg = GaugeConfig(include_global_scale=True)
        Q, info = compute_gauge_projection(cfg, model, params, return_info=True)

        hi = nk.hilbert.Spin(s=0.5, N=n_sites)
        states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
        amps, grads, _ = _value_and_grad(model, states, full_gradient=True)
        mask = jnp.abs(amps) > 1e-12
        o = grads[mask] / amps[mask, None]
        S = QGT(Jacobian(o), space=ParameterSpace()).to_dense()
        S_proj = Q.conj().T @ S @ Q

        n_params = int(sum(t.size for t in tensors))
        bond_dims = [int(t.shape[2]) for t in tensors[:-1]]
        n_gauge = sum(d * d for d in bond_dims) + 1
        self.assertEqual(info["n_keep"], n_params - n_gauge)
        self.assertEqual(S_proj.shape[0], info["n_keep"])

        eig_full = jnp.sort(jnp.real(jnp.linalg.eigvalsh(S)))
        eig_proj = jnp.sort(jnp.real(jnp.linalg.eigvalsh(S_proj)))
        null_full = int(jnp.sum(eig_full < 1e-10))
        null_proj = int(jnp.sum(eig_proj < 1e-10))
        self.assertGreater(null_full, null_proj)

    def test_peps_projection_not_implemented(self) -> None:
        model = PEPS(rngs=nnx.Rngs(2), shape=(2, 2), bond_dim=2)
        params = {"tensors": [[jnp.asarray(t) for t in row] for row in model.tensors]}
        cfg = GaugeConfig()
        with self.assertRaises(NotImplementedError):
            compute_gauge_projection(cfg, model, params)


if __name__ == "__main__":
    unittest.main()

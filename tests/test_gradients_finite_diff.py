"""Finite-difference checks for PEPS amplitude gradients."""
from __future__ import annotations

import unittest

import numpy as np

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.peps import NoTruncation, PEPS
from vmc.peps.standard.compat import _value_and_grad


def _central_diff_grad(func, x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x)
    grad = np.zeros(
        x.size,
        dtype=np.complex128 if np.iscomplexobj(x) else x.dtype,
    )
    epsd = np.zeros_like(x)
    for i in range(x.size):
        epsd[i] = eps
        f_plus = func(x + epsd)
        f_minus = func(x - epsd)
        grad[i] = 0.5 * (f_plus - f_minus) / eps
        epsd[i] = 0
    return grad


class FiniteDiffGradientTest(unittest.TestCase):
    def test_peps_amplitude_gradient(self) -> None:
        shape = (2, 3)
        model = PEPS(
            rngs=nnx.Rngs(1),
            shape=shape,
            bond_dim=3,
            contraction_strategy=NoTruncation(),
        )
        n_sites = shape[0] * shape[1]
        key = jax.random.key(1)
        sample = jax.random.bernoulli(key, 0.5, (n_sites,)).astype(jnp.int32)

        tensors = [[np.asarray(t) for t in row] for row in model.tensors]
        flat = np.concatenate([np.ravel(t) for row in tensors for t in row])
        shapes = [t.shape for row in tensors for t in row]

        def unflatten_peps(flat_params: np.ndarray) -> list[list[np.ndarray]]:
            nested = []
            offset = 0
            idx = 0
            for _ in range(shape[0]):
                row = []
                for _ in range(shape[1]):
                    size = int(np.prod(shapes[idx]))
                    tensor = np.asarray(
                        flat_params[offset : offset + size]
                    ).reshape(shapes[idx])
                    row.append(tensor)
                    offset += size
                    idx += 1
                nested.append(row)
            return nested

        def amp_from_flat(flat_params: np.ndarray) -> np.ndarray:
            tensors_local = unflatten_peps(flat_params)
            amp = PEPS.apply(
                tensors_local, sample, shape, model.strategy
            )
            return np.asarray(amp)

        _, grad, _ = _value_and_grad(model, sample, full_gradient=True)
        grad_fd = _central_diff_grad(amp_from_flat, flat, eps=1e-6)
        max_diff = np.max(np.abs(np.asarray(grad) - grad_fd))
        self.assertLess(max_diff, 1e-5)


if __name__ == "__main__":
    unittest.main()

"""Canonical standard PEPS model container."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.peps.common.strategy import ContractionStrategy, Variational
from vmc.peps.standard.compat import peps_apply
from vmc.operators.local_terms import support_span
from vmc.utils.utils import random_tensor

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = ["PEPS"]


class PEPS(nnx.Module):
    """Open-boundary PEPS on a rectangular lattice."""

    tensors: list[list[nnx.Param]] = nnx.data()

    @staticmethod
    def site_dims(
        row: int, col: int, n_rows: int, n_cols: int, bond_dim: int
    ) -> tuple[int, int, int, int]:
        up = 1 if row == 0 else bond_dim
        down = 1 if row == n_rows - 1 else bond_dim
        left = 1 if col == 0 else bond_dim
        right = 1 if col == n_cols - 1 else bond_dim
        return up, down, left, right

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        shape: tuple[int, int],
        bond_dim: int,
        phys_dim: int = 2,
        contraction_strategy: ContractionStrategy | None = None,
        dtype: "DTypeLike" = jnp.complex128,
    ):
        self.shape = (int(shape[0]), int(shape[1]))
        self.bond_dim = int(bond_dim)
        self.phys_dim = int(phys_dim)
        self.dtype = jnp.dtype(dtype)
        if contraction_strategy is None:
            contraction_strategy = Variational(
                truncate_bond_dimension=self.bond_dim * self.bond_dim
            )
        self.strategy = contraction_strategy

        n_rows, n_cols = self.shape
        self.tensors = [
            [
                nnx.Param(
                    random_tensor(
                        rngs,
                        (
                            self.phys_dim,
                            *self.site_dims(r, c, n_rows, n_cols, self.bond_dim),
                        ),
                        self.dtype,
                    ),
                    dtype=self.dtype,
                )
                for c in range(n_cols)
            ]
            for r in range(n_rows)
        ]

    apply = staticmethod(peps_apply)
    eval_span = staticmethod(support_span)

    @staticmethod
    def unflatten_sample(sample: jax.Array, shape: tuple[int, int]) -> jax.Array:
        return sample.reshape(shape)

    @nnx.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        amps = jax.vmap(
            lambda s: self.apply(self.tensors, s, self.shape, self.strategy)
        )(x)
        return jnp.log(amps)

    def random_physical_configuration(
        self, key: jax.Array, n_samples: int = 1
    ) -> jax.Array:
        return jax.random.randint(
            key,
            (n_samples, self.shape[0], self.shape[1]),
            0,
            self.phys_dim,
            dtype=jnp.int32,
        )

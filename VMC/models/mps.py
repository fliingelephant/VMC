"""Matrix Product State (MPS) model for variational wavefunctions.

This module provides a lightweight MPS implementation compatible with
NetKet's variational state interface.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from VMC.utils.utils import spin_to_occupancy

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = ["SimpleMPS"]


class SimpleMPS(nnx.Module):
    """Lightweight open-boundary MPS producing log-psi values.

    tensors: list of site tensors with shape (phys_dim, D_left, D_right).
    The boundary bond dimensions are fixed to 1 on the first/last site.

    Attributes:
        n_sites: Number of lattice sites.
        bond_dim: Virtual bond dimension.
        phys_dim: Physical dimension (default 2 for spins).
        dtype: Data type for tensors (default complex128).
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        n_sites: int,
        bond_dim: int,
        dtype: "DTypeLike" = jnp.complex128,
    ):
        """Initialize MPS with random tensors.

        Args:
            rngs: Flax NNX random key generator.
            n_sites: Number of lattice sites.
            bond_dim: Virtual bond dimension.
            dtype: Data type for tensors (default: complex128).
        """
        self.n_sites = n_sites
        self.bond_dim = bond_dim
        self.phys_dim = 2
        self.dtype = jnp.dtype(dtype)

        # Determine real dtype for random initialization
        is_complex = jnp.issubdtype(self.dtype, jnp.complexfloating)
        if is_complex:
            real_dtype = jnp.real(jnp.zeros((), dtype=self.dtype)).dtype
            complex_unit = jnp.array(1j, dtype=self.dtype)
        else:
            real_dtype = self.dtype
            complex_unit = None

        tensors = []
        for site in range(n_sites):
            left_dim = 1 if site == 0 else bond_dim
            right_dim = 1 if site == n_sites - 1 else bond_dim
            shape_t = (self.phys_dim, left_dim, right_dim)

            if is_complex:
                key_re, key_im = rngs.params(), rngs.params()
                tensor_val = (
                    1/2 * jax.random.uniform(key_re, shape_t, dtype=real_dtype)
                    + 1/2 * complex_unit
                    * jax.random.uniform(key_im, shape_t, dtype=real_dtype)
                )
            else:
                tensor_val = jax.random.uniform(
                    rngs.params(),
                    shape_t,
                    dtype=real_dtype,
                )

            tensors.append(nnx.Param(tensor_val, dtype=self.dtype))
        self.tensors = nnx.List(tensors)

    @staticmethod
    def _batch_amplitudes(tensors, samples: jax.Array) -> jax.Array:
        """Compute MPS amplitudes for a batch of spin configurations.

        Note: Uses Python loop because boundary tensors have different shapes
        (bond_dim=1 at edges), preventing use of jax.lax.scan with stacked tensors.

        Args:
            tensors: List of MPS site tensors.
            samples: Spin configurations with shape (batch, n_sites).

        Returns:
            Complex amplitudes with shape (batch,).
        """
        indices = spin_to_occupancy(samples)  # Map spins {-1, 1} -> {0, 1}
        state = jnp.ones((indices.shape[0], 1), dtype=tensors[0].dtype)
        for site, tensor in enumerate(tensors):
            mats = tensor[indices[:, site]]  # (batch, D_left, D_right)
            state = jnp.einsum("bi,bij->bj", state, mats)
        return state.squeeze(-1)

    @nnx.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute log-amplitudes for input spin configurations.

        Args:
            x: Spin configuration(s). Shape (n_sites,) for single sample,
               or (batch, n_sites) for batch.

        Returns:
            Log-amplitude(s). Scalar for single sample, shape (batch,) for batch.
        """
        samples = x if x.ndim == 2 else x[None, :]
        amps = self._batch_amplitudes(self.tensors, samples)
        log_amps = jnp.log(amps)
        return log_amps if x.ndim == 2 else log_amps[0]

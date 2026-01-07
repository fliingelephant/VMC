"""Gauge removal for tensor network variational states.

This module provides gauge projectors that remove redundant directions
from parameter updates, improving optimization stability.

Uses plum multiple dispatch to select the appropriate implementation
based on model type.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from plum import dispatch

from VMC.models.mps import SimpleMPS
from VMC.models.peps import SimplePEPS

__all__ = [
    "GaugeConfig",
    "compute_gauge_projection",
]


@dataclass(frozen=True)
class GaugeConfig:
    """Configuration for gauge removal.

    Gauge freedom in tensor networks corresponds to inserting X @ X^-1
    between contracted indices. This config specifies how to construct
    a projector onto the orthogonal complement of these gauge directions.

    Attributes:
        rcond: Relative condition number for rank determination.
        include_global_scale: Whether to remove global scale direction.
    """

    rcond: float | None = None
    include_global_scale: bool = True


def _flatten_tensor_list(tensor_list: list[jax.Array]) -> jax.Array:
    """Flatten a list of tensors into a single vector."""
    if not tensor_list:
        return jnp.array([], dtype=jnp.float32)
    flat_leaves = [t.reshape(-1) for t in tensor_list]
    return jnp.concatenate(flat_leaves, axis=0)


def _flatten_tensor_list_batched(tensor_list: list[jax.Array]) -> jax.Array:
    """Flatten a batched list of tensors into (batch, n_params)."""
    if not tensor_list:
        return jnp.zeros((0, 0), dtype=jnp.float32)
    batch = int(tensor_list[0].shape[0])
    flat_leaves = [t.reshape(batch, -1) for t in tensor_list]
    return jnp.concatenate(flat_leaves, axis=1)


def _null_vectors_for_bond(
    tensor_list: list[jax.Array],
    bond: int,
) -> jax.Array:
    """Build all gauge null vectors for a single bond."""
    A_i = tensor_list[bond]
    A_j = tensor_list[bond + 1]
    bond_dim = int(A_i.shape[2])
    if A_j.shape[1] != bond_dim:
        raise ValueError(
            f"Bond mismatch between sites {bond} and {bond + 1}: "
            f"{A_i.shape[2]} != {A_j.shape[1]}"
        )

    eye = jnp.eye(bond_dim, dtype=A_i.dtype)
    # delta_i[a, b, :, :, b] = A_i[:, :, a]
    A_i_right = jnp.transpose(A_i, (2, 0, 1))  # (D, phys, left)
    delta_i = A_i_right[:, None, :, :, None] * eye[None, :, None, None, :]
    delta_i = delta_i.reshape(bond_dim * bond_dim, *A_i.shape)

    # delta_j[a, b, :, a, :] = -A_j[:, b, :]
    A_j_left = jnp.transpose(A_j, (1, 0, 2))  # (D, phys, right)
    delta_j = -eye[:, None, :, None, None] * A_j_left[None, :, None, :, :]
    delta_j = jnp.transpose(delta_j, (0, 1, 3, 2, 4))
    delta_j = delta_j.reshape(bond_dim * bond_dim, *A_j.shape)

    n_pairs = bond_dim * bond_dim
    batched = []
    for idx, tensor in enumerate(tensor_list):
        if idx == bond:
            batched.append(delta_i)
        elif idx == bond + 1:
            batched.append(delta_j)
        else:
            batched.append(jnp.zeros((n_pairs,) + tensor.shape, dtype=tensor.dtype))
    return _flatten_tensor_list_batched(batched)


@dispatch
def compute_gauge_projection(
    cfg: GaugeConfig,
    model: "SimpleMPS",
    params: dict,
    *,
    return_info: bool = False,
) -> jax.Array | tuple[jax.Array, dict]:
    """Compute gauge projection matrix for MPS.

    Args:
        cfg: Gauge configuration.
        model: The SimpleMPS model instance.
        params: Parameter dict containing 'tensors'.
        return_info: If True, also return diagnostic info dict.

    Returns:
        Projection matrix Q, or (Q, info) if return_info=True.
    """
    tensors = params["tensors"]
    if isinstance(tensors, dict):
        tensor_keys = sorted(tensors.keys())
        tensor_list = [tensors[k] for k in tensor_keys]
    elif isinstance(tensors, (tuple, list)):
        tensor_list = list(tensors)
    else:
        raise TypeError(
            "SimpleMPS tensors must be a dict, tuple, or list of site tensors."
        )

    n_sites = len(tensor_list)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    flat_tensors = _flatten_tensor_list(tensor_list)
    if flat_params.shape != flat_tensors.shape:
        raise ValueError(
            "Gauge projection expects params to only contain the MPS tensors."
        )
    n_params = int(flat_tensors.shape[0])
    null_blocks = []

    # Build gauge null vectors for each bond (a,b pairs vectorized per bond).
    for bond in range(n_sites - 1):
        null_block = _null_vectors_for_bond(tensor_list, bond)
        null_blocks.append(null_block.T)

    # Optionally include global scale direction
    if cfg.include_global_scale:
        null_blocks.append(flat_tensors[:, None])

    if not null_blocks:
        Q = jnp.eye(n_params, dtype=flat_tensors.dtype)
        info = {
            "null_rank": 0,
            "rcond": None,
            "null_count": 0,
            "rank_T": 0,
            "n_keep": n_params,
        }
        return (Q, info) if return_info else Q

    T = jnp.concatenate(null_blocks, axis=1)
    q_full, _ = jsp.linalg.qr(T, pivoting=False, mode="full")
    n_cols = int(T.shape[1])
    n_keep = n_params - n_cols
    if n_keep <= 0:
        raise ValueError(
            "Gauge projector removed all parameters: "
            f"n_params={n_params}, n_gauge+extras={n_cols}"
        )
    Q = q_full[:, -n_keep:]

    if not return_info:
        return Q

    # Only compute rank when info is requested (avoids block_until_ready in hot path)
    if cfg.rcond is None:
        rank_t = int(jnp.linalg.matrix_rank(T))
        rcond_val = None
    else:
        rank_t = int(jnp.linalg.matrix_rank(T, rtol=cfg.rcond))
        rcond_val = float(cfg.rcond)

    info = {
        "null_rank": n_cols,
        "rcond": rcond_val,
        "null_count": n_cols,
        "rank_T": rank_t,
        "n_keep": n_keep,
    }
    return Q, info


@dispatch
def compute_gauge_projection(
    cfg: GaugeConfig,
    model: "SimplePEPS",
    params: dict,
    *,
    return_info: bool = False,
) -> jax.Array | tuple[jax.Array, dict]:
    """Compute gauge projection matrix for PEPS.

    Raises:
        NotImplementedError: PEPS gauge removal is not yet implemented.
    """
    raise NotImplementedError(
        "Gauge removal for PEPS is not yet implemented. "
        "Use gauge_config=None for PEPS models."
    )

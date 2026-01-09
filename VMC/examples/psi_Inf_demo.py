#!/usr/bin/env python3
"""Inspect finite/-inf log-amplitudes for `MPSOpen`."""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import logging

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

import netket as nk

logger = logging.getLogger(__name__)


def configure_delta_mps(mps, *, seed: int = 0):
    """Configure `MPSOpen` to support only a single product state.

    Because `MPSOpen` adds an identity-like tensor in `setup`, subtract the
    identity everywhere and re-add it only on the supported local state to
    enforce zero amplitude elsewhere.
    """
    hi = mps.hilbert

    # Dummy input to initialize parameters
    sample = jnp.ones((1, hi.size))
    variables = unfreeze(mps.init(jax.random.PRNGKey(seed), sample))

    params = variables["params"]
    dtype = params["middle_tensors"].dtype

    # Find the local-index that corresponds to physical value +1 (treated as "up").
    support_value = jnp.array([1])
    support_idx = int(hi.states_to_local_indices(support_value)[0])
    d = hi.local_size
    D = int(mps.bond_dim)

    # Boundary tensors: start from -1 to cancel the built-in "+1" bias, then
    # set the supported index back to 0 so that (+1) becomes 1 after addition.
    left = -jnp.ones_like(params["left_tensors"])
    left = left.at[support_idx, :].set(0.0)

    right = -jnp.ones_like(params["right_tensors"])
    right = right.at[support_idx, :].set(0.0)

    # Middle tensors: same idea, but cancel the identity matrix per site/local state.
    num_bulk = params["middle_tensors"].shape[0]
    minus_eye = -jnp.eye(D, dtype=dtype)
    middle = jnp.tile(minus_eye[None, None, :, :], (num_bulk, d, 1, 1))
    middle = middle.at[:, support_idx, :, :].set(jnp.zeros((D, D), dtype=dtype))

    params["left_tensors"] = left
    params["right_tensors"] = right
    params["middle_tensors"] = middle

    variables["params"] = params
    return freeze(variables)


def inspect_logpsi(mps, variables, configs: dict[str, jnp.ndarray]):
    for name, cfg in configs.items():
        logpsi = mps.apply(variables, cfg)
        psi = jnp.exp(logpsi)
        logger.info(
            "%s log(psi)=%s | isfinite=%s | psi=%s",
            name,
            logpsi,
            jnp.isfinite(logpsi),
            psi,
        )


def main():
    L = 4
    hi = nk.hilbert.Spin(s=0.5, N=L)
    mps = nk.models.tensor_networks.MPSOpen(
        hilbert=hi, bond_dim=1, param_dtype=jnp.complex128
    )
    variables = configure_delta_mps(mps)

    configs = {
        "all_up (finite)": jnp.ones((L,)),  # supported -> finite log amplitude
        "single_down_end": jnp.array([1, 1, 1, -1]),  # unsupported -> -inf
        "single_down_start": jnp.array([-1, 1, 1, 1]),  # unsupported -> -inf
    }

    inspect_logpsi(mps, variables, configs)


if __name__ == "__main__":
    main()

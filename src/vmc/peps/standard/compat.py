"""Compatibility surfaces for standard PEPS."""

from __future__ import annotations

import functools
from typing import Any

from vmc import config  # noqa: F401 - JAX config must be imported first

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from vmc.operators.local_terms import LocalHamiltonian, merge_operators
from vmc.peps.common.contraction import _build_row_mpo, _forward_with_cache
from vmc.peps.common.energy import (
    _estimate_sweep,
    _compute_all_gradients,
)
from vmc.peps.common.strategy import ContractionStrategy
from vmc.peps.grading import (
    FermionSigns,
    Grading,
    _grading_statics,
    column_prefix_parities,
)
from vmc.utils.utils import spin_to_occupancy

__all__ = [
    "graded_peps_apply",
    "peps_apply",
    "local_estimate",
    "_value",
    "_grad",
    "_value_and_grad",
]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def peps_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
) -> jax.Array:
    spins = spin_to_occupancy(sample).reshape(shape)
    amp, _ = _forward_with_cache(tensors, spins, shape, strategy)
    return amp


def _peps_apply_fwd(tensors, sample, shape, strategy):
    spins = spin_to_occupancy(sample).reshape(shape)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, strategy)
    return amp, (tensors, spins, top_envs)


def _peps_apply_bwd(shape, strategy, residuals, g):
    tensors, spins, top_envs = residuals
    n_rows, n_cols = shape
    env_grads = _compute_all_gradients(tensors, spins, shape, strategy, top_envs)
    grad_leaves = []
    for r in range(n_rows):
        for c in range(n_cols):
            grad_full = jnp.zeros_like(jnp.asarray(tensors[r][c]))
            grad_leaves.append(grad_full.at[spins[r, c]].set(g * env_grads[r][c]))
    return (
        jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(tensors), grad_leaves
        ),
        None,
    )


peps_apply.defvjp(_peps_apply_fwd, _peps_apply_bwd)


def _decorate(
    tensors: list[list[jax.Array]],
    prefix: jax.Array,
    masks: list,
    right_par: list,
) -> list[list[jax.Array]]:
    """Graded assembly rule: mask x right-leg gate sign ``(-1)^{prefix * P}``."""
    return [
        [
            jnp.asarray(tensors[r][c])
            * masks[r][c]
            * (1.0 - 2.0 * prefix[r, c] * right_par[r][c])
            for c in range(len(tensors[0]))
        ]
        for r in range(len(tensors))
    ]


def _graded_forward(
    tensors: list[list[jax.Array]],
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    grading: Grading,
) -> tuple[jax.Array, list[tuple]]:
    """Masked, gate-signed forward pass caching the top boundaries."""
    masks, right_par, _ = _grading_statics(grading, tensors)
    decorated = _decorate(
        tensors, column_prefix_parities(grading, spins), masks, right_par
    )
    return _forward_with_cache(decorated, spins, shape, strategy)


def graded_peps_apply(
    tensors: list[list[jax.Array]],
    sample: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    *,
    grading: Grading,
) -> jax.Array:
    """Amplitude of a graded PEPS sample."""
    spins = spin_to_occupancy(sample).reshape(shape)
    return _graded_forward(tensors, spins, shape, strategy, grading)[0]


def _value(
    model: Any,
    sample: jax.Array,
) -> jax.Array:
    """Compute amplitude for standard PEPS sample(s)."""
    sample = jnp.asarray(sample)
    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    if sample.ndim == 2:
        return jax.vmap(lambda s: model.apply(tensors, s, model.shape, model.strategy))(
            sample
        )
    return model.apply(tensors, sample, model.shape, model.strategy)


def _grad(
    model: Any,
    sample: jax.Array,
    *,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Compute amplitude gradient for standard PEPS sample(s)."""
    _, grad_row, p_row = _value_and_grad(model, sample, full_gradient=full_gradient)
    return grad_row, p_row


def _value_and_grad(
    model: Any,
    sample: jax.Array,
    *,
    full_gradient: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Compute amplitude and gradient for standard PEPS sample(s)."""
    sample = jnp.asarray(sample)
    if sample.ndim == 2:
        return jax.vmap(
            lambda s: _value_and_grad(model, s, full_gradient=full_gradient)
        )(sample)

    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    shape = model.shape
    bond_dim = model.bond_dim
    n_rows, n_cols = shape

    if full_gradient:
        amp, grad = jax.value_and_grad(model.apply, holomorphic=True)(
            tensors, sample, shape, model.strategy
        )
        grad_flat, _ = ravel_pytree(grad)
        return amp, grad_flat, None

    spins = spin_to_occupancy(sample).reshape(shape)
    if model.grading is not None:
        masks, right_par, _ = _grading_statics(model.grading, tensors)
        prefix = column_prefix_parities(model.grading, spins)
        tensors = _decorate(tensors, prefix, masks, right_par)
    amp, top_envs = _forward_with_cache(tensors, spins, shape, model.strategy)
    env_grads = _compute_all_gradients(tensors, spins, shape, model.strategy, top_envs)
    if model.grading is not None:
        for r in range(n_rows):
            for c in range(n_cols):
                env_grads[r][c] = env_grads[r][c] * (
                    masks[r][c][spins[r, c]]
                    * (1.0 - 2.0 * prefix[r, c] * right_par[r][c])
                )

    grad_parts, p_parts = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            grad_parts.append(env_grads[r][c].reshape(-1))
            up, down, left, right = model.site_dims(r, c, n_rows, n_cols, bond_dim)
            params_per_phys = up * down * left * right
            p_parts.append(jnp.full((params_per_phys,), spins[r, c], dtype=jnp.int8))

    return amp, jnp.concatenate(grad_parts), jnp.concatenate(p_parts)


def local_estimate(
    model: Any,
    samples: jax.Array,
    operator: LocalHamiltonian,
    amps: jax.Array,
    *,
    coeffs: jax.Array | None = None,
) -> jax.Array:
    """Compute local energy estimates for PEPS from local operator terms."""
    samples = jnp.asarray(samples)
    amps = jnp.asarray(amps)
    shape = model.shape
    bucketed_terms, coeff_structure = merge_operators(
        (operator,),
        shape,
        eval_span=type(model).eval_span,
    )
    base_coeffs = coeff_structure.build_coeffs()
    coeffs = base_coeffs if coeffs is None else base_coeffs * coeffs
    has_diag = bool(bucketed_terms.diagonal)
    has_offdiag = any(
        cell
        for row_passes in bucketed_terms.rows
        for _, cols in row_passes
        for cell in cols
    )

    if not has_diag and not has_offdiag:
        return jnp.zeros((samples.shape[0],), dtype=amps.dtype)

    if has_diag and not has_offdiag:
        phys_dim = model.phys_dim

        def diag_only(sample):
            spins = spin_to_occupancy(sample).reshape(shape)
            total = jnp.zeros((), dtype=amps.dtype)
            for term, contributions in bucketed_terms.diagonal:
                idx = jnp.asarray(0, dtype=jnp.int32)
                for row, col in term.sites:
                    idx = idx * phys_dim + spins[row, col]
                for _op_idx, coeff_idx in contributions:
                    total = total + coeffs[coeff_idx] * term.diag[idx]
            return total

        return jax.vmap(diag_only)(samples)

    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    grading = model.grading
    if grading is not None:
        masks, right_par, down_par = _grading_statics(grading, tensors)
        down_flip = [[1.0 - 2.0 * par for par in row] for row in down_par]
        right_flip = [[1.0 - 2.0 * par for par in row] for row in right_par]

    def per_sample(sample, amp):
        occupancy = spin_to_occupancy(sample)
        spins = occupancy.reshape(shape)
        if grading is None:
            decorated, env_config = tensors, None
        else:
            prefix = column_prefix_parities(grading, spins)
            parities = jnp.asarray(grading.phys_parity)[spins]
            suffix = (jnp.sum(parities, axis=0) + prefix + parities) % 2
            decorated = _decorate(tensors, prefix, masks, right_par)
            env_config = FermionSigns(prefix, suffix, down_flip, right_flip)
        _, top_envs = _forward_with_cache(decorated, spins, shape, model.strategy)

        def build_row_mpo(
            tensors: Any,
            sample: jax.Array,
            row: int,
        ) -> tuple:
            return _build_row_mpo(tensors, sample[row], row, shape[1])

        _, energies, _ = _estimate_sweep(
            decorated,
            spins,
            amp,
            top_envs,
            strategy=model.strategy,
            terms=bucketed_terms,
            build_row_mpo=build_row_mpo,
            env_config=env_config,
            coeffs=coeffs,
            collect_grads=False,
        )
        return energies[0]

    return jax.vmap(per_sample, in_axes=(0, 0))(samples, amps)

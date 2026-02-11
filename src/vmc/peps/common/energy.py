"""Common PEPS local-energy and derivative contractions."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from typing import Any

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import BucketedOperators
from vmc.peps.common.contraction import _apply_mpo_from_below, _build_row_mpo, _compute_right_envs
from vmc.peps.common.strategy import ContractionStrategy

__all__ = [
    "_compute_right_envs_2row",
    "_compute_row_pair_vertical_energy",
    "_compute_all_row_gradients",
    "_compute_all_env_grads_and_energy",
    "_compute_2site_horizontal_env",
    "_compute_single_gradient",
    "_compute_all_gradients",
]

def _compute_right_envs_2row(
    top_env: tuple,
    mpo_row0: tuple,
    mpo_row1: tuple,
    bottom_env: tuple,
    dtype,
) -> list[jax.Array]:
    """Compute right environments for 2-row contractions using direct einsum."""
    n_cols = len(mpo_row0)
    right_envs = [None] * n_cols
    right_envs[n_cols - 1] = jnp.ones((1, 1, 1, 1), dtype=dtype)
    for c in range(n_cols - 2, -1, -1):
        # Direct einsum: top @ mpo0 @ mpo1 @ bot @ right_env -> new_right_env
        # top: (a, u, b), mpo0: (l, r, u, v), mpo1: (x, y, v, w), bot: (e, w, f)
        # right_env: (b, r, y, f) -> output: (a, l, x, e)
        right_envs[c] = jnp.einsum(
            "aub,lruv,xyvw,ewf,bryf->alxe",
            top_env[c + 1], mpo_row0[c + 1], mpo_row1[c + 1], bottom_env[c + 1], right_envs[c + 1],
            optimize=[(0, 4), (0, 3), (0, 2), (0, 1)],
        )
    return right_envs


def _compute_row_pair_vertical_energy(
    top_mps: tuple,
    bottom_mps: tuple,
    mpo_row0: tuple,
    mpo_row1: tuple,
    tensors_row0: list[jax.Array],
    tensors_row1: list[jax.Array],
    spins_row0: jax.Array,
    spins_row1: jax.Array,
    terms_row: tuple[tuple[tuple[tuple[int, Any], ...], ...], ...] | list[list],
    amp: jax.Array,
    phys_dim: int,
    *,
    coeffs: jax.Array | None = None,
    right_envs_2row: list[jax.Array] | None = None,
) -> jax.Array:
    """Compute vertical 2-site energy contributions for a row pair."""
    if not any(terms_row):
        return jnp.zeros((), dtype=amp.dtype)
    n_cols = len(mpo_row0)
    dtype = mpo_row0[0].dtype
    if right_envs_2row is None:
        right_envs_2row = _compute_right_envs_2row(
            top_mps, mpo_row0, mpo_row1, bottom_mps, dtype
        )
    left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
    energy = jnp.zeros((), dtype=amp.dtype)
    for c in range(n_cols):
        col_terms = terms_row[c]
        if col_terms:
            # Direct einsum: left_env @ top @ tensor0 @ tensor1 @ bot @ right_env -> (p, q)
            # tensor0: (p, u, v, l, r), tensor1: (q, v, w, m, n)
            amps_edge = jnp.einsum(
                "almg,aub,puvlr,qvwmn,gwf,brnf->pq",
                left_env, top_mps[c], tensors_row0[c], tensors_row1[c], bottom_mps[c], right_envs_2row[c],
                optimize=[(0, 1), (2, 3), (0, 2), (1, 2), (0, 1)],
            )
            spin0 = spins_row0[c]
            spin1 = spins_row1[c]
            col_idx = spin0 * phys_dim + spin1
            amps_flat = amps_edge.reshape(-1)
            for term_idx, term in col_terms:
                coeff = 1.0 if coeffs is None else coeffs[term_idx]
                energy = energy + coeff * jnp.dot(term.op[:, col_idx], amps_flat) / amp
        # Direct einsum for left_env_2row update
        left_env = jnp.einsum(
            "alxe,aub,lruv,xyvw,ewf->bryf",
            left_env, top_mps[c], mpo_row0[c], mpo_row1[c], bottom_mps[c],
            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
        )
    return energy


def _compute_all_row_gradients(
    top_mps: tuple,
    bottom_mps: tuple,
    mpo: tuple,
) -> list[jax.Array]:
    """Compute gradients for all tensors in a row using direct einsum."""
    n_cols = len(mpo)
    dtype = mpo[0].dtype
    right_envs = _compute_right_envs(top_mps, mpo, bottom_mps, dtype)

    env_grads = []
    left_env = jnp.ones((1, 1, 1), dtype=dtype)
    for c in range(n_cols):
        env_grads.append(
            _compute_single_gradient(left_env, right_envs[c], top_mps[c], bottom_mps[c])
        )
        # Direct einsum for left_env update: left_env @ top @ mpo @ bot -> new_left_env
        left_env = jnp.einsum(
            "ace,aub,cduv,evf->bdf",
            left_env, top_mps[c], mpo[c], bottom_mps[c],
            optimize=[(0, 1), (0, 2), (0, 1)],
        )
    return env_grads


def _compute_all_env_grads_and_energy(
    tensors: Any,
    spins: jax.Array,
    amp: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    top_envs: list[tuple],
    *,
    terms: BucketedOperators,
    coeffs: jax.Array | None = None,
    collect_grads: bool = True,
) -> tuple[list[list[jax.Array]], jax.Array, list[tuple]]:
    """Backward pass: use cached top_envs, build and cache bottom_envs."""
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype
    phys_dim = int(jnp.asarray(tensors[0][0]).shape[0])

    env_grads = (
        [[None for _ in range(n_cols)] for _ in range(n_rows)]
        if collect_grads
        else []
    )
    bottom_envs_cache = [None] * n_rows
    energy = jnp.zeros((), dtype=amp.dtype)
    diagonal_terms = terms.diagonal
    span_11_terms = terms.span_11
    span_12_terms = terms.span_12
    span_21_terms = terms.span_21

    # Diagonal terms
    for term_idx, term in diagonal_terms:
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + spins[row, col]
        coeff = 1.0 if coeffs is None else coeffs[term_idx]
        energy = energy + coeff * term.diag[idx]

    # Backward pass: bottom → top
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    empty_row_terms = tuple(() for _ in range(n_cols))
    next_row_mpo = None
    for row in range(n_rows - 1, -1, -1):
        bottom_envs_cache[row] = bottom_env
        top_env = top_envs[row]
        mpo = _build_row_mpo(tensors, spins[row], row, n_cols)
        right_envs = _compute_right_envs(top_env, mpo, bottom_env, dtype)
        vertical_row_terms = span_21_terms[row] if row < n_rows - 1 else empty_row_terms
        has_vertical_terms = row < n_rows - 1 and any(vertical_row_terms)
        if has_vertical_terms:
            bottom_env_next = bottom_envs_cache[row + 1]
            right_envs_2row = _compute_right_envs_2row(
                top_env, mpo, next_row_mpo, bottom_env_next, dtype
            )
            left_env_2row = jnp.ones((1, 1, 1, 1), dtype=dtype)
        else:
            vertical_row_terms = empty_row_terms
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        for c in range(n_cols):
            site_terms = span_11_terms[row][c]
            horizontal_terms = span_12_terms[row][c] if c < n_cols - 1 else ()
            vertical_terms = vertical_row_terms[c]
            need_env_grad = collect_grads or bool(site_terms)
            amps_site = None
            if need_env_grad:
                env_grad = _compute_single_gradient(
                    left_env, right_envs[c], top_env[c], bottom_env[c]
                )
                if collect_grads:
                    env_grads[row][c] = env_grad
                if site_terms:
                    amps_site = jnp.einsum("pudlr,udlr->p", tensors[row][c], env_grad)
            amps_horizontal = None
            if horizontal_terms:
                amps_horizontal = jnp.einsum(
                    "ace,aub,edf,pudcr,qvwrx,bvg,fwi,gxi->pq",
                    left_env,
                    top_env[c],
                    bottom_env[c],
                    tensors[row][c],
                    tensors[row][c + 1],
                    top_env[c + 1],
                    bottom_env[c + 1],
                    right_envs[c + 1],
                    optimize=[(0, 1), (1, 6), (0, 5), (1, 3), (1, 2), (1, 2), (0, 1)],
                )
            amps_vertical = None
            if vertical_terms:
                amps_vertical = jnp.einsum(
                    "almg,aub,puvlr,qvwmn,gwf,brnf->pq",
                    left_env_2row,
                    top_env[c],
                    tensors[row][c],
                    tensors[row + 1][c],
                    bottom_env_next[c],
                    right_envs_2row[c],
                    optimize=[(0, 1), (2, 3), (0, 2), (1, 2), (0, 1)],
                )
            for term_idx, term in site_terms:
                coeff = 1.0 if coeffs is None else coeffs[term_idx]
                spin_idx = spins[row, c]
                energy = energy + coeff * jnp.dot(term.op[:, spin_idx], amps_site) / amp
            for term_idx, term in horizontal_terms:
                coeff = 1.0 if coeffs is None else coeffs[term_idx]
                spin0 = spins[row, c]
                spin1 = spins[row, c + 1]
                col_idx = spin0 * phys_dim + spin1
                energy = energy + coeff * jnp.dot(
                    term.op[:, col_idx], amps_horizontal.reshape(-1)
                ) / amp
            for term_idx, term in vertical_terms:
                coeff = 1.0 if coeffs is None else coeffs[term_idx]
                spin0 = spins[row, c]
                spin1 = spins[row + 1, c]
                col_idx = spin0 * phys_dim + spin1
                energy = energy + coeff * jnp.dot(
                    term.op[:, col_idx], amps_vertical.reshape(-1)
                ) / amp
            left_env = jnp.einsum(
                "ace,aub,cduv,evf->bdf",
                left_env, top_env[c], mpo[c], bottom_env[c],
                optimize=[(0, 1), (0, 2), (0, 1)],
            )
            if has_vertical_terms:
                left_env_2row = jnp.einsum(
                    "alxe,aub,lruv,xyvw,ewf->bryf",
                    left_env_2row,
                    top_env[c],
                    mpo[c],
                    next_row_mpo[c],
                    bottom_env_next[c],
                    optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
                )
        bottom_env = _apply_mpo_from_below(bottom_env, mpo, strategy)
        next_row_mpo = mpo

    return env_grads, energy, bottom_envs_cache


def _compute_2site_horizontal_env(
    left_env: jax.Array,
    right_env: jax.Array,
    top0: jax.Array,
    bot0: jax.Array,
    top1: jax.Array,
    bot1: jax.Array,
) -> jax.Array:
    """Compute 2-site environment for horizontal edge (c, c+1).

    Index conventions:
        left_env: (tL, mL, bL) - top/mpo/bottom left bonds
        right_env: (tR, mR, bR) - top/mpo/bottom right bonds
        top0/top1: (left, up, right) - boundary boundary state
        bot0/bot1: (left, down, right) - boundary boundary state

    Returns tensor with shape (up0, down0, mL, up1, down1, mR).
    """
    env = jnp.einsum(
        "ace,aub,edf,bvg,ghi,fwi->cudvhw",
        left_env,
        top0,
        bot0,
        top1,
        right_env,
        bot1,
        optimize=[(0, 1), (0, 1), (1, 2), (1, 2), (0, 1)],
    )
    # Transpose to (up0, down0, mL, up1, down1, mR)
    return jnp.transpose(env, (1, 2, 0, 3, 5, 4))


def _compute_single_gradient(
    left_env: jax.Array,
    right_env: jax.Array,
    top_tensor: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Compute gradient for a single tensor given left/right environments.

    Returns gradient tensor with shape (up, down, mL, mR).
    """
    grad = jnp.einsum(
        "ace,aub,evf,bdf->cuvd", left_env, top_tensor, bot_tensor, right_env,
        optimize=[(0, 1), (0, 1), (0, 1)],
    )
    return jnp.transpose(grad, (1, 2, 0, 3))


def _compute_all_gradients(
    tensors: Any,
    spins: jax.Array,
    shape: tuple[int, int],
    strategy: ContractionStrategy,
    top_envs: list[tuple],
    *,
    cache_bottom_envs: bool = False,
    row_mpos: list[tuple] | None = None,
) -> list[list[jax.Array]] | tuple[list[list[jax.Array]], list[tuple]]:
    """Compute gradients for all PEPS tensors using cached top environments."""
    n_rows, n_cols = shape
    dtype = jnp.asarray(tensors[0][0]).dtype

    grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    bottom_envs_cached = [None] * n_rows if cache_bottom_envs else None
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

    for row in range(n_rows - 1, -1, -1):
        if cache_bottom_envs:
            bottom_envs_cached[row] = bottom_env
        mpo = row_mpos[row] if row_mpos else _build_row_mpo(tensors, spins[row], row, n_cols)
        grads[row] = _compute_all_row_gradients(top_envs[row], bottom_env, mpo)
        bottom_env = _apply_mpo_from_below(bottom_env, mpo, strategy)

    return (grads, bottom_envs_cached) if cache_bottom_envs else grads

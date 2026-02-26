"""Common PEPS local-energy and derivative contractions."""
from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.operators.local_terms import (
    BucketedOperators,
    HorizontalTwoSiteOperator,
    OneSiteOperator,
    VerticalTwoSiteOperator,
)
from vmc.peps.common.contraction import _apply_mpo_from_below, _build_row_mpo, _compute_right_envs
from vmc.peps.common.strategy import ContractionStrategy

__all__ = [
    "RowEnvs",
    "TwoRowEnvs",
    "_eval_term",
    "_update_left_env_1row",
    "_update_left_env_2row",
    "_compute_right_envs_2row",
    "_compute_all_row_gradients",
    "_compute_all_env_grads_and_energy",
    "_compute_single_gradient",
    "_compute_all_gradients",
]


class RowEnvs(NamedTuple):
    """1-row environment context for dr=1 evaluation."""
    left_env: jax.Array
    right_envs: list
    top_env: tuple
    bottom_env: tuple
    env_grad: jax.Array
    row_tensors: list


class TwoRowEnvs(NamedTuple):
    """2-row environment context for dr=2 evaluation."""
    left_env: jax.Array
    right_envs: list
    top_env: tuple
    bottom_env_next: tuple
    row_tensors: list
    row_tensors_next: list

def _update_left_env_1row(left_env, top, mpo, bottom):
    """Advance 1-row left environment by one column."""
    return jnp.einsum(
        "ace,aub,cduv,evf->bdf",
        left_env, top, mpo, bottom,
        optimize=[(0, 1), (0, 2), (0, 1)],
    )


def _update_left_env_2row(left_env, top, mpo0, mpo1, bottom):
    """Advance 2-row left environment by one column."""
    return jnp.einsum(
        "alxe,aub,lruv,xyvw,ewf->bryf",
        left_env, top, mpo0, mpo1, bottom,
        optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
    )


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
        left_env = _update_left_env_1row(left_env, top_mps[c], mpo[c], bottom_mps[c])
    return env_grads


@dispatch
def _eval_term(
    term: OneSiteOperator,
    envs: RowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    amps = jnp.einsum("pudlr,udlr->p", envs.row_tensors[col], envs.env_grad)
    return jnp.dot(term.op[:, spins[row, col]], amps)


@_eval_term.dispatch
def _eval_term(
    term: HorizontalTwoSiteOperator,
    envs: RowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    amps = jnp.einsum(
        "ace,aub,edf,pudcr,qvwrx,bvg,fwi,gxi->pq",
        envs.left_env,
        envs.top_env[col],
        envs.bottom_env[col],
        envs.row_tensors[col],
        envs.row_tensors[col + 1],
        envs.top_env[col + 1],
        envs.bottom_env[col + 1],
        envs.right_envs[col + 1],
        optimize=[
            (0, 1), (1, 6), (0, 5), (1, 3), (1, 2), (1, 2), (0, 1),
        ],
    )
    s0, s1 = spins[row, col], spins[row, col + 1]
    return jnp.dot(term.op[:, s0 * phys_dim + s1], amps.reshape(-1))


@_eval_term.dispatch
def _eval_term(
    term: VerticalTwoSiteOperator,
    envs: TwoRowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    amps = jnp.einsum(
        "almg,aub,puvlr,qvwmn,gwf,brnf->pq",
        envs.left_env,
        envs.top_env[col],
        envs.row_tensors[col],
        envs.row_tensors_next[col],
        envs.bottom_env_next[col],
        envs.right_envs[col],
        optimize=[(0, 1), (2, 3), (0, 2), (1, 2), (0, 1)],
    )
    s0, s1 = spins[row, col], spins[row + 1, col]
    return jnp.dot(term.op[:, s0 * phys_dim + s1], amps.reshape(-1))


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

    # Diagonal terms
    for term_idx, term in terms.diagonal:
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + spins[row, col]
        coeff = 1.0 if coeffs is None else coeffs[term_idx]
        energy = energy + coeff * term.diag[idx]

    # Backward pass: bottom → top
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    empty_cols = tuple(() for _ in range(n_cols))
    next_row_mpo = None
    for row in range(n_rows - 1, -1, -1):
        bottom_envs_cache[row] = bottom_env
        top_env = top_envs[row]
        mpo = _build_row_mpo(tensors, spins[row], row, n_cols)
        row_passes = terms.rows[row]
        if collect_grads and not any(dr == 1 for dr, _ in row_passes):
            row_passes = ((1, empty_cols),) + row_passes

        def _eval_dr1(
            energy_acc: jax.Array,
            col_terms: tuple[tuple[tuple[int, object, tuple[int, int]], ...], ...],
        ) -> jax.Array:
            right_envs = _compute_right_envs(top_env, mpo, bottom_env, dtype)
            left_env = jnp.ones((1, 1, 1), dtype=dtype)
            for col in range(n_cols):
                env_grad = _compute_single_gradient(
                    left_env, right_envs[col], top_env[col], bottom_env[col]
                )
                if collect_grads:
                    env_grads[row][col] = env_grad
                envs = RowEnvs(left_env, right_envs, top_env, bottom_env, env_grad, tensors[row])
                for term_idx, term, span in col_terms[col]:
                    coeff = 1.0 if coeffs is None else coeffs[term_idx]
                    energy_acc = energy_acc + coeff * _eval_term(
                        term, envs, tensors, row, col, spins, phys_dim,
                    ) / amp
                left_env = _update_left_env_1row(left_env, top_env[col], mpo[col], bottom_env[col])
            return energy_acc

        def _eval_dr2(
            energy_acc: jax.Array,
            col_terms: tuple[tuple[tuple[int, object, tuple[int, int]], ...], ...],
        ) -> jax.Array:
            if row >= n_rows - 1:
                return energy_acc
            if next_row_mpo is None:
                raise NotImplementedError("Missing next-row MPO for dr=2 evaluation.")
            bottom_env_next = bottom_envs_cache[row + 1]
            right_envs_2row = _compute_right_envs_2row(
                top_env, mpo, next_row_mpo, bottom_env_next, dtype
            )
            left_env_2row = jnp.ones((1, 1, 1, 1), dtype=dtype)
            for col in range(n_cols):
                envs = TwoRowEnvs(
                    left_env_2row, right_envs_2row, top_env, bottom_env_next,
                    tensors[row], tensors[row + 1],
                )
                for term_idx, term, span in col_terms[col]:
                    coeff = 1.0 if coeffs is None else coeffs[term_idx]
                    energy_acc = energy_acc + coeff * _eval_term(
                        term, envs, tensors, row, col, spins, phys_dim,
                    ) / amp
                left_env_2row = _update_left_env_2row(left_env_2row, top_env[col], mpo[col], next_row_mpo[col], bottom_env_next[col])
            return energy_acc

        pass_evaluators = {
            1: _eval_dr1,
            2: _eval_dr2,
        }
        for dr, col_terms in row_passes:
            evaluator = pass_evaluators.get(dr)
            if evaluator is None:
                raise NotImplementedError(
                    f"dr={dr} transition evaluation is not implemented."
                )
            energy = evaluator(energy, col_terms)
        bottom_env = _apply_mpo_from_below(bottom_env, mpo, strategy)
        next_row_mpo = mpo

    return env_grads, energy, bottom_envs_cache


def _compute_single_gradient(
    left_env: jax.Array,
    right_env: jax.Array,
    top_tensor: jax.Array,
    bot_tensor: jax.Array,
) -> jax.Array:
    """Compute gradient for a single tensor given left/right environments.

    Returns gradient tensor with shape (up, down, mL, mR).
    """
    return jnp.einsum(
        "ace,aub,evf,bdf->uvcd", left_env, top_tensor, bot_tensor, right_env,
        optimize=[(0, 1), (0, 1), (0, 1)],
    )


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

"""DP-weighted initial-sample generation for non-Abelian GI-PEPS."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from plum import dispatch

from vmc.peps.non_abelian_gi.contraction import flatten_matter_sample, flatten_sample


# ---------- matter spec hierarchy (typed dispatch target) ----------


class MatterSpec(ABC):
    """Per-site matter Hilbert-space description."""


@dataclass(frozen=True)
class Vacuum(MatterSpec):
    """Pure-gauge: matter is fixed to the singlet vacuum at every site."""


@dataclass(frozen=True)
class ConservedMatter(MatterSpec):
    """Matter with conserved total particle number."""

    irreps: tuple[int, ...]
    numbers: tuple[int, ...]
    particle_number: int


# ---------- static metadata ----------


@dataclass(frozen=True)
class InitTables:
    """Precomputed static tables for DP-weighted initial sampling."""

    candidates: jax.Array  # (n_rows, n_cols, n_irreps, n_irreps, phys_dim, max_cand)
    n_candidates: jax.Array  # (n_rows, n_cols, n_irreps, n_irreps, phys_dim)
    j_r_by_block: jax.Array  # (n_rows, n_cols, max_blocks)
    j_d_by_block: jax.Array  # (n_rows, n_cols, max_blocks)
    iota_by_block: jax.Array  # (n_rows, n_cols, max_blocks)
    top_irrep_at: jax.Array  # (n_top_states, n_cols)
    replace_top_at: jax.Array  # (n_top_states, n_cols, n_irreps)
    partitions: jax.Array  # (n_partitions, n_sites); one all-vacuum row for Vacuum
    partition_multiplicities: jax.Array  # (n_partitions,)
    vacuum_Z: jax.Array
    n_irreps: int
    n_top_states: int
    shape: tuple[int, int]


def build_init_tables(tables: Any, spec: MatterSpec) -> InitTables:
    """Build static metadata closed over by the JIT'd init kernel."""
    n_rows, n_cols = tables.shape
    n_irreps = len(tables.group.irreps())
    phys_dim = int(tables.phys_dim)
    n_top_states = int(n_irreps**n_cols)

    j_l_b = np.asarray(tables.j_l_by_block)
    j_u_b = np.asarray(tables.j_u_by_block)
    matter_b = np.asarray(tables.matter_state_by_block)

    grouped: dict[tuple[int, int, int, int, int], list[int]] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            for b in range(tables.n_blocks(r, c)):
                key = (
                    r,
                    c,
                    int(j_l_b[r, c, b]),
                    int(j_u_b[r, c, b]),
                    int(matter_b[r, c, b]),
                )
                grouped.setdefault(key, []).append(b)
    max_cand = max((len(v) for v in grouped.values()), default=1)
    candidates = np.full(
        (n_rows, n_cols, n_irreps, n_irreps, phys_dim, max_cand), -1, dtype=np.int32
    )
    n_candidates = np.zeros(
        (n_rows, n_cols, n_irreps, n_irreps, phys_dim), dtype=np.int32
    )
    for (r, c, a, u, m), bids in grouped.items():
        n_candidates[r, c, a, u, m] = len(bids)
        candidates[r, c, a, u, m, : len(bids)] = bids

    strides = n_irreps ** np.arange(n_cols)  # (n_cols,)
    indices = np.arange(n_top_states)  # (n_top,)
    top_irrep_at = (indices[:, None] // strides) % n_irreps  # (n_top, n_cols)
    bases = indices[:, None] - top_irrep_at * strides  # (n_top, n_cols)
    replace_top_at = (
        bases[..., None] + np.arange(n_irreps) * strides[:, None]
    )  # (n_top, n_cols, n_irreps)
    partitions = _partitions(spec, n_rows * n_cols)

    init = InitTables(
        candidates=jnp.asarray(candidates),
        n_candidates=jnp.asarray(n_candidates),
        j_r_by_block=jnp.asarray(tables.j_r_by_block),
        j_d_by_block=jnp.asarray(tables.j_d_by_block),
        iota_by_block=jnp.asarray(tables.iota_by_block),
        top_irrep_at=jnp.asarray(top_irrep_at.astype(np.int32)),
        replace_top_at=jnp.asarray(replace_top_at.astype(np.int32)),
        partitions=jnp.asarray(partitions),
        partition_multiplicities=jnp.asarray(_partition_multiplicities(partitions)),
        vacuum_Z=jnp.empty((0,), dtype=jnp.float64),
        n_irreps=n_irreps,
        n_top_states=n_top_states,
        shape=(n_rows, n_cols),
    )
    if isinstance(spec, Vacuum):
        return replace(
            init,
            vacuum_Z=completion_counts(
                jnp.zeros(init.shape, dtype=jnp.int32),
                init,
            ),
        )
    return init


@dispatch
def _partitions(spec: Vacuum, n_sites: int) -> np.ndarray:
    del spec
    return np.zeros((1, n_sites), dtype=np.int32)


@_partitions.dispatch
def _partitions(spec: ConservedMatter, n_sites: int) -> np.ndarray:
    rows = [
        _row_from_counts(state_counts, n_sites)
        for state_counts in _enumerate_state_counts(
            spec.numbers, spec.particle_number, n_sites
        )
    ]
    if not rows:
        raise ValueError(
            f"No matter assignment satisfies particle_number={spec.particle_number}"
            f" with numbers={spec.numbers} on {n_sites} sites."
        )
    return np.stack(rows, axis=0)


def _partition_multiplicities(partitions: np.ndarray) -> np.ndarray:
    n_sites = partitions.shape[1]
    return np.asarray(
        [
            math.factorial(n_sites)
            // math.prod(math.factorial(int(count)) for count in np.bincount(row))
            for row in partitions
        ],
        dtype=np.float64,
    )


def _row_from_counts(state_counts: tuple[int, ...], n_sites: int) -> np.ndarray:
    row = np.zeros(n_sites, dtype=np.int32)
    cursor = 0
    for state, count in enumerate(state_counts):
        row[cursor : cursor + count] = state
        cursor += count
    return row


def _enumerate_state_counts(
    numbers: tuple[int, ...],
    particle_number: int,
    n_sites: int,
) -> list[tuple[int, ...]]:
    """All ``(c_0, ..., c_{S-1})`` with ``Σ c_s = n_sites`` and ``Σ c_s n_s = particle_number``."""
    out: list[tuple[int, ...]] = []

    def recurse(
        state: int, sites_left: int, particles_left: int, prefix: tuple[int, ...]
    ):
        if state == len(numbers):
            if sites_left == 0 and particles_left == 0:
                out.append(prefix)
            return
        n_s = numbers[state]
        cap = sites_left if n_s == 0 else min(sites_left, particles_left // n_s)
        for count in range(cap + 1):
            recurse(
                state + 1,
                sites_left - count,
                particles_left - count * n_s,
                prefix + (count,),
            )

    recurse(0, n_sites, particle_number, ())
    return out


# ---------- backward DP ----------


def completion_counts(matter: jax.Array, init: InitTables) -> jax.Array:
    """Z[r, c, j_l, top_idx] = number of valid completions from cell (r, c)."""
    n_rows, n_cols = init.shape
    n_irreps = init.n_irreps
    n_top_states = init.n_top_states
    matter_flat = matter.reshape(-1)

    Z_after = jnp.zeros((n_irreps, n_top_states), dtype=jnp.float64)
    Z_after = Z_after.at[0, 0].set(1.0)

    def cell_step(Z_next, idx):
        # idx counts cells from the last to the first. Advanced-index-in-middle
        # rule below leaves dim order as (top, j_l, cand); we transpose the final
        # Z_here to the (j_l, top) convention used by ``Z_next``.
        flat = (n_rows * n_cols - 1) - idx
        r = flat // n_cols
        c = flat % n_cols
        m = matter_flat[flat]

        j_us = init.top_irrep_at[:, c]  # (n_top,)
        n_cand = init.n_candidates[r, c, :, j_us, m]  # (n_top, n_irreps)
        cand = init.candidates[r, c, :, j_us, m, :]  # (n_top, n_irreps, max_cand)
        cand_safe = jnp.maximum(cand, 0)
        new_j_l = init.j_r_by_block[r, c, cand_safe]  # (n_top, n_irreps, max_cand)
        new_j_d = init.j_d_by_block[r, c, cand_safe]  # (n_top, n_irreps, max_cand)
        new_top = init.replace_top_at[
            jnp.arange(n_top_states)[:, None, None], c, new_j_d
        ]  # (n_top, n_irreps, max_cand)
        contributions = Z_next[new_j_l, new_top]  # (n_top, n_irreps, max_cand)
        valid = (jnp.arange(cand.shape[-1]) < n_cand[..., None]).astype(Z_next.dtype)
        Z_here = jnp.sum(valid * contributions, axis=-1).T  # (n_irreps, n_top)

        # End-of-row (we are about to step out of row r in reverse): j_l for the
        # next-processed cell (last column of row r-1) must see Z[r, 0, j_l=0, top]
        # because the next row begins with j_l = 0 (left boundary).
        end_of_row = c == 0
        Z_carry = jnp.where(
            end_of_row,
            jnp.broadcast_to(Z_here[0:1], Z_here.shape),
            Z_here,
        )
        return Z_carry, Z_here

    _, Z_per_cell = jax.lax.scan(cell_step, Z_after, jnp.arange(n_rows * n_cols))
    # Z_per_cell shape: (n_rows * n_cols, n_irreps, n_top_states), in reverse order.
    Z = jnp.flip(Z_per_cell, axis=0).reshape(n_rows, n_cols, n_irreps, n_top_states)
    return Z


# ---------- forward sampling ----------


def walk_blocks(
    key: jax.Array,
    matter: jax.Array,
    Z: jax.Array,
    init: InitTables,
) -> jax.Array:
    """Sample one block per site, weighted by completion counts."""
    n_rows, n_cols = init.shape
    n_sites = n_rows * n_cols
    matter_flat = matter.reshape(-1)
    keys = jax.random.split(key, n_sites)

    def step(carry, inputs):
        left_irrep, top_idx = carry
        flat, k = inputs
        r = flat // n_cols
        c = flat % n_cols
        m = matter_flat[flat]

        j_l_used = jnp.where(c == 0, jnp.int32(0), left_irrep)
        j_u = init.top_irrep_at[top_idx, c]
        n_cand = init.n_candidates[r, c, j_l_used, j_u, m]
        cand = init.candidates[r, c, j_l_used, j_u, m]  # (max_cand,)
        cand_safe = jnp.maximum(cand, 0)
        new_j_l = init.j_r_by_block[r, c, cand_safe]  # (max_cand,)
        new_j_d = init.j_d_by_block[r, c, cand_safe]  # (max_cand,)
        new_top = init.replace_top_at[top_idx, c, new_j_d]  # (max_cand,)

        # Future weight: continue within row, or roll to next row at end.
        next_c = c + 1
        within_row = next_c < n_cols
        z_within = Z[r, jnp.minimum(next_c, n_cols - 1), new_j_l, new_top]
        next_r = r + 1
        below = jnp.where(
            next_r < n_rows,
            Z[jnp.minimum(next_r, n_rows - 1), 0, 0, new_top],
            (new_top == 0).astype(Z.dtype),
        )
        future = jnp.where(within_row, z_within, below)
        valid = (jnp.arange(cand.shape[-1]) < n_cand).astype(Z.dtype)
        weights = valid * future

        idx_in_cand = jax.random.choice(k, cand.shape[-1], p=weights / weights.sum())
        block_id = cand_safe[idx_in_cand]
        return (new_j_l[idx_in_cand], new_top[idx_in_cand]), block_id

    _, block_ids = jax.lax.scan(
        step,
        (jnp.int32(0), jnp.int32(0)),
        (jnp.arange(n_sites), keys),
    )
    return block_ids.reshape(n_rows, n_cols)


# ---------- top-level entry (typed dispatch on spec) ----------


def _sample_fields_from_counts(
    key: jax.Array,
    matter: jax.Array,
    Z: jax.Array,
    init: InitTables,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    block_ids = walk_blocks(key, matter, Z, init)
    n_rows, n_cols = init.shape

    def take(table: jax.Array) -> jax.Array:
        return jnp.take_along_axis(table, block_ids[..., None], axis=-1).squeeze(-1)

    j_r = take(init.j_r_by_block)
    j_d = take(init.j_d_by_block)
    iotas = take(init.iota_by_block).astype(jnp.int32)
    return (
        matter,
        j_r[:, : n_cols - 1].astype(jnp.int32),
        j_d[: n_rows - 1, :].astype(jnp.int32),
        iotas,
    )


@dispatch
def sample_initial(
    key: jax.Array, spec: ConservedMatter, init: InitTables
) -> jax.Array:
    del spec
    key_m, key_walk = jax.random.split(key)
    p_key, perm_key = jax.random.split(key_m)
    p_idx = jax.random.choice(
        p_key,
        init.partitions.shape[0],
        p=init.partition_multiplicities / jnp.sum(init.partition_multiplicities),
    )
    matter = jax.random.permutation(perm_key, init.partitions[p_idx]).reshape(
        init.shape
    )
    return flatten_matter_sample(
        *_sample_fields_from_counts(
            key_walk,
            matter,
            completion_counts(matter, init),
            init,
        )
    )


@sample_initial.dispatch
def sample_initial(key: jax.Array, spec: Vacuum, init: InitTables) -> jax.Array:
    del spec
    _, h, v, iotas = _sample_fields_from_counts(
        key,
        jnp.zeros(init.shape, dtype=jnp.int32),
        init.vacuum_Z,
        init,
    )
    return flatten_sample(h, v, iotas)

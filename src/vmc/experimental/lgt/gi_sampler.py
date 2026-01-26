"""GI-PEPS sequential sampler with integrated energy/gradients."""
from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from plum import dispatch

from vmc.experimental.lgt.gi_local_terms import GILocalHamiltonian, LinkDiagonalTerm
from vmc.experimental.lgt.gi_peps import (
    GIPEPS,
    GIPEPSConfig,
    _link_value_or_zero,
)
from vmc.models.peps import (
    _apply_mpo_from_below,
    _compute_all_row_gradients,
    _contract_bottom,
    _contract_column_transfer_2row,
    _contract_left_partial_2row,
    _compute_right_envs_2row,
)
from vmc.samplers.sequential import _metropolis_ratio, _sample_counts, _trim_samples


def _assemble_site(
    tensors: list[list[jax.Array]],
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    r: int,
    c: int,
) -> jax.Array:
    k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
    k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
    k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
    k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
    return tensors[r][c][:, k_u, :, k_d, :, k_l, :, k_r, :]


def _build_row_mpos_with_sites(
    eff_tensors: list[list[jax.Array]], sites: jax.Array
) -> list[tuple]:
    return [
        tuple(jnp.transpose(t[sites[r, c]], (2, 3, 0, 1)) for c, t in enumerate(row))
        for r, row in enumerate(eff_tensors)
    ]


def _bottom_envs(row_mpos: list[tuple], strategy: Any) -> list[tuple]:
    n_rows = len(row_mpos)
    bottom_envs = [None] * n_rows
    n_cols = len(row_mpos[0])
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=row_mpos[0][0].dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        bottom_envs[row] = bottom_env
        bottom_env = _apply_mpo_from_below(bottom_env, row_mpos[row], strategy)
    return bottom_envs


def _compute_all_gradients_from_row_mpos(
    row_mpos: list[tuple],
    shape: tuple[int, int],
    strategy: Any,
    top_envs: list[tuple],
) -> list[list[jax.Array]]:
    n_rows, n_cols = shape
    grads = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=row_mpos[0][0].dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        grads[row] = _compute_all_row_gradients(top_envs[row], bottom_env, row_mpos[row])
        bottom_env = _apply_mpo_from_below(bottom_env, row_mpos[row], strategy)
    return grads


def _local_pair_amp(
    left_env: jax.Array,
    transfer0: jax.Array,
    transfer1: jax.Array,
    right_env: jax.Array,
) -> jax.Array:
    tmp = _contract_left_partial_2row(left_env, transfer0)
    tmp = _contract_left_partial_2row(tmp, transfer1)
    return jnp.einsum("aceg,aceg->", tmp, right_env)


def _plaquette_flip(
    h_links: jax.Array,
    v_links: jax.Array,
    r: int,
    c: int,
    *,
    delta: int,
    N: int,
) -> tuple[jax.Array, jax.Array]:
    h_links = h_links.at[r, c].set((h_links[r, c] + delta) % N)
    h_links = h_links.at[r + 1, c].set((h_links[r + 1, c] - delta) % N)
    v_links = v_links.at[r, c].set((v_links[r, c] - delta) % N)
    v_links = v_links.at[r, c + 1].set((v_links[r, c + 1] + delta) % N)
    return h_links, v_links


def _update_row_mpo_for_site(
    row_mpo: tuple, col: int, tensor: jax.Array, phys_index: jax.Array
) -> tuple:
    row_list = list(row_mpo)
    row_list[col] = jnp.transpose(tensor[phys_index], (2, 3, 0, 1))
    return tuple(row_list)


def _updated_transfers_for_plaquette(
    tensors: list[list[jax.Array]],
    row_mpos: list[tuple],
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
) -> tuple[jax.Array, jax.Array, tuple, tuple, list[list[jax.Array]]]:
    eff00 = _assemble_site(tensors, h_links, v_links, config, r, c)
    eff01 = _assemble_site(tensors, h_links, v_links, config, r, c + 1)
    eff10 = _assemble_site(tensors, h_links, v_links, config, r + 1, c)
    eff11 = _assemble_site(tensors, h_links, v_links, config, r + 1, c + 1)

    row_mpo0 = _update_row_mpo_for_site(row_mpos[r], c, eff00, sites[r, c])
    row_mpo0 = _update_row_mpo_for_site(row_mpo0, c + 1, eff01, sites[r, c + 1])
    row_mpo1 = _update_row_mpo_for_site(row_mpos[r + 1], c, eff10, sites[r + 1, c])
    row_mpo1 = _update_row_mpo_for_site(row_mpo1, c + 1, eff11, sites[r + 1, c + 1])

    transfer0 = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    transfer1 = _contract_column_transfer_2row(
        top_env[c + 1], row_mpo0[c + 1], row_mpo1[c + 1], bottom_env[c + 1]
    )

    eff_updates = [[None, None], [None, None]]
    eff_updates[0][0] = eff00
    eff_updates[0][1] = eff01
    eff_updates[1][0] = eff10
    eff_updates[1][1] = eff11
    return transfer0, transfer1, row_mpo0, row_mpo1, eff_updates


def _updated_transfers_for_row_update(
    tensors: list[list[jax.Array]],
    row_mpos: list[tuple],
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
    row_index: int,
) -> tuple[jax.Array, jax.Array, tuple, tuple, list[tuple[int, int, jax.Array]]]:
    row_mpo0 = row_mpos[r]
    row_mpo1 = row_mpos[r + 1]
    updates = []
    if row_index == r:
        eff0 = _assemble_site(tensors, h_links, v_links, config, r, c)
        eff1 = _assemble_site(tensors, h_links, v_links, config, r, c + 1)
        row_mpo0 = _update_row_mpo_for_site(row_mpo0, c, eff0, sites[r, c])
        row_mpo0 = _update_row_mpo_for_site(row_mpo0, c + 1, eff1, sites[r, c + 1])
        updates.append((r, c, eff0))
        updates.append((r, c + 1, eff1))
    else:
        eff0 = _assemble_site(tensors, h_links, v_links, config, r + 1, c)
        eff1 = _assemble_site(tensors, h_links, v_links, config, r + 1, c + 1)
        row_mpo1 = _update_row_mpo_for_site(row_mpo1, c, eff0, sites[r + 1, c])
        row_mpo1 = _update_row_mpo_for_site(row_mpo1, c + 1, eff1, sites[r + 1, c + 1])
        updates.append((r + 1, c, eff0))
        updates.append((r + 1, c + 1, eff1))

    transfer0 = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    transfer1 = _contract_column_transfer_2row(
        top_env[c + 1], row_mpo0[c + 1], row_mpo1[c + 1], bottom_env[c + 1]
    )
    return transfer0, transfer1, row_mpo0, row_mpo1, updates


def _updated_transfer_for_column(
    tensors: list[list[jax.Array]],
    row_mpos: list[tuple],
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
) -> tuple[jax.Array, tuple, tuple, list[tuple[int, int, jax.Array]]]:
    eff0 = _assemble_site(tensors, h_links, v_links, config, r, c)
    eff1 = _assemble_site(tensors, h_links, v_links, config, r + 1, c)
    row_mpo0 = _update_row_mpo_for_site(row_mpos[r], c, eff0, sites[r, c])
    row_mpo1 = _update_row_mpo_for_site(row_mpos[r + 1], c, eff1, sites[r + 1, c])
    transfer = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    updates = [(r, c, eff0), (r + 1, c, eff1)]
    return transfer, row_mpo0, row_mpo1, updates


def _row_pair_sweep(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    eff_tensors: list[list[jax.Array]],
    row_mpos: list[tuple],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
) -> tuple[jax.Array, list[list[jax.Array]], list[tuple], jax.Array, jax.Array, jax.Array]:
    n_cols = config.shape[1]
    transfers = [
        _contract_column_transfer_2row(
            top_env[c], row_mpos[r][c], row_mpos[r + 1][c], bottom_env[c]
        )
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs_2row(transfers, transfers[0].dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=transfers[0].dtype)

    for c in range(n_cols - 1):
        key, subkey = jax.random.split(key)
        delta = jnp.where(jax.random.bernoulli(subkey), 1, -1)

        amp_cur = _local_pair_amp(left_env, transfers[c], transfers[c + 1], right_envs[c + 1])
        h_prop, v_prop = _plaquette_flip(h_links, v_links, r, c, delta=delta, N=config.N)
        trans0, trans1, row_mpo0, row_mpo1, eff_updates = _updated_transfers_for_plaquette(
            tensors,
            row_mpos,
            h_prop,
            v_prop,
            sites,
            config,
            top_env,
            bottom_env,
            r,
            c,
        )
        amp_prop = _local_pair_amp(left_env, trans0, trans1, right_envs[c + 1])
        ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)

        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

        def accept_branch(_):
            updated_eff = [list(row) for row in eff_tensors]
            updated_eff[r][c] = eff_updates[0][0]
            updated_eff[r][c + 1] = eff_updates[0][1]
            updated_eff[r + 1][c] = eff_updates[1][0]
            updated_eff[r + 1][c + 1] = eff_updates[1][1]
            updated_row_mpos = list(row_mpos)
            updated_row_mpos[r] = row_mpo0
            updated_row_mpos[r + 1] = row_mpo1
            updated_transfers = list(transfers)
            updated_transfers[c] = trans0
            updated_transfers[c + 1] = trans1
            return h_prop, v_prop, updated_eff, updated_row_mpos, updated_transfers

        def reject_branch(_):
            return h_links, v_links, eff_tensors, row_mpos, transfers

        h_links, v_links, eff_tensors, row_mpos, transfers = jax.lax.cond(
            accept, accept_branch, reject_branch, operand=None
        )
        left_env = _contract_left_partial_2row(left_env, transfers[c])

    if config.phys_dim > 1:
        right_envs = _compute_right_envs_2row(transfers, transfers[0].dtype)
        left_env = jnp.ones((1, 1, 1, 1), dtype=transfers[0].dtype)
        for c in range(n_cols):
            key, subkey = jax.random.split(key)
            delta = jnp.where(jax.random.bernoulli(subkey), 1, -1)

            tmp = _contract_left_partial_2row(left_env, transfers[c])
            amp_cur = jnp.einsum("aceg,aceg->", tmp, right_envs[c])
            v_prop = v_links.at[r, c].set((v_links[r, c] + delta) % config.N)
            sites_prop = sites.at[r, c].set((sites[r, c] - delta) % config.phys_dim)
            sites_prop = sites_prop.at[r + 1, c].set(
                (sites_prop[r + 1, c] + delta) % config.phys_dim
            )
            trans, row_mpo0, row_mpo1, eff_updates = _updated_transfer_for_column(
                tensors,
                row_mpos,
                h_links,
                v_prop,
                sites_prop,
                config,
                top_env,
                bottom_env,
                r,
                c,
            )
            tmp_prop = _contract_left_partial_2row(left_env, trans)
            amp_prop = jnp.einsum("aceg,aceg->", tmp_prop, right_envs[c])
            ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)
            key, accept_key = jax.random.split(key)
            accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

            def accept_branch(_):
                updated_eff = [list(row) for row in eff_tensors]
                for rr, cc, eff in eff_updates:
                    updated_eff[rr][cc] = eff
                updated_row_mpos = list(row_mpos)
                updated_row_mpos[r] = row_mpo0
                updated_row_mpos[r + 1] = row_mpo1
                updated_transfers = list(transfers)
                updated_transfers[c] = trans
                return v_prop, sites_prop, updated_eff, updated_row_mpos, updated_transfers

            def reject_branch(_):
                return v_links, sites, eff_tensors, row_mpos, transfers

            v_links, sites, eff_tensors, row_mpos, transfers = jax.lax.cond(
                accept, accept_branch, reject_branch, operand=None
            )

            if c < n_cols - 1:
                for row_index in (r, r + 1):
                    key, subkey = jax.random.split(key)
                    delta = jnp.where(jax.random.bernoulli(subkey), 1, -1)

                    amp_cur = _local_pair_amp(
                        left_env, transfers[c], transfers[c + 1], right_envs[c + 1]
                    )
                    h_prop = h_links.at[row_index, c].set(
                        (h_links[row_index, c] + delta) % config.N
                    )
                    sites_prop = sites.at[row_index, c].set(
                        (sites[row_index, c] - delta) % config.phys_dim
                    )
                    sites_prop = sites_prop.at[row_index, c + 1].set(
                        (sites_prop[row_index, c + 1] + delta) % config.phys_dim
                    )
                    trans0, trans1, row_mpo0, row_mpo1, eff_updates = (
                        _updated_transfers_for_row_update(
                            tensors,
                            row_mpos,
                            h_prop,
                            v_links,
                            sites_prop,
                            config,
                            top_env,
                            bottom_env,
                            r,
                            c,
                            row_index,
                        )
                    )
                    amp_prop = _local_pair_amp(
                        left_env, trans0, trans1, right_envs[c + 1]
                    )
                    ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)
                    key, accept_key = jax.random.split(key)
                    accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

                    def accept_branch(_):
                        updated_eff = [list(row) for row in eff_tensors]
                        for rr, cc, eff in eff_updates:
                            updated_eff[rr][cc] = eff
                        updated_row_mpos = list(row_mpos)
                        updated_row_mpos[r] = row_mpo0
                        updated_row_mpos[r + 1] = row_mpo1
                        updated_transfers = list(transfers)
                        updated_transfers[c] = trans0
                        updated_transfers[c + 1] = trans1
                        return h_prop, sites_prop, updated_eff, updated_row_mpos, updated_transfers

                    def reject_branch(_):
                        return h_links, sites, eff_tensors, row_mpos, transfers

                    h_links, sites, eff_tensors, row_mpos, transfers = jax.lax.cond(
                        accept, accept_branch, reject_branch, operand=None
                    )

            left_env = _contract_left_partial_2row(left_env, transfers[c])

    return key, eff_tensors, row_mpos, sites, h_links, v_links


def _plaquette_energy(
    tensors: list[list[jax.Array]],
    row_mpos: list[tuple],
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    config: GIPEPSConfig,
    coeff: float,
    top_envs: list[tuple],
    bottom_envs: list[tuple],
) -> jax.Array:
    n_rows, n_cols = config.shape
    total = jnp.zeros((), dtype=row_mpos[0][0].dtype)
    for r in range(n_rows - 1):
        top_env = top_envs[r]
        bottom_env = bottom_envs[r + 1]
        transfers = [
            _contract_column_transfer_2row(
                top_env[c], row_mpos[r][c], row_mpos[r + 1][c], bottom_env[c]
            )
            for c in range(n_cols)
        ]
        right_envs = _compute_right_envs_2row(transfers, transfers[0].dtype)
        left_env = jnp.ones((1, 1, 1, 1), dtype=transfers[0].dtype)
        for c in range(n_cols - 1):
            amp_cur = _local_pair_amp(left_env, transfers[c], transfers[c + 1], right_envs[c + 1])
            h_plus, v_plus = _plaquette_flip(h_links, v_links, r, c, delta=1, N=config.N)
            trans0p, trans1p, _, _, _ = _updated_transfers_for_plaquette(
                tensors, row_mpos, h_plus, v_plus, sites, config, top_env, bottom_env, r, c
            )
            amp_plus = _local_pair_amp(left_env, trans0p, trans1p, right_envs[c + 1])
            h_minus, v_minus = _plaquette_flip(h_links, v_links, r, c, delta=-1, N=config.N)
            trans0m, trans1m, _, _, _ = _updated_transfers_for_plaquette(
                tensors, row_mpos, h_minus, v_minus, sites, config, top_env, bottom_env, r, c
            )
            amp_minus = _local_pair_amp(left_env, trans0m, trans1m, right_envs[c + 1])
            total = total + coeff * (amp_plus + amp_minus) / amp_cur
            left_env = _contract_left_partial_2row(left_env, transfers[c])
    return total


@functools.partial(
    jax.jit,
    static_argnames=("n_samples", "n_chains", "burn_in", "full_gradient"),
)
@dispatch
def sequential_sample_with_gradients(
    model: GIPEPS,
    operator: GILocalHamiltonian,
    *,
    n_samples: int = 1,
    n_chains: int = 1,
    key: jax.Array,
    initial_configuration: jax.Array,
    burn_in: int = 0,
    full_gradient: bool = True,
):
    num_samples, num_chains, num_burn_in, chain_length, total_samples = _sample_counts(
        n_samples, n_chains, burn_in
    )
    shape = model.shape
    n_rows, n_cols = shape
    config = model.config

    tensors = [[jnp.asarray(t) for t in row] for row in model.tensors]
    sites, h_links, v_links = jax.vmap(
        lambda conf: GIPEPS.unflatten_sample(conf, shape)
    )(initial_configuration)

    def cache_envs_fn(state):
        sites, h_links, v_links = state
        eff_tensors = [[_assemble_site(tensors, h_links, v_links, config, r, c)
                        for c in range(n_cols)]
                       for r in range(n_rows)]
        row_mpos = _build_row_mpos_with_sites(eff_tensors, sites)
        return _bottom_envs(row_mpos, model.strategy)

    bottom_envs = jax.vmap(cache_envs_fn)((sites, h_links, v_links))

    def sweep_with_envs(state, key, bottom_envs, collect_top_envs):
        sites, h_links, v_links = state
        eff_tensors = [
            [_assemble_site(tensors, h_links, v_links, config, r, c) for c in range(n_cols)]
            for r in range(n_rows)
        ]
        row_mpos = _build_row_mpos_with_sites(eff_tensors, sites)

        top_env = tuple(jnp.ones((1, 1, 1), dtype=eff_tensors[0][0].dtype) for _ in range(n_cols))
        for r in range(n_rows - 1):
            key, eff_tensors, row_mpos, sites, h_links, v_links = _row_pair_sweep(
                key,
                tensors,
                eff_tensors,
                row_mpos,
                sites,
                h_links,
                v_links,
                config,
                top_env,
                bottom_envs[r + 1],
                r,
            )
            top_env = model.strategy.apply(top_env, row_mpos[r])
        top_env = model.strategy.apply(top_env, row_mpos[n_rows - 1])
        amp = _contract_bottom(top_env) if collect_top_envs else jnp.zeros((), dtype=top_env[0].dtype)
        if collect_top_envs:
            top_envs = []
            top_env = tuple(
                jnp.ones((1, 1, 1), dtype=eff_tensors[0][0].dtype) for _ in range(n_cols)
            )
            for r in range(n_rows):
                top_envs.append(top_env)
                top_env = model.strategy.apply(top_env, row_mpos[r])
            aux = (top_envs, row_mpos)
        else:
            aux = ()
        return (sites, h_links, v_links), key, aux, amp

    def sweep_batched(state, chain_keys, bottom_envs, collect_top_envs):
        def sweep_single(state, key, envs):
            return sweep_with_envs(state, key, envs, collect_top_envs)

        return jax.vmap(sweep_single, in_axes=(0, 0, 0))(state, chain_keys, bottom_envs)

    key, chain_key = jax.random.split(key)
    chain_keys = jax.random.split(chain_key, num_chains)

    def burn_step(carry, _):
        state, chain_keys, bottom_envs = carry
        state, chain_keys, _, _ = sweep_batched(state, chain_keys, bottom_envs, False)
        bottom_envs = jax.vmap(cache_envs_fn)(state)
        return (state, chain_keys, bottom_envs), None

    ((sites, h_links, v_links), chain_keys, bottom_envs), _ = jax.lax.scan(
        burn_step,
        ((sites, h_links, v_links), chain_keys, bottom_envs),
        xs=None,
        length=num_burn_in,
    )

    def sample_step(carry, _):
        sites, h_links, v_links, chain_keys, bottom_envs = carry
        (sites, h_links, v_links), chain_keys, aux, amp = sweep_batched(
            (sites, h_links, v_links), chain_keys, bottom_envs, True
        )
        top_envs, row_mpos = aux
        bottom_envs = jax.vmap(lambda m: _bottom_envs(m, model.strategy))(row_mpos)

        env_grads = jax.vmap(
            lambda m, t: _compute_all_gradients_from_row_mpos(m, shape, model.strategy, t)
        )(row_mpos, top_envs)

        def chain_grads(env_grads, sites, h_links, v_links, amp):
            grad_parts = []
            p_parts = []
            for r in range(n_rows):
                for c in range(n_cols):
                    if full_gradient:
                        k_l = _link_value_or_zero(h_links, v_links, r, c, direction="left")
                        k_r = _link_value_or_zero(h_links, v_links, r, c, direction="right")
                        k_u = _link_value_or_zero(h_links, v_links, r, c, direction="up")
                        k_d = _link_value_or_zero(h_links, v_links, r, c, direction="down")
                        grad_site = jnp.zeros_like(tensors[r][c])
                        grad_site = grad_site.at[
                            sites[r, c], k_u, :, k_d, :, k_l, :, k_r, :
                        ].set(env_grads[r][c])
                        grad_parts.append(grad_site.reshape(-1))
                    else:
                        grad_parts.append(env_grads[r][c].reshape(-1))
                        params_per_site = env_grads[r][c].size
                        p_parts.append(
                            jnp.full((params_per_site,), sites[r, c], dtype=jnp.int8)
                        )
            grad_row = jnp.concatenate(grad_parts) / amp
            p_row = (
                jnp.zeros((0,), dtype=jnp.int8)
                if full_gradient
                else jnp.concatenate(p_parts)
            )
            return grad_row, p_row

        grad_row, p_row = jax.vmap(chain_grads)(env_grads, sites, h_links, v_links, amp)

        def electric_energy(h_links, v_links):
            total = jnp.zeros((), dtype=amp.dtype)
            for term in operator.electric_terms:
                if isinstance(term, LinkDiagonalTerm):
                    total = total + term.energy(h_links, v_links)
            return total

        electric_energy = jax.vmap(electric_energy)(h_links, v_links)
        plaquette_energy = jax.vmap(
            lambda m, h, v, s, t, b: _plaquette_energy(
                tensors,
                m,
                h,
                v,
                s,
                config,
                operator.plaquette.coeff,
                t,
                b,
            )
        )(row_mpos, h_links, v_links, sites, top_envs, bottom_envs)
        total_energy = electric_energy + plaquette_energy

        sample_flat = jax.vmap(GIPEPS.flatten_sample)(sites, h_links, v_links)
        return (sites, h_links, v_links, chain_keys, bottom_envs), (
            sample_flat,
            grad_row,
            p_row,
            amp,
            total_energy,
        )

    (sites, h_links, v_links, chain_keys, bottom_envs), outputs = jax.lax.scan(
        sample_step,
        (sites, h_links, v_links, chain_keys, bottom_envs),
        xs=None,
        length=chain_length,
    )
    samples, grads, p_rows, amps, energies = outputs

    samples = _trim_samples(samples, total_samples, num_samples)
    grads = _trim_samples(grads, total_samples, num_samples)
    amps = _trim_samples(amps, total_samples, num_samples)
    energies = _trim_samples(energies, total_samples, num_samples)
    p = None if full_gradient else _trim_samples(p_rows, total_samples, num_samples)

    return (
        samples,
        grads,
        p,
        key,
        jax.vmap(GIPEPS.flatten_sample)(sites, h_links, v_links),
        amps,
        energies,
    )

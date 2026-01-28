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
    _assemble_site,
    _link_value_or_zero,
    _sample_site_index_for_charge,
)
from vmc.models.peps import (
    _apply_mpo_from_below,
    _contract_bottom,
    _contract_column_transfer,
    _contract_column_transfer_2row,
    _contract_left_partial,
    _contract_left_partial_2row,
    _compute_2site_horizontal_env,
    _compute_right_envs,
    _compute_right_envs_2row,
    _compute_row_pair_vertical_energy,
    _compute_single_gradient,
)
from vmc.operators.local_terms import bucket_terms
from vmc.samplers.sequential import (
    _metropolis_ratio,
    _sample_counts,
    _trim_samples,
)


def _row_mpo_site(
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    r: int,
    c: int,
) -> jax.Array:
    eff = _assemble_site(tensors, h_links, v_links, r, c)
    return jnp.transpose(eff[sites[r, c]], (2, 3, 0, 1))


def _build_row_mpo_gi(
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    row: int,
    n_cols: int,
) -> tuple:
    return tuple(
        _row_mpo_site(tensors, sites, h_links, v_links, row, c)
        for c in range(n_cols)
    )


def _bottom_envs(
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    strategy: Any,
) -> list[tuple]:
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    bottom_envs = [None] * n_rows
    bottom_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    for row in range(n_rows - 1, -1, -1):
        bottom_envs[row] = bottom_env
        row_mpo = _build_row_mpo_gi(tensors, sites, h_links, v_links, row, n_cols)
        bottom_env = _apply_mpo_from_below(bottom_env, row_mpo, strategy)
    return bottom_envs




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
    row_mpo0: tuple,
    row_mpo1: tuple,
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
) -> tuple[jax.Array, jax.Array, tuple, tuple]:
    eff00 = _assemble_site(tensors, h_links, v_links, r, c)
    eff01 = _assemble_site(tensors, h_links, v_links, r, c + 1)
    eff10 = _assemble_site(tensors, h_links, v_links, r + 1, c)
    eff11 = _assemble_site(tensors, h_links, v_links, r + 1, c + 1)

    row_mpo0 = _update_row_mpo_for_site(row_mpo0, c, eff00, sites[r, c])
    row_mpo0 = _update_row_mpo_for_site(row_mpo0, c + 1, eff01, sites[r, c + 1])
    row_mpo1 = _update_row_mpo_for_site(row_mpo1, c, eff10, sites[r + 1, c])
    row_mpo1 = _update_row_mpo_for_site(row_mpo1, c + 1, eff11, sites[r + 1, c + 1])

    transfer0 = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    transfer1 = _contract_column_transfer_2row(
        top_env[c + 1], row_mpo0[c + 1], row_mpo1[c + 1], bottom_env[c + 1]
    )
    return transfer0, transfer1, row_mpo0, row_mpo1


def _updated_transfers_for_row_update(
    tensors: list[list[jax.Array]],
    row_mpo0: tuple,
    row_mpo1: tuple,
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
    row_index: int,
) -> tuple[jax.Array, jax.Array, tuple, tuple]:
    if row_index == r:
        eff0 = _assemble_site(tensors, h_links, v_links, r, c)
        eff1 = _assemble_site(tensors, h_links, v_links, r, c + 1)
        row_mpo0 = _update_row_mpo_for_site(row_mpo0, c, eff0, sites[r, c])
        row_mpo0 = _update_row_mpo_for_site(row_mpo0, c + 1, eff1, sites[r, c + 1])
    else:
        eff0 = _assemble_site(tensors, h_links, v_links, r + 1, c)
        eff1 = _assemble_site(tensors, h_links, v_links, r + 1, c + 1)
        row_mpo1 = _update_row_mpo_for_site(row_mpo1, c, eff0, sites[r + 1, c])
        row_mpo1 = _update_row_mpo_for_site(row_mpo1, c + 1, eff1, sites[r + 1, c + 1])

    transfer0 = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    transfer1 = _contract_column_transfer_2row(
        top_env[c + 1], row_mpo0[c + 1], row_mpo1[c + 1], bottom_env[c + 1]
    )
    return transfer0, transfer1, row_mpo0, row_mpo1


def _updated_transfer_for_column(
    tensors: list[list[jax.Array]],
    row_mpo0: tuple,
    row_mpo1: tuple,
    h_links: jax.Array,
    v_links: jax.Array,
    sites: jax.Array,
    top_env: tuple,
    bottom_env: tuple,
    r: int,
    c: int,
) -> tuple[jax.Array, tuple, tuple]:
    eff0 = _assemble_site(tensors, h_links, v_links, r, c)
    eff1 = _assemble_site(tensors, h_links, v_links, r + 1, c)
    row_mpo0 = _update_row_mpo_for_site(row_mpo0, c, eff0, sites[r, c])
    row_mpo1 = _update_row_mpo_for_site(row_mpo1, c, eff1, sites[r + 1, c])
    transfer = _contract_column_transfer_2row(
        top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
    )
    return transfer, row_mpo0, row_mpo1


def _row_pair_sweep(
    key: jax.Array,
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    config: GIPEPSConfig,
    top_env: tuple,
    bottom_env: tuple,
    row_mpo0: tuple,
    row_mpo1: tuple,
    r: int,
    charge_of_site: jax.Array,
    charge_to_indices: jax.Array,
    charge_deg: jax.Array,
) -> tuple[jax.Array, tuple, tuple, jax.Array, jax.Array, jax.Array]:
    n_cols = config.shape[1]
    transfers = [
        _contract_column_transfer_2row(
            top_env[c], row_mpo0[c], row_mpo1[c], bottom_env[c]
        )
        for c in range(n_cols)
    ]
    right_envs = _compute_right_envs_2row(transfers, transfers[0].dtype)
    left_env = jnp.ones((1, 1, 1, 1), dtype=transfers[0].dtype)

    for c in range(n_cols - 1):
        key, subkey = jax.random.split(key)
        delta = jnp.where(jax.random.bernoulli(subkey), 1, -1)

        left_partial = _contract_left_partial_2row(left_env, transfers[c])
        tmp = _contract_left_partial_2row(left_partial, transfers[c + 1])
        amp_cur = jnp.einsum("aceg,aceg->", tmp, right_envs[c + 1])
        h_prop, v_prop = _plaquette_flip(h_links, v_links, r, c, delta=delta, N=config.N)
        trans0, trans1, row_mpo0_prop, row_mpo1_prop = _updated_transfers_for_plaquette(
            tensors,
            row_mpo0,
            row_mpo1,
            h_prop,
            v_prop,
            sites,
            top_env,
            bottom_env,
            r,
            c,
        )
        left_partial_prop = _contract_left_partial_2row(left_env, trans0)
        tmp_prop = _contract_left_partial_2row(left_partial_prop, trans1)
        amp_prop = jnp.einsum("aceg,aceg->", tmp_prop, right_envs[c + 1])
        ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)

        key, accept_key = jax.random.split(key)
        accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

        def accept_branch(_):
            updated_transfers = list(transfers)
            updated_transfers[c] = trans0
            updated_transfers[c + 1] = trans1
            return h_prop, v_prop, row_mpo0_prop, row_mpo1_prop, updated_transfers, left_partial_prop

        def reject_branch(_):
            return h_links, v_links, row_mpo0, row_mpo1, transfers, left_partial

        h_links, v_links, row_mpo0, row_mpo1, transfers, left_env = jax.lax.cond(
            accept, accept_branch, reject_branch, operand=None
        )

    if config.phys_dim > 1:
        right_envs = _compute_right_envs_2row(transfers, transfers[0].dtype)
        left_env = jnp.ones((1, 1, 1, 1), dtype=transfers[0].dtype)
        for c in range(n_cols):
            key, subkey = jax.random.split(key)
            delta = jnp.where(jax.random.bernoulli(subkey), 1, -1)

            left_partial = _contract_left_partial_2row(left_env, transfers[c])
            amp_cur = jnp.einsum("aceg,aceg->", left_partial, right_envs[c])
            v_prop = v_links.at[r, c].set((v_links[r, c] + delta) % config.N)
            q_top = charge_of_site[sites[r, c]]
            q_bottom = charge_of_site[sites[r + 1, c]]
            q_top_new = (q_top - delta) % config.N
            q_bottom_new = (q_bottom + delta) % config.N
            key, site_key = jax.random.split(key)
            key_top, key_bottom = jax.random.split(site_key)
            site_top = _sample_site_index_for_charge(
                key_top, q_top_new, charge_to_indices, charge_deg
            )
            site_bottom = _sample_site_index_for_charge(
                key_bottom, q_bottom_new, charge_to_indices, charge_deg
            )
            sites_prop = sites.at[r, c].set(site_top)
            sites_prop = sites_prop.at[r + 1, c].set(site_bottom)
            trans, row_mpo0_prop, row_mpo1_prop = _updated_transfer_for_column(
                tensors,
                row_mpo0,
                row_mpo1,
                h_links,
                v_prop,
                sites_prop,
                top_env,
                bottom_env,
                r,
                c,
            )
            left_partial_prop = _contract_left_partial_2row(left_env, trans)
            amp_prop = jnp.einsum("aceg,aceg->", left_partial_prop, right_envs[c])
            ratio = _metropolis_ratio(jnp.abs(amp_cur) ** 2, jnp.abs(amp_prop) ** 2)
            prop_ratio = (
                charge_deg[q_top_new] * charge_deg[q_bottom_new]
            ) / (charge_deg[q_top] * charge_deg[q_bottom])
            ratio = ratio * prop_ratio
            key, accept_key = jax.random.split(key)
            accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

            def accept_branch(_):
                updated_transfers = list(transfers)
                updated_transfers[c] = trans
                return v_prop, sites_prop, row_mpo0_prop, row_mpo1_prop, updated_transfers

            def reject_branch(_):
                return v_links, sites, row_mpo0, row_mpo1, transfers

            v_links, sites, row_mpo0, row_mpo1, transfers = jax.lax.cond(
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
                    q_left = charge_of_site[sites[row_index, c]]
                    q_right = charge_of_site[sites[row_index, c + 1]]
                    q_left_new = (q_left - delta) % config.N
                    q_right_new = (q_right + delta) % config.N
                    key, site_key = jax.random.split(key)
                    key_left, key_right = jax.random.split(site_key)
                    site_left = _sample_site_index_for_charge(
                        key_left, q_left_new, charge_to_indices, charge_deg
                    )
                    site_right = _sample_site_index_for_charge(
                        key_right, q_right_new, charge_to_indices, charge_deg
                    )
                    sites_prop = sites.at[row_index, c].set(site_left)
                    sites_prop = sites_prop.at[row_index, c + 1].set(site_right)
                    trans0, trans1, row_mpo0_prop, row_mpo1_prop = (
                        _updated_transfers_for_row_update(
                            tensors,
                            row_mpo0,
                            row_mpo1,
                            h_prop,
                            v_links,
                            sites_prop,
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
                    prop_ratio = (
                        charge_deg[q_left_new] * charge_deg[q_right_new]
                    ) / (charge_deg[q_left] * charge_deg[q_right])
                    ratio = ratio * prop_ratio
                    key, accept_key = jax.random.split(key)
                    accept = jax.random.uniform(accept_key) < jnp.minimum(1.0, ratio)

                    def accept_branch(_):
                        updated_transfers = list(transfers)
                        updated_transfers[c] = trans0
                        updated_transfers[c + 1] = trans1
                        return h_prop, sites_prop, row_mpo0_prop, row_mpo1_prop, updated_transfers

                    def reject_branch(_):
                        return h_links, sites, row_mpo0, row_mpo1, transfers

                    h_links, sites, row_mpo0, row_mpo1, transfers = jax.lax.cond(
                        accept, accept_branch, reject_branch, operand=None
                    )

            left_env = _contract_left_partial_2row(left_env, transfers[c])

    return key, row_mpo0, row_mpo1, sites, h_links, v_links


def _compute_all_env_grads_and_energy_gi(
    tensors: list[list[jax.Array]],
    sites: jax.Array,
    h_links: jax.Array,
    v_links: jax.Array,
    amp: jax.Array,
    config: GIPEPSConfig,
    operator: GILocalHamiltonian,
    strategy: Any,
    *,
    collect_grads: bool = True,
) -> tuple[list[list[jax.Array]], jax.Array, list[tuple]]:
    n_rows, n_cols = config.shape
    dtype = tensors[0][0].dtype
    phys_dim = config.phys_dim

    (
        diagonal_terms,
        one_site_terms,
        horizontal_terms,
        vertical_terms,
        plaquette_terms,
    ) = bucket_terms(operator.terms, config.shape)

    env_grads = (
        [[None for _ in range(n_cols)] for _ in range(n_rows)]
        if collect_grads
        else []
    )
    bottom_envs = _bottom_envs(tensors, sites, h_links, v_links, config, strategy)
    energy = jnp.zeros((), dtype=amp.dtype)
    for term in diagonal_terms:
        if isinstance(term, LinkDiagonalTerm):
            energy = energy + term.energy(h_links, v_links)
            continue
        idx = jnp.asarray(0, dtype=jnp.int32)
        for row, col in term.sites:
            idx = idx * phys_dim + sites[row, col]
        energy = energy + term.diag[idx]

    top_env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
    row_mpo = _build_row_mpo_gi(tensors, sites, h_links, v_links, 0, n_cols)
    for row in range(n_rows):
        bottom_env = bottom_envs[row]
        transfers = [
            _contract_column_transfer(top_env[c], row_mpo[c], bottom_env[c])
            for c in range(n_cols)
        ]
        right_envs = _compute_right_envs(transfers, dtype)
        left_env = jnp.ones((1, 1, 1), dtype=dtype)
        row_has_one_site = any(one_site_terms[row][c] for c in range(n_cols))
        row_has_horizontal = any(horizontal_terms[row][c] for c in range(n_cols - 1))
        row_has_vertical = row < n_rows - 1 and any(
            vertical_terms[row][c] for c in range(n_cols)
        )
        row_has_plaquette = row < n_rows - 1 and any(
            plaquette_terms[row][c] for c in range(n_cols - 1)
        )
        eff_row = None
        eff_row_next = None
        if row_has_one_site or row_has_horizontal or row_has_vertical:
            eff_row = [
                _assemble_site(tensors, h_links, v_links, row, c)
                for c in range(n_cols)
            ]
        if row_has_vertical:
            eff_row_next = [
                _assemble_site(tensors, h_links, v_links, row + 1, c)
                for c in range(n_cols)
            ]
        for c in range(n_cols):
            site_terms = one_site_terms[row][c]
            need_env_grad = collect_grads or site_terms
            if need_env_grad:
                env_grad = _compute_single_gradient(
                    left_env, right_envs[c], top_env[c], bottom_env[c]
                )
                if collect_grads:
                    env_grads[row][c] = env_grad
                if site_terms:
                    amps_site = jnp.einsum("pudlr,udlr->p", eff_row[c], env_grad)
                    spin_idx = sites[row, c]
                    for term in site_terms:
                        energy = energy + jnp.dot(term.op[:, spin_idx], amps_site) / amp
            if c < n_cols - 1:
                edge_terms = horizontal_terms[row][c]
                if edge_terms:
                    env_2site = _compute_2site_horizontal_env(
                        left_env,
                        right_envs[c + 1],
                        top_env[c],
                        bottom_env[c],
                        top_env[c + 1],
                        bottom_env[c + 1],
                    )
                    amps_edge = jnp.einsum(
                        "pudlr,qverx,udlvex->pq",
                        eff_row[c],
                        eff_row[c + 1],
                        env_2site,
                    )
                    spin0 = sites[row, c]
                    spin1 = sites[row, c + 1]
                    col_idx = spin0 * phys_dim + spin1
                    amps_flat = amps_edge.reshape(-1)
                    for term in edge_terms:
                        energy = energy + jnp.dot(term.op[:, col_idx], amps_flat) / amp
            left_env = _contract_left_partial(left_env, transfers[c])
        if row < n_rows - 1:
            row_mpo_next = _build_row_mpo_gi(
                tensors, sites, h_links, v_links, row + 1, n_cols
            )
            if row_has_vertical or row_has_plaquette:
                bottom_env_next = bottom_envs[row + 1]
                transfers_2row = [
                    _contract_column_transfer_2row(
                        top_env[c], row_mpo[c], row_mpo_next[c], bottom_env_next[c]
                    )
                    for c in range(n_cols)
                ]
                right_envs_2row = _compute_right_envs_2row(
                    transfers_2row, transfers_2row[0].dtype
                )
                if row_has_vertical:
                    energy = energy + _compute_row_pair_vertical_energy(
                        top_env,
                        bottom_env_next,
                        row_mpo,
                        row_mpo_next,
                        eff_row,
                        eff_row_next,
                        sites[row],
                        sites[row + 1],
                        vertical_terms[row],
                        amp,
                        phys_dim,
                        transfers_2row=transfers_2row,
                        right_envs_2row=right_envs_2row,
                    )
                if row_has_plaquette:
                    left_env_2row = jnp.ones((1, 1, 1, 1), dtype=transfers_2row[0].dtype)
                    for c in range(n_cols - 1):
                        plaquette_here = plaquette_terms[row][c]
                        if not plaquette_here:
                            left_env_2row = _contract_left_partial_2row(
                                left_env_2row, transfers_2row[c]
                            )
                            continue
                        amp_cur = _local_pair_amp(
                            left_env_2row,
                            transfers_2row[c],
                            transfers_2row[c + 1],
                            right_envs_2row[c + 1],
                        )
                        h_plus, v_plus = _plaquette_flip(
                            h_links, v_links, row, c, delta=1, N=config.N
                        )
                        trans0p, trans1p, _, _ = _updated_transfers_for_plaquette(
                            tensors,
                            row_mpo,
                            row_mpo_next,
                            h_plus,
                            v_plus,
                            sites,
                            top_env,
                            bottom_env_next,
                            row,
                            c,
                        )
                        amp_plus = _local_pair_amp(
                            left_env_2row,
                            trans0p,
                            trans1p,
                            right_envs_2row[c + 1],
                        )
                        h_minus, v_minus = _plaquette_flip(
                            h_links, v_links, row, c, delta=-1, N=config.N
                        )
                        trans0m, trans1m, _, _ = _updated_transfers_for_plaquette(
                            tensors,
                            row_mpo,
                            row_mpo_next,
                            h_minus,
                            v_minus,
                            sites,
                            top_env,
                            bottom_env_next,
                            row,
                            c,
                        )
                        amp_minus = _local_pair_amp(
                            left_env_2row,
                            trans0m,
                            trans1m,
                            right_envs_2row[c + 1],
                        )
                        if len(plaquette_here) == 1:
                            coeff = plaquette_here[0].coeff
                        else:
                            coeff = jnp.sum(
                                jnp.asarray([term.coeff for term in plaquette_here])
                            )
                        energy = energy + coeff * (amp_plus + amp_minus) / amp_cur
                        left_env_2row = _contract_left_partial_2row(
                            left_env_2row, transfers_2row[c]
                        )
        top_env = strategy.apply(top_env, row_mpo)
        if row < n_rows - 1:
            row_mpo = row_mpo_next

    return env_grads, energy, bottom_envs


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
    charge_of_site = jnp.asarray(model.charge_of_site, dtype=jnp.int32)
    charge_to_indices = model.charge_to_indices
    charge_deg = model.charge_deg
    sites, h_links, v_links = jax.vmap(
        lambda conf: GIPEPS.unflatten_sample(conf, shape)
    )(initial_configuration)

    def cache_envs_fn(state):
        sites, h_links, v_links = state
        return _bottom_envs(tensors, sites, h_links, v_links, config, model.strategy)

    bottom_envs = jax.vmap(cache_envs_fn)((sites, h_links, v_links))

    def sweep_with_envs(state, key, bottom_envs, return_amp):
        sites, h_links, v_links = state
        top_env = tuple(jnp.ones((1, 1, 1), dtype=tensors[0][0].dtype) for _ in range(n_cols))
        if n_rows == 1:
            row_mpo = _build_row_mpo_gi(tensors, sites, h_links, v_links, 0, n_cols)
            top_env = model.strategy.apply(top_env, row_mpo)
        else:
            row_mpo0 = _build_row_mpo_gi(tensors, sites, h_links, v_links, 0, n_cols)
            row_mpo1 = _build_row_mpo_gi(tensors, sites, h_links, v_links, 1, n_cols)
            for r in range(n_rows - 1):
                key, row_mpo0, row_mpo1, sites, h_links, v_links = _row_pair_sweep(
                    key,
                    tensors,
                    sites,
                    h_links,
                    v_links,
                    config,
                    top_env,
                    bottom_envs[r + 1],
                    row_mpo0,
                    row_mpo1,
                    r,
                    charge_of_site,
                    charge_to_indices,
                    charge_deg,
                )
                top_env = model.strategy.apply(top_env, row_mpo0)
                if r + 2 < n_rows:
                    row_mpo0 = row_mpo1
                    row_mpo1 = _build_row_mpo_gi(
                        tensors, sites, h_links, v_links, r + 2, n_cols
                    )
            top_env = model.strategy.apply(top_env, row_mpo1)
        amp = _contract_bottom(top_env) if return_amp else jnp.zeros((), dtype=top_env[0].dtype)
        return (sites, h_links, v_links), key, amp

    def sweep_batched(state, chain_keys, bottom_envs, return_amp):
        def sweep_single(state, key, envs):
            return sweep_with_envs(state, key, envs, return_amp)

        return jax.vmap(sweep_single, in_axes=(0, 0, 0))(state, chain_keys, bottom_envs)

    key, chain_key = jax.random.split(key)
    chain_keys = jax.random.split(chain_key, num_chains)

    def burn_step(carry, _):
        state, chain_keys, bottom_envs = carry
        state, chain_keys, _ = sweep_batched(state, chain_keys, bottom_envs, False)
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
        (sites, h_links, v_links), chain_keys, amp = sweep_batched(
            (sites, h_links, v_links), chain_keys, bottom_envs, True
        )
        env_grads, total_energy, bottom_envs = jax.vmap(
            lambda s, h, v, a: _compute_all_env_grads_and_energy_gi(
                tensors,
                s,
                h,
                v,
                a,
                config,
                operator,
                model.strategy,
            )
        )(sites, h_links, v_links, amp)

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

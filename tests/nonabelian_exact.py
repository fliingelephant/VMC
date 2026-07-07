"""Exact enumeration of non-Abelian GI-PEPS states and operator outcomes.

Shared by the SU(2)/SU(3) kernel, group, and driver tests: decodes the
factored per-vertex operator tables into explicit ``(candidate, me)`` lists
and builds exact pure-gauge Hamiltonians over the full valid-sample basis.
"""

import itertools

import jax
import jax.numpy as jnp

from vmc.peps.non_abelian_gi import NonAbelianGIPEPS
from vmc.peps.non_abelian_gi.contraction import flatten_like_sample
from vmc.peps.non_abelian_gi.factors import PLAQUETTE_KEY_LEGS, PLAQUETTE_LEG_FWD


def decode_rows(table, block: int, key: tuple[int, ...]) -> list[tuple[int, complex]]:
    start = int(table.group_starts[(block, *key)])
    count = int(table.group_counts[(block, *key)])
    return [
        (int(table.out_blocks[i]), complex(table.factors[i]))
        for i in range(start, start + count)
    ]


def plaquette_outcomes(
    model: NonAbelianGIPEPS,
    sample: jax.Array,
    *,
    row: int = 0,
    col: int = 0,
) -> list[tuple[jax.Array, complex]]:
    """All ``(candidate_sample, me)`` of one plaquette window, factor tables."""
    tabs = model.plaquette_factor_tables
    tables = model.tables
    matter, h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_spin_network_sample(
        sample, model.shape
    )
    active = model.active_block_ids(sample)
    blocks = (
        int(active[row, col]),
        int(active[row, col + 1]),
        int(active[row + 1, col]),
        int(active[row + 1, col + 1]),
    )
    links = (
        int(tables.j_r_by_block[row, col, blocks[0]]),
        int(tables.j_d_by_block[row, col + 1, blocks[1]]),
        int(tables.j_r_by_block[row + 1, col, blocks[2]]),
        int(tables.j_d_by_block[row, col, blocks[0]]),
    )
    corner_tables = (
        tabs.tl[row][col],
        tabs.tr[row][col + 1],
        tabs.bl[row + 1][col],
        tabs.br[row + 1][col + 1],
    )
    offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
    outcomes = []
    for orient in range(2):
        maps = [
            tabs.fuse_op if fwd == (orient == 0) else tabs.fuse_rev
            for fwd in PLAQUETTE_LEG_FWD
        ]
        leg_choices = [
            [int(m.outputs[link, k]) for k in range(int(m.counts[link]))]
            for m, link in zip(maps, links)
        ]
        for legs in itertools.product(*leg_choices):
            cand_lists = [
                decode_rows(
                    pair[orient],
                    blocks[v],
                    tuple(legs[p] for p in PLAQUETTE_KEY_LEGS[v]),
                )
                for v, pair in enumerate(corner_tables)
            ]
            for picks in itertools.product(*cand_lists):
                me = 1.0 + 0.0j
                h_new = h_links.at[row, col].set(legs[0])
                h_new = h_new.at[row + 1, col].set(legs[2])
                v_new = v_links.at[row, col + 1].set(legs[1])
                v_new = v_new.at[row, col].set(legs[3])
                iota_new = iotas
                for (dr, dc), (out_id, lam) in zip(offsets, picks, strict=True):
                    me *= lam
                    iota_new = iota_new.at[row + dr, col + dc].set(
                        tables.iota_by_block[row + dr, col + dc, out_id]
                    )
                outcomes.append(
                    (flatten_like_sample(sample, matter, h_new, v_new, iota_new), me)
                )
    return outcomes


def hopping_outcomes(
    model: NonAbelianGIPEPS,
    sample: jax.Array,
    *,
    row: int,
    col: int,
    horizontal: bool,
) -> list[tuple[jax.Array, complex]]:
    """All ``(candidate_sample, me)`` of one hopping window, factor tables."""
    hop = model.hopping_factor_tables
    tables = model.tables
    matter, h_links, v_links, iotas = NonAbelianGIPEPS.unflatten_spin_network_sample(
        sample, model.shape
    )
    active = model.active_block_ids(sample)
    if horizontal:
        sites = ((row, col), (row, col + 1))
        pairs = (hop.h_src[row][col], hop.h_tgt[row][col + 1])
        link = int(tables.j_r_by_block[row, col, active[row, col]])
    else:
        sites = ((row, col), (row + 1, col))
        pairs = (hop.v_src[row][col], hop.v_tgt[row + 1][col])
        link = int(tables.j_d_by_block[row, col, active[row, col]])
    blocks = tuple(int(active[r, c]) for r, c in sites)
    outcomes = []
    for orient in range(2):
        fuse = hop.fuse_op if orient == 0 else hop.fuse_rev
        for k in range(int(fuse.counts[link])):
            new_link = int(fuse.outputs[link, k])
            cand_lists = [
                decode_rows(pair[orient], block, (new_link,))
                for pair, block in zip(pairs, blocks)
            ]
            for picks in itertools.product(*cand_lists):
                me = 1.0 + 0.0j
                matter_new, iota_new = matter, iotas
                for (r, c), (out_id, lam) in zip(sites, picks, strict=True):
                    me *= lam
                    matter_new = matter_new.at[r, c].set(
                        tables.matter_state_by_block[r, c, out_id]
                    )
                    iota_new = iota_new.at[r, c].set(tables.iota_by_block[r, c, out_id])
                h_new, v_new = h_links, v_links
                if horizontal:
                    h_new = h_links.at[row, col].set(new_link)
                else:
                    v_new = v_links.at[row, col].set(new_link)
                outcomes.append(
                    (
                        flatten_like_sample(
                            sample, matter_new, h_new, v_new, iota_new
                        ),
                        me,
                    )
                )
    return outcomes


def valid_samples(model: NonAbelianGIPEPS) -> tuple[jax.Array, ...]:
    n_rows, n_cols = model.shape
    link_irreps = model.gauge_group.irreps()
    samples = []
    for h_values in itertools.product(link_irreps, repeat=n_rows * (n_cols - 1)):
        h_links = jnp.asarray(h_values, dtype=jnp.int32).reshape((n_rows, n_cols - 1))
        for v_values in itertools.product(link_irreps, repeat=(n_rows - 1) * n_cols):
            v_links = jnp.asarray(v_values, dtype=jnp.int32).reshape(
                (n_rows - 1, n_cols)
            )
            iota_choices = []
            for row in range(n_rows):
                for col in range(n_cols):
                    choices = []
                    for iota in range(model.tables.max_iotas):
                        block_id = model.tables.block_id_lookup[
                            row,
                            col,
                            0,
                            h_links[row, col - 1] if col > 0 else 0,
                            v_links[row - 1, col] if row > 0 else 0,
                            h_links[row, col] if col < n_cols - 1 else 0,
                            v_links[row, col] if row < n_rows - 1 else 0,
                            iota,
                        ]
                        if int(block_id) >= 0:
                            choices.append(iota)
                    if not choices:
                        break
                    iota_choices.append(tuple(choices))
                else:
                    continue
                break
            if len(iota_choices) != n_rows * n_cols:
                continue
            for iotas in itertools.product(*iota_choices):
                samples.append(
                    NonAbelianGIPEPS.flatten_sample(
                        h_links,
                        v_links,
                        jnp.asarray(iotas, dtype=jnp.int32).reshape(model.shape),
                    )
                )
    return tuple(samples)


def exact_pure_gauge_hamiltonian(
    model: NonAbelianGIPEPS,
    *,
    electric_coeff: float,
    plaquette_coeff: float,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    samples = valid_samples(model)
    sample_keys = {tuple(sample.tolist()): idx for idx, sample in enumerate(samples)}
    hamiltonian = jnp.zeros((len(samples), len(samples)), dtype=jnp.complex128)
    for source_idx, sample in enumerate(samples):
        h_links, v_links, _iotas = NonAbelianGIPEPS.unflatten_sample(
            sample, model.shape
        )
        electric = sum(
            electric_coeff * model.gauge_group.casimir(int(link))
            for link in (*h_links.reshape(-1), *v_links.reshape(-1))
        )
        hamiltonian = hamiltonian.at[source_idx, source_idx].set(electric)
        for row in range(model.shape[0] - 1):
            for col in range(model.shape[1] - 1):
                for candidate, me in plaquette_outcomes(
                    model, sample, row=row, col=col
                ):
                    target_idx = sample_keys[tuple(candidate.tolist())]
                    hamiltonian = hamiltonian.at[target_idx, source_idx].add(
                        plaquette_coeff * me
                    )
    return samples, hamiltonian

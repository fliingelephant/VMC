"""Group-generic Schur-factor construction for string/loop operators.

Every off-diagonal term in this family is a string or loop of link operators
in one irrep ``R``.  Its spin-network matrix element factorizes over the
visited vertices (``docs/superpowers/specs/2026-07-06-non-abelian-factor-
tables.md``):

    <out|O|in> = prod_x lambda_x(b_in_x, b_out_x)

The split is exact and every factor is a plain scalar: writing daggered
links (``U^dagger``, the bottom/left plaquette links) with sector-swapped
conjugated couplings of the *same* operator irrep makes every operator index
tie a corner-local bra-ket contraction, in any basis of any compact group.
The dual irrep never enters operator tensors; it only shows up implicitly in
reverse fusion for daggered legs.  This module owns the generic ring/string
oracles, the per-role factor extraction, and the ``VertexFactorTable``
builders with build-time factorization asserts.  Group backends supply only
the typed primitives ``fundamental_irrep``, ``coupling_tensor``, and
``vertex_tensor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math
from typing import Any

import jax
import numpy as np
from plum import dispatch

from vmc.peps.non_abelian_gi.tables import (
    FusionOutputs,
    PureGaugeTables,
    VertexFactorTable,
    pack_fusion_outputs,
)

__all__ = [
    "HOPPING_KEY_LEGS",
    "HOPPING_LEG_FWD",
    "HoppingFactorTables",
    "PLAQUETTE_KEY_LEGS",
    "PLAQUETTE_LEG_FWD",
    "PlaquetteFactorTables",
    "build_hopping_factor_tables",
    "build_plaquette_factor_tables",
    "coupling_tensor",
    "fundamental_irrep",
    "vertex_tensor",
]

_SPLIT_ATOL = 1e-9
_DROP_TOL = 1e-12

# Window geometry shared by table builders, kernels, and exact-enumeration
# tests.  A plaquette window moves four links, ordered (top', right',
# bottom', left'); each vertex table key reads its two touched legs at
# ``KEY_LEGS`` positions, and ``LEG_FWD`` says whether that moved link is an
# undaggered U (top/right, forward fusion with op) or a daggered U^dagger
# (bottom/left, reverse fusion) under the forward orientation.  Hopping
# windows move one link.  Vertex order: (tl, tr, bl, br) / (src, tgt).
PLAQUETTE_KEY_LEGS = ((0, 3), (0, 1), (3, 2), (2, 1))
PLAQUETTE_LEG_FWD = (True, True, False, False)
HOPPING_KEY_LEGS = ((0,), (0,))
HOPPING_LEG_FWD = (True,)

# Intertwiner axes touched by the plaquette loop at each corner, in the
# (l, u, r, d[, matter]) axis convention of vertex tensors.
_PLAQUETTE_AXES = {"tl": (2, 3), "tr": (0, 3), "bl": (1, 2), "br": (0, 1)}
# Touched legs per corner in key-axis order.
_PLAQUETTE_TOUCHED = {
    "tl": ("j_r", "j_d"),
    "tr": ("j_l", "j_d"),
    "bl": ("j_u", "j_r"),
    "br": ("j_l", "j_u"),
}
_PLAQUETTE_LEG_IS_U = {
    role: tuple(PLAQUETTE_LEG_FWD[p] for p in legs)
    for role, legs in zip(("tl", "tr", "bl", "br"), PLAQUETTE_KEY_LEGS)
}
_LEG_KEY_INDEX = {"j_l": 1, "j_u": 2, "j_r": 3, "j_d": 4}


# ---------- group primitives (typed dispatch, registered by backends) ----------


@dispatch
def fundamental_irrep(group: object) -> int:
    """Return the irrep label carried by the Hamiltonian link operators."""
    raise NotImplementedError(f"No fundamental irrep for {type(group)!r}.")


@dispatch
def coupling_tensor(group: object, out: int, op: int, inp: int) -> jax.Array:
    """Return the isometric coupling ``<out m_o|op m_a, in m_i>``.

    Shape ``(dim out, dim op, dim in)`` with orthonormal rows (the raw
    Wigner coupling, ``sum_{a,i} |C[o,a,i]|^2 = 1`` per ``o``); backends fix
    its phase convention once and use it everywhere.  Phases cancel exactly
    in every matrix element because each coupling enters once plain and once
    conjugated per link or matter operator.
    """
    raise NotImplementedError(f"No coupling tensor for {type(group)!r}.")


@dispatch
def vertex_tensor(group: object, block: object) -> jax.Array:
    """Return the orthonormal magnetic intertwiner tensor of one block."""
    raise NotImplementedError(f"No vertex tensor for {type(group)!r}.")


# ---------- generic constructions ----------


def _u_src_half(group: Any, in_irrep: int, out_irrep: int, op: int) -> np.ndarray:
    """Src half ``(out_m, op, in_m)`` of ``U``, carrying the link prefactor."""
    return math.sqrt(group.dim(in_irrep) / group.dim(out_irrep)) * np.asarray(
        coupling_tensor(group, out_irrep, op, in_irrep)
    )


def _u_tgt_half(group: Any, in_irrep: int, out_irrep: int, op: int) -> np.ndarray:
    """Tgt half ``(out_m, op, in_m)`` of ``U``, conjugated, no prefactor."""
    return np.asarray(coupling_tensor(group, out_irrep, op, in_irrep)).conj()


def _udag_src_half(group: Any, in_irrep: int, out_irrep: int, op: int) -> np.ndarray:
    """Src half ``(in_m, op, out_m)`` of ``U^dagger`` with its prefactor."""
    return (
        math.sqrt(group.dim(out_irrep) / group.dim(in_irrep))
        * np.asarray(coupling_tensor(group, in_irrep, op, out_irrep)).conj()
    )


def _udag_tgt_half(group: Any, in_irrep: int, out_irrep: int, op: int) -> np.ndarray:
    """Tgt half ``(in_m, op, out_m)`` of ``U^dagger``, plain."""
    return np.asarray(coupling_tensor(group, in_irrep, op, out_irrep))


def vertex_overlap(
    group: Any,
    block_in: Any,
    block_out: Any,
    *,
    affected_axes: tuple[int, ...],
) -> np.ndarray:
    """Overlap of two intertwiners over all axes except ``affected_axes``.

    Returns ``(out affected dims..., in affected dims...)`` with the output
    tensor conjugated.  Pure-gauge tensors get a trivial matter axis when
    axis 4 is affected.
    """
    input_tensor = np.asarray(vertex_tensor(group, block_in))
    output_tensor = np.asarray(vertex_tensor(group, block_out))
    if 4 in affected_axes:
        if input_tensor.ndim == 4:
            input_tensor = input_tensor[..., None]
        if output_tensor.ndim == 4:
            output_tensor = output_tensor[..., None]
    external_axes = tuple(
        axis for axis in range(output_tensor.ndim) if axis not in affected_axes
    )
    out_dims = tuple(output_tensor.shape[axis] for axis in affected_axes)
    in_dims = tuple(input_tensor.shape[axis] for axis in affected_axes)
    ext_dim = math.prod(output_tensor.shape[axis] for axis in external_axes)
    output_flat = np.transpose(output_tensor, affected_axes + external_axes).reshape(
        math.prod(out_dims), ext_dim
    )
    input_flat = np.transpose(input_tensor, affected_axes + external_axes).reshape(
        math.prod(in_dims), ext_dim
    )
    return (output_flat.conj() @ input_flat.T).reshape(*out_dims, *in_dims)


# ---------- oracles (build-time asserts and tests only) ----------


def _oriented_plaquette_matrix_element(
    group: Any,
    input_blocks: tuple[Any, Any, Any, Any],
    output_blocks: tuple[Any, Any, Any, Any],
    op: int,
) -> complex:
    """Ring contraction of ``<out|U_square|in>`` in one einsum.

    Top/right links carry ``U`` in irrep ``op``; bottom/left carry
    ``U^dagger``.  All operator ties are corner-local.
    """
    tl_in, tr_in, bl_in, br_in = input_blocks
    tl_out, tr_out, bl_out, br_out = output_blocks
    overlaps = tuple(
        vertex_overlap(group, b_in, b_out, affected_axes=_PLAQUETTE_AXES[role])
        for role, b_in, b_out in zip(
            ("tl", "tr", "bl", "br"), input_blocks, output_blocks, strict=True
        )
    )
    return complex(
        np.einsum(
            "ABab,Axa,bxB,CDcd,Cyc,Dyd,EFef,ezE,Fzf,GHgh,gwG,hwH->",
            overlaps[0],
            _u_src_half(group, tl_in.j_r, tl_out.j_r, op),
            _udag_src_half(group, tl_in.j_d, tl_out.j_d, op),
            overlaps[1],
            _u_tgt_half(group, tl_in.j_r, tl_out.j_r, op),
            _u_src_half(group, tr_in.j_d, tr_out.j_d, op),
            overlaps[3],
            _udag_tgt_half(group, bl_in.j_r, bl_out.j_r, op),
            _u_tgt_half(group, tr_in.j_d, tr_out.j_d, op),
            overlaps[2],
            _udag_tgt_half(group, tl_in.j_d, tl_out.j_d, op),
            _udag_src_half(group, bl_in.j_r, bl_out.j_r, op),
            # greedy: optimal path search is exponential in the 12 operands
            optimize="greedy",
        )
    )


def _oriented_hopping_matrix_element(
    group: Any,
    input_blocks: tuple[Any, Any],
    output_blocks: tuple[Any, Any],
    op: int,
    *,
    horizontal: bool,
) -> complex:
    """``<out|psi^dag_src U_link psi_tgt|in>``: matter created at the src site."""
    src_in, tgt_in = input_blocks
    src_out, tgt_out = output_blocks
    src_axes, tgt_axes = ((2, 4), (0, 4)) if horizontal else ((3, 4), (1, 4))
    link_in = src_in.j_r if horizontal else src_in.j_d
    link_out = src_out.j_r if horizontal else src_out.j_d
    return complex(
        np.einsum(
            "PMpm,Pxp,Mxm,QNqn,Qyq,nyN->",
            vertex_overlap(group, src_in, src_out, affected_axes=src_axes),
            _u_src_half(group, link_in, link_out, op),
            np.asarray(
                coupling_tensor(group, src_out.matter_irrep, op, src_in.matter_irrep)
            ),
            vertex_overlap(group, tgt_in, tgt_out, affected_axes=tgt_axes),
            _u_tgt_half(group, link_in, link_out, op),
            np.asarray(
                coupling_tensor(group, tgt_in.matter_irrep, op, tgt_out.matter_irrep)
            ).conj(),
            optimize="greedy",
        )
    )


# ---------- factor extraction (all corner factors are scalars) ----------


@cache
def _plaquette_corner_factor(
    group: Any,
    role: str,
    block_in: Any,
    block_out: Any,
    op: int,
) -> complex:
    """Return ``lambda_role(block_in -> block_out)`` of the oriented loop.

    Exact regrouping of the ring einsum: every operator tie is corner-local,
    so each corner contracts to a plain scalar.
    """
    overlap = vertex_overlap(
        group, block_in, block_out, affected_axes=_PLAQUETTE_AXES[role]
    )
    if role == "tl":
        return complex(
            np.einsum(
                "ABab,Axa,bxB->",
                overlap,
                _u_src_half(group, block_in.j_r, block_out.j_r, op),
                _udag_src_half(group, block_in.j_d, block_out.j_d, op),
                optimize=True,
            )
        )
    if role == "tr":
        return complex(
            np.einsum(
                "CDcd,Cyc,Dyd->",
                overlap,
                _u_tgt_half(group, block_in.j_l, block_out.j_l, op),
                _u_src_half(group, block_in.j_d, block_out.j_d, op),
                optimize=True,
            )
        )
    if role == "bl":
        return complex(
            np.einsum(
                "GHgh,gwG,hwH->",
                overlap,
                _udag_tgt_half(group, block_in.j_u, block_out.j_u, op),
                _udag_src_half(group, block_in.j_r, block_out.j_r, op),
                optimize=True,
            )
        )
    return complex(
        np.einsum(
            "EFef,ezE,Fzf->",
            overlap,
            _udag_tgt_half(group, block_in.j_l, block_out.j_l, op),
            _u_tgt_half(group, block_in.j_u, block_out.j_u, op),
            optimize=True,
        )
    )


@cache
def _hopping_endpoint_factor(
    group: Any,
    role: str,
    block_in: Any,
    block_out: Any,
    op: int,
    *,
    horizontal: bool,
) -> complex:
    """Return the endpoint scalar of the oriented hop (creation at src)."""
    if role == "src":
        link_in = block_in.j_r if horizontal else block_in.j_d
        link_out = block_out.j_r if horizontal else block_out.j_d
        return complex(
            np.einsum(
                "PMpm,Pxp,Mxm->",
                vertex_overlap(
                    group,
                    block_in,
                    block_out,
                    affected_axes=(2, 4) if horizontal else (3, 4),
                ),
                _u_src_half(group, link_in, link_out, op),
                np.asarray(
                    coupling_tensor(
                        group, block_out.matter_irrep, op, block_in.matter_irrep
                    )
                ),
                optimize=True,
            )
        )
    link_in = block_in.j_l if horizontal else block_in.j_u
    link_out = block_out.j_l if horizontal else block_out.j_u
    return complex(
        np.einsum(
            "QNqn,Qyq,nyN->",
            vertex_overlap(
                group,
                block_in,
                block_out,
                affected_axes=(0, 4) if horizontal else (1, 4),
            ),
            _u_tgt_half(group, link_in, link_out, op),
            np.asarray(
                coupling_tensor(
                    group, block_in.matter_irrep, op, block_out.matter_irrep
                )
            ).conj(),
            optimize=True,
        )
    )


# ---------- table builders ----------


@dataclass(frozen=True)
class PlaquetteFactorTables:
    """Per-site plaquette corner factors, both orientations.

    Grids are ``[r][c] -> (fwd, bwd) | None`` where fwd holds
    ``lambda(b_in -> b_out)`` of ``U_square`` and bwd of ``U_square^dagger``
    (``lambda_bwd(in -> out) = conj(lambda_fwd(out -> in))``).  Key order is
    the axis-ordered touched legs: tl ``(top', left')``, tr ``(top',
    right')``, bl ``(left', bottom')``, br ``(bottom', right')``.  Under
    fwd, top/right legs fuse forward with ``op`` and bottom/left in reverse;
    bwd swaps the two fusion tables.  The full matrix element is the plain
    product of the four corner factors.
    """

    fuse_op: FusionOutputs
    fuse_rev: FusionOutputs
    tl: tuple
    tr: tuple
    bl: tuple
    br: tuple


@dataclass(frozen=True)
class HoppingFactorTables:
    """Per-site hopping endpoint factors, both orientations.

    fwd creates matter at the src site (link fuses forward with ``op``); bwd
    is the Hermitian conjugate (reverse fusion).  Key = new link irrep; the
    matrix element is ``lambda_src * lambda_tgt``.
    """

    fuse_op: FusionOutputs
    fuse_rev: FusionOutputs
    h_src: tuple
    h_tgt: tuple
    v_src: tuple
    v_tgt: tuple


def _factor_rows(
    tables: PureGaugeTables,
    r: int,
    c: int,
    *,
    touched: tuple[str, ...],
    leg_outputs: tuple[dict, ...],
    factor: Any,
    matter_shift: int | None = None,
) -> dict[tuple[int, ...], list[tuple[int, complex]]]:
    """Enumerate factor-table rows for one (site, role, orientation).

    ``factor(block_in, block_out)`` computes the scalar.  Candidates share
    all untouched legs, take touched legs from ``leg_outputs`` (forward or
    reverse fusion maps), and keep the matter state (``matter_shift=None``)
    or shift its occupation number by ``matter_shift``.
    """
    site_blocks = tables.blocks[r][c]
    grouped: dict[tuple[int, int, int, int, int], list[int]] = {}
    for block_id, block in enumerate(site_blocks):
        site_key = (
            block.matter_state,
            block.j_l,
            block.j_u,
            block.j_r,
            block.j_d,
        )
        grouped.setdefault(site_key, []).append(block_id)
    rows: dict[tuple[int, ...], list[tuple[int, complex]]] = {}
    for block_id, block in enumerate(site_blocks):
        state_in = block.matter_state
        if matter_shift is None:
            out_states: tuple[int, ...] = (state_in,)
        else:
            target = tables.matter_numbers[state_in] + matter_shift
            out_states = tuple(
                state
                for state, number in enumerate(tables.matter_numbers)
                if number == target
            )
        combos: list[tuple[int, ...]] = [()]
        for leg, outputs in zip(touched, leg_outputs, strict=True):
            leg_in = getattr(block, leg)
            combos = [combo + (out,) for combo in combos for out in outputs[leg_in]]
        for legs_out in combos:
            candidates: list[tuple[int, complex]] = []
            for state in out_states:
                key = [state, block.j_l, block.j_u, block.j_r, block.j_d]
                for leg, new_irrep in zip(touched, legs_out, strict=True):
                    key[_LEG_KEY_INDEX[leg]] = new_irrep
                for out_id in grouped.get(tuple(key), ()):
                    value = factor(block, site_blocks[out_id])
                    if abs(value) > _DROP_TOL:
                        candidates.append((out_id, value))
            if candidates:
                rows[(block_id, *legs_out)] = candidates
    return rows


def _fusion_maps(group: Any, op: int) -> tuple[dict, dict]:
    irreps = group.irreps()
    forward = {j: group.fuse(j, op) for j in irreps}
    reverse = {j: tuple(k for k in irreps if j in forward[k]) for j in irreps}
    return forward, reverse


def build_plaquette_factor_tables(
    group: Any, tables: PureGaugeTables
) -> PlaquetteFactorTables:
    op = fundamental_irrep(group)
    forward_map, reverse_map = _fusion_maps(group, op)
    n_rows, n_cols = tables.shape
    n_irreps = len(group.irreps())
    in_plaquette = {
        "tl": lambda r, c: r < n_rows - 1 and c < n_cols - 1,
        "tr": lambda r, c: r < n_rows - 1 and c > 0,
        "bl": lambda r, c: r > 0 and c < n_cols - 1,
        "br": lambda r, c: r > 0 and c > 0,
    }
    rows: dict[tuple[str, str, int, int], dict] = {}
    for role, leg_is_u in _PLAQUETTE_LEG_IS_U.items():
        for orientation in ("fwd", "bwd"):
            forward = orientation == "fwd"
            leg_outputs = tuple(
                (forward_map if is_u == forward else reverse_map) for is_u in leg_is_u
            )

            def corner(
                b_in: Any, b_out: Any, role: str = role, forward: bool = forward
            ) -> complex:
                if forward:
                    return _plaquette_corner_factor(group, role, b_in, b_out, op)
                return complex(
                    np.conj(_plaquette_corner_factor(group, role, b_out, b_in, op))
                )

            for r in range(n_rows):
                for c in range(n_cols):
                    if in_plaquette[role](r, c):
                        rows[(role, orientation, r, c)] = _factor_rows(
                            tables,
                            r,
                            c,
                            touched=_PLAQUETTE_TOUCHED[role],
                            leg_outputs=leg_outputs,
                            factor=corner,
                        )
    _assert_plaquette_split(group, tables, rows, op)

    def grid(role: str) -> tuple:
        return tuple(
            tuple(
                tuple(
                    VertexFactorTable.from_rows(
                        tables.n_blocks(r, c),
                        n_irreps,
                        2,
                        rows[(role, orientation, r, c)],
                        factor_dtype=np.complex128,
                    )
                    for orientation in ("fwd", "bwd")
                )
                if in_plaquette[role](r, c)
                else None
                for c in range(n_cols)
            )
            for r in range(n_rows)
        )

    return PlaquetteFactorTables(
        fuse_op=pack_fusion_outputs(n_irreps, forward_map),
        fuse_rev=pack_fusion_outputs(n_irreps, reverse_map),
        tl=grid("tl"),
        tr=grid("tr"),
        bl=grid("bl"),
        br=grid("br"),
    )


def _assert_plaquette_split(
    group: Any,
    tables: PureGaugeTables,
    rows: dict,
    op: int,
) -> None:
    """Assert ``ring == prod lambda`` on a few joined instances."""
    n_rows, n_cols = tables.shape
    checked = 0
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            tl_blocks = tables.blocks[r][c]
            tr_blocks = tables.blocks[r][c + 1]
            bl_blocks = tables.blocks[r + 1][c]
            br_blocks = tables.blocks[r + 1][c + 1]
            tr_index: dict[tuple[int, int], list] = {}
            for (tr_id, top_new, right_new), cands in rows[
                ("tr", "fwd", r, c + 1)
            ].items():
                tr_index.setdefault((top_new, tr_blocks[tr_id].j_l), []).append(
                    (tr_id, right_new, cands)
                )
            bl_index: dict[tuple[int, int], list] = {}
            for (bl_id, left_new, bottom_new), cands in rows[
                ("bl", "fwd", r + 1, c)
            ].items():
                bl_index.setdefault((left_new, bl_blocks[bl_id].j_u), []).append(
                    (bl_id, bottom_new, cands)
                )
            br_index: dict[tuple[int, int, int, int], list] = {}
            for (br_id, bottom_new, right_new), cands in rows[
                ("br", "fwd", r + 1, c + 1)
            ].items():
                block = br_blocks[br_id]
                br_index.setdefault(
                    (bottom_new, right_new, block.j_l, block.j_u), []
                ).append((br_id, cands))
            for (tl_id, top_new, left_new), tl_cands in rows[
                ("tl", "fwd", r, c)
            ].items():
                tl_in = tl_blocks[tl_id]
                for tr_id, right_new, tr_cands in tr_index.get(
                    (top_new, tl_in.j_r), ()
                ):
                    tr_in = tr_blocks[tr_id]
                    for bl_id, bottom_new, bl_cands in bl_index.get(
                        (left_new, tl_in.j_d), ()
                    ):
                        bl_in = bl_blocks[bl_id]
                        for br_id, br_cands in br_index.get(
                            (bottom_new, right_new, bl_in.j_r, tr_in.j_d), ()
                        ):
                            ring = _oriented_plaquette_matrix_element(
                                group,
                                (tl_in, tr_in, bl_in, br_blocks[br_id]),
                                (
                                    tl_blocks[tl_cands[0][0]],
                                    tr_blocks[tr_cands[0][0]],
                                    bl_blocks[bl_cands[0][0]],
                                    br_blocks[br_cands[0][0]],
                                ),
                                op,
                            )
                            product = (
                                tl_cands[0][1]
                                * tr_cands[0][1]
                                * bl_cands[0][1]
                                * br_cands[0][1]
                            )
                            if abs(ring - product) > _SPLIT_ATOL * max(1.0, abs(ring)):
                                raise AssertionError(
                                    f"Plaquette split inconsistency: ring {ring}"
                                    f" != prod(lambda) {product}."
                                )
                            checked += 1
                            if checked >= 3:
                                return


def build_hopping_factor_tables(
    group: Any, tables: PureGaugeTables
) -> HoppingFactorTables:
    op = fundamental_irrep(group)
    forward_map, reverse_map = _fusion_maps(group, op)
    n_rows, n_cols = tables.shape
    n_irreps = len(group.irreps())
    specs = {
        "h_src": ("src", True, lambda r, c: c < n_cols - 1, "j_r"),
        "h_tgt": ("tgt", True, lambda r, c: c > 0, "j_l"),
        "v_src": ("src", False, lambda r, c: r < n_rows - 1, "j_d"),
        "v_tgt": ("tgt", False, lambda r, c: r > 0, "j_u"),
    }
    rows: dict[tuple[str, str, int, int], dict] = {}
    if tables.phys_dim > 1:
        for name, (role, horizontal, in_lattice, leg) in specs.items():
            fwd_shift = 1 if role == "src" else -1
            for orientation in ("fwd", "bwd"):
                forward = orientation == "fwd"

                def endpoint(
                    b_in: Any,
                    b_out: Any,
                    role: str = role,
                    horizontal: bool = horizontal,
                    forward: bool = forward,
                ) -> complex:
                    if forward:
                        return _hopping_endpoint_factor(
                            group, role, b_in, b_out, op, horizontal=horizontal
                        )
                    return complex(
                        np.conj(
                            _hopping_endpoint_factor(
                                group, role, b_out, b_in, op, horizontal=horizontal
                            )
                        )
                    )

                for r in range(n_rows):
                    for c in range(n_cols):
                        if in_lattice(r, c):
                            rows[(name, orientation, r, c)] = _factor_rows(
                                tables,
                                r,
                                c,
                                touched=(leg,),
                                leg_outputs=(forward_map if forward else reverse_map,),
                                factor=endpoint,
                                matter_shift=fwd_shift if forward else -fwd_shift,
                            )
        _assert_hopping_split(group, tables, rows, op)

    def grid(name: str) -> tuple:
        in_lattice = specs[name][2]
        return tuple(
            tuple(
                tuple(
                    VertexFactorTable.from_rows(
                        tables.n_blocks(r, c),
                        n_irreps,
                        1,
                        rows[(name, orientation, r, c)],
                        factor_dtype=np.complex128,
                    )
                    for orientation in ("fwd", "bwd")
                )
                if tables.phys_dim > 1 and in_lattice(r, c)
                else None
                for c in range(n_cols)
            )
            for r in range(n_rows)
        )

    return HoppingFactorTables(
        fuse_op=pack_fusion_outputs(n_irreps, forward_map),
        fuse_rev=pack_fusion_outputs(n_irreps, reverse_map),
        h_src=grid("h_src"),
        h_tgt=grid("h_tgt"),
        v_src=grid("v_src"),
        v_tgt=grid("v_tgt"),
    )


def _assert_hopping_split(
    group: Any,
    tables: PureGaugeTables,
    rows: dict,
    op: int,
) -> None:
    """Assert ``oriented hop == lambda_src * lambda_tgt`` on a few instances."""
    n_rows, n_cols = tables.shape
    for horizontal, src_name, tgt_name in (
        (True, "h_src", "h_tgt"),
        (False, "v_src", "v_tgt"),
    ):
        checked = 0
        for r in range(n_rows if horizontal else n_rows - 1):
            if checked >= 3:
                break
            for c in range(n_cols - 1 if horizontal else n_cols):
                if checked >= 3:
                    break
                tgt_r, tgt_c = (r, c + 1) if horizontal else (r + 1, c)
                src_blocks = tables.blocks[r][c]
                tgt_blocks = tables.blocks[tgt_r][tgt_c]
                tgt_index: dict[tuple[int, int], list] = {}
                for (tgt_id, link_new), cands in rows[
                    (tgt_name, "fwd", tgt_r, tgt_c)
                ].items():
                    block = tgt_blocks[tgt_id]
                    link_in = block.j_l if horizontal else block.j_u
                    tgt_index.setdefault((link_new, link_in), []).append(
                        (tgt_id, cands)
                    )
                for (src_id, link_new), src_cands in rows[
                    (src_name, "fwd", r, c)
                ].items():
                    if checked >= 3:
                        break
                    src_in = src_blocks[src_id]
                    link_in = src_in.j_r if horizontal else src_in.j_d
                    for tgt_id, tgt_cands in tgt_index.get((link_new, link_in), ()):
                        oriented = _oriented_hopping_matrix_element(
                            group,
                            (src_in, tgt_blocks[tgt_id]),
                            (
                                src_blocks[src_cands[0][0]],
                                tgt_blocks[tgt_cands[0][0]],
                            ),
                            op,
                            horizontal=horizontal,
                        )
                        product = src_cands[0][1] * tgt_cands[0][1]
                        if abs(oriented - product) > _SPLIT_ATOL * max(
                            1.0, abs(oriented)
                        ):
                            raise AssertionError(
                                f"Hopping split inconsistency: oriented {oriented}"
                                f" != lambda_src*lambda_tgt {product}."
                            )
                        checked += 1
                        if checked >= 3:
                            break

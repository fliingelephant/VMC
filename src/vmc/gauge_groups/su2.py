"""Truncated SU(2) gauge-group backend for non-Abelian GI-PEPS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math

import jax
import jax.numpy as jnp
import numpy as np

from vmc.peps.non_abelian_gi.builders import build_pure_gauge_tables
from vmc.peps.non_abelian_gi.factors import (
    coupling_tensor,
    fundamental_irrep,
    vertex_tensor,
)
from vmc.peps.non_abelian_gi.tables import PureGaugeTables


@dataclass(frozen=True)
class SU2:
    """Truncated SU(2) irreps using integer labels ``j_twice = 2j``."""

    j_max_twice: int

    def __post_init__(self) -> None:
        if not isinstance(self.j_max_twice, int):
            raise ValueError("j_max_twice must be an integer.")
        if self.j_max_twice < 0:
            raise ValueError("j_max_twice must be non-negative.")

    def irreps(self) -> tuple[int, ...]:
        """Return link irrep labels inside the Hilbert-space truncation."""
        return tuple(range(self.j_max_twice + 1))

    def dim(self, j_twice: int) -> int:
        """Return ``2j + 1`` for an allowed truncated irrep."""
        self._validate_link_irrep(j_twice)
        return j_twice + 1

    def casimir(self, j_twice: int) -> float:
        """Return ``j(j + 1)``."""
        self._validate_link_irrep(j_twice)
        return 0.25 * j_twice * (j_twice + 2)

    def fuse(self, a_twice: int, b_twice: int) -> tuple[int, ...]:
        """Return truncated fusion outputs of a link irrep with any irrep.

        The second factor is an operator irrep and is not cut off by the
        link truncation; only the outputs are.
        """
        self._validate_link_irrep(a_twice)
        if not isinstance(b_twice, int) or b_twice < 0:
            raise ValueError("Expected a valid SU(2) irrep label.")
        return tuple(
            c_twice
            for c_twice in _fuse_untruncated(a_twice, b_twice)
            if c_twice <= self.j_max_twice
        )

    def tensor_product(self, a_twice: int, b_twice: int) -> tuple[int, ...]:
        """Return untruncated SU(2) tensor-product outputs.

        Vertex-internal fusion channels are not link Hilbert-space sectors, so
        they are not cut off by ``j_max_twice``.
        """
        self._validate_link_irrep(a_twice)
        self._validate_link_irrep(b_twice)
        return _fuse_untruncated(a_twice, b_twice)

    def _validate_link_irrep(self, j_twice: int) -> None:
        if not isinstance(j_twice, int):
            raise ValueError("SU(2) irrep labels must be integers.")
        if j_twice < 0:
            raise ValueError("Expected a valid SU(2) irrep label.")
        if j_twice > self.j_max_twice:
            raise ValueError("SU(2) irrep label must be within truncation.")


@dataclass(frozen=True, order=True)
class VertexBlock:
    """One vertex block in the sampled spin-network basis."""

    j_l: int
    j_u: int
    j_r: int
    j_d: int
    iota: int
    internal_irreps: tuple[int, ...]
    matter_state: int = 0
    matter_irrep: int = 0
    matter_number: int = 0


def _fuse_untruncated(a_twice: int, b_twice: int) -> tuple[int, ...]:
    return tuple(range(abs(a_twice - b_twice), a_twice + b_twice + 1, 2))


def clebsch_gordan(
    j1_twice: int,
    m1_twice: int,
    j2_twice: int,
    m2_twice: int,
    j_out_twice: int,
    m_out_twice: int,
) -> float:
    """Return a Condon-Shortley SU(2) Clebsch-Gordan coefficient."""
    if not (
        _is_valid_magnetic_label(j1_twice, m1_twice)
        and _is_valid_magnetic_label(j2_twice, m2_twice)
        and _is_valid_magnetic_label(j_out_twice, m_out_twice)
    ):
        return 0.0
    if m_out_twice != m1_twice + m2_twice:
        return 0.0
    if j_out_twice not in _fuse_untruncated(j1_twice, j2_twice):
        return 0.0

    triangle_prefactor = (
        (j_out_twice + 1)
        * math.factorial(_half_sum(j_out_twice, j1_twice, -j2_twice))
        * math.factorial(_half_sum(j_out_twice, -j1_twice, j2_twice))
        * math.factorial(_half_sum(j1_twice, j2_twice, -j_out_twice))
        / math.factorial(_half_sum(j1_twice, j2_twice, j_out_twice) + 1)
    )
    magnetic_prefactor = (
        math.factorial(_half_sum(j_out_twice, m_out_twice))
        * math.factorial(_half_sum(j_out_twice, -m_out_twice))
        * math.factorial(_half_sum(j1_twice, -m1_twice))
        * math.factorial(_half_sum(j1_twice, m1_twice))
        * math.factorial(_half_sum(j2_twice, -m2_twice))
        * math.factorial(_half_sum(j2_twice, m2_twice))
    )

    a = _half_sum(j1_twice, j2_twice, -j_out_twice)
    b = _half_sum(j1_twice, -m1_twice)
    c = _half_sum(j2_twice, m2_twice)
    d = _half_sum(j_out_twice, -j2_twice, m1_twice)
    e = _half_sum(j_out_twice, -j1_twice, -m2_twice)
    total = 0.0
    for k in range(max(0, -d, -e), min(a, b, c) + 1):
        sign = -1.0 if k % 2 else 1.0
        denom = (
            math.factorial(k)
            * math.factorial(a - k)
            * math.factorial(b - k)
            * math.factorial(c - k)
            * math.factorial(d + k)
            * math.factorial(e + k)
        )
        total += sign / denom
    return float(jnp.sqrt(triangle_prefactor * magnetic_prefactor) * total)


@cache
def vertex_intertwiner_tensor(block: VertexBlock) -> jnp.ndarray:
    """Return the canonical oriented singlet intertwiner tensor.

    Lattice links are oriented right/down. At a vertex, the left/up legs are
    incoming dual legs and the right/down legs are outgoing legs. Matter blocks
    carry one additional matter magnetic axis at the end; the matter slot
    transforms like an incoming leg and is dualized the same way.
    """
    if len(block.internal_irreps) == 2 and block.matter_irrep == 0:
        return _pure_vertex_intertwiner_tensor(block)
    if len(block.internal_irreps) == 3:
        return _matter_vertex_intertwiner_tensor(block)
    raise ValueError("Unexpected SU(2) vertex-block fusion path.")


def _pure_vertex_intertwiner_tensor(block: VertexBlock) -> jnp.ndarray:
    j_mid, j_pair = block.internal_irreps
    tensor = np.zeros(
        (
            block.j_l + 1,
            block.j_u + 1,
            block.j_r + 1,
            block.j_d + 1,
        ),
        dtype=np.float64,
    )
    for l_idx, m_l in enumerate(_magnetic_labels(block.j_l)):
        for u_idx, m_u in enumerate(_magnetic_labels(block.j_u)):
            for r_idx, m_r in enumerate(_magnetic_labels(block.j_r)):
                for d_idx, m_d in enumerate(_magnetic_labels(block.j_d)):
                    value = 0.0
                    for m_mid in _magnetic_labels(j_mid):
                        for m_pair in _magnetic_labels(j_pair):
                            value += (
                                clebsch_gordan(
                                    block.j_l, m_l, block.j_u, m_u, j_mid, m_mid
                                )
                                * clebsch_gordan(
                                    block.j_r, m_r, block.j_d, m_d, j_pair, m_pair
                                )
                                * clebsch_gordan(j_mid, m_mid, j_pair, m_pair, 0, 0)
                            )
                    tensor[l_idx, u_idx, r_idx, d_idx] = value
    return jnp.einsum(
        "la,ub,abrd->lurd",
        _dual_metric(block.j_l),
        _dual_metric(block.j_u),
        tensor,
        optimize=True,
    )


def _matter_vertex_intertwiner_tensor(block: VertexBlock) -> jnp.ndarray:
    j_mid, j_pair, j_gauge = block.internal_irreps
    tensor = np.zeros(
        (
            block.j_l + 1,
            block.j_u + 1,
            block.j_r + 1,
            block.j_d + 1,
            block.matter_irrep + 1,
        ),
        dtype=np.float64,
    )
    for l_idx, m_l in enumerate(_magnetic_labels(block.j_l)):
        for u_idx, m_u in enumerate(_magnetic_labels(block.j_u)):
            for r_idx, m_r in enumerate(_magnetic_labels(block.j_r)):
                for d_idx, m_d in enumerate(_magnetic_labels(block.j_d)):
                    for q_idx, m_q in enumerate(_magnetic_labels(block.matter_irrep)):
                        value = 0.0
                        for m_mid in _magnetic_labels(j_mid):
                            for m_pair in _magnetic_labels(j_pair):
                                for m_gauge in _magnetic_labels(j_gauge):
                                    value += (
                                        clebsch_gordan(
                                            block.j_l, m_l, block.j_u, m_u, j_mid, m_mid
                                        )
                                        * clebsch_gordan(
                                            block.j_r,
                                            m_r,
                                            block.j_d,
                                            m_d,
                                            j_pair,
                                            m_pair,
                                        )
                                        * clebsch_gordan(
                                            j_mid,
                                            m_mid,
                                            j_pair,
                                            m_pair,
                                            j_gauge,
                                            m_gauge,
                                        )
                                        * clebsch_gordan(
                                            j_gauge,
                                            m_gauge,
                                            block.matter_irrep,
                                            m_q,
                                            0,
                                            0,
                                        )
                                    )
                        tensor[l_idx, u_idx, r_idx, d_idx, q_idx] = value
    return jnp.einsum(
        "la,ub,qv,abrdv->lurdq",
        _dual_metric(block.j_l),
        _dual_metric(block.j_u),
        _dual_metric(block.matter_irrep),
        tensor,
        optimize=True,
    )


def _magnetic_labels(j_twice: int) -> tuple[int, ...]:
    return tuple(range(-j_twice, j_twice + 1, 2))


@cache
def _dual_metric(j_twice: int) -> jax.Array:
    metric = np.zeros((j_twice + 1, j_twice + 1), dtype=np.float64)
    labels = _magnetic_labels(j_twice)
    prefactor = math.sqrt(j_twice + 1)
    for row, m_row in enumerate(labels):
        for col, m_col in enumerate(labels):
            metric[row, col] = prefactor * clebsch_gordan(
                j_twice,
                m_row,
                j_twice,
                m_col,
                0,
                0,
            )
    return jnp.asarray(metric)


@fundamental_irrep.dispatch
def fundamental_irrep(group: SU2) -> int:
    return 1


@coupling_tensor.dispatch
def coupling_tensor(group: SU2, out: int, op: int, inp: int) -> jax.Array:
    return _coupling_tensor(out, op, inp)


@vertex_tensor.dispatch
def vertex_tensor(group: SU2, block: VertexBlock) -> jax.Array:
    return vertex_intertwiner_tensor(block)


@cache
def _coupling_tensor(out: int, op: int, inp: int) -> jax.Array:
    """Isometric coupling ``<out m_o|op m_a, in m_i>`` (raw Clebsch-Gordan)."""
    tensor = np.zeros((out + 1, op + 1, inp + 1), dtype=np.float64)
    for o_idx, m_o in enumerate(_magnetic_labels(out)):
        for a_idx, m_a in enumerate(_magnetic_labels(op)):
            for i_idx, m_i in enumerate(_magnetic_labels(inp)):
                tensor[o_idx, a_idx, i_idx] = clebsch_gordan(
                    op, m_a, inp, m_i, out, m_o
                )
    return jnp.asarray(tensor)


def _is_valid_magnetic_label(j_twice: int, m_twice: int) -> bool:
    return (
        isinstance(j_twice, int)
        and isinstance(m_twice, int)
        and j_twice >= 0
        and abs(m_twice) <= j_twice
        and (j_twice - m_twice) % 2 == 0
    )


def _half_sum(*twice_values: int) -> int:
    total = sum(twice_values)
    if total % 2 != 0:
        raise ValueError("Expected an integer half-sum.")
    return total // 2


def _pure_gauge_intertwiner_paths(
    group: SU2,
    *,
    j_l: int,
    j_u: int,
    j_r: int,
    j_d: int,
    target_twice: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (j_m, j_n)
        for j_m in group.tensor_product(j_l, j_u)
        for j_n in group.tensor_product(j_r, j_d)
        if target_twice in _fuse_untruncated(j_m, j_n)
    )


def _matter_intertwiner_paths(
    group: SU2,
    *,
    j_l: int,
    j_u: int,
    j_r: int,
    j_d: int,
    matter_irrep: int,
    target_twice: int,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (j_m, j_n, j_gauge)
        for j_m in group.tensor_product(j_l, j_u)
        for j_n in group.tensor_product(j_r, j_d)
        for j_gauge in _fuse_untruncated(j_m, j_n)
        if target_twice in _fuse_untruncated(j_gauge, matter_irrep)
    )


def build_pure_gauge_vertex_blocks(
    group: SU2,
    *,
    active_legs: tuple[bool, bool, bool, bool],
    target_twice: int = 0,
    matter_irreps: tuple[int, ...] = (0,),
    matter_numbers: tuple[int, ...] = (0,),
) -> tuple[VertexBlock, ...]:
    """Enumerate valid vertex blocks for one site.

    ``active_legs`` is ordered as ``(left, up, right, down)``. Inactive
    boundary legs are fixed to the singlet irrep ``j_twice = 0``.
    """
    if len(active_legs) != 4:
        raise ValueError("active_legs must be ordered as (left, up, right, down).")
    if target_twice < 0:
        raise ValueError("target_twice must be non-negative.")
    if len(matter_irreps) != len(matter_numbers):
        raise ValueError("matter_irreps and matter_numbers must have equal length.")

    link_irreps = group.irreps()
    choices = tuple(link_irreps if active else (0,) for active in active_legs)
    blocks: list[VertexBlock] = []
    for matter_state, (matter_irrep, matter_number) in enumerate(
        zip(matter_irreps, matter_numbers, strict=True)
    ):
        if matter_irrep < 0:
            raise ValueError("matter_irreps must be non-negative SU(2) labels.")
        for j_l in choices[0]:
            for j_u in choices[1]:
                for j_r in choices[2]:
                    for j_d in choices[3]:
                        if matter_irrep == 0:
                            paths = _pure_gauge_intertwiner_paths(
                                group,
                                j_l=j_l,
                                j_u=j_u,
                                j_r=j_r,
                                j_d=j_d,
                                target_twice=target_twice,
                            )
                        else:
                            paths = _matter_intertwiner_paths(
                                group,
                                j_l=j_l,
                                j_u=j_u,
                                j_r=j_r,
                                j_d=j_d,
                                matter_irrep=matter_irrep,
                                target_twice=target_twice,
                            )
                        for iota, internal_irreps in enumerate(paths):
                            blocks.append(
                                VertexBlock(
                                    j_l=j_l,
                                    j_u=j_u,
                                    j_r=j_r,
                                    j_d=j_d,
                                    iota=iota,
                                    internal_irreps=internal_irreps,
                                    matter_state=matter_state,
                                    matter_irrep=matter_irrep,
                                    matter_number=matter_number,
                                )
                            )
    return tuple(sorted(blocks))


@build_pure_gauge_tables.dispatch
def build_pure_gauge_tables(
    group: SU2,
    *,
    shape: tuple[int, int],
    target_charge: int = 0,
    matter_irreps: tuple[int, ...] = (0,),
    matter_numbers: tuple[int, ...] = (0,),
) -> PureGaugeTables:
    """Build boundary-aware vertex block tables."""
    target_twice = int(target_charge)
    n_rows, n_cols = shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("shape must have positive dimensions.")
    if len(matter_irreps) != len(matter_numbers):
        raise ValueError("matter_irreps and matter_numbers must have equal length.")

    return PureGaugeTables.from_blocks(
        group=group,
        shape=shape,
        matter_irreps=tuple(int(irrep) for irrep in matter_irreps),
        matter_numbers=tuple(int(number) for number in matter_numbers),
        blocks=tuple(
            tuple(
                build_pure_gauge_vertex_blocks(
                    group,
                    active_legs=(c > 0, r > 0, c < n_cols - 1, r < n_rows - 1),
                    target_twice=target_twice,
                    matter_irreps=matter_irreps,
                    matter_numbers=matter_numbers,
                )
                for c in range(n_cols)
            )
            for r in range(n_rows)
        ),
    )

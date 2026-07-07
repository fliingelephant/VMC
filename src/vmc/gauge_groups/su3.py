"""Fundamental-truncated SU(3) gauge-group backend for non-Abelian GI-PEPS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import product
import math

import jax
import jax.numpy as jnp

from vmc.peps.non_abelian_gi.builders import build_pure_gauge_tables
from vmc.peps.non_abelian_gi.factors import (
    coupling_tensor,
    fundamental_irrep,
    vertex_tensor,
)
from vmc.peps.non_abelian_gi.tables import PureGaugeTables


@dataclass(frozen=True)
class SU3:
    """SU(3) irreps ``(p, q)`` truncated to ``p + q <= max_weight_sum``.

    The current backend implements exact magnetic tensors for the fundamental
    truncation ``max_weight_sum=1``: singlet, fundamental, and antifundamental.
    """

    max_weight_sum: int

    def __post_init__(self) -> None:
        if not isinstance(self.max_weight_sum, int):
            raise ValueError("max_weight_sum must be an integer.")
        if self.max_weight_sum != 1:
            raise NotImplementedError(
                "SU(3) backend currently supports max_weight_sum=1."
            )

    @property
    def fundamental(self) -> int:
        return self.label((1, 0))

    @property
    def antifundamental(self) -> int:
        return self.label((0, 1))

    def irreps(self) -> tuple[int, ...]:
        return tuple(range(len(_IRREP_WEIGHTS)))

    def highest_weight(self, irrep: int) -> tuple[int, int]:
        self._validate_irrep(irrep)
        return _IRREP_WEIGHTS[irrep]

    def label(self, highest_weight: tuple[int, int]) -> int:
        if highest_weight not in _IRREP_LABELS:
            raise ValueError(f"SU(3) irrep {highest_weight} is outside truncation.")
        return _IRREP_LABELS[highest_weight]

    def dual(self, irrep: int) -> int:
        p, q = self.highest_weight(irrep)
        return self.label((q, p))

    def dim(self, irrep: int) -> int:
        p, q = self.highest_weight(irrep)
        return (p + 1) * (q + 1) * (p + q + 2) // 2

    def casimir(self, irrep: int) -> float:
        p, q = self.highest_weight(irrep)
        return (p * p + q * q + p * q + 3 * p + 3 * q) / 3.0

    def fuse(self, irrep: int, operator_irrep: int) -> tuple[int, ...]:
        self._validate_irrep(irrep)
        self._validate_irrep(operator_irrep)
        return _fuse_labels(irrep, operator_irrep)

    def _validate_irrep(self, irrep: int) -> None:
        if not isinstance(irrep, int):
            raise ValueError("SU(3) irrep labels must be integers.")
        if irrep not in self.irreps():
            raise ValueError("SU(3) irrep label is outside truncation.")


def _fuse_labels(irrep: int, operator_irrep: int) -> tuple[int, ...]:
    if irrep == 0:
        return (operator_irrep,)
    if operator_irrep == 0:
        return (irrep,)
    weight = _IRREP_WEIGHTS[irrep]
    if operator_irrep == _IRREP_LABELS[(1, 0)]:
        return _filter_truncated(_tensor_with_fundamental(weight))
    if operator_irrep == _IRREP_LABELS[(0, 1)]:
        return _filter_truncated(_tensor_with_antifundamental(weight))
    raise NotImplementedError(
        "Only fundamental and antifundamental SU(3) fusions are implemented."
    )


@dataclass(frozen=True, order=True)
class VertexBlock:
    """One pure-gauge SU(3) vertex block in the sampled spin-network basis."""

    j_l: int
    j_u: int
    j_r: int
    j_d: int
    iota: int
    internal_irreps: tuple[int, ...]
    matter_state: int = 0
    matter_irrep: int = 0
    matter_number: int = 0


_IRREP_WEIGHTS = ((0, 0), (1, 0), (0, 1))
_IRREP_LABELS = {weight: label for label, weight in enumerate(_IRREP_WEIGHTS)}


def _tensor_with_fundamental(weight: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    p, q = weight
    outputs = [(p + 1, q)]
    if p > 0:
        outputs.append((p - 1, q + 1))
    if q > 0:
        outputs.append((p, q - 1))
    return tuple(outputs)


def _tensor_with_antifundamental(
    weight: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    p, q = weight
    outputs = [(p, q + 1)]
    if q > 0:
        outputs.append((p + 1, q - 1))
    if p > 0:
        outputs.append((p - 1, q))
    return tuple(outputs)


def _filter_truncated(weights: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    labels = []
    for weight in weights:
        if weight in _IRREP_LABELS:
            labels.append(_IRREP_LABELS[weight])
    return tuple(labels)


@cache
def _fundamental_generators() -> tuple[jax.Array, ...]:
    z = 0.0 + 0.0j
    one = 1.0 + 0.0j
    imag = 0.0 + 1.0j
    matrices = (
        ((z, one, z), (one, z, z), (z, z, z)),
        ((z, -imag, z), (imag, z, z), (z, z, z)),
        ((one, z, z), (z, -one, z), (z, z, z)),
        ((z, z, one), (z, z, z), (one, z, z)),
        ((z, z, -imag), (z, z, z), (imag, z, z)),
        ((z, z, z), (z, z, one), (z, one, z)),
        ((z, z, z), (z, z, -imag), (z, imag, z)),
        (
            (one / math.sqrt(3), z, z),
            (z, one / math.sqrt(3), z),
            (z, z, -2 / math.sqrt(3)),
        ),
    )
    return tuple(0.5 * jnp.asarray(matrix, dtype=jnp.complex128) for matrix in matrices)


@cache
def _generators(irrep: int) -> tuple[jax.Array, ...]:
    if irrep == 0:
        return tuple(jnp.zeros((1, 1), dtype=jnp.complex128) for _ in range(8))
    fundamental = _fundamental_generators()
    if irrep == 1:
        return fundamental
    if irrep == 2:
        return tuple(-generator.conj() for generator in fundamental)
    raise ValueError("SU(3) generator requested outside fundamental truncation.")


def _canonical_tensor(tensor: jax.Array) -> jax.Array:
    flat = tensor.reshape(-1)
    for value in flat:
        magnitude = float(jnp.abs(value))
        if magnitude > 1e-10:
            return tensor / (value / magnitude)
    return tensor


def _kron_all(matrices: tuple[jax.Array, ...]) -> jax.Array:
    out = jnp.ones((1, 1), dtype=jnp.complex128)
    for matrix in matrices:
        out = jnp.kron(out, matrix)
    return out


@cache
def _invariant_basis(labels: tuple[int, ...]) -> tuple[jax.Array, ...]:
    dims = tuple(_IRREP_DIMS[label] for label in labels)
    total_dim = math.prod(dims)
    generators_by_leg = tuple(_generators(label) for label in labels)
    constraints = []
    for generator_idx in range(8):
        total = jnp.zeros((total_dim, total_dim), dtype=jnp.complex128)
        for leg_idx, leg_generators in enumerate(generators_by_leg):
            factors = tuple(
                leg_generators[generator_idx]
                if idx == leg_idx
                else jnp.eye(dims[idx], dtype=jnp.complex128)
                for idx in range(len(labels))
            )
            total = total + _kron_all(factors)
        constraints.append(total)
    constraint = jnp.concatenate(constraints, axis=0)
    _u, singular_values, vh = jnp.linalg.svd(constraint, full_matrices=True)
    rank = int(jnp.sum(singular_values > 1e-10))
    return tuple(_canonical_tensor(vector.reshape(dims)) for vector in vh[rank:])


@cache
def vertex_intertwiner_tensor(block: VertexBlock) -> jax.Array:
    labels = (_dual_label(block.j_l), _dual_label(block.j_u), block.j_r, block.j_d)
    return _invariant_basis(labels)[block.iota]


@cache
def _coupling_tensor(
    output_irrep: int,
    operator_irrep: int,
    input_irrep: int,
) -> jax.Array:
    labels = (_dual_label(output_irrep), operator_irrep, input_irrep)
    basis = _invariant_basis(labels)
    if len(basis) != 1:
        raise ValueError(
            f"Expected a unique SU(3) coupling for {operator_irrep} x {input_irrep} -> {output_irrep}."
        )
    return basis[0]


def _dual_label(irrep: int) -> int:
    if irrep == 1:
        return 2
    if irrep == 2:
        return 1
    return 0


@fundamental_irrep.dispatch
def fundamental_irrep(group: SU3) -> int:
    return group.fundamental


@coupling_tensor.dispatch
def coupling_tensor(group: SU3, out: int, op: int, inp: int) -> jax.Array:
    # The invariant-basis coupling is unit-Frobenius; its columns are equal
    # by Schur, so sqrt(dim) rescales it to the isometric Wigner coupling.
    return _coupling_tensor(out, op, inp) * math.sqrt(group.dim(out))


@vertex_tensor.dispatch
def vertex_tensor(group: SU3, block: VertexBlock) -> jax.Array:
    return vertex_intertwiner_tensor(block)


_IRREP_DIMS = tuple((p + 1) * (q + 1) * (p + q + 2) // 2 for p, q in _IRREP_WEIGHTS)


def build_pure_gauge_vertex_blocks(
    group: SU3,
    *,
    active_legs: tuple[bool, bool, bool, bool],
    target_charge: int = 0,
) -> tuple[VertexBlock, ...]:
    if target_charge != 0:
        raise NotImplementedError(
            "SU(3) backend currently supports singlet target_charge=0."
        )
    if len(active_legs) != 4:
        raise ValueError("active_legs must be ordered as (left, up, right, down).")
    choices = tuple(group.irreps() if active else (0,) for active in active_legs)
    blocks: list[VertexBlock] = []
    for j_l, j_u, j_r, j_d in product(*choices):
        labels = (group.dual(j_l), group.dual(j_u), j_r, j_d)
        for iota, _tensor in enumerate(_invariant_basis(labels)):
            blocks.append(
                VertexBlock(
                    j_l=j_l,
                    j_u=j_u,
                    j_r=j_r,
                    j_d=j_d,
                    iota=iota,
                    internal_irreps=(),
                )
            )
    return tuple(sorted(blocks))


@build_pure_gauge_tables.dispatch
def build_pure_gauge_tables(
    group: SU3,
    *,
    shape: tuple[int, int],
    target_charge: int = 0,
    matter_irreps: tuple[int, ...] = (0,),
    matter_numbers: tuple[int, ...] = (0,),
) -> PureGaugeTables:
    if matter_irreps != (0,) or matter_numbers != (0,):
        raise NotImplementedError("SU(3) matter tables are not implemented yet.")
    n_rows, n_cols = shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("shape must have positive dimensions.")
    return PureGaugeTables.from_blocks(
        group=group,
        shape=shape,
        matter_irreps=(0,),
        matter_numbers=(0,),
        blocks=tuple(
            tuple(
                build_pure_gauge_vertex_blocks(
                    group,
                    active_legs=(c > 0, r > 0, c < n_cols - 1, r < n_rows - 1),
                    target_charge=target_charge,
                )
                for c in range(n_cols)
            )
            for r in range(n_rows)
        ),
    )

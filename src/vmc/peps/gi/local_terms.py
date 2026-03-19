"""Local terms for gauge-invariant PEPS."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import (
    DiagonalOperator,
    LocalHamiltonian as GILocalHamiltonian,
    TransitionOperator,
    support_span,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinkDiagonalTerm(DiagonalOperator):
    """Diagonal term on link degrees of freedom."""

    orientation: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.orientation not in ("h", "v"):
            raise ValueError("orientation must be 'h' or 'v'")

    def energy(self, h_links: jax.Array, v_links: jax.Array) -> jax.Array:
        links = h_links if self.orientation == "h" else v_links
        total = jnp.zeros((), dtype=self.diag.dtype)
        for row, col in self.sites:
            total = total + self.diag[links[row, col]]
        return total

    def tree_flatten(self):
        return (self.diag,), (self.sites, self.orientation)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (diag,) = children
        sites, orientation = aux_data
        return cls(sites=sites, diag=diag, orientation=orientation)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MatterMassTerm(DiagonalOperator):
    """Matter mass term m_x n_x (diagonal on the matter site)."""

    def __init__(
        self,
        *,
        row: int,
        col: int,
        charge_of_site: tuple[int, ...],
    ) -> None:
        diag = jnp.asarray(charge_of_site, dtype=jnp.int32)
        super().__init__(sites=((row, col),), diag=diag)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HorizontalMatterHoppingTerm(TransitionOperator):
    """Gauge-covariant hard-core hopping on a horizontal link."""

    row: int
    col: int

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row, self.col + 1))

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HorizontalHiggsLinkTerm(TransitionOperator):
    """Z2 Higgs link term sigma_x X sigma_x on a horizontal link."""

    row: int
    col: int

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row, self.col + 1))

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VerticalMatterHoppingTerm(TransitionOperator):
    """Gauge-covariant hard-core hopping on a vertical link."""

    row: int
    col: int

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row + 1, self.col))

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VerticalHiggsLinkTerm(TransitionOperator):
    """Z2 Higgs link term sigma_x X sigma_x on a vertical link."""

    row: int
    col: int

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return ((self.row, self.col), (self.row + 1, self.col))

    def tree_flatten(self):
        return (), (self.row, self.col)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        row, col = aux_data
        return cls(row=row, col=col)


@support_span.dispatch
def support_span(_: HorizontalMatterHoppingTerm) -> tuple[int, int]:
    return 1, 2


@support_span.dispatch
def support_span(_: HorizontalHiggsLinkTerm) -> tuple[int, int]:
    return 1, 2


@support_span.dispatch
def support_span(_: VerticalMatterHoppingTerm) -> tuple[int, int]:
    return 2, 1


@support_span.dispatch
def support_span(_: VerticalHiggsLinkTerm) -> tuple[int, int]:
    return 2, 1


def build_electric_terms(
    shape: tuple[int, int],
    N: int,
) -> tuple[LinkDiagonalTerm, ...]:
    n_rows, n_cols = shape
    diag = 2.0 - 2.0 * jnp.cos(2.0 * jnp.pi * jnp.arange(N) / N)
    terms: list[LinkDiagonalTerm] = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            terms.append(LinkDiagonalTerm(sites=((r, c),), diag=diag, orientation="h"))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            terms.append(LinkDiagonalTerm(sites=((r, c),), diag=diag, orientation="v"))
    return tuple(terms)

"""Experimental local terms for GI-PEPS (ZN pure gauge)."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import DiagonalTerm


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinkDiagonalTerm(DiagonalTerm):
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
class PlaquetteTerm:
    """Plaquette term coefficient for U_p + U_p^dagger."""

    coeff: float

    def tree_flatten(self):
        return (jnp.asarray(self.coeff),), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (coeff,) = children
        return cls(coeff=coeff)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GILocalHamiltonian:
    """Local Hamiltonian container for GI-PEPS."""

    shape: tuple[int, int]
    electric_terms: tuple[LinkDiagonalTerm, ...]
    plaquette: PlaquetteTerm

    def tree_flatten(self):
        return (self.electric_terms, self.plaquette), (self.shape,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (electric_terms, plaquette) = children
        (shape,) = aux_data
        return cls(shape=shape, electric_terms=electric_terms, plaquette=plaquette)


def build_electric_terms(
    shape: tuple[int, int],
    coeff: float,
    N: int,
) -> tuple[LinkDiagonalTerm, ...]:
    n_rows, n_cols = shape
    diag = coeff * (2.0 - 2.0 * jnp.cos(2.0 * jnp.pi * jnp.arange(N) / N))
    terms: list[LinkDiagonalTerm] = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            terms.append(LinkDiagonalTerm(sites=((r, c),), diag=diag, orientation="h"))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            terms.append(LinkDiagonalTerm(sites=((r, c),), diag=diag, orientation="v"))
    return tuple(terms)

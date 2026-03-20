"""Shared physics helpers for Z2 hard-core-boson example scripts."""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from vmc.operators import PlaquetteOperator
from vmc.peps import GIPEPS, GIPEPSConfig, Variational
from vmc.peps.gi.local_terms import (
    GILocalHamiltonian,
    HorizontalMatterHoppingTerm,
    LinkDiagonalTerm,
    MatterMassTerm,
    VerticalMatterHoppingTerm,
    build_electric_terms,
)


CHARGE_OF_SITE = (0, 1)
QX = 0

DEFAULT_H = 1.0
DEFAULT_G = 0.33
DEFAULT_J = 0.5
DEFAULT_M = 0.0

DEFAULT_BOUNDARY_SWEEPS = 2
DEFAULT_DT = 0.01
DEFAULT_DIAG_SHIFT = 1e-4


def half_filling(shape: tuple[int, int]) -> int:
    """Return the half-filling hard-core boson number for a lattice."""
    n_sites = shape[0] * shape[1]
    if n_sites % 2:
        raise ValueError(f"Half filling requires an even number of sites, got {shape}.")
    return n_sites // 2


def format_token(value: int | float) -> str:
    """Format numeric values for filesystem-safe run-directory names."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), ".17g").replace("-", "m").replace(".", "p")


def coupling_suffix(*, h: float, g: float, J: float, m: float) -> str:
    """Format the Hamiltonian couplings for a run-directory suffix."""
    return (
        f"h{format_token(h)}_g{format_token(g)}_"
        f"J{format_token(J)}_m{format_token(m)}"
    )


def build_z2_hardcore_boson_hamiltonian(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
    J: float,
    m: float,
) -> GILocalHamiltonian:
    """Build the Z2 gauge theory with hard-core bosons Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=2)
    mass_terms = tuple(
        MatterMassTerm(
            row=row,
            col=col,
            charge_of_site=CHARGE_OF_SITE,
        )
        for row in range(n_rows)
        for col in range(n_cols)
    )
    horizontal_hops = tuple(
        HorizontalMatterHoppingTerm(row=row, col=col)
        for row in range(n_rows)
        for col in range(n_cols - 1)
    )
    vertical_hops = tuple(
        VerticalMatterHoppingTerm(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols)
    )
    terms = electric_terms + plaquette_terms + mass_terms + horizontal_hops + vertical_hops
    coeffs = (
        (jnp.asarray(g),) * len(electric_terms)
        + (jnp.asarray(-h),) * len(plaquette_terms)
        + (jnp.asarray(m),) * len(mass_terms)
        + (jnp.asarray(J),) * len(horizontal_hops)
        + (jnp.asarray(J),) * len(vertical_hops)
    )
    return GILocalHamiltonian(
        shape=shape,
        terms=terms,
        coeffs=coeffs,
    )


def build_model(
    shape: tuple[int, int],
    *,
    particle_number: int,
    bond_dim_per_charge: int,
    boundary_dim: int,
    boundary_sweeps: int,
    seed: int,
) -> GIPEPS:
    """Build a GI-PEPS model for Z2 hard-core bosons."""
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=shape,
            N=2,
            phys_dim=2,
            Qx=QX,
            degeneracy_per_charge=(bond_dim_per_charge, bond_dim_per_charge),
            charge_of_site=CHARGE_OF_SITE,
            particle_number=particle_number,
        ),
        contraction_strategy=Variational(boundary_dim, n_sweeps=boundary_sweeps),
    )


def _site_in_box(row: int, col: int, row0: int, row1: int, col0: int, col1: int) -> bool:
    return row0 <= row < row1 and col0 <= col < col1


def build_central_bulk_observable(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
    J: float,
    m: float,
    bulk_size: int,
) -> GILocalHamiltonian:
    """Build the central-bulk energy observable used in the finite-size analysis."""
    n_rows, n_cols = shape
    if bulk_size > min(shape):
        raise ValueError(f"bulk_size={bulk_size} exceeds lattice shape {shape}.")
    row0 = (n_rows - bulk_size) // 2
    col0 = (n_cols - bulk_size) // 2
    row1 = row0 + bulk_size
    col1 = col0 + bulk_size

    terms = []
    coeffs = []
    operator = build_z2_hardcore_boson_hamiltonian(shape, h=h, g=g, J=J, m=m)
    for term, coeff in zip(operator.terms, operator.coeffs):
        if isinstance(term, MatterMassTerm):
            row, col = term.sites[0]
            if _site_in_box(row, col, row0, row1, col0, col1):
                terms.append(term)
                coeffs.append(coeff)
        elif isinstance(term, HorizontalMatterHoppingTerm):
            if (
                _site_in_box(term.row, term.col, row0, row1, col0, col1)
                and _site_in_box(term.row, term.col + 1, row0, row1, col0, col1)
            ):
                terms.append(term)
                coeffs.append(coeff)
        elif isinstance(term, VerticalMatterHoppingTerm):
            if (
                _site_in_box(term.row, term.col, row0, row1, col0, col1)
                and _site_in_box(term.row + 1, term.col, row0, row1, col0, col1)
            ):
                terms.append(term)
                coeffs.append(coeff)
        elif isinstance(term, PlaquetteOperator):
            if (
                _site_in_box(term.row, term.col, row0, row1, col0, col1)
                and _site_in_box(term.row + 1, term.col + 1, row0, row1, col0, col1)
            ):
                terms.append(term)
                coeffs.append(coeff)
        elif isinstance(term, LinkDiagonalTerm):
            if term.orientation == "h":
                keep = (
                    _site_in_box(term.sites[0][0], term.sites[0][1], row0, row1, col0, col1)
                    and _site_in_box(
                        term.sites[0][0], term.sites[0][1] + 1, row0, row1, col0, col1
                    )
                )
            else:
                keep = (
                    _site_in_box(term.sites[0][0], term.sites[0][1], row0, row1, col0, col1)
                    and _site_in_box(
                        term.sites[0][0] + 1, term.sites[0][1], row0, row1, col0, col1
                    )
                )
            if keep:
                terms.append(term)
                coeffs.append(coeff)
        else:
            raise TypeError(f"Unsupported term type: {type(term)!r}")
    return GILocalHamiltonian(shape=shape, terms=tuple(terms), coeffs=tuple(coeffs))

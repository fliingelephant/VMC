"""Shared Z2 Higgs physics for vison confinement examples.

Provides the Higgs Hamiltonian, parity-sector model construction,
all-plaquette observables, and interior vison pair insertion.

The paper Hamiltonian (Wu & Nys 2026):
    H = -sum_i sigma_i^z - sum_p B_p - J sum_l sigma^-_l X_l sigma^+_l - g sum_l Z_l
is implemented up to additive constants in the binary occupancy basis:
    +2*n  +0.5*g*(2-2Z)  -J*sigma_x X sigma_x  -h*B_p
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.operators import PlaquetteOperator
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational
from vmc.peps.gi.local_terms import (
    HorizontalHiggsLinkTerm,
    MatterMassTerm,
    VerticalHiggsLinkTerm,
    build_electric_terms,
)


CHARGE_OF_SITE = (0, 1)


def build_model(
    shape: tuple[int, int],
    *,
    bond_dim: int,
    boundary_dim: int,
    boundary_sweeps: int,
    seed: int,
) -> GIPEPS:
    """Build the parity-sector Z2 GIPEPS for the Higgs example."""
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=shape,
            N=2,
            phys_dim=2,
            Qx=0,
            degeneracy_per_charge=(bond_dim, bond_dim),
            charge_of_site=CHARGE_OF_SITE,
            conserve_particle_number=False,
            particle_number=None,
        ),
        contraction_strategy=Variational(boundary_dim, n_sweeps=boundary_sweeps),
    )


def build_z2_higgs_hamiltonian(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
    J: float,
    sigma_z_field: float,
) -> GILocalHamiltonian:
    """Build the Z2 gauge-theory Hamiltonian with Higgs-link terms."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=2)
    matter_terms = tuple(
        MatterMassTerm(row=row, col=col, charge_of_site=CHARGE_OF_SITE)
        for row in range(n_rows)
        for col in range(n_cols)
    )
    horizontal_higgs = tuple(
        HorizontalHiggsLinkTerm(row=row, col=col)
        for row in range(n_rows)
        for col in range(n_cols - 1)
    )
    vertical_higgs = tuple(
        VerticalHiggsLinkTerm(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols)
    )
    terms = electric_terms + plaquette_terms + matter_terms + horizontal_higgs + vertical_higgs
    coeffs = (
        (jnp.asarray(0.5 * g),) * len(electric_terms)
        + (jnp.asarray(-h),) * len(plaquette_terms)
        + (jnp.asarray(2.0 * sigma_z_field),) * len(matter_terms)
        + (jnp.asarray(-J),) * len(horizontal_higgs)
        + (jnp.asarray(-J),) * len(vertical_higgs)
    )
    return GILocalHamiltonian(shape=shape, terms=terms, coeffs=coeffs)


def build_all_plaquette_observables(
    shape: tuple[int, int],
) -> tuple[GILocalHamiltonian, ...]:
    """Build one observable per plaquette for full 2D map snapshots.

    PlaquetteOperator evaluates P + P†; for Z2 P = P†, so coefficient
    0.5 yields the plaquette expectation value.
    """
    n_rows, n_cols = shape
    return tuple(
        GILocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(row=row, col=col),),
            coeffs=(jnp.asarray(0.5),),
        )
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )


def plaquette_observable_names(shape: tuple[int, int]) -> tuple[str, ...]:
    """Return names for all plaquette observables in row-major order."""
    n_rows, n_cols = shape
    return tuple(
        f"P_{r}_{c}"
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )


# ---------------------------------------------------------------------------
# Interior vison pair insertion
# ---------------------------------------------------------------------------

def _site_independent_directions(
    shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[str, ...]:
    """Return the locally independent link directions on one GIPEPS site."""
    n_rows, n_cols = shape
    active = {
        "left": col > 0,
        "right": col < n_cols - 1,
        "up": row > 0,
        "down": row < n_rows - 1,
    }
    dependent = next(
        d for d in ("right", "down", "up", "left") if active[d]
    )
    return tuple(
        d for d in ("left", "up", "down", "right")
        if active[d] and d != dependent
    )


def _z2_phase_for_direction(
    shape: tuple[int, int],
    row: int,
    col: int,
    direction: str,
) -> jax.Array:
    """Return the sigma_z phase on the site's Nc slices for one link direction."""
    directions = _site_independent_directions(shape, row, col)
    if direction not in directions:
        raise ValueError(
            f"Direction {direction!r} is not independent at site {(row, col)}."
        )
    n_configs = 1 << len(directions)
    cfg_indices = jnp.arange(n_configs, dtype=jnp.int32)
    digit_index = directions.index(direction)
    divisor = 1 << (len(directions) - digit_index - 1)
    values = (cfg_indices // divisor) % 2
    return (1 - 2 * values).astype(jnp.complex128)


def default_vison_link(
    shape: tuple[int, int],
    orientation: str,
) -> tuple[int, int]:
    """Return a central interior link for creating the default vison pair."""
    n_rows, n_cols = shape
    if min(shape) < 4:
        raise ValueError("Interior vison-pair insertion requires L >= 4.")
    if orientation == "v":
        return (n_rows - 2) // 2, n_cols // 2
    if orientation == "h":
        return n_rows // 2, (n_cols - 2) // 2
    raise ValueError(f"Unsupported orientation {orientation!r}.")


def create_interior_vison_pair(
    model: GIPEPS,
    *,
    orientation: str,
    row: int,
    col: int,
) -> GIPEPS:
    """Act with sigma_z on one interior link to create a vison pair."""
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    tensors = nnx.to_pure_dict(params)["tensors"]
    tensors = {
        sr: {sc: jnp.asarray(t) for sc, t in rd.items()}
        for sr, rd in tensors.items()
    }
    if orientation == "v":
        phase = _z2_phase_for_direction(model.shape, row, col, "down")
        tensors[row][col] = tensors[row][col] * phase[None, :, None, None, None, None]
    elif orientation == "h":
        phase = _z2_phase_for_direction(model.shape, row, col + 1, "left")
        tensors[row][col + 1] = (
            tensors[row][col + 1] * phase[None, :, None, None, None, None]
        )
    else:
        raise ValueError(f"Unsupported orientation {orientation!r}.")
    return nnx.merge(graphdef, {"tensors": tensors}, model_state)

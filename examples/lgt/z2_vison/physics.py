"""Shared Z2 pure gauge physics for vison propagation examples.

Provides Hamiltonian construction, model construction, plaquette observable
selection (with open-data coordinate transform), and boundary vison insertion.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.operators import PlaquetteOperator
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational
from vmc.peps.gi.local_terms import build_electric_terms


def build_z2_hamiltonian(
    shape: tuple[int, int],
    *,
    h: float,
    g: float,
) -> GILocalHamiltonian:
    """Build the pure Z2 gauge Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=row, col=col)
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=2)
    return GILocalHamiltonian(
        shape=shape,
        terms=electric_terms + plaquette_terms,
        coeffs=(jnp.asarray(g),) * len(electric_terms)
        + (jnp.asarray(-h),) * len(plaquette_terms),
    )


def build_model(
    shape: tuple[int, int],
    *,
    bond_dim: int,
    seed: int,
) -> GIPEPS:
    """Build a pure-gauge GI-PEPS model."""
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=shape,
            N=2,
            phys_dim=1,
            Qx=0,
            degeneracy_per_charge=(bond_dim, bond_dim),
            charge_of_site=(0,),
        ),
        contraction_strategy=Variational(truncate_bond_dimension=3 * bond_dim),
    )


# ---------------------------------------------------------------------------
# Plaquette observables with open-data coordinate transform
# ---------------------------------------------------------------------------

def selected_open_plaquettes(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return selected plaquettes in the open-data (bottom-up) convention.

    For L=6: (0,0), (0,1), (2,2) — matches Wu & Liu (2025) Fig 5a.
    For L=10: (0,0), (0,1), (4,4) — matches Fig 5b.
    """
    center = (shape[0] - 2) // 2
    return ((0, 0), (0, 1), (center, center))


def open_to_internal_plaquette(
    shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[int, int]:
    """Convert open-data plaquette coords (bottom-up) to internal (top-down).

    The Wu/open-data convention counts plaquette rows from the bottom;
    the internal PlaquetteOperator counts from the top.
    """
    return shape[0] - 2 - row, col


def build_selected_plaquette_observables(
    shape: tuple[int, int],
    plaquettes: tuple[tuple[int, int], ...] | None = None,
) -> tuple[GILocalHamiltonian, ...]:
    """Build observables for selected plaquettes.

    Plaquette coordinates use the open-data (bottom-up) convention.
    PlaquetteOperator evaluates P + P†; for Z2 P = P†, so coefficient
    0.5 yields the plaquette expectation value.
    """
    if plaquettes is None:
        plaquettes = selected_open_plaquettes(shape)
    return tuple(
        GILocalHamiltonian(
            shape=shape,
            terms=(PlaquetteOperator(*open_to_internal_plaquette(shape, row, col)),),
            coeffs=(jnp.asarray(0.5),),
        )
        for row, col in plaquettes
    )


# ---------------------------------------------------------------------------
# Boundary vison insertion
# ---------------------------------------------------------------------------

def _site_independent_directions(
    shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[str, ...]:
    """Return the independent local link directions on one GI-PEPS site."""
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


def create_bottom_left_vison(model: GIPEPS) -> GIPEPS:
    """Act with sigma_z on the bottom-left vertical boundary link."""
    n_rows = model.shape[0]
    if n_rows < 2 or model.shape[1] < 2:
        raise ValueError("The bottom-left vison construction requires L >= 2.")
    site_row = n_rows - 2
    site_col = 0
    phase = _z2_phase_for_direction(model.shape, site_row, site_col, "down")
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    tensors = nnx.to_pure_dict(params)["tensors"]
    tensors = {
        r: {c: jnp.asarray(t) for c, t in rd.items()}
        for r, rd in tensors.items()
    }
    tensors[site_row][site_col] = (
        tensors[site_row][site_col] * phase[None, :, None, None, None, None]
    )
    return nnx.merge(graphdef, {"tensors": tensors}, model_state)

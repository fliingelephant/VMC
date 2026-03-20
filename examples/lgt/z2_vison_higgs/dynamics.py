"""Z2 Higgs vison confinement dynamics with GI-PEPS.

Loads a ground state, creates an interior vison pair, and runs real-time
TDVP tracking all plaquette observables for 2D map snapshots.

Reproduces Wu & Nys (2026) Fig. 4:
  Deconfined phase: J=0.1, g=0.1
  Higgs phase:      J=0.5, g=0.1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402
import json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import PlaquetteOperator  # noqa: E402
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational  # noqa: E402
from vmc.peps.gi.local_terms import (  # noqa: E402
    HorizontalHiggsLinkTerm,
    MatterMassTerm,
    VerticalHiggsLinkTerm,
    build_electric_terms,
)
from vmc.preconditioners import SRPreconditioner  # noqa: E402
from vmc.qgt import ParameterSpace, SampleSpace  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402


CHARGE_OF_SITE = (0, 1)


# ---------------------------------------------------------------------------
# Physics (shared with ground_state.py)
# ---------------------------------------------------------------------------

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

    PlaquetteOperator evaluates P + P†. For Z2, P = P†, so coefficient
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


# ---------------------------------------------------------------------------
# Model state I/O
# ---------------------------------------------------------------------------

def load_model_state(input_path: Path) -> tuple[GIPEPS, dict]:
    """Load a saved GI-PEPS state."""
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        shape = tuple(int(x) for x in metadata["shape"])
        model = build_model(
            shape,
            bond_dim=int(metadata["bond_dim"]),
            boundary_dim=int(metadata["boundary_dim"]),
            boundary_sweeps=int(metadata["boundary_sweeps"]),
            seed=int(metadata["seed"]),
        )
        graphdef, _, model_state = nnx.split(model, nnx.Param, ...)
        tensors = {
            row: {
                col: jnp.asarray(data[f"tensor_{row}_{col}"])
                for col in range(shape[1])
            }
            for row in range(shape[0])
        }
    return nnx.merge(graphdef, {"tensors": tensors}, model_state), metadata


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Z2 Higgs vison confinement dynamics.",
    )
    parser.add_argument("--state", type=Path, required=True,
                        help="Path to ground-state .npz from ground_state.py")
    parser.add_argument("--vison-orientation", choices=("v", "h"), default="v")
    parser.add_argument("--vison-row", type=int, default=None)
    parser.add_argument("--vison-col", type=int, default=None)
    add_common_args(parser)
    parser.set_defaults(
        bond_dim=2, dt=0.005, diag_shift=1e-8,
        n_samples=4096, n_chains=512,
        solver_space="minsr", save_every=20, log_every=10,
    )
    args = parser.parse_args()

    model, metadata = load_model_state(args.state)
    shape = model.shape
    L = shape[0]
    h = float(metadata["h"])
    g = float(metadata["g"])
    J = float(metadata["J"])
    sigma_z_field = float(metadata["sigma_z_field"])

    # Determine vison link
    orientation = args.vison_orientation
    if args.vison_row is not None and args.vison_col is not None:
        vison_row, vison_col = args.vison_row, args.vison_col
    else:
        vison_row, vison_col = default_vison_link(shape, orientation)

    model = create_interior_vison_pair(
        model, orientation=orientation, row=vison_row, col=vison_col,
    )

    hamiltonian = build_z2_higgs_hamiltonian(
        shape, h=h, g=g, J=J, sigma_z_field=sigma_z_field,
    )

    # All plaquettes as observables for 2D map snapshots
    observables = build_all_plaquette_observables(shape)
    n_rows, n_cols = shape
    plaq_names = tuple(
        f"P_{r}_{c}"
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )

    seed = args.seed + 1
    space = SampleSpace() if args.solver_space == "minsr" else ParameterSpace()
    driver = TDVPDriver(
        model, hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=space,
            diag_shift=args.diag_shift,
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(seed),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        full_gradient=args.full_gradient,
    )

    g_tok = format(g, ".3f").replace(".", "p")
    J_tok = format(J, ".3f").replace(".", "p")
    run_dir = (
        f"data/z2_vison_higgs/L{L}_g{g_tok}_J{J_tok}_Dk{int(metadata['bond_dim'])}"
        f"_rt_vison_{orientation}_r{vison_row}_c{vison_col}"
    )
    run(
        driver,
        T_final=args.T_final,
        run_dir=run_dir,
        observable_names=plaq_names,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "model": "z2_vison_higgs_confinement",
            "L": L, "h": h, "g": g, "J": J,
            "sigma_z_field": sigma_z_field,
            "vison_orientation": orientation,
            "vison_row": vison_row,
            "vison_col": vison_col,
        },
    )


if __name__ == "__main__":
    main()

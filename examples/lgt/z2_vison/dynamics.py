"""Z2 vison propagation real-time dynamics with GI-PEPS.

Loads a saved ground state, inserts a boundary vison (sigma_z on the
bottom-left vertical link), and runs real-time TDVP tracking selected
plaquette expectation values.

Reproduces the real-time dynamics for Wu & Liu (2025) Fig. 5:
  --L 6  --bond-dim 3  → Fig 5a (6×6, exact comparison available)
  --L 10 --bond-dim 4  → Fig 5b (10×10)

The selected plaquettes follow the open-data convention: row-major
indices on the (L-1)×(L-1) plaquette grid, counted from the bottom.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import PlaquetteOperator  # noqa: E402
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational  # noqa: E402
from vmc.peps.gi.local_terms import build_electric_terms  # noqa: E402
from vmc.preconditioners import SRPreconditioner  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402


# ---------------------------------------------------------------------------
# Physics (shared with ground_state.py)
# ---------------------------------------------------------------------------

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
# Model state I/O
# ---------------------------------------------------------------------------

def load_model_state(input_path: Path) -> tuple[GIPEPS, dict]:
    """Load a saved optimized GI-PEPS state."""
    import json
    import numpy as np

    with np.load(input_path) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        shape = tuple(int(x) for x in metadata["shape"])
        bond_dim = int(metadata["bond_dim"])
        model = build_model(shape, bond_dim=bond_dim, seed=int(metadata["seed"]))
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
# Vison insertion
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
        direction
        for direction in ("right", "down", "up", "left")
        if active[direction]
    )
    return tuple(
        direction
        for direction in ("left", "up", "down", "right")
        if active[direction] and direction != dependent
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
    n_rows, n_cols = model.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError("The bottom-left vison construction requires L >= 2.")
    site_row = n_rows - 2
    site_col = 0
    phase = _z2_phase_for_direction(model.shape, site_row, site_col, "down")
    graphdef, params, model_state = nnx.split(model, nnx.Param, ...)
    tensors = nnx.to_pure_dict(params)["tensors"]
    tensors = {
        row: {col: jnp.asarray(tensor) for col, tensor in row_dict.items()}
        for row, row_dict in tensors.items()
    }
    tensors[site_row][site_col] = (
        tensors[site_row][site_col] * phase[None, :, None, None, None, None]
    )
    return nnx.merge(graphdef, {"tensors": tensors}, model_state)


# ---------------------------------------------------------------------------
# Plaquette observables
# ---------------------------------------------------------------------------

def selected_open_plaquettes(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return selected plaquettes in the open-data (bottom-up) convention."""
    center = (shape[0] - 2) // 2
    return ((0, 0), (0, 1), (center, center))


def open_to_internal_plaquette(
    shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[int, int]:
    """Convert open-data plaquette coords (bottom-up) to internal (top-down)."""
    return shape[0] - 2 - row, col


def build_selected_plaquette_observables(
    shape: tuple[int, int],
    plaquettes: tuple[tuple[int, int], ...] | None = None,
) -> tuple[GILocalHamiltonian, ...]:
    """Build observables for selected plaquettes.

    ``PlaquetteOperator`` evaluates ``P + P†``. For Z2, ``P = P†``, so a
    coefficient of ``0.5`` yields the plaquette expectation value.
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Z2 vison propagation real-time dynamics.",
    )
    parser.add_argument("--state", type=Path, required=True,
                        help="Path to ground-state .npz from ground_state.py")
    add_common_args(parser)
    # Override defaults for real-time dynamics
    parser.set_defaults(bond_dim=3, dt=0.01, diag_shift=1e-8)
    args = parser.parse_args()

    model, metadata = load_model_state(args.state)
    model = create_bottom_left_vison(model)
    shape = model.shape
    h = float(metadata["h"])
    g = float(metadata["g"])
    bond_dim = int(metadata["bond_dim"])
    L = shape[0]

    hamiltonian = build_z2_hamiltonian(shape, h=h, g=g)
    plaquettes = selected_open_plaquettes(shape)
    observables = build_selected_plaquette_observables(shape, plaquettes)
    plaq_names = tuple(f"P_{r}{c}" for r, c in plaquettes)

    seed = args.seed + 1  # different seed from ground state
    driver = TDVPDriver(
        model,
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
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

    g_token = format(g, ".3f").replace(".", "p")
    run_dir = f"data/z2_vison/L{L}_g{g_token}_Dk{bond_dim}_rt"
    run(
        driver,
        T_final=args.T_final,
        run_dir=run_dir,
        observable_names=plaq_names,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "gauge_group": "Z2", "L": L,
            "h": h, "g": g,
            "vison": "sigma_z on bottom-left vertical link",
            "selected_plaquettes": [list(p) for p in plaquettes],
        },
    )


if __name__ == "__main__":
    main()

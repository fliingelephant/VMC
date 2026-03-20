"""Z2 pure gauge ground-state optimization with GI-PEPS.

Optimizes the deconfined ground state via imaginary-time SR and saves the
model tensors for subsequent real-time dynamics.

Reproduces the ground-state preparation for Wu & Liu (2025) Fig. 5:
  --L 6  --bond-dim 3  → Fig 5a (6×6)
  --L 10 --bond-dim 4  → Fig 5b (10×10)
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

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import PlaquetteOperator  # noqa: E402
from vmc.peps import GIPEPS, GIPEPSConfig, GILocalHamiltonian, Variational  # noqa: E402
from vmc.peps.gi.local_terms import build_electric_terms  # noqa: E402
from vmc.preconditioners import SRPreconditioner  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402


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


def save_model_state(model: GIPEPS, metadata: dict, output_path: Path) -> None:
    """Save the optimized PEPS tensors and minimal rebuild metadata."""
    _, params, model_state = nnx.split(model, nnx.Param, ...)
    if nnx.to_pure_dict(model_state):
        raise ValueError("Expected an empty non-parameter GIPEPS state.")
    tensors = nnx.to_pure_dict(params)["tensors"]
    arrays = {"metadata_json": np.asarray(json.dumps(metadata))}
    for row, row_dict in tensors.items():
        for col, tensor in row_dict.items():
            arrays[f"tensor_{row}_{col}"] = np.asarray(tensor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Saved {output_path}", flush=True)


def load_model_state(input_path: Path) -> tuple[GIPEPS, dict]:
    """Load a saved optimized GI-PEPS state."""
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


def _default_run_dir(*, L: int, g: float, bond_dim: int) -> str:
    g_token = format(g, ".3f").replace(".", "p")
    return f"data/z2_vison/L{L}_g{g_token}_Dk{bond_dim}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize Z2 pure gauge ground state with GI-PEPS.",
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.1)
    add_common_args(parser)
    # Override defaults for this problem
    parser.set_defaults(bond_dim=3, dt=0.005, diag_shift=1e-6, n_steps=400)
    args = parser.parse_args()

    shape = (args.L, args.L)
    model = build_model(shape, bond_dim=args.bond_dim, seed=args.seed)
    hamiltonian = build_z2_hamiltonian(shape, h=args.h, g=args.g)

    driver = TDVPDriver(
        model,
        hamiltonian,
        preconditioner=SRPreconditioner(
            diag_shift=args.diag_shift,
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(args.seed),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        full_gradient=args.full_gradient,
    )

    run_dir = _default_run_dir(L=args.L, g=args.g, bond_dim=args.bond_dim)
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=run_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "gauge_group": "Z2", "L": args.L,
            "h": args.h, "g": args.g,
        },
    )

    # Save model state for dynamics handoff
    state_path = Path(run_dir) / "ground_state.npz"
    save_model_state(
        driver.model,
        {
            "gauge_group": "Z2",
            "shape": list(shape),
            "L": args.L,
            "h": args.h,
            "g": args.g,
            "bond_dim": args.bond_dim,
            "seed": args.seed,
        },
        state_path,
    )


if __name__ == "__main__":
    main()

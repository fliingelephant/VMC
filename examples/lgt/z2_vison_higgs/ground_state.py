"""Z2 Higgs ground-state optimization with GI-PEPS.

Prepares the parity-sector ground state of the Z2 gauge theory with Higgs
field, targeting the vison confinement dynamics in Wu & Nys (2026) Fig. 4.

The paper Hamiltonian
    H = -sum_i sigma_i^z - sum_p B_p - J sum_l sigma^-_l X_l sigma^+_l - g sum_l Z_l
is implemented up to additive constants in the binary occupancy basis:
    +2*n  +0.5*g*(2-2Z)  -J*sigma_x X sigma_x  -h*B_p
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


def save_model_state(model: GIPEPS, metadata: dict, output_path: Path) -> None:
    """Save model tensors for dynamics handoff."""
    _, params, model_state = nnx.split(model, nnx.Param, ...)
    if nnx.to_pure_dict(model_state):
        raise ValueError("Expected an empty non-parameter GIPEPS state.")
    tensors = nnx.to_pure_dict(params)["tensors"]
    arrays = {"metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}
    for row, row_dict in tensors.items():
        for col, tensor in row_dict.items():
            arrays[f"tensor_{row}_{col}"] = np.asarray(tensor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Saved {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize Z2 Higgs ground state with parity-sector GI-PEPS.",
    )
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.1)
    parser.add_argument("--J", type=float, default=0.1)
    parser.add_argument("--sigma-z-field", type=float, default=1.0)
    parser.add_argument("--boundary-sweeps", type=int, default=2)
    add_common_args(parser)
    parser.set_defaults(
        bond_dim=2, dt=0.005, diag_shift=1e-4,
        n_steps=200, n_samples=4096, n_chains=512,
        solver_space="minsr", save_every=20, log_every=10,
    )
    args = parser.parse_args()

    shape = (args.L, args.L)
    boundary_dim = 3 * args.bond_dim
    model = build_model(
        shape,
        bond_dim=args.bond_dim,
        boundary_dim=boundary_dim,
        boundary_sweeps=args.boundary_sweeps,
        seed=args.seed,
    )
    hamiltonian = build_z2_higgs_hamiltonian(
        shape, h=args.h, g=args.g, J=args.J, sigma_z_field=args.sigma_z_field,
    )
    space = SampleSpace() if args.solver_space == "minsr" else ParameterSpace()
    driver = TDVPDriver(
        model, hamiltonian,
        preconditioner=SRPreconditioner(
            space=space,
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

    g_tok = format(args.g, ".3f").replace(".", "p")
    J_tok = format(args.J, ".3f").replace(".", "p")
    run_dir = f"data/z2_vison_higgs/L{args.L}_g{g_tok}_J{J_tok}_Dk{args.bond_dim}"
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=run_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "model": "z2_vison_higgs_confinement",
            "L": args.L, "h": args.h, "g": args.g,
            "J": args.J, "sigma_z_field": args.sigma_z_field,
        },
    )

    state_path = Path(run_dir) / "ground_state.npz"
    save_model_state(
        driver.model,
        {
            "model": "z2_vison_higgs_confinement",
            "gauge_group": "Z2",
            "shape": list(shape),
            "L": args.L,
            "h": args.h,
            "g": args.g,
            "J": args.J,
            "sigma_z_field": args.sigma_z_field,
            "bond_dim": args.bond_dim,
            "boundary_dim": boundary_dim,
            "boundary_sweeps": args.boundary_sweeps,
            "seed": args.seed,
            "solver_space": args.solver_space,
        },
        state_path,
    )


if __name__ == "__main__":
    main()

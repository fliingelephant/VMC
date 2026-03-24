"""Fixed-step SR benchmark for pure Z2 lattice gauge theory.

Records total energy, mean plaquette value, mean horizontal-link Z, and mean
vertical-link Z expectation values.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import PlaquetteOperator  # noqa: E402
from vmc.peps import ZipUp  # noqa: E402
from vmc.peps.gi import GILocalHamiltonian, GIPEPS, GIPEPSConfig  # noqa: E402
from vmc.peps.gi.local_terms import LinkDiagonalTerm, build_electric_terms  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky  # noqa: E402
from vmc.qgt import ParameterSpace  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402


def build_z2_hamiltonian(
    shape: tuple[int, int],
    h: float,
    g: float,
) -> GILocalHamiltonian:
    """Build the pure Z2 gauge Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=r, col=c)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=2)
    return GILocalHamiltonian(
        shape=shape,
        terms=electric_terms + plaquette_terms,
        coeffs=(jnp.asarray(g),) * len(electric_terms)
        + (jnp.asarray(-h),) * len(plaquette_terms),
    )


def build_mean_plaquette_observable(shape: tuple[int, int]) -> GILocalHamiltonian:
    """Build the average plaquette operator."""
    n_rows, n_cols = shape
    n_plaquettes = (n_rows - 1) * (n_cols - 1)
    coeff = jnp.asarray(0.5 / n_plaquettes, dtype=jnp.complex128)
    terms = tuple(
        PlaquetteOperator(row=r, col=c)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    return GILocalHamiltonian(shape=shape, terms=terms, coeffs=(coeff,) * len(terms))


def build_mean_link_z_observable(
    shape: tuple[int, int],
    *,
    orientation: str,
) -> GILocalHamiltonian:
    """Build the average Z expectation on horizontal or vertical links."""
    n_rows, n_cols = shape
    if orientation == "h":
        count = n_rows * (n_cols - 1)
        row_range = range(n_rows)
        col_range = range(n_cols - 1)
    elif orientation == "v":
        count = (n_rows - 1) * n_cols
        row_range = range(n_rows - 1)
        col_range = range(n_cols)
    else:
        raise ValueError(f"Unsupported orientation: {orientation!r}")

    diag = jnp.asarray([1.0, -1.0], dtype=jnp.complex128) / count
    terms = tuple(
        LinkDiagonalTerm(sites=((r, c),), diag=diag, orientation=orientation)
        for r in row_range
        for c in col_range
    )
    return GILocalHamiltonian(shape=shape, terms=terms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-step SR benchmark for pure Z2 lattice gauge theory.",
    )
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.2)
    add_common_args(parser)
    parser.set_defaults(
        bond_dim=2, n_samples=1024, n_chains=64,
        n_steps=100, dt=0.01, diag_shift=1e-8, seed=42,
        log_every=1,
    )
    args = parser.parse_args()

    shape = (args.L, args.L)
    boundary_dim = 3 * args.bond_dim
    hamiltonian = build_z2_hamiltonian(shape, args.h, args.g)
    observables = (
        build_mean_plaquette_observable(shape),
        build_mean_link_z_observable(shape, orientation="h"),
        build_mean_link_z_observable(shape, orientation="v"),
    )

    model = GIPEPS(
        rngs=nnx.Rngs(args.seed),
        config=GIPEPSConfig(
            shape=shape, N=2, phys_dim=1, Qx=0,
            degeneracy_per_charge=(args.bond_dim, args.bond_dim),
            charge_of_site=(0,),
        ),
        contraction_strategy=ZipUp(truncate_bond_dimension=boundary_dim),
    )

    driver = TDVPDriver(
        model,
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=ParameterSpace(),
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=args.diag_shift,
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(args.seed),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
    )

    g_tok = format(args.g, ".3f").replace(".", "p")
    default_dir = (
        Path(__file__).resolve().parent
        / f"z2_pure_gauge_benchmark_{args.L}x{args.L}_g{g_tok}_ns{args.n_samples}_{args.n_steps}"
    )
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=args.output or str(default_dir),
        observable_names=("plaquette", "z_h", "z_v"),
        log_every=args.log_every,
        save_every=args.save_every,
        extra_config={
            "gauge_group": "Z2", "L": args.L,
            "h": args.h, "g": args.g,
        },
    )


if __name__ == "__main__":
    main()

"""Ground state of pure Z3 lattice gauge theory.

The Z3 gauge theory exhibits a first-order phase transition at g_c ~ 0.375.

Hamiltonian:
    H = -h sum_x (P_x + P_x^dag) + g sum_links (2 - 2cos(2 pi E/3))

Reference: Wu & Liu, Phys. Rev. Lett. 135, 130401 (2025), Fig 2(a,b)
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

from vmc.drivers import TDVPDriver, ImaginaryTimeUnit  # noqa: E402
from vmc.operators import PlaquetteOperator  # noqa: E402
from vmc.peps import ZipUp  # noqa: E402
from vmc.peps.gi.local_terms import GILocalHamiltonian, build_electric_terms  # noqa: E402
from vmc.peps.gi.model import GIPEPS, GIPEPSConfig  # noqa: E402
from vmc.preconditioners import SRPreconditioner  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402


def build_z3_hamiltonian(
    shape: tuple[int, int],
    h: float = 1.0,
    g: float = 0.375,
) -> GILocalHamiltonian:
    """Build pure Z3 LGT Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=r, col=c)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=3)
    return GILocalHamiltonian(
        shape=shape,
        terms=electric_terms + plaquette_terms,
        coeffs=(jnp.asarray(g),) * len(electric_terms)
        + (jnp.asarray(-h),) * len(plaquette_terms),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure Z3 lattice gauge theory ground state optimization.",
    )
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--g", type=float, default=0.375)
    parser.add_argument("--h", type=float, default=1.0)
    add_common_args(parser)
    parser.set_defaults(
        bond_dim=2, n_samples=2000, n_chains=1, n_steps=500,
        full_gradient=True, log_every=5, save_every=100,
    )
    args = parser.parse_args()

    shape = (args.L, args.L)
    model = GIPEPS(
        rngs=nnx.Rngs(args.seed),
        config=GIPEPSConfig(
            shape=shape, N=3, phys_dim=1, Qx=0,
            degeneracy_per_charge=(args.bond_dim, args.bond_dim, args.bond_dim),
            charge_of_site=(0,),
        ),
        contraction_strategy=ZipUp(truncate_bond_dimension=3 * args.bond_dim),
    )
    hamiltonian = build_z3_hamiltonian(shape, h=args.h, g=args.g)

    driver = TDVPDriver(
        model, hamiltonian,
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

    g_tok = format(args.g, ".3f").replace(".", "p")
    run_dir = args.output or f"data/z3_pure/L{args.L}_g{g_tok}"
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=run_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "gauge_group": "Z3", "L": args.L,
            "h": args.h, "g": args.g,
            "critical_point": "g_c ~ 0.375 (first-order)",
        },
    )


if __name__ == "__main__":
    main()

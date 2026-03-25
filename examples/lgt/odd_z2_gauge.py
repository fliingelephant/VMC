"""Ground state of odd-Z2 lattice gauge theory.

The odd-Z2 gauge theory has background charge Q_x = 1 at every site,
relevant for understanding spin liquids and dimer models.

By varying g, it experiences a continuous transition between:
- Deconfined phase (g < g_c): Uniform plaquette expectation values
- Confined phase (g > g_c): Translation symmetry breaking (VBS order)

Critical point: g_c ~ 0.64

Reference: Wu & Liu, Phys. Rev. Lett. 135, 130401 (2025)
"""
from __future__ import annotations

from pathlib import Path


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

from vmc.workflow import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402


def build_odd_z2_hamiltonian(
    shape: tuple[int, int],
    h: float = 1.0,
    g: float = 0.5,
) -> GILocalHamiltonian:
    """Build odd-Z2 LGT Hamiltonian (Q_x = 1 everywhere)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Odd-Z2 lattice gauge theory ground state optimization.",
    )
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--g", type=float, default=0.5)
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
            shape=shape, N=2, phys_dim=1, Qx=1,
            degeneracy_per_charge=(args.bond_dim, args.bond_dim),
            charge_of_site=(0,),
        ),
        contraction_strategy=ZipUp(truncate_bond_dimension=3 * args.bond_dim),
    )
    hamiltonian = build_odd_z2_hamiltonian(shape, h=args.h, g=args.g)

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
    run_dir = args.output or f"data/odd_z2/L{args.L}_g{g_tok}"
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=run_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "gauge_group": "odd_Z2", "L": args.L,
            "h": args.h, "g": args.g, "Qx": 1,
            "phase": "Deconfined" if args.g < 0.64 else "Confined (VBS)",
        },
    )


if __name__ == "__main__":
    main()

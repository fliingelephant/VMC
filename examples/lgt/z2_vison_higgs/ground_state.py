"""Z2 Higgs ground-state optimization with parity-sector GI-PEPS.

Targets Wu & Nys (2026) Fig. 4 ground-state preparation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.preconditioners import SRPreconditioner  # noqa: E402
from vmc.qgt import ParameterSpace, SampleSpace  # noqa: E402

from vmc.workflow import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402
from physics import build_model, build_z2_higgs_hamiltonian  # noqa: E402


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
    parser.add_argument("--n-steps", type=int, default=200)
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
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=args.output or f"data/z2_vison_higgs/L{args.L}_g{g_tok}_J{J_tok}_Dk{args.bond_dim}",
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "model": "z2_vison_higgs", "L": args.L,
            "h": args.h, "g": args.g, "J": args.J,
            "sigma_z_field": args.sigma_z_field,
            "bond_dim": args.bond_dim,
            "boundary_dim": boundary_dim,
            "boundary_sweeps": args.boundary_sweeps,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()

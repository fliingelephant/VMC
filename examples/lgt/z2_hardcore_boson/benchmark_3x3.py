"""3x3 quoted-ED benchmark for Z2 gauge fields coupled to hard-core bosons."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky  # noqa: E402
from vmc.qgt import ParameterSpace  # noqa: E402

from vmc.workflow import DEFAULT_METRICS_CONFIG, add_common_args, run  # noqa: E402
from common import build_model, build_z2_hardcore_boson_hamiltonian  # noqa: E402


L = 3
SHAPE = (L, L)
PARTICLE_NUMBER = 2
H_COUPLING = 1.0
G_COUPLING = 0.33
J_COUPLING = 0.5
M_COUPLING = 0.0
BOND_DIM_PER_CHARGE = 3
BOUNDARY_DIM = 9
QUOTED_ED_ENERGY_PER_SITE = -0.4707135061


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 3x3 quoted-ED benchmark for Z2 hard-core bosons.",
    )
    parser.add_argument("--boundary-sweeps", type=int, default=2)
    add_common_args(parser)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.set_defaults(
        n_samples=4096, n_chains=64, n_steps=200,
        dt=0.01, diag_shift=1e-4, seed=42,
        log_every=50, save_every=50,
    )
    args = parser.parse_args()

    model = build_model(
        SHAPE,
        particle_number=PARTICLE_NUMBER,
        bond_dim_per_charge=BOND_DIM_PER_CHARGE,
        boundary_dim=BOUNDARY_DIM,
        boundary_sweeps=args.boundary_sweeps,
        seed=args.seed,
    )
    hamiltonian = build_z2_hardcore_boson_hamiltonian(
        SHAPE, h=H_COUPLING, g=G_COUPLING, J=J_COUPLING, m=M_COUPLING,
    )
    driver = TDVPDriver(
        model, hamiltonian,
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
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=args.output or str(Path(__file__).resolve().parent / "data" / "benchmark_3x3"),
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "quoted_ed_energy_per_site": QUOTED_ED_ENERGY_PER_SITE,
            "particle_number": PARTICLE_NUMBER,
            "h": H_COUPLING, "g": G_COUPLING, "J": J_COUPLING, "m": M_COUPLING,
        },
    )


if __name__ == "__main__":
    main()

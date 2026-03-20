"""3x3 quoted-ED benchmark for Z2 gauge fields coupled to hard-core bosons."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky  # noqa: E402
from vmc.qgt import ParameterSpace  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, run  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_BOUNDARY_SWEEPS, DEFAULT_DIAG_SHIFT, DEFAULT_DT,
    build_model, build_z2_hardcore_boson_hamiltonian,
)


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

N_SAMPLES = 4096
N_CHAINS = 64
DEFAULT_N_STEPS = 200
SEED = 42


def _run_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "benchmark_3x3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 3x3 quoted-ED benchmark for Z2 hard-core bosons.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--diag-shift", type=float, default=DEFAULT_DIAG_SHIFT)
    parser.add_argument("--boundary-sweeps", type=int, default=DEFAULT_BOUNDARY_SWEEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
        run_dir=_run_dir(),
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

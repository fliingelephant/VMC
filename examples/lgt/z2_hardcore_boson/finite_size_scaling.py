"""One half-filled finite-size point with central-bulk energies for the paper scan."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner, solve_cholesky  # noqa: E402
from vmc.qgt import ParameterSpace  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, run  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_BOUNDARY_SWEEPS, DEFAULT_DIAG_SHIFT, DEFAULT_DT,
    DEFAULT_G, DEFAULT_H, DEFAULT_J, DEFAULT_M,
    build_central_bulk_observable, build_model,
    build_z2_hardcore_boson_hamiltonian,
    coupling_suffix, half_filling,
)


DEFAULT_L = 16
DEFAULT_DK = 6
DEFAULT_N_SAMPLES = 4096
DEFAULT_N_CHAINS = 64
DEFAULT_N_STEPS = 200
DEFAULT_SEED = 42


def _bulk_sizes(L: int) -> tuple[int, ...]:
    return tuple(s for s in (L, L - 2, L - 4) if s > 0)


def _run_dir(args: argparse.Namespace) -> Path:
    return (
        Path(__file__).resolve().parent
        / "data" / "finite_size_scaling"
        / (
            f"L{args.L}_{coupling_suffix(h=args.h, g=args.g, J=args.J, m=args.m)}"
            f"_Dk{args.bond_dim_per_charge}"
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one finite-size point with central-bulk energies for the Z2 hard-core-boson paper scan.",
    )
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--bond-dim-per-charge", type=int, default=DEFAULT_DK)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--g", type=float, default=DEFAULT_G)
    parser.add_argument("--J", type=float, default=DEFAULT_J)
    parser.add_argument("--m", type=float, default=DEFAULT_M)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--diag-shift", type=float, default=DEFAULT_DIAG_SHIFT)
    parser.add_argument("--boundary-sweeps", type=int, default=DEFAULT_BOUNDARY_SWEEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shape = (args.L, args.L)
    particle_number = half_filling(shape)
    bulk_sizes = _bulk_sizes(args.L)
    observables = tuple(
        build_central_bulk_observable(
            shape, h=args.h, g=args.g, J=args.J, m=args.m, bulk_size=bs,
        )
        for bs in bulk_sizes
    )
    observable_names = tuple(f"bulk{bs}" for bs in bulk_sizes)
    model = build_model(
        shape,
        particle_number=particle_number,
        bond_dim_per_charge=args.bond_dim_per_charge,
        boundary_dim=3 * args.bond_dim_per_charge,
        boundary_sweeps=args.boundary_sweeps,
        seed=args.seed,
    )
    hamiltonian = build_z2_hardcore_boson_hamiltonian(
        shape, h=args.h, g=args.g, J=args.J, m=args.m,
    )
    driver = TDVPDriver(
        model, hamiltonian,
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
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=_run_dir(args),
        observable_names=observable_names,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "L": args.L, "particle_number": particle_number,
            "bulk_sizes": list(bulk_sizes),
            "h": args.h, "g": args.g, "J": args.J, "m": args.m,
            "bond_dim_per_charge": args.bond_dim_per_charge,
        },
    )


if __name__ == "__main__":
    main()

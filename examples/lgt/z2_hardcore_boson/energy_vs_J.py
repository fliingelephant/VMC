"""One half-filled energy-vs-J point for the Z2 hard-core-boson paper scan."""
from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    from common import (
        DEFAULT_BOUNDARY_SWEEPS,
        DEFAULT_DIAG_SHIFT,
        DEFAULT_DT,
        DEFAULT_G,
        DEFAULT_H,
        DEFAULT_J,
        coupling_suffix,
        DEFAULT_LOG_EVERY,
        DEFAULT_M,
        DEFAULT_SAVE_EVERY,
        build_ground_state_driver,
        format_token,
        half_filling,
        maybe_resume,
        prepare_run_dir,
        run_ground_state_steps,
    )
else:
    from .common import (
        DEFAULT_BOUNDARY_SWEEPS,
        DEFAULT_DIAG_SHIFT,
        DEFAULT_DT,
        DEFAULT_G,
        DEFAULT_H,
        DEFAULT_J,
        coupling_suffix,
        DEFAULT_LOG_EVERY,
        DEFAULT_M,
        DEFAULT_SAVE_EVERY,
        build_ground_state_driver,
        format_token,
        half_filling,
        maybe_resume,
        prepare_run_dir,
        run_ground_state_steps,
    )


DEFAULT_L = 16
DEFAULT_G_POINT = DEFAULT_G
DEFAULT_J_POINT = DEFAULT_J
DEFAULT_DK = 6
DEFAULT_N_SAMPLES = 4096
DEFAULT_N_CHAINS = 64
DEFAULT_N_STEPS = 200
DEFAULT_SEED = 42


def _run_dir(args: argparse.Namespace) -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "energy_vs_J"
        / (
            f"L{args.L}_{coupling_suffix(h=args.h, g=args.g, J=args.J, m=args.m)}"
            f"_Dk{args.bond_dim_per_charge}"
        )
    )


def _problem(args: argparse.Namespace, particle_number: int) -> dict:
    return {
        "script": "energy_vs_J",
        "shape": [args.L, args.L],
        "particle_number": particle_number,
        "h": args.h,
        "g": args.g,
        "J": args.J,
        "m": args.m,
        "bond_dim_per_charge": args.bond_dim_per_charge,
        "boundary_dimension": 3 * args.bond_dim_per_charge,
        "boundary_sweeps": args.boundary_sweeps,
        "n_samples": args.n_samples,
        "n_chains": args.n_chains,
        "dt": args.dt,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one half-filled energy-vs-J point for the Z2 hard-core-boson paper scan.",
    )
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--g", type=float, default=DEFAULT_G_POINT)
    parser.add_argument("--J", type=float, default=DEFAULT_J_POINT)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--m", type=float, default=DEFAULT_M)
    parser.add_argument("--bond-dim-per-charge", type=int, default=DEFAULT_DK)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--diag-shift", type=float, default=DEFAULT_DIAG_SHIFT)
    parser.add_argument("--boundary-sweeps", type=int, default=DEFAULT_BOUNDARY_SWEEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shape = (args.L, args.L)
    particle_number = half_filling(shape)
    problem = _problem(args, particle_number)
    driver = build_ground_state_driver(
        shape=shape,
        h=args.h,
        g=args.g,
        J=args.J,
        m=args.m,
        particle_number=particle_number,
        bond_dim_per_charge=args.bond_dim_per_charge,
        boundary_dim=3 * args.bond_dim_per_charge,
        boundary_sweeps=args.boundary_sweeps,
        seed=args.seed,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        dt=args.dt,
        diag_shift=args.diag_shift,
    )
    run_dir = prepare_run_dir(_run_dir(args), resume=args.resume)
    maybe_resume(run_dir, problem=problem, driver=driver, resume=args.resume, label="energy_vs_J")
    run_ground_state_steps(
        label="energy_vs_J",
        driver=driver,
        run_dir=run_dir,
        problem=problem,
        n_steps=args.n_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        energy_scale=shape[0] * shape[1],
    )


if __name__ == "__main__":
    main()

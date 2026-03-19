"""3x3 quoted-ED benchmark for Z2 gauge fields coupled to hard-core bosons."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

if __package__ in (None, ""):
    from common import (
        DEFAULT_BOUNDARY_SWEEPS,
        DEFAULT_DIAG_SHIFT,
        DEFAULT_DT,
        DEFAULT_LOG_EVERY,
        DEFAULT_SAVE_EVERY,
        build_ground_state_driver,
        maybe_resume,
        prepare_run_dir,
        run_ground_state_steps,
    )
else:
    from .common import (
        DEFAULT_BOUNDARY_SWEEPS,
        DEFAULT_DIAG_SHIFT,
        DEFAULT_DT,
        DEFAULT_LOG_EVERY,
        DEFAULT_SAVE_EVERY,
        build_ground_state_driver,
        maybe_resume,
        prepare_run_dir,
        run_ground_state_steps,
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


def _problem(args: argparse.Namespace) -> dict:
    return {
        "script": "benchmark_3x3",
        "shape": [3, 3],
        "particle_number": PARTICLE_NUMBER,
        "h": H_COUPLING,
        "g": G_COUPLING,
        "J": J_COUPLING,
        "m": M_COUPLING,
        "bond_dim_per_charge": BOND_DIM_PER_CHARGE,
        "boundary_dimension": BOUNDARY_DIM,
        "boundary_sweeps": args.boundary_sweeps,
        "quoted_ed_energy_per_site": QUOTED_ED_ENERGY_PER_SITE,
        "n_samples": args.n_samples,
        "n_chains": args.n_chains,
        "dt": args.dt,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
    }


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
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    problem = _problem(args)
    driver = build_ground_state_driver(
        shape=SHAPE,
        h=H_COUPLING,
        g=G_COUPLING,
        J=J_COUPLING,
        m=M_COUPLING,
        particle_number=PARTICLE_NUMBER,
        bond_dim_per_charge=BOND_DIM_PER_CHARGE,
        boundary_dim=BOUNDARY_DIM,
        boundary_sweeps=args.boundary_sweeps,
        seed=args.seed,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        dt=args.dt,
        diag_shift=args.diag_shift,
    )
    run_dir = prepare_run_dir(_run_dir(), resume=args.resume)
    maybe_resume(run_dir, problem=problem, driver=driver, resume=args.resume, label="benchmark_3x3")
    run_ground_state_steps(
        label="benchmark_3x3",
        driver=driver,
        run_dir=run_dir,
        problem=problem,
        n_steps=args.n_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        energy_scale=SHAPE[0] * SHAPE[1],
        update_row=lambda _driver, row: row.update(
            absolute_error_vs_quoted_ed=abs(
                row["energy_mean"] - QUOTED_ED_ENERGY_PER_SITE
            )
        ),
        format_extra=lambda row: (
            "abs_err_vs_ed"
            if not row
            else f"{row['absolute_error_vs_quoted_ed']:.6e}"
        ),
    )


if __name__ == "__main__":
    main()

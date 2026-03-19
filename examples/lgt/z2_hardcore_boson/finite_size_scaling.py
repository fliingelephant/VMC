"""One half-filled finite-size point with central-bulk energies for the paper scan."""
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
        build_central_bulk_observable,
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
        build_central_bulk_observable,
        build_ground_state_driver,
        format_token,
        half_filling,
        maybe_resume,
        prepare_run_dir,
        run_ground_state_steps,
    )


DEFAULT_L = 16
DEFAULT_DK = 6
DEFAULT_N_SAMPLES = 4096
DEFAULT_N_CHAINS = 64
DEFAULT_N_STEPS = 200
DEFAULT_SEED = 42


def _bulk_sizes(L: int) -> tuple[int, ...]:
    sizes = []
    for bulk_size in (L, L - 2, L - 4):
        if bulk_size > 0:
            sizes.append(bulk_size)
    return tuple(sizes)


def _run_dir(args: argparse.Namespace) -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "finite_size_scaling"
        / (
            f"L{args.L}_{coupling_suffix(h=args.h, g=args.g, J=args.J, m=args.m)}"
            f"_Dk{args.bond_dim_per_charge}"
        )
    )


def _problem(args: argparse.Namespace, particle_number: int) -> dict:
    return {
        "script": "finite_size_scaling",
        "shape": [args.L, args.L],
        "particle_number": particle_number,
        "bulk_sizes": list(_bulk_sizes(args.L)),
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
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shape = (args.L, args.L)
    particle_number = half_filling(shape)
    bulk_sizes = _bulk_sizes(args.L)
    observables = tuple(
        build_central_bulk_observable(
            shape,
            h=args.h,
            g=args.g,
            J=args.J,
            m=args.m,
            bulk_size=bulk_size,
        )
        for bulk_size in bulk_sizes
    )
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
        observables=observables,
    )
    run_dir = prepare_run_dir(_run_dir(args), resume=args.resume)
    maybe_resume(
        run_dir,
        problem=problem,
        driver=driver,
        resume=args.resume,
        label="finite_size_scaling",
    )
    run_ground_state_steps(
        label="finite_size_scaling",
        driver=driver,
        run_dir=run_dir,
        problem=problem,
        n_steps=args.n_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        energy_scale=shape[0] * shape[1],
        update_row=lambda current_driver, row: row.update(
            {
                key: value
                for bulk_size, stats in zip(bulk_sizes, current_driver.observable_stats)
                for key, value in (
                    (
                        f"bulk{bulk_size}_energy_per_site",
                        float(stats.mean.real) / (bulk_size * bulk_size),
                    ),
                    (
                        f"bulk{bulk_size}_error",
                        float(stats.error_of_mean.real) / (bulk_size * bulk_size),
                    ),
                )
            }
        ),
        format_extra=lambda row: (
            " ".join(
                f"bulk{bulk_size}_energy_per_site bulk{bulk_size}_err"
                for bulk_size in bulk_sizes
            )
            if not row
            else " ".join(
                (
                    f"{row[f'bulk{bulk_size}_energy_per_site']:.10f} "
                    f"{row[f'bulk{bulk_size}_error']:.6e}"
                )
                for bulk_size in bulk_sizes
            )
        ),
    )


if __name__ == "__main__":
    main()

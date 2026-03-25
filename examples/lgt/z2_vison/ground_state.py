"""Z2 pure gauge ground-state optimization with GI-PEPS.

Reproduces the ground-state preparation for Wu & Liu (2025) Fig. 5:
  --L 6  --bond-dim 3  -> Fig 5a (6x6)
  --L 10 --bond-dim 4  -> Fig 5b (10x10)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.gauge import GaugeConfig  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner  # noqa: E402

from vmc.workflow import DEFAULT_METRICS_CONFIG, SOLVERS, SPACES, add_common_args, run  # noqa: E402
from physics import build_model, build_z2_hamiltonian  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize Z2 pure gauge ground state with GI-PEPS.",
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.1)
    add_common_args(parser)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.set_defaults(bond_dim=3, dt=0.005, diag_shift=1e-6, n_steps=400)
    args = parser.parse_args()

    shape = (args.L, args.L)
    model = build_model(shape, bond_dim=args.bond_dim, seed=args.seed)
    hamiltonian = build_z2_hamiltonian(shape, h=args.h, g=args.g)

    driver = TDVPDriver(
        model,
        hamiltonian,
        preconditioner=SRPreconditioner(
            space=SPACES[args.solver_space](),
            strategy=DirectSolve(solver=SOLVERS[args.solver]),
            diag_shift=args.diag_shift,
            gauge_config=GaugeConfig() if args.gauge_removal else None,
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
    run(
        driver,
        n_steps=args.n_steps,
        run_dir=args.output or f"data/z2_vison/L{args.L}_g{g_tok}_Dk{args.bond_dim}",
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "gauge_group": "Z2", "L": args.L,
            "h": args.h, "g": args.g,
            "bond_dim": args.bond_dim, "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()

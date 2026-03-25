"""Z2 Higgs vison confinement dynamics with GI-PEPS.

Loads a ground state, creates an interior vison pair, and runs real-time
TDVP tracking all plaquette observables for 2D map snapshots.

Wu & Nys (2026) Fig. 4:
  Deconfined: J=0.1, g=0.1
  Higgs:      J=0.5, g=0.1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver  # noqa: E402
from vmc.gauge import GaugeConfig  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner  # noqa: E402

from vmc.workflow import (
    SOLVERS, SPACES,  # noqa: E402
    DEFAULT_METRICS_CONFIG,
    add_common_args,
    load_model_from_checkpoint,
    read_config,
    run,
)
from physics import (  # noqa: E402
    build_all_plaquette_observables,
    build_model,
    build_z2_higgs_hamiltonian,
    create_interior_vison_pair,
    default_vison_link,
    plaquette_observable_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Z2 Higgs vison confinement dynamics.",
    )
    parser.add_argument("--state", type=str, required=True,
                        help="Path to ground-state run_dir from ground_state.py")
    parser.add_argument("--vison-orientation", choices=("v", "h"), default="v")
    parser.add_argument("--vison-row", type=int, default=None)
    parser.add_argument("--vison-col", type=int, default=None)
    add_common_args(parser)
    parser.add_argument("--T-final", type=float, default=20.0, dest="T_final")
    parser.set_defaults(
        bond_dim=2, dt=0.005, diag_shift=1e-8,
        n_samples=4096, n_chains=512,
        solver_space="minsr", save_every=20, log_every=10,
    )
    args = parser.parse_args()

    # Read ground-state config from checkpoint metadata
    extra = read_config(args.state).get("extra", {})
    L = int(extra["L"])
    h = float(extra["h"])
    g = float(extra["g"])
    J = float(extra["J"])
    sigma_z_field = float(extra["sigma_z_field"])
    bond_dim = int(extra["bond_dim"])
    boundary_dim = int(extra["boundary_dim"])
    boundary_sweeps = int(extra["boundary_sweeps"])
    seed = int(extra["seed"])

    shape = (L, L)
    model = build_model(
        shape, bond_dim=bond_dim, boundary_dim=boundary_dim,
        boundary_sweeps=boundary_sweeps, seed=seed,
    )
    model, _ = load_model_from_checkpoint(args.state, model)

    # Insert vison pair
    orientation = args.vison_orientation
    if args.vison_row is not None and args.vison_col is not None:
        vison_row, vison_col = args.vison_row, args.vison_col
    else:
        vison_row, vison_col = default_vison_link(shape, orientation)
    model = create_interior_vison_pair(
        model, orientation=orientation, row=vison_row, col=vison_col,
    )

    hamiltonian = build_z2_higgs_hamiltonian(
        shape, h=h, g=g, J=J, sigma_z_field=sigma_z_field,
    )
    observables = build_all_plaquette_observables(shape)
    plaq_names = plaquette_observable_names(shape)

    driver = TDVPDriver(
        model, hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=SPACES[args.solver_space](),
            strategy=DirectSolve(solver=SOLVERS[args.solver]),
            diag_shift=args.diag_shift,
            gauge_config=GaugeConfig() if args.gauge_removal else None,
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(seed + 1),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        full_gradient=args.full_gradient,
    )

    g_tok = format(g, ".3f").replace(".", "p")
    J_tok = format(J, ".3f").replace(".", "p")
    run(
        driver,
        T_final=args.T_final,
        run_dir=args.output or (
            f"data/z2_vison_higgs/L{L}_g{g_tok}_J{J_tok}_Dk{bond_dim}"
            f"_rt_vison_{orientation}_r{vison_row}_c{vison_col}"
        ),
        observable_names=plaq_names,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "model": "z2_vison_higgs", "L": L,
            "h": h, "g": g, "J": J, "sigma_z_field": sigma_z_field,
            "vison_orientation": orientation,
            "vison_row": vison_row, "vison_col": vison_col,
        },
    )


if __name__ == "__main__":
    main()

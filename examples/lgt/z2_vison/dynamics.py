"""Z2 vison propagation real-time dynamics with GI-PEPS.

Loads a ground-state checkpoint, inserts a boundary vison, and runs
real-time TDVP tracking selected plaquette expectation values.

Reproduces Wu & Liu (2025) Fig. 5:
  --L 6  -> Fig 5a (exact comparison available via plot.py)
  --L 10 -> Fig 5b
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver  # noqa: E402
from vmc.preconditioners import SRPreconditioner  # noqa: E402

from vmc.workflow import (  # noqa: E402
    DEFAULT_METRICS_CONFIG,
    add_common_args,
    load_model_from_checkpoint,
    run,
)
from physics import (  # noqa: E402
    build_model,
    build_selected_plaquette_observables,
    build_z2_hamiltonian,
    create_bottom_left_vison,
    selected_open_plaquettes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Z2 vison propagation real-time dynamics.",
    )
    parser.add_argument("--state", type=str, required=True,
                        help="Path to ground-state run_dir from ground_state.py")
    add_common_args(parser)
    parser.set_defaults(bond_dim=3, dt=0.01, diag_shift=1e-8)
    args = parser.parse_args()

    # Load ground state from runner checkpoint
    gs_config = args.state
    dummy_model = build_model(
        (2, 2), bond_dim=args.bond_dim, seed=args.seed,
    )
    # Read metadata to get actual shape/bond_dim
    import json
    with open(Path(gs_config) / "latest.json") as f:
        metadata = json.load(f)
    gs_extra = metadata.get("config", {}).get("extra", {})
    L = int(gs_extra.get("L", args.L if hasattr(args, "L") else 6))
    bond_dim = int(gs_extra.get("bond_dim", args.bond_dim))
    h = float(gs_extra.get("h", 1.0))
    g = float(gs_extra.get("g", 0.1))
    seed = int(gs_extra.get("seed", args.seed))

    shape = (L, L)
    model = build_model(shape, bond_dim=bond_dim, seed=seed)
    model, _ = load_model_from_checkpoint(gs_config, model)
    model = create_bottom_left_vison(model)

    hamiltonian = build_z2_hamiltonian(shape, h=h, g=g)
    plaquettes = selected_open_plaquettes(shape)
    observables = build_selected_plaquette_observables(shape, plaquettes)
    plaq_names = tuple(f"P_{r}{c}" for r, c in plaquettes)

    driver = TDVPDriver(
        model,
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            diag_shift=args.diag_shift,
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
    run(
        driver,
        T_final=args.T_final,
        run_dir=args.output or f"data/z2_vison/L{L}_g{g_tok}_Dk{bond_dim}_rt",
        observable_names=plaq_names,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "gauge_group": "Z2", "L": L, "h": h, "g": g,
            "vison": "sigma_z on bottom-left vertical link",
            "selected_plaquettes": [list(p) for p in plaquettes],
        },
    )


if __name__ == "__main__":
    main()

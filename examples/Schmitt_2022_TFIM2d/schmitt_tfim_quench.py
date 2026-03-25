"""PEPS tVMC quench for the 2D transverse-field Ising model.

This script follows the smooth quench used for tensor-network simulations in
Schmitt et al. (2022):

    epsilon(t) = t / tau_q - 4 t^3 / (27 tau_q^3)
    J(t) = 1 + epsilon(t)
    g(t) = g_c (1 - epsilon(t))

with t in [-3 tau_q / 2, 3 tau_q / 2] on an odd-L open square lattice.

The initial state is the exact |+x>^N product state embedded into a PEPS of
bond dimension D.
"""
from __future__ import annotations

from pathlib import Path


from vmc import config  # noqa: F401, E402

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from vmc.drivers import RK4, RealTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import (  # noqa: E402
    CubicSchedule,
    DiagonalOperator,
    LocalHamiltonian,
    OneSiteOperator,
    TimeDependentHamiltonian,
)
from vmc.peps import PEPS, Variational  # noqa: E402
from vmc.preconditioners import DirectSolve, SRPreconditioner  # noqa: E402

from vmc.workflow import (  # noqa: E402
    DEFAULT_METRICS_CONFIG,
    SOLVERS,
    SPACES,
    add_common_args,
    run,
)


GC = 3.04438
SIGMA_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
SIGMA_ZZ_DIAG = jnp.asarray([1.0, -1.0, -1.0, 1.0], dtype=jnp.complex128)
PLUS_X = jnp.asarray([1.0, 1.0], dtype=jnp.complex128) / jnp.sqrt(2.0)


def smooth_time_window(tau_q: float) -> tuple[float, float]:
    """Return the Schmitt smooth-ramp time window."""
    return (-1.5 * tau_q, 1.5 * tau_q)


def center_site(shape: tuple[int, int]) -> tuple[int, int]:
    """Return the unique central site for an odd-by-odd lattice."""
    if shape[0] % 2 == 0 or shape[1] % 2 == 0:
        raise ValueError("Center-spin correlations require an odd-by-odd lattice.")
    return (shape[0] // 2, shape[1] // 2)


def center_axis_distances(shape: tuple[int, int]) -> tuple[int, ...]:
    """Return paper-aligned axis distances from the central spin."""
    row, col = center_site(shape)
    max_distance = min(row, shape[0] - 1 - row, col, shape[1] - 1 - col)
    return tuple(range(1, max_distance + 1))


def build_center_czz_observables(
    shape: tuple[int, int],
) -> tuple[LocalHamiltonian, ...]:
    """Build Czz(R) observables relative to the central spin along lattice axes."""
    row, col = center_site(shape)
    observables = []
    for distance in center_axis_distances(shape):
        sites = (
            ((row, col), (row, col - distance)),
            ((row, col), (row, col + distance)),
            ((row, col), (row - distance, col)),
            ((row, col), (row + distance, col)),
        )
        diag = SIGMA_ZZ_DIAG / len(sites)
        observables.append(
            LocalHamiltonian(
                shape=shape,
                terms=tuple(DiagonalOperator(sites=pair, diag=diag) for pair in sites),
            )
        )
    return tuple(observables)


def build_mx_observable(shape: tuple[int, int]) -> LocalHamiltonian:
    """Build average transverse magnetization."""
    coeff = 1.0 / (shape[0] * shape[1])
    return LocalHamiltonian(
        shape=shape,
        terms=tuple(
            OneSiteOperator(row=row, col=col, op=coeff * SIGMA_X)
            for row in range(shape[0])
            for col in range(shape[1])
        ),
    )


def build_plus_x_product_peps(
    shape: tuple[int, int],
    bond_dim: int,
    boundary_dim: int,
    seed: int,
) -> PEPS:
    """Build the exact |+x>^N product state as a PEPS."""
    model = PEPS(
        rngs=nnx.Rngs(seed),
        shape=shape,
        bond_dim=bond_dim,
        phys_dim=2,
        contraction_strategy=Variational(boundary_dim),
        dtype=jnp.complex128,
    )
    n_rows, n_cols = shape
    for row in range(n_rows):
        for col in range(n_cols):
            up, down, left, right = model.site_dims(
                row, col, n_rows, n_cols, bond_dim
            )
            tensor = jnp.zeros((2, up, down, left, right), dtype=jnp.complex128)
            model.tensors[row][col][...] = tensor.at[:, 0, 0, 0, 0].set(PLUS_X)
    return model


def build_schmitt_smooth_tfim(
    shape: tuple[int, int],
    tau_q: float,
    gc: float,
) -> TimeDependentHamiltonian:
    """Build the smooth-ramp Schmitt TFIM Hamiltonian."""
    terms = []
    constant = []
    linear = []
    quadratic = []
    cubic = []
    j_cubic = -4.0 / (27.0 * tau_q**3)
    g_cubic = 4.0 * gc / (27.0 * tau_q**3)

    for row in range(shape[0]):
        for col in range(shape[1]):
            terms.append(OneSiteOperator(row=row, col=col, op=-SIGMA_X))
            constant.append(gc)
            linear.append(-gc / tau_q)
            quadratic.append(0.0)
            cubic.append(g_cubic)

            if col + 1 < shape[1]:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row, col + 1)),
                        diag=-SIGMA_ZZ_DIAG,
                    )
                )
                constant.append(1.0)
                linear.append(1.0 / tau_q)
                quadratic.append(0.0)
                cubic.append(j_cubic)

            if row + 1 < shape[0]:
                terms.append(
                    DiagonalOperator(
                        sites=((row, col), (row + 1, col)),
                        diag=-SIGMA_ZZ_DIAG,
                    )
                )
                constant.append(1.0)
                linear.append(1.0 / tau_q)
                quadratic.append(0.0)
                cubic.append(j_cubic)

    return TimeDependentHamiltonian(
        base=LocalHamiltonian(shape=shape, terms=tuple(terms)),
        schedule=CubicSchedule(
            constant=jnp.asarray(constant, dtype=jnp.float64),
            linear=jnp.asarray(linear, dtype=jnp.float64),
            quadratic=jnp.asarray(quadratic, dtype=jnp.float64),
            cubic=jnp.asarray(cubic, dtype=jnp.float64),
        ),
    )


def run_single_quench(args, tau_q: float) -> None:
    """Run one Schmitt-style smooth quench."""
    shape = (args.L, args.L)
    center_site(shape)  # validate odd lattice
    t0, t1 = smooth_time_window(tau_q)
    distances = center_axis_distances(shape)
    obs_names = ("mx", *(f"czz_r{d}" for d in distances))
    observables = (build_mx_observable(shape), *build_center_czz_observables(shape))

    model = build_plus_x_product_peps(shape, args.bond_dim, args.boundary_dim, args.seed)
    driver = TDVPDriver(
        model,
        build_schmitt_smooth_tfim(shape, tau_q, GC),
        observables=observables,
        preconditioner=SRPreconditioner(
            space=SPACES[args.solver_space](),
            diag_shift=args.diag_shift,
            strategy=DirectSolve(solver=SOLVERS[args.solver]),
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=args.dt,
        t0=t0,
        time_unit=RealTimeUnit(),
        integrator=RK4(),
        sampler_key=jax.random.key(args.seed + 17),
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        full_gradient=args.full_gradient,
    )

    tau_q_tok = format(tau_q, ".4f").replace(".", "p")
    run_dir = args.output or f"data/tfim_quench/L{args.L}_tauq{tau_q_tok}_D{args.bond_dim}"
    run(
        driver,
        T_final=t1,
        run_dir=run_dir,
        observable_names=obs_names,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        extra_config={
            "L": args.L, "tau_q": tau_q, "gc": GC,
            "initial_state": "product |+x>",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a PEPS Schmitt-style smooth TFIM quench on an odd OBC lattice.",
    )
    parser.add_argument("--L", type=int, default=7)
    parser.add_argument("--tau-q", type=float, nargs="+", default=[0.8])
    add_common_args(parser)
    parser.set_defaults(bond_dim=4, boundary_dim=16, diag_shift=1e-8)
    args = parser.parse_args()

    for tau_q in args.tau_q:
        run_single_quench(args, tau_q)


if __name__ == "__main__":
    main()

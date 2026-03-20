"""Fixed-step SR benchmark for pure Z2 lattice gauge theory.

Records total energy, mean plaquette value, mean horizontal-link Z, and mean
vertical-link Z expectation values.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmc import config  # noqa: F401, E402 - JAX config must be imported first

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from vmc.drivers import ImaginaryTimeUnit, TDVPDriver  # noqa: E402
from vmc.operators import PlaquetteOperator  # noqa: E402
from vmc.peps import ZipUp  # noqa: E402
from vmc.peps.gi import GILocalHamiltonian, GIPEPS, GIPEPSConfig  # noqa: E402
from vmc.peps.gi.local_terms import LinkDiagonalTerm, build_electric_terms  # noqa: E402
from vmc.preconditioners import (  # noqa: E402
    DirectSolve,
    SRPreconditioner,
    solve_cholesky,
)
from vmc.qgt import ParameterSpace  # noqa: E402

from runner import DEFAULT_METRICS_CONFIG, run  # noqa: E402


L = 3
SHAPE = (L, L)
H_COUPLING = 1.0
G_COUPLING = 0.2

BOND_DIM = 2
BOUNDARY_DIM = 3 * BOND_DIM

N_SAMPLES = 1024
N_CHAINS = 64
SEED = 42

SR_FIXED_STEPS = 100
SR_FIXED_DT = 0.01
SR_DIAG_SHIFT = 1e-8


def build_z2_hamiltonian(
    shape: tuple[int, int],
    h: float,
    g: float,
) -> GILocalHamiltonian:
    """Build the pure Z2 gauge Hamiltonian."""
    n_rows, n_cols = shape
    plaquette_terms = tuple(
        PlaquetteOperator(row=r, col=c)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    electric_terms = build_electric_terms(shape, N=2)
    return GILocalHamiltonian(
        shape=shape,
        terms=electric_terms + plaquette_terms,
        coeffs=(jnp.asarray(g),) * len(electric_terms)
        + (jnp.asarray(-h),) * len(plaquette_terms),
    )


def build_mean_plaquette_observable(shape: tuple[int, int]) -> GILocalHamiltonian:
    """Build the average plaquette operator.

    ``PlaquetteOperator`` evaluates ``P + P†``. For Z2, ``P = P†``, so the
    average plaquette value is obtained with a coefficient of
    ``1 / (2 * n_plaquettes)``.
    """
    n_rows, n_cols = shape
    n_plaquettes = (n_rows - 1) * (n_cols - 1)
    coeff = jnp.asarray(0.5 / n_plaquettes, dtype=jnp.complex128)
    terms = tuple(
        PlaquetteOperator(row=r, col=c)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    return GILocalHamiltonian(shape=shape, terms=terms, coeffs=(coeff,) * len(terms))


def build_mean_link_z_observable(
    shape: tuple[int, int],
    *,
    orientation: str,
) -> GILocalHamiltonian:
    """Build the average Z expectation on horizontal or vertical links."""
    n_rows, n_cols = shape
    if orientation == "h":
        count = n_rows * (n_cols - 1)
        row_range = range(n_rows)
        col_range = range(n_cols - 1)
    elif orientation == "v":
        count = (n_rows - 1) * n_cols
        row_range = range(n_rows - 1)
        col_range = range(n_cols)
    else:
        raise ValueError(f"Unsupported orientation: {orientation!r}")

    diag = jnp.asarray([1.0, -1.0], dtype=jnp.complex128) / count
    terms = tuple(
        LinkDiagonalTerm(
            sites=((r, c),),
            diag=diag,
            orientation=orientation,
        )
        for r in row_range
        for c in col_range
    )
    return GILocalHamiltonian(shape=shape, terms=terms)


def build_problem() -> tuple[GILocalHamiltonian, tuple[GILocalHamiltonian, ...]]:
    """Build the Hamiltonian and benchmark observables."""
    return (
        build_z2_hamiltonian(SHAPE, H_COUPLING, G_COUPLING),
        (
            build_mean_plaquette_observable(SHAPE),
            build_mean_link_z_observable(SHAPE, orientation="h"),
            build_mean_link_z_observable(SHAPE, orientation="v"),
        ),
    )


def build_model(seed: int) -> GIPEPS:
    """Build a fresh GIPEPS model."""
    return GIPEPS(
        rngs=nnx.Rngs(seed),
        config=GIPEPSConfig(
            shape=SHAPE,
            N=2,
            phys_dim=1,
            Qx=0,
            degeneracy_per_charge=(BOND_DIM, BOND_DIM),
            charge_of_site=(0,),
        ),
        contraction_strategy=ZipUp(truncate_bond_dimension=BOUNDARY_DIM),
    )


def benchmark_output_dir() -> Path:
    """Build the output directory for the current benchmark settings."""
    g_token = format(G_COUPLING, ".3f").replace(".", "p")
    return (
        Path(__file__).resolve().parent
        / f"z2_pure_gauge_benchmark_{L}x{L}_g{g_token}_ns{N_SAMPLES}_{SR_FIXED_STEPS}"
    )


def main() -> None:
    """Run the fixed-step SR benchmark."""
    hamiltonian, observables = build_problem()
    print(
        (
            f"Benchmarking pure Z2 gauge theory on {SHAPE}, "
            f"h={H_COUPLING:.3f}, g={G_COUPLING:.3f}, "
            f"Dk={BOND_DIM}, Dc={BOUNDARY_DIM}, nsamples={N_SAMPLES}"
        ),
        flush=True,
    )
    driver = TDVPDriver(
        build_model(SEED),
        hamiltonian,
        observables=observables,
        preconditioner=SRPreconditioner(
            space=ParameterSpace(),
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=SR_DIAG_SHIFT,
            metrics_config=DEFAULT_METRICS_CONFIG,
        ),
        dt=SR_FIXED_DT,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(SEED),
        n_samples=N_SAMPLES,
        n_chains=N_CHAINS,
        full_gradient=False,
    )
    run(
        driver,
        n_steps=SR_FIXED_STEPS,
        run_dir=benchmark_output_dir(),
        observable_names=("plaquette", "z_h", "z_v"),
        log_every=1,
        save_every=SR_FIXED_STEPS,
        extra_config={
            "gauge_group": "Z2",
            "L": L,
            "h": H_COUPLING,
            "g": G_COUPLING,
        },
    )


if __name__ == "__main__":
    main()

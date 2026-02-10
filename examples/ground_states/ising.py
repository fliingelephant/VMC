"""Ground state optimization for 2D transverse-field Ising model.

Uses a rectangular grid with open boundary conditions:
- shape = (n_rows, n_cols)
- Coupling J = -1.0
- Transverse field h = 0.5
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import logging

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.drivers import TDVPDriver, ImaginaryTimeUnit
from vmc.operators import DiagonalTerm, LocalHamiltonian, OneSiteTerm
from vmc.peps import PEPS, ZipUp
from vmc.preconditioners import SRPreconditioner, DirectSolve
from vmc.qgt import SampleSpace
from vmc.qgt.solvers import solve_cholesky

logger = logging.getLogger(__name__)


def build_ising_2d(
    shape: tuple[int, int] = (5, 5),
    J: float = -1.0,
    h: float = 0.5,
) -> LocalHamiltonian:
    """Build 2D transverse-field Ising Hamiltonian."""
    n_rows, n_cols = shape
    diag_zz = -J * jnp.array([1, -1, -1, 1], dtype=jnp.complex128)
    op_x = -h * jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    terms = []
    for row in range(n_rows):
        for col in range(n_cols):
            terms.append(OneSiteTerm(row, col, op_x))
            if col + 1 < n_cols:
                terms.append(DiagonalTerm(((row, col), (row, col + 1)), diag_zz))
            if row + 1 < n_rows:
                terms.append(DiagonalTerm(((row, col), (row + 1, col)), diag_zz))
    return LocalHamiltonian(shape=shape, terms=tuple(terms))


def run_optimization(
    model,
    H,
    exact_e: float | None = None,
    n_samples=2048,
    n_steps=120,
    dt=0.01,
    diag_shift=0.01,
    seed=42,
):
    """Run ground state optimization loop."""
    driver = TDVPDriver(
        model,
        H,
        preconditioner=SRPreconditioner(
            space=SampleSpace(),
            strategy=DirectSolve(solver=solve_cholesky),
            diag_shift=diag_shift
        ),
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        n_chains=64,
        full_gradient=False,
    )

    k = 5
    n_chunks = n_steps // k
    assert n_steps == n_chunks * k, (
        f"n_steps={n_steps} must be a multiple of chunk size k={k}"
    )
    for _ in range(n_chunks):
        driver.run(k * dt)

    e = driver.energy
    logger.info("Final: E = %.6f ± %.4f [σ²=%.4f]", e.mean.real, e.error_of_mean.real, e.variance.real)
    if exact_e is not None:
        logger.info("Exact: E = %.6f", exact_e)
        logger.info("Absolute error: %.2e", abs(e.mean.real - exact_e))


def main(shape: tuple[int, int] = (4, 4), J: float = -1.0, h: float = 0.5):
    """Run ground state optimization with PEPS."""
    n_sites = shape[0] * shape[1]
    graph = nk.graph.Grid(extent=shape, pbc=False)
    hi = nk.hilbert.Spin(s=0.5, N=n_sites)
    H_ed = nk.operator.Ising(hi, graph, h=h, J=J, dtype=jnp.complex128)
    exact_e = nk.exact.lanczos_ed(H_ed, k=1)[0].real
    H = build_ising_2d(shape, J, h)
    logger.info("Hamiltonian: transverse-field Ising, J=%.3f, h=%.3f", J, h)
    logger.info("System size: %d sites", n_sites)
    logger.info("Exact ground state energy: %.6f (%.6f per site)", exact_e, exact_e / n_sites)

    # PEPS
    logger.info("=" * 60)
    logger.info("PEPS with shape=%s, bond_dim=4", shape)
    logger.info("=" * 60)
    peps = PEPS(
        rngs=nnx.Rngs(42),
        shape=shape,
        bond_dim=4,
        contraction_strategy=ZipUp(truncate_bond_dimension=16),
    )
    run_optimization(peps, H, exact_e=exact_e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

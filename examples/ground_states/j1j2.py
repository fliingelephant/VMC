"""Ground state optimization for J1-J2 Model.

This example matches the NetKet tutorial parameters:
- 1D chain with 14 sites and periodic boundary conditions
- J1 = 1.0 (nearest-neighbor coupling)
- J2 = 0.2 (next-nearest-neighbor coupling, low frustration)

Note: NetKet tutorial uses total_sz=0 constraint with MetropolisExchange sampler.
Our sequential sampler uses single-site flips, so we don't enforce this constraint.

Runs both MPS (natural for 1D) and PEPS for comparison.

Reference: https://netket.readthedocs.io/en/stable/tutorials/gs-j1j2.html
"""

from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import functools
import logging

import jax
import jax.numpy as jnp
import netket as nk
from flax import nnx

from vmc.drivers import DynamicsDriver, ImaginaryTimeUnit
from vmc.models.mps import MPS
from vmc.models.peps import PEPS, ZipUp
from vmc.preconditioners import SRPreconditioner
from vmc.samplers.sequential import sequential_sample_with_gradients

logger = logging.getLogger(__name__)


def build_j1j2_graph(n_sites: int) -> nk.graph.Graph:
    """Build J1-J2 chain graph with edge coloring."""
    edges_j1 = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    edges_j2 = [(i, (i + 2) % n_sites) for i in range(n_sites)]
    edge_colors = [0] * len(edges_j1) + [1] * len(edges_j2)
    return nk.graph.Graph(edges=edges_j1 + edges_j2, edge_colors=edge_colors, n_nodes=n_sites)


def build_j1j2_1d(
    n_sites: int = 14,
    j1: float = 1.0,
    j2: float = 0.2,
) -> tuple[nk.hilbert.Spin, nk.operator.Heisenberg]:
    """Build 1D J1-J2 Heisenberg Hamiltonian."""
    graph = build_j1j2_graph(n_sites)
    hi = nk.hilbert.Spin(s=0.5, N=n_sites)
    return hi, nk.operator.Heisenberg(hi, graph, J=(j1, j2), dtype=jnp.complex128)


def run_optimization(model, H, exact_e, n_samples=1008, n_steps=600, dt=0.01, diag_shift=0.01, seed=42, log_interval=20):
    """Run ground state optimization loop."""
    sampler = functools.partial(
        sequential_sample_with_gradients,
        n_samples=n_samples,
        burn_in=8,
        full_gradient=False,
    )
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=SRPreconditioner(diag_shift=diag_shift),
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(seed),
    )

    for step in range(n_steps):
        driver.step()
        if step % log_interval == 0 and driver.energy is not None:
            e = driver.energy
            logger.info(
                "Step %4d | E = %.6f ± %.4f [σ²=%.4f] | Error = %.2e",
                step,
                e.mean.real,
                e.error_of_mean.real,
                e.variance.real,
                abs(e.mean.real - exact_e),
            )

    e = driver.energy
    logger.info("Final: E = %.6f ± %.4f [σ²=%.4f]", e.mean.real, e.error_of_mean.real, e.variance.real)
    logger.info("Exact: E = %.6f", exact_e)
    logger.info("Absolute error: %.2e", abs(e.mean.real - exact_e))


def main(n_sites: int = 14, j1: float = 1.0, j2: float = 0.2):
    """Run ground state optimization with both MPS and PEPS."""
    _, H = build_j1j2_1d(n_sites, j1, j2)
    exact_e = nk.exact.lanczos_ed(H, k=1)[0].real
    logger.info("J1-J2 model: J1=%.2f, J2=%.2f (frustration ratio=%.2f)", j1, j2, j2 / j1)
    logger.info("Exact ground state energy: %.6f (%.6f per site)", exact_e, exact_e / n_sites)

    # MPS (natural for 1D chains)
    logger.info("=" * 60)
    logger.info("MPS with bond_dim=16")
    logger.info("=" * 60)
    mps = MPS(rngs=nnx.Rngs(42), n_sites=n_sites, bond_dim=16)
    run_optimization(mps, H, exact_e)

    # PEPS (for comparison)
    logger.info("=" * 60)
    logger.info("PEPS with shape=(2,7), bond_dim=4")
    logger.info("=" * 60)
    peps = PEPS(
        rngs=nnx.Rngs(42),
        shape=(2, 7),
        bond_dim=4,
        contraction_strategy=ZipUp(truncate_bond_dimension=16),
    )
    run_optimization(peps, H, exact_e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

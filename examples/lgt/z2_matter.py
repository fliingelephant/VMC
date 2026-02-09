"""Z2 gauge theory coupled to hard-core bosons.

This example demonstrates GI-PEPS simulation of Z2 gauge fields coupled
to dynamical matter (hard-core bosons). The matter fields live on vertices
and interact with gauge fields via gauge-covariant hopping.

The Hamiltonian is:
    H = H_M + H_B + H_E

    H_M = Σ_x m_x n_x + Σ_{x,α} (J c†_x U_{x,α} c_{x+e_α} + h.c.)
    H_B = -h Σ_x (P_x + P_x†)
    H_E = g Σ_links (2 - 2cos(2πE/N))

where n_x = c†_x c_x is the boson number operator.

With dynamical matter fields, the Wilson loop exhibits perimeter-law
behavior even in the confinement regime due to screening.

To replicate Fig 4:
- (a): Energy convergence vs bond dimension D
- (b): Finite-size scaling using central bulk energy
- (c): Energy vs J at fixed g
- (d): Wilson loops showing perimeter-law behavior

Reference: Wu & Liu, Phys. Rev. Lett. 135, 130401 (2025)
"""

from __future__ import annotations

from vmc import config  # noqa: F401

import logging
from pathlib import Path

import jax
from flax import nnx

from vmc.drivers import TDVPDriver, ImaginaryTimeUnit
from vmc.operators import PlaquetteTerm
from vmc.peps import ZipUp
from vmc.peps.gi.local_terms import GILocalHamiltonian, build_electric_terms
from vmc.peps.gi.model import GIPEPS, GIPEPSConfig
from vmc.preconditioners import SRPreconditioner

from .observables import SimulationData, format_step_log

logger = logging.getLogger(__name__)


def build_z2_matter_hamiltonian(
    shape: tuple[int, int],
    h: float = 1.0,
    g: float = 0.33,
    J: float = 0.5,
    m: float = 0.0,
) -> GILocalHamiltonian:
    """Build Z2 gauge + matter Hamiltonian.
    
    Args:
        shape: Lattice shape.
        h: Magnetic (plaquette) coupling.
        g: Electric field coupling.
        J: Hopping strength.
        m: Chemical potential.
    
    Returns:
        GILocalHamiltonian for Z2 gauge + hard-core bosons.
    """
    n_rows, n_cols = shape
    
    plaquette_terms = tuple(
        PlaquetteTerm(row=r, col=c, coeff=-h)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    
    electric_terms = build_electric_terms(shape, coeff=g, N=2)
    
    # TODO: Add hopping terms c†_x U_{x,α} c_{x+e_α}
    # This requires implementing HorizontalHoppingTerm and VerticalHoppingTerm
    # For now, only plaquette + electric terms are included
    
    return GILocalHamiltonian(shape=shape, terms=electric_terms + plaquette_terms)


def run_optimization(
    model: GIPEPS,
    operator: GILocalHamiltonian,
    *,
    n_samples: int = 500,
    n_steps: int = 200,
    dt: float = 0.01,
    diag_shift: float = 0.1,
    seed: int = 42,
    log_interval: int = 10,
    data: SimulationData | None = None,
) -> TDVPDriver:
    """Run imaginary-time ground state optimization."""
    driver = TDVPDriver(
        model,
        operator,
        preconditioner=SRPreconditioner(diag_shift=diag_shift),
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(seed),
        n_samples=n_samples,
        full_gradient=True,
    )

    for step in range(n_steps):
        driver.step()
        if step % log_interval == 0 and driver.energy is not None:
            e = driver.energy
            logger.info(format_step_log(
                step=step,
                energy=e.mean.real,
                energy_error=e.error_of_mean.real,
                energy_variance=e.variance.real,
            ))
            if data is not None:
                data.add_step(
                    step=step,
                    time=driver.t,
                    energy=e.mean.real,
                    energy_error=e.error_of_mean.real,
                    energy_variance=e.variance.real,
                )

    e = driver.energy
    logger.info("Final: E = %.6f ± %.4f [σ²=%.4f]", e.mean.real, e.error_of_mean.real, e.variance.real)
    return driver


def main(
    size: int = 6,
    g: float = 0.33,
    h: float = 1.0,
    J: float = 0.5,
    bond_dim: int = 4,
    n_samples: int = 5000,
    n_steps: int = 500,
    dt: float = 0.01,
    output_dir: str | None = None,
):
    """Run Z2 gauge + matter ground state optimization.
    
    Args:
        size: Lattice size (size x size). Paper uses up to 16×16.
        g: Electric field coupling.
        h: Magnetic (plaquette) coupling.
        J: Hopping strength.
        bond_dim: Bond dimension per charge sector (D_k). Paper needs D=12 for convergence.
        n_samples: MC samples per step. Paper uses 10^5 for matter coupling.
        n_steps: Optimization steps.
        dt: Imaginary time step.
        output_dir: Directory to save output data (JSON). None to skip saving.
    """
    shape = (size, size)
    
    logger.info("=" * 60)
    logger.info("Z2 Gauge Theory + Hard-Core Bosons")
    logger.info("=" * 60)
    logger.info(f"Lattice: {size}x{size}, h={h}, g={g}, J={J}")
    logger.info(f"Bond dimension per charge: D_k={bond_dim} (total D={2*bond_dim})")
    
    # Matter: phys_dim=2 (empty/occupied), charges 0 and 1
    cfg = GIPEPSConfig(
        shape=shape,
        N=2,
        phys_dim=2,
        Qx=0,
        degeneracy_per_charge=(bond_dim, bond_dim),
        charge_of_site=(0, 1),  # Empty → charge 0, Occupied → charge 1
    )
    
    model = GIPEPS(
        rngs=nnx.Rngs(42),
        config=cfg,
        contraction_strategy=ZipUp(truncate_bond_dimension=3 * bond_dim),
    )
    
    operator = build_z2_matter_hamiltonian(shape, h=h, g=g, J=J)
    
    logger.info("Note: Hopping terms not yet implemented. Only H_B + H_E included.")
    
    # Setup data logging
    data = SimulationData(
        model_type="Z2_matter",
        lattice_size=shape,
        parameters={"h": h, "g": g, "J": J, "bond_dim": bond_dim, "n_samples": n_samples},
    )
    
    run_optimization(model, operator, n_samples=n_samples, n_steps=n_steps, dt=dt, data=data)
    
    # Save data if output_dir specified
    if output_dir:
        path = Path(output_dir) / f"z2_matter_L{size}_g{g:.3f}_J{J:.2f}.json"
        data.save(path)
    
    return data


def scan_bond_dim(
    size: int = 10,
    bond_dims: tuple[int, ...] = (2, 3, 4, 5, 6),
    g: float = 0.33,
    J: float = 0.5,
    n_samples: int = 5000,
    n_steps: int = 500,
    dt: float = 0.01,
    output_dir: str = "output/z2_matter_Dscan",
):
    """Scan over bond dimensions to check convergence (replicates Fig 4a).
    
    The paper finds that D=12 (D_k=6) is needed for convergence with matter.
    """
    logger.info("=" * 60)
    logger.info("Z2+Matter Bond Dimension Scan")
    logger.info(f"Lattice: {size}x{size}, g={g}, J={J}")
    logger.info(f"Bond dims: {bond_dims}")
    logger.info("=" * 60)
    
    results = []
    for bond_dim in bond_dims:
        logger.info(f"\n{'='*40}\nRunning D_k = {bond_dim}\n{'='*40}")
        data = main(
            size=size, g=g, J=J, bond_dim=bond_dim,
            n_samples=n_samples, n_steps=n_steps, dt=dt,
            output_dir=output_dir,
        )
        results.append(data)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCAN SUMMARY")
    logger.info("=" * 60)
    for data in results:
        D_k = data.parameters["bond_dim"]
        e = data.energies[-1] if data.energies else float('nan')
        logger.info(f"D_k = {D_k} (D = {2*D_k}): E = {e:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

"""Ground state of pure Z2 lattice gauge theory.

This example demonstrates gauge-invariant PEPS simulation of pure Z2 LGT
following Wu & Liu (2025). The Hamiltonian is:

    H = -h Σ_x (P_x + P_x†) + g Σ_links (2 - 2cos(2πE/N))

where P_x is the plaquette operator and E is the electric field.

For Z2: cos(2πE/2) = cos(πE) = (-1)^E, so H_E = g Σ (2 - 2*(-1)^E) = g Σ 2*(1 - (-1)^E)

Phase transition at g_c ≈ 0.3285 (from QMC of dual transverse-field Ising model).
- g < g_c: Deconfined phase (perimeter-law Wilson loop)
- g > g_c: Confined phase (area-law Wilson loop)

Reference: Wu & Liu, Phys. Rev. Lett. 135, 130401 (2025)
"""

from __future__ import annotations

from vmc import config  # noqa: F401

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.drivers import TDVPDriver, ImaginaryTimeUnit
from vmc.operators import PlaquetteTerm
from vmc.peps import ZipUp
from vmc.peps.gi.local_terms import GILocalHamiltonian, build_electric_terms
from vmc.peps.gi.model import GIPEPS, GIPEPSConfig
from vmc.preconditioners import SRPreconditioner

from .observables import SimulationData, format_step_log

logger = logging.getLogger(__name__)


def build_z2_hamiltonian(
    shape: tuple[int, int],
    h: float = 1.0,
    g: float = 0.33,
) -> GILocalHamiltonian:
    """Build pure Z2 LGT Hamiltonian.
    
    Args:
        shape: Lattice shape (n_rows, n_cols).
        h: Magnetic (plaquette) coupling.
        g: Electric field coupling.
    
    Returns:
        GILocalHamiltonian for Z2 LGT.
    """
    n_rows, n_cols = shape
    
    # Plaquette terms: -h (P + P†) = -2h Re(P)
    plaquette_terms = tuple(
        PlaquetteTerm(row=r, col=c, coeff=-h)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    
    # Electric terms: g * (2 - 2*(-1)^E) = 4g for E=1, 0 for E=0
    electric_terms = build_electric_terms(shape, coeff=g, N=2)
    
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
    g: float | None = None,
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
    size: int = 8,
    g: float = 0.33,
    h: float = 1.0,
    bond_dim: int = 2,
    n_samples: int = 2000,
    n_steps: int = 500,
    dt: float = 0.01,
    output_dir: str | None = None,
):
    """Run Z2 pure gauge ground state optimization.
    
    Args:
        size: Lattice size (size x size). Paper uses up to 32×32.
        g: Electric field coupling (phase transition at g_c ≈ 0.3285).
        h: Magnetic (plaquette) coupling.
        bond_dim: Bond dimension per charge sector (D_k). Paper uses D_k=2.
        n_samples: MC samples per step. Paper uses ~10^4.
        n_steps: Optimization steps.
        dt: Imaginary time step.
        output_dir: Directory to save output data (JSON). None to skip saving.
    """
    shape = (size, size)
    
    logger.info("=" * 60)
    logger.info("Pure Z2 Lattice Gauge Theory")
    logger.info("=" * 60)
    logger.info(f"Lattice: {size}x{size}, h={h}, g={g}")
    logger.info(f"Bond dimension per charge: D_k={bond_dim} (total D={2*bond_dim})")
    logger.info(f"Phase: {'Deconfined' if g < 0.3285 else 'Near critical' if g < 0.35 else 'Confined'}")
    
    # Pure gauge: phys_dim=1 (no matter), single charge sector
    cfg = GIPEPSConfig(
        shape=shape,
        N=2,
        phys_dim=1,
        Qx=0,
        degeneracy_per_charge=(bond_dim, bond_dim),
        charge_of_site=(0,),
    )
    
    model = GIPEPS(
        rngs=nnx.Rngs(42),
        config=cfg,
        contraction_strategy=ZipUp(truncate_bond_dimension=3 * bond_dim),
    )
    
    operator = build_z2_hamiltonian(shape, h=h, g=g)
    
    logger.info(f"Number of plaquettes: {(size-1)**2}")
    logger.info(f"Number of links: {2*size*(size-1)}")
    
    # Setup data logging
    data = SimulationData(
        model_type="Z2_pure_gauge",
        lattice_size=shape,
        parameters={"h": h, "g": g, "bond_dim": bond_dim, "n_samples": n_samples},
    )
    
    run_optimization(model, operator, n_samples=n_samples, n_steps=n_steps, dt=dt, data=data, g=g)
    
    # Save data if output_dir specified
    if output_dir:
        path = Path(output_dir) / f"z2_pure_L{size}_g{g:.3f}.json"
        data.save(path)
    
    return data


def scan_g(
    size: int = 8,
    g_values: tuple[float, ...] = (0.1, 0.2, 0.3, 0.33, 0.35, 0.4, 0.5),
    bond_dim: int = 2,
    n_samples: int = 2000,
    n_steps: int = 500,
    dt: float = 0.01,
    output_dir: str = "output/z2_scan",
):
    """Scan over g values to map out phase diagram (replicates Fig 2 style).
    
    This produces E(g) data that can be used to compute:
    - dE/dg = (1/g)⟨H_E⟩  
    - d²E/dg² via finite difference
    
    Args:
        size: Lattice size.
        g_values: Values of g to scan.
        bond_dim: Bond dimension per charge.
        n_samples: MC samples per step.
        n_steps: Optimization steps.
        dt: Imaginary time step.
        output_dir: Directory to save output data.
    """
    logger.info("=" * 60)
    logger.info("Z2 Phase Diagram Scan")
    logger.info(f"Lattice: {size}x{size}, D_k={bond_dim}")
    logger.info(f"g values: {g_values}")
    logger.info("=" * 60)
    
    results = []
    for g in g_values:
        logger.info(f"\n{'='*40}\nRunning g = {g}\n{'='*40}")
        data = main(
            size=size, g=g, bond_dim=bond_dim,
            n_samples=n_samples, n_steps=n_steps, dt=dt,
            output_dir=output_dir,
        )
        results.append(data)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCAN SUMMARY")
    logger.info("=" * 60)
    for data in results:
        g = data.parameters["g"]
        e = data.energies[-1] if data.energies else float('nan')
        logger.info(f"g = {g:.3f}: E = {e:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

"""Ground state of odd-Z2 lattice gauge theory.

The odd-Z2 gauge theory has background charge Q_x = 1 at every site,
which is relevant for understanding spin liquids and dimer models.

By varying g, it experiences a continuous transition between:
- Deconfined phase (g < g_c): Uniform plaquette expectation values
- Confined phase (g > g_c): Translation symmetry breaking (VBS order)

Critical point: g_c ≈ 0.64

The dual model is the fully frustrated transverse field Ising model.

To replicate Fig 3:
- (a,b): Plaquette ⟨P_x⟩ at each site showing uniform vs VBS pattern
- (c): VBS order parameters D_x, D_y vs g
- (d): Wilson loop slope σ vs g

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


def build_odd_z2_hamiltonian(
    shape: tuple[int, int],
    h: float = 1.0,
    g: float = 0.5,
) -> GILocalHamiltonian:
    """Build odd-Z2 LGT Hamiltonian (Q_x = 1 everywhere)."""
    n_rows, n_cols = shape
    
    plaquette_terms = tuple(
        PlaquetteTerm(row=r, col=c, coeff=-h)
        for r in range(n_rows - 1)
        for c in range(n_cols - 1)
    )
    
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

    k = 5
    n_chunks = n_steps // k
    assert n_steps == n_chunks * k, (
        f"n_steps={n_steps} must be a multiple of chunk size k={k}"
    )
    for chunk in range(n_chunks):
        driver.run(k * dt)
        completed_steps = (chunk + 1) * k
        if driver.energy is not None:
            e = driver.energy
            logger.info(format_step_log(
                step=completed_steps,
                energy=e.mean.real,
                energy_error=e.error_of_mean.real,
                energy_variance=e.variance.real,
            ))
            if data is not None:
                data.add_step(
                    step=completed_steps,
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
    g: float = 0.5,
    h: float = 1.0,
    bond_dim: int = 2,
    n_samples: int = 2000,
    n_steps: int = 500,
    dt: float = 0.01,
    output_dir: str | None = None,
):
    """Run odd-Z2 gauge ground state optimization.
    
    Args:
        size: Lattice size (size x size). Paper uses up to 32×32.
        g: Electric field coupling (phase transition at g_c ≈ 0.64).
        h: Magnetic (plaquette) coupling.
        bond_dim: Bond dimension per charge sector (D_k). Paper uses D_k=2.
        n_samples: MC samples per step. Paper uses ~10^4.
        n_steps: Optimization steps.
        dt: Imaginary time step.
        output_dir: Directory to save output data (JSON). None to skip saving.
    """
    shape = (size, size)
    
    logger.info("=" * 60)
    logger.info("Odd-Z2 Lattice Gauge Theory (Q_x = 1)")
    logger.info("=" * 60)
    logger.info(f"Lattice: {size}x{size}, h={h}, g={g}")
    logger.info(f"Bond dimension per charge: D_k={bond_dim} (total D={2*bond_dim})")
    logger.info(f"Phase: {'Deconfined' if g < 0.64 else 'Confined (VBS)'}")
    logger.info("Dual model: Fully frustrated transverse field Ising model")
    
    # Odd-Z2: Q_x = 1 at every site
    cfg = GIPEPSConfig(
        shape=shape,
        N=2,
        phys_dim=1,
        Qx=1,  # Background charge = 1
        degeneracy_per_charge=(bond_dim, bond_dim),
        charge_of_site=(0,),
    )
    
    model = GIPEPS(
        rngs=nnx.Rngs(42),
        config=cfg,
        contraction_strategy=ZipUp(truncate_bond_dimension=3 * bond_dim),
    )
    
    operator = build_odd_z2_hamiltonian(shape, h=h, g=g)
    
    # Setup data logging
    data = SimulationData(
        model_type="odd_Z2_gauge",
        lattice_size=shape,
        parameters={"h": h, "g": g, "bond_dim": bond_dim, "n_samples": n_samples, "Qx": 1},
    )
    
    run_optimization(model, operator, n_samples=n_samples, n_steps=n_steps, dt=dt, data=data)
    
    # Save data if output_dir specified
    if output_dir:
        path = Path(output_dir) / f"odd_z2_L{size}_g{g:.3f}.json"
        data.save(path)
    
    return data


def scan_g(
    size: int = 8,
    g_values: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.64, 0.7, 0.8, 0.9),
    bond_dim: int = 2,
    n_samples: int = 2000,
    n_steps: int = 500,
    dt: float = 0.01,
    output_dir: str = "output/odd_z2_scan",
):
    """Scan over g values to detect VBS transition (replicates Fig 3c).
    
    This produces data for computing:
    - VBS order parameters D_x, D_y
    - Plaquette maps showing uniform (g<g_c) vs VBS pattern (g>g_c)
    
    The continuous transition should show VBS order emerging at g_c ≈ 0.64.
    """
    logger.info("=" * 60)
    logger.info("Odd-Z2 Phase Diagram Scan (VBS Transition)")
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
        phase = "Deconfined" if g < 0.64 else "Confined (VBS)"
        logger.info(f"g = {g:.3f}: E = {e:.6f} [{phase}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

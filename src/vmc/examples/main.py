from __future__ import annotations

from vmc import config  # noqa: F401 - JAX config must be imported first

import functools
import logging

import jax
import netket as nk
from flax import nnx

from vmc.drivers import DynamicsDriver, ImaginaryTimeUnit
from vmc.examples.real_time import build_heisenberg_square
from vmc.gauge import GaugeConfig
from vmc.models.mps import MPS
from vmc.preconditioners import SRPreconditioner
from vmc.samplers.sequential import sequential_sample_with_gradients

__all__ = [
    "MPS",
    "build_heisenberg_square",
    "main",
]

logger = logging.getLogger(__name__)


def main():
    length = 3
    n_sites = length * length
    mps_bond = 12
    n_samples = 512
    n_steps = 100
    dt = 3e-2

    hi, H, _ = build_heisenberg_square(length, pbc=False)

    exact_e = nk.exact.lanczos_ed(H, k=1)[0].real
    logger.info("=" * 60)
    logger.info(
        "Exact ground-state energy (Heisenberg %dx%d): %.6f",
        length,
        length,
        exact_e,
    )

    model = MPS(rngs=nnx.Rngs(6), n_sites=n_sites, bond_dim=mps_bond)
    sampler = functools.partial(
        sequential_sample_with_gradients,
        n_samples=n_samples,
        burn_in=8,
        full_gradient=False,
    )
    preconditioner = SRPreconditioner(
        diag_shift=1e-8,
        gauge_config=GaugeConfig(),
    )
    driver = DynamicsDriver(
        model,
        H,
        sampler=sampler,
        preconditioner=preconditioner,
        dt=dt,
        time_unit=ImaginaryTimeUnit(),
        sampler_key=jax.random.key(0),
    )

    logger.info("=" * 60)
    logger.info("MPS + SR (imaginary time)")
    logger.info("=" * 60)
    logger.info("samples=%d", n_samples)
    for step in range(n_steps):
        driver.step()
        stats = driver.energy
        if step % 10 == 0 and stats is not None:
            logger.info("step=%d energy=%.6f", step, float(stats.mean.real))

    if driver.energy is not None:
        logger.info("final energy: %.6f", float(driver.energy.mean.real))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

"""MinSR demo and verification using QGT with sliced Jacobian."""
from __future__ import annotations

from vmc import config  # noqa: F401

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from vmc.core import _value_and_grad
from vmc.models.mps import MPS
from vmc.models.peps import PEPS, ZipUp
from vmc.qgt import QGT, Jacobian, SlicedJacobian, SliceOrdering, SiteOrdering
from vmc.samplers.sequential import sequential_sample
from vmc.utils.smallo import params_per_site
from vmc.utils.vmc_utils import flatten_samples

logger = logging.getLogger(__name__)


def minsr_demo_mps(
    length: int = 8,
    bond_dim: int = 4,
    n_samples: int = 256,
    diag_shift: float = 1e-8,
    seed: int = 0,
):
    """Demo minSR solve with MPS using sliced Jacobian."""
    logger.info("MinSR Demo (MPS): length=%d, bond_dim=%d, n_samples=%d",
                length, bond_dim, n_samples)

    model = MPS(rngs=nnx.Rngs(seed), n_sites=length, bond_dim=bond_dim)
    samples = sequential_sample(
        model,
        n_samples=n_samples,
        key=jax.random.key(seed),
        initial_configuration=model.random_physical_configuration(
            jax.random.key(seed + 1), n_samples=1
        ),
    )

    # Build QGT from sliced Jacobian
    jac = SlicedJacobian.from_samples(model, samples)
    qgt = QGT(jac, space="sample")
    logger.info("QGT matrix shape: %s", qgt.matrix.shape)

    # Solve with random RHS
    dv = jax.random.normal(
        jax.random.key(seed + 1), (n_samples,), dtype=jnp.complex128
    )
    updates, _ = qgt.solve(dv, diag_shift, samples=samples)
    logger.info("Updates shape: %s", updates.shape)

    return {"qgt_shape": qgt.matrix.shape, "updates_shape": updates.shape}


def minsr_demo_peps(
    length: int = 3,
    bond_dim: int = 2,
    n_samples: int = 64,
    diag_shift: float = 1e-8,
    seed: int = 0,
):
    """Demo minSR solve with PEPS using sliced Jacobian."""
    logger.info("MinSR Demo (PEPS): %dx%d, bond_dim=%d, n_samples=%d",
                length, length, bond_dim, n_samples)

    model = PEPS(
        rngs=nnx.Rngs(seed),
        shape=(length, length),
        bond_dim=bond_dim,
        contraction_strategy=ZipUp(bond_dim ** 2),
    )
    samples = sequential_sample(
        model,
        n_samples=n_samples,
        key=jax.random.key(seed),
        initial_configuration=model.random_physical_configuration(
            jax.random.key(seed + 1), n_samples=1
        ),
    )

    # Build QGT with site ordering
    pps = tuple(params_per_site(model))
    jac = SlicedJacobian.from_samples(model, samples, ordering=SiteOrdering(pps))
    qgt = QGT(jac, space="sample")
    logger.info("QGT matrix shape: %s", qgt.matrix.shape)

    # Solve
    dv = jax.random.normal(
        jax.random.key(seed + 1), (n_samples,), dtype=jnp.complex128
    )
    updates, _ = qgt.solve(dv, diag_shift, samples=samples)
    logger.info("Updates shape: %s", updates.shape)

    return {"qgt_shape": qgt.matrix.shape, "updates_shape": updates.shape}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 50)
    minsr_demo_mps()
    logger.info("=" * 50)
    minsr_demo_peps()


if __name__ == "__main__":
    main()

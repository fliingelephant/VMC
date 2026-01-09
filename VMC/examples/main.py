from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import logging

import jax.numpy as jnp
import netket as nk
from flax import nnx

from VMC.drivers.custom_driver import CustomVMC, CustomVMC_SR
from VMC.gauge import GaugeConfig
from VMC.models.mps import SimpleMPS
from VMC.qgt.dense_qgt_operator import MinimalDenseSR

__all__ = [
    "MinimalDenseSR",
    "SimpleMPS",
    "build_heisenberg_square",
    "main",
]

logger = logging.getLogger(__name__)


def build_heisenberg_square(length: int, *, pbc: bool = False):
    graph = nk.graph.Square(length=length, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    H = nk.operator.Heisenberg(hi, graph, dtype=jnp.complex128)
    return hi, H, (length, length)


def main():
    length = 3
    n_sites = length * length
    mps_bond = 12
    n_samples = 4096
    n_iter = 100

    hi, H, _ = build_heisenberg_square(length, pbc=False)
    sampler_kwargs = {"n_chains": 8}

    # Exact ground-state energy for reference (small 3x3 lattice).
    exact_e = nk.exact.lanczos_ed(H, k=1)[0].real
    logger.info("=" * 60)
    logger.info(
        "Exact ground-state energy (Heisenberg %dx%d): %.6f",
        length,
        length,
        exact_e,
    )

    def run_case(name: str, driver):
        logger.info("\n%s", "=" * 60)
        logger.info(name)
        logger.info("=" * 60)
        driver.run(n_iter=n_iter, show_progress=True, out=None)
        energy = driver.state.expect(H).mean
        logger.info("%s final energy: %.6f", name, energy.real)
        if hasattr(driver, "diag_shift_error") or hasattr(driver, "residual_error"):
            diag_shift_error = getattr(driver, "diag_shift_error", None)
            residual_error = getattr(driver, "residual_error", None)
            if diag_shift_error is not None:
                logger.info(
                    "%s final diag_shift_error: %.3e", name, diag_shift_error
                )
            if residual_error is not None:
                logger.info("%s final residual_error: %.3e", name, residual_error)

    optimizer = nk.optimizer.Sgd(learning_rate=3e-2)

    mps_gauge_removed = SimpleMPS(
        rngs=nnx.Rngs(6), n_sites=n_sites, bond_dim=mps_bond
    )
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, **sampler_kwargs),
        mps_gauge_removed,
        n_samples=n_samples,
        n_discard_per_chain=16,
    )
    run_case(
        "MPS + Custom VMC SR (gauge removal)",
        CustomVMC_SR(
            H,
            optimizer,
            variational_state=vstate,
            diag_shift=1e-8,
            use_ntk=False,
            gauge_config=GaugeConfig(),
        ),
    )

    mps_custom_driver_netket_qgt = SimpleMPS(
        rngs=nnx.Rngs(6), n_sites=n_sites, bond_dim=mps_bond
    )
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, **sampler_kwargs),
        mps_custom_driver_netket_qgt,
        n_samples=n_samples,
        n_discard_per_chain=16,
    )
    sr_netket_custom = nk.optimizer.SR(
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
        diag_shift=1e-2,
    )
    run_case(
        "MPS + Custom VMC (NetKet QGT)",
        CustomVMC(
            H,
            optimizer,
            variational_state=vstate,
            preconditioner=sr_netket_custom,
        ),
    )

    mps_custom_driver_custom_qgt = SimpleMPS(
        rngs=nnx.Rngs(6), n_sites=n_sites, bond_dim=mps_bond
    )
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, **sampler_kwargs),
        mps_custom_driver_custom_qgt,
        n_samples=n_samples,
        n_discard_per_chain=16,
    )
    sr_custom_driver = MinimalDenseSR(diag_shift=1e-2)
    run_case(
        "MPS + Custom VMC (custom QGT)",
        CustomVMC(
            H,
            optimizer,
            variational_state=vstate,
            preconditioner=sr_custom_driver,
        ),
    )

    mps_netket_custom = SimpleMPS(
        rngs=nnx.Rngs(6), n_sites=n_sites, bond_dim=mps_bond
    )
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, **sampler_kwargs),
        mps_netket_custom,
        n_samples=n_samples,
        n_discard_per_chain=16,
    )
    sr_custom = MinimalDenseSR(diag_shift=1e-2)
    run_case(
        "MPS + NetKet VMC (custom dense SR)",
        nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr_custom),
    )

    mps_netket_driver = SimpleMPS(
        rngs=nnx.Rngs(6), n_sites=n_sites, bond_dim=mps_bond
    )
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, **sampler_kwargs),
        mps_netket_driver,
        n_samples=n_samples,
        n_discard_per_chain=16,
    )

    sr_netket = nk.optimizer.SR(
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
        diag_shift=1e-2,
    )
    run_case(
        "MPS + NetKet VMC (NetKet SR)",
        nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr_netket),
    )

if __name__ == "__main__":
    main()

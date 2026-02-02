"""Open quantum systems simulation via quantum trajectories.

This module provides tools for simulating open quantum systems with T1 (amplitude
damping) and T2 (dephasing) noise using the quantum trajectory method.

The quantum trajectory method unravels the Lindblad master equation into an
ensemble of stochastic pure-state evolutions:
    1. No-jump evolution with H_eff = H - (i/2) sum_k gamma_k L_k^dag L_k
    2. Stochastic jumps |psi> -> L_k|psi> with probability dp_k = gamma_k dt <L_k^dag L_k>

Example:
    >>> from vmc.experimental.open_systems import QuantumTrajectoryDriver, JumpOperator
    >>> # Create jump operators for all sites
    >>> n_rows, n_cols = 4, 4
    >>> T1, T_phi = 10.0, 20.0
    >>> jumps = [JumpOperator.t1(r, c, T1) for r in range(n_rows) for c in range(n_cols)]
    >>> jumps += [JumpOperator.dephasing(r, c, T_phi) for r in range(n_rows) for c in range(n_cols)]
    >>> # Create driver
    >>> driver = QuantumTrajectoryDriver(
    ...     model=peps,
    ...     hamiltonian=H,
    ...     jump_operators=jumps,
    ...     sampler=sampler,
    ...     preconditioner=preconditioner,
    ...     dt=0.01,
    ... )
    >>> driver.step()  # Single trajectory step
"""

from vmc.experimental.open_systems.jump_operators import (
    JumpOperator,
    SIGMA_MINUS,
    SIGMA_Z,
)
from vmc.experimental.open_systems.trajectory_driver import QuantumTrajectoryDriver

__all__ = [
    "QuantumTrajectoryDriver",
    "JumpOperator",
    "SIGMA_MINUS",
    "SIGMA_Z",
]

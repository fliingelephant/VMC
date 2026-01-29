[![codecov](https://codecov.io/gh/fliingelephant/VMC/graph/badge.svg?token=MONPD2TS5Y)](https://codecov.io/gh/fliingelephant/VMC)

# VMC

> **Note:** This project is under active development and not ready for use.

Variational Monte Carlo dynamics for tensor-network states (MPS/PEPS), implemented with [JAX] primitives and [NetKet] interfaces.

## Features (for tensor-network states)

- Sequential sampling [^1]
- Efficient energy and gradient evaluation with environment reuse
- Gauge-invariant PEPS for finite Abelian lattice gauge theory ($Z_N$ groups) [^3]
- Sliced gradients for memory-efficient stochastic reconfiguration
- Gauge removal for improved numerical stability [^2]

[^1]: W.-Y. Liu, Y.-Z. Huang, S.-S. Gong, and Z.-C. Gu. **Accurate Simulation for Finite Projected Entangled Pair States in Two Dimensions**. *Physical Review B* 103(23):235155, 2021. https://doi.org/10.1103/PhysRevB.103.235155

[^2]: Y. Wu. **Real-Time Dynamics in Two Dimensions with Tensor Network States via Time-Dependent Variational Monte Carlo**. arXiv:2512.06768, 2025. https://doi.org/10.48550/arXiv.2512.06768

[^3]: Y. Wu and W.-Y. Liu. **Accurate Gauge-Invariant Tensor-Network Simulations for Abelian Lattice Gauge Theory in (2+1)D: Ground-State and Real-Time Dynamics**. *Physical Review Letters* 135(13):130401, 2025. https://doi.org/10.1103/3m3j-ds18

[NetKet]: https://github.com/netket/netket
[JAX]: https://github.com/jax-ml/jax

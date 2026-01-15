## VMC inventory

### Configuration
- `VMC/config.py`: JAX x64 setup and logging config (VMC_LOG_LEVEL).

### Models
- `VMC/models/mps.py`: `MPS` open-boundary MPS log-amplitude model.
- `VMC/models/peps.py`: `PEPS` open-boundary PEPS with boundary-MPS contraction.

### Contraction + Amplitudes (PEPS)
- `VMC/models/peps.py`: `ContractionStrategy` ABC with `NoTruncation`, `ZipUp`, `DensityMatrix`.
- `VMC/models/peps.py`: Custom-VJP amplitude pipeline and environment-gradient helpers.

### Drivers
- `VMC/drivers/custom_driver.py`: `Integrator` ABC (`Euler`, `RK4`), `PropagationType` ABC (`RealTime`, `ImaginaryTime`).
- `VMC/drivers/custom_driver.py`: `CustomVMC`, `CustomVMC_SR`, `CustomVMC_QR`, `CustomTDVP_SR`.

### QGT Module
- `VMC/qgt/jacobian.py`: `Jacobian`, `SlicedJacobian`, `PhysicalOrdering`, `SiteOrdering`.
- `VMC/qgt/qgt.py`: `QGT`, `DiagonalQGT`, `ParameterSpace`, `SampleSpace` with lazy matvec via plum dispatch.
- `VMC/qgt/solvers.py`: `solve_cg`, `solve_cholesky`, `solve_svd`.
- `VMC/qgt/netket_compat.py`: `QGTOperator`, `DenseSR` (NetKet LinearOperator adapter).

### Preconditioners
- `VMC/preconditioners/preconditioners.py`: `DirectSolve`, `QRSolve`, `DiagonalSolve`, `SRPreconditioner`.

### Gauge
- `VMC/gauge/gauge.py`: `GaugeConfig`, `compute_gauge_projection` for MPS.

### Utilities
- `VMC/utils/vmc_utils.py`: `flatten_samples`, `get_apply_fun`, `build_dense_jac`.

## QGT Class Diagram

```mermaid
classDiagram
    class Jacobian {
        O: Array
    }
    class SlicedJacobian {
        o: Array
        p: Array
        phys_dim: int
        ordering: Ordering
    }
    class PhysicalOrdering
    class SiteOrdering {
        params_per_site: tuple
    }

    class ParameterSpace
    class SampleSpace

    class QGT {
        jac: Jacobian | SlicedJacobian
        space: ParameterSpace | SampleSpace
        __matmul__(v) Array
        to_dense() Array
    }

    class DiagonalQGT {
        jac: Jacobian | SlicedJacobian
        space: ParameterSpace | SampleSpace
        params_per_site: tuple | None
        __matmul__(v) Array
        to_dense() Array
    }

    class QGTOperator {
        _qgt: QGT
        diag_shift: float
        __matmul__(v)
        _solve(solve_fun, y)
        to_dense()
    }
    class DenseSR {
        diag_shift: float
        lhs_constructor(vstate) QGTOperator
    }

    SlicedJacobian --> PhysicalOrdering
    SlicedJacobian --> SiteOrdering
    QGT --> Jacobian
    QGT --> SlicedJacobian
    QGT --> ParameterSpace
    QGT --> SampleSpace
    DiagonalQGT --> Jacobian
    DiagonalQGT --> SlicedJacobian
    DiagonalQGT --> ParameterSpace
    QGTOperator --> QGT : wraps
    DenseSR --> QGTOperator : creates
    QGTOperator --|> LinearOperator
```

## Preconditioner Class Diagram

```mermaid
classDiagram
    class DirectSolve {
        solver: LinearSolver
    }
    class QRSolve {
        rcond: float | None
        min_norm: bool
    }
    class DiagonalSolve {
        solver: LinearSolver
        params_per_site: tuple | None
    }

    class SRPreconditioner {
        space: ParameterSpace | SampleSpace
        strategy: DirectSolve | QRSolve | DiagonalSolve
        diag_shift: float
        gauge_config: GaugeConfig | None
        apply(state, local_energies) dict
    }

    SRPreconditioner --> DirectSolve
    SRPreconditioner --> QRSolve
    SRPreconditioner --> DiagonalSolve
    SRPreconditioner --> ParameterSpace
    SRPreconditioner --> SampleSpace
```

## Flowchart

```mermaid
flowchart TD
    subgraph Models
        MPS["MPS"]
        PEPS["PEPS"]
    end

    subgraph QGT["QGT Module"]
        Jac["Jacobian<br/>SlicedJacobian"]
        QGTCore["QGT<br/>DiagonalQGT"]
        Space["ParameterSpace<br/>SampleSpace"]
        Solvers["solve_cg<br/>solve_cholesky<br/>solve_svd"]
        NetKet["QGTOperator<br/>DenseSR"]
    end

    subgraph Preconditioners
        SRP["SRPreconditioner"]
        Strategy["DirectSolve<br/>QRSolve<br/>DiagonalSolve"]
    end

    subgraph Drivers
        CustomVMC["CustomVMC"]
        CustomSR["CustomVMC_SR"]
        TDVP["CustomTDVP_SR"]
    end

    Models --> Drivers
    Jac --> QGTCore
    Space --> QGTCore
    QGTCore --> Solvers
    QGTCore --> NetKet
    Solvers --> SRP
    Strategy --> SRP
    SRP --> Drivers
    NetKet --> Drivers
```

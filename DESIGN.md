# VMC Design Overview

## Package Layout (src)
- `src/vmc/config.py`: JAX x64 setup and logging config (`VMC_LOG_LEVEL`).
- `src/vmc/core/eval.py`: Unified eval API (`_value`, `_grad`, `_value_and_grad`) with plum dispatch.
- `src/vmc/models/mps.py`: `MPS` open-boundary model; `MPS.site_dims` for boundary bond sizes.
- `src/vmc/models/peps.py`: `PEPS` open-boundary model; `PEPS.site_dims`, contraction strategies, VJP.
- `src/vmc/samplers/sequential.py`: Sequential Metropolis samplers for MPS/PEPS.
- `src/vmc/qgt/*`: QGT, Jacobians, solvers, NetKet compatibility.
- `src/vmc/preconditioners/preconditioners.py`: SR/QGT preconditioners and solvers.
- `src/vmc/drivers/custom_driver.py`: time evolution drivers and integrators.
- `src/vmc/utils/vmc_utils.py`: Jacobian helpers, batching, local energy.
- `src/vmc/utils/smallo.py`: `params_per_site` (uses model static site dims).

## Core Eval API
- `_value(model, sample)`: amplitude; auto-vmaps for 2D samples.
- `_grad(model, sample)`: gradient; wraps `_value_and_grad`.
- `_value_and_grad(model, sample)`: amplitude + gradient; base entrypoint.

## Models
- `MPS`: site tensors `(phys_dim, D_left, D_right)` with open boundaries.
- `PEPS`: site tensors `(phys_dim, up, down, left, right)` with open boundaries.
- `MPS.site_dims` / `PEPS.site_dims`: boundary-aware bond sizes used by eval/sampling.

## PEPS Contraction + Gradients
- `ContractionStrategy` ABC with `NoTruncation`, `ZipUp`, `DensityMatrix`.
- `PEPS.apply`: static method with custom-VJP using cached environments.
- `_compute_all_gradients`: environment-based row gradients.

## Sampling
- `sequential_sample`: sequential Metropolis for MPS/PEPS.
- `sequential_sample_with_gradients`: sampling + gradient collection.
- `peps_sequential_sweep`: single PEPS sweep utility.
- Stateful sampling threads `(final_configurations, key)` for reproducible chains.

## QGT
- `Jacobian` / `SlicedJacobian` with `PhysicalOrdering` / `SiteOrdering`.
- `QGT`: lazy matvec in parameter or sample space.
- `QGTOperator` / `DenseSR`: NetKet compatibility wrappers.

## Class Diagrams

### QGT
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
    QGTOperator --> QGT : wraps
    DenseSR --> QGTOperator : creates
    QGTOperator --|> LinearOperator
```

### Preconditioners
```mermaid
classDiagram
    class DirectSolve {
        solver: LinearSolver
    }
    class QRSolve {
        rcond: float | None
        min_norm: bool
    }

    class SRPreconditioner {
        space: ParameterSpace | SampleSpace
        strategy: DirectSolve | QRSolve
        diag_shift: float
        gauge_config: GaugeConfig | None
        ordering: PhysicalOrdering | SiteOrdering
        apply(model, samples, o, p, local_energies) dict
    }

    SRPreconditioner --> DirectSolve
    SRPreconditioner --> QRSolve
    SRPreconditioner --> ParameterSpace
    SRPreconditioner --> SampleSpace
```

### Drivers
```mermaid
classDiagram
    class TimeUnit {
        <<abstract>>
        grad_factor: complex
        default_integrator()
    }
    class RealTimeUnit
    class ImaginaryTimeUnit

    class Integrator {
        <<abstract>>
        step(driver, params, t, dt)
    }
    class Euler
    class RK4

    class DynamicsDriver {
        model
        operator
        sampler
        preconditioner
        time_unit
        integrator
        dt: float
        t: float
        _sampler_configuration
        step()
        run(T)
    }

    TimeUnit <|-- RealTimeUnit
    TimeUnit <|-- ImaginaryTimeUnit
    Integrator <|-- Euler
    Integrator <|-- RK4
    DynamicsDriver --> TimeUnit
    DynamicsDriver --> Integrator
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
        QGTCore["QGT"]
        Space["ParameterSpace<br/>SampleSpace"]
        Solvers["solve_cg<br/>solve_cholesky<br/>solve_svd"]
        NetKet["QGTOperator<br/>DenseSR"]
    end

    subgraph Preconditioners
        SRP["SRPreconditioner"]
        Strategy["DirectSolve<br/>QRSolve"]
    end

    subgraph Drivers
        Dynamics["DynamicsDriver"]
    end
    subgraph Samplers
        Seq["Sequential Samplers"]
    end

    Models --> Drivers
    Samplers --> Drivers
    Jac --> QGTCore
    Space --> QGTCore
    QGTCore --> Solvers
    QGTCore --> NetKet
    Solvers --> SRP
    Strategy --> SRP
    SRP --> Drivers
    NetKet --> Drivers
```

"""Local operator definitions."""
from __future__ import annotations

from vmc.operators.local_terms import (
    BucketedOperators,
    CoefficientStructure,
    LocalHamiltonian,
    DiagonalOperator,
    HorizontalTwoSiteOperator,
    Operator,
    OneSiteOperator,
    PlaquetteOperator,
    TransitionOperator,
    VerticalTwoSiteOperator,
    merge_operators,
    support_span,
)
from vmc.operators.time_dependent import (
    AffineSchedule,
    TermCoefficientSchedule,
    TimeDependentHamiltonian,
    coeffs_at,
    operator_schedule,
)

__all__ = [
    "BucketedOperators",
    "CoefficientStructure",
    "LocalHamiltonian",
    "DiagonalOperator",
    "HorizontalTwoSiteOperator",
    "Operator",
    "OneSiteOperator",
    "PlaquetteOperator",
    "TransitionOperator",
    "VerticalTwoSiteOperator",
    "merge_operators",
    "support_span",
    "TermCoefficientSchedule",
    "AffineSchedule",
    "TimeDependentHamiltonian",
    "coeffs_at",
    "operator_schedule",
]

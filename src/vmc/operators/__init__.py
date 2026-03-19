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
    CubicSchedule,
    TermCoefficientSchedule,
    TimeDependentHamiltonian,
    coeffs_at,
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
    "CubicSchedule",
    "TimeDependentHamiltonian",
    "coeffs_at",
]

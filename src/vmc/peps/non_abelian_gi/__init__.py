"""Generic sampled-block non-Abelian gauge-invariant PEPS machinery."""

from __future__ import annotations

from vmc.peps.non_abelian_gi.builders import build_pure_gauge_tables
from vmc.peps.non_abelian_gi.contraction import build_row_mpo, non_abelian_gi_apply
from vmc.peps.non_abelian_gi.factors import (
    HoppingFactorTables,
    PlaquetteFactorTables,
    build_hopping_factor_tables,
    build_plaquette_factor_tables,
)
from vmc.peps.non_abelian_gi.kernels import build_mc_kernels
from vmc.peps.non_abelian_gi.local_terms import (
    FermionicHorizontalMatterHoppingTerm,
    FermionicVerticalMatterHoppingTerm,
    HorizontalLinkCasimirTerm,
    HorizontalMatterHoppingTerm,
    MatterNumberTerm,
    PlaquetteTerm,
    VerticalLinkCasimirTerm,
    VerticalMatterHoppingTerm,
    build_link_casimir_terms,
    build_matter_number_terms,
    casimir_diagonal,
    link_casimir_energy,
    matter_number_energy,
)
from vmc.peps.non_abelian_gi.model import NonAbelianGIPEPS, NonAbelianGIPEPSConfig
from vmc.peps.non_abelian_gi.tables import PureGaugeTables

__all__ = [
    "FermionicHorizontalMatterHoppingTerm",
    "FermionicVerticalMatterHoppingTerm",
    "HoppingFactorTables",
    "HorizontalLinkCasimirTerm",
    "HorizontalMatterHoppingTerm",
    "MatterNumberTerm",
    "NonAbelianGIPEPS",
    "NonAbelianGIPEPSConfig",
    "PlaquetteFactorTables",
    "PlaquetteTerm",
    "PureGaugeTables",
    "VerticalLinkCasimirTerm",
    "VerticalMatterHoppingTerm",
    "build_hopping_factor_tables",
    "build_link_casimir_terms",
    "build_matter_number_terms",
    "build_mc_kernels",
    "build_plaquette_factor_tables",
    "build_pure_gauge_tables",
    "build_row_mpo",
    "casimir_diagonal",
    "link_casimir_energy",
    "matter_number_energy",
    "non_abelian_gi_apply",
]

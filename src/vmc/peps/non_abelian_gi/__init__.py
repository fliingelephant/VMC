"""Generic sampled-block non-Abelian gauge-invariant PEPS machinery."""
from __future__ import annotations

from vmc.peps.non_abelian_gi.builders import (
    build_horizontal_hopping_matrix_table,
    build_horizontal_hopping_matrix_tables,
    build_plaquette_link_transitions,
    build_plaquette_matrix_table,
    build_plaquette_matrix_tables,
    build_pure_gauge_tables,
    build_vertical_hopping_matrix_table,
    build_vertical_hopping_matrix_tables,
)
from vmc.peps.non_abelian_gi.contraction import build_row_mpo, non_abelian_gi_apply
from vmc.peps.non_abelian_gi.kernels import build_mc_kernels
from vmc.peps.non_abelian_gi.local_terms import (
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
from vmc.peps.non_abelian_gi.tables import (
    HoppingMatrixTable,
    PlaquetteLinkTransitions,
    PlaquetteMatrixTable,
    PureGaugeTables,
)

__all__ = [
    "HoppingMatrixTable",
    "HorizontalLinkCasimirTerm",
    "HorizontalMatterHoppingTerm",
    "MatterNumberTerm",
    "NonAbelianGIPEPS",
    "NonAbelianGIPEPSConfig",
    "PlaquetteLinkTransitions",
    "PlaquetteMatrixTable",
    "PlaquetteTerm",
    "PureGaugeTables",
    "VerticalLinkCasimirTerm",
    "VerticalMatterHoppingTerm",
    "build_horizontal_hopping_matrix_table",
    "build_horizontal_hopping_matrix_tables",
    "build_link_casimir_terms",
    "build_matter_number_terms",
    "build_mc_kernels",
    "build_plaquette_link_transitions",
    "build_plaquette_matrix_table",
    "build_plaquette_matrix_tables",
    "build_pure_gauge_tables",
    "build_vertical_hopping_matrix_table",
    "build_vertical_hopping_matrix_tables",
    "build_row_mpo",
    "casimir_diagonal",
    "link_casimir_energy",
    "matter_number_energy",
    "non_abelian_gi_apply",
]

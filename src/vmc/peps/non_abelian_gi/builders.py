"""Typed static-metadata builders for non-Abelian GI-PEPS backends."""
from __future__ import annotations

from typing import Any

from plum import dispatch


@dispatch
def build_pure_gauge_tables(
    group: object,
    *,
    shape: tuple[int, int],
    target_charge: Any = 0,
    matter_irreps: tuple[int, ...] = (0,),
    matter_numbers: tuple[int, ...] = (0,),
) -> object:
    del shape, target_charge, matter_irreps, matter_numbers
    raise NotImplementedError(f"No pure-gauge table builder registered for {type(group)!r}.")


@dispatch
def build_plaquette_link_transitions(group: object) -> object:
    raise NotImplementedError(
        f"No plaquette link-transition builder registered for {type(group)!r}."
    )


@dispatch
def build_plaquette_matrix_table(
    group: object,
    tables: object,
    *,
    row: int,
    col: int,
) -> object:
    del tables, row, col
    raise NotImplementedError(
        f"No plaquette matrix-table builder registered for {type(group)!r}."
    )


@dispatch
def build_plaquette_matrix_tables(group: object, tables: object) -> object:
    del tables
    raise NotImplementedError(
        f"No plaquette matrix-table builder registered for {type(group)!r}."
    )


@dispatch
def build_horizontal_hopping_matrix_table(
    group: object,
    tables: object,
    *,
    row: int,
    col: int,
) -> object:
    del tables, row, col
    raise NotImplementedError(
        f"No horizontal hopping matrix-table builder registered for {type(group)!r}."
    )


@dispatch
def build_horizontal_hopping_matrix_tables(group: object, tables: object) -> object:
    del tables
    raise NotImplementedError(
        f"No horizontal hopping matrix-table builder registered for {type(group)!r}."
    )


@dispatch
def build_vertical_hopping_matrix_table(
    group: object,
    tables: object,
    *,
    row: int,
    col: int,
) -> object:
    del tables, row, col
    raise NotImplementedError(
        f"No vertical hopping matrix-table builder registered for {type(group)!r}."
    )


@dispatch
def build_vertical_hopping_matrix_tables(group: object, tables: object) -> object:
    del tables
    raise NotImplementedError(
        f"No vertical hopping matrix-table builder registered for {type(group)!r}."
    )

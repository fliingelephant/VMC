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
    raise NotImplementedError(
        f"No pure-gauge table builder registered for {type(group)!r}."
    )

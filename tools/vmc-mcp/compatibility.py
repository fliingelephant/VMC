"""Compatibility matrix and feasibility checker for the PEPS-tVMC codebase."""

from __future__ import annotations

# Which term types are valid with which model family
TERM_MODEL_COMPAT = {
    "PEPS": {
        "OneSiteOperator",
        "DiagonalOperator",
        "HorizontalTwoSiteOperator",
        "VerticalTwoSiteOperator",
        "PlaquetteOperator",
    },
    "GIPEPS": {
        "PlaquetteOperator",
        "LinkDiagonalTerm",
        "MatterMassTerm",
        "HorizontalMatterHoppingTerm",
        "VerticalMatterHoppingTerm",
        "HorizontalHiggsLinkTerm",
        "VerticalHiggsLinkTerm",
    },
    "BlockadePEPS": {
        "OneSiteOperator",
        "DiagonalOperator",
        "HorizontalTwoSiteOperator",
        "VerticalTwoSiteOperator",
        "PlaquetteOperator",
    },
}

# Map user-friendly term names to canonical operator class names
TERM_ALIASES: dict[str, str | tuple[str, ...]] = {
    "electric": "LinkDiagonalTerm",
    "plaquette": "PlaquetteOperator",
    "matter_mass": "MatterMassTerm",
    "hopping": ("HorizontalMatterHoppingTerm", "VerticalMatterHoppingTerm"),
    "higgs": ("HorizontalHiggsLinkTerm", "VerticalHiggsLinkTerm"),
    "onsite": "OneSiteOperator",
    "diagonal": "DiagonalOperator",
    "exchange_h": "HorizontalTwoSiteOperator",
    "exchange_v": "VerticalTwoSiteOperator",
}


def _resolve_terms(term_names: list[str]) -> list[str]:
    """Resolve user-friendly aliases to canonical operator class names."""
    resolved = []
    for name in term_names:
        alias = TERM_ALIASES.get(name)
        if alias is None:
            resolved.append(name)
        elif isinstance(alias, tuple):
            resolved.extend(alias)
        else:
            resolved.append(alias)
    return resolved


def check_compatibility(model: str, term_types: list[str]) -> dict:
    """Check if a model supports the given operator term types."""
    supported = TERM_MODEL_COMPAT.get(model)
    if supported is None:
        return {
            "compatible": False,
            "reason": f"Unknown model {model!r}. Known: {list(TERM_MODEL_COMPAT)}",
            "incompatible_terms": term_types,
        }
    incompatible = [t for t in term_types if t not in supported]
    if incompatible:
        return {
            "compatible": False,
            "reason": f"{model} does not support: {incompatible}",
            "incompatible_terms": incompatible,
        }
    return {
        "compatible": True,
        "reason": "All terms supported.",
        "incompatible_terms": [],
    }


def check_feasibility(config: dict) -> dict:
    """Check if a simulation config is feasible with this codebase.

    Config keys:
        N: int | None       Z_N group order (None = no gauge symmetry → PEPS)
        Qx: int             Background charge per site (0 = even, 1 = odd)
        lattice: tuple[int, int]
        terms: list[str]    User-friendly names or operator class names
        dynamics: str | None  ("imaginary_time", "real_time", None)
    """
    N = config.get("N")
    lattice = config.get("lattice", (4, 4))
    terms = config.get("terms", [])
    notes: list[str] = []

    # Determine model family
    if N is not None:
        suggested_model = "GIPEPS"
        Qx = config.get("Qx", 0)
        if Qx != 0:
            notes.append(f"Odd gauge theory (Qx={Qx}): set Qx={Qx} in GIPEPSConfig.")
    else:
        suggested_model = "PEPS"

    # Resolve term aliases
    resolved_terms = _resolve_terms(terms)

    # Check term compatibility
    compat = check_compatibility(suggested_model, resolved_terms)
    if not compat["compatible"]:
        return {
            "feasible": False,
            "suggested_model": suggested_model,
            "reason": compat["reason"],
            "missing_features": compat["incompatible_terms"],
            "notes": notes,
        }

    # Lattice size check for plaquette terms
    if "PlaquetteOperator" in resolved_terms:
        if lattice[0] < 2 or lattice[1] < 2:
            return {
                "feasible": False,
                "suggested_model": suggested_model,
                "reason": "PlaquetteOperator requires lattice >= 2x2.",
                "missing_features": ["lattice too small for plaquette terms"],
                "notes": notes,
            }

    # Higgs terms note
    higgs_terms = {"HorizontalHiggsLinkTerm", "VerticalHiggsLinkTerm"}
    if higgs_terms & set(resolved_terms):
        notes.append(
            "Higgs terms require conserve_particle_number=False in GIPEPSConfig."
        )

    return {
        "feasible": True,
        "suggested_model": suggested_model,
        "reason": "All terms and features are supported.",
        "missing_features": [],
        "notes": notes,
    }

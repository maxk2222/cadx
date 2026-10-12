"""Isotropic linear-elastic material table for FEM verification and TopOpt.

Values are typical bulk properties — printed parts are anisotropic in
reality, so treat results as engineering estimates, not certification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialProps:
    name: str
    elastic_modulus_mpa: float  # Young's modulus E (MPa = N/mm^2)
    poisson: float  # Poisson's ratio nu
    yield_mpa: float  # tensile yield strength (design allowable)
    density_g_cm3: float


_MATERIALS: dict[str, MaterialProps] = {
    "pla": MaterialProps("PLA", 3500.0, 0.36, 50.0, 1.24),
    "petg": MaterialProps("PETG", 2100.0, 0.38, 53.0, 1.27),
    "abs": MaterialProps("ABS", 2300.0, 0.35, 40.0, 1.05),
    "asa": MaterialProps("ASA", 2400.0, 0.36, 42.0, 1.06),
    "pa": MaterialProps("PA (nylon)", 1700.0, 0.40, 45.0, 1.14),
    "pc": MaterialProps("PC", 2400.0, 0.37, 60.0, 1.20),
    "resin": MaterialProps("SLA resin", 2800.0, 0.40, 60.0, 1.10),
    "aluminum": MaterialProps("Aluminum 6061", 68_900.0, 0.33, 276.0, 2.70),
    "steel": MaterialProps("Steel S235", 210_000.0, 0.30, 235.0, 7.85),
}

_ALIASES = {
    "nylon": "pa",
    "pa6": "pa",
    "pa12": "pa",
    "pet-g": "petg",
    "aluminium": "aluminum",
    "alu": "aluminum",
    "6061": "aluminum",
    "aluminium 6061": "aluminum",
    "s235": "steel",
    "stahl": "steel",
    "structural steel": "steel",
}

DEFAULT = _MATERIALS["pla"]


def material_names() -> list[str]:
    return sorted(_MATERIALS)


def resolve_material(
    name: str,
    *,
    e_mpa: float | None = None,
    nu: float | None = None,
    yield_mpa: float | None = None,
) -> MaterialProps:
    """Look up a material by name (falls back to PLA) and apply overrides."""
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    base = _MATERIALS.get(key, DEFAULT)
    if e_mpa is None and nu is None and yield_mpa is None:
        return base
    return MaterialProps(
        name=base.name,
        elastic_modulus_mpa=e_mpa or base.elastic_modulus_mpa,
        poisson=nu if nu is not None else base.poisson,
        yield_mpa=yield_mpa or base.yield_mpa,
        density_g_cm3=base.density_g_cm3,
    )

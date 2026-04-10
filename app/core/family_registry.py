from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import FamilyProfileConstraintError, UnsupportedAntennaFamilyError
from app.core.schemas import OptimizeRequest


@dataclass(frozen=True)
class FamilyProfile:
    family: str
    allowed_materials: tuple[str, ...]
    allowed_substrates: tuple[str, ...]
    default_material: str
    default_substrate: str


_COMMON_CONDUCTORS: tuple[str, ...] = (
    "Copper (annealed)",
    "Aluminum",
    "Silver",
    "Gold",
)

_COMMON_SUBSTRATES: tuple[str, ...] = (
    "FR-4 (lossy)",
    "Rogers RT/duroid 5880",
    "Rogers RO3003",
    "Rogers RO4350B",
)


_FAMILY_REGISTRY: dict[str, FamilyProfile] = {
    "amc_patch": FamilyProfile(
        family="amc_patch",
        allowed_materials=_COMMON_CONDUCTORS,
        allowed_substrates=_COMMON_SUBSTRATES,
        default_material="Copper (annealed)",
        default_substrate="FR-4 (lossy)",
    ),
    "microstrip_patch": FamilyProfile(
        family="microstrip_patch",
        allowed_materials=_COMMON_CONDUCTORS,
        allowed_substrates=_COMMON_SUBSTRATES,
        default_material="Copper (annealed)",
        default_substrate="Rogers RT/duroid 5880",
    ),
    "wban_patch": FamilyProfile(
        family="wban_patch",
        allowed_materials=_COMMON_CONDUCTORS,
        allowed_substrates=_COMMON_SUBSTRATES,
        default_material="Copper (annealed)",
        default_substrate="Rogers RO3003",
    ),
}


def list_supported_families() -> list[str]:
    return sorted(_FAMILY_REGISTRY.keys())


def get_family_profile(family: str) -> FamilyProfile:
    normalized = family.strip().lower()
    if normalized not in _FAMILY_REGISTRY:
        supported = ", ".join(sorted(_FAMILY_REGISTRY.keys()))
        raise UnsupportedAntennaFamilyError(
            f"Unsupported antenna family '{family}'. Supported families: {supported}."
        )
    return _FAMILY_REGISTRY[normalized]


def apply_family_profile(request: OptimizeRequest) -> OptimizeRequest:
    """Return a normalized request with family profile defaults/constraints applied."""
    normalized = request.model_copy(deep=True)
    profile = get_family_profile(normalized.target_spec.antenna_family)
    normalized.target_spec.antenna_family = profile.family

    # Honor caller-provided materials/substrates exactly as supplied and only validate them.
    for material in normalized.design_constraints.allowed_materials:
        if material not in profile.allowed_materials:
            raise FamilyProfileConstraintError(
                f"Material '{material}' is not allowed for family '{profile.family}'."
            )

    for substrate in normalized.design_constraints.allowed_substrates:
        if substrate not in profile.allowed_substrates:
            raise FamilyProfileConstraintError(
                f"Substrate '{substrate}' is not allowed for family '{profile.family}'."
            )

    return normalized

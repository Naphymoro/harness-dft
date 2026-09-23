"""Pseudopotential family selection and validation against a structure."""
from __future__ import annotations

from ase import Atoms

# The full SSSP 1.3 library this harness installs by default: both
# functionals used in the QE/AiiDA ecosystem, at both accuracy protocols.
# `aiida_quantumespresso`'s `get_builder_from_protocol()` hardcodes
# SSSP/1.3/PBEsol/* for its own "fast"/"balanced"/"stringent" protocols --
# passing PBE requires an explicit override (see dft-pseudo-select skill) --
# so both functionals need to actually be installed, not just PBE.
SSSP_FULL_LIBRARY = (
    "SSSP/1.3/PBE/efficiency",
    "SSSP/1.3/PBE/precision",
    "SSSP/1.3/PBEsol/efficiency",
    "SSSP/1.3/PBEsol/precision",
)


class MissingPseudopotentialError(RuntimeError):
    pass


def list_installed_families() -> list[str]:
    """Return the labels of every aiida-pseudo family group actually
    installed in the loaded AiiDA profile (not just the ones this harness
    knows the canonical names for)."""
    from aiida import orm
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    return sorted(group.label for group in orm.QueryBuilder().append(PseudoPotentialFamily).all(flat=True))


def validate_family_covers_structure(atoms: Atoms, family_label: str) -> None:
    """Raise if the named aiida-pseudo family doesn't have a pseudo for every
    element in the structure. Call before building a workchain to fail fast
    with a clear message instead of a QE crash deep in a submitted job."""
    from aiida import orm

    family = orm.load_group(family_label)
    available = set(family.elements)
    missing = sorted(set(atoms.get_chemical_symbols()) - available)
    if missing:
        raise MissingPseudopotentialError(
            f"Pseudopotential family '{family_label}' has no entry for: {', '.join(missing)}"
        )


def recommend_family(functional: str = "PBE", protocol: str = "efficiency") -> str:
    """Return the conventional aiida-pseudo SSSP family label for a given
    functional/protocol combination. Doesn't check installation -- pair with
    `aiida-pseudo list` or `validate_family_covers_structure`."""
    return f"SSSP/1.3/{functional}/{protocol}"

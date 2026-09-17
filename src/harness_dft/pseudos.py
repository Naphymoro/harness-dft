"""Pseudopotential family selection and validation against a structure."""
from __future__ import annotations

from ase import Atoms


class MissingPseudopotentialError(RuntimeError):
    pass


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

"""Read an ASE Atoms object from inline text instead of a host filesystem
path. The MCP server may run on a different machine/filesystem than the
agent sandbox that has the structure file, so every tool that needs a
structure takes its content as a string plus an ASE format name rather than
a path (see `dft-harness` skill for the list of formats agents should use)."""
from __future__ import annotations

import io

from ase import Atoms
from ase.io import read


def read_atoms(structure_text: str, structure_format: str) -> Atoms:
    atoms = read(io.StringIO(structure_text), format=structure_format)
    if isinstance(atoms, list):  # some formats (e.g. multi-frame xyz) return a list
        atoms = atoms[-1]
    return atoms

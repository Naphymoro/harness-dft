"""Structure conversion and electron-counting helpers bridging ASE and AiiDA."""
from __future__ import annotations

from ase import Atoms


def ase_to_structure_data(atoms: Atoms):
    """Wrap an ASE Atoms object as an AiiDA StructureData. Import of aiida.orm
    is local so this module stays importable without a loaded AiiDA profile."""
    from aiida import orm
    return orm.StructureData(ase=atoms)


def count_valence_electrons(atoms: Atoms, pseudo_family) -> float:
    """Sum z_valence over all atoms using an aiida-pseudo family (e.g. an
    SsspFamily loaded via `aiida.orm.load_group('SSSP/1.3/PBE/efficiency')`)."""
    return sum(pseudo_family.get_pseudo(symbol).z_valence for symbol in atoms.get_chemical_symbols())


def is_likely_metal(atoms: Atoms) -> bool:
    """Coarse heuristic only: elemental metals from a fixed set, else assume
    insulator/semiconductor. Real electronic-type detection needs a
    calculation; callers should override this when they know better."""
    metallic_symbols = {
        "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Al", "Ga", "In", "Sn", "Pb", "Y", "Zr", "Nb", "Mo", "Tc",
        "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg",
    }
    symbols = set(atoms.get_chemical_symbols())
    return symbols.issubset(metallic_symbols) and len(symbols) > 0

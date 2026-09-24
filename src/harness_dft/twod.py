"""2D monolayer structure prototypes for elemental single-species screening
(the "Xene" family: graphene/silicene/borophene-type honeycombs, and
black-phosphorus-type puckered pnictogens).

Every generator here produces a *starting guess* for a subsequent DFT
relaxation, not a claimed final structure. Bond lengths default to the sum
of ASE's tabulated covalent radii (`ase.data.covalent_radii`) for the two
bonding atoms -- a standard, transparent rule-of-thumb, not a literature
number copied from memory. The puckered-pnictogen prototype's exact pucker
angle/amplitude is similarly a generic topological seed (not element-specific
literature lattice constants), because seed precision matters far less than
topology: `cell_dofree="2Dxy"` relaxation (see `harness_dft.builders`) will
find the real bond lengths and buckling amplitude for a given element and
functional. Treat every candidate as *unrelaxed* until a real DFT relaxation
has actually run on it.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import graphene
from ase.data import atomic_numbers, covalent_radii


def _pair_bond_length(symbol: str) -> float:
    """Sum of covalent radii for two atoms of `symbol` -- a standard
    nearest-neighbor-distance estimate, not an element-specific literature
    value. Full relaxation refines this."""
    z = atomic_numbers[symbol]
    return 2 * covalent_radii[z]


def is_2d_periodic(atoms: Atoms) -> bool:
    """True if `atoms` is periodic in the first two lattice directions only
    (a slab-with-vacuum convention), matching what every function in this
    module produces."""
    pbc = atoms.pbc
    return bool(pbc[0]) and bool(pbc[1]) and not pbc[2]


def default_2d_kpoints_mesh(in_plane_mesh: tuple[int, int] = (9, 9)) -> tuple[int, int, int]:
    """A 2D structure has no periodicity along the vacuum direction -- the
    third mesh index must always be 1, never sampled. Wrap
    `in_plane_mesh` into the (Nx, Ny, 1) triple every QE input needs, so
    callers don't have to remember this convention by hand."""
    nx, ny = in_plane_mesh
    return (nx, ny, 1)


def build_honeycomb_monolayer(
    symbol: str,
    bond_length: float | None = None,
    vacuum: float = 18.0,
    buckle_seed: float = 0.08,
) -> Atoms:
    """A 2-atom honeycomb monolayer (the graphene/silicene/germanene/
    stanene/borophene-buckled-honeycomb/arsenene-alpha-phase family), seeded
    with a small alternating out-of-plane displacement so a subsequent
    relaxation isn't stuck exactly at the high-symmetry planar saddle point
    for elements whose true minimum is buckled (silicene, germanene,
    arsenene, antimonene, bismuthene, ...). If the true minimum is planar
    (graphene), relaxation just relaxes `buckle_seed` back to ~0.

    `bond_length` defaults to the sum of `symbol`'s covalent radii (nearest-
    neighbor distance); `a = bond_length * sqrt(3)` is the resulting
    honeycomb lattice constant, matching `ase.build.graphene`'s convention.
    """
    if bond_length is None:
        bond_length = _pair_bond_length(symbol)
    lattice_a = bond_length * np.sqrt(3)

    atoms = graphene(formula=f"{symbol}2", a=lattice_a, vacuum=vacuum)
    if buckle_seed:
        z = atoms.positions[:, 2].copy()
        z[0] += buckle_seed / 2
        z[1] -= buckle_seed / 2
        atoms.positions[:, 2] = z
    return atoms


def build_puckered_monolayer(
    symbol: str,
    bond_length: float | None = None,
    vacuum: float = 18.0,
    pucker_amplitude_fraction: float = 0.35,
) -> Atoms:
    """A 4-atom orthorhombic puckered monolayer -- the black-phosphorus/
    phosphorene topology, also the relevant prototype for arsenene/
    antimonene/bismuthene's beta phase: two zigzag sublattices offset along
    the vacuum direction, each atom 3-fold coordinated (2 in-plane-ish
    neighbors within its own zigzag chain, 1 neighbor connecting to the
    other sublattice's chain).

    This is a generic topological seed built from `symbol`'s covalent-radius
    bond length and a fractional pucker amplitude, NOT a copy of any
    element's literature lattice constants -- only the connectivity/symmetry
    (Pmna-type puckered honeycomb) is asserted here; a real relaxation
    determines the actual geometry. `pucker_amplitude_fraction` (of the bond
    length) sets how strongly seeded the initial fold is; the QE default's
    `cell_dofree="2Dxy"` relaxation is free to flatten or deepen it.
    """
    if bond_length is None:
        bond_length = _pair_bond_length(symbol)

    # Zigzag chain geometry: two atoms per chain unit, related by a mirror,
    # separated in-plane by `dx` and out-of-plane by `dz` (the pucker), with
    # `dx**2 + dz**2 == bond_length**2` for the in-chain bond. A second,
    # symmetry-equivalent chain is offset by half the cell along the
    # zigzag-perpendicular in-plane direction and by the full pucker along
    # the vacuum direction, giving the third (inter-chain) bond its own
    # in-plane offset `dy`.
    dz = bond_length * pucker_amplitude_fraction
    dx = np.sqrt(max(bond_length**2 - dz**2, 0.0))
    dy = bond_length  # inter-chain bond, taken in-plane (dz already used for the intra-chain pucker)

    a_lattice = 2 * dx  # periodicity along the zigzag direction
    b_lattice = 2 * dy  # periodicity along the inter-chain direction

    positions = [
        (0.0, 0.0, 0.0),
        (dx, 0.0, dz),
        (dx, dy, dz),
        (0.0, dy, 0.0),
    ]
    cell = [a_lattice, b_lattice, vacuum]
    atoms = Atoms(
        symbols=[symbol] * 4,
        positions=positions,
        cell=cell,
        pbc=(True, True, False),
    )
    atoms.center(vacuum=vacuum / 2, axis=2)
    return atoms

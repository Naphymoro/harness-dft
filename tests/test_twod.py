"""Pure geometry tests -- no AiiDA needed, unlike most of this test suite."""
import numpy as np
import pytest

from harness_dft.twod import (
    build_honeycomb_monolayer,
    build_puckered_monolayer,
    default_2d_kpoints_mesh,
    is_2d_periodic,
)


@pytest.mark.parametrize("symbol", ["Al", "Ga", "In", "Tl", "C", "Si"])
def test_honeycomb_nearest_neighbor_distance_matches_intended_bond_length(symbol):
    atoms = build_honeycomb_monolayer(symbol, buckle_seed=0.0)
    assert is_2d_periodic(atoms)
    assert len(atoms) == 2
    d = atoms.get_distance(0, 1, mic=True)
    from ase.data import atomic_numbers, covalent_radii
    expected_bond = 2 * covalent_radii[atomic_numbers[symbol]]
    assert d == pytest.approx(expected_bond, abs=1e-6)


def test_honeycomb_buckle_seed_breaks_planar_symmetry():
    flat = build_honeycomb_monolayer("Si", buckle_seed=0.0)
    buckled = build_honeycomb_monolayer("Si", buckle_seed=0.1)
    assert flat.positions[0, 2] == pytest.approx(flat.positions[1, 2])
    assert buckled.positions[0, 2] != pytest.approx(buckled.positions[1, 2])


@pytest.mark.parametrize("symbol", ["P", "As", "Sb", "Bi"])
def test_puckered_monolayer_has_threefold_coordination_at_intended_bond_length(symbol):
    from ase.data import atomic_numbers, covalent_radii

    atoms = build_puckered_monolayer(symbol)
    assert is_2d_periodic(atoms)
    assert len(atoms) == 4
    expected_bond = 2 * covalent_radii[atomic_numbers[symbol]]

    supercell = atoms.repeat((2, 2, 1))
    for i in range(len(atoms)):
        distances = np.sort(supercell.get_distances(i, [j for j in range(len(supercell)) if j != i], mic=True))
        nearest_three = distances[:3]
        assert nearest_three == pytest.approx(expected_bond, abs=1e-6), (
            f"atom {i} of {symbol} puckered monolayer: expected 3 neighbors at {expected_bond}, got {nearest_three}"
        )


def test_default_2d_kpoints_mesh_forces_z_to_one():
    assert default_2d_kpoints_mesh((9, 9)) == (9, 9, 1)
    assert default_2d_kpoints_mesh((4, 6)) == (4, 6, 1)


def test_is_2d_periodic_rejects_bulk_pbc():
    from ase.build import bulk

    silicon_bulk = bulk("Si", "diamond", a=5.43)
    assert not is_2d_periodic(silicon_bulk)

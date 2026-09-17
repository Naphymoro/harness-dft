"""k-point mesh and ecutwfc convergence sweeps: run successive SCF
calculations and stop once the energy-per-atom change drops below threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ase import Atoms

from harness_dft.workflows.eos import build_scf_inputs


@dataclass
class ConvergencePoint:
    parameter_value: object
    energy_ev: float
    energy_per_atom_ev: float
    delta_ev_per_atom: float | None


@dataclass
class ConvergenceResult:
    converged_value: object
    points: list[ConvergencePoint] = field(default_factory=list)
    converged: bool = False


def _run_scf_energy(atoms: Atoms, code_label, **scf_kwargs) -> float:
    from aiida.engine import run_get_node

    builder, _plan = build_scf_inputs(atoms, code_label, **scf_kwargs)
    results, node = run_get_node(builder)
    if not node.is_finished_ok:
        raise RuntimeError(f"SCF failed: exit status {node.exit_status}")
    return results["output_parameters"].get_dict()["energy"]


def converge_ecutwfc(
    atoms: Atoms,
    code_label,
    ecutwfc_values: list[float] = (30.0, 40.0, 50.0, 60.0, 80.0),
    threshold_ev_per_atom: float = 0.01,
    kpoints_mesh: tuple[int, int, int] = (4, 4, 4),
    **scf_kwargs,
) -> ConvergenceResult:
    n_atoms = len(atoms)
    result = ConvergenceResult(converged_value=ecutwfc_values[-1])
    previous_epa = None

    for ecutwfc in ecutwfc_values:
        energy = _run_scf_energy(
            atoms, code_label, ecutwfc_ry=ecutwfc, kpoints_mesh=kpoints_mesh, **scf_kwargs,
        )
        epa = energy / n_atoms
        delta = None if previous_epa is None else abs(epa - previous_epa)
        result.points.append(ConvergencePoint(ecutwfc, energy, epa, delta))

        if delta is not None and delta < threshold_ev_per_atom:
            result.converged_value = ecutwfc
            result.converged = True
            break
        previous_epa = epa

    return result


def converge_kpoints(
    atoms: Atoms,
    code_label,
    kpoints_meshes: list[tuple[int, int, int]] = ((2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8)),
    threshold_ev_per_atom: float = 0.01,
    ecutwfc_ry: float = 40.0,
    **scf_kwargs,
) -> ConvergenceResult:
    n_atoms = len(atoms)
    result = ConvergenceResult(converged_value=kpoints_meshes[-1])
    previous_epa = None

    for mesh in kpoints_meshes:
        energy = _run_scf_energy(
            atoms, code_label, ecutwfc_ry=ecutwfc_ry, kpoints_mesh=mesh, **scf_kwargs,
        )
        epa = energy / n_atoms
        delta = None if previous_epa is None else abs(epa - previous_epa)
        result.points.append(ConvergencePoint(mesh, energy, epa, delta))

        if delta is not None and delta < threshold_ev_per_atom:
            result.converged_value = mesh
            result.converged = True
            break
        previous_epa = epa

    return result

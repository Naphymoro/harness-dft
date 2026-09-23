"""Equation-of-state fitting: scan isotropic volume scaling with SCF
calculations (PwBaseWorkChain) and fit with ASE's EquationOfState.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.eos import EquationOfState

from harness_dft.builders import apply_calculation_settings, apply_resource_plan, get_electronic_type
from harness_dft.pseudos import validate_family_covers_structure


def _scaled_copy(atoms: Atoms, scale: float) -> Atoms:
    scaled = atoms.copy()
    scaled.set_cell(atoms.get_cell() * scale ** (1 / 3), scale_atoms=True)
    return scaled


def build_scf_inputs(
    atoms: Atoms,
    code_label,
    pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
    protocol: str = "fast",
    kpoints_mesh: tuple[int, int, int] = (4, 4, 4),
    ecutwfc_ry: float = 40.0,
    allow_remote: bool = False,
    allow_gpu: bool = False,
    cpu_batch_size: int = 8,
    local_atom_ceiling: int = 40,
):
    """Return (builder, plan) for a single-point SCF via PwBaseWorkChain."""
    from aiida import orm
    from aiida.orm import load_code
    from aiida.plugins import WorkflowFactory

    validate_family_covers_structure(atoms, pseudo_family_label)

    code = load_code(code_label)
    structure = orm.StructureData(ase=atoms)
    electronic_type = get_electronic_type(atoms)

    PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
    builder = PwBaseWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        overrides={"pseudo_family": pseudo_family_label},
        electronic_type=electronic_type,
    )
    apply_calculation_settings(builder, kpoints_mesh, ecutwfc_ry)
    plan = apply_resource_plan(
        builder.pw, atoms, pseudo_family_label, ecutwfc_ry, kpoints_mesh,
        allow_remote=allow_remote, allow_gpu=allow_gpu,
        cpu_batch_size=cpu_batch_size, local_atom_ceiling=local_atom_ceiling,
    )
    return builder, plan


def run_eos(
    atoms: Atoms,
    code_label,
    scale_range: tuple[float, float] = (0.94, 1.06),
    n_points: int = 7,
    **scf_kwargs,
):
    """Run n_points SCF calculations across isotropic volume scaling and fit
    an equation of state. Returns (eos, volumes, energies_ev, nodes)."""
    from aiida.engine import run_get_node

    scales = np.linspace(scale_range[0], scale_range[1], n_points)
    volumes, energies, nodes = [], [], []

    for scale in scales:
        scaled = _scaled_copy(atoms, scale)
        builder, _plan = build_scf_inputs(scaled, code_label, **scf_kwargs)
        results, node = run_get_node(builder)
        if not node.is_finished_ok:
            raise RuntimeError(f"SCF failed at scale={scale:.4f}: exit status {node.exit_status}")
        energy_ev = results["output_parameters"].get_dict()["energy"]
        volumes.append(scaled.get_volume())
        energies.append(energy_ev)
        nodes.append(node)

    eos = EquationOfState(volumes, energies)
    return eos, volumes, energies, nodes

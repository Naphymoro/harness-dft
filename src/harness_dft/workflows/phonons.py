"""Phonon dispersion pipeline: PhBaseWorkChain (DFPT force constants) ->
Q2rBaseWorkChain (real-space force constants) -> MatdynBaseWorkChain
(interpolated dispersion). Q2r/Matdyn have no `get_builder_from_protocol` in
aiida-quantumespresso -- their builders are assembled by hand, but they're
cheap serial post-processing steps so they don't need the adaptive resource
plan the DFPT step does.
"""
from __future__ import annotations

from harness_dft.builders import apply_resource_plan, get_electronic_type


def build_ph_inputs(
    parent_scf_node,
    ph_code_label,
    atoms,
    pseudo_family_label: str,
    ecutwfc_ry: float,
    qpoints_mesh: tuple[int, int, int] = (2, 2, 2),
    protocol: str = "fast",
    is_metal: bool = False,
    allow_remote: bool = False,
    allow_gpu: bool = False,
    cpu_batch_size: int = 8,
    local_atom_ceiling: int = 40,
):
    """Build a PhBaseWorkChain from a finished PwBaseWorkChain/PwCalculation
    node's remote_folder. `atoms` is the same structure the parent SCF ran
    on -- DFPT cost scales similarly to SCF per-atom, so it reuses the same
    resource estimator rather than a separate one. Returns (builder, plan)."""
    from aiida.orm import load_code
    from aiida.plugins import WorkflowFactory
    from aiida_quantumespresso.common.types import ElectronicType

    PhBaseWorkChain = WorkflowFactory("quantumespresso.ph.base")
    code = load_code(ph_code_label)
    parent_folder = parent_scf_node.outputs.remote_folder

    builder = PhBaseWorkChain.get_builder_from_protocol(
        code=code,
        parent_folder=parent_folder,
        protocol=protocol,
        overrides={"qpoints": list(qpoints_mesh)},
        electronic_type=ElectronicType.METAL if is_metal else ElectronicType.INSULATOR,
    )

    plan = apply_resource_plan(
        builder.ph, atoms, pseudo_family_label, ecutwfc_ry, qpoints_mesh, allow_remote=allow_remote,
        allow_gpu=allow_gpu, cpu_batch_size=cpu_batch_size, local_atom_ceiling=local_atom_ceiling,
    )
    return builder, plan


def build_q2r_inputs(ph_node, q2r_code_label):
    """Build a Q2rBaseWorkChain from a finished PhBaseWorkChain's remote_folder."""
    from aiida.orm import load_code
    from aiida.plugins import WorkflowFactory

    Q2rBaseWorkChain = WorkflowFactory("quantumespresso.q2r.base")
    code = load_code(q2r_code_label)

    builder = Q2rBaseWorkChain.get_builder()
    builder.q2r.code = code
    builder.q2r.parent_folder = ph_node.outputs.remote_folder
    builder.q2r.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    builder.q2r.metadata.options.max_wallclock_seconds = 600
    return builder


def build_matdyn_inputs(q2r_node, matdyn_code_label, kpoints):
    """Build a MatdynBaseWorkChain from a finished Q2rBaseWorkChain's force
    constants output. `kpoints` is an AiiDA KpointsData defining the
    dispersion path or mesh to interpolate onto."""
    from aiida.orm import load_code
    from aiida.plugins import WorkflowFactory

    MatdynBaseWorkChain = WorkflowFactory("quantumespresso.matdyn.base")
    code = load_code(matdyn_code_label)

    builder = MatdynBaseWorkChain.get_builder()
    builder.matdyn.code = code
    builder.matdyn.force_constants = q2r_node.outputs.force_constants
    builder.matdyn.kpoints = kpoints
    builder.matdyn.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    builder.matdyn.metadata.options.max_wallclock_seconds = 600
    return builder


def run_phonon_dispersion(
    parent_scf_node,
    ph_code_label,
    q2r_code_label,
    matdyn_code_label,
    dispersion_kpoints,
    atoms,
    pseudo_family_label: str,
    ecutwfc_ry: float,
    qpoints_mesh: tuple[int, int, int] = (2, 2, 2),
    protocol: str = "fast",
    is_metal: bool = False,
):
    """Blocking convenience wrapper chaining Ph -> Q2r -> Matdyn. Returns the
    three finished workchain nodes; `dispersion_kpoints` is an AiiDA
    KpointsData (band path or mesh) for the final interpolation."""
    from aiida.engine import run_get_node

    ph_builder, _ = build_ph_inputs(
        parent_scf_node, ph_code_label, atoms, pseudo_family_label, ecutwfc_ry,
        qpoints_mesh=qpoints_mesh, protocol=protocol, is_metal=is_metal,
    )
    _, ph_node = run_get_node(ph_builder)
    if not ph_node.is_finished_ok:
        raise RuntimeError(f"PhBaseWorkChain failed: exit status {ph_node.exit_status}")

    q2r_builder = build_q2r_inputs(ph_node, q2r_code_label)
    _, q2r_node = run_get_node(q2r_builder)
    if not q2r_node.is_finished_ok:
        raise RuntimeError(f"Q2rBaseWorkChain failed: exit status {q2r_node.exit_status}")

    matdyn_builder = build_matdyn_inputs(q2r_node, matdyn_code_label, dispersion_kpoints)
    _, matdyn_node = run_get_node(matdyn_builder)
    if not matdyn_node.is_finished_ok:
        raise RuntimeError(f"MatdynBaseWorkChain failed: exit status {matdyn_node.exit_status}")

    return ph_node, q2r_node, matdyn_node

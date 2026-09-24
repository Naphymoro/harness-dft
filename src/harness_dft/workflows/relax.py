"""Structure relaxation via AiiDA's PwRelaxWorkChain, with resource-adaptive
metadata.options applied to both the relax and final-SCF sub-calculations.
"""
from __future__ import annotations

from ase import Atoms

from harness_dft.builders import apply_calculation_settings, apply_cell_dofree, apply_resource_plan, get_electronic_type
from harness_dft.pseudos import validate_family_covers_structure


def build_relax_inputs(
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
    cell_dofree: str | None = None,
    force_metal: bool | None = None,
):
    """Return (builder, plan) for a PwRelaxWorkChain run on `atoms`.

    `cell_dofree` (e.g. `"2Dxy"` for a slab-with-vacuum structure -- see
    `harness_dft.twod`) restricts the vc-relax step's cell degrees of
    freedom; only applied to `base` (the vc-relax step), never
    `base_final_scf` (a fixed-cell static SCF, where QE ignores CELL anyway).

    `force_metal` overrides the electronic-type heuristic (see
    `get_electronic_type`) -- real need: `is_likely_metal` doesn't know a 2D
    monolayer of a nominally non-metallic element can be near-metallic, and
    QE's fixed-occupations (insulator) SCF can fail to converge there where
    smearing (metal) works fine.

    Doesn't submit -- call `aiida.engine.run_get_node(builder)` for a
    blocking run or `aiida.engine.submit(builder)` to hand it to the daemon.
    """
    from aiida import orm
    from aiida.orm import load_code
    from aiida.plugins import WorkflowFactory

    validate_family_covers_structure(atoms, pseudo_family_label)

    code = load_code(code_label)
    structure = orm.StructureData(ase=atoms)
    electronic_type = get_electronic_type(atoms, force_metal=force_metal)

    PwRelaxWorkChain = WorkflowFactory("quantumespresso.pw.relax")
    pseudo_override = {"pseudo_family": pseudo_family_label}
    builder = PwRelaxWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        overrides={"base": pseudo_override, "base_final_scf": pseudo_override},
        electronic_type=electronic_type,
    )

    apply_calculation_settings(builder.base, kpoints_mesh, ecutwfc_ry)
    apply_calculation_settings(builder.base_final_scf, kpoints_mesh, ecutwfc_ry)
    apply_cell_dofree(builder.base, cell_dofree)
    resource_kwargs = dict(
        allow_remote=allow_remote, allow_gpu=allow_gpu,
        cpu_batch_size=cpu_batch_size, local_atom_ceiling=local_atom_ceiling,
    )
    plan = apply_resource_plan(
        builder.base.pw, atoms, pseudo_family_label, ecutwfc_ry, kpoints_mesh, **resource_kwargs,
    )
    apply_resource_plan(
        builder.base_final_scf.pw, atoms, pseudo_family_label, ecutwfc_ry, kpoints_mesh, **resource_kwargs,
    )
    return builder, plan


def run_relax(atoms: Atoms, code_label, **kwargs):
    """Blocking convenience wrapper: builds and runs a relaxation, returning
    (results_dict, workchain_node)."""
    from aiida.engine import run_get_node

    builder, plan = build_relax_inputs(atoms, code_label, **kwargs)
    results, node = run_get_node(builder)
    return results, node, plan

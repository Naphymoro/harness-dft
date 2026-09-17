"""Band structure (PwBandsWorkChain) and density of states (PdosWorkChain)
on an already-relaxed structure. Both take a resource plan applied to their
`scf`/`bands`/`nscf` sub-namespaces.
"""
from __future__ import annotations

from ase import Atoms

from harness_dft.builders import apply_calculation_settings, apply_ecutwfc, apply_resource_plan, get_electronic_type
from harness_dft.pseudos import validate_family_covers_structure


def build_bands_inputs(
    atoms: Atoms,
    code_label,
    pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
    protocol: str = "fast",
    kpoints_mesh: tuple[int, int, int] = (4, 4, 4),
    ecutwfc_ry: float = 40.0,
    allow_remote: bool = False,
):
    """Build a PwBandsWorkChain (SCF + auto k-path bands via seekpath) for an
    already-relaxed structure. Returns (builder, scf_plan, bands_plan)."""
    from aiida import orm
    from aiida.orm import load_code
    from aiida.plugins import WorkflowFactory

    validate_family_covers_structure(atoms, pseudo_family_label)

    code = load_code(code_label)
    structure = orm.StructureData(ase=atoms)
    electronic_type = get_electronic_type(atoms)

    PwBandsWorkChain = WorkflowFactory("quantumespresso.pw.bands")
    pseudo_override = {"pseudo_family": pseudo_family_label}
    builder = PwBandsWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        overrides={"scf": pseudo_override, "bands": pseudo_override},
        electronic_type=electronic_type,
    )

    apply_calculation_settings(builder.scf, kpoints_mesh, ecutwfc_ry)
    apply_ecutwfc(builder.bands, ecutwfc_ry)  # bands k-path is auto (seekpath) -- don't touch kpoints here

    scf_plan = apply_resource_plan(
        builder.scf.pw, atoms, pseudo_family_label, ecutwfc_ry, kpoints_mesh, allow_remote=allow_remote,
    )
    bands_plan = apply_resource_plan(
        builder.bands.pw, atoms, pseudo_family_label, ecutwfc_ry, kpoints_mesh, allow_remote=allow_remote,
    )
    return builder, scf_plan, bands_plan


def build_pdos_inputs(
    atoms: Atoms,
    pw_code_label,
    dos_code_label,
    projwfc_code_label,
    pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
    protocol: str = "fast",
    kpoints_mesh: tuple[int, int, int] = (4, 4, 4),
    nscf_kpoints_mesh: tuple[int, int, int] | None = None,
    ecutwfc_ry: float = 40.0,
    allow_remote: bool = False,
):
    """Build a PdosWorkChain (SCF + NSCF + dos.x + projwfc.x) for an
    already-relaxed structure. NSCF conventionally uses a denser mesh than
    SCF for smooth DOS -- defaults to doubling `kpoints_mesh` if not given.
    Returns (builder, scf_plan, nscf_plan)."""
    if nscf_kpoints_mesh is None:
        nscf_kpoints_mesh = tuple(2 * k for k in kpoints_mesh)
    from aiida import orm
    from aiida.orm import load_code
    from aiida.plugins import WorkflowFactory

    validate_family_covers_structure(atoms, pseudo_family_label)

    pw_code = load_code(pw_code_label)
    dos_code = load_code(dos_code_label)
    projwfc_code = load_code(projwfc_code_label)
    structure = orm.StructureData(ase=atoms)
    electronic_type = get_electronic_type(atoms)

    PdosWorkChain = WorkflowFactory("quantumespresso.pdos")
    pseudo_override = {"pseudo_family": pseudo_family_label}
    builder = PdosWorkChain.get_builder_from_protocol(
        pw_code=pw_code,
        dos_code=dos_code,
        projwfc_code=projwfc_code,
        structure=structure,
        protocol=protocol,
        overrides={"scf": pseudo_override, "nscf": pseudo_override},
        electronic_type=electronic_type,
    )

    apply_calculation_settings(builder.scf, kpoints_mesh, ecutwfc_ry)
    apply_calculation_settings(builder.nscf, nscf_kpoints_mesh, ecutwfc_ry)

    scf_plan = apply_resource_plan(
        builder.scf.pw, atoms, pseudo_family_label, ecutwfc_ry, kpoints_mesh, allow_remote=allow_remote,
    )
    nscf_plan = apply_resource_plan(
        builder.nscf.pw, atoms, pseudo_family_label, ecutwfc_ry, nscf_kpoints_mesh, allow_remote=allow_remote,
    )
    return builder, scf_plan, nscf_plan

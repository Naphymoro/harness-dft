---
name: dft-phonons
description: Compute phonon dispersion via DFPT (PhBaseWorkChain -> Q2rBaseWorkChain -> MatdynBaseWorkChain) starting from a converged SCF. Use when the user wants phonon dispersion, dynamical stability (checking for imaginary modes), or vibrational properties.
---

# Phonon dispersion

Three-stage pipeline: `ph.x` (DFPT force constants at a q-point mesh) -> `q2r.x` (real-space force constants) -> `matdyn.x` (interpolated dispersion on a path/mesh).

## Prerequisites

- A **finished, converged SCF** `PwBaseWorkChain` node to restart from (its `remote_folder` output is the DFPT starting point) — run `dft-relax` (or a plain SCF) first and keep the resulting node.
- `ph.x`, `q2r.x`, `matdyn.x` all registered as separate AiiDA codes.

## Steps

```python
from harness_dft.workflows.phonons import run_phonon_dispersion
from aiida.orm import KpointsData

dispersion_kpoints = KpointsData()
dispersion_kpoints.set_kpoints_path()  # or set_kpoints_mesh(...) for a mesh instead of a path

ph_node, q2r_node, matdyn_node = run_phonon_dispersion(
    parent_scf_node=scf_node,  # a finished PwBaseWorkChain node
    ph_code_label="ph-7.5@localhost",
    q2r_code_label="q2r-7.5@localhost",
    matdyn_code_label="matdyn-7.5@localhost",
    dispersion_kpoints=dispersion_kpoints,
    atoms=relaxed_atoms,
    pseudo_family_label="SSSP/1.3/PBE/efficiency",
    ecutwfc_ry=50.0,
    qpoints_mesh=(2, 2, 2),  # coarse q-mesh for DFPT; interpolated finer by matdyn
)
```

- Check `matdyn_node.outputs.output_phonon_bands` for the dispersion. Negative (imaginary) frequencies indicate dynamical instability at that q-point — worth flagging explicitly to the user, it's a common thing they're checking for.

## Known gotchas

- `q2r.x` and `matdyn.x` have **no** `get_builder_from_protocol()` in `aiida-quantumespresso` — their builders are hand-assembled in `harness_dft.workflows.phonons` (`build_q2r_inputs`, `build_matdyn_inputs`) with fixed minimal resources (1 process, 10 min walltime), since they're cheap serial post-processing, not adaptive-resourced like the DFPT step.
- The DFPT (`ph.x`) step reuses the harness's standard resource estimator (same per-atom cost model as SCF) via `build_ph_inputs` — this is a coarse approximation; DFPT cost also scales with the number of irreducible atomic displacements, which the estimator doesn't account for. If a `ph.x` run times out repeatedly even after AiiDA's automatic walltime-based resubmission, that's likely why — consider bumping `qpoints_mesh` coarser or resources manually.
- **Not live-tested** during harness development — the DFPT pipeline is expensive to iterate on. Builder namespaces (`builder.ph`, `builder.q2r.parent_folder`, `builder.matdyn.force_constants`) were verified against the installed source, not run. Treat the first real invocation as validation, not a known-good path.
- `is_metal` must be passed explicitly and correctly to `build_ph_inputs` — DFPT dielectric/Born-effective-charge handling (`epsil` flag) differs for insulators vs. metals, and getting this wrong doesn't necessarily crash, it silently changes what's computed.

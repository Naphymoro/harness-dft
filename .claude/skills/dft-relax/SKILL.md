---
name: dft-relax
description: Relax an atomic structure (ions and/or cell) with Quantum ESPRESSO via AiiDA's PwRelaxWorkChain, with resource-adaptive mpiprocs/npool/walltime. Use when the user wants to optimize a structure's geometry, find its equilibrium lattice, or "relax"/"optimize" a crystal or molecule before further DFT analysis.
---

# DFT structure relaxation

Runs `PwRelaxWorkChain` (AiiDA + Quantum ESPRESSO) on a structure, with the harness's resource estimator picking mpiprocs/npool/walltime automatically.

## When to use this

- User gives a structure (CIF, POSCAR, ASE-buildable, or an already-loaded `ase.Atoms`) and wants it relaxed/optimized.
- Before running bands, DOS, phonons, or EOS on a structure that isn't already at its DFT-relaxed geometry — those workflows expect a relaxed input.

## Prerequisites

- An AiiDA profile loaded (`aiida.load_profile()`), a registered `pw.x` code, and an installed pseudopotential family (see `dft-pseudo-select` skill if unsure these exist).
- `harness_dft` importable (this repo's `src/` on `PYTHONPATH`, or installed via `pip install -e .`).

## Steps

1. Confirm/obtain the structure as an `ase.Atoms` object (`ase.io.read(path)` or `ase.build.*`).
2. Confirm the pseudopotential family covers all elements (`harness_dft.pseudos.validate_family_covers_structure` — `build_relax_inputs` already calls this and raises `MissingPseudopotentialError` with a clear message if not).
3. Pick a `protocol`: `"fast"` for quick/test runs, `"balanced"` (default AiiDA-QE choice) for production, `"stringent"` for high precision.
4. Call:
   ```python
   from harness_dft.workflows.relax import run_relax
   results, node, plan = run_relax(
       atoms, code_label="pw-7.5@localhost", protocol="balanced",
       kpoints_mesh=(4, 4, 4), ecutwfc_ry=50.0,
   )
   ```
5. Report `plan` (what resources were chosen and why) alongside the result — this is the harness's adaptivity in action, worth surfacing to the user.
6. On success (`node.is_finished_ok`), the relaxed structure is `results['output_structure']` (an AiiDA `StructureData` — `.get_ase()` converts back) and the final energy is `results['output_parameters'].get_dict()['energy']` (eV).
7. On failure, run `verdi process report <node.pk>` and read it before guessing — AiiDA's automatic error handlers (out-of-walltime, diagonalization errors, etc.) already retry recoverable failures, so a final failure usually means something structural (bad pseudopotential match, unphysical starting geometry, wrong `ElectronicType` guess).

## Known gotchas (hit and fixed while building this harness)

- `get_electronic_type()` uses a coarse heuristic (`harness_dft.structures.is_likely_metal`) based on element symbols only. It's wrong for e.g. metallic alloys behaving as semiconductors, or vice versa. If a relaxation fails with `ERROR_CHARGE_IS_WRONG` or convergence issues tied to occupations/smearing, try overriding the electronic type explicitly rather than trusting the heuristic.
- `allow_remote=True` only *recommends* routing to remote — it doesn't submit there unless the `code_label` you pass actually resolves to a code on a remote computer. Local vs. remote is decided by which code you load, not by this flag alone.

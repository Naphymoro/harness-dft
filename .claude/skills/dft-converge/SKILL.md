---
name: dft-converge
description: Run a k-point mesh or plane-wave cutoff (ecutwfc) convergence sweep against energy-per-atom, and pick the first value under a threshold. Use when the user asks to "converge" a calculation, validate a cutoff/k-mesh choice, or wants confidence that subsequent DFT results aren't basis-set artifacts.
---

# k-point / ecutwfc convergence testing

Runs successive SCF calculations (`PwBaseWorkChain`) at increasing parameter values and stops once the energy-per-atom change between steps drops below a threshold.

## When to use this

- Before a production run (relax, bands, EOS, phonons) on a new structure/pseudopotential combination the user hasn't validated cutoffs for.
- When the user explicitly asks to check/tune k-points or ecutwfc.

## Steps

1. Ensure the structure, code, and pseudopotential family are ready (see `dft-pseudo-select` if unsure).
2. Run the ecutwfc sweep first (cheaper to iterate), holding k-points fixed at a reasonably dense mesh:
   ```python
   from harness_dft.workflows.converge import converge_ecutwfc
   result = converge_ecutwfc(
       atoms, code_label="pw-7.5@localhost",
       ecutwfc_values=[30, 40, 50, 60, 80], threshold_ev_per_atom=0.01,
       kpoints_mesh=(4, 4, 4),
   )
   ```
3. Then sweep k-points at the converged ecutwfc:
   ```python
   from harness_dft.workflows.converge import converge_kpoints
   result = converge_kpoints(
       atoms, code_label="pw-7.5@localhost",
       kpoints_meshes=[(2,2,2), (4,4,4), (6,6,6), (8,8,8)],
       ecutwfc_ry=result.converged_value, threshold_ev_per_atom=0.01,
   )
   ```
4. Report `result.points` (the full trace) and `result.converged_value`. If `result.converged` is `False`, the sweep ran out of values without meeting the threshold — extend the range rather than trusting the last point.
5. Feed the converged `ecutwfc_ry` and `kpoints_mesh` into whichever production workflow (`dft-relax`, `dft-bands-dos`, `dft-eos`) comes next.

## Known gotchas

- This module directly overrides `SYSTEM.ecutwfc`/`ecutrho` and the k-point mesh on the builder (`harness_dft.builders.apply_calculation_settings`) — a real bug was caught during development where `ecutwfc_ry` was only feeding the resource *estimator* and never reaching the actual QE input, silently making convergence sweeps meaningless (every point ran at the protocol's default cutoff). If a sweep ever comes back with byte-identical energies across different parameter values, that's the symptom to check for again — the values genuinely aren't reaching the calculation.
- Convergence in energy-per-atom doesn't guarantee convergence in forces/stress if the target is a relaxation — for that, also check `forc_conv_thr` sensitivity, which this module doesn't sweep.

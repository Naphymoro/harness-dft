---
name: dft-eos
description: Fit an equation of state (bulk modulus, equilibrium volume/energy) by running SCF at a range of isotropic volume scalings and fitting with ASE's EquationOfState. Use when the user wants bulk modulus, equilibrium lattice constant/volume, or an E-V curve.
---

# Equation of state

```python
from harness_dft.workflows.eos import run_eos

eos, volumes, energies, nodes = run_eos(
    atoms, code_label="pw-7.5@localhost",
    scale_range=(0.94, 1.06), n_points=7,
    protocol="balanced", kpoints_mesh=(6, 6, 6), ecutwfc_ry=50.0,
)
v0, e0, B = eos.fit()  # equilibrium volume (A^3), energy (eV), bulk modulus (eV/A^3)
```

## Steps

1. Start from a structure already reasonably close to equilibrium (run `dft-converge` first if cutoffs aren't validated — EOS fitting is sensitive to basis-set-incomplete energies, since it fits *curvature*, not just absolute energy).
2. `n_points=7` across `scale_range=(0.94, 1.06)` is a reasonable default (isotropic volume scaling, not just lattice-constant scaling — the harness scales the cell by `scale**(1/3)` per axis so this is a true volume scale). Widen the range if the fit residual is poor or the minimum falls near an edge.
3. Convert bulk modulus to GPa if reporting to a user: `B_GPa = B * 160.21766208` (eV/Å³ → GPa).
4. If any point raises `RuntimeError` (SCF failed at that scale), narrow the range — large compressions/expansions can push SCF into non-convergence.

## Known gotchas

- Each point is an independent SCF (fixed cell shape, scaled isotropically) — not a full `vc-relax`, so this assumes the structure's *shape* (not just volume) is already close to correct. For strongly anisotropic materials, an isotropic EOS scan can miss the true minimum; consider per-axis or `vc-relax`-based sweeps instead if the user's material is anisotropic.
- Not live-tested end-to-end during harness development (reuses the same `build_scf_inputs` path validated by `dft-converge`'s live test, so the underlying builder construction is trusted, but the EOS-specific volume-scaling + `ase.eos.EquationOfState` fit wasn't run against a real multi-point dataset).

---
name: dft-neb
description: Estimate a reaction/diffusion barrier using ASE's own NEB (ase.mep), with Quantum ESPRESSO as the force engine via ASE's Espresso calculator directly (no AiiDA provenance). Use when the user wants a migration barrier, transition state energy, or reaction pathway.
---

# Nudged elastic band (NEB)

Deliberately bypasses AiiDA/QE's own `neb.x` — ASE's NEB (`ase.mep`, moved there from `ase.neb` as of recent ASE versions) is an independent implementation, and barrier searches are typically iterative/exploratory rather than needing full provenance tracking, which is why this sits in the ASE layer rather than the AiiDA layer (see the layering principle in `harness_dft.builders`'s module docstring).

## Steps

1. Have two **already-relaxed** endpoint structures (`initial`, `final`) as `ase.Atoms` — relax each with `dft-relax` first (via AiiDA, for provenance on the endpoints), then convert to plain `ase.Atoms` for the NEB itself.
2. Build a calculator factory — each NEB image needs its own calculator instance, not a shared one:
   ```python
   from harness_dft.workflows.neb import build_espresso_calculator, run_neb, barrier_energy

   def make_calc():
       return build_espresso_calculator(
           pseudopotentials={"Si": "Si.upf", ...},
           pseudo_dir="/path/to/pseudos",
           pw_command="pw.x",
           kpts=(2, 2, 2), ecutwfc=40.0,
       )

   neb, images, optimizer = run_neb(initial, final, make_calc, n_images=5, climb=True, fmax=0.05)
   barrier_ev = barrier_energy(images, initial_energy=images[0].get_potential_energy())
   ```
3. `use_idpp=True` (default) interpolates the initial path with image-dependent pair potentials rather than linear interpolation — keep this on unless the user has a specific reason not to; linear interpolation on anything but the simplest paths tends to produce unphysical starting guesses that fail to converge.
4. Report the barrier in eV, and note which image is the transition-state guess (`max(images, key=lambda im: im.get_potential_energy())`).

## Known gotchas

- This path uses ASE's `Espresso` calculator directly (subprocess to `pw.x`), **not** AiiDA — no provenance graph, no automatic error handling/resubmission on convergence failures. If an image's SCF fails to converge, ASE will raise or return a bad energy silently depending on calculator settings; check `EspressoProfile`/calculator logs per image if the NEB path looks physically wrong.
- `pseudo_dir` must point to wherever the actual `.UPF` files live on disk — this is separate from AiiDA's pseudopotential family database (`aiida-pseudo`), which stores pseudopotentials as AiiDA data nodes, not as loose files. If you've only installed pseudopotentials via `aiida-pseudo install`, you'll need to export the specific UPF files to a directory first (or look them up via `aiida.orm.load_group(family_label).get_pseudo(symbol).get_content()` and write them out).
- Not implemented/tested via AiiDA's own `NebBaseWorkChain` at all — that's a documented alternative in the capability map if the user specifically wants AiiDA provenance for the barrier search instead of ASE's implementation.

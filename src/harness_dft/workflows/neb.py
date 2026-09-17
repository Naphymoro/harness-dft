"""Reaction/diffusion barriers via ASE's own NEB implementation (ase.mep),
using QE as the force engine through ASE's `Espresso` calculator directly
(subprocess, no AiiDA provenance). This deliberately skips AiiDA/QE's
`neb.x` -- ASE's NEB is a separate, independent implementation, and this is
exactly the "fast local, no provenance needed" case the harness's ASE layer
is for (see harness_dft.builders module docstring for the layering
rationale).
"""
from __future__ import annotations

from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.mep import NEB, idpp_interpolate
from ase.optimize import FIRE


def build_espresso_calculator(
    pseudopotentials: dict[str, str],
    pseudo_dir: str,
    pw_command: str = "pw.x",
    kpts: tuple[int, int, int] = (2, 2, 2),
    ecutwfc: float = 40.0,
    ecutrho: float | None = None,
    input_data: dict | None = None,
) -> Espresso:
    """ASE's Espresso calculator, driving `pw.x` directly as a subprocess
    (no AiiDA). `pseudopotentials` maps element symbol -> UPF filename."""
    profile = EspressoProfile(command=pw_command, pseudo_dir=pseudo_dir)
    data = {"system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho or ecutwfc * 8}}
    if input_data:
        data.update(input_data)
    return Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        kpts=kpts,
        input_data=data,
    )


def run_neb(
    initial: Atoms,
    final: Atoms,
    calculator_factory,
    n_images: int = 5,
    climb: bool = True,
    fmax: float = 0.05,
    use_idpp: bool = True,
):
    """Run an NEB between `initial` and `final` (already-relaxed endpoints).
    `calculator_factory` is a zero-arg callable returning a fresh ASE
    calculator instance (e.g. `lambda: build_espresso_calculator(...)`) --
    each image needs its own calculator instance.

    Returns (neb, images, optimizer) after optimization; the transition
    state is the image with the highest energy in `images`.
    """
    images = [initial.copy()]
    images += [initial.copy() for _ in range(n_images)]
    images.append(final.copy())

    if use_idpp:
        idpp_interpolate(images)

    for image in images[1:-1]:
        image.calc = calculator_factory()

    neb = NEB(images, climb=climb)
    optimizer = FIRE(neb)
    optimizer.run(fmax=fmax)

    return neb, images, optimizer


def barrier_energy(images, initial_energy: float | None = None) -> float:
    """Energy barrier (eV) as max(image energies) - initial energy."""
    energies = [image.get_potential_energy() for image in images]
    reference = initial_energy if initial_energy is not None else energies[0]
    return max(energies) - reference

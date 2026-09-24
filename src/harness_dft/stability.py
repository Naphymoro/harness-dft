"""Dynamical-stability assessment from a finished phonon dispersion
(harness_dft.workflows.phonons' ph -> q2r -> matdyn chain).

"Stable" here means exactly one thing: no real (non-numerical-noise)
negative/imaginary phonon frequency anywhere in the sampled part of the
Brillouin zone. This is necessary but not sufficient for a candidate 2D
structure being an actual synthesizable/observed material -- it only rules
out the structure being a saddle point (or worse) on the potential-energy
surface at zero temperature. Formation energy relative to competing
prototypes/the bulk allotrope, thermal stability at finite temperature, and
electronic-structure sanity are separate questions this module says nothing
about.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImaginaryMode:
    qpoint: tuple[float, float, float]
    frequency_thz: float


@dataclass(frozen=True)
class StabilityResult:
    stable: bool
    min_frequency_thz: float
    tolerance_thz: float
    n_qpoints: int
    n_modes: int
    imaginary_modes: list[ImaginaryMode] = field(default_factory=list)


def check_dynamical_stability(matdyn_pk: int, tolerance_thz: float = -0.5) -> StabilityResult:
    """Read a finished MatdynBaseWorkChain's `output_phonon_bands` and check
    for imaginary (negative) frequencies.

    `tolerance_thz` (default -0.5 THz) is a numerical-noise allowance, not a
    physical one: small negative acoustic-branch frequencies right at Gamma
    are a known artifact of imperfect translational-invariance enforcement,
    even with `asr` applied (see `build_matdyn_inputs`) -- real dynamical
    instabilities are typically much larger in magnitude and/or span an
    extended region of the Brillouin zone, not a single point. A frequency
    below this tolerance is reported as imaginary; anything above it (even
    if formally negative) is treated as numerical noise. Tighten this
    (closer to 0) once `asr` correction is confirmed to clean up a given
    system's Gamma-point noise, or loosen it if a coarser/less-converged
    calculation shows more numerical spread.
    """
    from aiida import orm

    node = orm.load_node(matdyn_pk)
    if not node.is_finished_ok:
        raise RuntimeError(f"pk={matdyn_pk} is not a finished, successful matdyn job (exit_status={node.exit_status})")

    bands_node = node.outputs.output_phonon_bands
    frequencies = bands_node.get_bands()  # shape (n_qpoints, n_modes), THz
    qpoints = bands_node.get_kpoints()  # shape (n_qpoints, 3), reduced coordinates

    min_frequency = float(frequencies.min())
    imaginary_modes = []
    for q_index, q in enumerate(qpoints):
        row_min = frequencies[q_index].min()
        if row_min < tolerance_thz:
            imaginary_modes.append(ImaginaryMode(qpoint=tuple(float(x) for x in q), frequency_thz=float(row_min)))

    return StabilityResult(
        stable=min_frequency >= tolerance_thz,
        min_frequency_thz=min_frequency,
        tolerance_thz=tolerance_thz,
        n_qpoints=frequencies.shape[0],
        n_modes=frequencies.shape[1],
        imaginary_modes=imaginary_modes,
    )

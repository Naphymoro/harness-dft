# Upstream bugs found while building harness-dft

Bugs in third-party projects (QE itself, `aiida-quantumespresso`) discovered while validating this harness,
root-caused with a real reproduction and (where applicable) a real fix -- not just worked around in harness code.

## `qe-phq-summary-missing-comma.patch` -- QE 7.5, `PHonon/PH/phq_summary.f90`

**Symptom**: `ph.x` crashes within seconds of starting a phonon calculation on certain structures, with a
`libgfortran` I/O-layer crash (`data_transfer_init`). Looked at first like an MPI/ScaLAPACK/rank-count issue
(the crash signature is generic and easy to misattribute) -- it is not.

**Root cause**: a genuinely missing comma in a `WRITE` FORMAT string in `phq_summary.f90`, in the block that only
executes when printing a symmetry operation with a nonzero fractional translation (a non-symmorphic operation).
Every sibling `WRITE` statement in the same subroutine has the comma; this one doesn't. Confirmed two ways:

1. A debug rebuild (plain `gfortran -O0`, no MPI, no HDF5) reproduces a clear, human-readable compiler runtime
   error naming the exact file and line: `Fortran runtime error: Missing comma between descriptors`.
2. A full production-equivalent rebuild (MPI + HDF5, matching the harness's actual QE install) with the one-line
   fix applied completes an otherwise 100%-reproducible crash case (a 2-atom hexagonal `ibrav=0` cell, 4-qpoint
   phonon dispersion) to completion, producing correct, physically sane frequencies.

**Why it correlates with `ibrav=0`**: non-symmorphic symmetry operations (nonzero fractional translation) are far
more likely to appear when QE has to determine symmetry from generic `CELL_PARAMETERS`/`ATOMIC_POSITIONS`
(`ibrav=0`) than when a recognized Bravais lattice lets it choose a symmetric origin. `ibrav=0` is exactly what
every AiiDA `StructureData` (and therefore every ASE/pymatgen-built structure passed through AiiDA) produces --
this is why `pw.x`'s own "using `ibrav=0` with symmetry is DISCOURAGED" warning turns out to matter for `ph.x`,
even though `pw.x` itself tolerates `ibrav=0` fine (it never reaches this buggy printing code).

**A wrong diagnosis this bug produced first, corrected**: the crash was initially (incorrectly) attributed to
ScaLAPACK-based parallel diagonalization failing for small matrices, because `-northo 0` (disabling ScaLAPACK)
appeared to avoid the crash in an early manual reproduction. That reproduction used a hand-typed, not-fully-
faithful SCF input (different smearing/cutoff details than the real failing case); re-tested against the *exact*
real failing input, `-northo 0` did **not** avoid the crash. The harness briefly shipped a `disable_scalapack`
parameter based on this wrong diagnosis; it has been removed. The comma fix above is the confirmed, actual root
cause and fix -- verified against the real case, not a simplified one.

**Status**: fixed and rebuilt locally (`~/.local/share/cyranoid/gpu-build/q-e-qe-7.5/build-fixed/`, registered in
this profile as `ph-7.5-fixed@localhost`). A ready-to-paste GitLab issue is drafted at
`gitlab-issue-draft.md` (title + full body, patch included) for
[QEF/q-e](https://gitlab.com/QEF/q-e) -- **not yet submitted**, since posting to a third-party project's issue
tracker under someone's identity is their action to take, not something done unilaterally on their behalf. Update
this line once it's actually filed (link the real issue here).

## `aiida-quantumespresso`'s `PhBaseWorkChain` restart handler drops `INPUTPH`

Separate, still-open issue -- see `docs/2d-screening.md`'s writeup for P/Sb's phonon failures. Not a QE bug;
`aiida_quantumespresso/calculations/ph.py:365` raises `KeyError: 'INPUTPH'` when a `PhCalculation` restarts after
a handled failure (e.g. walltime), because the restart path reconstructs calculation inputs without the
`INPUTPH` namelist. No fix attempted here (different codebase, different maintainers); avoiding it entirely
requires the calculation to finish in one attempt (no restart), which for P/Sb exceeded even a 2-hour walltime
budget on this hardware at the pilot's q-mesh/cutoff.

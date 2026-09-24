# Draft GitLab issue for gitlab.com/QEF/q-e

Ready to paste as-is into a new issue at `gitlab.com/QEF/q-e/-/issues/new`. **Not yet submitted** -- filing it
under the user's identity is their action to take, not something done unilaterally here.

## Title

```
PHonon/PH: ph.x crashes with "Missing comma between descriptors" when printing non-symmorphic symmetry operations (ibrav=0)
```

## Body

```markdown
## Summary

`ph.x` aborts within seconds of starting a phonon calculation, on certain structures, with a Fortran runtime
I/O error. The crash signature (`libgfortran`'s `data_transfer_init`, an MPI rank aborting with a generic
non-zero exit) is generic and easy to misattribute to MPI/ScaLAPACK/parallelization settings -- it is not related
to any of those. The actual cause is a genuine syntax defect in a `WRITE` format string.

## Root cause

`PHonon/PH/phq_summary.f90`, in the block that prints a symmetry operation's crystal-coordinate representation
when it has a nonzero fractional translation (i.e. a **non-symmorphic** operation):

```fortran
           WRITE(stdout, '(1x,"cryst.",3x,"s(",i2,") = (",3(i6,5x) &
                &                    " )    f =( ",f10.7," )")') isymq,  &
                & (s(1,ipol,isym), ipol = 1, 3), ft(1,isym)
```

There is **no comma** between the repeated numeric descriptor `3(i6,5x)` and the following character-literal
descriptor `" )    f =( "`. Every sibling `WRITE` in the same subroutine has the comma -- e.g. the very next
statement two lines below:

```fortran
           WRITE(stdout, '(17x," (",3(i6,5x), &
                &                    " )       ( ",f10.7," )")')  &
                & (s(2,ipol,isym), ipol = 1, 3), ft(2,isym)
```

Some Fortran compilers silently accept the missing comma as a non-standard extension; a strict compiler (in our
case, `gfortran` 15.2.0) aborts at runtime with:

```
Fortran runtime error: Missing comma between descriptors
(1x,"cryst.",3x,"s(",i2,") = (",3(i6,5x)                     " )    f =( ",f10.7
```

## Why this correlates with `ibrav=0`

This block only executes when a symmetry operation's fractional translation is nonzero (`ft(1,isym)**2 + ... >
1.0d-8`). Non-symmorphic operations are far more likely to appear when the symmetry finder has to work from
generic `CELL_PARAMETERS`/`ATOMIC_POSITIONS` (`ibrav=0`) than when a recognized Bravais lattice lets it choose a
symmetric origin. This is presumably why `pw.x`'s own "`Message from routine setup: using ibrav=0 with symmetry
is DISCOURAGED, use correct ibrav instead`" warning matters here -- `pw.x` itself never reaches this printing
code, so it tolerates `ibrav=0` fine; `ph.x` does reach it and aborts.

This also explains why the crash is easy to misdiagnose as parallelization-related: it only reproduces for
structures whose symmetry happens to include a non-symmorphic operation, which can look like it "depends on rank
count" or process-grid layout if you're comparing different structures/settings rather than isolating the actual
variable.

## Steps to reproduce

1. Build any structure with `ibrav=0` whose space group includes a non-symmorphic operation (e.g. a 2-atom
   hexagonal monolayer with a glide-plane-type symmetry; this was found via a 2D honeycomb-lattice monolayer
   built from generic `CELL_PARAMETERS`, no recognized `ibrav`).
2. Run a self-consistent `pw.x` calculation (completes fine, may print the "ibrav=0 with symmetry is
   DISCOURAGED" warning).
3. Run `ph.x` on the resulting charge density with `verbosity='high'` (or default verbosity where this print
   path is reached) at `q=(0,0,0)` (or any q-point whose symmetry group contains a non-symmorphic operation).
4. `ph.x` aborts within seconds with the error above, before any real DFPT computation begins.

## Fix

One-line patch (tested against QE 7.5):

```diff
--- a/PHonon/PH/phq_summary.f90
+++ b/PHonon/PH/phq_summary.f90
@@ -186,7 +186,7 @@
            ft1 = at(1,1)*ft(1,isym) + at(1,2)*ft(2,isym) + at(1,3)*ft(3,isym)
            ft2 = at(2,1)*ft(1,isym) + at(2,2)*ft(2,isym) + at(2,3)*ft(3,isym)
            ft3 = at(3,1)*ft(1,isym) + at(3,2)*ft(2,isym) + at(3,3)*ft(3,isym)
-           WRITE(stdout, '(1x,"cryst.",3x,"s(",i2,") = (",3(i6,5x) &
+           WRITE(stdout, '(1x,"cryst.",3x,"s(",i2,") = (",3(i6,5x), &
                 &                    " )    f =( ",f10.7," )")') isymq,  &
                 & (s(1,ipol,isym), ipol = 1, 3), ft(1,isym)
            WRITE(stdout, '(17x," (",3(i6,5x), &
```

I checked the rest of this subroutine and the equivalent routines in `TDDFPT/src/lr_summary.f90` and
`EPW/src/summaries.f90` for the same pattern -- no other instances found; this appears to be an isolated
copy-paste slip in this one line.

## Verification

- A debug rebuild (`gfortran -O0`, no MPI) reproduced the exact compiler error above, naming this file and line.
- A full rebuild with the one-line fix applied (MPI + HDF5 enabled) took the previously-crashing case to
  completion (`JOB DONE`), producing correct, physically sane phonon frequencies for the test structure.
- Re-ran end-to-end through an AiiDA-driven workflow against the patched binary: the full
  SCF -> DFPT -> force-constants -> dispersion-interpolation chain completes successfully where it previously
  aborted every time.

## Environment

- Quantum ESPRESSO 7.5 (tag `qe-7.5`)
- Reproduced with: `gfortran` 15.2.0, OpenMPI, HDF5-enabled build
- Also reproduced (crash only, not yet re-verified against the fix) with an NVIDIA HPC SDK (`nvfortran`)
  25.7 CUDA-enabled build
```

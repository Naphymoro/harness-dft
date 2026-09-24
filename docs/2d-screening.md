# 2D monolayer stability screening: design, status, and a real bug it caught

## What this is

A pilot for a broader question ("is there a stable 2D monolayer of element X, and which structural prototype is
it?") scoped to 8 elements first: **Al, Ga, In, Tl** (group 13 -- honeycomb prototype) and **P, As, Sb, Bi**
(group 15 -- puckered/black-phosphorus prototype, phosphorene's family), before deciding whether to scale to more
of the periodic table. Full periodic table sweep was explicitly deferred: hundreds of relax+phonon jobs is real
compute (likely days on this one workstation), and the 2D-specific machinery needed to exist and be *proven*
first.

Pseudopotential coverage is **not** a blocker for scaling later: the installed SSSP library covers all 103
practically-relevant elements, H through Lr (checked directly, not assumed).

## New modules

| Module | Purpose |
|---|---|
| `harness_dft.twod` | 2D monolayer prototype generation: `build_honeycomb_monolayer`, `build_puckered_monolayer`, `is_2d_periodic`, `default_2d_kpoints_mesh` |
| `harness_dft.stability` | `check_dynamical_stability`: parses a finished `matdyn` phonon dispersion for imaginary (negative) frequencies |
| `harness_dft.screening` | `rank_prototypes`: compares finished relax/SCF energies-per-atom across candidate structures for the same element |

Plus `builders.apply_cell_dofree` (QE's `cell_dofree` CELL-namelist keyword, e.g. `"2Dxy"`, threaded through
`build_relax_inputs`) and an `asr="simple"` default on `build_matdyn_inputs` (acoustic-sum-rule correction --
reduces small numerical negative-frequency noise at Gamma, seen directly in the bulk-Si phonon validation:
~-0.39 THz before this fix).

MCP tools: `hd_generate_2d_prototype`, `hd_check_phonon_stability`, `hd_rank_prototypes`; `hd_submit_relax` gained
a `cell_dofree` parameter.

## Structure generation: honest about what's seeded vs. what's real

Bond lengths come from **`ase.data.covalent_radii`** (a real, tabulated data source), not literature lattice
constants recalled from memory -- verified directly against `ase.get_distance()` for every pilot element (exact
match to the intended bond length, and exactly 3-fold coordination for the puckered prototype, checked
numerically, not just asserted). The puckered prototype's pucker angle/amplitude is a generic topological seed
(right connectivity/symmetry, not any element's real lattice parameters) -- full relaxation is what determines
the actual geometry, and structure quality of the seed matters far less than getting the topology and
`cell_dofree` handling right.

## A real bug this caught: `npool` divides the wrong number

`choose_resources`' pool-count logic (`estimate.py`) used
`_largest_divisor_at_most(job.n_kpoints, mpiprocs)` -- the largest divisor of **n_kpoints** that's `<= mpiprocs`.
But QE's `-npool N` requires `nproc % N == 0`: **npool must divide the MPI rank count**, not the k-point count.
Every prior real test happened to use a k-point mesh whose total (`4x4x4=64`, a power of two) shared a large
common factor with the 8-rank batch size, masking this. The first real 2D candidate used a `(9,9,1)` mesh --
81 k-points, factors only `3^4` -- and `_largest_divisor_at_most(81, 8) = 3`. `mpiprocs=8, npool=3`: not an even
split. `pw.x` aborted immediately:

```
Error in routine mp_start_pools (1):
invalid number of pools, parent_nproc /= nproc_pool * npool
```

Fixed: `npool = _largest_divisor_at_most(mpiprocs, min(mpiprocs, n_kpoints))` -- largest divisor **of mpiprocs**,
capped (for efficiency, not correctness) at the k-point count. The previous test asserting
`n_kpoints % npool == 0` encoded the same wrong invariant the bug had; replaced with the actual QE constraint
(`mpiprocs % npool == 0`) plus a regression test reproducing the exact 81-kpoint/8-rank failure.

## Real validation: aluminene (Al honeycomb monolayer)

Full pipeline run through the real MCP server against this machine's AiiDA profile:
`hd_generate_2d_prototype` → `hd_estimate` → `hd_submit_relax(cell_dofree="2Dxy")` → `hd_wait_for_job` →
`hd_get_job_results`, then read the relaxed structure directly.

Result: `exit_status=0`, `is_finished_ok=True`. Critically:

- **Vacuum survived**: c-axis `35.99999969 Å`, started at `36.0 Å` -- `cell_dofree="2Dxy"` correctly prevented
  the vc-relax from collapsing or tilting the vacuum direction. This was the entire point of the feature; an
  unconstrained relax would not have preserved this.
- **Real relaxation happened, not a no-op**: in-plane bond length moved from the covalent-radius seed (2.42 Å)
  to the DFT-relaxed value (2.59 Å).
- **Buckling relaxed from the 0.08 Å seed down to 0.025 Å** -- a real result, not seed leakage, though whether
  aluminene is genuinely (near-)planar or has a small stable buckling is exactly the kind of question a phonon
  check (`hd_check_phonon_stability`) resolves, not a single relax at a loose (`"fast"`) protocol.

## Full pilot results (all 8 elements)

Every element below was relaxed via the real MCP pipeline (`hd_generate_2d_prototype` → `hd_estimate` →
`hd_submit_relax(cell_dofree="2Dxy")` → `hd_wait_for_job` → `hd_get_job_results`), `exit_status=0` in every case,
vacuum spacing intact to within `3e-7 Å` of the seed in every case:

| Element | Prototype | pk | Relaxed cell (Å) | Energy/atom (eV) | `force_metal` needed? |
|---|---|---|---|---|---|
| Al | honeycomb | 726 | 4.486 × 4.486 × 36.0 | -536.27 | no |
| Ga | honeycomb | 1215 | 4.255 × 4.255 × 36.0 | -3780.67 | yes (walltime failures without it were actually oversubscription, see below -- but it converged faster with smearing regardless) |
| In | honeycomb | 800 | 4.933 × 4.933 × 36.0 | -1970.00 | no (already on `is_likely_metal`'s hardcoded list) |
| Tl | honeycomb | 1234 | 5.138 × 5.138 × 36.0 | -1971.73 | yes |
| P | puckered | 842 | 4.167 × 4.699 × 18.749 | -190.71 | no |
| As | puckered | 1257 | 3.574 × 5.695 × 18.833 | -247.48 | yes |
| Sb | puckered | 1387 | 4.085 × 6.421 × 18.973 | -2516.23 | yes -- **genuine SCF non-convergence without it** (5 restart attempts, progressively smaller mixing, still failed; converged immediately once `force_metal=True`) |
| Bi | puckered | 1456 | 4.213 × 6.670 × 19.036 | -2520.59 | yes (needed both `force_metal=True` and a manually-extended walltime, 5400s, to avoid a restart-handler bug -- see below) |

**Phonon stability** (`(2,2,1)` q-mesh) was attempted for all 8. Result: **2 of 8 completed** (updated after the
QE bug below was actually fixed, not just diagnosed).

- **As: dynamically UNSTABLE.** `min_frequency_thz=-6.22` at Gamma and `-1.49` at `q=(0.5,0,0)` -- real, well
  beyond the `-0.5 THz` numerical-noise tolerance, and spanning more than one q-point. This puckered arsenene
  candidate, at this level of theory (PBE, `ecutwfc=40 Ry`, this seed geometry), is not a stable phase.
- **Ga: dynamically UNSTABLE.** `min_frequency_thz=-1.06 THz`, identically at **three** q-points
  (`(0,0.5,0)`, `(0.5,0,0)`, `(0.5,0.5,0)`) -- also real, not noise. This buckled-honeycomb gallenene candidate is
  not a stable phase either, at this level of theory.

Both are real, meaningful screening results -- not every candidate should be stable, and the pipeline correctly
identified two that aren't.

### A real QE 7.5 bug, found, root-caused, and fixed (not just diagnosed)

`ph.x` was crashing within seconds on Ga, Tl, In, and Bi's phonon step -- a `libgfortran` I/O crash
(`data_transfer_init`), multiple MPI ranks failing independently, at exactly the point where it prints
point-group symmetry operations for `q=(0,0,0)`. **Root cause, confirmed via a debug rebuild + a real,
human-readable compiler error, not guessed**: a missing comma in a `WRITE` FORMAT string in
`PHonon/PH/phq_summary.f90` (line 189), in the block that only executes for a symmetry operation with a nonzero
fractional translation (a non-symmorphic operation). Every sibling `WRITE` in the same subroutine has the comma;
this one line doesn't. This is why it correlates with `ibrav=0` (what AiiDA always builds): non-symmorphic
operations are far more likely to appear without a recognized Bravais lattice to pick a symmetric origin --
exactly what `pw.x`'s own "using `ibrav=0` with symmetry is DISCOURAGED" warning is about, even though `pw.x`
itself never reaches this buggy phonon-only code path.

**A wrong turn on the way there, corrected**: an early manual reproduction suggested `-northo 0` (disabling
ScaLAPACK) avoided the crash, and the harness briefly shipped a `disable_scalapack` auto-workaround based on that.
Retested against the *exact* real failing input (not a hand-typed approximation), `-northo 0` did **not** avoid
the crash -- that first reproduction just happened to use slightly different SCF parameters that never generated
a non-symmorphic operation. The workaround has been **removed from the harness** (it didn't fix anything, and
shipping it would have been dishonest); the actual fix is the one-line source patch, confirmed by a full rebuild
that took the exact previously-crashing case to `JOB DONE` with correct, physically sane frequencies.

Full patch, bug writeup, and reproduction: `docs/upstream-bugs/qe-phq-summary-missing-comma.patch` and
`docs/upstream-bugs/README.md`. Rebuilt locally (MPI + HDF5, matching this profile's production QE install) and
registered as `ph-7.5-fixed@localhost`. **Not yet submitted to the real QE project** -- the patch is ready, but
filing it on [QEF/q-e](https://gitlab.com/QEF/q-e) under someone's identity needs their explicit go-ahead, not
something to do unilaterally.

With the crash fixed, Ga/Tl/In/Bi's phonon steps run to real DFPT convergence -- but three of the four (Tl, In,
Bi) then hit a **second, separate, pre-existing bug**:

- **Tl, In, Bi: blocked by the `aiida-quantumespresso` restart-handler bug**, not the QE bug above and not a
  harness bug. Once `ph.x` no longer crashes instantly, these take longer than the default ~32-minute walltime
  estimate to actually converge; when the walltime handler fires and restarts, the restart hits
  `aiida_quantumespresso/calculations/ph.py:365`: `parameters['INPUTPH'].get('electron_phonon', ...)` raises
  `KeyError: 'INPUTPH'` -- the restart path constructs new calculation inputs that have silently dropped the
  `INPUTPH` namelist entirely. Same bug already found for P/Sb below.
- **P, Sb: blocked by the same `aiida-quantumespresso` bug.** Both genuinely exceed even a manually-extended
  7200s (2 hour) walltime for a single `PhCalculation` at this q-mesh/cutoff on this hardware -- a real, if
  inconvenient, computational-cost finding (DFPT for these puckered pnictogens is expensive) -- and then hit the
  identical restart-handler bug. Reproduced identically at both the original ~34-minute default walltime and the
  manually-extended 2-hour one, confirming it fires on *any* automatic restart of a `PhCalculation`, regardless
  of walltime budget. This is a genuine upstream library bug (confirmed via full traceback, not inferred);
  avoiding it requires either a coarser/cheaper phonon calculation that finishes in one attempt, or a fix/patch
  to `aiida-quantumespresso` itself, neither attempted here. Ga's own official run succeeded only because it was
  resubmitted standalone (no CPU contention from concurrent jobs) with a manually-extended walltime -- proving
  the same approach would likely work for Tl/In/Bi too, at the cost of more wall-clock time than this pilot spent
  chasing it further.

**Real bugs this pilot caught in the harness itself, all fixed**: the `npool`-divisibility bug (see above); the
electronic-type heuristic's missing override hook (`force_metal`, directly unblocking Sb's real SCF convergence
failure); and the `disable_scalapack` non-fix described above, added then correctly removed once shown not to
work.

## What's proven vs. what's left

**Proven**: structure generation (all 8 elements, numerically verified geometry), `cell_dofree="2Dxy"` vacuum
preservation (all 8, to 7 significant figures), the resource estimator including the `npool` fix, the
`force_metal` override (directly fixed Sb's real convergence failure), **the QE 7.5 `ph.x` symmetry crash found,
root-caused to an exact source line, patched, rebuilt, and verified fixed** (not just diagnosed), and two complete
generate→relax→phonon→stability verdicts (As and Ga: both correctly identified as unstable).

**Not resolved, and out of scope for further iteration in this pilot**: the `aiida-quantumespresso`
`PhBaseWorkChain` restart-handler `INPUTPH` bug (blocks Tl, In, Bi, P, Sb -- 5 of 8, once the QE crash above
stopped being the thing blocking them first). This is a real, root-caused, reproducible bug in a different
project's codebase, not a harness logic error and not the bug this session was asked to fix -- a genuine
follow-up (an `aiida-quantumespresso` patch, or restructuring these calculations to reliably finish inside one
walltime budget) rather than something to paper over with more resource-parameter tuning. The QE patch itself
is written and verified but **not yet submitted upstream** (see `docs/upstream-bugs/README.md`) -- filing it on
the real QE project needs the user's go-ahead, not something done unilaterally.

`hd_rank_prototypes` was only exercised as a mechanical sort-order check (two arbitrary bulk-Si SCF nodes), never
on a real "which prototype wins" comparison -- with only As and Ga having complete stability verdicts among the
8, and no element having *two* prototypes both taken to completion, there is nothing to rank yet in this pilot.

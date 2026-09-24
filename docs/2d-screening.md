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

**Phonon stability** (`(2,2,1)` q-mesh) was attempted for all 8. Result: **1 of 8 completed**.

- **As: dynamically UNSTABLE.** `min_frequency_thz=-6.22` at Gamma and `-1.49` at `q=(0.5,0,0)` -- real, well
  beyond the `-0.5 THz` numerical-noise tolerance, and spanning more than one q-point. This puckered arsenene
  candidate, at this level of theory (PBE, `ecutwfc=40 Ry`, this seed geometry), is not a stable phase. This is
  itself a real, meaningful screening result -- not every candidate should be stable, and the pipeline correctly
  identified one that isn't.
- **Ga, Tl, In, Bi: blocked by a real QE 7.5 crash**, not a harness bug. `ph.x` crashes within seconds at exactly
  the point where it prints/processes point-group symmetry operations for `q=(0,0,0)` -- a Fortran runtime I/O
  crash (`libgfortran/io/transfer.c: data_transfer_init`), multiple MPI ranks failing independently. Confirmed
  **not** caused by: MPI rank count (reproduced identically at `mpiprocs=8` and `mpiprocs=2`), k-point pool count
  (ph.x doesn't even receive an `-npool` flag in this workflow), or symmetry use itself (`nosym=true` in the
  `INPUTPH` namelist did not avoid it). Correlates with the number of symmetry operations at the crash point (Ga:
  13 sym ops, Bi: 9), not with honeycomb-vs-puckered topology (Bi is puckered and still crashes; As is puckered
  and doesn't). AiiDA always builds QE's `CELL_PARAMETERS`/`ibrav=0` (a generic cell, never a symmetry-aware
  `ibrav`) -- QE's own `pw.x` output for every one of these structures includes the warning `using ibrav=0 with
  symmetry is DISCOURAGED, use correct ibrav instead`, which `pw.x` tolerates but `ph.x`'s stricter symmetry
  machinery apparently does not, for high-enough symmetry counts. This needs either an upstream QE fix or setting
  an explicit symmetry-matched `ibrav` for these structures (not attempted here -- real scope beyond this pilot).
- **P, Sb: blocked by a real `aiida-quantumespresso` bug**, also not a harness bug. Both genuinely exceed even a
  manually-extended 7200s (2 hour) walltime for a single `PhCalculation` at this q-mesh/cutoff on this hardware --
  a real, if inconvenient, computational-cost finding (DFPT for these puckered pnictogens is expensive). When the
  walltime handler fires and restarts, the restart hits `aiida_quantumespresso/calculations/ph.py:365`:
  `parameters['INPUTPH'].get('electron_phonon', ...)` raises `KeyError: 'INPUTPH'` -- the restart path constructs
  new calculation inputs that have silently dropped the `INPUTPH` namelist entirely. Reproduced identically at
  both the original ~34-minute default walltime and the manually-extended 2-hour one. This is a genuine upstream
  library bug (confirmed via full traceback, not inferred) that fires on *any* automatic restart of a
  `PhCalculation`, regardless of walltime budget -- avoiding it requires either a coarser/cheaper phonon
  calculation that finishes in one shot, or a fix/patch to `aiida-quantumespresso` itself, neither attempted here.

**Two more real bugs this pilot caught in the harness itself, both fixed** (see "Real bugs found" above for the
first: `npool`): the electronic-type heuristic had no override hook (`force_metal` added, directly unblocking
Sb's real convergence failure); nothing else new during the full-pilot run beyond what's documented above.

## What's proven vs. what's left

**Proven**: structure generation (all 8 elements, numerically verified geometry), `cell_dofree="2Dxy"` vacuum
preservation (all 8, to 7 significant figures), the resource estimator including the `npool` fix, the
`force_metal` override (directly fixed Sb's real convergence failure), and one complete
generate→relax→phonon→stability verdict (As: correctly identified as unstable).

**Not resolved, and out of scope for further iteration in this pilot**: the QE 7.5 `ph.x` symmetry-count crash
(4 elements blocked), and the `aiida-quantumespresso` restart-handler `INPUTPH` bug (2 elements blocked). Both
are real, root-caused, reproducible external bugs, not harness logic errors -- fixing either is a genuine
follow-up project (upstream QE investigation / an `ibrav` fix for the former; an `aiida-quantumespresso` patch or
workaround for the latter), not something to paper over with more resource-parameter tuning.

`hd_rank_prototypes` was only exercised as a mechanical sort-order check (two arbitrary bulk-Si SCF nodes), never
on a real "which prototype wins" comparison -- with only As having a complete stability verdict among the 8, and
no element having *two* prototypes both taken to completion, there is nothing to rank yet in this pilot.

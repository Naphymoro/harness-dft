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

## What's proven vs. what's left of the pilot

**Proven** (this element, this run): structure generation, `cell_dofree` vacuum preservation, the resource
estimator (including the npool fix), and the full generate→relax MCP pipeline.

**Not yet run**: the phonon chain and `hd_check_phonon_stability` on any 2D candidate (only exercised against
bulk Si so far); the puckered prototype through a real relax (only geometry-checked, not DFT-relaxed); the
remaining 7 pilot elements; `hd_rank_prototypes` comparing honeycomb vs. puckered for the same element (only
exercised with two arbitrary bulk-Si SCF nodes as a mechanical sort-order check, not a real "which prototype
wins" comparison).

A real phonon check needs more than the Gamma-only q-mesh used for the bulk-Si validation (a single q-point
cannot catch instabilities away from Gamma) -- a 2D q-mesh like `(4,4,1)` is a reasonable next step, at real
additional compute cost (the phonon chain already took several minutes per q-point on bulk Si).

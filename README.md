# harness-dft

A resource-adaptive DFT orchestration harness built on **Quantum ESPRESSO**, **AiiDA** (via `aiida-quantumespresso`), and **ASE** — drivable directly (CLI/Python) or through **DeerFlow** via the `harness-dft-mcp` server and `dft-harness` skill.

This is not another DFT engine — it's an automation layer. Quantum ESPRESSO does the physics, AiiDA drives it with provenance and remote HPC submission, and ASE handles structure building/analysis and the cases that don't need full provenance. `src/harness_dft/` adds one thing on top: an estimate of how much compute a job actually needs, applied automatically to mpiprocs/npool/walltime, so calculations don't have to be hand-tuned for every structure. CPU allocations are quantized to whole batches of `cpu_batch_size` CPUs (default 8, matching typical socket/NUMA-node scheduling granularity); GPU allocations use one MPI rank per GPU when a CUDA-built QE code is available and the job warrants it.

(This is a separate project from `lengau_v221`/AtomX in this workspace, which is an independent from-scratch DFT implementation — harness-dft wraps the real QE package instead.)

## Architecture

- **ASE** owns structure building/analysis and any calculation you want fast and local without provenance (see `workflows/neb.py`).
- **AiiDA workchains** own anything needing provenance, automatic error-handling/restart, or remote execution. The harness rarely calls `pw.x` directly — it builds AiiDA workchain inputs and lets `PwBaseWorkChain`'s existing error handlers (out-of-walltime, diagonalization errors, convergence failures) do the retry logic.
- **`harness_dft.estimate`** fills the one real gap in both ecosystems: no built-in atom-count → resource estimator exists in AiiDA or QE. `harness_dft.builders.apply_resource_plan` uses it to set `metadata.options` (mpiprocs, walltime) and `parallelization` (npool) on every workchain builder, and to *recommend* local vs. local-GPU vs. remote routing (`harness_dft.estimate.choose_resources`) — though remote execution itself needs a configured remote AiiDA computer (see `dft-submit-remote` skill; no specific cluster is pre-wired), and GPU execution needs a CUDA-built QE code registered (see "GPU build" below).
- **`harness_dft.jobs`** is the submit-and-poll layer every `hd_submit_*` MCP tool uses: `aiida.engine.submit` + pk, rather than the blocking `run_get_node` the CLI/examples use, since an MCP tool call can't block for a multi-minute DFT run. Needs `verdi daemon start`.
- **`mcp_server/`** exposes all of this to an agent harness (DeerFlow) as typed MCP tools; see `mcp_server/README.md` and the `deer-flow/skills/public/dft-harness/` skill. Its first rule: ask whether to run locally or on remote HPC before planning anything, rather than assuming.

## Setup

Environment already built as a conda env (`dft-harness`, via `micromamba`) alongside a pre-existing Quantum ESPRESSO 7.5 install:

```bash
ENV=~/.local/share/cyranoid/conda/envs/dft-harness
export PATH=$ENV/bin:$PATH

# AiiDA profile (SQLite + ZeroMQ, no Postgres/RabbitMQ needed)
verdi presto --profile-name dft-harness

# Register QE binaries as AiiDA codes (adjust paths if QE lives elsewhere)
verdi code create core.code.installed -L pw-7.5 -Y localhost -X ~/.local/bin/pw.x -P quantumespresso.pw --with-mpi --non-interactive
# ...repeat for dos.x/projwfc.x/ph.x/q2r.x/matdyn.x/neb.x with matching plugin entry points

# Pseudopotentials -- the full SSSP 1.3 library (both functionals used across the QE/AiiDA
# ecosystem, at both accuracy protocols; get_builder_from_protocol()'s own fast/moderate/precise
# protocols default to PBEsol, so PBE-only coverage is not enough even for PBE-family work).
aiida-pseudo install sssp --version 1.3 --functional PBE --protocol efficiency
aiida-pseudo install sssp --version 1.3 --functional PBE --protocol precision
aiida-pseudo install sssp --version 1.3 --functional PBEsol --protocol efficiency
aiida-pseudo install sssp --version 1.3 --functional PBEsol --protocol precision
```

Install the harness package itself:

```bash
$ENV/bin/pip install -e .
```

Note: `aiida-quantumespresso` 4.17.0 needs `qe-tools` >= 2.3.0 for stress-tensor parsing, but conda-forge only ships 2.0.0 as of this writing — the install pulls the newer version via pip:

```bash
$ENV/bin/pip install --upgrade "qe-tools==2.3.0"
```

## Verify it works

```bash
AIIDA_PROFILE=dft-harness PYTHONPATH=src $ENV/bin/python examples/smoke_test_si.py
```

Relaxes bulk Si and prints the equilibrium lattice parameter — should land close to 5.43-5.48 Å depending on protocol/cutoff.

## Modules

| Module | Purpose |
|---|---|
| `environment.py` | Detect local CPU/RAM/GPU (name, count, VRAM) |
| `estimate.py` | Heuristic resource estimation + local/local-GPU/remote routing decision, CPU-batch quantization |
| `structures.py` | ASE <-> AiiDA structure conversion, electron counting |
| `pseudos.py` | Pseudopotential family validation, full-library constant, installed-family listing |
| `builders.py` | Shared logic: apply calculation settings (ecutwfc/kpoints) + resource plan to any PwBaseWorkChain-shaped builder |
| `remote.py` | Generic (cluster-agnostic) SSH computer/code setup |
| `jobs.py` | Submit-and-poll layer (`aiida.engine.submit` + pk) for callers that can't block for a full run, e.g. the MCP server |
| `workflows/relax.py` | `PwRelaxWorkChain` wrapper |
| `workflows/converge.py` | k-point / ecutwfc convergence sweeps |
| `workflows/eos.py` | Volume scan + ASE `EquationOfState` fit |
| `workflows/bands_dos.py` | `PwBandsWorkChain` + `PdosWorkChain` wrappers |
| `workflows/phonons.py` | `PhBaseWorkChain` -> `Q2rBaseWorkChain` -> `MatdynBaseWorkChain` pipeline |
| `workflows/neb.py` | ASE-native NEB with QE as the force engine (no AiiDA provenance) |
| `mcp_server/` | `harness-dft-mcp`: exposes the above to DeerFlow as MCP tools (`hd_*`) |

## Claude Code Skills

`.claude/skills/` has one playbook per workflow (`dft-relax`, `dft-converge`, `dft-bands-dos`, `dft-eos`, `dft-phonons`, `dft-neb`, `dft-pseudo-select`, `dft-submit-remote`) — each documents prerequisites, exact API usage, and gotchas actually hit while building this (e.g. the ecutwfc-override bug, the PBE-vs-PBEsol pseudo family mismatch). Read the relevant one before using a workflow module for the first time in a session.

For driving this through DeerFlow instead, see `deer-flow/skills/public/dft-harness/` and `mcp_server/README.md`.

## GPU build

QE-GPU needs `nvfortran`/CUDA Fortran, which is **not available via conda-forge or the `nvidia` conda channel**
(checked directly — no `nvhpc`/`nvfortran`/`hpc-sdk` package exists there as of this writing). It's built here
instead from NVIDIA's own HPC SDK tarball installer (self-contained, installs into a user-writable prefix, no
sudo needed), against QE's CMake build with `-DQE_ENABLE_CUDA=ON -DQE_GPU_ARCHS=sm_89 -DQE_ENABLE_OPENMP=OFF`
(the last flag avoids a real GNU-libgomp/NVHPC-OpenMP runtime clash hit during the build). `make all` from that
same build tree produced the entire QE suite; every code this harness has workflow support for is registered
GPU-side: `pw-7.5-gpu`, `ph-7.5-gpu`, `dos-7.5-gpu`, `projwfc-7.5-gpu`, `q2r-7.5-gpu`, `matdyn-7.5-gpu`,
`neb-7.5-gpu` (the last unused by any current workflow — see below), each distinct from its CPU `*-7.5@localhost`
counterpart. See `docs/gpu-build.md` for the full runbook, all four build/integration bugs hit and fixed, and
validation results: GPU vs. CPU SCF energy for bulk Si agrees to 9 significant figures, and the full phonon chain
(`ph`→`q2r`→`matdyn`) and PDOS both ran to completion on GPU codes through the harness's own automatic
`local-gpu` routing. Built and validated on one GPU architecture (Ada Lovelace, compute capability 8.9) and one
small test structure — treat a different architecture, a larger system, or `hd_submit_relax`/`hd_submit_bands` on
GPU (registered and code-path-identical, but not separately re-run after the routing fix) as unvalidated until
tried.

## What's verified vs. not

Live-tested against a real `pw.x` 7.5 run: `relax.py`, `converge.py` (both ecutwfc and k-point sweeps, via shared `eos.py` SCF builder). Also live-tested: `jobs.py`'s submit/poll/results round trip (including its recursive handling of namespaced outputs, e.g. PdosWorkChain's `dos.*`/`projwfc.*`) and the `harness-dft-mcp` server's read-only tools + `hd_submit_scf`/`hd_submit_ph`/`hd_submit_q2r`/`hd_submit_matdyn`/`hd_submit_pdos` on both CPU and GPU codes, against this machine's real AiiDA profile (see `mcp_server/README.md`'s "Verified vs. not" for the exact list). Structurally verified against installed `aiida-quantumespresso` 4.17.0 source but **not run live**: `eos.py`'s multi-point volume scan, `neb.py`, `remote.py` (no real SSH target), and `hd_submit_relax`/`hd_submit_bands` specifically on a GPU code (the CPU path and the analogous `hd_submit_scf`/`hd_submit_ph` GPU paths are both live-tested, but these two combinations weren't separately re-run). Treat first real use of the untested combinations as validation, not a known-good path.

## Not yet wired up

Remote HPC execution has no specific cluster configured — `remote.py`/`dft-submit-remote` are generic scaffolding, deliberately not tied to any particular cluster's hostname/scheduler/module system. GPU QE covers every code this harness has workflow support for, on one GPU architecture so far (see `docs/gpu-build.md`); `neb.x`, `cp.x`, `pwcond.x`, `epw.x`, `hp.x`, `xspectra.x` and the rest of the wider QE suite were built (via `make all`, they came along for free) but are not registered as AiiDA codes since nothing in this harness drives them yet.

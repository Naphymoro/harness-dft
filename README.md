# harness-dft

A resource-adaptive DFT orchestration harness built on **Quantum ESPRESSO**, **AiiDA** (via `aiida-quantumespresso`), and **ASE**.

This is not another DFT engine — it's an automation layer. Quantum ESPRESSO does the physics, AiiDA drives it with provenance and (eventually) remote HPC submission, and ASE handles structure building/analysis and the cases that don't need full provenance. `src/harness_dft/` adds one thing on top: an estimate of how much compute a job actually needs, applied automatically to mpiprocs/npool/walltime, so calculations don't have to be hand-tuned for every structure.

(This is a separate project from `lengau_v221`/AtomX in this workspace, which is an independent from-scratch DFT implementation — harness-dft wraps the real QE package instead.)

## Architecture

- **ASE** owns structure building/analysis and any calculation you want fast and local without provenance (see `workflows/neb.py`).
- **AiiDA workchains** own anything needing provenance, automatic error-handling/restart, or (future) remote execution. The harness rarely calls `pw.x` directly — it builds AiiDA workchain inputs and lets `PwBaseWorkChain`'s existing error handlers (out-of-walltime, diagonalization errors, convergence failures) do the retry logic.
- **`harness_dft.estimate`** fills the one real gap in both ecosystems: no built-in atom-count → resource estimator exists in AiiDA or QE. `harness_dft.builders.apply_resource_plan` uses it to set `metadata.options` (mpiprocs, walltime) and `parallelization` (npool) on every workchain builder, and to *recommend* local vs. remote routing (`harness_dft.estimate.choose_resources`) — though remote execution itself needs a configured remote AiiDA computer (see `dft-submit-remote` skill; no specific cluster is pre-wired).

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

# Pseudopotentials
aiida-pseudo install sssp --version 1.3 --functional PBE --protocol efficiency
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
| `environment.py` | Detect local CPU/RAM/GPU |
| `estimate.py` | Heuristic resource estimation + local/remote routing decision |
| `structures.py` | ASE <-> AiiDA structure conversion, electron counting |
| `pseudos.py` | Pseudopotential family validation |
| `builders.py` | Shared logic: apply calculation settings (ecutwfc/kpoints) + resource plan to any PwBaseWorkChain-shaped builder |
| `remote.py` | Generic (cluster-agnostic) SSH computer/code setup |
| `workflows/relax.py` | `PwRelaxWorkChain` wrapper |
| `workflows/converge.py` | k-point / ecutwfc convergence sweeps |
| `workflows/eos.py` | Volume scan + ASE `EquationOfState` fit |
| `workflows/bands_dos.py` | `PwBandsWorkChain` + `PdosWorkChain` wrappers |
| `workflows/phonons.py` | `PhBaseWorkChain` -> `Q2rBaseWorkChain` -> `MatdynBaseWorkChain` pipeline |
| `workflows/neb.py` | ASE-native NEB with QE as the force engine (no AiiDA provenance) |

## Claude Code Skills

`.claude/skills/` has one playbook per workflow (`dft-relax`, `dft-converge`, `dft-bands-dos`, `dft-eos`, `dft-phonons`, `dft-neb`, `dft-pseudo-select`, `dft-submit-remote`) — each documents prerequisites, exact API usage, and gotchas actually hit while building this (e.g. the ecutwfc-override bug, the PBE-vs-PBEsol pseudo family mismatch). Read the relevant one before using a workflow module for the first time in a session.

## What's verified vs. not

Live-tested against a real `pw.x` 7.5 run: `relax.py`, `converge.py` (both ecutwfc and k-point sweeps, via shared `eos.py` SCF builder). Structurally verified against installed `aiida-quantumespresso` 4.17.0 source but **not run live**: `bands_dos.py`, `phonons.py`, `eos.py`'s multi-point volume scan, `neb.py`, `remote.py` (no real SSH target). Treat first real use of the untested modules as validation, not a known-good path.

## Not yet wired up

Remote HPC execution has no specific cluster configured — `remote.py`/`dft-submit-remote` are generic scaffolding, deliberately not tied to any particular cluster's hostname/scheduler/module system.

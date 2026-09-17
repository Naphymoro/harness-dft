"""Heuristic resource estimation for plane-wave DFT jobs.

Nothing here claims exactness — QE's real memory/PW-count depends on the FFT
grid and exact G-vector sphere. These are order-of-magnitude estimates meant
to pick a *starting* resource allocation (mpiprocs, npool, walltime). AiiDA's
`PwBaseWorkChain` already handles the fine correction: it automatically
resubmits with more walltime/resources on `handle_out_of_walltime`,
`handle_diagonalization_errors`, etc. The harness's job is to not start
absurdly wrong, not to predict exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from harness_dft.environment import LocalResources

BYTES_PER_COMPLEX = 16  # complex128
SAFETY_FACTOR = 3.0  # wavefunctions + charge density + workspace, very rough


@dataclass(frozen=True)
class JobEstimate:
    n_atoms: int
    n_electrons: float
    n_bands: int
    n_kpoints: int
    estimated_pw_per_kpoint: int
    estimated_memory_gb: float


@dataclass(frozen=True)
class ExecutionPlan:
    target: str  # "local" or "remote"
    mpiprocs: int
    npool: int
    walltime_seconds: int
    reason: str


def estimate_plane_waves(cell_volume_ang3: float, ecutwfc_ry: float) -> int:
    """Standard plane-wave counting estimate: N_pw ~ V * Ecut^1.5 / (6*pi^2),
    with V in bohr^3 and Ecut in Rydberg atomic units."""
    bohr_to_ang = 0.52917720859
    volume_bohr3 = cell_volume_ang3 / (bohr_to_ang**3)
    return max(1, round(volume_bohr3 * ecutwfc_ry**1.5 / (6 * math.pi**2)))


def estimate_bands(n_electrons: float, is_metal: bool) -> int:
    """Occupied bands plus a safety margin of empty bands for convergence."""
    occupied = n_electrons / 2
    buffer_factor = 1.3 if is_metal else 1.15
    extra_bands = 6 if is_metal else 4
    return max(4, math.ceil(occupied * buffer_factor) + extra_bands)


def estimate_job(
    n_atoms: int,
    n_electrons: float,
    cell_volume_ang3: float,
    ecutwfc_ry: float,
    n_kpoints: int,
    is_metal: bool = False,
) -> JobEstimate:
    n_bands = estimate_bands(n_electrons, is_metal)
    pw_per_kpoint = estimate_plane_waves(cell_volume_ang3, ecutwfc_ry)
    memory_bytes = n_bands * n_kpoints * pw_per_kpoint * BYTES_PER_COMPLEX * SAFETY_FACTOR
    return JobEstimate(
        n_atoms=n_atoms,
        n_electrons=n_electrons,
        n_bands=n_bands,
        n_kpoints=n_kpoints,
        estimated_pw_per_kpoint=pw_per_kpoint,
        estimated_memory_gb=round(memory_bytes / 1e9, 2),
    )


def _largest_divisor_at_most(n: int, limit: int) -> int:
    """Largest divisor of n that is <= limit (falls back to 1)."""
    for candidate in range(min(n, limit), 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def choose_resources(
    job: JobEstimate,
    local: LocalResources,
    allow_remote: bool = False,
    local_atom_ceiling: int = 40,
) -> ExecutionPlan:
    """Pick an initial mpiprocs/npool/walltime and decide local vs remote.

    Routing rule: prefer local unless the job's estimated memory exceeds a
    safe fraction of local RAM, or the atom count passes a configurable
    ceiling meant to keep local runs fast for interactive use. Remote is only
    chosen if `allow_remote` is True and a remote computer/code has actually
    been configured by the caller -- this function doesn't know whether one
    exists, it only recommends.
    """
    memory_headroom_gb = local.memory_gb * 0.7
    exceeds_memory = job.estimated_memory_gb > memory_headroom_gb
    exceeds_atom_ceiling = job.n_atoms > local_atom_ceiling

    if allow_remote and (exceeds_memory or exceeds_atom_ceiling):
        target = "remote"
        reason = (
            f"estimated {job.estimated_memory_gb} GB / {job.n_atoms} atoms "
            f"exceeds local comfort threshold (RAM headroom {memory_headroom_gb:.1f} GB, "
            f"atom ceiling {local_atom_ceiling})"
        )
        mpiprocs = min(job.n_kpoints * 4, 128)  # generic remote default; refine per-cluster
    else:
        target = "local"
        reason = "fits within local resource budget" if not (exceeds_memory or exceeds_atom_ceiling) else (
            "exceeds local comfort threshold but no remote target configured; running local anyway"
        )
        mpiprocs = min(local.cpu_count, max(1, job.n_atoms))

    npool = _largest_divisor_at_most(job.n_kpoints, mpiprocs) if job.n_kpoints > 1 else 1

    # Coarse walltime starting point; AiiDA's out-of-walltime handler resubmits
    # with more time if this is too low, so err on the short side for local dev.
    base_seconds = 1800
    walltime = base_seconds + job.n_atoms * 60

    return ExecutionPlan(
        target=target,
        mpiprocs=mpiprocs,
        npool=npool,
        walltime_seconds=walltime,
        reason=reason,
    )

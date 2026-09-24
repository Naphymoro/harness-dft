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
    target: str  # "local", "local-gpu", or "remote"
    mpiprocs: int
    npool: int
    walltime_seconds: int
    reason: str
    cpu_batch_size: int = 8


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


def _quantize_mpiprocs(raw: int, batch_size: int, cpu_ceiling: int) -> int:
    """Snap an mpiprocs estimate to a whole number of `batch_size`-CPU
    batches (default 8, a common socket/NUMA-node granularity), rounding
    *up* so small jobs still get one full batch's worth of ranks -- QE's
    per-rank overhead makes single-digit rank counts inefficient anyway --
    but never past what the machine (or `cpu_ceiling`) actually has.

    A machine/allocation with fewer CPUs than one batch just gets all of
    them; batching only kicks in once there's at least one full batch to give.
    """
    if cpu_ceiling < batch_size:
        return max(1, cpu_ceiling)
    batches_available = cpu_ceiling // batch_size
    batches_wanted = max(1, math.ceil(raw / batch_size))
    return min(batches_wanted, batches_available) * batch_size


def choose_resources(
    job: JobEstimate,
    local: LocalResources,
    allow_remote: bool = False,
    allow_gpu: bool = False,
    local_atom_ceiling: int = 40,
    cpu_batch_size: int = 8,
    remote_mpiprocs_ceiling: int = 128,
    gpu_min_atoms: int = 8,
) -> ExecutionPlan:
    """Pick an initial mpiprocs/npool/walltime and decide local vs local-gpu
    vs remote.

    Routing rule: prefer local unless the job's estimated memory exceeds a
    safe fraction of local RAM, or the atom count passes a configurable
    ceiling meant to keep local runs fast for interactive use. Remote is only
    chosen if `allow_remote` is True and a remote computer/code has actually
    been configured by the caller -- this function doesn't know whether one
    exists, it only recommends. `allow_gpu` similarly only recommends
    "local-gpu"; the caller must have an actual CUDA-built QE code registered
    to act on it (see `harness_dft.remote`/the `dft-harness` MCP tools).

    CPU-target mpiprocs (`local` and `remote`) are quantized to whole
    `cpu_batch_size`-CPU batches -- resource allocation in increments of a
    socket/NUMA node, not single cores, is both what real clusters schedule
    in and what keeps QE's MPI communication pattern efficient. GPU targets
    use a different rule entirely: one MPI rank per GPU is QE-GPU's supported
    parallelization model, so batching by CPU count does not apply there.
    """
    memory_headroom_gb = local.memory_gb * 0.7
    exceeds_memory = job.estimated_memory_gb > memory_headroom_gb
    exceeds_atom_ceiling = job.n_atoms > local_atom_ceiling
    fits_locally = not (exceeds_memory or exceeds_atom_ceiling)

    gpu_memory_headroom_gb = local.gpu_memory_gb * 0.7
    exceeds_gpu_memory = local.gpu_memory_gb > 0 and job.estimated_memory_gb > gpu_memory_headroom_gb

    if allow_remote and (exceeds_memory or exceeds_atom_ceiling):
        target = "remote"
        reason = (
            f"estimated {job.estimated_memory_gb} GB / {job.n_atoms} atoms "
            f"exceeds local comfort threshold (RAM headroom {memory_headroom_gb:.1f} GB, "
            f"atom ceiling {local_atom_ceiling})"
        )
        raw_remote = min(job.n_kpoints * 4, remote_mpiprocs_ceiling)  # generic remote default; refine per-cluster
        mpiprocs = _quantize_mpiprocs(raw_remote, cpu_batch_size, remote_mpiprocs_ceiling)
    elif allow_gpu and local.gpu_count > 0 and fits_locally and not exceeds_gpu_memory and job.n_atoms >= gpu_min_atoms:
        target = "local-gpu"
        mpiprocs = local.gpu_count  # one MPI rank per GPU -- QE-GPU's supported model, not a CPU-core count
        reason = (
            f"GPU offload available ({local.gpu_count}x {local.gpu_name or 'GPU'}); "
            f"{job.n_atoms} atoms clears the {gpu_min_atoms}-atom floor where kernel-launch "
            "overhead starts paying off, and the job fits GPU VRAM"
        )
    else:
        target = "local"
        raw_local = min(local.cpu_count, max(1, job.n_atoms))
        mpiprocs = _quantize_mpiprocs(raw_local, cpu_batch_size, local.cpu_count)
        if fits_locally:
            reason = "fits within local resource budget"
        elif allow_gpu and local.gpu_count > 0:
            reason = "exceeds local comfort threshold; GPU offload rejected (too small or exceeds GPU VRAM), running local CPU"
        else:
            reason = "exceeds local comfort threshold but no remote target configured; running local anyway"

    # QE's `-npool N` requires nproc % N == 0 (each pool gets an equal share
    # of the MPI ranks) -- npool must divide mpiprocs, NOT n_kpoints. Getting
    # this backwards (a divisor of n_kpoints capped at mpiprocs) crashes QE
    # with "invalid number of pools, parent_nproc /= nproc_pool * npool" the
    # moment n_kpoints and mpiprocs don't happen to share a large common
    # factor -- every prior test structure's mesh total (a power of two)
    # happened to divide evenly into an 8-rank batch, masking this until a
    # 9x9x1 mesh (81 kpoints, all-odd factors) hit it for real. Capping the
    # search at n_kpoints (not mpiprocs) is just an efficiency choice --
    # a pool with zero kpoints assigned is wasted, not incorrect -- so it's
    # fine that the cap and the divisibility target are different numbers.
    npool = _largest_divisor_at_most(mpiprocs, min(mpiprocs, job.n_kpoints)) if job.n_kpoints > 1 else 1

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
        cpu_batch_size=cpu_batch_size,
    )

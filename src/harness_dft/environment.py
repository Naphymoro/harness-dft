"""Detection of locally available compute resources."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalResources:
    cpu_count: int
    memory_gb: float
    gpu_count: int
    gpu_name: str | None = None
    gpu_memory_gb: float = 0.0  # per-GPU VRAM, assumes a homogeneous set
    gpu_compute_capable: bool = False  # True once a CUDA-enabled QE code is registered for this machine


def _query_gpus() -> list[tuple[str, float]]:
    """Return [(name, memory_gb), ...] for each visible NVIDIA GPU, or []
    if nvidia-smi is absent/fails (no GPU, or driver not installed)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        name, _, mib = line.rpartition(",")
        try:
            gpus.append((name.strip(), round(float(mib.strip()) / 1024, 1)))
        except ValueError:
            continue
    return gpus


def get_local_resources() -> LocalResources:
    """Detect CPU count, total RAM (GB), and GPU count/name/VRAM on this machine.

    `gpu_compute_capable` is always False here -- it only means "a GPU is
    visible to the driver", not "a CUDA-built QE code is registered in AiiDA".
    Callers that need to route to GPU execution should check for a registered
    GPU code (see `harness_dft.pseudos`/CLI `resources --codes`) as well."""
    cpu_count = os.cpu_count() or 1

    memory_gb = 0.0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    memory_gb = kib / (1024 * 1024)
                    break
    except FileNotFoundError:
        import psutil  # fallback for non-Linux
        memory_gb = psutil.virtual_memory().total / (1024**3)

    gpus = _query_gpus()
    gpu_name = gpus[0][0] if gpus else None
    gpu_memory_gb = gpus[0][1] if gpus else 0.0

    return LocalResources(
        cpu_count=cpu_count,
        memory_gb=round(memory_gb, 1),
        gpu_count=len(gpus),
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )

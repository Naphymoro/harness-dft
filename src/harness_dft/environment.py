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


def _detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return len(lines)


def get_local_resources() -> LocalResources:
    """Detect CPU count, total RAM (GB), and GPU count on this machine."""
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

    return LocalResources(
        cpu_count=cpu_count,
        memory_gb=round(memory_gb, 1),
        gpu_count=_detect_gpu_count(),
    )

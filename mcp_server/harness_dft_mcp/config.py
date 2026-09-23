"""Runtime settings, read once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    aiida_profile: str | None = None  # None -> AiiDA's own default-profile resolution
    default_cpu_batch_size: int = 8
    default_local_atom_ceiling: int = 40
    default_gpu_min_atoms: int = 8
    max_wait_seconds: int = 120

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        return cls(
            aiida_profile=env.get("AIIDA_PROFILE") or None,
            default_cpu_batch_size=int(env.get("HARNESS_DFT_CPU_BATCH_SIZE", cls.default_cpu_batch_size)),
            default_local_atom_ceiling=int(env.get("HARNESS_DFT_LOCAL_ATOM_CEILING", cls.default_local_atom_ceiling)),
            default_gpu_min_atoms=int(env.get("HARNESS_DFT_GPU_MIN_ATOMS", cls.default_gpu_min_atoms)),
            max_wait_seconds=int(env.get("HARNESS_DFT_MCP_MAX_WAIT_SECONDS", cls.max_wait_seconds)),
        )

---
name: dft-submit-remote
description: Configure a remote HPC computer (SSH + Slurm) and QE codes in AiiDA, for jobs the resource estimator routes away from local execution. Use when a job is too large for local resources, or the user explicitly wants to set up/verify a remote cluster target.
---

# Remote HPC computer + code setup

Generic, cluster-agnostic helpers in `harness_dft.remote` — no specific cluster's hostname/scheduler details are hardcoded. You (or the user) supply those per-cluster.

## Setup

```python
from harness_dft.remote import RemoteComputerSpec, setup_remote_computer, register_remote_code

spec = RemoteComputerSpec(
    label="my-cluster",
    hostname="login.example-hpc.ac.za",
    username="myusername",
    scheduler="slurm",
    work_dir="/scratch/{username}/aiida_run",
    mpiprocs_per_machine=24,
    prepend_text="module load quantum-espresso/7.2\n",
)
computer = setup_remote_computer(spec)

pw_code = register_remote_code(
    computer_label="my-cluster",
    label="pw-remote",
    remote_executable_path="/apps/qe/7.2/bin/pw.x",  # confirm with `which pw.x` after module load
    plugin_entry_point="quantumespresso.pw",
    prepend_text=spec.prepend_text,
)
```

## Before running this

1. **Verify key-based SSH auth works first**, outside AiiDA: `ssh <username>@<hostname> echo ok`. `setup_remote_computer` assumes this already works (`look_for_keys=True`, `allow_agent=True`) — it does not set up SSH keys for you.
2. Confirm the scheduler type (`slurm` is assumed by default — check `sinfo`/`squeue` availability on the cluster, or ask the user).
3. Get the exact remote executable path and any module-load commands needed (`prepend_text`) — these are cluster-specific and the harness has no way to guess them. Ask the user if unknown, or have them check with `ssh <cluster> 'module load <qe-module> && which pw.x'`.
4. After setup, sanity-check with `verdi computer test <label>` before submitting real work.

## Routing jobs here

The resource estimator (`harness_dft.estimate.choose_resources`) only *recommends* `target="remote"` when a job's estimated memory/atom-count exceeds the local comfort threshold and `allow_remote=True` was passed. It does not automatically pick a remote code — the caller must load whichever code (local or remote) matches the recommendation:

```python
plan = ...  # from apply_resource_plan / choose_resources
code_label = "pw-remote@my-cluster" if plan.target == "remote" else "pw-7.5@localhost"
```

## Known gotchas

- **Entirely unverified against a real cluster** — this was built and reviewed against the installed `aiida-core` SSH transport source (parameter names for `computer.configure()` confirmed to exist: `username`, `port`, `look_for_keys`, `key_filename`, `timeout`, `allow_agent`, `proxy_jump`, `compress`, `gss_*`, `load_system_host_keys`, `key_policy`, `use_login_shell`, `safe_interval`), but no live SSH connection was ever tested. Treat the first real use as the actual validation.
- `register_remote_code`'s `remote_executable_path` must be the path *after* whatever `prepend_text` module-loads run — get this wrong and jobs will fail at submission with a "command not found"-style error on the remote scheduler, not a clear AiiDA-side error.
- The generic remote resource default in `choose_resources` (`mpiprocs = min(n_kpoints * 4, 128)`) is a placeholder, not tuned to any real cluster's node topology — override `metadata.options.resources` explicitly per-cluster once you know its node size (cores/node), rather than trusting this default for production remote jobs.

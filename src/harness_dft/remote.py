"""Generic (cluster-agnostic) helpers for configuring a remote AiiDA
computer + codes over SSH with a Slurm-family scheduler. Deliberately holds
no cluster-specific defaults -- callers supply hostname/username/scheduler
details for whatever HPC target they're wiring up.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RemoteComputerSpec:
    label: str
    hostname: str
    username: str
    scheduler: str = "slurm"
    work_dir: str = "/scratch/{username}/aiida_run"
    mpiprocs_per_machine: int = 24
    default_memory_kb: int | None = None
    prepend_text: str = ""  # e.g. module-load lines for QE on this cluster


def setup_remote_computer(spec: RemoteComputerSpec):
    """Idempotently create+configure an AiiDA Computer over SSH. Requires
    key-based SSH auth to already work to `spec.hostname` as `spec.username`
    (verify with a plain `ssh` call before calling this). Returns the
    configured, enabled Computer node."""
    from aiida import orm
    from aiida.common import NotExistent

    try:
        computer = orm.load_computer(spec.label)
    except NotExistent:
        computer = orm.Computer(
            label=spec.label,
            hostname=spec.hostname,
            transport_type="core.ssh",
            scheduler_type=f"core.{spec.scheduler}",
            workdir=spec.work_dir.format(username=spec.username),
        )
        computer.store()

    computer.set_default_mpiprocs_per_machine(spec.mpiprocs_per_machine)
    if spec.default_memory_kb:
        computer.set_default_memory_per_machine(spec.default_memory_kb)

    computer.configure(
        username=spec.username,
        port=22,
        look_for_keys=True,
        key_filename=None,  # uses default SSH identity unless overridden
        timeout=60,
        allow_agent=True,
        proxy_jump=None,
        compress=True,
        gss_auth=False,
        gss_kex=False,
        gss_deleg_creds=False,
        gss_host=spec.hostname,
        load_system_host_keys=True,
        key_policy="WarningPolicy",
        use_login_shell=True,
        safe_interval=30.0,
    )

    if not computer.is_enabled():
        computer.set_enabled_state(True)

    return computer


def register_remote_code(computer_label: str, label: str, remote_executable_path: str,
                          plugin_entry_point: str, prepend_text: str = ""):
    """Register a QE binary already installed on a configured remote
    Computer as an AiiDA InstalledCode. `remote_executable_path` is the
    absolute path on the remote machine (e.g. after `module load ...` puts
    it on PATH -- resolve it there first with `which pw.x`)."""
    from aiida import orm
    from aiida.common import NotExistent

    full_label = f"{label}@{computer_label}"
    try:
        return orm.load_code(full_label)
    except NotExistent:
        pass

    computer = orm.load_computer(computer_label)
    code = orm.InstalledCode(
        label=label,
        computer=computer,
        filepath_executable=remote_executable_path,
        default_calc_job_plugin=plugin_entry_point,
        prepend_text=prepend_text,
        with_mpi=True,
    )
    code.store()
    return code

"""FastMCP server exposing harness-dft (QE + AiiDA + ASE) as typed tools.

Unlike `ndim-mcp` (which proxies HTTP to a separate engine process), this
server imports `harness_dft` and AiiDA directly and runs in the same
process/host as the AiiDA profile and the `pw.x`/etc. binaries -- there is no
separate "engine" to call over the network. That means it must run
somewhere that actually has the `dft-harness` conda env, a loaded AiiDA
profile, and (for local execution) QE on PATH -- typically the host, not
inside the DeerFlow sandbox container. See the `dft-harness` DeerFlow skill
for the "ask local machine or remote HPC first" workflow this server exists
to support, and the package README for how to point DeerFlow at it.

DFT jobs are long (minutes to hours), so write-path tools submit to the
AiiDA daemon (`aiida.engine.submit`) and return a pk immediately rather than
blocking for the full run -- `hd_wait_for_job`/`hd_get_job` poll it. This
needs `verdi daemon start` to actually be running; `hd_status` reports it.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from .atoms_io import read_atoms
from .config import Settings

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

INSTRUCTIONS = """harness-dft: resource-adaptive Quantum ESPRESSO orchestration on AiiDA + ASE.

FIRST, before planning any calculation: ask the researcher whether this job should run on THIS machine or on a
remote HPC cluster. Do not assume -- call hd_status to see what's actually available (local CPU/RAM/GPU, whether a
GPU-enabled QE code is registered, and any remote computers already configured), present that, and let them choose.
If remote and no computer is configured yet, walk them through hd_setup_remote_computer / hd_register_remote_code
(needs their cluster hostname, username, scheduler and QE module path -- this harness never guesses those).

Workflow shape: hd_status -> hd_estimate (see the resource plan before committing) -> hd_validate_pseudo_coverage ->
hd_submit_* (returns a pk immediately) -> hd_wait_for_job / hd_get_job -> hd_get_job_results.

Every hd_submit_* estimates its own mpiprocs/npool/walltime via the same estimator hd_estimate uses; CPU allocations
are quantized to whole batches of `cpu_batch_size` (default 8) CPUs, GPU allocations use one MPI rank per GPU.
Nothing here claims exact resource prediction -- these are order-of-magnitude starting points; AiiDA's own
PwBaseWorkChain error handlers (out-of-walltime, diagonalization failures) resubmit with more resources if the
estimate was too low. Never claim a DFT result is final without checking hd_get_job's is_finished_ok."""

# Module-level, not local to create_server(): with `from __future__ import
# annotations` in effect, every annotation is stored as a string and resolved
# later by name against the function's __globals__ -- a local variable inside
# create_server() would not be visible to that lookup and FastMCP's tool
# registration would fail with a NameError at server-construction time.
Structure = Annotated[str, Field(description="Structure file content, e.g. a CIF or POSCAR body, as text.")]
StructureFormat = Annotated[str, Field(description='ASE format name, e.g. "cif", "vasp", "extxyz", "xyz", "json".')]
KMesh = Annotated[tuple[int, int, int], Field(description="Monkhorst-Pack mesh, e.g. [4, 4, 4].")]
JobPk = Annotated[int, Field(description="AiiDA node pk returned by an hd_submit_* tool.")]


def create_server(settings=None, host="127.0.0.1", port=8000):
    settings = settings or Settings.from_env()

    from aiida import load_profile
    load_profile(settings.aiida_profile)

    mcp = FastMCP("harness-dft", instructions=INSTRUCTIONS, host=host, port=port)

    # NEB (ase.mep, no AiiDA provenance -- see harness_dft.workflows.neb) has no
    # AiiDA node to poll, so it gets its own tiny in-memory job table instead.
    # Lost on server restart; that's an accepted limitation for the one
    # workflow that was never meant to be provenance-tracked in the first place.
    _neb_jobs: dict[str, dict] = {}
    _neb_lock = threading.Lock()

    def _plan_dict(plan) -> dict:
        return asdict(plan)

    def _resources_kwargs(allow_remote: bool, allow_gpu: bool, cpu_batch_size: int, local_atom_ceiling: int) -> dict:
        return dict(
            allow_remote=allow_remote,
            allow_gpu=allow_gpu,
            cpu_batch_size=cpu_batch_size,
            local_atom_ceiling=local_atom_ceiling,
        )

    # ---- status / discovery -------------------------------------------------

    @mcp.tool(annotations=READ)
    def hd_status() -> dict:
        """Local hardware, AiiDA daemon/profile state, installed pseudopotential
        families, and registered codes/computers. Call this first, and show the
        researcher local CPU/RAM/GPU plus any already-configured remote
        computers before asking "local or remote?"."""
        from aiida import orm

        from harness_dft.environment import get_local_resources
        from harness_dft.jobs import is_daemon_running
        from harness_dft.pseudos import list_installed_families

        local = get_local_resources()
        codes = [
            {"label": code.full_label, "plugin": code.default_calc_job_plugin, "computer": code.computer.label}
            for code in orm.QueryBuilder().append(orm.InstalledCode).all(flat=True)
        ]
        computers = [c.label for c in orm.QueryBuilder().append(orm.Computer).all(flat=True)]
        return {
            "local_resources": asdict(local),
            "cpu_batch_size_default": settings.default_cpu_batch_size,
            "daemon_running": is_daemon_running(),
            "pseudo_families_installed": list_installed_families(),
            "codes": codes,
            "computers": computers,
            "gpu_code_registered": any("gpu" in c["label"].lower() for c in codes),
        }

    @mcp.tool(annotations=READ)
    def hd_list_pseudo_families() -> dict:
        """List every aiida-pseudo family installed in this profile."""
        from harness_dft.pseudos import list_installed_families
        return {"families": list_installed_families()}

    # ---- estimate / validate (no submission) --------------------------------

    @mcp.tool(annotations=READ)
    def hd_estimate(
        structure_text: Structure,
        structure_format: StructureFormat,
        pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
        ecutwfc_ry: float = 40.0,
        kpoints_mesh: KMesh = (4, 4, 4),
        allow_remote: bool = False,
        allow_gpu: bool = False,
        cpu_batch_size: Annotated[int, Field(ge=1, le=256)] = 8,
        local_atom_ceiling: int = 40,
    ) -> dict:
        """Estimate memory/bands/plane-waves and the resource plan (target,
        mpiprocs, npool, walltime) for a structure WITHOUT submitting anything.
        Show this to the researcher before calling any hd_submit_* tool."""
        from aiida import orm

        from harness_dft.environment import get_local_resources
        from harness_dft.estimate import choose_resources, estimate_job
        from harness_dft.structures import count_valence_electrons, is_likely_metal

        atoms = read_atoms(structure_text, structure_format)
        pseudo_family = orm.load_group(pseudo_family_label)
        n_electrons = count_valence_electrons(atoms, pseudo_family)
        n_kpoints = kpoints_mesh[0] * kpoints_mesh[1] * kpoints_mesh[2]
        is_metal = is_likely_metal(atoms)

        job = estimate_job(
            n_atoms=len(atoms), n_electrons=n_electrons, cell_volume_ang3=atoms.get_volume(),
            ecutwfc_ry=ecutwfc_ry, n_kpoints=max(1, n_kpoints), is_metal=is_metal,
        )
        plan = choose_resources(
            job, get_local_resources(),
            **_resources_kwargs(allow_remote, allow_gpu, cpu_batch_size, local_atom_ceiling),
        )
        return {"job": asdict(job), "plan": _plan_dict(plan), "is_likely_metal": is_metal}

    @mcp.tool(annotations=READ)
    def hd_validate_pseudo_coverage(structure_text: Structure, structure_format: StructureFormat,
                                     pseudo_family_label: str) -> dict:
        """Check whether a pseudopotential family covers every element in a
        structure. Call before hd_submit_* -- a missing element fails fast here
        instead of deep inside a submitted QE job."""
        from harness_dft.pseudos import MissingPseudopotentialError, validate_family_covers_structure

        atoms = read_atoms(structure_text, structure_format)
        try:
            validate_family_covers_structure(atoms, pseudo_family_label)
            return {"covered": True, "missing": []}
        except MissingPseudopotentialError as exc:
            missing = str(exc).rsplit(": ", 1)[-1].split(", ")
            return {"covered": False, "missing": missing, "message": str(exc)}

    # ---- job polling (generic across every submitted workflow) --------------

    @mcp.tool(annotations=READ)
    def hd_get_job(pk: JobPk) -> dict:
        """Current status of a submitted job (process_state, exit_status)."""
        from harness_dft.jobs import get_status
        return asdict(get_status(pk))

    @mcp.tool(annotations=READ)
    def hd_wait_for_job(pk: JobPk, timeout_seconds: Annotated[int, Field(ge=0, le=600)] = 60) -> dict:
        """Poll a submitted job until it finishes or timeout_seconds elapses.
        DFT jobs commonly run far longer than one call's timeout -- call this
        repeatedly rather than assuming one call is enough."""
        from harness_dft.jobs import wait_for_job
        return asdict(wait_for_job(pk, timeout_seconds=timeout_seconds))

    @mcp.tool(annotations=READ)
    def hd_get_job_results(pk: JobPk) -> dict:
        """Output namespace of a finished, successful job as a plain dict.
        Raises (returned as an error) if the job hasn't finished or failed --
        check hd_get_job first."""
        from harness_dft.jobs import get_results
        return get_results(pk)

    # ---- submit: relax -------------------------------------------------------

    @mcp.tool(annotations=WRITE)
    def hd_submit_relax(
        structure_text: Structure,
        structure_format: StructureFormat,
        code_label: Annotated[str, Field(description='AiiDA code label, e.g. "pw-7.5@localhost" or "pw-7.5-gpu@localhost".')],
        pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
        protocol: Literal["fast", "moderate", "precise"] = "fast",
        kpoints_mesh: KMesh = (4, 4, 4),
        ecutwfc_ry: float = 40.0,
        allow_remote: bool = False,
        allow_gpu: bool = False,
        cpu_batch_size: Annotated[int, Field(ge=1, le=256)] = 8,
        local_atom_ceiling: int = 40,
    ) -> dict:
        """Submit a structure relaxation (PwRelaxWorkChain). Returns immediately
        with a pk; poll with hd_wait_for_job. `code_label` must match the
        resource plan's target -- call hd_estimate first and pick a code whose
        computer matches plan.target (local vs remote) and whose binary matches
        plan.target=="local-gpu" (a CUDA-built code)."""
        from harness_dft.workflows.relax import build_relax_inputs
        from harness_dft.jobs import submit_builder

        atoms = read_atoms(structure_text, structure_format)
        builder, plan = build_relax_inputs(
            atoms, code_label, pseudo_family_label=pseudo_family_label, protocol=protocol,
            kpoints_mesh=kpoints_mesh, ecutwfc_ry=ecutwfc_ry, allow_remote=allow_remote,
        )
        pk = submit_builder(builder, label="harness-dft relax (MCP)")
        return {"pk": pk, "plan": _plan_dict(plan)}

    # ---- submit: single-point SCF (covers EOS points / convergence points) --

    @mcp.tool(annotations=WRITE)
    def hd_submit_scf(
        structure_text: Structure,
        structure_format: StructureFormat,
        code_label: str,
        pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
        protocol: Literal["fast", "moderate", "precise"] = "fast",
        kpoints_mesh: KMesh = (4, 4, 4),
        ecutwfc_ry: float = 40.0,
        allow_remote: bool = False,
        allow_gpu: bool = False,
        cpu_batch_size: Annotated[int, Field(ge=1, le=256)] = 8,
        local_atom_ceiling: int = 40,
    ) -> dict:
        """Submit a single-point SCF (PwBaseWorkChain). This is the building
        block for equation-of-state and convergence work: call it once per
        volume scaling (EOS) or once per ecutwfc/k-mesh value (convergence),
        then compare hd_get_job_results()['output_parameters']['energy']
        across the runs yourself -- there is no separate EOS/convergence tool,
        this composes into both."""
        from harness_dft.workflows.eos import build_scf_inputs
        from harness_dft.jobs import submit_builder

        atoms = read_atoms(structure_text, structure_format)
        builder, plan = build_scf_inputs(
            atoms, code_label, pseudo_family_label=pseudo_family_label, protocol=protocol,
            kpoints_mesh=kpoints_mesh, ecutwfc_ry=ecutwfc_ry, allow_remote=allow_remote,
        )
        pk = submit_builder(builder, label="harness-dft scf (MCP)")
        return {"pk": pk, "plan": _plan_dict(plan), "cell_volume_ang3": atoms.get_volume()}

    # ---- submit: bands / pdos -------------------------------------------------

    @mcp.tool(annotations=WRITE)
    def hd_submit_bands(
        structure_text: Structure,
        structure_format: StructureFormat,
        code_label: str,
        pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
        protocol: Literal["fast", "moderate", "precise"] = "fast",
        kpoints_mesh: KMesh = (4, 4, 4),
        ecutwfc_ry: float = 40.0,
        allow_remote: bool = False,
    ) -> dict:
        """Submit PwBandsWorkChain (SCF + auto k-path bands via seekpath) on an
        ALREADY-RELAXED structure. Use hd_submit_relax first if it isn't."""
        from harness_dft.workflows.bands_dos import build_bands_inputs
        from harness_dft.jobs import submit_builder

        atoms = read_atoms(structure_text, structure_format)
        builder, scf_plan, bands_plan = build_bands_inputs(
            atoms, code_label, pseudo_family_label=pseudo_family_label, protocol=protocol,
            kpoints_mesh=kpoints_mesh, ecutwfc_ry=ecutwfc_ry, allow_remote=allow_remote,
        )
        pk = submit_builder(builder, label="harness-dft bands (MCP)")
        return {"pk": pk, "scf_plan": _plan_dict(scf_plan), "bands_plan": _plan_dict(bands_plan)}

    @mcp.tool(annotations=WRITE)
    def hd_submit_pdos(
        structure_text: Structure,
        structure_format: StructureFormat,
        pw_code_label: str,
        dos_code_label: str,
        projwfc_code_label: str,
        pseudo_family_label: str = "SSSP/1.3/PBE/efficiency",
        protocol: Literal["fast", "moderate", "precise"] = "fast",
        kpoints_mesh: KMesh = (4, 4, 4),
        ecutwfc_ry: float = 40.0,
        allow_remote: bool = False,
    ) -> dict:
        """Submit PdosWorkChain (SCF + denser NSCF + dos.x + projwfc.x) on an
        ALREADY-RELAXED structure."""
        from harness_dft.workflows.bands_dos import build_pdos_inputs
        from harness_dft.jobs import submit_builder

        atoms = read_atoms(structure_text, structure_format)
        builder, scf_plan, nscf_plan = build_pdos_inputs(
            atoms, pw_code_label, dos_code_label, projwfc_code_label,
            pseudo_family_label=pseudo_family_label, protocol=protocol,
            kpoints_mesh=kpoints_mesh, ecutwfc_ry=ecutwfc_ry, allow_remote=allow_remote,
        )
        pk = submit_builder(builder, label="harness-dft pdos (MCP)")
        return {"pk": pk, "scf_plan": _plan_dict(scf_plan), "nscf_plan": _plan_dict(nscf_plan)}

    # ---- submit: phonon chain, one composable step at a time -----------------

    @mcp.tool(annotations=WRITE)
    def hd_submit_ph(
        parent_scf_pk: Annotated[int, Field(description="pk of a FINISHED hd_submit_scf/hd_submit_relax job.")],
        ph_code_label: str,
        structure_text: Structure,
        structure_format: StructureFormat,
        pseudo_family_label: str,
        ecutwfc_ry: float,
        qpoints_mesh: KMesh = (2, 2, 2),
        protocol: Literal["fast", "moderate", "precise"] = "fast",
        is_metal: bool = False,
        allow_remote: bool = False,
    ) -> dict:
        """Submit PhBaseWorkChain (DFPT) from a finished SCF's remote_folder.
        Step 1 of the phonon chain: hd_submit_ph -> hd_submit_q2r -> hd_submit_matdyn."""
        from aiida import orm
        from harness_dft.workflows.phonons import build_ph_inputs
        from harness_dft.jobs import submit_builder

        atoms = read_atoms(structure_text, structure_format)
        parent_node = orm.load_node(parent_scf_pk)
        builder, plan = build_ph_inputs(
            parent_node, ph_code_label, atoms, pseudo_family_label, ecutwfc_ry,
            qpoints_mesh=qpoints_mesh, protocol=protocol, is_metal=is_metal, allow_remote=allow_remote,
        )
        pk = submit_builder(builder, label="harness-dft ph (MCP)")
        return {"pk": pk, "plan": _plan_dict(plan)}

    @mcp.tool(annotations=WRITE)
    def hd_submit_q2r(ph_pk: Annotated[int, Field(description="pk of a FINISHED hd_submit_ph job.")],
                       q2r_code_label: str) -> dict:
        """Step 2 of the phonon chain: real-space force constants from a
        finished PhBaseWorkChain."""
        from aiida import orm
        from harness_dft.workflows.phonons import build_q2r_inputs
        from harness_dft.jobs import submit_builder

        ph_node = orm.load_node(ph_pk)
        builder = build_q2r_inputs(ph_node, q2r_code_label)
        pk = submit_builder(builder, label="harness-dft q2r (MCP)")
        return {"pk": pk}

    @mcp.tool(annotations=WRITE)
    def hd_submit_matdyn(q2r_pk: Annotated[int, Field(description="pk of a FINISHED hd_submit_q2r job.")],
                          matdyn_code_label: str,
                          dispersion_kpoints_mesh: KMesh = (8, 8, 8)) -> dict:
        """Step 3 of the phonon chain: interpolate the dispersion onto a
        q-point mesh from finished Q2r force constants. (Mesh only for now --
        a true high-symmetry q-path is not yet wired up; see the dft-phonons
        skill.)"""
        from aiida import orm
        from harness_dft.workflows.phonons import build_matdyn_inputs
        from harness_dft.jobs import submit_builder

        q2r_node = orm.load_node(q2r_pk)
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh(dispersion_kpoints_mesh)
        builder = build_matdyn_inputs(q2r_node, matdyn_code_label, kpoints)
        pk = submit_builder(builder, label="harness-dft matdyn (MCP)")
        return {"pk": pk}

    # ---- NEB (ASE-native, no AiiDA provenance -> in-memory job table) -------

    @mcp.tool(annotations=WRITE)
    def hd_submit_neb(
        initial_structure_text: Structure,
        initial_structure_format: StructureFormat,
        final_structure_text: Structure,
        final_structure_format: StructureFormat,
        pseudopotentials: Annotated[dict[str, str], Field(description="element symbol -> UPF filename, e.g. {\"Li\": \"Li.upf\"}.")],
        pseudo_dir: str,
        pw_command: str = "pw.x",
        kpoints_mesh: KMesh = (2, 2, 2),
        ecutwfc_ry: float = 40.0,
        n_images: Annotated[int, Field(ge=1, le=15)] = 5,
        climb: bool = True,
        fmax: float = 0.05,
    ) -> dict:
        """Submit an ASE-native NEB barrier estimate (QE as the force engine via
        ASE's Espresso calculator, subprocess -- NOT AiiDA-tracked). Runs in a
        background thread on THIS server's host; `pw_command`/`pseudo_dir` are
        paths on that host, not the agent sandbox. Poll with hd_get_neb_job.
        Not provenance-tracked: treat results as exploratory, per the harness's
        own ASE/AiiDA layering (see dft-neb skill)."""
        from harness_dft.workflows.neb import build_espresso_calculator, run_neb, barrier_energy

        initial = read_atoms(initial_structure_text, initial_structure_format)
        final = read_atoms(final_structure_text, final_structure_format)
        job_id = uuid.uuid4().hex

        def factory():
            return build_espresso_calculator(
                pseudopotentials, pseudo_dir, pw_command=pw_command, kpts=kpoints_mesh, ecutwfc=ecutwfc_ry,
            )

        def worker():
            try:
                _neb, images, _opt = run_neb(initial, final, factory, n_images=n_images, climb=climb, fmax=fmax)
                barrier = barrier_energy(images)
                with _neb_lock:
                    _neb_jobs[job_id] = {
                        "status": "finished", "barrier_ev": barrier,
                        "image_energies_ev": [img.get_potential_energy() for img in images],
                    }
            except Exception as exc:  # surfaced via hd_get_neb_job rather than crashing the server
                with _neb_lock:
                    _neb_jobs[job_id] = {"status": "failed", "error": str(exc)}

        with _neb_lock:
            _neb_jobs[job_id] = {"status": "running"}
        threading.Thread(target=worker, daemon=True).start()
        return {"neb_job_id": job_id}

    @mcp.tool(annotations=READ)
    def hd_get_neb_job(neb_job_id: str) -> dict:
        """Status/result of a hd_submit_neb job. Only lives in this server
        process's memory -- lost if the server restarts."""
        with _neb_lock:
            job = _neb_jobs.get(neb_job_id)
        if job is None:
            return {"status": "unknown", "error": "no such neb_job_id (or the server was restarted)"}
        return job

    # ---- remote HPC setup ------------------------------------------------

    @mcp.tool(annotations=WRITE)
    def hd_setup_remote_computer(
        label: str,
        hostname: str,
        username: str,
        scheduler: Literal["slurm", "pbspro", "sge", "torque"] = "slurm",
        work_dir: str = "/scratch/{username}/aiida_run",
        mpiprocs_per_machine: int = 24,
        prepend_text: str = "",
    ) -> dict:
        """Idempotently register a remote HPC computer over SSH. Requires
        key-based SSH auth to already work (verify with a plain `ssh` call
        first -- this tool does not set up keys). Ask the researcher for
        hostname/username/scheduler/module-load lines; never guess a specific
        cluster's details."""
        from harness_dft.remote import RemoteComputerSpec, setup_remote_computer

        spec = RemoteComputerSpec(
            label=label, hostname=hostname, username=username, scheduler=scheduler,
            work_dir=work_dir, mpiprocs_per_machine=mpiprocs_per_machine, prepend_text=prepend_text,
        )
        computer = setup_remote_computer(spec)
        return {"label": computer.label, "hostname": computer.hostname, "enabled": computer.is_enabled()}

    @mcp.tool(annotations=WRITE)
    def hd_register_remote_code(
        computer_label: str,
        label: str,
        remote_executable_path: str,
        plugin_entry_point: Annotated[str, Field(description='e.g. "quantumespresso.pw", "quantumespresso.ph".')],
        prepend_text: str = "",
    ) -> dict:
        """Idempotently register a QE binary already installed on a configured
        remote computer. `remote_executable_path` must be the path AFTER
        whatever module-loads run in prepend_text -- confirm with
        `ssh <host> 'module load ... && which pw.x'` first."""
        from harness_dft.remote import register_remote_code

        code = register_remote_code(computer_label, label, remote_executable_path, plugin_entry_point, prepend_text)
        return {"full_label": code.full_label, "pk": code.pk}

    return mcp


__all__ = ["create_server"]

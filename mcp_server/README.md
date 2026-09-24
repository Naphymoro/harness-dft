# harness-dft-mcp — harness-dft as an MCP server

Lets an agent harness such as DeerFlow drive `harness-dft` (resource-adaptive Quantum ESPRESSO orchestration on
AiiDA + ASE) without touching its code. The matching DeerFlow skill is `deer-flow/skills/public/dft-harness/`.
Read it for how agents should *use* these tools — in particular, its hard rule #1: **ask the researcher whether to
run locally or on remote HPC before planning anything**.

## Design

| Decision | Why |
|---|---|
| Imports `harness_dft`/AiiDA directly, does not call over HTTP | Unlike `ndim-mcp` (which proxies to a separate engine process), there is no separate "DFT engine" service — this server *is* the AiiDA client, running wherever the AiiDA profile and QE binaries actually live. |
| Every `hd_submit_*` tool submits (`aiida.engine.submit`) and returns a pk immediately | DFT jobs take minutes to hours; an MCP tool call cannot block for that. Requires the AiiDA daemon running (`verdi daemon start`) — `hd_status` reports whether it is. |
| Structures are passed as inline text + ASE format name, never a file path | This server may run on a different machine/filesystem than the agent sandbox holding the structure file. |
| No auto target selection | `choose_resources` (used by `hd_estimate` and every `hd_submit_*`) only *recommends* local/local-gpu/remote; the caller supplies `code_label` explicitly. This mirrors the underlying harness's own design and keeps the researcher's local-vs-remote choice authoritative. |
| CPU allocations quantized to whole `cpu_batch_size` batches (default 8) | Matches real HPC scheduling granularity and avoids QE's per-rank overhead at tiny rank counts. GPU allocations use one rank per GPU instead — a different rule, not a batch. |
| NEB gets its own in-memory job table, not AiiDA pks | `workflows/neb.py` is deliberately ASE-native with no AiiDA provenance (see the harness's own module docstring). `hd_submit_neb`/`hd_get_neb_job` track it in this process's memory instead; lost on server restart. |
| Remote/GPU setup tools are idempotent, never guess cluster/GPU specifics | `hd_setup_remote_computer`/`hd_register_remote_code` require the researcher's own hostname/username/scheduler/module details, matching `harness_dft.remote`'s cluster-agnostic design. |

## Tools (21)

Read: `hd_status`, `hd_list_pseudo_families`, `hd_estimate`, `hd_validate_pseudo_coverage`,
`hd_generate_2d_prototype`, `hd_check_phonon_stability`, `hd_rank_prototypes`, `hd_get_job`, `hd_wait_for_job`,
`hd_get_job_results`, `hd_get_neb_job`.
Write: `hd_submit_relax`, `hd_submit_scf`, `hd_submit_bands`, `hd_submit_pdos`, `hd_submit_ph`, `hd_submit_q2r`,
`hd_submit_matdyn`, `hd_submit_neb`, `hd_setup_remote_computer`, `hd_register_remote_code`.

`hd_generate_2d_prototype`/`hd_check_phonon_stability`/`hd_rank_prototypes` support elemental 2D-monolayer
screening (see `docs/2d-screening.md`) -- `hd_submit_relax` also gained a `cell_dofree` parameter for this
(`"2Dxy"` keeps a slab's vacuum spacing intact during vc-relax).

Full argument/return reference: `deer-flow/skills/public/dft-harness/references/tool-reference.md`.

## Run it

Needs the `dft-harness` conda env (see the package README's Setup section) with an AiiDA profile loaded, QE codes
registered, and the daemon running:

```bash
ENV=~/.local/share/cyranoid/conda/envs/dft-harness
export PATH="$ENV/bin:$PATH"
export AIIDA_PROFILE=dft-harness
verdi daemon start   # required -- hd_submit_* jobs never run without it

cd harness-dft
$ENV/bin/pip install -e .              # the harness_dft package itself
$ENV/bin/pip install -e mcp_server --no-deps
$ENV/bin/pip install "mcp>=1.2,<2"     # if not already present

python -m harness_dft_mcp                                                   # stdio, what DeerFlow spawns directly on the host
python -m harness_dft_mcp --transport streamable-http --port 8767           # http://127.0.0.1:8767/mcp
```

| Env var | Default | Meaning |
|---|---|---|
| `AIIDA_PROFILE` | AiiDA's own default-profile resolution | Which profile to load |
| `HARNESS_DFT_CPU_BATCH_SIZE` | `8` | Default CPU allocation batch size |
| `HARNESS_DFT_LOCAL_ATOM_CEILING` | `40` | Default atom-count ceiling for staying local |
| `HARNESS_DFT_GPU_MIN_ATOMS` | `8` | Default minimum atom count before GPU offload is recommended |
| `HARNESS_DFT_MCP_MAX_WAIT_SECONDS` | `120` | Reserved for future use; `hd_wait_for_job`'s `timeout_seconds` is capped per-call at 600 already |

## Wire into DeerFlow

`deer-flow/extensions_config.json` contains a `harness-dft` entry, **disabled**, pointing at
`http://host.docker.internal:8767/mcp`, following the same pattern as the `ndim-engine` entry (DeerFlow's gateway
here runs in Docker). Steps:

1. Start the daemon and this server on the host, bound where the container can reach it —
   `host.docker.internal` resolves to the Docker bridge address (commonly `172.17.0.1`) inside the gateway:
   `python -m harness_dft_mcp --transport streamable-http --host 172.17.0.1 --port 8767`
2. Enable `harness-dft` (DeerFlow MCP settings, or `"enabled": true` in `extensions_config.json`).

**Firewall caveat (same one hit wiring up `ndim-engine` on this machine).** From inside the gateway container,
connections to arbitrary host ports can time out even with a plain `0.0.0.0` listener, if a host firewall drops
container→host traffic. Fixing this needs root: `ufw allow from 172.16.0.0/12 to any port 8767 proto tcp`, then
re-test with
`docker exec -i deer-flow-gateway python -c "import httpx;print(httpx.get('http://host.docker.internal:8767/mcp').status_code)"`
(any HTTP status, even 4xx, means it's reachable). This was **not verified end to end from the container** for
the same reason it wasn't for `ndim-engine` — see its README for the alternative (containers on the compose
network).

If DeerFlow runs directly on the host (not Docker), use stdio instead, reusing DeerFlow's own venv is not an
option here (this server needs the `dft-harness` conda env's AiiDA/ASE/QE stack, not DeerFlow's Python packages):

```json
"harness-dft": {"enabled": true, "type": "stdio",
  "command": "<path-to>/.local/share/cyranoid/conda/envs/dft-harness/bin/python", "args": ["-m", "harness_dft_mcp"],
  "env": {"AIIDA_PROFILE": "dft-harness"},
  "tool_name_prefix": false, "tool_call_timeout": 650}
```

**Security:** this server does not authenticate callers, and its write tools submit real compute jobs and can
register remote SSH computers. Keep the HTTP transport on `127.0.0.1`, the docker0 address, or behind your own
authenticating proxy, never `0.0.0.0`.

## Verified vs. not

Live-verified against this machine's real AiiDA profile and QE 7.5, on both CPU (`*-7.5@localhost`) and GPU
(`*-7.5-gpu@localhost`, built per `docs/gpu-build.md`) codes:

- `hd_status`, `hd_estimate`, `hd_validate_pseudo_coverage` (read-only).
- `hd_submit_scf` → `hd_wait_for_job` → `hd_get_job_results` on bulk Si: CPU vs. GPU energy agree to 9 significant
  figures; `allow_gpu=True` correctly picks `target="local-gpu"`/`mpiprocs=1`.
- The full phonon chain, GPU codes: `hd_submit_ph` → `hd_submit_q2r` → `hd_submit_matdyn`, all
  `exit_status=0`/`is_finished_ok=True`, `matdyn`'s output includes `output_phonon_bands`.
- `hd_submit_pdos`, GPU codes (`pw`+`dos`+`projwfc`): finishes successfully, results correctly include the
  namespaced `dos.output_dos`/`projwfc.Dos`/`projwfc.Pdos`/`projwfc.projections` outputs.
- `hd_generate_2d_prototype` → `hd_estimate` → `hd_submit_relax(cell_dofree="2Dxy")` on a real 2D candidate
  (aluminene, CPU code): `exit_status=0`; vacuum spacing survived vc-relax intact (c-axis unchanged to 8
  significant figures) while the in-plane bond length and buckling genuinely relaxed. See `docs/2d-screening.md`.

This exercise caught **three real bugs**, all fixed with regression tests: (1) `hd_submit_relax`/`hd_submit_scf`/
`hd_submit_ph`/`hd_submit_bands`/`hd_submit_pdos` variously either dropped `allow_gpu`/`cpu_batch_size`/
`local_atom_ceiling` on the way to their builder functions, or (for `hd_submit_ph`/`hd_submit_bands`/
`hd_submit_pdos`) didn't declare those parameters at all — caught by a real GPU submission coming back with a
CPU-style 8-rank plan instead of the correct 1-rank-per-GPU one; (2) `hd_get_job_results`/`harness_dft.jobs.
get_results` silently dropped every namespaced output (`PdosWorkChain`'s `dos.*`/`projwfc.*`), so a successfully
finished PDOS job returned an empty dict with no error — fixed by recursing into nested output namespaces instead
of skipping them; (3) `choose_resources`' `npool` picked the largest divisor of **n_kpoints** capped at
`mpiprocs`, when QE's `-npool` actually requires a divisor **of mpiprocs** — every prior test's k-point mesh
total happened to share a large common factor with the 8-rank batch size, until a real `(9,9,1)`-mesh 2D candidate
(81 kpoints, factors only 3^4) computed `npool=3` against `mpiprocs=8` and `pw.x` aborted immediately
(`mp_start_pools`: `parent_nproc /= nproc_pool * npool`). Fixed, with a regression test reproducing the exact
failure.

**Not live-verified**: `hd_submit_relax`/`hd_submit_bands` specifically on a GPU code (the parameter-forwarding
fix is identical to the live-tested `hd_submit_scf`/`hd_submit_ph` paths and covered by a regression test, but
these two combinations weren't separately re-run end-to-end), `hd_submit_neb` (never run live at all),
`hd_setup_remote_computer`/`hd_register_remote_code` (no real SSH target), the puckered 2D prototype through a
real relax (only geometry-checked, not DFT-relaxed), `hd_check_phonon_stability` on any 2D candidate (only
exercised against bulk Si), and `hd_rank_prototypes` on a real "which prototype wins" comparison (only mechanically
tested against two arbitrary bulk-Si SCF nodes). Treat first real use of each as validation.

## Tests

```bash
AIIDA_PROFILE=dft-harness $ENV/bin/python -m pytest mcp_server/tests
```

Integration-style against the real local AiiDA profile (there is no lightweight fake AiiDA to mock against, unlike
`ndim-mcp`'s `httpx.MockTransport`) — skipped if no profile loads. Covers tool registration and the read-only
tools (`hd_status`, `hd_estimate`, `hd_validate_pseudo_coverage`); does not submit jobs (that's exercised by hand,
see "Verified vs. not" above, to avoid every test run queuing real QE calculations).

## Known limits

- No approval gate on `hd_submit_*` (unlike `ndim-mcp`'s `approval_statement` requirement) — the skill's hard
  rule #1 (ask local-vs-remote first) is the trust boundary here, enforced by instruction, not by the tool schema.
  Harden this in DeerFlow (guardrails / human-in-the-loop) if agents run unattended.
- `hd_get_job_results` only surfaces `Dict`/scalar outputs and `{node_type, pk, uuid}` references for everything
  else (structures, trajectories, remote/retrieved folders) — no tool yet to pull those into a sandbox file.
- No tool exposes `verdi process report` — failures surface only via `exit_status`/`exit_message`.

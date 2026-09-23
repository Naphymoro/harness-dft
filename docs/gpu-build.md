# GPU-enabled Quantum ESPRESSO: build runbook and status

## Why this isn't a simple `pip`/`conda install`

QE's GPU offload (`-DQE_ENABLE_CUDA=ON` in its CMake build) needs `nvfortran` — CUDA Fortran — which ships only in
NVIDIA's HPC SDK. Checked directly on this machine: **no `nvhpc`/`nvfortran`/`hpc-sdk` package exists on
conda-forge or the `nvidia` conda channel** (`micromamba search` against both returns nothing). The only real
distribution channels are NVIDIA's own installer tarball, an apt/yum repo, or a prebuilt NGC container — the first
is the only one that needs neither root nor a container runtime change, which is why it's the one used here.

## Hardware/software on this machine

- GPU: 1x NVIDIA RTX 2000 Ada Generation, 16 GB VRAM, driver 595.91.07, CUDA 13.2 (driver-reported)
- `nvcc` 12.4 already present (CUDA toolkit), but that alone cannot compile QE's Fortran GPU code — `nvfortran`
  specifically is what's missing, hence the HPC SDK
- No `nvidia-container-toolkit`, no passwordless sudo — ruled out the Docker/NGC prebuilt-container path for this
  reason (would need a `sudo apt install` + Docker daemon restart)
- Existing CPU QE: conda-forge `qe-7.5` package, registered as `pw-7.5@localhost` etc. in AiiDA

## Plan

1. Download NVIDIA HPC SDK (Linux x86_64, CUDA-multi build) tarball directly — no login required for the SDK
   itself: `https://developer.download.nvidia.com/hpc-sdk/25.7/nvhpc_2025_257_Linux_x86_64_cuda_multi.tar.gz`
   (~12.6 GB). Downloaded to `~/.local/share/cyranoid/gpu-build/`.
2. Run its installer (`install` script inside the extracted tarball) with a **user-writable install prefix**
   (e.g. `~/.local/share/cyranoid/nvhpc`), answering "single system install" / non-interactive mode
   (`NVHPC_SILENT=true NVHPC_INSTALL_DIR=<prefix> NVHPC_INSTALL_TYPE=single ./install`) — no sudo needed for a
   non-default prefix.
3. Source the SDK's environment module (`<prefix>/Linux_x86_64/<ver>/compilers/bin`,
   `.../comm_libs/mpi/bin` for its bundled OpenMPI) onto `PATH` for the build shell only — deliberately not
   added to the user's default shell profile, to avoid silently shadowing the existing `dft-harness` conda
   toolchain.
4. Download QE source matching the installed CPU version's major release (7.5, from
   `https://gitlab.com/QEF/q-e/-/archive/qe-7.5/q-e-qe-7.5.tar.gz`) — using the same major version keeps
   `aiida-quantumespresso`'s output parsing (XML schema, `qe-tools`) compatible between the CPU and GPU codes.
5. Configure with QE's CMake build against `nvfortran`/`nvc` and the SDK's bundled OpenMPI. Confirmed directly
   from `q-e-qe-7.5/CMakeLists.txt` and `cmake/NVFortranCompiler.cmake` (so this is read from source, not
   guessed): `QE_ENABLE_CUDA` requires `CMAKE_Fortran_COMPILER_ID` to be `PGI` or `NVHPC` (i.e.
   `CMAKE_Fortran_COMPILER=nvfortran`), and the target GPU architecture is set via `QE_GPU_ARCHS=sm_89` for this
   card (Ada Lovelace, compute capability 8.9 -- QE translates `sm_89` into nvfortran's `-gpu=cc89`).

   ```bash
   NVHPC=<install-prefix>/Linux_x86_64/<ver>
   export PATH="$NVHPC/compilers/bin:$NVHPC/comm_libs/mpi/bin:$PATH"
   cd q-e-qe-7.5 && mkdir build-gpu && cd build-gpu
   cmake -DCMAKE_Fortran_COMPILER=nvfortran -DCMAKE_C_COMPILER=nvc \
         -DQE_ENABLE_CUDA=ON -DQE_GPU_ARCHS=sm_89 -DQE_ENABLE_MPI=ON \
         -DCMAKE_BUILD_TYPE=Release ..
   make -j"$(nproc)" pw
   ```

   Build `pw.x` at minimum; the rest of the GPU-enabled suite (`ph.x` etc.) if time and the build allow. No
   `ninja` is installed on this host -- using the CMake default `Unix Makefiles` generator (`make -j24` still
   parallelizes fine).
6. Register the resulting binary as a **separate** AiiDA code, e.g. `pw-7.5-gpu@localhost` (`core.code.installed`,
   `quantumespresso.pw` plugin) — never overwrite `pw-7.5@localhost`; the harness's routing
   (`choose_resources`/`hd_status().gpu_code_registered`) expects the CPU and GPU codes to coexist under distinct
   labels so a job can be sent to either deliberately.
7. Validate: run `hd_submit_scf`/`hd_submit_relax` with `code_label="pw-7.5-gpu@localhost"` on a structure above
   `gpu_min_atoms`, confirm it actually uses the GPU (`nvidia-smi` showing utilization during the run) and that
   `output_parameters.convergence_info.scf_conv.convergence_achieved` matches the CPU result on the same
   structure within normal numerical tolerance.

## Two real problems hit during the build, and their fixes

Both confirmed by reading the actual error and tracing it, not guessed:

1. **`MPI_Init` aborts with "opal_init:startup:internal-failure"` / help files not found at
   `/proj/nv/libraries/...`.** The bundled HPC-X OpenMPI (`comm_libs/mpi` symlinks to `comm_libs/12.9/hpcx/hpcx-2.22.1`)
   has NVIDIA's internal build-machine path baked in for locating its own runtime data, and doesn't auto-relocate.
   Fix: set `OPAL_PREFIX` to the real local path before running anything MPI-linked against this SDK:
   ```bash
   export OPAL_PREFIX="$NVHPC/comm_libs/12.9/hpcx/hpcx-2.22.1/ompi"
   ```
   This must also go in the AiiDA code's `prepend_text` (see step 6 below), or every submitted job fails the
   same way.
2. **`pw.x` aborts immediately with `libgomp: TODO`.** QE enables `QE_ENABLE_OPENMP` by default whenever
   `QE_ENABLE_CUDA` is on, and its CMake `find_package(FFTW3)` picked up Ubuntu's system FFTW3 including
   `libfftw3_omp.so.3` (built against GNU's `libgomp`). The resulting `pw.x` links **both** `libgomp.so.1` (via
   FFTW3) and NVHPC's own `libnvomp.so` (from nvfortran-compiled code) — two independent OpenMP runtimes in one
   process, which crashes on the first parallel region. Fix: disable QE's own OpenMP layer, which only affects
   host-side threading (GPU offload via CUDA Fortran/OpenACC is untouched):
   ```bash
   cmake ... -DQE_ENABLE_OPENMP=OFF ...
   ```
   After this, `ldd bin/pw.x` shows only `libfftw3.so.3` (no `_omp` variant) and `libnvomp.so` — one OpenMP
   runtime, no crash.

## Status: done and validated

The build succeeded and is registered as `pw-7.5-gpu@localhost` (`core.code.installed`, pk 550, pointing at
`~/.local/share/cyranoid/gpu-build/q-e-qe-7.5/build-gpu2/bin/pw.x`, `prepend_text` setting `PATH` for
`nvfortran`/HPC-X `mpirun` and `OPAL_PREFIX` per the fix above). Validated three ways:

- QE's own routine-timing breakdown for a real SCF run shows explicit `GPU` wall-clock entries for
  `cdiaghg`/`vloc_psi`/`fft`/`ffts`/`fftw` — confirms kernels actually executed on the device, not just that CUDA
  libraries are linked.
- Ran the identical bulk-Si structure (8-atom conventional cell, `ecutwfc_ry=30`, `kpoints_mesh=[2,2,2]`,
  `SSSP/1.3/PBE/efficiency`) through both `pw-7.5@localhost` (CPU) and `pw-7.5-gpu@localhost` (GPU) via
  `hd_submit_scf`. Energies: CPU `-1242.1392117417 eV`, GPU `-1242.1392117388 eV` — agree to 9 significant
  figures. Both reported `convergence_info.scf_conv.convergence_achieved: true`.
- Went through the harness's own automatic routing, not a manual override: `hd_estimate`/`hd_submit_scf` with
  `allow_gpu=True` on this 8-atom structure correctly picked `target="local-gpu"`, `mpiprocs=1` (one MPI rank per
  the single GPU) with no cluster/GPU details hand-specified.

**A real bug was caught by this validation and fixed**: `hd_submit_relax`/`hd_submit_scf` in
`mcp_server/harness_dft_mcp/server.py` declared `allow_gpu`/`cpu_batch_size`/`local_atom_ceiling` as tool
parameters but only forwarded `allow_remote` to the builder functions — the GPU/batch-size choice was silently
dropped. First attempt at the comparison above submitted with `code_label="pw-7.5-gpu@localhost"` but got back
`target="local"`, `mpiprocs=8` (8 CPU-side ranks all sharing the one GPU context) instead of the expected
`local-gpu`/`mpiprocs=1`. It still finished and gave a numerically correct energy (GPUs tolerate multiple
processes attaching, just inefficiently), but the resource plan was wrong. Fixed by forwarding all three
parameters; a regression test (`test_submit_scf_forwards_allow_gpu_and_cpu_batch_size_to_the_plan`) now checks
non-`allow_remote` kwargs actually reach the plan.

## Full suite build (`make all`) and the rest of the harness's codes

The `pw`-only build above was step one; `make -j"$(nproc)" all` from the *same* `build-gpu2` directory (same CMake
cache: CUDA on, `sm_89`, OpenMP off) built the entire QE distribution in one pass, reusing everything already
compiled for `pw.x` -- `ph.x`, `dos.x`, `projwfc.x`, `q2r.x`, `matdyn.x`, `neb.x`, `cp.x`, `pwcond.x`, `epw.x`,
`hp.x`, `xspectra.x`, and the rest, over 80 executables in total, all built with the same CUDA-enabled shared
libraries `pw.x` uses. Registered as `<name>-7.5-gpu@localhost` for every code this harness already had a CPU
version of: `ph-7.5-gpu`, `dos-7.5-gpu`, `projwfc-7.5-gpu`, `q2r-7.5-gpu`, `matdyn-7.5-gpu`, `neb-7.5-gpu` (same
`prepend_text` as `pw-7.5-gpu`: `PATH` for `nvfortran`/HPC-X `mpirun`, `OPAL_PREFIX`).

**Note on `neb-7.5-gpu`:** registered for parity/completeness, but nothing in this harness currently calls it --
`harness_dft.workflows.neb` is deliberately ASE-native, driving `pw.x` directly via ASE's `Espresso` calculator,
not QE's own `neb.x`. It's there if a future AiiDA-native NEB workflow gets added.

**Two more real bugs found by validating the phonon chain and PDOS, both fixed:**

3. **`hd_submit_ph`/`hd_submit_bands`/`hd_submit_pdos` had no `allow_gpu`/`cpu_batch_size`/`local_atom_ceiling`
   parameters at all** (not even declared, let alone forwarded) -- `build_ph_inputs`/`build_bands_inputs`/
   `build_pdos_inputs` in `workflows/phonons.py`/`workflows/bands_dos.py` only ever accepted `allow_remote`. Caught
   the same way as bug 3 in the pw/scf case: a real `hd_submit_ph` call with `ph_code_label="ph-7.5-gpu@localhost"`
   and `allow_gpu=True` came back with `target="local"`, `mpiprocs=8` -- 8 CPU ranks all attaching to the one GPU
   (confirmed via `nvidia-smi --query-compute-apps`, all 8 processes visible on-device, 99% GPU utilization) instead
   of the correct 1-rank plan. Fixed by threading the three parameters through both workflow modules and all three
   MCP tools, with regression tests for each (`test_submit_bands_forwards_...`, `test_submit_ph_forwards_...`).
4. **`harness_dft.jobs.get_results` silently dropped every namespaced output.** `PdosWorkChain`'s outputs are
   namespaced (`dos.output_dos`, `projwfc.Pdos`, `nscf.output_parameters`, etc.), which `.nested()` returns as
   nested dicts -- the original code did `if isinstance(value, dict): continue`, meaning a real, successfully
   finished PDOS submission (`exit_status=0`, `is_finished_ok=True`) came back from `hd_get_job_results` with an
   **empty dict**, silently. No exception, no warning -- just nothing, which looks exactly like "this workflow has
   no outputs" instead of "this code has a bug." Fixed by recursing into nested namespaces instead of skipping
   them; `output_dos`/`Pdos`/etc. now come back as `results["dos"]["output_dos"]` etc. Regression test:
   `test_get_results_recurses_into_namespaced_sub_outputs` (`tests/test_jobs.py`).

## Full-suite validation results

All run through the real `harness-dft-mcp` server against this machine's AiiDA profile, on bulk Si (8 atoms):

- **Phonon chain** (`hd_submit_ph` → `hd_submit_q2r` → `hd_submit_matdyn`, GPU codes, Gamma-only q-mesh, chained
  from the GPU SCF at pk 573): all three steps `exit_status=0`, `is_finished_ok=True`. `hd_submit_ph` with
  `allow_gpu=True` correctly picked `target="local-gpu"`, `mpiprocs=1` (confirmed via `ps`/`nvidia-smi`: a single
  `ph.x` process, GPU actively utilized). `matdyn`'s output included `output_phonon_bands` as expected.
- **PDOS** (`hd_submit_pdos`, `pw-7.5-gpu` + `dos-7.5-gpu` + `projwfc-7.5-gpu`, `allow_gpu=True`): both the SCF and
  NSCF sub-steps correctly picked `target="local-gpu"`/`mpiprocs=1`; the workchain finished
  `exit_status=0`/`is_finished_ok=True`; results correctly include `dos.output_dos`, `projwfc.Dos`,
  `projwfc.Pdos`, `projwfc.projections` (after the `get_results` fix above).

**Known remaining gaps:**
- `hd_submit_relax`/`hd_submit_bands` on a GPU code were not separately re-validated after the parameter-forwarding
  fix (only `hd_submit_scf`/`hd_submit_ph`/`hd_submit_pdos` were actually re-run end-to-end) -- the code path is
  identical and covered by a regression test, but treat an actual GPU relax/bands submission as the first live
  check.
- Validated on exactly one structure (bulk Si, 8 atoms) and one machine (1x RTX 2000 Ada, 16GB VRAM). Larger
  systems, multi-GPU, or a different GPU architecture are untested -- `QE_GPU_ARCHS=sm_89` was compiled for this
  specific card's compute capability (8.9) and would need rebuilding for a different one.
- `neb.x`, `cp.x`, `pwcond.x`, `epw.x`, `hp.x`, `xspectra.x`, and the rest of the ~80 executables in `make all`
  compiled successfully but were not run at all -- only the six codes this harness already has workflow code for
  (`pw`, `ph`, `dos`, `projwfc`, `q2r`, `matdyn`) were registered and tested.
- The performance benefit was not measured (the test structures are tiny -- runs finished in seconds to a few
  minutes). This build proves *correctness*, not speedup; benchmark before relying on it for throughput.

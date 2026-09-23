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

**Known remaining gaps:**
- Only `pw.x` was built. The rest of the GPU-enabled suite (`ph.x`, etc.) was not attempted — phonon workflows
  still route to the CPU codes regardless of `allow_gpu`.
- Validated on exactly one structure (bulk Si, 8 atoms) and one machine (1x RTX 2000 Ada, 16GB VRAM). Larger
  systems, multi-GPU, or a different GPU architecture are untested — `QE_GPU_ARCHS=sm_89` was compiled for this
  specific card's compute capability (8.9) and would need rebuilding for a different one.
- The performance benefit was not measured (the test structure is tiny -- both runs finished in under a second).
  This build proves *correctness*, not speedup; benchmark before relying on it for throughput.

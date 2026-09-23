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

## Status

As of this writing: the HPC SDK tarball download was in progress (step 1, ~12.6 GB, background). QE 7.5 source
(step 4) is already downloaded and extracted to `~/.local/share/cyranoid/gpu-build/q-e-qe-7.5/`, and the exact
CMake invocation for step 5 has been confirmed by reading QE's own `CMakeLists.txt`/`NVFortranCompiler.cmake`
(above) rather than guessed. Steps 2, 3, 5, 6, 7 had not been attempted yet. **No GPU-built QE code exists or is
registered in AiiDA at this point** — `allow_gpu=True` on any `harness_dft`/MCP tool call will fall back to CPU
routing regardless, since `choose_resources` only recommends a target; nothing enforces that a matching code
actually exists until submission, and there is no `pw-*-gpu@*` code to submit to yet.

**Update this file** (status, exact CMake invocation that worked, any errors hit and their fixes) once the build
is actually attempted — this is the newest, least-travelled part of the whole harness and the most likely place
for the plan above to need correction against what QE 7.5's CMake build actually accepts from NVHPC 25.7.

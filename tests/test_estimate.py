from harness_dft.environment import LocalResources
from harness_dft.estimate import (
    choose_resources,
    estimate_bands,
    estimate_job,
    estimate_plane_waves,
    _largest_divisor_at_most,
    _quantize_mpiprocs,
)


def test_estimate_bands_insulator_vs_metal():
    insulator_bands = estimate_bands(n_electrons=8, is_metal=False)
    metal_bands = estimate_bands(n_electrons=8, is_metal=True)
    assert metal_bands > insulator_bands
    assert insulator_bands >= 4


def test_estimate_plane_waves_scales_with_volume_and_cutoff():
    small = estimate_plane_waves(cell_volume_ang3=100.0, ecutwfc_ry=30.0)
    larger_volume = estimate_plane_waves(cell_volume_ang3=200.0, ecutwfc_ry=30.0)
    higher_cutoff = estimate_plane_waves(cell_volume_ang3=100.0, ecutwfc_ry=60.0)
    assert larger_volume > small
    assert higher_cutoff > small


def test_largest_divisor_at_most():
    assert _largest_divisor_at_most(8, 4) == 4
    assert _largest_divisor_at_most(7, 4) == 1  # 7 is prime
    assert _largest_divisor_at_most(12, 5) == 4
    assert _largest_divisor_at_most(1, 10) == 1


def test_choose_resources_stays_local_when_small():
    job = estimate_job(
        n_atoms=2, n_electrons=8, cell_volume_ang3=40.0, ecutwfc_ry=40.0, n_kpoints=8,
    )
    local = LocalResources(cpu_count=24, memory_gb=30.0, gpu_count=1)
    plan = choose_resources(job, local, allow_remote=True)
    assert plan.target == "local"
    assert plan.mpiprocs >= 1
    assert plan.npool >= 1


def test_choose_resources_routes_remote_for_large_system_when_allowed():
    job = estimate_job(
        n_atoms=200, n_electrons=800, cell_volume_ang3=4000.0, ecutwfc_ry=60.0, n_kpoints=8,
    )
    local = LocalResources(cpu_count=24, memory_gb=30.0, gpu_count=1)
    plan = choose_resources(job, local, allow_remote=True, local_atom_ceiling=40)
    assert plan.target == "remote"


def test_choose_resources_stays_local_without_remote_allowed():
    job = estimate_job(
        n_atoms=200, n_electrons=800, cell_volume_ang3=4000.0, ecutwfc_ry=60.0, n_kpoints=8,
    )
    local = LocalResources(cpu_count=24, memory_gb=30.0, gpu_count=1)
    plan = choose_resources(job, local, allow_remote=False, local_atom_ceiling=40)
    assert plan.target == "local"


def test_npool_divides_kpoint_count():
    job = estimate_job(
        n_atoms=2, n_electrons=8, cell_volume_ang3=40.0, ecutwfc_ry=40.0, n_kpoints=6,
    )
    local = LocalResources(cpu_count=24, memory_gb=30.0, gpu_count=0)
    plan = choose_resources(job, local, allow_remote=False)
    assert job.n_kpoints % plan.npool == 0


def test_quantize_mpiprocs_rounds_up_to_a_full_batch():
    assert _quantize_mpiprocs(raw=2, batch_size=8, cpu_ceiling=24) == 8
    assert _quantize_mpiprocs(raw=10, batch_size=8, cpu_ceiling=24) == 16
    assert _quantize_mpiprocs(raw=30, batch_size=8, cpu_ceiling=24) == 24  # capped at 3 batches
    assert _quantize_mpiprocs(raw=1, batch_size=8, cpu_ceiling=4) == 4  # fewer CPUs than one batch


def test_choose_resources_quantizes_local_mpiprocs_to_batches_of_8():
    job = estimate_job(
        n_atoms=2, n_electrons=8, cell_volume_ang3=40.0, ecutwfc_ry=40.0, n_kpoints=1,
    )
    local = LocalResources(cpu_count=24, memory_gb=30.0, gpu_count=0)
    plan = choose_resources(job, local, allow_remote=False)
    assert plan.mpiprocs % 8 == 0
    assert plan.mpiprocs == 8  # a 2-atom job still gets one full batch, capped by cpu_count


def test_choose_resources_routes_to_gpu_when_allowed_and_available():
    job = estimate_job(
        n_atoms=32, n_electrons=128, cell_volume_ang3=500.0, ecutwfc_ry=40.0, n_kpoints=8,
    )
    local = LocalResources(cpu_count=24, memory_gb=30.0, gpu_count=1, gpu_name="RTX 2000 Ada", gpu_memory_gb=16.0)
    plan = choose_resources(job, local, allow_gpu=True, gpu_min_atoms=8)
    assert plan.target == "local-gpu"
    assert plan.mpiprocs == local.gpu_count  # one rank per GPU, not a CPU batch


def test_choose_resources_skips_gpu_below_atom_floor():
    job = estimate_job(
        n_atoms=2, n_electrons=8, cell_volume_ang3=40.0, ecutwfc_ry=40.0, n_kpoints=1,
    )
    local = LocalResources(cpu_count=24, memory_gb=30.0, gpu_count=1, gpu_name="RTX 2000 Ada", gpu_memory_gb=16.0)
    plan = choose_resources(job, local, allow_gpu=True, gpu_min_atoms=8)
    assert plan.target == "local"


def test_choose_resources_skips_gpu_when_job_exceeds_gpu_vram():
    job = estimate_job(
        n_atoms=32, n_electrons=128, cell_volume_ang3=5_000_000.0, ecutwfc_ry=200.0, n_kpoints=8,
    )
    local = LocalResources(cpu_count=24, memory_gb=512.0, gpu_count=1, gpu_name="RTX 2000 Ada", gpu_memory_gb=16.0)
    plan = choose_resources(job, local, allow_gpu=True, gpu_min_atoms=8, local_atom_ceiling=1000)
    assert plan.target == "local"
    assert "GPU" in plan.reason

from harness_dft.environment import LocalResources
from harness_dft.estimate import (
    choose_resources,
    estimate_bands,
    estimate_job,
    estimate_plane_waves,
    _largest_divisor_at_most,
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

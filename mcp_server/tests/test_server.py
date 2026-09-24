"""Integration-style tests against a real AiiDA profile (no engine to mock,
unlike ndim-mcp -- this server imports AiiDA directly). Skipped entirely if
no profile can be loaded, e.g. in a CI environment without the dft-harness
conda env set up; run these for real inside that env before relying on the
MCP layer for anything, since neither jobs.py's outgoing-links traversal nor
the FastMCP tool signatures were exercised anywhere else.
"""
import asyncio
import json

import pytest

pytest.importorskip("aiida")

from harness_dft_mcp.atoms_io import read_atoms  # noqa: E402
from harness_dft_mcp.server import create_server  # noqa: E402

SI_CIF = """data_Si
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt "F d -3 m"
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.0 0.0 0.0
"""


@pytest.fixture(scope="module")
def mcp():
    try:
        return create_server()
    except Exception as exc:  # no loadable AiiDA profile in this environment
        pytest.skip(f"no AiiDA profile available: {exc}")


def _call(mcp, name, args):
    result = asyncio.run(mcp.call_tool(name, args))
    return json.loads(result[0].text)


def test_tool_registration_covers_every_workflow(mcp):
    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {
        "hd_status", "hd_list_pseudo_families", "hd_estimate", "hd_validate_pseudo_coverage",
        "hd_generate_2d_prototype", "hd_check_phonon_stability", "hd_rank_prototypes",
        "hd_get_job", "hd_wait_for_job", "hd_get_job_results",
        "hd_submit_relax", "hd_submit_scf", "hd_submit_bands", "hd_submit_pdos",
        "hd_submit_ph", "hd_submit_q2r", "hd_submit_matdyn",
        "hd_submit_neb", "hd_get_neb_job",
        "hd_setup_remote_computer", "hd_register_remote_code",
    }
    assert expected <= tools


def test_status_reports_local_hardware_and_pseudo_families(mcp):
    status = _call(mcp, "hd_status", {})
    assert status["local_resources"]["cpu_count"] >= 1
    assert "SSSP/1.3/PBE/efficiency" in status["pseudo_families_installed"]
    assert isinstance(status["daemon_running"], bool)


def test_submit_scf_forwards_allow_gpu_and_cpu_batch_size_to_the_plan(mcp, monkeypatch):
    """Regression test: hd_submit_scf/hd_submit_relax used to accept allow_gpu/
    cpu_batch_size/local_atom_ceiling as tool parameters but never pass them to
    build_scf_inputs/build_relax_inputs -- only allow_remote was forwarded, so
    the returned plan silently ignored the caller's GPU/batch-size choice
    (caught by an actual GPU submission returning target="local" with
    allow_gpu=True instead of "local-gpu"). This doesn't submit for real: it
    patches submit_builder to capture the plan without touching the daemon."""
    captured = {}

    def fake_submit(builder, label=None):
        captured["called"] = True
        return -1  # sentinel pk, never a real node

    # server.py does `from harness_dft.jobs import submit_builder` *inside* the
    # tool function, so patching the attribute on harness_dft.jobs (looked up
    # at call time) intercepts it without touching the real daemon.
    import harness_dft.jobs as jobs_module
    monkeypatch.setattr(jobs_module, "submit_builder", fake_submit)

    result = _call(mcp, "hd_submit_scf", {
        "structure_text": SI_CIF, "structure_format": "cif", "code_label": "pw-7.5@localhost",
        "kpoints_mesh": [2, 2, 2], "ecutwfc_ry": 30.0,
        "allow_gpu": True, "cpu_batch_size": 4, "local_atom_ceiling": 40,
    })
    assert captured.get("called"), "submit_builder was never reached"
    # With allow_gpu=True and a real GPU present, the plan must reflect it --
    # not silently fall back to a CPU plan because the flag was dropped.
    assert result["plan"]["cpu_batch_size"] == 4


def test_submit_bands_forwards_allow_gpu_and_cpu_batch_size_to_the_plan(mcp, monkeypatch):
    """Same regression as above, for hd_submit_bands (build_bands_inputs also
    used to drop allow_gpu/cpu_batch_size/local_atom_ceiling)."""
    import harness_dft.jobs as jobs_module
    monkeypatch.setattr(jobs_module, "submit_builder", lambda builder, label=None: -1)

    result = _call(mcp, "hd_submit_bands", {
        "structure_text": SI_CIF, "structure_format": "cif", "code_label": "pw-7.5@localhost",
        "kpoints_mesh": [2, 2, 2], "ecutwfc_ry": 30.0, "cpu_batch_size": 4,
    })
    assert result["scf_plan"]["cpu_batch_size"] == 4
    assert result["bands_plan"]["cpu_batch_size"] == 4


def test_submit_ph_forwards_allow_gpu_and_cpu_batch_size_to_the_plan(mcp, monkeypatch):
    """Same regression, for hd_submit_ph (build_ph_inputs had no
    allow_gpu/cpu_batch_size/local_atom_ceiling parameters at all until this
    was caught by an actual GPU phonon submission getting a CPU-style plan
    despite allow_gpu=True). Uses the most recent finished SCF-like node
    already in this dev profile as the DFPT parent, since PhBaseWorkChain's
    builder validates parent_folder against a real RemoteData output -- skips
    if none exists (e.g. a fresh profile with no prior harness-dft runs)."""
    from aiida import orm
    from aiida.common import LinkType

    query = orm.QueryBuilder().append(
        orm.WorkChainNode, filters={"attributes.process_label": "PwBaseWorkChain"}, tag="wc",
    ).order_by({"wc": {"id": "desc"}})
    parent_pk = None
    for (node,) in query.iterall():
        if node.is_finished_ok and node.base.links.get_outgoing(link_type=LinkType.RETURN).nested().get("remote_folder"):
            parent_pk = node.pk
            break
    if parent_pk is None:
        pytest.skip("no finished PwBaseWorkChain with a remote_folder output in this profile yet")

    import harness_dft.jobs as jobs_module
    monkeypatch.setattr(jobs_module, "submit_builder", lambda builder, label=None: -1)

    result = _call(mcp, "hd_submit_ph", {
        "parent_scf_pk": parent_pk, "ph_code_label": "ph-7.5@localhost",
        "structure_text": SI_CIF, "structure_format": "cif",
        "pseudo_family_label": "SSSP/1.3/PBE/efficiency", "ecutwfc_ry": 30.0,
        "qpoints_mesh": [1, 1, 1], "allow_gpu": True, "cpu_batch_size": 4,
    })
    assert result["plan"]["cpu_batch_size"] == 4


def test_estimate_quantizes_mpiprocs_to_a_batch_and_does_not_submit(mcp):
    before = _call(mcp, "hd_status", {})
    estimate = _call(mcp, "hd_estimate", {"structure_text": SI_CIF, "structure_format": "cif"})
    assert estimate["job"]["n_atoms"] == 8
    assert estimate["plan"]["mpiprocs"] % estimate["plan"]["cpu_batch_size"] == 0
    after = _call(mcp, "hd_status", {})
    assert before["codes"] == after["codes"]  # read-only: nothing registered/submitted


def test_generate_2d_prototype_honeycomb_is_ready_for_2d_relax(mcp):
    result = _call(mcp, "hd_generate_2d_prototype", {"element": "Al", "prototype": "honeycomb"})
    assert result["n_atoms"] == 2
    assert result["kpoints_mesh_suggestion"] == [9, 9, 1]
    assert result["required_relax_settings"]["cell_dofree"] == "2Dxy"
    # round-trips through read_atoms the same way any hd_submit_* tool would
    atoms = read_atoms(result["structure_text"], result["structure_format"])
    assert len(atoms) == 2
    assert not atoms.pbc[2]


def test_generate_2d_prototype_puckered_has_fourfold_cell(mcp):
    result = _call(mcp, "hd_generate_2d_prototype", {"element": "P", "prototype": "puckered"})
    assert result["n_atoms"] == 4
    atoms = read_atoms(result["structure_text"], result["structure_format"])
    assert len(atoms) == 4


def test_validate_pseudo_coverage_covers_silicon_in_every_installed_family(mcp):
    for family in ("SSSP/1.3/PBE/efficiency", "SSSP/1.3/PBE/precision",
                   "SSSP/1.3/PBEsol/efficiency", "SSSP/1.3/PBEsol/precision"):
        result = _call(mcp, "hd_validate_pseudo_coverage",
                        {"structure_text": SI_CIF, "structure_format": "cif", "pseudo_family_label": family})
        assert result["covered"] is True, f"{family} should cover Si: {result}"


def test_check_phonon_stability_tool_reaches_a_real_finished_matdyn_job(mcp):
    from aiida import orm

    query = orm.QueryBuilder().append(
        orm.WorkChainNode, filters={"attributes.process_label": "MatdynBaseWorkChain"}, tag="wc",
    ).order_by({"wc": {"id": "desc"}})
    pk = next((node.pk for (node,) in query.iterall() if node.is_finished_ok), None)
    if pk is None:
        pytest.skip("no finished MatdynBaseWorkChain in this profile yet")

    result = _call(mcp, "hd_check_phonon_stability", {"matdyn_pk": pk})
    assert "stable" in result and "min_frequency_thz" in result


def test_rank_prototypes_tool_reaches_real_finished_scf_jobs(mcp):
    from aiida import orm

    query = orm.QueryBuilder().append(
        orm.WorkChainNode, filters={"attributes.process_label": "PwBaseWorkChain"}, tag="wc",
    ).order_by({"wc": {"id": "desc"}})
    pks = []
    for (node,) in query.iterall():
        if node.is_finished_ok:
            params = getattr(node.outputs, "output_parameters", None)
            if params is not None and "energy" in params.get_dict():
                pks.append(node.pk)
        if len(pks) >= 2:
            break
    if len(pks) < 2:
        pytest.skip("need at least 2 finished SCF-like PwBaseWorkChain nodes in this profile")

    result = _call(mcp, "hd_rank_prototypes", {"candidates": [{"label": f"c{i}", "pk": pk} for i, pk in enumerate(pks)]})
    assert len(result["ranked"]) == 2
    assert result["ground_state_label"] == result["ranked"][0]["label"]

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


def test_estimate_quantizes_mpiprocs_to_a_batch_and_does_not_submit(mcp):
    before = _call(mcp, "hd_status", {})
    estimate = _call(mcp, "hd_estimate", {"structure_text": SI_CIF, "structure_format": "cif"})
    assert estimate["job"]["n_atoms"] == 8
    assert estimate["plan"]["mpiprocs"] % estimate["plan"]["cpu_batch_size"] == 0
    after = _call(mcp, "hd_status", {})
    assert before["codes"] == after["codes"]  # read-only: nothing registered/submitted


def test_validate_pseudo_coverage_covers_silicon_in_every_installed_family(mcp):
    for family in ("SSSP/1.3/PBE/efficiency", "SSSP/1.3/PBE/precision",
                   "SSSP/1.3/PBEsol/efficiency", "SSSP/1.3/PBEsol/precision"):
        result = _call(mcp, "hd_validate_pseudo_coverage",
                        {"structure_text": SI_CIF, "structure_format": "cif", "pseudo_family_label": family})
        assert result["covered"] is True, f"{family} should cover Si: {result}"

"""Integration-style test against real finished SCF/relax jobs already in
this AiiDA profile (bulk Si, from earlier CPU/GPU validation runs)."""
import pytest

pytest.importorskip("aiida")

from aiida import load_profile  # noqa: E402

try:
    load_profile()
except Exception as exc:
    pytest.skip(f"no AiiDA profile available: {exc}", allow_module_level=True)

from aiida import orm  # noqa: E402

from harness_dft.screening import rank_prototypes  # noqa: E402


def _two_most_recent_finished_scf(process_label: str, n=2):
    """Only nodes whose output_parameters actually report a total energy --
    an SCF/relax step, not e.g. an NSCF/bands sub-step (which QE doesn't
    compute a total energy for)."""
    query = orm.QueryBuilder().append(
        orm.WorkChainNode, filters={"attributes.process_label": process_label}, tag="wc",
    ).order_by({"wc": {"id": "desc"}})
    pks = []
    for (node,) in query.iterall():
        if not node.is_finished_ok:
            continue
        output_parameters = getattr(node.outputs, "output_parameters", None)
        if output_parameters is not None and "energy" in output_parameters.get_dict():
            pks.append(node.pk)
        if len(pks) >= n:
            break
    return pks


def test_rank_prototypes_sorts_by_energy_per_atom_and_flags_the_lowest():
    pks = _two_most_recent_finished_scf("PwBaseWorkChain")
    if len(pks) < 2:
        pytest.skip("need at least 2 finished PwBaseWorkChain nodes with output_parameters in this profile")

    candidates = [{"label": f"candidate-{i}", "pk": pk} for i, pk in enumerate(pks)]
    result = rank_prototypes(candidates)

    assert len(result.ranked) == 2
    assert result.ground_state_label == result.ranked[0].label
    # sorted ascending by energy per atom (most stable/lowest energy first)
    assert result.ranked[0].energy_per_atom_ev <= result.ranked[1].energy_per_atom_ev
    assert result.ranked[0].delta_from_lowest_ev_per_atom == 0.0
    assert result.ranked[1].delta_from_lowest_ev_per_atom >= 0.0

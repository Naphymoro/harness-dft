"""Integration-style tests against a real AiiDA profile (no lightweight fake
AiiDA to mock against). Skipped entirely if no profile can be loaded."""
import pytest

pytest.importorskip("aiida")

from aiida import load_profile  # noqa: E402

try:
    load_profile()
except Exception as exc:  # no loadable AiiDA profile in this environment
    pytest.skip(f"no AiiDA profile available: {exc}", allow_module_level=True)

from aiida import orm  # noqa: E402

from harness_dft.jobs import get_results  # noqa: E402


def _most_recent_finished(process_label: str):
    query = orm.QueryBuilder().append(
        orm.WorkChainNode, filters={"attributes.process_label": process_label}, tag="wc",
    ).order_by({"wc": {"id": "desc"}})
    for (node,) in query.iterall():
        if node.is_finished_ok:
            return node.pk
    return None


def test_get_results_flattens_flat_output_namespace():
    """A plain PwBaseWorkChain's outputs (output_parameters etc.) are not
    namespaced -- must come back as top-level keys, not nested under
    anything."""
    pk = _most_recent_finished("PwBaseWorkChain")
    if pk is None:
        pytest.skip("no finished PwBaseWorkChain in this profile yet")
    results = get_results(pk)
    assert "output_parameters" in results
    # Not nested under anything -- a flat top-level key, unlike the namespaced
    # case below. Don't assert on 'energy' specifically: the most recent
    # PwBaseWorkChain in this dev profile may be an NSCF/bands sub-step,
    # which doesn't report a total energy the way an SCF step does.
    assert results["output_parameters"], "output_parameters must not be empty"


def test_get_results_recurses_into_namespaced_sub_outputs():
    """Regression test: get_results used to check `isinstance(value, dict)`
    on the *nested()* output tree and silently `continue` past namespaced
    sub-outputs (e.g. a PdosWorkChain's `dos.output_dos`, `projwfc.Pdos`) --
    caught by a real GPU PDOS submission coming back with an empty results
    dict despite finishing successfully. Must recurse instead of dropping."""
    pk = _most_recent_finished("PdosWorkChain")
    if pk is None:
        pytest.skip("no finished PdosWorkChain in this profile yet")
    results = get_results(pk)
    assert results, "results must not be empty for a successfully finished PdosWorkChain"
    for namespace in ("dos", "projwfc", "nscf"):
        assert namespace in results, f"missing namespace {namespace!r} in {list(results)}"
        assert isinstance(results[namespace], dict) and results[namespace], (
            f"{namespace!r} sub-namespace was dropped or left empty"
        )
    assert "output_parameters" in results["dos"]
    assert "Pdos" in results["projwfc"]

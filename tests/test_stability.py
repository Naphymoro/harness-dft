"""Integration-style test against a real finished matdyn job in this AiiDA
profile (bulk Si, pk 631 from earlier GPU-phonon-chain validation)."""
import pytest

pytest.importorskip("aiida")

from aiida import load_profile  # noqa: E402

try:
    load_profile()
except Exception as exc:
    pytest.skip(f"no AiiDA profile available: {exc}", allow_module_level=True)

from aiida import orm  # noqa: E402

from harness_dft.stability import check_dynamical_stability  # noqa: E402


BULK_SI_MATDYN_PK = 631  # pinned, not "most recent": this profile now also holds a genuinely
                          # unstable 2D candidate's matdyn result (see docs/2d-screening.md), and
                          # "most recent finished MatdynBaseWorkChain" would silently pick that up
                          # instead of bulk Si once it exists -- exactly what happened here first.


def _node_exists(pk):
    try:
        orm.load_node(pk)
        return True
    except Exception:
        return False


def test_bulk_silicon_phonons_are_dynamically_stable():
    """Bulk Si has no imaginary phonon modes -- a real, physically known-good
    case. The historical run (pk 631) predates the asr='simple' fix, so its
    small Gamma-point numerical noise (~-0.39 THz, seen directly in that
    run's raw output) should still fall inside the default tolerance."""
    if not _node_exists(BULK_SI_MATDYN_PK):
        pytest.skip(f"pk={BULK_SI_MATDYN_PK} (bulk Si matdyn) not present in this profile")
    result = check_dynamical_stability(BULK_SI_MATDYN_PK)
    assert result.stable, f"bulk Si should be dynamically stable, got: {result}"
    assert result.n_qpoints > 0
    assert result.n_modes > 0
    assert not result.imaginary_modes


def test_tolerance_can_be_tightened_to_flag_the_known_numerical_noise():
    """The same run's small Gamma-point artifact (~-0.39 THz) is real enough
    to show up if the tolerance is tightened past it -- confirms the
    tolerance actually does something, rather than always reporting stable."""
    if not _node_exists(BULK_SI_MATDYN_PK):
        pytest.skip(f"pk={BULK_SI_MATDYN_PK} (bulk Si matdyn) not present in this profile")
    loose = check_dynamical_stability(BULK_SI_MATDYN_PK, tolerance_thz=-0.5)
    tight = check_dynamical_stability(BULK_SI_MATDYN_PK, tolerance_thz=-0.01)
    if loose.min_frequency_thz < -0.01:
        assert not tight.stable
        assert tight.imaginary_modes

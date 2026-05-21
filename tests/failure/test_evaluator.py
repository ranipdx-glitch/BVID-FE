import numpy as np
import pytest

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.failure.evaluator import FailureEvaluator, LaminateFailureReport


def test_evaluator_tsai_wu_single_stress_at_Xt():
    m = MATERIAL_LIBRARY["IM7/8552"]
    ev = FailureEvaluator(m, criterion="tsai_wu")
    stress_field = np.array([[[m.Xt, 0, 0, 0, 0, 0]]])  # 1 elem, 1 gp, 6 components
    rpt = ev.evaluate(stress_field)
    assert isinstance(rpt, LaminateFailureReport)
    assert abs(rpt.max_index - 1.0) < 0.05
    assert rpt.critical_element == 0
    assert rpt.critical_gauss_point == 0


def test_evaluator_larc05_matches_expected_mode():
    m = MATERIAL_LIBRARY["IM7/8552"]
    ev = FailureEvaluator(m, criterion="larc05")
    stress_field = np.array([[[0, m.Yt, 0, 0, 0, 0]]])
    rpt = ev.evaluate(stress_field)
    assert abs(rpt.max_index - 1.0) < 1e-6


def test_evaluator_critical_element_picked():
    m = MATERIAL_LIBRARY["IM7/8552"]
    ev = FailureEvaluator(m, criterion="tsai_wu")
    stress_field = np.zeros((3, 2, 6))
    stress_field[1, 1, :] = [m.Xt, 0, 0, 0, 0, 0]  # element 1, gp 1 is critical
    rpt = ev.evaluate(stress_field)
    assert rpt.critical_element == 1
    assert rpt.critical_gauss_point == 1


def test_evaluator_unknown_criterion_raises():
    m = MATERIAL_LIBRARY["IM7/8552"]
    with pytest.raises(ValueError):
        FailureEvaluator(m, criterion="bogus")


def test_criterion_registry_contains_known_keys():
    """The registry is the single source of truth for available criteria;
    asserting its keys here pins the supported set so any future addition
    has to update this test deliberately."""
    from bvidfe.failure.evaluator import _CRITERION_REGISTRY

    assert set(_CRITERION_REGISTRY.keys()) == {"tsai_wu", "larc05", "puck"}


def test_evaluator_unknown_criterion_error_lists_valid_names():
    """The ValueError raised on bad input must name every valid criterion
    so users get an actionable message instead of a bare 'unknown
    criterion' string."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    with pytest.raises(ValueError) as excinfo:
        FailureEvaluator(m, criterion="bogus")
    msg = str(excinfo.value)
    assert "tsai_wu" in msg
    assert "larc05" in msg


def test_evaluator_caches_function_pointer():
    """``__init__`` must store the dispatch function on ``self`` so
    ``evaluate`` no longer branches on raw strings."""
    from bvidfe.failure.larc05 import larc05_index_batch
    from bvidfe.failure.tsai_wu import tsai_wu_index_batch

    m = MATERIAL_LIBRARY["IM7/8552"]
    ev_tw = FailureEvaluator(m, criterion="tsai_wu")
    ev_lc = FailureEvaluator(m, criterion="larc05")
    assert ev_tw._evaluate_fn is tsai_wu_index_batch
    assert ev_lc._evaluate_fn is larc05_index_batch


def test_evaluate_matches_scalar_loop():
    """Lock the vectorised FailureEvaluator.evaluate against the prior
    nested-Python-loop form on a random (3, 4, 6) stress field. Tsai-Wu
    and LaRC05 each evaluated separately because the two criteria have
    different numerical pathways."""
    from bvidfe.failure.larc05 import larc05_index
    from bvidfe.failure.tsai_wu import tsai_wu_index

    m = MATERIAL_LIBRARY["IM7/8552"]
    rng = np.random.default_rng(7)
    field = rng.standard_normal((3, 4, 6)) * 200.0  # MPa-scale stresses

    for crit, scalar_fn in (("tsai_wu", tsai_wu_index), ("larc05", larc05_index)):
        ev = FailureEvaluator(m, criterion=crit)
        rpt = ev.evaluate(field)
        # Reconstruct the same answer with the explicit scalar loop
        max_idx, crit_e, crit_g = -1.0, 0, 0
        for e in range(field.shape[0]):
            for g in range(field.shape[1]):
                idx = scalar_fn(m, field[e, g])
                if idx > max_idx:
                    max_idx, crit_e, crit_g = idx, e, g
        assert rpt.max_index == pytest.approx(max_idx, rel=1e-12, abs=1e-9)
        assert rpt.critical_element == crit_e
        assert rpt.critical_gauss_point == crit_g


def test_larc05_index_batch_matches_scalar():
    """Numerical equivalence between larc05_index_batch and the scalar
    larc05_index on a random batch — locks the np.where mode-selection
    against the if/else scalar branches."""
    from bvidfe.failure.larc05 import larc05_index, larc05_index_batch

    m = MATERIAL_LIBRARY["IM7/8552"]
    rng = np.random.default_rng(11)
    stresses = rng.standard_normal((25, 6)) * 300.0
    batch = larc05_index_batch(m, stresses)
    scalar = np.array([larc05_index(m, s) for s in stresses])
    np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-9)


def test_tsai_wu_index_batch_matches_scalar():
    """Numerical equivalence between tsai_wu_index_batch and the scalar
    tsai_wu_index on a random batch — locks the einsum form against
    F.dot(s) + s.dot(Q.dot(s))."""
    from bvidfe.failure.tsai_wu import tsai_wu_index, tsai_wu_index_batch

    m = MATERIAL_LIBRARY["IM7/8552"]
    rng = np.random.default_rng(13)
    stresses = rng.standard_normal((25, 6)) * 300.0
    batch = tsai_wu_index_batch(m, stresses)
    scalar = np.array([tsai_wu_index(m, s) for s in stresses])
    np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-9)


def test_index_batch_preserves_leading_axes():
    """Both batch helpers must accept (n, m, 6) and return (n, m), so
    FailureEvaluator can pass its (n_elem, n_gp, 6) array unchanged."""
    from bvidfe.failure.larc05 import larc05_index_batch
    from bvidfe.failure.tsai_wu import tsai_wu_index_batch

    m = MATERIAL_LIBRARY["IM7/8552"]
    rng = np.random.default_rng(17)
    stresses = rng.standard_normal((4, 8, 6)) * 100.0
    assert larc05_index_batch(m, stresses).shape == (4, 8)
    assert tsai_wu_index_batch(m, stresses).shape == (4, 8)

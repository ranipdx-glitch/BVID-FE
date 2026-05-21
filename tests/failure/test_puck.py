"""Tests for the Puck (1998) action-plane failure criterion.

Reference points are computed by hand from the Puck-Schürmann master
fracture body. The three pinned cases are:

  * Pure axial tension sigma_1 = X_t        -> fiber-mode index = 1.0
  * Pure axial compression sigma_1 = -X_c   -> fiber-mode index = 1.0
  * Pure transverse tension sigma_2 = Y_t   -> IFF Mode-A index = 1.0
"""

from __future__ import annotations

import numpy as np
import pytest

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.failure.evaluator import (
    _CRITERION_REGISTRY,
    _CRITERION_SCALAR_REGISTRY,
    FailureEvaluator,
)
from bvidfe.failure.puck import puck_index, puck_index_batch

# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_puck_in_batch_registry():
    assert "puck" in _CRITERION_REGISTRY
    assert _CRITERION_REGISTRY["puck"] is puck_index_batch


def test_puck_in_scalar_registry():
    assert "puck" in _CRITERION_SCALAR_REGISTRY
    assert _CRITERION_SCALAR_REGISTRY["puck"] is puck_index


def test_failure_evaluator_constructs_with_puck():
    m = MATERIAL_LIBRARY["IM7/8552"]
    ev = FailureEvaluator(m, criterion="puck")
    assert ev.criterion == "puck"
    assert ev._evaluate_fn is puck_index_batch
    assert ev._scalar_fn is puck_index


# ---------------------------------------------------------------------------
# Hand-computed reference points
# ---------------------------------------------------------------------------


def test_pure_axial_tension_at_Xt_is_one():
    """sigma_1 = X_t, all other components zero -> fiber-mode index = 1.0."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    idx = puck_index(m, [m.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert idx == pytest.approx(1.0, abs=1e-9)


def test_pure_axial_compression_at_minus_Xc_is_one():
    """sigma_1 = -X_c, all other components zero -> fiber-mode index = 1.0."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    idx = puck_index(m, [-m.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert idx == pytest.approx(1.0, abs=1e-9)


def test_pure_transverse_tension_at_Yt_is_one():
    """sigma_2 = Y_t, all other components zero -> IFF Mode-A index = 1.0 on
    the theta = 0 fracture plane (sigma_n = Y_t, tau_nt = tau_n1 = 0)."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    idx = puck_index(m, [0.0, m.Yt, 0.0, 0.0, 0.0, 0.0])
    # theta = 0 lies exactly on the 37-point grid (centre sample), so the
    # IFF index there is analytic-exact; no angular-resolution slop.
    assert idx == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Benign-state sanity check
# ---------------------------------------------------------------------------


def test_benign_stress_state_well_below_failure():
    """A stress state at 10% of every strength loaded *one component at a
    time* should yield a Puck index well below 0.2 — far from failure
    for any criterion mode. (Loading every component simultaneously
    sums under the square-root, so the SRSS-style envelope climbs faster
    than a Tsai-Wu quadratic; pinning each component in isolation is
    the cleaner sanity bound.)"""
    m = MATERIAL_LIBRARY["IM7/8552"]
    for component_idx, scale in enumerate([m.Xt, m.Yt, m.Yt, m.S23, m.S12, m.S12]):
        stress = [0.0] * 6
        stress[component_idx] = 0.1 * scale
        idx = puck_index(m, stress)
        assert idx < 0.2, f"component {component_idx} gave index {idx}"


# ---------------------------------------------------------------------------
# Vectorised vs scalar equivalence
# ---------------------------------------------------------------------------


def test_puck_index_batch_matches_scalar():
    """Numerical equivalence between puck_index_batch and the scalar
    puck_index across a random batch — both walk the same 37-point
    action-plane sweep."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    rng = np.random.default_rng(19)
    stresses = rng.standard_normal((25, 6)) * 300.0
    batch = puck_index_batch(m, stresses)
    scalar = np.array([puck_index(m, s) for s in stresses])
    np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-9)


def test_puck_index_batch_preserves_leading_axes():
    """``puck_index_batch`` must accept ``(n, m, 6)`` and return ``(n, m)``
    so ``FailureEvaluator.evaluate`` can pass its ``(n_elem, n_gp, 6)``
    stress array unchanged."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    rng = np.random.default_rng(23)
    stresses = rng.standard_normal((4, 8, 6)) * 100.0
    assert puck_index_batch(m, stresses).shape == (4, 8)


def test_puck_zero_stress_gives_zero():
    m = MATERIAL_LIBRARY["IM7/8552"]
    assert puck_index(m, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == 0.0


def test_puck_via_evaluator():
    """FailureEvaluator with criterion='puck' returns the same index as
    direct puck_index calls — pins the dispatcher wiring."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    ev = FailureEvaluator(m, criterion="puck")
    stress_field = np.array([[[0.0, m.Yt, 0.0, 0.0, 0.0, 0.0]]])
    rpt = ev.evaluate(stress_field)
    assert rpt.max_index == pytest.approx(1.0, abs=1e-9)
    assert rpt.criterion == "puck"

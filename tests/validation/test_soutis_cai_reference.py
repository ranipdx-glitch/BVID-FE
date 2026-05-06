"""Soutis CAI knockdown self-consistent regression tests.

The Soutis & Curtis (1996) compression-after-impact model implemented in
``bvidfe.failure.soutis_openhole.soutis_cai`` is

    knockdown = 1 / (1 + k_s * (DPA / A_panel)^m)
    sigma_CAI = knockdown * sigma_pristine

with material-specific constants ``k_s`` and ``m`` (defaults 2.5 and 0.5
for typical CFRP, calibrated to T300/914 in the original paper).

These tests pin the formula's behaviour to itself — i.e. the
implementation must remain in agreement with the closed-form expression
across boundary conditions (zero damage, normalised damage invariance,
known calibration constants). True literature-pinned values would
require digitising Soutis 1996 Figure 3 and is left as a follow-up.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.failure.soutis_openhole import soutis_cai


def test_soutis_cai_returns_pristine_at_zero_dpa():
    """DPA = 0 must short-circuit to sigma_pristine exactly."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    sigma = soutis_cai(m, dpa_mm2=0.0, A_panel_mm2=15000.0, sigma_pristine_MPa=500.0)
    assert sigma == 500.0


def test_soutis_cai_matches_closed_form_low_damage():
    """Implementation must match knockdown = 1 / (1 + k_s*(DPA/A)^m)."""
    m = MATERIAL_LIBRARY["IM7/8552"]  # k_s=2.5, m=0.5
    dpa, A_panel, sigma_0 = 100.0, 15000.0, 500.0
    sigma = soutis_cai(m, dpa, A_panel, sigma_0)
    expected = sigma_0 / (1.0 + m.soutis_k_s * (dpa / A_panel) ** m.soutis_m)
    assert sigma == pytest.approx(expected, rel=1e-12)


def test_soutis_cai_matches_closed_form_moderate_damage():
    m = MATERIAL_LIBRARY["IM7/8552"]
    dpa, A_panel, sigma_0 = 500.0, 15000.0, 500.0
    sigma = soutis_cai(m, dpa, A_panel, sigma_0)
    expected = sigma_0 / (1.0 + m.soutis_k_s * (dpa / A_panel) ** m.soutis_m)
    assert sigma == pytest.approx(expected, rel=1e-12)


def test_soutis_cai_dimensionless_invariance():
    """Same DPA/A_panel ratio must yield the same knockdown regardless of
    absolute scale — the formula is purely dimensionless in that ratio."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    sigma_0 = 400.0
    s_small = soutis_cai(m, dpa_mm2=100.0, A_panel_mm2=10_000.0, sigma_pristine_MPa=sigma_0)
    s_large = soutis_cai(m, dpa_mm2=400.0, A_panel_mm2=40_000.0, sigma_pristine_MPa=sigma_0)
    assert s_small == pytest.approx(s_large, rel=1e-12)


def test_soutis_cai_monotonically_decreases_with_damage():
    """For a fixed panel + material, the residual strength must decrease as
    DPA grows."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    A, sigma_0 = 15_000.0, 500.0
    series = [soutis_cai(m, dpa, A, sigma_0) for dpa in (10, 100, 500, 1500, 5000)]
    for prev, nxt in zip(series, series[1:]):
        assert prev > nxt, f"non-monotone: {series}"


def test_soutis_cai_responds_to_calibration_constants():
    """Doubling k_s at a fixed DPA/A must lower the knockdown — confirms
    the calibration constants flow into the formula correctly."""
    base = MATERIAL_LIBRARY["IM7/8552"]
    stiffer = replace(base, soutis_k_s=2 * base.soutis_k_s)
    sigma_base = soutis_cai(base, 200.0, 15_000.0, 500.0)
    sigma_stiff = soutis_cai(stiffer, 200.0, 15_000.0, 500.0)
    assert sigma_stiff < sigma_base

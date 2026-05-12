"""Whitney-Nuismer TAI analytical-limit validation.

The point-stress TAI formula in ``failure.soutis_openhole.whitney_nuismer_tai``
has two analytical limits that must hold for ANY material and any DPA:

  (a) DPA -> 0 : knockdown -> 1.0 (no hole, no strength loss).
  (b) DPA -> inf : xi -> 1, denom = 2 + 1 + 3 - (Kt_inf - 3)*(5 - 7) = 2 * Kt_inf;
      so knockdown -> 2 / (2 * Kt_inf) = 1 / Kt_inf. With the default
      Kt_inf = 3 the asymptote is exactly 1/3.

  (c) Monotonicity: knockdown decreases as DPA increases (the hole-effect
      stress concentration intensifies with hole size).

These three limits exercise pure algebra in the formula and should hold
to machine precision once xi is sufficiently far from its boundaries.
"""

from __future__ import annotations

import math

import pytest

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.failure.soutis_openhole import whitney_nuismer_tai


def test_wn_returns_pristine_at_zero_dpa():
    """The formula short-circuits to sigma_pristine at dpa <= 0."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    assert whitney_nuismer_tai(m, dpa_mm2=0.0, sigma_pristine_MPa=600.0) == 600.0


def test_wn_knockdown_approaches_unity_for_tiny_dpa():
    """As DPA -> 0+, knockdown should approach (but not exactly equal) 1.0."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    sigma_0 = 600.0
    # Pick DPA so xi = R/(R + d0) is small (<< 1)
    sigma = whitney_nuismer_tai(m, dpa_mm2=1e-4, sigma_pristine_MPa=sigma_0)
    assert sigma == pytest.approx(sigma_0, rel=1e-3)
    assert sigma <= sigma_0  # never exceeds pristine


def test_wn_knockdown_asymptote_at_infinite_dpa_equals_one_third():
    """For Kt_inf=3, the formula's asymptote is 1/3 as DPA -> inf (xi -> 1)."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    sigma_0 = 600.0
    sigma = whitney_nuismer_tai(m, dpa_mm2=1e10, sigma_pristine_MPa=sigma_0, Kt_inf=3.0)
    expected_asymptote = sigma_0 / 3.0
    assert sigma == pytest.approx(expected_asymptote, rel=1e-4)


def test_wn_asymptote_scales_with_inverse_Kt_inf():
    """For arbitrary Kt_inf the asymptote is 1 / Kt_inf (algebraic, see
    module docstring). Verify with Kt_inf = 2 and Kt_inf = 5."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    sigma_0 = 600.0
    sigma2 = whitney_nuismer_tai(m, dpa_mm2=1e10, sigma_pristine_MPa=sigma_0, Kt_inf=2.0)
    sigma5 = whitney_nuismer_tai(m, dpa_mm2=1e10, sigma_pristine_MPa=sigma_0, Kt_inf=5.0)
    assert sigma2 == pytest.approx(sigma_0 / 2.0, rel=1e-4)
    # Convergence to the algebraic asymptote is slower at higher Kt_inf;
    # at dpa=1e10 the Kt_inf=5 residual is ~1.2e-4 relative.
    assert sigma5 == pytest.approx(sigma_0 / 5.0, rel=5e-4)


def test_wn_monotonically_decreases_with_dpa():
    """sigma_TAI must decrease as DPA grows."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    sigma_0 = 600.0
    series = [
        whitney_nuismer_tai(m, dpa_mm2=dpa, sigma_pristine_MPa=sigma_0)
        for dpa in (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)
    ]
    for prev, nxt in zip(series, series[1:]):
        assert prev > nxt, f"non-monotone: {series}"


def test_wn_knockdown_bounded_below_by_asymptote():
    """For Kt_inf=3 no finite DPA should yield knockdown < 1/3 (the
    asymptote is the lower bound)."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    sigma_0 = 600.0
    for dpa in (1, 10, 100, 1000, 10_000, 1e6):
        sigma = whitney_nuismer_tai(m, dpa_mm2=dpa, sigma_pristine_MPa=sigma_0, Kt_inf=3.0)
        # Allow a tiny slack for floating-point drift near the asymptote.
        assert sigma >= sigma_0 / 3.0 - 1e-9, f"undershoot at DPA={dpa}: {sigma}"
        assert math.isfinite(sigma)

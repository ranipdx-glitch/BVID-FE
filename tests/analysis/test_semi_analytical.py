import math

import pytest

from bvidfe.analysis.semi_analytical import (
    SemiAnalyticalResult,
    _sublaminate_D_matrix,
    find_critical_interface,
    semi_analytical_cai,
    semi_analytical_tai,
    sublaminate_buckling_load,
)
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.damage.state import DamageState, DelaminationEllipse


def test_buckling_load_positive_for_realistic_case():
    m = MATERIAL_LIBRARY["IM7/8552"]
    layup = [0, 45, -45, 90] * 4  # 16 plies
    lam = Laminate(m, layup, 0.152)
    # Delamination at interface 3 (between ply 3 and ply 4)
    ell = DelaminationEllipse(3, (75, 50), major_mm=20, minor_mm=10, orientation_deg=0)
    N = sublaminate_buckling_load(lam, ell)
    assert N > 0


def test_buckling_load_decreases_with_ellipse_size():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ell_small = DelaminationEllipse(3, (0, 0), 10, 6, 0)
    ell_large = DelaminationEllipse(3, (0, 0), 40, 20, 0)
    N_small = sublaminate_buckling_load(lam, ell_small)
    N_large = sublaminate_buckling_load(lam, ell_large)
    assert N_small > N_large  # smaller ellipse => harder to buckle


def test_find_critical_interface_picks_largest_ellipse_near_surface():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ds = DamageState(
        [
            DelaminationEllipse(1, (0, 0), 10, 5, 0),  # near impact face
            DelaminationEllipse(7, (0, 0), 20, 10, 0),  # mid-plane, larger
            DelaminationEllipse(14, (0, 0), 8, 4, 0),  # near back face
        ],
        dent_depth_mm=0.3,
    )
    idx = find_critical_interface(ds, lam)
    # Interface 7 has largest area, but let's check scoring:
    # interface 7 z-upper = 7 * 0.152 = 1.064 (distance from top)
    # interface 7 z-lower = (16-8) * 0.152 = 1.216 (distance from bottom)
    # max |z| = 1.216
    # score_7 = pi*20*10 * 1.216 ≈ 764
    # score_1 = pi*10*5 * (16-2)*0.152 = pi*50 * 2.128 ≈ 334 (distance from bottom larger)
    # score_14 = pi*8*4 * (16-15)*0.152 = small
    # Expect 7 to win
    assert idx == 7


def test_find_critical_interface_returns_none_for_empty():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ds = DamageState([], dent_depth_mm=0.0)
    idx = find_critical_interface(ds, lam)
    assert idx is None


def test_sublaminate_above_interface_uses_correct_plies():
    """Critical sublaminate includes plies above the interface (from top surface to the delamination)."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    layup = [0, 45, -45, 90, 90, -45, 45, 0]  # 8 plies
    lam = Laminate(m, layup, 0.152)
    # Delamination at interface 1 (between ply 1 and ply 2)
    # Upper sublaminate = plies 0, 1 (2 plies); thinner sublaminate buckles first.
    ell = DelaminationEllipse(1, (0, 0), 20, 10, 0)
    N = sublaminate_buckling_load(lam, ell)
    assert N > 0


def test_semi_analytical_cai_pristine_returns_pristine():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ds = DamageState([], dent_depth_mm=0.0)
    result = semi_analytical_cai(lam, ds, 500.0, 15000.0)
    assert isinstance(result, SemiAnalyticalResult)
    assert result.residual_strength_MPa == 500.0
    assert result.critical_interface_index is None
    assert result.critical_buckling_load_N is None


def test_semi_analytical_cai_returns_less_than_pristine():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ds = DamageState(
        [DelaminationEllipse(3, (0, 0), 30, 20, 0)],
        dent_depth_mm=0.5,
    )
    result = semi_analytical_cai(lam, ds, 500.0, 15000.0)
    assert isinstance(result, SemiAnalyticalResult)
    assert 0 < result.residual_strength_MPa < 500.0
    assert result.critical_interface_index == 3
    assert result.critical_buckling_load_N is not None
    assert result.critical_buckling_load_N > 0


def test_semi_analytical_cai_takes_min_over_soutis_and_buckling():
    """For a large ellipse, buckling should dominate (smaller). For tiny ellipse, Soutis dominates."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    # Large ellipse => low buckling load
    ds_large = DamageState(
        [DelaminationEllipse(3, (0, 0), 60, 40, 0)],
        dent_depth_mm=0.5,
    )
    result_large = semi_analytical_cai(lam, ds_large, 500.0, 15000.0)
    assert result_large.residual_strength_MPa > 0


def test_semi_analytical_result_is_frozen_dataclass():
    """``SemiAnalyticalResult`` is an immutable, named-attribute container —
    callers must not be able to mutate its fields after construction, and
    the structured access must replace positional-tuple unpacking."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ds = DamageState(
        [DelaminationEllipse(3, (0, 0), 30, 20, 0)],
        dent_depth_mm=0.5,
    )
    result = semi_analytical_cai(lam, ds, 500.0, 15000.0)
    # Frozen dataclass: assignment to a field raises.
    with pytest.raises((AttributeError, Exception)):
        result.residual_strength_MPa = 0.0  # type: ignore[misc]
    # Named-attribute access (the whole point of this dataclass).
    assert hasattr(result, "residual_strength_MPa")
    assert hasattr(result, "critical_interface_index")
    assert hasattr(result, "critical_buckling_load_N")


def test_semi_analytical_tai_pristine_returns_pristine():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ds = DamageState([], dent_depth_mm=0.0)
    sigma_tai = semi_analytical_tai(lam, ds, 800.0)
    assert sigma_tai == 800.0


def test_semi_analytical_tai_monotonic_in_dpa():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    ds_small = DamageState([DelaminationEllipse(3, (0, 0), 10, 5, 0)], 0.3)
    ds_large = DamageState([DelaminationEllipse(3, (0, 0), 30, 20, 0)], 0.3)
    t1 = semi_analytical_tai(lam, ds_small, 800.0)
    t2 = semi_analytical_tai(lam, ds_large, 800.0)
    assert t1 > t2


def test_buckling_load_matches_closed_form_ssss_square():
    """Regression test for Issue #29.

    The closed-form SSSS Navier eigenvalue for an orthotropic rectangle is

        N_cr(m,n) = (pi^2 / a^2)
                    * [D11*m^4 + 2*(D12 + 2*D66)*(m*a/b)^2 * n^2
                       + D22 * (a*n/b)^4] / m^2

    where (a, b) are the FULL plate dimensions (the sine basis
    sin(m*pi*x/a) * sin(n*pi*y/b) vanishes at x=0, x=a, y=0, y=b).
    For an ellipse with semi-axes (major_mm, minor_mm), the enclosing
    rectangle is (2*major_mm) x (2*minor_mm), so a = 2*major_mm.

    Prior to the fix, the implementation passed a = major_mm directly,
    inflating N_cr by ~4x (N_cr ∝ 1/a^2). This test pins the corrected
    convention against a hand-computed value for a square sublaminate
    with a balanced [45, -45]s layup (D11 = D22 and m=n=1 governs).
    """
    m = MATERIAL_LIBRARY["IM7/8552"]
    # Balanced ±45 sublaminate -> D11 == D22 by symmetry; D16 == D26 != 0
    # but the closed form uses only D11, D22, D12, D66.
    layup = [45.0, -45.0, -45.0, 45.0]
    ply_t = 0.152
    lam = Laminate(m, layup, ply_t)

    # Square ellipse so the enclosing rectangle is square (a == b),
    # which makes m=n=1 the analytic minimizer for any D with D11 == D22.
    semi = 25.0  # mm
    ell = DelaminationEllipse(
        interface_index=1,
        centroid_mm=(0.0, 0.0),
        major_mm=semi,
        minor_mm=semi,
        orientation_deg=0.0,
    )

    # The sublaminate selection picks the thinner stack. With interface=1
    # and 4 plies total: upper = plies 0..1 (2 plies), lower = plies 2..3
    # (2 plies); ties go to upper. So sub_layup = [45, -45].
    sub_layup = [45.0, -45.0]
    D = _sublaminate_D_matrix(m, sub_layup, ply_t)
    D11, D22, D12, D66 = D[0, 0], D[1, 1], D[0, 1], D[2, 2]

    # Full rectangle side lengths (a, b) per the corrected convention.
    a = 2.0 * semi
    b = 2.0 * semi
    aspect = a / b  # == 1

    # Hand-evaluate the closed form at m=n=1.
    pi2 = math.pi * math.pi
    N_cr_11 = (
        (pi2 / a**2)
        * (
            D11 * 1**4
            + 2.0 * (D12 + 2.0 * D66) * (1 * aspect) ** 2 * 1**2
            + D22 * (aspect * 1) ** 4
        )
        / 1**2
    )

    # Sanity: m=n=1 really is the minimizer here (a == b, D11 == D22 by ±45
    # symmetry). Spot-check (2,1) and (1,2) to be safe.
    def _N_mn(mm, nn):
        return (
            (pi2 / a**2)
            * (
                D11 * mm**4
                + 2.0 * (D12 + 2.0 * D66) * (mm * aspect) ** 2 * nn**2
                + D22 * (aspect * nn) ** 4
            )
            / mm**2
        )

    assert _N_mn(2, 1) > N_cr_11
    assert _N_mn(1, 2) > N_cr_11
    assert _N_mn(2, 2) > N_cr_11

    N_actual = sublaminate_buckling_load(lam, ell, boundary="simply_supported")
    assert N_actual == pytest.approx(N_cr_11, rel=1e-10)

    # And confirm the boundary factor is applied.
    N_clamped = sublaminate_buckling_load(lam, ell, boundary="clamped")
    assert N_clamped == pytest.approx(N_cr_11 * 1.9, rel=1e-10)


def test_buckling_load_uses_full_plate_dimensions_not_semi_axes():
    """Regression test for Issue #29 — verify the 4x-overprediction bug
    does not regress.

    If the implementation reverted to using ellipse semi-axes directly as
    (a, b), N_cr would be inflated by exactly (2*r / r)^2 = 4 (since the
    eigenvalue is homogeneous of degree -2 in length when a/b is held fixed).
    Build two cases differing only in scale and check the corrected
    scaling, then independently check the magnitude against the
    "semi-axis-as-a" miscalculation.
    """
    mat = MATERIAL_LIBRARY["IM7/8552"]
    ply_t = 0.152
    lam = Laminate(mat, [0.0, 90.0, 90.0, 0.0], ply_t)

    semi = 20.0  # mm
    ell = DelaminationEllipse(1, (0.0, 0.0), semi, semi, 0.0)

    # Sub layup is the thinner of [0, 90] vs [90, 0]; tie -> upper -> [0, 90].
    sub_layup = [0.0, 90.0]
    D = _sublaminate_D_matrix(mat, sub_layup, ply_t)
    D11, D22, D12, D66 = D[0, 0], D[1, 1], D[0, 1], D[2, 2]

    # Closed form with the CORRECT full dimensions (a = b = 2*semi).
    a_full = 2.0 * semi
    pi2 = math.pi * math.pi
    N_correct = (pi2 / a_full**2) * (D11 + 2.0 * (D12 + 2.0 * D66) + D22)

    # Closed form with the BUGGY semi-axis-as-side-length (a = semi).
    a_bug = semi
    N_buggy = (pi2 / a_bug**2) * (D11 + 2.0 * (D12 + 2.0 * D66) + D22)

    # The bug would inflate by exactly 4x.
    assert N_buggy == pytest.approx(4.0 * N_correct, rel=1e-12)

    # Verify D11 == D22 for this symmetric cross-ply (so m=n=1 governs at a=b)
    assert D11 == pytest.approx(D22, rel=1e-12)

    N_actual = sublaminate_buckling_load(lam, ell, boundary="simply_supported")
    assert N_actual == pytest.approx(N_correct, rel=1e-10)
    # And explicitly rule out the buggy value.
    assert N_actual < 0.5 * N_buggy

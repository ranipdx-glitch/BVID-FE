import math

from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.failure.soutis_openhole import (
    lekhnitskii_kt_infinity,
    soutis_cai,
    whitney_nuismer_tai,
)


def test_soutis_monotone_decreasing_in_dpa():
    m = MATERIAL_LIBRARY["IM7/8552"]
    A_panel = 150 * 100
    s1 = soutis_cai(m, dpa_mm2=100, A_panel_mm2=A_panel, sigma_pristine_MPa=500)
    s2 = soutis_cai(m, dpa_mm2=500, A_panel_mm2=A_panel, sigma_pristine_MPa=500)
    assert s1 > s2


def test_soutis_zero_dpa_returns_pristine():
    m = MATERIAL_LIBRARY["IM7/8552"]
    s = soutis_cai(m, dpa_mm2=0, A_panel_mm2=15000, sigma_pristine_MPa=500)
    assert math.isclose(s, 500, rel_tol=1e-6)


def test_soutis_pristine_strength_cap():
    m = MATERIAL_LIBRARY["IM7/8552"]
    s = soutis_cai(m, dpa_mm2=50, A_panel_mm2=15000, sigma_pristine_MPa=500)
    assert 0 < s < 500


def test_wn_tai_zero_dpa_returns_pristine():
    m = MATERIAL_LIBRARY["IM7/8552"]
    s = whitney_nuismer_tai(m, dpa_mm2=0, sigma_pristine_MPa=800)
    assert math.isclose(s, 800, rel_tol=1e-6)


def test_wn_tai_monotone_decreasing_in_dpa():
    m = MATERIAL_LIBRARY["IM7/8552"]
    t1 = whitney_nuismer_tai(m, dpa_mm2=100, sigma_pristine_MPa=800)
    t2 = whitney_nuismer_tai(m, dpa_mm2=400, sigma_pristine_MPa=800)
    assert t1 > t2


def test_wn_tai_pristine_strength_cap():
    m = MATERIAL_LIBRARY["IM7/8552"]
    s = whitney_nuismer_tai(m, dpa_mm2=200, sigma_pristine_MPa=800)
    assert 0 < s < 800


def test_lekhnitskii_kt_unidirectional_exceeds_isotropic():
    """Issue #30: a highly orthotropic IM7/8552 [0]_8 stack must have an
    infinite-plate Kt well above the isotropic value 3.0."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0] * 8, 0.152)
    assert lekhnitskii_kt_infinity(lam) > 3.0


def test_lekhnitskii_kt_quasi_iso_reduces_to_isotropic():
    """Issue #30: a quasi-isotropic layup must collapse to the isotropic
    Kt_inf = 3.0."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 2, 0.125)
    assert math.isclose(lekhnitskii_kt_infinity(lam), 3.0, rel_tol=1e-3)

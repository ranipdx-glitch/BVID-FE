import dataclasses
import math

import pytest

from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.impact.olsson import (
    NAVIER_N,
    _k_bending_ssss,
    _navier_basis_ssss,
    onset_energy,
    threshold_load,
)


def test_threshold_load_scales_with_sqrt_G_IIc():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    pan = PanelGeometry(150, 100)
    imp = ImpactorGeometry()
    Pc1 = threshold_load(lam, pan, imp)
    m4 = dataclasses.replace(m, G_IIc=m.G_IIc * 4)
    lam2 = Laminate(m4, [0, 45, -45, 90] * 4, 0.152)
    Pc2 = threshold_load(lam2, pan, imp)
    # Pc ∝ sqrt(G_IIc), so 4x G_IIc => 2x Pc
    assert math.isclose(Pc2 / Pc1, 2.0, rel_tol=0.05)


def test_onset_energy_positive_and_monotonic_in_thickness():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam_thin = Laminate(m, [0, 90] * 4, 0.125)  # 8 plies
    lam_thick = Laminate(m, [0, 90] * 12, 0.125)  # 24 plies
    pan = PanelGeometry(150, 100)
    imp = ImpactorGeometry()
    E_thin = onset_energy(lam_thin, pan, imp)
    E_thick = onset_energy(lam_thick, pan, imp)
    assert E_thin > 0 and E_thick > 0
    assert E_thick > E_thin  # thicker plate harder to damage


def test_navier_n_is_11_by_default():
    assert NAVIER_N == 11


def test_onset_energy_scales_with_material_G_IIc():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam1 = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    m4 = dataclasses.replace(m, G_IIc=m.G_IIc * 4)
    lam2 = Laminate(m4, [0, 45, -45, 90] * 4, 0.152)
    pan = PanelGeometry(150, 100)
    imp = ImpactorGeometry()
    E1 = onset_energy(lam1, pan, imp)
    E2 = onset_energy(lam2, pan, imp)
    # onset energy ∝ Pc^2 ∝ G_IIc, so 4x => 4x (approximately)
    assert E2 > E1


@pytest.mark.parametrize(
    "x0, y0",
    [
        (0.0, 0.0),  # corner (origin)
        (150.0, 100.0),  # opposite corner (Lx, Ly)
        (0.0, 50.0),  # mid-left edge (x0 == 0)
        (75.0, 100.0),  # top edge (y0 == Ly)
        (-1.0, 50.0),  # outside the panel
    ],
)
def test_k_bending_ssss_boundary_raises_valueerror(x0, y0):
    """A point load on/outside the SSSS plate boundary is degenerate: the
    bending compliance is singular, so we raise ValueError rather than
    crashing with ZeroDivisionError (issue #19)."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    pan = PanelGeometry(150, 100)
    with pytest.raises(ValueError, match="singular at the boundary"):
        _k_bending_ssss(lam, pan, x0, y0)


def test_k_bending_ssss_interior_returns_finite_positive():
    """Regression guard: a normal interior location still yields a finite,
    strictly positive bending stiffness."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    pan = PanelGeometry(150, 100)
    k = _k_bending_ssss(lam, pan, 75.0, 50.0)
    assert math.isfinite(k)
    assert k > 0.0


def test_k_bending_ssss_cached_matches_scalar_navier_reference():
    """Issue #56: the cached/vectorised Navier sum must equal the original
    scalar double-loop implementation to machine epsilon."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    pan = PanelGeometry(150, 100)
    x0, y0 = 70.0, 40.0
    _, _, D = lam.abd_matrices()
    D11, D22, D12, D66 = D[0, 0], D[1, 1], D[0, 1], D[2, 2]
    a, b = pan.Lx_mm, pan.Ly_mm
    w_over_P = 0.0
    for mm in range(1, NAVIER_N + 1):
        for nn in range(1, NAVIER_N + 1):
            s = math.sin(mm * math.pi * x0 / a) * math.sin(nn * math.pi * y0 / b)
            dmn = (
                D11 * (mm * math.pi / a) ** 4
                + 2 * (D12 + 2 * D66) * (mm * math.pi / a) ** 2 * (nn * math.pi / b) ** 2
                + D22 * (nn * math.pi / b) ** 4
            )
            w_over_P += s * s / dmn
    w_over_P *= 4.0 / (a * b)
    reference = 1.0 / w_over_P
    assert _k_bending_ssss(lam, pan, x0, y0) == pytest.approx(reference, rel=1e-12, abs=0.0)


def test_navier_basis_is_cached_keyed_on_geometry_not_laminate():
    """Issue #56: identical (a, b, x0, y0, n_modes) reuse the cached basis
    even across different laminates — the expensive trig grid is computed once."""
    _navier_basis_ssss.cache_clear()
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam_a = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    lam_b = Laminate(m, [0, 90] * 6, 0.152)  # different D, same geometry/location
    pan = PanelGeometry(150, 100)
    _k_bending_ssss(lam_a, pan, 60.0, 40.0)
    _k_bending_ssss(lam_b, pan, 60.0, 40.0)
    _k_bending_ssss(lam_a, pan, 60.0, 40.0)
    info = _navier_basis_ssss.cache_info()
    assert info.misses == 1  # basis trig grid computed exactly once
    assert info.hits >= 2  # later identical-geometry calls reuse it

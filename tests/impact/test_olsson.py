import dataclasses
import math

import pytest

from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.impact.olsson import (
    NAVIER_N,
    _k_bending_ssss,
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


@pytest.mark.parametrize(
    "x0,y0",
    [(0.0, 50.0), (150.0, 50.0), (75.0, 0.0), (75.0, 100.0), (0.0, 0.0), (150.0, 100.0)],
)
def test_k_bending_ssss_rejects_boundary_point(x0, y0):
    """A point load on the SSSS edge/corner has infinite bending compliance
    (every Navier sin term vanishes); it must raise rather than divide by 0."""
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], [0, 45, -45, 90] * 2, 0.152)
    pan = PanelGeometry(150, 100, boundary="simply_supported")
    with pytest.raises(ValueError, match="boundary"):
        _k_bending_ssss(lam, pan, x0, y0)


def test_k_bending_ssss_matches_reference_scalar_sum():
    """Cached/vectorised _k_bending_ssss must equal the original scalar
    double-sum to machine epsilon (issue #56 acceptance)."""
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], [0, 45, -45, 90] * 2, 0.152)
    pan = PanelGeometry(150, 100, boundary="simply_supported")
    x0, y0 = 61.0, 43.0
    n_modes = NAVIER_N
    _, _, D = lam.abd_matrices()
    D11, D22, D12, D66 = D[0, 0], D[1, 1], D[0, 1], D[2, 2]
    a, b = pan.Lx_mm, pan.Ly_mm
    w_over_P = 0.0
    for m in range(1, n_modes + 1):
        for n in range(1, n_modes + 1):
            sin_mx = math.sin(m * math.pi * x0 / a)
            sin_ny = math.sin(n * math.pi * y0 / b)
            Dmn = (
                D11 * (m * math.pi / a) ** 4
                + 2 * (D12 + 2 * D66) * (m * math.pi / a) ** 2 * (n * math.pi / b) ** 2
                + D22 * (n * math.pi / b) ** 4
            )
            w_over_P += (sin_mx * sin_ny) ** 2 / Dmn
    w_over_P *= 4.0 / (a * b)
    ref = 1.0 / w_over_P
    assert _k_bending_ssss(lam, pan, x0, y0) == pytest.approx(ref, rel=1e-15, abs=0.0)


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

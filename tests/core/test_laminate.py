import numpy as np
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.core.laminate import Laminate


def test_symmetric_laminate_has_zero_B():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(material=m, layup_deg=[0, 45, -45, 90, 90, -45, 45, 0], ply_thickness_mm=0.152)
    A, B, D = lam.abd_matrices()
    assert np.allclose(B, 0.0, atol=1e-6)


def test_quasi_isotropic_A_is_isotropic():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(material=m, layup_deg=[0, 45, -45, 90] * 2, ply_thickness_mm=0.125)
    A, _, _ = lam.abd_matrices()
    # A11 == A22, A16 == A26 == 0 for quasi-iso
    assert abs(A[0, 0] - A[1, 1]) / A[0, 0] < 0.02
    assert abs(A[0, 2]) / A[0, 0] < 0.02
    assert abs(A[1, 2]) / A[0, 0] < 0.02


def test_effective_Ex_matches_ply_when_all_zero():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(material=m, layup_deg=[0] * 8, ply_thickness_mm=0.125)
    Ex, Ey, Gxy, nuxy = lam.effective_engineering_constants()
    assert abs(Ex - m.E11) / m.E11 < 0.01


def test_thickness_property():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(material=m, layup_deg=[0, 90, 0, 90], ply_thickness_mm=0.2)
    assert abs(lam.thickness_mm - 0.8) < 1e-9


def test_D_eff_positive():
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(material=m, layup_deg=[0, 45, -45, 90] * 4, ply_thickness_mm=0.152)
    assert lam.flexural_rigidity_Deff() > 0


def test_effective_engineering_constants_valid_laminate_finite_positive():
    """Issue #21 regression: the new ill-conditioning guard must NOT
    false-trigger on a normal laminate."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    Ex, Ey, Gxy, nuxy = lam.effective_engineering_constants()
    for v in (Ex, Ey, Gxy):
        assert np.isfinite(v) and v > 0.0
    assert np.isfinite(nuxy)


def test_effective_engineering_constants_raises_on_singular_A():
    """Issue #21: a numerically singular A must raise loudly instead of
    silently returning NaN/Inf that corrupts downstream stress recovery."""
    import pytest

    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    # Corrupt the cached A to a (nearly) singular matrix.
    lam._A = np.full((3, 3), 1e-14)
    with pytest.raises(ValueError, match="ill-conditioned"):
        lam.effective_engineering_constants()

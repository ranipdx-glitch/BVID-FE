import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Per-ply (non-uniform) thickness support
# ---------------------------------------------------------------------------


def test_per_ply_thickness_uniform_list_matches_scalar():
    """A per-ply thickness list of identical values must produce numerically
    identical ABD matrices, total thickness, and engineering constants as
    the scalar form. This is the no-regression guarantee."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    layup = [0, 45, -45, 90, 90, -45, 45, 0]
    t = 0.152
    lam_scalar = Laminate(material=m, layup_deg=layup, ply_thickness_mm=t)
    lam_list = Laminate(material=m, layup_deg=layup, ply_thickness_mm=[t] * len(layup))
    A_s, B_s, D_s = lam_scalar.abd_matrices()
    A_l, B_l, D_l = lam_list.abd_matrices()
    assert np.allclose(A_s, A_l)
    assert np.allclose(B_s, B_l)
    assert np.allclose(D_s, D_l)
    assert lam_scalar.thickness_mm == lam_list.thickness_mm
    assert lam_scalar.is_uniform_thickness
    assert lam_list.is_uniform_thickness


def test_per_ply_thickness_total_h_matches_sum():
    m = MATERIAL_LIBRARY["IM7/8552"]
    thicknesses = [0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.10, 0.10]
    lam = Laminate(
        material=m,
        layup_deg=[0, 90, 45, -45, -45, 45, 90, 0],
        ply_thickness_mm=thicknesses,
    )
    assert abs(lam.thickness_mm - sum(thicknesses)) < 1e-12
    assert lam.ply_thicknesses_mm == thicknesses
    assert not lam.is_uniform_thickness


def test_per_ply_thickness_z_coords_track_per_ply_steps():
    """z[k+1] - z[k] must equal the k-th ply thickness exactly."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    thicknesses = [0.08, 0.20, 0.30, 0.10]
    lam = Laminate(
        material=m,
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=thicknesses,
    )
    z = lam._z_coords()  # noqa: SLF001 — internal API needed for the geometry assertion
    assert abs(z[0] + lam.thickness_mm / 2.0) < 1e-12
    for k, t in enumerate(thicknesses):
        assert abs((z[k + 1] - z[k]) - t) < 1e-12


def test_per_ply_thickness_length_mismatch_raises():
    m = MATERIAL_LIBRARY["IM7/8552"]
    with pytest.raises(ValueError, match="length"):
        Laminate(
            material=m,
            layup_deg=[0, 45, -45, 90],
            ply_thickness_mm=[0.10, 0.20],  # only 2 vs 4 plies
        )


def test_per_ply_thickness_non_positive_raises():
    m = MATERIAL_LIBRARY["IM7/8552"]
    with pytest.raises(ValueError, match="must be > 0"):
        Laminate(
            material=m,
            layup_deg=[0, 90, 0, 90],
            ply_thickness_mm=[0.1, 0.0, 0.1, 0.1],
        )


def test_thicker_outer_plies_increase_D11_vs_uniform():
    """Moving thickness outboard increases bending stiffness D11 — basic
    sanity check that per-ply z accounting actually flows through CLT."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    layup = [0, 90, 0, 90, 90, 0, 90, 0]
    # Reference uniform stack: 8 plies x 0.15 mm = 1.20 mm total.
    uniform = Laminate(material=m, layup_deg=layup, ply_thickness_mm=0.15)
    # Same total thickness, but thicker 0-degree outer plies and thinner
    # interior — D11 (bending about y) should rise for the thick-outer stack.
    outer_thick = [0.25, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.25]
    assert abs(sum(outer_thick) - uniform.thickness_mm) < 1e-12
    thick_outer = Laminate(material=m, layup_deg=layup, ply_thickness_mm=outer_thick)
    _, _, D_uni = uniform.abd_matrices()
    _, _, D_thk = thick_outer.abd_matrices()
    assert D_thk[0, 0] > D_uni[0, 0]

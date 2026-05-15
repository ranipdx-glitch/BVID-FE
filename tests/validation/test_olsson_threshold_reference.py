"""Olsson threshold-load + onset-energy validation.

The Olsson (2001) quasi-static damage-threshold model predicts
   Pc = pi * sqrt(8 * G_IIc * D_eff / 9)
where D_eff is the geometric-mean flexural rigidity. These tests pin the
shape of the prediction (toughness scaling, thickness scaling, panel
size invariance) rather than absolute literature values — the absolute
calibration would need digitised Olsson plots, which is out of scope
for a self-contained regression test.

The shape relationships, in contrast, follow directly from the closed
form and must hold for every BVID-FE material:
  * Pc scales as sqrt(G_IIc) when D_eff is held fixed.
  * Pc scales as h^{3/2} when E_ij are held fixed (D_eff ~ E*h^3, sqrt
    gives h^{3/2}; this is exact CLT).
  * Pc is a property of the laminate + impactor and is INDEPENDENT of
    panel size, so the same Laminate+ImpactorGeometry should yield the
    same Pc on a 100x100 mm and 200x150 mm panel.
"""

from __future__ import annotations

import math

import pytest

from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.impact.olsson import threshold_load


def _laminate(material_name: str, layup, t_ply: float = 0.152):
    return Laminate(MATERIAL_LIBRARY[material_name], list(layup), t_ply)


def _panel(Lx: float = 150.0, Ly: float = 100.0):
    return PanelGeometry(Lx_mm=Lx, Ly_mm=Ly, boundary="simply_supported")


def _impactor():
    return ImpactorGeometry(diameter_mm=16.0, shape="hemispherical")


def test_threshold_load_is_positive_and_finite():
    """A standard CFRP layup must produce a positive, finite Pc."""
    Pc = threshold_load(
        _laminate("IM7/8552", [0, 45, -45, 90, 90, -45, 45, 0]), _panel(), _impactor()
    )
    assert math.isfinite(Pc)
    assert Pc > 0


def test_threshold_load_scales_as_sqrt_of_giic():
    """Pc ~ sqrt(G_IIc) when D_eff is fixed.

    Hold the laminate stack fixed (same E_ij, same ply count) and swap
    materials with different G_IIc only. We use IM7/8552 (G_IIc=0.79)
    and T700/2510 (G_IIc=0.60) and check the ratio matches sqrt of the
    G_IIc ratio. The two materials have different E_ij too, so we
    isolate the G_IIc effect by comparing Pc / sqrt(D_eff)."""
    lam_a = _laminate("IM7/8552", [0, 45, -45, 90, 90, -45, 45, 0])
    lam_b = _laminate("T700/2510", [0, 45, -45, 90, 90, -45, 45, 0])
    pan, imp = _panel(), _impactor()
    Pc_a = threshold_load(lam_a, pan, imp)
    Pc_b = threshold_load(lam_b, pan, imp)
    # Pc / sqrt(D_eff * G_IIc) must be the same constant for both materials
    # (it equals pi * sqrt(8/9)).
    expected_const = math.pi * math.sqrt(8.0 / 9.0)
    Deff_a = lam_a.flexural_rigidity_Deff()
    Deff_b = lam_b.flexural_rigidity_Deff()
    actual_a = Pc_a / math.sqrt(Deff_a * lam_a.material.G_IIc)
    actual_b = Pc_b / math.sqrt(Deff_b * lam_b.material.G_IIc)
    assert actual_a == pytest.approx(expected_const, rel=1e-9)
    assert actual_b == pytest.approx(expected_const, rel=1e-9)


def test_threshold_load_invariant_to_panel_size():
    """Pc depends on the LAMINATE, not the panel (per the Olsson model)."""
    lam = _laminate("IM7/8552", [0, 45, -45, 90, 90, -45, 45, 0])
    imp = _impactor()
    Pc_small = threshold_load(lam, _panel(100, 80), imp)
    Pc_large = threshold_load(lam, _panel(300, 200), imp)
    assert Pc_small == pytest.approx(Pc_large, rel=1e-12)


def test_threshold_load_scales_with_thickness_to_the_three_halves():
    """Pc ~ h^{3/2} under *homogeneous* thickness scaling (exact CLT).

    Pc = pi*sqrt(8*G_IIc*D_eff/9) with D_eff = sqrt(D11*D22), and the D
    matrix is sum_k Qbar_k*(z_k^3 - z_{k-1}^3)/3. The h^3 (hence Pc ~ h^{3/2})
    law is exact ONLY when the through-thickness Qbar(z) distribution is
    preserved under scaling — i.e. the SAME layup with thicker plies. It is
    NOT a property of "doubling the ply count": [0,45,-45,90]*2 is a
    different stacking sequence (the 0 ply moves off the surface), so its
    D_eff/h^3 is not unity (here D_eff ratio ~7.0, giving Pc ratio ~2.65 =
    sqrt(7), which is correct physics, not a bug). We therefore scale h
    homogeneously by doubling t_ply with the layup held fixed."""
    base_layup = [0, 45, -45, 90]
    lam_thin = _laminate("IM7/8552", base_layup, t_ply=0.152)
    lam_thick = _laminate("IM7/8552", base_layup, t_ply=0.304)  # exact h -> 2h
    pan, imp = _panel(), _impactor()
    Pc_thin = threshold_load(lam_thin, pan, imp)
    Pc_thick = threshold_load(lam_thick, pan, imp)
    # h_thick / h_thin = 2 -> Pc_thick / Pc_thin = 2^{3/2} = 2.828... (exact)
    assert Pc_thick / Pc_thin == pytest.approx(2**1.5, rel=1e-9)

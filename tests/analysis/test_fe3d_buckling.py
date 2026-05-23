"""Tests for the fe3d CAI buckling channel (Rayleigh-Ritz closed-form
delegation, #129)."""

import pytest

from bvidfe.analysis import AnalysisConfig, BvidAnalysis, MeshParams
from bvidfe.analysis.fe_tier import fe3d_cai_buckling
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.impact.mapping import ImpactEvent


@pytest.fixture
def small_cfg():
    return AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(10, 5),
        loading="compression",
        tier="fe3d",
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=2.5),
    )


def test_fe3d_cai_buckling_returns_positive_strength(small_cfg):
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    damage = DamageState([], dent_depth_mm=0.0)
    sigma, lambda_crit, notes = fe3d_cai_buckling(small_cfg, damage, lam, sigma_pristine_MPa=500.0)
    assert sigma > 0
    assert lambda_crit > 0
    # Clean solve on a pristine input should not generate any diagnostic notes.
    assert notes == []


def test_fe3d_cai_buckling_damaged_less_than_pristine(small_cfg):
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    ds = DamageState([DelaminationEllipse(1, (5, 2.5), 3, 1.5, 0)], dent_depth_mm=0.2)
    sigma_pristine, _, _ = fe3d_cai_buckling(small_cfg, DamageState([], 0.0), lam, 500.0)
    sigma_damaged, _, _ = fe3d_cai_buckling(small_cfg, ds, lam, 500.0)
    assert 0 < sigma_damaged <= sigma_pristine


def test_fe3d_cai_buckling_sublaminate_path_binds_on_thin_panel():
    """The damaged-vs-pristine ordering on ``small_cfg`` saturates against
    ``sigma_pristine_MPa`` (the small coupon's plate buckling load is huge),
    so the test above can't verify the sublaminate path actually engages.
    This test uses a wider panel with a large delamination so the
    sublaminate buckling stress is strictly below pristine and strictly
    below the full-panel buckling stress — proving min(panel, sublam) picks
    the sublaminate when it should.
    """
    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150.0, 100.0, boundary="simply_supported"),
        loading="compression",
        tier="fe3d",
        damage=DamageState(
            [DelaminationEllipse(3, (75.0, 50.0), 40.0, 30.0, 0.0)], dent_depth_mm=0.5
        ),
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=10.0),
    )
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], cfg.layup_deg, cfg.ply_thickness_mm)
    sigma_pristine_panel, _, _ = fe3d_cai_buckling(
        cfg, DamageState([], 0.0), lam, sigma_pristine_MPa=600.0
    )
    sigma_damaged, _, _ = fe3d_cai_buckling(cfg, cfg.damage, lam, sigma_pristine_MPa=600.0)
    assert sigma_damaged < sigma_pristine_panel
    assert sigma_damaged < 600.0  # neither path saturates


def test_fe3d_cai_buckling_canonical_panel_value():
    """Regression test pinning the buckling stress on the canonical
    200x150 mm 8-ply quasi-isotropic IM7/8552 pristine reference (issue #129).

    The Rayleigh-Ritz closed form is exact for this configuration; a
    regression to a different formulation (or a unit-conversion bug) would
    move this number by an order of magnitude. The expected value is
    derived from D11/D22/D12/D66 of the laminate and the
    Navier sine-basis solution minimised over (m, n) in [1..5].
    """
    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(200.0, 150.0, boundary="simply_supported"),
        loading="compression",
        tier="fe3d",
        damage=DamageState([], dent_depth_mm=0.0),
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=10.0),
    )
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], cfg.layup_deg, cfg.ply_thickness_mm)
    sigma, _, _ = fe3d_cai_buckling(cfg, cfg.damage, lam, sigma_pristine_MPa=887.5)
    assert 12.0 < sigma < 13.0, f"expected ~12.6 MPa, got {sigma}"


def test_bvid_analysis_fe3d_cai_uses_buckling_path(small_cfg):
    """BvidAnalysis with tier=fe3d should populate buckling_eigenvalues."""
    r = BvidAnalysis(small_cfg).run()
    assert r.tier_used == "fe3d"
    assert r.buckling_eigenvalues is not None
    assert len(r.buckling_eigenvalues) >= 1
    assert r.buckling_eigenvalues[0] > 0


def test_fe3d_cai_buckling_degenerate_closed_form_emits_note(small_cfg, monkeypatch):
    """If the Rayleigh-Ritz closed-form returns a degenerate (non-finite or
    non-positive) value for both the full-panel and the worst-sublaminate
    paths, fe3d_cai_buckling falls back to pristine AND emits a runtime
    note. Without the note the user would see knockdown=1.0 with no
    indication that the buckling channel was inactive."""
    from bvidfe.analysis import fe_tier as ft
    from bvidfe.core.laminate import Laminate as Lam

    monkeypatch.setattr(ft, "panel_buckling_load", lambda *_a, **_kw: float("inf"))
    monkeypatch.setattr(ft, "sublaminate_buckling_load", lambda *_a, **_kw: float("inf"))
    lam = Lam(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    sigma, lambda_crit, notes = fe3d_cai_buckling(
        small_cfg, DamageState([], 0.0), lam, sigma_pristine_MPa=500.0
    )
    assert sigma == 500.0  # fell back to pristine
    assert lambda_crit == 0.0
    assert any("rayleigh-ritz" in n.lower() or "degenerate" in n.lower() for n in notes), notes


def test_bvid_analysis_fe3d_surfaces_buckling_fallback_note(small_cfg, monkeypatch):
    """End-to-end: a buckling fallback inside fe3d_cai_buckling should
    surface in ``AnalysisResults.notes`` so callers (CLI / GUI) can see
    that the reported knockdown is the pristine fallback rather than a
    real result."""
    from bvidfe.analysis import fe_tier as ft

    monkeypatch.setattr(ft, "panel_buckling_load", lambda *_a, **_kw: float("inf"))
    monkeypatch.setattr(ft, "sublaminate_buckling_load", lambda *_a, **_kw: float("inf"))
    r = BvidAnalysis(small_cfg).run()
    assert any("rayleigh-ritz" in n.lower() or "degenerate" in n.lower() for n in r.notes), r.notes

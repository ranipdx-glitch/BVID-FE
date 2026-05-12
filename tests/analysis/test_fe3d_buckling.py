"""Tests for the fe3d linear buckling CAI path."""

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


def test_bvid_analysis_fe3d_cai_uses_buckling_path(small_cfg):
    """BvidAnalysis with tier=fe3d should populate buckling_eigenvalues."""
    r = BvidAnalysis(small_cfg).run()
    assert r.tier_used == "fe3d"
    assert r.buckling_eigenvalues is not None
    assert len(r.buckling_eigenvalues) >= 1
    assert r.buckling_eigenvalues[0] > 0


def test_fe3d_cai_buckling_eigensolver_failure_emits_note(small_cfg, monkeypatch):
    """If linear_buckling raises, fe3d_cai_buckling falls back to pristine
    AND emits a runtime note explaining the fallback. Without the note the
    user sees knockdown=1.0 with no indication that the solver failed."""
    from bvidfe.analysis import fe_tier as ft
    from bvidfe.core.laminate import Laminate as Lam

    def _raise(*_args, **_kwargs):
        raise RuntimeError("synthetic ARPACK failure")

    monkeypatch.setattr(ft, "linear_buckling", _raise)
    lam = Lam(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    sigma, lambda_crit, notes = fe3d_cai_buckling(
        small_cfg, DamageState([], 0.0), lam, sigma_pristine_MPa=500.0
    )
    assert sigma == 500.0  # fell back to pristine
    assert lambda_crit == 0.0
    assert any("eigensolver" in n.lower() for n in notes), notes


def test_bvid_analysis_fe3d_surfaces_buckling_fallback_note(small_cfg, monkeypatch):
    """End-to-end: a buckling fallback inside fe3d_cai_buckling should
    surface in ``AnalysisResults.notes`` so callers (CLI / GUI) can see
    that the reported knockdown is the pristine fallback rather than a
    real result."""
    from bvidfe.analysis import fe_tier as ft

    def _raise(*_args, **_kwargs):
        raise RuntimeError("synthetic ARPACK failure")

    monkeypatch.setattr(ft, "linear_buckling", _raise)
    r = BvidAnalysis(small_cfg).run()
    assert any("eigensolver" in n.lower() for n in r.notes), r.notes

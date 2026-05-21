import pytest

from bvidfe.analysis.config import AnalysisConfig, MeshParams
from bvidfe.analysis.fe_tier import (
    FE3DSizeError,
    _fe3d_cai_first_ply_failure,
    fe3d_cai,
    fe3d_tai,
)
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
        panel=PanelGeometry(10.0, 5.0),
        loading="compression",
        tier="fe3d",
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=2.5),
    )


def test_fe3d_cai_pristine_returns_positive_strength(small_cfg):
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    damage = DamageState([], dent_depth_mm=0.0)
    sigma = fe3d_cai(small_cfg, damage, lam, sigma_pristine_MPa=500.0)
    assert sigma > 0


def test_fe3d_cai_damaged_less_than_pristine(small_cfg):
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    # Damage with a modest ellipse at interface 1
    ds = DamageState([DelaminationEllipse(1, (5, 2.5), 3, 1.5, 0)], dent_depth_mm=0.2)
    sigma_pristine = fe3d_cai(small_cfg, DamageState([], 0.0), lam, sigma_pristine_MPa=500.0)
    sigma_damaged = fe3d_cai(small_cfg, ds, lam, sigma_pristine_MPa=500.0)
    assert 0 < sigma_damaged <= sigma_pristine


def test_fe3d_tai_pristine_positive(small_cfg):
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    damage = DamageState([], 0.0)
    sigma = fe3d_tai(small_cfg, damage, lam, sigma_pristine_MPa=800.0)
    assert sigma > 0


def test_fe3d_rejects_oversized_mesh():
    """Defense-in-depth: fe3d solvers raise FE3DSizeError before calling
    scipy's native code on an OOM-risk problem. Prevents the SIGSEGV that
    the v0.1.0 build exhibited on a 150x100 panel with default mesh."""
    oversized_cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90] * 4,  # 16 plies
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="compression",
        tier="fe3d",
        impact=ImpactEvent(20.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=4, in_plane_size_mm=1.0),  # the old default
    )
    lam = Laminate(
        MATERIAL_LIBRARY["IM7/8552"],
        oversized_cfg.layup_deg,
        oversized_cfg.ply_thickness_mm,
    )
    with pytest.raises(FE3DSizeError):
        fe3d_cai(oversized_cfg, DamageState([], 0.0), lam, 500.0)
    with pytest.raises(FE3DSizeError):
        fe3d_tai(oversized_cfg, DamageState([], 0.0), lam, 800.0)


def test_fe3d_fpf_accepts_explicit_criterion(small_cfg):
    """``_fe3d_cai_first_ply_failure`` must accept ``criterion="tsai_wu"``
    instead of hardcoding LaRC05. The default behaviour (no kwarg) stays
    LaRC05 — verified by comparing the explicit-larc05 call against the
    default call.
    """
    lam = Laminate(MATERIAL_LIBRARY["IM7/8552"], small_cfg.layup_deg, small_cfg.ply_thickness_mm)
    damage = DamageState([], dent_depth_mm=0.0)

    sigma_default = _fe3d_cai_first_ply_failure(small_cfg, damage, lam, 500.0)
    sigma_larc05 = _fe3d_cai_first_ply_failure(small_cfg, damage, lam, 500.0, criterion="larc05")
    sigma_tsai = _fe3d_cai_first_ply_failure(small_cfg, damage, lam, 500.0, criterion="tsai_wu")

    # Default must match explicit larc05 (preserves prior behaviour).
    assert sigma_default == pytest.approx(sigma_larc05, rel=1e-12)
    # All paths return positive residual strength on a pristine panel.
    assert sigma_tsai > 0

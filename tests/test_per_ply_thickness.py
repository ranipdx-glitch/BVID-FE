"""End-to-end coverage for per-ply (non-uniform) thickness laminates (#5).

Guarantees that ``ply_thickness_mm`` accepts both a scalar (uniform, the
legacy behaviour) and a per-ply sequence, that a uniform list reproduces
scalar results bit-for-bit across all three tiers, and that genuine
non-uniform stacks change the physics as expected. Covers the user-visible
surfaces: AnalysisConfig validation, the CLI parser, the analysis pipeline
(pristine-strength weighting, semi-analytical sublaminate selection, fe3d
z-grid).
"""

from __future__ import annotations

import math

import pytest

from bvidfe.analysis import AnalysisConfig, BvidAnalysis, MeshParams
from bvidfe.cli import _build_parser
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.impact.mapping import ImpactEvent

_LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
_T = 0.152


def _make_cfg(thickness, *, tier: str = "empirical", loading: str = "compression"):
    return AnalysisConfig(
        material="IM7/8552",
        layup_deg=list(_LAYUP),
        ply_thickness_mm=thickness,
        panel=PanelGeometry(150, 100),
        loading=loading,
        tier=tier,
        impact=ImpactEvent(30.0, ImpactorGeometry(), mass_kg=5.5),
    )


# --- AnalysisConfig validation --------------------------------------------


def test_config_accepts_scalar_thickness():
    assert _make_cfg(_T).ply_thickness_mm == _T


def test_config_accepts_list_thickness():
    assert _make_cfg([_T] * len(_LAYUP)).ply_thickness_mm == [_T] * len(_LAYUP)


def test_config_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        _make_cfg([_T, _T])


def test_config_rejects_non_positive_entry():
    bad = [_T] * len(_LAYUP)
    bad[3] = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        _make_cfg(bad)


# --- Uniform list ≡ scalar (bit-for-bit) across tiers ---------------------


@pytest.mark.parametrize("tier", ["empirical", "semi_analytical"])
def test_uniform_list_matches_scalar_knockdown(tier):
    r_scalar = BvidAnalysis(_make_cfg(_T, tier=tier)).run()
    r_list = BvidAnalysis(_make_cfg([_T] * len(_LAYUP), tier=tier)).run()
    assert r_scalar.knockdown == pytest.approx(r_list.knockdown, rel=1e-12)
    assert r_scalar.residual_strength_MPa == pytest.approx(r_list.residual_strength_MPa, rel=1e-12)
    assert r_scalar.pristine_strength_MPa == pytest.approx(r_list.pristine_strength_MPa, rel=1e-12)


def test_uniform_list_matches_scalar_tai():
    ds = DamageState([DelaminationEllipse(3, (75, 50), 20, 12, 45)], dent_depth_mm=0.4)
    common = dict(
        material="IM7/8552",
        layup_deg=list(_LAYUP),
        panel=PanelGeometry(150, 100),
        loading="tension",
        tier="empirical",
        damage=ds,
    )
    r_scalar = BvidAnalysis(AnalysisConfig(ply_thickness_mm=_T, **common)).run()
    r_list = BvidAnalysis(AnalysisConfig(ply_thickness_mm=[_T] * len(_LAYUP), **common)).run()
    assert r_scalar.knockdown == pytest.approx(r_list.knockdown, rel=1e-12)


def test_uniform_list_matches_scalar_fe3d():
    ds = DamageState([DelaminationEllipse(2, (5, 5), 2.0, 1.5, 0.0)], dent_depth_mm=0.2)
    common = dict(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90],
        panel=PanelGeometry(10, 10),
        loading="compression",
        tier="fe3d",
        damage=ds,
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=2.0),
    )
    r_scalar = BvidAnalysis(AnalysisConfig(ply_thickness_mm=0.15, **common)).run()
    r_list = BvidAnalysis(AnalysisConfig(ply_thickness_mm=[0.15] * 4, **common)).run()
    assert r_scalar.residual_strength_MPa == pytest.approx(r_list.residual_strength_MPa, rel=1e-12)


# --- Genuine non-uniform behaviour ----------------------------------------


def test_thickness_weighted_pristine_matches_handwritten_sum():
    from bvidfe.analysis.bvid import _pristine_strength
    from bvidfe.core.laminate import Laminate
    from bvidfe.core.material import MATERIAL_LIBRARY

    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(material=m, layup_deg=[0, 90], ply_thickness_mm=[0.20, 0.10])
    expected = (0.20 * m.Xc + 0.10 * m.Yc) / (0.20 + 0.10)
    assert _pristine_strength(lam, "compression") == pytest.approx(expected, rel=1e-12)


def test_per_ply_thickness_changes_pristine_strength():
    r_uni = BvidAnalysis(_make_cfg(_T)).run()
    r_thk = BvidAnalysis(_make_cfg([0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.30])).run()
    assert not math.isclose(r_uni.pristine_strength_MPa, r_thk.pristine_strength_MPa)


def test_per_ply_thickness_preserves_finite_knockdown():
    r = BvidAnalysis(_make_cfg([0.10, 0.20, 0.20, 0.10, 0.10, 0.20, 0.20, 0.10])).run()
    assert math.isfinite(r.knockdown)
    assert 0.0 < r.knockdown <= 1.0
    assert math.isfinite(r.residual_strength_MPa)


def test_fe3d_runs_with_per_ply_thickness():
    ds = DamageState([DelaminationEllipse(2, (5, 5), 2.0, 1.5, 0.0)], dent_depth_mm=0.2)
    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90],
        ply_thickness_mm=[0.10, 0.10, 0.20, 0.20],
        panel=PanelGeometry(10, 10),
        loading="compression",
        tier="fe3d",
        damage=ds,
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=2.0),
    )
    r = BvidAnalysis(cfg).run()
    assert math.isfinite(r.residual_strength_MPa)
    assert math.isfinite(r.knockdown)
    assert r.tier_used == "fe3d"


# --- CLI parser -----------------------------------------------------------

_CLI_BASE = [
    "--material",
    "IM7/8552",
    "--layup",
    "0,45,-45,90,90,-45,45,0",
    "--panel",
    "150x100",
    "--loading",
    "compression",
    "--energy",
    "30",
]


def test_cli_thickness_accepts_scalar():
    args = _build_parser().parse_args([*_CLI_BASE, "--thickness", "0.152"])
    assert args.thickness == 0.152


def test_cli_thickness_accepts_csv_list():
    args = _build_parser().parse_args(
        [*_CLI_BASE, "--thickness", "0.10,0.10,0.20,0.20,0.20,0.20,0.10,0.10"]
    )
    assert args.thickness == [0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.10, 0.10]


def test_cli_thickness_rejects_zero_in_list():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "--material",
                "IM7/8552",
                "--layup",
                "0,90,0,90",
                "--thickness",
                "0.1,0.0,0.1,0.1",
                "--panel",
                "150x100",
                "--loading",
                "compression",
                "--energy",
                "30",
            ]
        )

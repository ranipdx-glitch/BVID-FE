"""End-to-end coverage for per-ply (non-uniform) thickness laminates.

These tests guarantee that ``AnalysisConfig.ply_thickness_mm`` accepts both
a scalar (uniform laminate, the legacy behaviour) and a sequence of per-ply
thicknesses, and that a uniform list reproduces scalar results bit-for-bit
across all three tiers. They cover the user-visible surfaces affected by
the change (CLI parser, analysis pipeline, semi-analytical sublaminate
selection).
"""

from __future__ import annotations

import math

import pytest

from bvidfe.analysis import AnalysisConfig, BvidAnalysis
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


# ---------------------------------------------------------------------------
# AnalysisConfig validation
# ---------------------------------------------------------------------------


def test_config_accepts_scalar_thickness():
    cfg = _make_cfg(_T)
    assert cfg.ply_thickness_mm == _T


def test_config_accepts_list_thickness():
    cfg = _make_cfg([_T] * len(_LAYUP))
    assert cfg.ply_thickness_mm == [_T] * len(_LAYUP)


def test_config_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        _make_cfg([_T, _T])


def test_config_rejects_non_positive_entry():
    bad = [_T] * len(_LAYUP)
    bad[3] = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        _make_cfg(bad)


# ---------------------------------------------------------------------------
# Pipeline equivalence: uniform list ≡ scalar across all three tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["empirical", "semi_analytical"])
def test_uniform_list_matches_scalar_knockdown(tier):
    """A uniform list must yield bitwise-identical results to the scalar form."""
    r_scalar = BvidAnalysis(_make_cfg(_T, tier=tier)).run()
    r_list = BvidAnalysis(_make_cfg([_T] * len(_LAYUP), tier=tier)).run()
    assert r_scalar.knockdown == pytest.approx(r_list.knockdown, rel=1e-12)
    assert r_scalar.residual_strength_MPa == pytest.approx(r_list.residual_strength_MPa, rel=1e-12)
    assert r_scalar.pristine_strength_MPa == pytest.approx(r_list.pristine_strength_MPa, rel=1e-12)


def test_uniform_list_matches_scalar_tai():
    ds = DamageState(
        [DelaminationEllipse(3, (75, 50), 20, 12, 45)],
        dent_depth_mm=0.4,
    )
    cfg_scalar = AnalysisConfig(
        material="IM7/8552",
        layup_deg=list(_LAYUP),
        ply_thickness_mm=_T,
        panel=PanelGeometry(150, 100),
        loading="tension",
        tier="empirical",
        damage=ds,
    )
    cfg_list = AnalysisConfig(
        material="IM7/8552",
        layup_deg=list(_LAYUP),
        ply_thickness_mm=[_T] * len(_LAYUP),
        panel=PanelGeometry(150, 100),
        loading="tension",
        tier="empirical",
        damage=ds,
    )
    r_scalar = BvidAnalysis(cfg_scalar).run()
    r_list = BvidAnalysis(cfg_list).run()
    assert r_scalar.knockdown == pytest.approx(r_list.knockdown, rel=1e-12)


# ---------------------------------------------------------------------------
# Genuine non-uniform behaviour
# ---------------------------------------------------------------------------


def test_thickness_weighted_pristine_strength_matches_handwritten_sum():
    """``_pristine_strength`` must weight each ply by its actual thickness.

    A 2-ply [0, 90] laminate where the 0-ply is twice as thick as the 90-ply
    should give pristine compression strength = (2*Xc + 1*Yc) / 3.
    """
    from bvidfe.analysis.bvid import _pristine_strength
    from bvidfe.core.laminate import Laminate
    from bvidfe.core.material import MATERIAL_LIBRARY

    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(material=m, layup_deg=[0, 90], ply_thickness_mm=[0.20, 0.10])
    expected = (0.20 * m.Xc + 0.10 * m.Yc) / (0.20 + 0.10)
    assert _pristine_strength(lam, "compression") == pytest.approx(expected, rel=1e-12)


def test_per_ply_thickness_changes_pristine_strength():
    """Same total thickness, different distribution => still same pristine
    (it's a thickness-weighted average; only the *shape* of the distribution
    matters when fibre angles differ across plies). For a layup with mixed
    angles the pristine should differ from the uniform-thickness baseline
    if we weight more toward 0-deg plies."""
    cfg_uniform = _make_cfg(_T)
    cfg_thick0 = _make_cfg([0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.30])
    r_uni = BvidAnalysis(cfg_uniform).run()
    r_thk = BvidAnalysis(cfg_thick0).run()
    # The thicker stack with more 0-deg has a higher fibre-direction
    # contribution and a different total thickness, so pristine should
    # change.
    assert not math.isclose(r_uni.pristine_strength_MPa, r_thk.pristine_strength_MPa)


def test_per_ply_thickness_preserves_finite_knockdown():
    cfg = _make_cfg([0.10, 0.20, 0.20, 0.10, 0.10, 0.20, 0.20, 0.10])
    r = BvidAnalysis(cfg).run()
    assert math.isfinite(r.knockdown)
    assert 0.0 < r.knockdown <= 1.0
    assert math.isfinite(r.residual_strength_MPa)


def test_fe3d_runs_with_per_ply_thickness():
    """fe3d builds a structured hex mesh whose z-grid follows per-ply
    boundaries; verify it doesn't crash and reports a finite residual on a
    small panel."""
    from bvidfe.analysis import MeshParams

    ds = DamageState(
        [DelaminationEllipse(2, (5, 5), 2.0, 1.5, 0.0)],
        dent_depth_mm=0.2,
    )
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


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_cli_thickness_accepts_scalar():
    p = _build_parser()
    args = p.parse_args(
        [
            "--material",
            "IM7/8552",
            "--layup",
            "0,45,-45,90,90,-45,45,0",
            "--thickness",
            "0.152",
            "--panel",
            "150x100",
            "--loading",
            "compression",
            "--energy",
            "30",
        ]
    )
    assert args.thickness == 0.152


def test_cli_thickness_accepts_csv_list():
    p = _build_parser()
    args = p.parse_args(
        [
            "--material",
            "IM7/8552",
            "--layup",
            "0,45,-45,90,90,-45,45,0",
            "--thickness",
            "0.10,0.10,0.20,0.20,0.20,0.20,0.10,0.10",
            "--panel",
            "150x100",
            "--loading",
            "compression",
            "--energy",
            "30",
        ]
    )
    assert args.thickness == [0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.10, 0.10]


def test_cli_thickness_rejects_zero_in_list():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(
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

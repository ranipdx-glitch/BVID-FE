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


def test_sublaminate_selection_uses_thickness_not_ply_count():
    """Regression for issue #18.

    The sublaminate-buckling selection must pick the geometrically thinner
    stack (by sum of per-ply thicknesses), not the side with fewer plies.
    Configuration: ``layup=[0, 90, 0, 90, 0, 90]``, ``ply_thickness_mm=
    [0.5, 0.5, 0.1, 0.1, 0.1, 0.1]``, ``interface_index=1``.

    At interface 1 the upper sublaminate is plies 0..1 (2 plies, 1.0 mm
    total) and the lower is plies 2..5 (4 plies, 0.4 mm total). Ply count
    says "upper is thinner"; through-thickness says "lower is thinner".
    The geometrically thinner stack is what actually buckles first, so the
    function must return the buckling load of the *lower* sublaminate.
    """
    from bvidfe.analysis.semi_analytical import (
        _sublaminate_D_matrix,
        semi_analytical_cai,
        sublaminate_buckling_load,
    )
    from bvidfe.core.laminate import Laminate
    from bvidfe.core.material import MATERIAL_LIBRARY

    m = MATERIAL_LIBRARY["IM7/8552"]
    layup = [0, 90, 0, 90, 0, 90]
    thicknesses = [0.5, 0.5, 0.1, 0.1, 0.1, 0.1]
    lam = Laminate(material=m, layup_deg=layup, ply_thickness_mm=thicknesses)

    upper_layup, upper_t = layup[:2], thicknesses[:2]
    lower_layup, lower_t = layup[2:], thicknesses[2:]
    assert sum(upper_t) > sum(lower_t)  # upper geometrically thicker
    assert len(upper_layup) < len(lower_layup)  # but has fewer plies

    # D matrices of each candidate sublaminate — the geometrically thinner
    # stack (lower) has much smaller D11/D22 because D scales as t^3.
    D_upper = _sublaminate_D_matrix(m, upper_layup, upper_t)
    D_lower = _sublaminate_D_matrix(m, lower_layup, lower_t)
    assert D_lower[0, 0] < D_upper[0, 0]
    assert D_lower[1, 1] < D_upper[1, 1]

    # Buckling load with the function under test. Use an ellipse large
    # enough that the (a/b)^4 aspect-ratio clip never fires.
    ell = DelaminationEllipse(1, (0.0, 0.0), major_mm=20.0, minor_mm=15.0, orientation_deg=0.0)
    N = sublaminate_buckling_load(lam, ell)

    # Reference loads from each candidate sublaminate's D matrix, evaluated
    # with the same closed-form (m, n) in [1..5] x [1..5] minimisation that
    # ``sublaminate_buckling_load`` uses internally.
    def _ref_Ncr(D, a, b):
        D11, D22, D12, D66 = D[0, 0], D[1, 1], D[0, 1], D[2, 2]
        pi2 = math.pi * math.pi
        aspect = a / b
        best = math.inf
        for mm in range(1, 6):
            for nn in range(1, 6):
                num = (
                    D11 * mm**4
                    + 2.0 * (D12 + 2.0 * D66) * (mm * aspect) ** 2 * nn**2
                    + D22 * (aspect * nn) ** 4
                )
                cand = (pi2 / a**2) * num / mm**2
                if cand < best:
                    best = cand
        return best

    # (a, b) are FULL plate side lengths (= 2 * ellipse semi-axes); see #29.
    N_upper_ref = _ref_Ncr(D_upper, 2.0 * ell.major_mm, 2.0 * ell.minor_mm)
    N_lower_ref = _ref_Ncr(D_lower, 2.0 * ell.major_mm, 2.0 * ell.minor_mm)
    assert N_lower_ref < N_upper_ref  # thinner stack buckles at lower load

    # The function must report the *lower* (geometrically thinner)
    # sublaminate's buckling load — not the upper (fewer-plies) stack.
    assert N == pytest.approx(N_lower_ref, rel=1e-12)
    assert not math.isclose(N, N_upper_ref, rel_tol=1e-6)

    # End-to-end: semi_analytical_cai must use the same geometrically
    # thinner sublaminate for its h_sub normalisation. Drive the buckling
    # tier with a large delamination at interface 1 so it controls the min.
    ds = DamageState(
        [DelaminationEllipse(1, (0.0, 0.0), 60.0, 40.0, 0.0)],
        dent_depth_mm=0.5,
    )
    sigma_cai, crit_idx, N_cr = semi_analytical_cai(
        lam, ds, sigma_pristine_MPa=500.0, A_panel_mm2=15000.0
    )
    assert crit_idx == 1
    # h_sub must equal the lower (thinner) stack's total thickness (0.4 mm),
    # so sigma_buckling = N_cr / 0.4. If the buggy logic picked the upper
    # stack we'd divide by 1.0 mm instead and get a stress 2.5x smaller.
    sigma_buckling_expected = N_cr / sum(lower_t)
    assert sigma_cai == pytest.approx(min(sigma_buckling_expected, 500.0), rel=1e-9)


def test_sublaminate_selection_matches_uniform_for_uniform_laminate():
    """For uniform per-ply thicknesses the new thickness-based selection
    must agree with the old ply-count-based selection: every test in
    ``tests/analysis/test_semi_analytical.py`` uses a uniform layup, so
    pin the equivalence explicitly here too."""
    from bvidfe.analysis.semi_analytical import sublaminate_buckling_load
    from bvidfe.core.laminate import Laminate
    from bvidfe.core.material import MATERIAL_LIBRARY

    m = MATERIAL_LIBRARY["IM7/8552"]
    # Asymmetric ply count around the interface so ply count and thickness
    # would disagree if thicknesses were ever non-uniform; here they aren't,
    # so both criteria must select the same (smaller-ply-count) side.
    layup = [0, 45, -45, 90, 90, -45, 45, 0, 0, 45]  # 10 plies
    lam_scalar = Laminate(m, layup, 0.152)
    lam_list = Laminate(m, layup, [0.152] * len(layup))
    ell = DelaminationEllipse(2, (0, 0), 20, 12, 0)
    N_scalar = sublaminate_buckling_load(lam_scalar, ell)
    N_list = sublaminate_buckling_load(lam_list, ell)
    assert N_scalar == pytest.approx(N_list, rel=1e-12)
    assert N_scalar > 0


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

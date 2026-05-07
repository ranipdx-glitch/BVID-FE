"""Cross-tier knockdown consistency checks.

Issue #11: existing tests cover each tier in isolation but never assert the
documented cross-tier relationships from README "Knockdown definition and
cross-tier comparability":

  * All three tiers share the same pristine baseline (thickness-weighted
    ply-average), so ``pristine_strength_MPa`` must be identical.
  * For TAI, ``empirical`` and ``semi_analytical`` delegate to the same
    Whitney-Nuismer point-stress formula, so their knockdowns must match
    exactly.
  * For CAI, ``semi_analytical`` adds a sublaminate-buckling floor to the
    empirical Soutis term, so its residual is always <= the empirical
    residual on the same input — and therefore ``semi_analytical.knockdown
    <= empirical.knockdown``.
  * On a pristine input (no impact damage, zero DPA) all three tiers must
    return ``knockdown == 1.0`` exactly (empirical / semi_analytical) or
    very close to 1.0 (fe3d, capped at sigma_pristine_MPa).
"""

from __future__ import annotations

import math

import pytest

from bvidfe.analysis import AnalysisConfig, BvidAnalysis, MeshParams
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.impact.mapping import ImpactEvent


def _base_kwargs(loading: str = "compression"):
    return dict(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading=loading,
        impact=ImpactEvent(20.0, ImpactorGeometry(), mass_kg=5.5),
    )


def _run(tier: str, loading: str = "compression"):
    return BvidAnalysis(AnalysisConfig(tier=tier, **_base_kwargs(loading))).run()


def test_pristine_strength_identical_across_tiers():
    """All three tiers share _pristine_strength()."""
    e = _run("empirical")
    s = _run("semi_analytical")
    assert e.pristine_strength_MPa == s.pristine_strength_MPa


def test_tai_empirical_equals_semi_analytical():
    """semi_analytical_tai delegates unchanged to whitney_nuismer_tai."""
    e = _run("empirical", loading="tension")
    s = _run("semi_analytical", loading="tension")
    assert e.knockdown == pytest.approx(s.knockdown, rel=1e-12)


def test_cai_semi_analytical_le_empirical():
    """semi_analytical_cai = min(empirical_soutis, sublaminate_buckling)."""
    e = _run("empirical", loading="compression")
    s = _run("semi_analytical", loading="compression")
    assert s.knockdown <= e.knockdown + 1e-9


def test_pristine_input_yields_unity_knockdown():
    """With explicit DamageState() containing no delaminations and zero
    dent depth, both empirical and semi_analytical must return knockdown
    exactly 1.0 (no early-exit short-circuits, no floating-point drift)."""
    pristine = DamageState(delaminations=[], dent_depth_mm=0.0)
    common = dict(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(100, 50),
        loading="compression",
        damage=pristine,
    )
    for tier in ("empirical", "semi_analytical"):
        cfg = AnalysisConfig(tier=tier, **common)
        r = BvidAnalysis(cfg).run()
        assert r.knockdown == pytest.approx(1.0, rel=1e-12), (
            f"tier={tier} knockdown={r.knockdown!r} should be 1.0 on pristine input"
        )


def test_knockdown_is_finite_and_bounded():
    """No tier should ever return NaN, Inf, or knockdown outside [0, 1+ε]."""
    for tier in ("empirical", "semi_analytical"):
        r = _run(tier)
        assert math.isfinite(r.knockdown)
        assert 0.0 <= r.knockdown <= 1.0 + 1e-9


def test_residual_never_exceeds_pristine():
    """Per the per-tier sigma cap (semi_analytical & fe3d explicitly,
    empirical implicitly via Soutis kd <= 1) residual <= pristine."""
    for tier in ("empirical", "semi_analytical"):
        r = _run(tier)
        assert r.residual_strength_MPa <= r.pristine_strength_MPa + 1e-6

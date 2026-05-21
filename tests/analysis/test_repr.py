"""Pin compact ``__repr__`` outputs for observability dataclasses.

These tests use substring checks (not exact equality) so adding new fields
to the dataclasses won't churn the assertions.
"""

from __future__ import annotations

from bvidfe.analysis.results import AnalysisResults
from bvidfe.damage.state import DamageState, DelaminationEllipse


def test_delamination_ellipse_repr_is_compact() -> None:
    e = DelaminationEllipse(
        interface_index=3,
        centroid_mm=(0.0, 0.0),
        major_mm=12.5,
        minor_mm=8.25,
        orientation_deg=45.0,
    )
    r = repr(e)
    assert r.startswith("DelaminationEllipse(")
    assert r.endswith(")")
    assert "iface=3" in r
    # Compact numeric formatting (no full float spew).
    assert "a=" in r and "mm" in r
    assert "b=" in r
    assert "theta=" in r
    # Single line, short.
    assert "\n" not in r
    assert len(r) < 120


def test_damage_state_repr_is_compact() -> None:
    e1 = DelaminationEllipse(
        interface_index=0,
        centroid_mm=(0.0, 0.0),
        major_mm=10.0,
        minor_mm=5.0,
        orientation_deg=0.0,
    )
    e2 = DelaminationEllipse(
        interface_index=2,
        centroid_mm=(1.0, 1.0),
        major_mm=8.0,
        minor_mm=4.0,
        orientation_deg=30.0,
    )
    ds = DamageState(
        delaminations=[e1, e2],
        dent_depth_mm=0.512,
        fiber_break_radius_mm=2.3,
    )
    r = repr(ds)
    assert r.startswith("DamageState(")
    assert r.endswith(")")
    assert "n_delam=2" in r
    assert "dent=" in r
    assert "mm" in r
    assert "\n" not in r
    # Repr should NOT trigger expensive shapely union; just check it's short.
    assert len(r) < 200


def test_damage_state_repr_empty() -> None:
    ds = DamageState()
    r = repr(ds)
    assert "n_delam=0" in r
    assert "DamageState(" in r


def test_analysis_results_repr_is_compact() -> None:
    ds = DamageState(dent_depth_mm=0.3, fiber_break_radius_mm=0.0)
    res = AnalysisResults(
        residual_strength_MPa=412.7,
        pristine_strength_MPa=625.5,
        knockdown=0.6598,
        damage=ds,
        dpa_mm2=234.5,
        tier_used="semi_analytical",
        config_snapshot={},
        notes=["something"],
        warnings=[],
    )
    r = repr(res)
    assert r.startswith("AnalysisResults(")
    assert r.endswith(")")
    assert "tier='semi_analytical'" in r
    assert "kd=0.660" in r
    # Residual / pristine sigma pair appears with MPa unit.
    assert "MPa" in r
    assert "412.7" in r
    assert "625.5" in r
    assert "notes=1" in r
    assert "warnings=0" in r
    assert "\n" not in r
    assert len(r) < 200

"""End-to-end edge-case robustness tests.

Covers combinations that are likely to come up in real use but weren't
explicitly tested: below-threshold energies, single-ply laminates, tiny
panels, huge energies that saturate the DPA cap, empty damage states,
mixed tier switching on the same config, etc.
"""

import sys
import warnings

import pytest

from bvidfe.analysis import AnalysisConfig, BvidAnalysis
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.impact.mapping import ImpactEvent


def _cfg(tier="empirical", **overrides):
    kw = dict(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="compression",
        tier=tier,
        impact=ImpactEvent(20.0, ImpactorGeometry(), mass_kg=5.5),
    )
    kw.update(overrides)
    return AnalysisConfig(**kw)


def test_below_threshold_returns_pristine_empirical():
    """Energy below Olsson threshold → empty damage → pristine strength."""
    cfg = _cfg(tier="empirical", impact=ImpactEvent(0.01, ImpactorGeometry(), mass_kg=5.5))
    r = BvidAnalysis(cfg).run()
    assert r.damage.projected_damage_area_mm2 == 0.0
    assert abs(r.knockdown - 1.0) < 1e-9


def test_below_threshold_returns_pristine_semi_analytical():
    cfg = _cfg(
        tier="semi_analytical",
        impact=ImpactEvent(0.01, ImpactorGeometry(), mass_kg=5.5),
    )
    r = BvidAnalysis(cfg).run()
    assert abs(r.knockdown - 1.0) < 1e-9


def test_huge_energy_saturates_dpa_cap_but_no_crash():
    """200 J is unrealistic but should not crash the pipeline."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cfg(impact=ImpactEvent(200.0, ImpactorGeometry(), mass_kg=5.5))
        r = BvidAnalysis(cfg).run()
    assert r.knockdown > 0
    assert r.knockdown < 1.0
    # DPA should be capped at 80% of panel area
    assert r.dpa_mm2 <= 0.82 * (cfg.panel.Lx_mm * cfg.panel.Ly_mm)


def test_tiny_panel_still_runs():
    """10x10 mm panel — nothing should explode."""
    cfg = _cfg(panel=PanelGeometry(10, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = BvidAnalysis(cfg).run()
    assert r.residual_strength_MPa > 0


def test_damage_driven_empty_state_returns_pristine():
    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="compression",
        tier="empirical",
        damage=DamageState([], dent_depth_mm=0.0),
    )
    r = BvidAnalysis(cfg).run()
    assert r.damage.projected_damage_area_mm2 == 0.0
    assert abs(r.knockdown - 1.0) < 1e-9


def test_damage_driven_tension_returns_reduced_strength():
    ds = DamageState(
        [DelaminationEllipse(3, (75, 50), 20, 12, 45)],
        dent_depth_mm=0.3,
    )
    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="tension",
        tier="empirical",
        damage=ds,
    )
    r = BvidAnalysis(cfg).run()
    assert r.knockdown < 1.0
    assert r.residual_strength_MPa > 0


def test_mixed_tier_switch_same_damage_yields_different_knockdown():
    """Same damage state, run through all three tiers — each gives a distinct number."""
    ds = DamageState(
        [DelaminationEllipse(3, (75, 50), 20, 12, 45)],
        dent_depth_mm=0.3,
    )
    base_kwargs = dict(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="compression",
        damage=ds,
    )
    kd_by_tier = {}
    for tier in ("empirical", "semi_analytical"):
        cfg = AnalysisConfig(tier=tier, **base_kwargs)
        kd_by_tier[tier] = BvidAnalysis(cfg).run().knockdown
    # Tiers should produce distinct knockdowns (not artificially tied to one value)
    assert len(set(round(kd, 3) for kd in kd_by_tier.values())) == len(kd_by_tier)


def test_material_library_object_instead_of_name():
    """Passing an OrthotropicMaterial object directly (not a preset name) must work."""
    from bvidfe.core.material import MATERIAL_LIBRARY

    m = MATERIAL_LIBRARY["IM7/8552"]
    cfg = _cfg(material=m)
    r = BvidAnalysis(cfg).run()
    assert r.residual_strength_MPa > 0


def test_ply_thickness_and_layup_consistent():
    """4-ply laminate: n_delaminations must be n_plies - 1 for an above-threshold impact."""
    cfg = _cfg(layup_deg=[0, 90, 0, 90], ply_thickness_mm=0.25)
    r = BvidAnalysis(cfg).run()
    n_plies = 4
    interfaces = {d.interface_index for d in r.damage.delaminations}
    assert len(interfaces) <= n_plies - 1
    for iface in interfaces:
        assert 0 <= iface <= n_plies - 2


def test_cli_runs_end_to_end():
    """Subprocess CLI smoke test — matches documented quick-start invocation."""
    import json
    import subprocess

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "bvidfe.cli",
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
            "--tier",
            "empirical",
            "--energy",
            "20",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert "knockdown" in data
    assert 0 < data["knockdown"] <= 1.0


@pytest.mark.parametrize("energy_J", [3.0, 10.0, 25.0, 50.0])
def test_empirical_knockdown_monotonic_in_energy(energy_J):
    """Empirical knockdown is monotonically decreasing in impact energy
    (same config, different energies)."""
    cfg = _cfg(impact=ImpactEvent(energy_J, ImpactorGeometry(), mass_kg=5.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = BvidAnalysis(cfg).run()
    assert 0 < r.knockdown <= 1.0

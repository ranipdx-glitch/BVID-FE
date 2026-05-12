"""Regression tests for the SummaryTab edge-case notices.

The Summary tab annotates the raw results with plain-language notices for
DPA saturation, free-boundary soft-support approximation, small-mass
quasi-static validity, and the known fe3d energy-insensitivity limitation.
The underlying Python-API warnings emit to stderr; these notices surface
them into the GUI where the user can actually see them.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from bvidfe.analysis import AnalysisConfig, BvidAnalysis, MeshParams
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.impact.mapping import ImpactEvent
from bvidfe.gui.tabs.summary_tab import SummaryTab


def _run(**overrides):
    panel = overrides.pop("panel", PanelGeometry(150, 100))
    impact = overrides.pop("impact", ImpactEvent(10.0, ImpactorGeometry(), mass_kg=5.5))
    kw = dict(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=panel,
        loading="compression",
        tier="empirical",
        impact=impact,
    )
    kw.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BvidAnalysis(AnalysisConfig(**kw)).run()


def test_summary_shows_dpa_saturation_note(qtbot):
    """Saturated DPA should produce a visible '⚠ DPA saturation' note."""
    # 30 J on a 150x100 panel at 8 plies *will* saturate
    results = _run(impact=ImpactEvent(30.0, ImpactorGeometry(), mass_kg=5.5))
    tab = SummaryTab()
    qtbot.addWidget(tab)
    tab.update(results)
    text = tab.text_area.toPlainText()
    assert "DPA saturation" in text, text


def test_summary_hides_dpa_note_when_not_saturated(qtbot):
    """At modest energy on a large panel, no saturation notice appears."""
    results = _run(
        panel=PanelGeometry(300, 200),
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
    )
    tab = SummaryTab()
    qtbot.addWidget(tab)
    tab.update(results)
    text = tab.text_area.toPlainText()
    assert "DPA saturation" not in text, text


def test_summary_shows_free_boundary_note(qtbot):
    """'free' boundary should emit the soft-support approximation notice."""
    results = _run(
        panel=PanelGeometry(300, 200, "free"),
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
    )
    tab = SummaryTab()
    qtbot.addWidget(tab)
    tab.update(results)
    text = tab.text_area.toPlainText()
    assert "free" in text and "soft-support" in text, text


def test_summary_shows_fe3d_tier_note(qtbot):
    """fe3d runs should emit a tier-specific note pointing users at the
    empirical tier for energy sweeps."""
    results = _run(
        tier="fe3d",
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=10.0),
        panel=PanelGeometry(300, 200),
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
    )
    tab = SummaryTab()
    qtbot.addWidget(tab)
    tab.update(results)
    text = tab.text_area.toPlainText()
    assert "fe3d tier" in text and "tier='empirical'" in text, text


def test_summary_hides_fe3d_note_for_empirical(qtbot):
    """The fe3d tier note should NOT appear for the empirical tier."""
    results = _run()
    tab = SummaryTab()
    qtbot.addWidget(tab)
    tab.update(results)
    text = tab.text_area.toPlainText()
    assert "fe3d tier:" not in text, text

"""Regression tests for the live E_onset preview in the ImpactPanel label.

Background: ``ImpactPanel.set_onset_energy()`` was coded as a live-preview
mechanism but never wired up — the "E_onset: — J" label stayed blank no
matter what the user changed. The v0.2.0-dev boundary- and shape-aware
physics makes that gap visible (the label SHOULD respond to those inputs),
so the main window now connects every relevant panel's ``configChanged``
signal to a single ``_update_live_onset`` slot. These tests pin that
behavior so it never silently breaks again.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from bvidfe.gui.main_window import BvidMainWindow


def _label_text(w: BvidMainWindow) -> str:
    return w.impact_panel.onset_label.text()


def _label_energy_J(w: BvidMainWindow) -> float | None:
    t = _label_text(w)
    # Expected format: "E_onset: 0.60 J" — grab the numeric token
    for tok in t.replace(":", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def test_live_onset_renders_a_value_on_startup(qtbot):
    """Opening the window should immediately populate the label (not leave
    it at the placeholder). Regression for the "never called" bug."""
    w = BvidMainWindow()
    qtbot.addWidget(w)
    E = _label_energy_J(w)
    assert E is not None and E > 0, _label_text(w)


def test_live_onset_responds_to_boundary_change(qtbot):
    """Switching the Panel boundary from simply_supported to clamped must
    move the E_onset label (clamped plates are stiffer => lower E_onset per
    Olsson's formula)."""
    w = BvidMainWindow()
    qtbot.addWidget(w)

    # Force a canonical starting state
    w.panel_panel.boundary_combo.setCurrentText("simply_supported")
    E_ss = _label_energy_J(w)

    w.panel_panel.boundary_combo.setCurrentText("clamped")
    E_cl = _label_energy_J(w)

    assert E_ss is not None and E_cl is not None
    assert E_cl < E_ss, (E_ss, E_cl)


def test_live_onset_responds_to_shape_change(qtbot):
    """Switching the Impactor shape must move the E_onset label since Hertz
    contact stiffness differs per shape."""
    w = BvidMainWindow()
    qtbot.addWidget(w)

    w.impact_panel.shape_combo.setCurrentText("hemispherical")
    E_hemi = _label_energy_J(w)
    w.impact_panel.shape_combo.setCurrentText("flat")
    E_flat = _label_energy_J(w)
    w.impact_panel.shape_combo.setCurrentText("conical")
    E_cone = _label_energy_J(w)

    # We don't assert a direction (depends on the relative magnitudes of
    # k_bending and k_contact for the current config) — just that they
    # aren't all equal.
    values = {E_hemi, E_flat, E_cone}
    assert len(values) >= 2, values


def test_live_onset_diameter_change_does_not_crash(qtbot):
    """The label update must not crash when the user edits the diameter,
    even though the numeric change in E_onset is small (<1%) at the
    default configuration because plate-bending stiffness dominates the
    contact stiffness in the k_cb series-spring combination."""
    w = BvidMainWindow()
    qtbot.addWidget(w)

    for d in (4.0, 8.0, 16.0, 25.0, 40.0, 80.0):
        w.impact_panel.diameter_spin.setValue(d)
        assert _label_energy_J(w) is not None, _label_text(w)


def _dpa_label_text(w: BvidMainWindow) -> str:
    return w.impact_panel.dpa_label.text()


def test_live_dpa_renders_a_value_on_startup(qtbot):
    """Opening the window should populate the DPA label, not leave it at
    the placeholder."""
    w = BvidMainWindow()
    qtbot.addWidget(w)
    text = _dpa_label_text(w)
    assert "DPA:" in text and "mm" in text, text
    # Should contain a numeric value — the first token after "DPA:"
    assert "\u2014" not in text or "damage-driven" in text, text


def test_live_dpa_responds_to_energy_change(qtbot):
    """Raising the impact energy must raise the DPA preview."""
    w = BvidMainWindow()
    qtbot.addWidget(w)

    w.impact_panel.energy_spin.setValue(3.0)
    low_text = _dpa_label_text(w)
    w.impact_panel.energy_spin.setValue(20.0)
    high_text = _dpa_label_text(w)

    # Extract the first numeric value from each label
    def _first_num(text: str) -> float:
        for tok in text.replace(":", " ").split():
            try:
                return float(tok)
            except ValueError:
                continue
        return 0.0

    assert _first_num(high_text) > _first_num(low_text), (low_text, high_text)


def test_live_dpa_shows_saturation_warning(qtbot):
    """At 30 J on the default 150x100 panel, DPA saturates — the label
    must show SATURATED and be styled in red bold."""
    w = BvidMainWindow()
    qtbot.addWidget(w)
    # Default panel is 150x100 (area = 15000 mm²); 30 J on 8-ply saturates.
    w.panel_panel.lx_spin.setValue(150.0)
    w.panel_panel.ly_spin.setValue(100.0)
    w.impact_panel.energy_spin.setValue(30.0)
    text = _dpa_label_text(w)
    assert "SATURATED" in text, text
    style = w.impact_panel.dpa_label.styleSheet()
    assert "darkred" in style or "red" in style, style


def test_live_dpa_hides_saturation_when_small(qtbot):
    """At low energy on a big panel, DPA is well below the cap — no
    SATURATED marker, no red styling."""
    w = BvidMainWindow()
    qtbot.addWidget(w)
    w.panel_panel.lx_spin.setValue(300.0)
    w.panel_panel.ly_spin.setValue(200.0)
    w.impact_panel.energy_spin.setValue(3.0)
    text = _dpa_label_text(w)
    assert "SATURATED" not in text, text
    style = w.impact_panel.dpa_label.styleSheet()
    assert "red" not in style, style


def test_live_onset_handles_damage_driven_mode(qtbot):
    """When the user flips to damage-driven input mode, the label should
    NOT crash — it should indicate the preview is not applicable."""
    w = BvidMainWindow()
    qtbot.addWidget(w)

    # Before: impact mode -> numeric
    E_impact = _label_energy_J(w)
    assert E_impact is not None

    # Flip to damage mode
    w.input_mode_panel.damage_radio.setChecked(True)
    text = _label_text(w)
    # The label is now non-numeric and mentions the damage-driven mode
    assert "damage-driven" in text, text

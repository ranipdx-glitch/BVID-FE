import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

from bvidfe.analysis import AnalysisConfig
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.gui.config_io import config_to_dict, config_from_dict
from bvidfe.impact.mapping import ImpactEvent


def _sample_cfg():
    return AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="compression",
        tier="empirical",
        impact=ImpactEvent(30.0, ImpactorGeometry(20.0, "flat"), mass_kg=4.0),
    )


def test_config_round_trip():
    cfg = _sample_cfg()
    d = config_to_dict(cfg)
    cfg2 = config_from_dict(d)
    assert cfg2.material == cfg.material
    assert cfg2.layup_deg == cfg.layup_deg
    assert cfg2.panel.Lx_mm == cfg.panel.Lx_mm
    assert cfg2.impact is not None
    assert cfg2.impact.impactor.shape == "flat"
    assert cfg2.impact.impactor.diameter_mm == 20.0
    assert cfg2.impact.mass_kg == 4.0


def test_config_json_serializable():
    cfg = _sample_cfg()
    d = config_to_dict(cfg)
    s = json.dumps(d)  # must not raise
    assert isinstance(s, str)


def test_damage_only_config_round_trip():
    from bvidfe.damage.state import DamageState, DelaminationEllipse

    ds = DamageState(
        [DelaminationEllipse(3, (75, 50), 20, 12, 45)],
        dent_depth_mm=0.4,
        fiber_break_radius_mm=1.5,
    )
    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(100, 50),
        loading="tension",
        tier="semi_analytical",
        damage=ds,
    )
    d = config_to_dict(cfg)
    cfg2 = config_from_dict(d)
    assert cfg2.damage is not None
    assert len(cfg2.damage.delaminations) == 1
    assert cfg2.damage.delaminations[0].interface_index == 3


def test_main_window_has_file_menu(qtbot):
    from bvidfe.gui.main_window import BvidMainWindow

    w = BvidMainWindow()
    qtbot.addWidget(w)
    actions = [a.text() for a in w.menuBar().actions()]
    # File menu should exist (title "&File" or "File")
    assert any("File" in t for t in actions)


def _drive_load_config(
    qtbot, monkeypatch, tmp_path, file_text: str | None, *, write_invalid_json: bool = False
):
    """Helper: monkeypatch QFileDialog + QMessageBox, drive _load_config,
    return (title, body) of the QMessageBox.warning call (or (None, None) if
    no warning was raised)."""
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from bvidfe.gui.main_window import BvidMainWindow

    path = tmp_path / "cfg.json"
    if write_invalid_json:
        path.write_text(file_text or "{not valid json")
    elif file_text is not None:
        path.write_text(file_text)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(path), "")),
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda parent, title, body, *a, **kw: captured.append((title, body))),
    )

    w = BvidMainWindow()
    qtbot.addWidget(w)
    w._load_config()
    return captured[0] if captured else (None, None)


def test_load_config_malformed_json_distinct_message(qtbot, monkeypatch, tmp_path):
    title, body = _drive_load_config(
        qtbot,
        monkeypatch,
        tmp_path,
        "{not valid json",
        write_invalid_json=True,
    )
    assert title == "Malformed JSON"
    assert "JSON" in body or "json" in body


def test_load_config_missing_field_distinct_message(qtbot, monkeypatch, tmp_path):
    # Valid JSON but missing required 'material' key
    title, body = _drive_load_config(
        qtbot,
        monkeypatch,
        tmp_path,
        '{"layup_deg": [0, 90], "ply_thickness_mm": 0.152}',
    )
    assert title == "Missing field"
    assert "missing" in body.lower()


def test_load_config_invalid_value_distinct_message(qtbot, monkeypatch, tmp_path):
    # Valid JSON, all keys present, but ply_thickness is a non-numeric string
    bad = {
        "material": "IM7/8552",
        "layup_deg": [0, 90, 0, 90],
        "ply_thickness_mm": "abc",
        "panel": {"Lx_mm": 100, "Ly_mm": 50, "boundary": "simply_supported"},
        "loading": "compression",
        "tier": "empirical",
        "impact": {
            "energy_J": 10.0,
            "impactor": {"diameter_mm": 16.0, "shape": "hemispherical"},
            "mass_kg": 5.5,
        },
    }
    title, body = _drive_load_config(qtbot, monkeypatch, tmp_path, json.dumps(bad))
    assert title == "Invalid value"


def test_damage_panel_skipped_rows_recorded_not_silent(qtbot):
    """Issue #6: malformed delamination rows must be visible (via
    skipped_rows + log warning) — not silently dropped."""
    from PyQt6.QtWidgets import QTableWidgetItem

    from bvidfe.gui.panels.damage_panel import DamagePanel

    panel = DamagePanel()
    qtbot.addWidget(panel)
    # Add one valid row programmatically
    panel.add_delamination_row(0, 5.0, 5.0, 4.0, 2.0, 0.0)
    # Add a second row whose 'iface' cell is non-numeric
    panel.add_delamination_row(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    panel.table.setItem(1, 0, QTableWidgetItem("not-a-number"))

    ds = panel.get_damage_state()
    assert len(ds.delaminations) == 1  # bad row was skipped
    assert hasattr(panel, "skipped_rows")
    assert len(panel.skipped_rows) == 1
    assert panel.skipped_rows[0][0] == 1  # the second row (zero-indexed)

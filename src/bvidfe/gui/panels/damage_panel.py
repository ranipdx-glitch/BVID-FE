"""Damage state input panel (delamination table + dent depth + C-scan import)."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bvidfe.damage.io import CScanSchemaError, load_cscan_json
from bvidfe.damage.state import DamageState, DelaminationEllipse

_log = logging.getLogger("bvidfe.gui")


class DamagePanel(QWidget):
    """Input panel for a DamageState (inspection-driven path)."""

    configChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Delamination table: iface | cx | cy | major | minor | angle
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["iface", "cx", "cy", "major", "minor", "angle"])

        self.dent_spin = QDoubleSpinBox()
        self.dent_spin.setRange(0.0, 10.0)
        self.dent_spin.setDecimals(3)
        self.dent_spin.setValue(0.0)

        self.fb_spin = QDoubleSpinBox()
        self.fb_spin.setRange(0.0, 100.0)
        self.fb_spin.setDecimals(3)
        self.fb_spin.setValue(0.0)

        self.add_button = QPushButton("Add row")
        self.remove_button = QPushButton("Remove row")
        self.import_button = QPushButton("Import C-scan\u2026")

        self.add_button.clicked.connect(lambda: self.add_delamination_row())
        self.remove_button.clicked.connect(self._remove_current_row)
        self.import_button.clicked.connect(self._import_cscan)

        for w in (self.dent_spin, self.fb_spin):
            w.valueChanged.connect(lambda _: self.configChanged.emit())
        self.table.itemChanged.connect(lambda _: self.configChanged.emit())

        v = QVBoxLayout(self)
        v.addWidget(self.table)

        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.import_button)
        v.addLayout(buttons)

        form = QFormLayout()
        form.addRow("Dent depth (mm):", self.dent_spin)
        form.addRow("Fiber break radius (mm):", self.fb_spin)
        v.addLayout(form)

    def add_delamination_row(
        self,
        interface_index: int = 0,
        cx: float = 0.0,
        cy: float = 0.0,
        major: float = 10.0,
        minor: float = 5.0,
        angle: float = 0.0,
    ) -> None:
        """Insert a new delamination row into the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate([interface_index, cx, cy, major, minor, angle]):
            self.table.setItem(row, col, QTableWidgetItem(str(val)))
        self.configChanged.emit()

    def _remove_current_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.configChanged.emit()

    def _import_cscan(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Import C-scan JSON", "", "JSON (*.json)")
        if not path_str:
            return
        try:
            ds = load_cscan_json(Path(path_str))
        except CScanSchemaError as exc:
            QMessageBox.warning(self, "Invalid C-scan", str(exc))
            return

        # Populate table from loaded DamageState
        self.table.setRowCount(0)
        for d in ds.delaminations:
            self.add_delamination_row(
                d.interface_index,
                d.centroid_mm[0],
                d.centroid_mm[1],
                d.major_mm,
                d.minor_mm,
                d.orientation_deg,
            )
        self.dent_spin.setValue(ds.dent_depth_mm)
        self.fb_spin.setValue(ds.fiber_break_radius_mm)

    def get_damage_state(self) -> DamageState:
        """Read the table and spinboxes into a DamageState.

        Skipped rows are reported via ``self.skipped_rows`` (zero-indexed
        row numbers) and as a ``bvidfe.gui`` log warning, so a typo or
        partially-edited cell can no longer silently drop a delamination
        from the analysis. The main window inspects ``skipped_rows`` after
        the call to surface a status-bar message; programmatic callers can
        do the same.
        """
        dels: list[DelaminationEllipse] = []
        skipped: list[tuple[int, str]] = []
        for row in range(self.table.rowCount()):
            try:
                iface = int(float(self.table.item(row, 0).text()))
                cx = float(self.table.item(row, 1).text())
                cy = float(self.table.item(row, 2).text())
                major = float(self.table.item(row, 3).text())
                minor = float(self.table.item(row, 4).text())
                angle = float(self.table.item(row, 5).text())
                dels.append(DelaminationEllipse(iface, (cx, cy), major, minor, angle))
            except (ValueError, AttributeError) as exc:
                skipped.append((row, str(exc)))
                _log.warning(
                    "DamagePanel: skipping malformed delamination row %d (%s)",
                    row,
                    exc,
                )
        self.skipped_rows: list[tuple[int, str]] = skipped
        return DamageState(
            delaminations=dels,
            dent_depth_mm=float(self.dent_spin.value()),
            fiber_break_radius_mm=float(self.fb_spin.value()),
        )

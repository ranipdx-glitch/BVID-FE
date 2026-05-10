"""BVID-FE main window (QMainWindow)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from bvidfe.gui.workers import AnalysisWorker, SweepWorker
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bvidfe.gui.config_io import config_from_dict, config_to_dict

from bvidfe.gui.panels.analysis_panel import AnalysisPanel
from bvidfe.gui.panels.damage_panel import DamagePanel
from bvidfe.gui.panels.impact_panel import ImpactPanel
from bvidfe.gui.panels.input_mode_panel import InputModePanel
from bvidfe.gui.panels.material_panel import MaterialPanel
from bvidfe.gui.panels.panel_panel import PanelPanel
from bvidfe.gui.panels.sweep_panel import SweepPanel


class BvidMainWindow(QMainWindow):
    """Main application window for BVID-FE."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BVID-FE")
        self.resize(1200, 800)

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

        # --- Input panels ---
        self.material_panel = MaterialPanel(self)
        self.panel_panel = PanelPanel(self)
        self.input_mode_panel = InputModePanel(self)
        self.impact_panel = ImpactPanel(self)
        self.damage_panel = DamagePanel(self)
        self.analysis_panel = AnalysisPanel(self)
        self.sweep_panel = SweepPanel(self)

        # Wire input mode toggle to enable/disable the relevant data panels
        self.input_mode_panel.configChanged.connect(self._on_mode_changed)
        self._on_mode_changed()  # set initial enabled state

        # Wire live-preview of Olsson onset energy. Every input that feeds into
        # E_onset(lam, panel, impactor) now triggers a recomputation so the user
        # sees the threshold update as they type — including the boundary-aware
        # bending stiffness and the shape-aware contact stiffness added in
        # v0.2.0-dev. The ``ImpactPanel.set_onset_energy`` method already
        # existed but was never being called by anything; this plumbing wires
        # it in for the first time.
        self.material_panel.configChanged.connect(self._update_live_onset)
        self.panel_panel.configChanged.connect(self._update_live_onset)
        self.impact_panel.configChanged.connect(self._update_live_onset)
        self.input_mode_panel.configChanged.connect(self._update_live_onset)
        # Initial render
        self._update_live_onset()

        # Keep a reference to workers to prevent garbage-collection during run
        self._analysis_worker: AnalysisWorker | None = None
        self._sweep_worker: SweepWorker | None = None
        self._tier_compare_worker = None  # type: ignore[assignment]  # TierComparisonWorker
        self.analysis_panel.runRequested.connect(self._run_analysis)
        self.sweep_panel.sweepRequested.connect(self._run_sweep)

        panel_container = QWidget()
        layout = QVBoxLayout(panel_container)
        for p in (
            self.material_panel,
            self.panel_panel,
            self.input_mode_panel,
            self.impact_panel,
            self.damage_panel,
            self.analysis_panel,
            self.sweep_panel,
        ):
            layout.addWidget(p)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel_container)

        dock = QDockWidget("Inputs", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # Central tabbed results area
        from bvidfe.gui.tabs.buckling_tab import BucklingTab
        from bvidfe.gui.tabs.damage_map_tab import DamageMapTab
        from bvidfe.gui.tabs.knockdown_tab import KnockdownTab
        from bvidfe.gui.tabs.mesh_3d_tab import Mesh3DTab
        from bvidfe.gui.tabs.stress_field_tab import StressFieldTab
        from bvidfe.gui.tabs.summary_tab import SummaryTab

        self.results_tabs = QTabWidget(self)
        self.summary_tab = SummaryTab(self)
        self.damage_map_tab = DamageMapTab(self)
        self.knockdown_tab = KnockdownTab(self)
        self.mesh_tab = Mesh3DTab(self)
        self.buckling_tab = BucklingTab(self)
        self.stress_tab = StressFieldTab(self)

        self.results_tabs.addTab(self.summary_tab, "Summary")
        self.results_tabs.addTab(self.damage_map_tab, "Damage Map")
        self.results_tabs.addTab(self.knockdown_tab, "Knockdown Curve")
        self.results_tabs.addTab(self.mesh_tab, "Damage View")
        self.results_tabs.addTab(self.buckling_tab, "Buckling Eigenvalues")
        self.results_tabs.addTab(self.stress_tab, "Damage Severity")
        self.setCentralWidget(self.results_tabs)

        self._last_result = None
        self._last_config = None
        self._build_file_menu()
        self._build_help_menu()

    def _on_mode_changed(self) -> None:
        """Toggle impact / damage panels based on the selected input mode."""
        mode = self.input_mode_panel.current_mode()
        self.impact_panel.setEnabled(mode == "impact")
        self.damage_panel.setEnabled(mode == "damage")

    def _update_live_onset(self) -> None:
        """Recompute and display the Olsson onset energy AND the predicted
        DPA preview in the Impact panel.

        Called every time any input that feeds ``onset_energy()`` or the DPA
        target changes — material, layup, ply thickness, panel dimensions /
        boundary, impactor diameter / shape / mass, and impact energy. Both
        previews are boundary- and shape-aware since v0.2.0-dev, so the user
        should see the labels move immediately in response to a boundary,
        shape, or mass toggle.

        The DPA preview also surfaces the 80% panel-area saturation cap in
        red bold text so the user sees "⚠ SATURATED" before clicking Run —
        previously they'd only discover saturation from the Summary-tab
        notice after a completed run.

        Invalid / partial inputs (e.g. the user is mid-edit) silently blank
        both labels rather than crashing the GUI.
        """
        # Only show the preview in impact-driven mode; in damage-driven mode
        # there is no impact event and E_onset / DPA are undefined.
        if self.input_mode_panel.current_mode() != "impact":
            self.impact_panel.onset_label.setText("E_onset: \u2014 J (damage-driven mode)")
            self.impact_panel.dpa_label.setText("DPA: \u2014 mm\u00b2 (damage-driven mode)")
            self.impact_panel.dpa_label.setStyleSheet("")
            return
        try:
            import warnings
            from bvidfe.core.laminate import Laminate
            from bvidfe.core.material import MATERIAL_LIBRARY
            from bvidfe.impact.mapping import impact_to_damage
            from bvidfe.impact.olsson import onset_energy

            mat_name = self.material_panel.get_material_name()
            mat = MATERIAL_LIBRARY[mat_name]
            lam = Laminate(
                mat,
                self.material_panel.get_layup_deg(),
                self.material_panel.get_ply_thickness_mm(),
            )
            panel = self.panel_panel_as_geometry()
            event = self.impact_panel.get_impact_event()
            with warnings.catch_warnings():
                # Suppress the "free" boundary / small-mass / DPA-cap
                # warnings from the live preview — users will see them in
                # the Summary tab after running.
                warnings.simplefilter("ignore")
                E = onset_energy(lam, panel, event.impactor)
                damage = impact_to_damage(event, lam, panel)
            self.impact_panel.set_onset_energy(E)
            A_panel = panel.Lx_mm * panel.Ly_mm
            self.impact_panel.set_dpa_preview(damage.projected_damage_area_mm2, A_panel)
        except Exception:
            # Any malformed intermediate state during typing — just blank both.
            self.impact_panel.onset_label.setText("E_onset: \u2014 J")
            self.impact_panel.dpa_label.setText("DPA: \u2014 mm\u00b2")
            self.impact_panel.dpa_label.setStyleSheet("")

    def _build_config(self):
        """Assemble an AnalysisConfig from current panel state."""
        from bvidfe.analysis import AnalysisConfig, MeshParams

        panel = self.panel_panel_as_geometry()
        mode = self.input_mode_panel.current_mode()
        impact = self.impact_panel.get_impact_event() if mode == "impact" else None
        damage = self.damage_panel.get_damage_state() if mode == "damage" else None
        if mode == "damage":
            skipped = getattr(self.damage_panel, "skipped_rows", [])
            if skipped:
                rows = ", ".join(str(r) for r, _ in skipped)
                self.statusBar().showMessage(
                    f"Damage panel: skipped {len(skipped)} malformed row(s): {rows}",
                    8000,
                )
        mesh_params = None
        if self.analysis_panel.get_tier() == "fe3d":
            mesh_params = MeshParams(
                elements_per_ply=self.analysis_panel.get_elements_per_ply(),
                in_plane_size_mm=self.analysis_panel.get_in_plane_size_mm(),
            )
        return AnalysisConfig(
            material=self.material_panel.get_material_name(),
            layup_deg=self.material_panel.get_layup_deg(),
            ply_thickness_mm=self.material_panel.get_ply_thickness_mm(),
            panel=panel,
            loading=self.analysis_panel.get_loading(),
            tier=self.analysis_panel.get_tier(),
            impact=impact,
            damage=damage,
            mesh=mesh_params,
        )

    def _run_analysis(self) -> None:
        from bvidfe.gui.workers import AnalysisWorker

        try:
            cfg = self._build_config()
        except (ValueError, AssertionError) as exc:
            self.statusBar().showMessage(f"Invalid config: {exc}", 10000)
            return

        # fe3d mesh-size sanity check. Oversized problems can both hang the
        # GUI and, more seriously, crash the process outright via SIGSEGV from
        # scipy's native sparse solvers when memory is exhausted. The solver
        # itself enforces a hard cap (FE3D_MAX_DOF) via FE3DSizeError; this
        # GUI dialog is a friendly early check.
        if cfg.tier == "fe3d":
            from bvidfe.analysis.fe_mesh import estimate_fe_mesh_size
            from bvidfe.analysis.fe_tier import FE3D_MAX_DOF

            stats = estimate_fe_mesh_size(cfg)
            if stats["n_dof"] > FE3D_MAX_DOF:
                # Past the hard cap: block run outright, explain clearly
                QMessageBox.critical(
                    self,
                    "Mesh too large for fe3d tier",
                    f"The requested fe3d mesh has {stats['n_elements']:,} elements "
                    f"({stats['n_dof']:,} DOFs), which exceeds the safe-size cap "
                    f"of {FE3D_MAX_DOF:,} DOFs.\n\n"
                    f"At this size the pure-Python FE assembler and scipy sparse "
                    f"solvers can exhaust memory and crash the process.\n\n"
                    f"Please do one of:\n"
                    f"  \u2022 increase 'In-plane size (mm)' in the Analysis panel\n"
                    f"  \u2022 decrease 'Elements per ply'\n"
                    f"  \u2022 switch tier to empirical or semi_analytical",
                )
                self.statusBar().showMessage("Run cancelled: mesh too large", 5000)
                return

            if stats["n_elements"] > 50_000 or stats["n_dof"] > 150_000:
                msg = (
                    f"The requested fe3d mesh has {stats['n_elements']:,} elements "
                    f"({stats['n_dof']:,} DOFs).\n\n"
                    f"Python FE in this version is single-threaded; expect multi-minute "
                    f"wall time at this size.\n\n"
                    f"Suggested tweaks:\n"
                    f"  \u2022 increase 'In-plane size (mm)' in the Analysis panel\n"
                    f"  \u2022 decrease 'Elements per ply'\n"
                    f"  \u2022 switch tier to empirical or semi_analytical\n\n"
                    f"Run anyway?"
                )
                result = QMessageBox.question(
                    self,
                    "Large mesh",
                    msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if result != QMessageBox.StandardButton.Yes:
                    self.statusBar().showMessage("Run cancelled", 3000)
                    return

        self._last_config = cfg
        self.statusBar().showMessage("Running analysis\u2026")
        worker = AnalysisWorker(cfg)
        worker.resultReady.connect(self._on_analysis_ready)
        worker.error.connect(self._on_worker_error)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda w=worker: self._on_analysis_worker_finished(w))
        self._analysis_worker = worker
        worker.start()

    def _on_analysis_worker_finished(self, worker) -> None:
        """Cleanup when an AnalysisWorker finishes. Mirrors the SweepWorker
        variant so neither main-window reference can dangle past its Qt
        object being deleteLater()'d."""
        try:
            worker.deleteLater()
        except RuntimeError:
            pass
        if self._analysis_worker is worker:
            self._analysis_worker = None

    def _run_sweep(self) -> None:
        from bvidfe.gui.workers import SweepWorker

        try:
            cfg = self._build_config()
        except (ValueError, AssertionError) as exc:
            self.statusBar().showMessage(f"Invalid config: {exc}", 10000)
            return
        energies = self.sweep_panel.get_energies_J()
        if not energies:
            self.statusBar().showMessage("No energies specified for sweep", 5000)
            return
        csv_path = self.sweep_panel.get_csv_path() or None

        # Same fe3d mesh-size guard as single-run. A sweep is N x single-run
        # cost, so the soft warning threshold here is stricter.
        if cfg.tier == "fe3d":
            from bvidfe.analysis.fe_mesh import estimate_fe_mesh_size
            from bvidfe.analysis.fe_tier import FE3D_MAX_DOF

            stats = estimate_fe_mesh_size(cfg)
            n_runs = len(energies)
            if stats["n_dof"] > FE3D_MAX_DOF:
                QMessageBox.critical(
                    self,
                    "Mesh too large for fe3d tier",
                    f"Sweep mesh has {stats['n_elements']:,} elements / "
                    f"{stats['n_dof']:,} DOFs, exceeding the safe-size cap "
                    f"of {FE3D_MAX_DOF:,}.\n\n"
                    f"Increase 'In-plane size (mm)', decrease 'Elements per ply', "
                    f"or switch to tier='empirical' / 'semi_analytical'.",
                )
                self.statusBar().showMessage("Sweep cancelled: mesh too large", 5000)
                return
            # Soft threshold: N runs * ~single-run cost. Warn at 10k elements * N.
            if stats["n_elements"] * n_runs > 50_000 or n_runs > 20:
                msg = (
                    f"Sweep: {n_runs} fe3d runs, each with {stats['n_elements']:,} "
                    f"elements ({stats['n_dof']:,} DOFs).\n\n"
                    f"Total projected cost \u2248 {n_runs * stats['n_elements']:,} "
                    f"element-solves. Expect multi-minute wall time.\n\n"
                    f"Suggested: start with tier='empirical' for a quick sweep, "
                    f"then use fe3d for spot-checks at key energies.\n\n"
                    f"Run anyway?"
                )
                result = QMessageBox.question(
                    self,
                    "Large fe3d sweep",
                    msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if result != QMessageBox.StandardButton.Yes:
                    self.statusBar().showMessage("Sweep cancelled", 3000)
                    return

        self.statusBar().showMessage("Running sweep\u2026")
        worker = SweepWorker(cfg, energies_J=energies, csv_path=csv_path)
        worker.resultReady.connect(self._on_sweep_ready)
        worker.error.connect(self._on_worker_error)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda w=worker: self._on_sweep_worker_finished(w))
        self._sweep_worker = worker
        worker.start()

    def _on_analysis_ready(self, result) -> None:
        """Called on the Qt main thread when an AnalysisWorker finishes.

        Wrapped in per-tab try/except so any matplotlib/VTK hiccup in one
        tab cannot take down the whole app — Qt 6.5+ aborts the process
        on any unhandled exception raised inside a slot. Each failure
        gets logged to stderr + the status bar; other tabs still update.
        """
        import traceback
        import logging

        log = logging.getLogger("bvidfe.gui")

        self._last_result = result

        tab_updates = [
            ("Summary", lambda: self.summary_tab.update(result)),
            (
                "Damage Map",
                lambda: self.damage_map_tab.update(result, panel=self.panel_panel_as_geometry()),
            ),
        ]
        if self._last_config is not None:
            tab_updates.extend(
                [
                    ("Damage View", lambda: self.mesh_tab.update(self._last_config, result)),
                    (
                        "Damage Severity",
                        lambda: self.stress_tab.update(self._last_config, result),
                    ),
                ]
            )
        tab_updates.append(("Buckling", lambda: self.buckling_tab.update(result)))

        failed: list[str] = []
        for name, fn in tab_updates:
            try:
                fn()
            except Exception:  # noqa: BLE001
                failed.append(name)
                log.exception("Tab update failed: %s", name)
                traceback.print_exc()

        msg = f"Analysis complete: knockdown = {result.knockdown:.3f}"
        if failed:
            msg += f" — tab update errors: {', '.join(failed)} (see terminal)"
        self.statusBar().showMessage(msg, 10000)

        # Auto-populate the Knockdown Curve tab with a quick empirical sweep
        # around the current impact energy. Runs in the background at empirical
        # tier (sub-second) so the user always has a knockdown-vs-energy plot
        # for context, without having to click Run energy sweep explicitly.
        try:
            self._auto_populate_knockdown_curve()
        except Exception:  # noqa: BLE001
            log.exception("Failed to start auto knockdown curve sweep")
            traceback.print_exc()

    def _auto_populate_knockdown_curve(self) -> None:
        """Kick off a quick empirical-tier energy sweep in a background thread
        and update the Knockdown Curve tab when it finishes.

        Skipped if the user ran a damage-driven analysis (no impact energy to
        sweep around) or if a sweep is already running.
        """
        if self._last_config is None or self._last_config.impact is None:
            return
        # The previous SweepWorker may have been deleteLater()'d by Qt. In that
        # case the Python wrapper raises `RuntimeError: wrapped C/C++ object
        # of type SweepWorker has been deleted` when we call methods on it.
        # We treat that as "no worker running, safe to start a new one" and
        # clear the dangling reference.
        if self._sweep_worker is not None:
            try:
                if self._sweep_worker.isRunning():
                    return
            except RuntimeError:
                self._sweep_worker = None
        from dataclasses import replace

        import numpy as np

        from bvidfe.gui.workers import SweepWorker

        # Build a pure-empirical config so the sweep runs fast regardless of
        # the user's currently-selected tier.
        base = self._last_config
        empirical_cfg = replace(base, tier="empirical", mesh=None)
        # Sweep 8 energies between 2 J and 1.5 * current energy, so the
        # current single-run point sits roughly in the middle of the curve.
        e_cur = base.impact.energy_J
        e_max = max(5.0, 1.5 * e_cur)
        energies = list(np.linspace(2.0, e_max, 8))

        worker = SweepWorker(empirical_cfg, energies_J=energies, csv_path=None)
        worker.resultReady.connect(self._on_auto_sweep_ready)
        worker.error.connect(lambda tb: None)  # swallow errors silently for auto-sweep
        # On finish: deleteLater() the Qt object AND clear the main-window's
        # Python reference so the next _auto_populate call sees a None worker
        # instead of a deleted one.
        worker.finished.connect(lambda w=worker: self._on_sweep_worker_finished(w))
        self._sweep_worker = worker
        worker.start()

    def _on_sweep_worker_finished(self, worker) -> None:
        """Cleanup when a SweepWorker finishes: deleteLater the Qt object and
        null out our Python reference so _auto_populate_knockdown_curve won't
        poke a dangling pointer next time."""
        try:
            worker.deleteLater()
        except RuntimeError:
            pass
        if self._sweep_worker is worker:
            self._sweep_worker = None

    def _on_auto_sweep_ready(self, df) -> None:
        """Populate the Knockdown Curve tab with the auto-sweep results."""
        energies = df["energy_J"].tolist() if "energy_J" in df.columns else list(range(len(df)))
        knockdowns = df["knockdown"].tolist()
        self.knockdown_tab.update_series(
            energies,
            knockdowns,
            tier_label="empirical (auto)",
        )

    def _on_sweep_ready(self, df) -> None:
        energies = df["energy_J"].tolist() if "energy_J" in df.columns else list(range(len(df)))
        knockdowns = df["knockdown"].tolist()
        self.knockdown_tab.update_series(
            energies,
            knockdowns,
            tier_label=self.analysis_panel.get_tier(),
        )
        self.statusBar().showMessage(f"Sweep complete: {len(df)} points", 10000)

    def _on_worker_error(self, tb: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.critical(self, "Analysis error", tb)
        self.statusBar().showMessage("Analysis failed", 5000)

    def _on_progress(self, percent: int) -> None:
        self.statusBar().showMessage(f"Running\u2026 {percent}%")

    def panel_panel_as_geometry(self):
        from bvidfe.core.geometry import PanelGeometry

        return PanelGeometry(
            Lx_mm=self.panel_panel.get_Lx_mm(),
            Ly_mm=self.panel_panel.get_Ly_mm(),
            boundary=self.panel_panel.get_boundary(),
        )

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------

    def _build_file_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")

        save_cfg = QAction("Save Config\u2026", self)
        save_cfg.triggered.connect(self._save_config)
        menu.addAction(save_cfg)

        load_cfg = QAction("Load Config\u2026", self)
        load_cfg.triggered.connect(self._load_config)
        menu.addAction(load_cfg)

        menu.addSeparator()

        export_json = QAction("Export Results JSON\u2026", self)
        export_json.triggered.connect(self._export_results_json)
        menu.addAction(export_json)

        export_png = QAction("Export Damage Map PNG\u2026", self)
        export_png.triggered.connect(self._export_damage_png)
        menu.addAction(export_png)

        menu.addSeparator()

        compare_tiers = QAction("Compare Tiers (empirical + semi_analytical)\u2026", self)
        compare_tiers.triggered.connect(self._compare_tiers)
        menu.addAction(compare_tiers)

    def _compare_tiers(self) -> None:
        """Run empirical + semi_analytical sweeps on the current config in a
        background ``TierComparisonWorker`` and overlay them on the Knockdown
        Curve tab when the worker finishes.

        Both tiers are fast (sub-second for empirical; ~1 second for
        semi_analytical) but on the GUI thread the combined ~12 s sweep
        froze the UI completely. fe3d is skipped because a sweep at fe3d
        can take many minutes; users who want fe3d in the comparison can
        trigger a dedicated sweep via the Sweep panel.
        """
        if self._tier_compare_worker is not None:
            self.statusBar().showMessage("Tier comparison already running\u2026", 3000)
            return
        try:
            cfg = self._build_config()
        except (ValueError, AssertionError) as exc:
            self.statusBar().showMessage(f"Invalid config: {exc}", 10000)
            return
        if cfg.impact is None:
            self.statusBar().showMessage("Tier comparison requires an impact-driven config.", 5000)
            return

        import numpy as np

        from bvidfe.gui.workers import TierComparisonWorker

        e_cur = cfg.impact.energy_J
        energies = list(np.linspace(2.0, max(5.0, 1.5 * e_cur), 8))
        tiers = ("empirical", "semi_analytical")

        self.statusBar().showMessage("Running tier comparison\u2026")
        worker = TierComparisonWorker(cfg, tiers, energies)
        worker.resultReady.connect(self._on_tier_compare_ready)
        worker.error.connect(self._on_worker_error)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda w=worker: self._on_tier_compare_finished(w))
        self._tier_compare_worker = worker
        worker.start()

    def _on_tier_compare_ready(self, payload) -> None:
        """Push the worker result into the Knockdown Curve tab and report
        any per-(tier, energy) failures via the status bar."""
        energies, kd_by_tier, failed_pairs = payload
        self.knockdown_tab.update_tier_comparison(energies, kd_by_tier)
        if failed_pairs:
            self.statusBar().showMessage(
                f"Tier comparison complete: {len(energies)} energies x "
                f"{len(kd_by_tier)} tiers ({len(failed_pairs)} pair(s) skipped \u2014 see log)",
                10000,
            )
        else:
            self.statusBar().showMessage(
                f"Tier comparison complete: {len(energies)} energies x " f"{len(kd_by_tier)} tiers",
                8000,
            )

    def _on_tier_compare_finished(self, worker) -> None:
        """Cleanup when a TierComparisonWorker finishes; mirrors the
        AnalysisWorker / SweepWorker variants."""
        try:
            worker.deleteLater()
        except RuntimeError:
            pass
        if self._tier_compare_worker is worker:
            self._tier_compare_worker = None

    def _save_config(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save AnalysisConfig", "bvidfe_config.json", "JSON (*.json)"
        )
        if not path_str:
            return
        try:
            cfg = self._build_config()
        except (ValueError, AssertionError) as exc:
            QMessageBox.warning(self, "Invalid config", str(exc))
            return
        Path(path_str).write_text(json.dumps(config_to_dict(cfg), indent=2))
        self.statusBar().showMessage(f"Saved config to {path_str}", 5000)

    def _load_config(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load AnalysisConfig", "", "JSON (*.json)")
        if not path_str:
            return
        # Read + parse + apply in three explicit steps so each failure mode
        # surfaces a specific message instead of the bare "list index out of
        # range" string the previous catch-all displayed.
        try:
            text = Path(path_str).read_text()
        except OSError as exc:
            QMessageBox.warning(self, "Cannot read config", f"{path_str}: {exc}")
            return
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            QMessageBox.warning(
                self,
                "Malformed JSON",
                f"{path_str} is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno}).",
            )
            return
        try:
            cfg = config_from_dict(d)
        except KeyError as exc:
            QMessageBox.warning(
                self,
                "Missing field",
                f"Config is missing required field: {exc}.",
            )
            return
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "Invalid value",
                f"Config has an invalid value: {exc}.",
            )
            return
        self._apply_config_to_panels(cfg)
        self.statusBar().showMessage(f"Loaded config from {path_str}", 5000)

    def _apply_config_to_panels(self, cfg) -> None:
        """Push config values back into the input panels."""
        # Material panel
        idx = self.material_panel.material_combo.findText(cfg.material)
        if idx >= 0:
            self.material_panel.material_combo.setCurrentIndex(idx)
        self.material_panel.layup_edit.setText(", ".join(f"{a:g}" for a in cfg.layup_deg))
        self.material_panel.thickness_spin.setValue(cfg.ply_thickness_mm)
        # Panel panel
        self.panel_panel.lx_spin.setValue(cfg.panel.Lx_mm)
        self.panel_panel.ly_spin.setValue(cfg.panel.Ly_mm)
        bi = self.panel_panel.boundary_combo.findText(cfg.panel.boundary)
        if bi >= 0:
            self.panel_panel.boundary_combo.setCurrentIndex(bi)
        # Input mode + impact/damage
        if cfg.impact is not None:
            self.input_mode_panel.impact_radio.setChecked(True)
            self.impact_panel.energy_spin.setValue(cfg.impact.energy_J)
            self.impact_panel.diameter_spin.setValue(cfg.impact.impactor.diameter_mm)
            si = self.impact_panel.shape_combo.findText(cfg.impact.impactor.shape)
            if si >= 0:
                self.impact_panel.shape_combo.setCurrentIndex(si)
            self.impact_panel.mass_spin.setValue(cfg.impact.mass_kg)
            self.impact_panel.location_x.setValue(cfg.impact.location_xy_mm[0])
            self.impact_panel.location_y.setValue(cfg.impact.location_xy_mm[1])
        elif cfg.damage is not None:
            self.input_mode_panel.damage_radio.setChecked(True)
            self.damage_panel.table.setRowCount(0)
            for d in cfg.damage.delaminations:
                self.damage_panel.add_delamination_row(
                    d.interface_index,
                    d.centroid_mm[0],
                    d.centroid_mm[1],
                    d.major_mm,
                    d.minor_mm,
                    d.orientation_deg,
                )
            self.damage_panel.dent_spin.setValue(cfg.damage.dent_depth_mm)
            self.damage_panel.fb_spin.setValue(cfg.damage.fiber_break_radius_mm)
        # Analysis panel
        ti = self.analysis_panel.tier_combo.findText(cfg.tier)
        if ti >= 0:
            self.analysis_panel.tier_combo.setCurrentIndex(ti)
        li = self.analysis_panel.loading_combo.findText(cfg.loading)
        if li >= 0:
            self.analysis_panel.loading_combo.setCurrentIndex(li)

    def _export_results_json(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, "No results", "Run an analysis first.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export results JSON", "bvidfe_results.json", "JSON (*.json)"
        )
        if not path_str:
            return
        Path(path_str).write_text(json.dumps(self._last_result.to_dict(), indent=2, default=str))
        self.statusBar().showMessage(f"Exported results to {path_str}", 5000)

    def _export_damage_png(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, "No results", "Run an analysis first.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export damage map", "damage_map.png", "PNG (*.png)"
        )
        if not path_str:
            return
        self.damage_map_tab.canvas.figure.savefig(path_str, dpi=150)
        self.statusBar().showMessage(f"Saved PNG to {path_str}", 5000)

    # ------------------------------------------------------------------
    # Help menu
    # ------------------------------------------------------------------

    def _build_help_menu(self) -> None:
        menu = self.menuBar().addMenu("&Help")
        about = QAction("About BVID-FE\u2026", self)
        about.triggered.connect(self._show_about_dialog)
        menu.addAction(about)
        how_to = QAction("How to interpret the tiers\u2026", self)
        how_to.triggered.connect(self._show_tier_help)
        menu.addAction(how_to)

    def _show_about_dialog(self) -> None:
        import bvidfe

        QMessageBox.about(
            self,
            "About BVID-FE",
            f"<h3>BVID-FE {bvidfe.__version__}</h3>"
            "<p>Barely Visible Impact Damage residual-strength analysis for "
            "fiber-reinforced composite laminates.</p>"
            "<p>MIT License. Third in a family of defect-specific composite "
            "tools (alongside PorosityFE and WrinkleFE).</p>"
            "<p>Three modeling tiers: empirical (Soutis / Whitney-Nuismer), "
            "semi-analytical (Rayleigh-Ritz sublaminate buckling), and 3D FE "
            "(first-ply-failure on damaged hex mesh). Two workflow paths: "
            "impact-driven (Olsson threshold + peanut-template DPA) and "
            "inspection-driven (C-scan JSON import).</p>"
            "<p><a href='https://github.com/elhajjar1/BVID-FE'>"
            "github.com/elhajjar1/BVID-FE</a></p>",
        )

    def _show_tier_help(self) -> None:
        QMessageBox.information(
            self,
            "BVID-FE modeling tiers",
            "<h4>Which tier should I use?</h4>"
            "<table>"
            "<tr><th>Tier</th><th>Runtime</th><th>Good for</th></tr>"
            "<tr><td><b>empirical</b></td><td>&lt; 1 s</td>"
            "<td>Design allowables, energy-knockdown curves, quick screening. "
            "Soutis formula scales with DPA.</td></tr>"
            "<tr><td><b>semi_analytical</b></td><td>~ 1 s</td>"
            "<td>More conservative than empirical when large sublaminate "
            "buckling governs. Scales with the largest ellipse. Good for "
            "post-buckling CAI estimates.</td></tr>"
            "<tr><td><b>fe3d</b></td><td>~ 10 s to several minutes</td>"
            "<td>Stress-field context, damage-through-thickness visualisation, "
            "first-ply-failure upper bound. <b>Not recommended for "
            "energy-knockdown sweeps</b> — knockdown is approximately flat "
            "vs. energy on the current simplified model. For energy-dependent "
            "studies use empirical or semi_analytical.</td></tr>"
            "</table>"
            "<p>Quick start: run <b>empirical</b> first for the "
            "knockdown-vs-energy shape, then spot-check interesting energies "
            "with <b>semi_analytical</b> to see the buckling contribution.</p>",
        )

"""QThread workers that run BvidAnalysis off the UI thread."""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from bvidfe.analysis import AnalysisConfig, BvidAnalysis
from bvidfe.analysis.fe_tier import _resolve_log_level

# Logger used by the GUI workers + main window. Mirrors bvidfe.fe3d:
# streams to stderr so launching the app from a terminal shows worker
# heartbeats and per-stage timings alongside the fe3d pipeline log.
_log = logging.getLogger("bvidfe.gui")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[bvidfe.gui %(asctime)s] %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(_resolve_log_level())


def _run_with_heartbeat(
    work: Callable[[], Any],
    on_progress: Callable[[int], None],
    *,
    start_pct: int,
    end_pct: int,
    interval_s: float = 2.0,
    label: str = "worker",
) -> Any:
    """Run ``work()`` in a daemon thread, emit heartbeat progress, and
    return its result (re-raising any exception it produced).

    This factors the duplicated heartbeat-progress loop shared by
    ``AnalysisWorker.run`` and the per-iteration body of
    ``SweepWorker.run``: spawn a daemon thread, tick a progress callback
    from ``start_pct`` toward ``end_pct`` at ``interval_s`` intervals
    while the worker is alive, then surface its result or its
    exception. Each tick advances by ``max(1, (end_pct - start_pct) //
    5)`` so the percentage shape across the whole interval is the same
    coarse 5-step ramp the previous code used directly.
    """
    result_box: list[Any] = [None]
    error_box: list[str] = []
    t_start = time.time()

    def _do_work() -> None:
        try:
            result_box[0] = work()
        except Exception:  # noqa: BLE001
            error_box.append(traceback.format_exc())

    t = threading.Thread(target=_do_work, daemon=True)
    t.start()
    pct = start_pct
    on_progress(pct)
    step = max(1, (end_pct - start_pct) // 5)
    while t.is_alive():
        t.join(timeout=interval_s)
        if t.is_alive() and pct < end_pct:
            pct = min(end_pct, pct + step)
            on_progress(pct)
            _log.info("%s heartbeat: %d%% (%.1fs)", label, pct, time.time() - t_start)

    if error_box:
        # Re-raise as a RuntimeError carrying the original traceback string;
        # callers turn that into a Qt error signal.
        raise RuntimeError(error_box[0])
    return result_box[0]


class AnalysisWorker(QThread):
    """Runs `BvidAnalysis(config).run()` in a background thread.

    BvidAnalysis is synchronous and can take tens of seconds on the fe3d
    tier. To keep the status bar from appearing frozen, we run the
    analysis in a daemon worker thread and emit heartbeat progress from
    the QThread itself every few seconds until the work thread finishes.
    """

    resultReady = pyqtSignal(object)  # AnalysisResults
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    HEARTBEAT_INTERVAL_S: float = 2.0

    def __init__(self, config: AnalysisConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config

    def run(self) -> None:  # type: ignore[override]
        t_start = time.time()
        _log.info(
            "AnalysisWorker started: tier=%s loading=%s", self.config.tier, self.config.loading
        )
        try:
            result = _run_with_heartbeat(
                work=lambda: BvidAnalysis(self.config).run(),
                on_progress=self.progress.emit,
                start_pct=10,
                end_pct=90,
                interval_s=self.HEARTBEAT_INTERVAL_S,
                label="AnalysisWorker",
            )
        except RuntimeError as exc:
            tb = str(exc)
            _log.warning(
                "AnalysisWorker error after %.1fs: %s",
                time.time() - t_start,
                tb.splitlines()[-1],
            )
            self.error.emit(tb)
            return
        _log.info("AnalysisWorker done (%.1fs)", time.time() - t_start)
        self.progress.emit(100)
        self.resultReady.emit(result)


class SweepWorker(QThread):
    """Runs a parametric energy sweep in a background thread."""

    resultReady = pyqtSignal(object)  # pandas.DataFrame
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        base_config: AnalysisConfig,
        energies_J: Sequence[float],
        csv_path: Optional[str | Path] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.base_config = base_config
        self.energies_J = list(energies_J)
        self.csv_path = csv_path

    def run(self) -> None:  # type: ignore[override]
        """Run one BvidAnalysis per energy and report progress after each point.

        This in-lines the sweep_energies logic so we can emit per-energy
        progress signals instead of only 10% at the start and 100% at the end.
        The functional behavior is identical to sweep_energies (same dataframe
        columns and CSV format).
        """
        try:
            n = len(self.energies_J)
            rows: list[dict] = []
            if self.base_config.impact is None:
                raise ValueError("sweep requires base_config.impact to be set")
            self.progress.emit(5)
            for i, E in enumerate(self.energies_J):
                new_impact = replace(self.base_config.impact, energy_J=float(E))
                cfg = replace(self.base_config, impact=new_impact)
                base_pct = 5 + int(90 * i / n)
                next_pct = 5 + int(90 * (i + 1) / n)
                # _run_with_heartbeat raises RuntimeError(traceback) on
                # failure; the outer except below converts that into the
                # error signal. Per-energy heartbeats tick from base_pct
                # toward next_pct so the bar moves visibly during a single
                # long fe3d analysis.
                result = _run_with_heartbeat(
                    work=lambda cfg=cfg: BvidAnalysis(cfg).run(),
                    on_progress=self.progress.emit,
                    start_pct=base_pct,
                    end_pct=next_pct,
                    interval_s=2.0,
                    label=f"SweepWorker[E={float(E):g}J]",
                )
                rows.append(
                    {
                        "energy_J": float(E),
                        "knockdown": result.knockdown,
                        "residual_MPa": result.residual_strength_MPa,
                        "pristine_MPa": result.pristine_strength_MPa,
                        "dpa_mm2": result.dpa_mm2,
                        "dent_mm": result.damage.dent_depth_mm,
                        "n_delaminations": len(result.damage.delaminations),
                        "tier_used": result.tier_used,
                    }
                )
                self.progress.emit(next_pct)
            df = pd.DataFrame(rows)
            if self.csv_path is not None:
                df.to_csv(Path(self.csv_path), index=False)
            self.progress.emit(100)
            self.resultReady.emit(df)
        except Exception:
            self.error.emit(traceback.format_exc())


class TierComparisonWorker(QThread):
    """Runs an N-tier x M-energy knockdown sweep in a background thread.

    Replaces the old synchronous loop in ``BvidMainWindow._compare_tiers``
    that ran 16 ``BvidAnalysis`` calls (2 tiers x 8 energies, ~12 s wall
    clock at default settings) on the GUI thread. Results are emitted as a
    ``(energies, kd_by_tier)`` tuple matching the existing
    ``KnockdownTab.update_tier_comparison`` signature; per-(tier, energy)
    failures are absorbed into NaN entries (no abort) and described in a
    third tuple element so the caller can surface them in the status bar.
    """

    # (energies: list[float], kd_by_tier: dict[str, list[float]],
    #  failed_pairs: list[tuple[str, float, str]])
    resultReady = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        base_config: AnalysisConfig,
        tiers: Sequence[str],
        energies_J: Sequence[float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.base_config = base_config
        self.tiers = list(tiers)
        self.energies_J = list(energies_J)

    def run(self) -> None:  # type: ignore[override]
        try:
            if self.base_config.impact is None:
                raise ValueError("tier comparison requires base_config.impact to be set")
            t_start = time.time()
            _log.info(
                "TierComparisonWorker started: tiers=%s n_energies=%d",
                self.tiers,
                len(self.energies_J),
            )

            total_pairs = max(1, len(self.tiers) * len(self.energies_J))
            kd_by_tier: dict[str, list[float]] = {t: [] for t in self.tiers}
            failed_pairs: list[tuple[str, float, str]] = []
            self.progress.emit(2)

            done = 0
            for tier in self.tiers:
                for E in self.energies_J:
                    new_impact = replace(self.base_config.impact, energy_J=float(E))
                    cfg = replace(self.base_config, impact=new_impact, tier=tier, mesh=None)
                    try:
                        result = BvidAnalysis(cfg).run()
                        kd_by_tier[tier].append(float(result.knockdown))
                    except Exception:  # noqa: BLE001
                        kd_by_tier[tier].append(float("nan"))
                        msg = traceback.format_exc().splitlines()[-1]
                        failed_pairs.append((tier, float(E), msg))
                        _log.warning(
                            "TierComparisonWorker: tier=%s energy=%.2fJ failed: %s",
                            tier,
                            E,
                            msg,
                        )
                    done += 1
                    self.progress.emit(2 + int(96 * done / total_pairs))

            _log.info(
                "TierComparisonWorker done: %d pair(s) (%d failed) in %.1fs",
                total_pairs,
                len(failed_pairs),
                time.time() - t_start,
            )
            self.progress.emit(100)
            self.resultReady.emit((list(self.energies_J), kd_by_tier, failed_pairs))
        except Exception:
            self.error.emit(traceback.format_exc())

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from bvidfe.analysis import AnalysisConfig
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.gui.workers import AnalysisWorker, SweepWorker, TierComparisonWorker
from bvidfe.impact.mapping import ImpactEvent


@pytest.fixture
def cfg():
    return AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="compression",
        tier="empirical",
        impact=ImpactEvent(30.0, ImpactorGeometry(), mass_kg=5.5),
    )


def test_analysis_worker_emits_result_ready(qtbot, cfg):
    worker = AnalysisWorker(cfg)
    with qtbot.waitSignal(worker.resultReady, timeout=10_000) as blocker:
        worker.start()
    (result,) = blocker.args
    assert 0 < result.knockdown <= 1.0


def test_analysis_worker_emits_error_on_bad_config(qtbot):
    # Build a config that will raise at run time (tier not recognized)
    bad_cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(100, 50),
        loading="compression",
        tier="empirical",
        impact=ImpactEvent(10.0, ImpactorGeometry(), mass_kg=5.5),
    )
    # Corrupt it to force a failure inside run()
    bad_cfg.tier = "bogus"  # type: ignore[assignment]

    worker = AnalysisWorker(bad_cfg)
    with qtbot.waitSignal(worker.error, timeout=10_000):
        worker.start()


def test_sweep_worker_emits_result_ready(qtbot, cfg):
    worker = SweepWorker(cfg, energies_J=[5, 10], csv_path=None)
    with qtbot.waitSignal(worker.resultReady, timeout=20_000) as blocker:
        worker.start()
    (df,) = blocker.args
    assert len(df) == 2
    assert "knockdown" in df.columns


def test_tier_comparison_worker_emits_result_ready(qtbot, cfg):
    """The new TierComparisonWorker replaces the synchronous loop in
    BvidMainWindow._compare_tiers — it must emit (energies, kd_by_tier,
    failed_pairs) for every requested (tier, energy) pair."""
    tiers = ("empirical", "semi_analytical")
    energies = [5.0, 10.0, 15.0]
    worker = TierComparisonWorker(cfg, tiers, energies)
    with qtbot.waitSignal(worker.resultReady, timeout=30_000) as blocker:
        worker.start()
    (payload,) = blocker.args
    out_energies, kd_by_tier, failed_pairs = payload
    assert out_energies == energies
    assert set(kd_by_tier.keys()) == set(tiers)
    for tier in tiers:
        assert len(kd_by_tier[tier]) == len(energies)
        assert all(0.0 < kd <= 1.0 for kd in kd_by_tier[tier])
    assert failed_pairs == []


def test_tier_comparison_worker_absorbs_per_pair_failure(qtbot, cfg):
    """If one tier raises mid-sweep, the worker keeps going with NaN entries
    and reports the failed pair instead of aborting the whole comparison."""
    import math

    worker = TierComparisonWorker(
        cfg,
        tiers=("empirical", "bogus_tier"),  # second tier triggers an exception
        energies_J=[5.0, 10.0],
    )
    with qtbot.waitSignal(worker.resultReady, timeout=20_000) as blocker:
        worker.start()
    (payload,) = blocker.args
    _energies, kd_by_tier, failed_pairs = payload
    # Empirical tier still produced valid knockdowns
    assert all(0.0 < kd <= 1.0 for kd in kd_by_tier["empirical"])
    # Bogus tier produced NaNs and was reported via failed_pairs
    assert all(math.isnan(kd) for kd in kd_by_tier["bogus_tier"])
    assert len(failed_pairs) == 2
    assert all(t == "bogus_tier" for t, _e, _msg in failed_pairs)

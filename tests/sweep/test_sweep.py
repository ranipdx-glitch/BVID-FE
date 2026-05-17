import warnings

import pandas as pd

from bvidfe.analysis import AnalysisConfig
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.impact.mapping import DPACapClipWarning, ImpactEvent
from bvidfe.sweep.parametric_sweep import (
    sweep_energies,
    sweep_layups,
    sweep_thicknesses,
)


def _base_impact_cfg(**overrides):
    kw = dict(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90] * 4,
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150, 100),
        loading="compression",
        tier="empirical",
        impact=ImpactEvent(10.0, ImpactorGeometry(), mass_kg=5.5),
    )
    kw.update(overrides)
    return AnalysisConfig(**kw)


def test_sweep_energies_returns_dataframe_with_expected_columns():
    cfg = _base_impact_cfg()
    df = sweep_energies(cfg, energies_J=[5, 10, 20, 30])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    for col in ["energy_J", "knockdown", "residual_MPa", "dpa_mm2", "dent_mm"]:
        assert col in df.columns


def test_sweep_energies_writes_csv(tmp_path):
    cfg = _base_impact_cfg()
    csv_path = tmp_path / "sweep.csv"
    sweep_energies(cfg, energies_J=[5, 10], csv_path=csv_path)
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == 2


def test_sweep_layups():
    cfg = _base_impact_cfg()
    layups = [
        [0, 45, -45, 90] * 4,
        [0, 90] * 8,
        [0, 60, -60] * 5 + [0],
    ]
    df = sweep_layups(cfg, layups=layups)
    assert len(df) == 3
    assert "layup" in df.columns
    assert "knockdown" in df.columns


def test_sweep_thicknesses():
    cfg = _base_impact_cfg()
    df = sweep_thicknesses(cfg, ply_thicknesses_mm=[0.125, 0.152, 0.2])
    assert len(df) == 3
    assert "ply_thickness_mm" in df.columns


def test_sweep_energies_default_raises_on_iteration_failure(monkeypatch):
    """Issue #8 baseline: with on_error='raise' (default) a failed iteration
    aborts the sweep — preserving the legacy behaviour for callers that want
    a hard fail."""
    cfg = _base_impact_cfg()
    from bvidfe.sweep import parametric_sweep as ps

    real_run_one = ps._run_one

    def _patched(cfg):
        if cfg.impact.energy_J == 10.0:
            raise RuntimeError("synthetic failure at 10J")
        return real_run_one(cfg)

    monkeypatch.setattr(ps, "_run_one", _patched)
    import pytest

    with pytest.raises(RuntimeError, match="synthetic failure"):
        sweep_energies(cfg, energies_J=[5, 10, 20])


def test_sweep_energies_skip_preserves_partial_results(monkeypatch, tmp_path):
    """Issue #8: with on_error='skip' a per-iteration failure must not
    abort the whole sweep — the failed energy is recorded with NaN numerics
    and an 'error' column, the surrounding successful iterations land in
    the DataFrame, and the CSV is still written."""
    import math

    cfg = _base_impact_cfg()
    from bvidfe.sweep import parametric_sweep as ps

    real_run_one = ps._run_one

    def _patched(cfg):
        if cfg.impact.energy_J == 10.0:
            raise RuntimeError("synthetic failure at 10J")
        return real_run_one(cfg)

    monkeypatch.setattr(ps, "_run_one", _patched)
    csv_path = tmp_path / "partial.csv"
    df = sweep_energies(cfg, energies_J=[5, 10, 20], csv_path=csv_path, on_error="skip")
    assert len(df) == 3
    bad = df.loc[df["energy_J"] == 10.0].iloc[0]
    assert math.isnan(bad["knockdown"])
    assert "error" in df.columns
    assert "synthetic failure" in str(bad["error"])
    # Surrounding iterations still produced valid knockdowns
    good = df.loc[df["energy_J"] != 10.0]
    assert all(0.0 < kd <= 1.0 for kd in good["knockdown"])
    # Partial CSV is preserved
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == 3


def test_sweep_energies_warn_emits_userwarning(monkeypatch):
    """Issue #8: on_error='warn' is identical to 'skip' but adds a
    UserWarning per failure so callers running interactively see the cause."""
    import warnings as _warnings

    cfg = _base_impact_cfg()
    from bvidfe.sweep import parametric_sweep as ps

    real_run_one = ps._run_one

    def _patched(cfg):
        if cfg.impact.energy_J == 10.0:
            raise RuntimeError("synthetic warn-mode failure")
        return real_run_one(cfg)

    monkeypatch.setattr(ps, "_run_one", _patched)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        df = sweep_energies(cfg, energies_J=[5, 10, 20], on_error="warn")
    assert any("synthetic warn-mode failure" in str(w.message) for w in caught)
    assert len(df) == 3


def test_sweep_energies_progress_callback_is_invoked():
    """Issue #8: progress_callback receives (i_done, n_total) after each
    iteration, allowing the GUI worker / a long-running script to drive a
    progress bar without re-implementing the loop."""
    cfg = _base_impact_cfg()
    seen: list[tuple[int, int]] = []
    sweep_energies(
        cfg,
        energies_J=[5, 10, 20],
        progress_callback=lambda i, n: seen.append((i, n)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_sweep_energies_rejects_unknown_on_error():
    cfg = _base_impact_cfg()
    import pytest

    with pytest.raises(ValueError, match="on_error must be one of"):
        sweep_energies(cfg, energies_J=[5], on_error="bogus")


def test_sweep_column_schema_is_deterministic():
    """Issue #38: column order is fixed (swept var first, `error` always
    present) regardless of whether any iteration failed."""
    expected_results = [
        "knockdown",
        "residual_MPa",
        "pristine_MPa",
        "dpa_mm2",
        "dent_mm",
        "n_delaminations",
        "tier_used",
        "error",
    ]
    cfg = _base_impact_cfg()

    # All-success run with on_error='raise': `error` column still present.
    df_e = sweep_energies(cfg, energies_J=[5, 10, 20])
    assert list(df_e.columns) == ["energy_J", *expected_results]
    assert df_e["error"].isna().all()

    df_l = sweep_layups(cfg, layups=[[0, 90] * 8, [0, 45, -45, 90] * 4])
    assert list(df_l.columns) == ["layup", *expected_results]

    df_t = sweep_thicknesses(cfg, ply_thicknesses_mm=[0.125, 0.152])
    assert list(df_t.columns) == ["ply_thickness_mm", *expected_results]


def test_sweep_dedupes_dpa_cap_warning_to_once_per_sweep():
    """Issue #55: the DPA-cap diagnostic must fire once for the whole sweep,
    not once per point. A small panel + high energies makes every iteration
    exceed the 80% DPA clip; each warning's message embeds the per-point DPA
    value, so Python's 'once' filter (keyed on message text) cannot collapse
    them — _dedupe_impact_warnings() collapses by category instead."""
    cfg = _base_impact_cfg(panel=PanelGeometry(40, 30))
    energies = [80.0, 100.0, 120.0, 140.0, 160.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = sweep_energies(cfg, energies_J=energies)
    assert len(df) == len(energies)
    dpa_warnings = [w for w in caught if issubclass(w.category, DPACapClipWarning)]
    # Without per-category dedup this equals len(energies) (one per point).
    assert len(dpa_warnings) == 1

"""Tests for BVID-FE CLI entry point."""

import json
import subprocess
import sys


def _run_cli(*args):
    # Invoke the CLI via ``python -m bvidfe.cli`` so the same command works
    # whether the package is installed in .venv/ (local dev) or on the system
    # PATH (CI runners). Hardcoding "./.venv/bin/bvidfe" breaks in CI.
    return subprocess.run(
        [sys.executable, "-m", "bvidfe.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_runs_empirical_impact():
    res = _run_cli(
        "--tier",
        "empirical",
        "--material",
        "IM7/8552",
        "--layup",
        "0,45,-45,90,0,45,-45,90",
        "--thickness",
        "0.152",
        "--panel",
        "150x100",
        "--loading",
        "compression",
        "--energy",
        "25",
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert "knockdown" in data
    assert 0 < data["knockdown"] <= 1.0


def test_cli_runs_tension():
    res = _run_cli(
        "--tier",
        "empirical",
        "--material",
        "IM7/8552",
        "--layup",
        "0,45,-45,90,0,45,-45,90",
        "--thickness",
        "0.152",
        "--panel",
        "150x100",
        "--loading",
        "tension",
        "--energy",
        "25",
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["tier_used"] == "empirical"


def test_cli_help_works():
    res = _run_cli("--help")
    assert res.returncode == 0
    assert "BVID" in res.stdout or "bvid" in res.stdout.lower()


def test_cli_rejects_bad_panel_format():
    res = _run_cli(
        "--tier",
        "empirical",
        "--material",
        "IM7/8552",
        "--layup",
        "0,90,0,90",
        "--thickness",
        "0.2",
        "--panel",
        "notaxspec",
        "--loading",
        "compression",
        "--energy",
        "10",
    )
    assert res.returncode != 0


def test_cli_rejects_unknown_material():
    """Issue #7: --material typo must produce argparse 'invalid choice' with
    the list of valid presets, not a downstream KeyError."""
    res = _run_cli(
        "--tier", "empirical",
        "--material", "IM7/8553",  # typo (no such preset)
        "--layup", "0,90,0,90",
        "--thickness", "0.2",
        "--panel", "100x100",
        "--loading", "compression",
        "--energy", "10",
    )
    assert res.returncode == 2
    assert "IM7/8552" in res.stderr  # at least one valid preset listed


def test_cli_rejects_zero_energy():
    """Issue #7: --energy must reject non-positive values at parse time."""
    res = _run_cli(
        "--tier", "empirical",
        "--material", "IM7/8552",
        "--layup", "0,90,0,90",
        "--thickness", "0.2",
        "--panel", "100x100",
        "--loading", "compression",
        "--energy", "0",
    )
    assert res.returncode == 2
    assert "must be > 0" in res.stderr


def test_cli_rejects_missing_cscan_path(tmp_path):
    """Issue #7: --cscan with a nonexistent path must fail at parse time
    (argparse 'invalid type'), not deep inside load_cscan_json."""
    missing = tmp_path / "does_not_exist.json"
    res = _run_cli(
        "--tier", "empirical",
        "--material", "IM7/8552",
        "--layup", "0,90,0,90",
        "--thickness", "0.2",
        "--panel", "100x100",
        "--loading", "compression",
        "--cscan", str(missing),
    )
    assert res.returncode == 2
    assert "file not found" in res.stderr.lower()


def test_cli_quick_json_emits_machine_readable_object():
    """Issue #7: --quick-json emits a single-line JSON object with knockdown
    plus tier provenance, in contrast to --quick (bare float)."""
    res = _run_cli(
        "--tier", "empirical",
        "--material", "IM7/8552",
        "--layup", "0,45,-45,90,90,-45,45,0",
        "--thickness", "0.152",
        "--panel", "150x100",
        "--loading", "compression",
        "--energy", "20",
        "--quick-json",
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert set(data.keys()) == {
        "knockdown", "residual_strength_MPa", "pristine_strength_MPa", "tier_used",
    }
    assert 0 < data["knockdown"] <= 1.0
    assert data["tier_used"] == "empirical"

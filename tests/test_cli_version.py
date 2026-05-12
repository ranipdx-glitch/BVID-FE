"""Tests for CLI version + quick flags."""

import subprocess
import sys


def test_cli_version_flag_prints_version():
    res = subprocess.run(
        [sys.executable, "-m", "bvidfe.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert res.stdout.strip().startswith("bvidfe ")
    # Version is consumed from bvidfe.__version__
    import bvidfe

    assert bvidfe.__version__ in res.stdout


def test_cli_list_materials_prints_all_presets():
    """--list-materials prints all 4 material presets and exits cleanly."""
    res = subprocess.run(
        [sys.executable, "-m", "bvidfe.cli", "--list-materials"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    for name in ("AS4/3501-6", "IM7/8552", "T700/2510", "T800/epoxy"):
        assert name in res.stdout


def test_cli_cscan_flag_runs_inspection_driven():
    """--cscan runs the inspection-driven path from CLI (mutually exclusive with --energy)."""
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
            "semi_analytical",
            "--cscan",
            "examples/sample_cscan.json",
            "--quick",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    kd = float(res.stdout.strip())
    assert 0.0 < kd < 1.0


def test_cli_rejects_energy_and_cscan_together():
    """--energy and --cscan are mutually exclusive."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "bvidfe.cli",
            "--material",
            "IM7/8552",
            "--layup",
            "0,90,0,90",
            "--thickness",
            "0.2",
            "--panel",
            "100x50",
            "--loading",
            "compression",
            "--energy",
            "20",
            "--cscan",
            "examples/sample_cscan.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "mutually exclusive" in res.stderr


def test_cli_quick_flag_prints_only_knockdown():
    """--quick prints just the knockdown as a scalar, no JSON."""
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
            "--quick",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    # stdout should be a single float (to 6 decimals), no JSON braces
    stdout = res.stdout.strip()
    assert "{" not in stdout
    assert "}" not in stdout
    # Must parse as a float in (0, 1]
    kd = float(stdout)
    assert 0.0 < kd <= 1.0

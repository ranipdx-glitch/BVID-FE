"""Smoke tests for the validation harness."""

import subprocess
import sys
from pathlib import Path


def test_validator_module_imports():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "validate_bvid_public",
        Path(__file__).parent.parent / "validation" / "validate_bvid_public.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so @dataclass can resolve cls.__module__
    # on Python 3.9 (needed when loading a file outside any package).
    sys.modules["validate_bvid_public"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("validate_bvid_public", None)
    assert hasattr(mod, "DatasetCase")
    assert hasattr(mod, "run_dataset")
    assert hasattr(mod, "main")


def test_validator_runs_synthetic_dataset():
    res = subprocess.run(
        [
            sys.executable,
            "validation/validate_bvid_public.py",
            "--dataset",
            "synthetic_selfcheck",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "synthetic_selfcheck" in res.stdout
    assert "MAE" in res.stdout


def test_validator_gate_passes_on_synthetic_dataset():
    res = subprocess.run(
        [
            sys.executable,
            "validation/validate_bvid_public.py",
            "--dataset",
            "synthetic_selfcheck",
            "--gate",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0, res.stderr

"""Direct unit tests for bvidfe.cli helper functions.

Issue #11: these helpers (_parse_panel, _parse_layup, _positive_float,
_existing_path) were exercised only end-to-end via subprocess in
test_cli.py. Direct tests fail much faster and cover edge cases like
zero / NaN / negative which would otherwise need a full CLI invocation.
"""

import argparse

import pytest

from bvidfe.cli import _existing_path, _parse_layup, _parse_panel, _positive_float


# ---------- _parse_panel ----------


def test_parse_panel_accepts_lx_x_ly():
    p = _parse_panel("150x100")
    assert p.Lx_mm == 150.0
    assert p.Ly_mm == 100.0


def test_parse_panel_is_case_insensitive():
    p = _parse_panel("150X100")
    assert p.Lx_mm == 150.0


@pytest.mark.parametrize(
    "spec",
    ["100", "abcxdef", "100x", "x100", "100x100x50", ""],
)
def test_parse_panel_rejects_malformed(spec):
    with pytest.raises(argparse.ArgumentTypeError, match=r"--panel"):
        _parse_panel(spec)


@pytest.mark.parametrize("spec", ["0x100", "100x0", "-50x100", "100x-50"])
def test_parse_panel_rejects_non_positive_dimensions(spec):
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_panel(spec)


# ---------- _parse_layup ----------


def test_parse_layup_basic():
    assert _parse_layup("0,45,-45,90") == [0.0, 45.0, -45.0, 90.0]


def test_parse_layup_rejects_non_numeric():
    with pytest.raises(argparse.ArgumentTypeError, match=r"--layup"):
        _parse_layup("0,45,abc,90")


def test_parse_layup_rejects_empty_token():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_layup("0,,45")


# ---------- _positive_float ----------


def test_positive_float_accepts_positive():
    assert _positive_float("0.152") == 0.152
    assert _positive_float("1e3") == 1000.0


@pytest.mark.parametrize("bad", ["0", "-1", "0.0", "-1e-6"])
def test_positive_float_rejects_non_positive(bad):
    with pytest.raises(argparse.ArgumentTypeError, match=r"> 0"):
        _positive_float(bad)


def test_positive_float_rejects_non_numeric():
    with pytest.raises(argparse.ArgumentTypeError, match="number"):
        _positive_float("abc")


# ---------- _existing_path ----------


def test_existing_path_accepts_real_file(tmp_path):
    f = tmp_path / "real.json"
    f.write_text("{}")
    assert _existing_path(str(f)).resolve() == f.resolve()


def test_existing_path_rejects_missing(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(argparse.ArgumentTypeError, match="file not found"):
        _existing_path(str(missing))


def test_existing_path_rejects_directory(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError, match="not a file"):
        _existing_path(str(tmp_path))

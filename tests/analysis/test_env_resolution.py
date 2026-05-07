"""Tests for the BVIDFE_LOG_LEVEL / BVIDFE_FE3D_MAX_DOF env-var resolution.

Both env vars are read at module import; an invalid value used to raise
``ValueError`` and prevent ``bvidfe`` from being imported at all. The
resolvers below fall back to the documented defaults with a stderr warning
instead — failure-tolerant for typos like ``BVIDFE_LOG_LEVEL=DUBG``.
"""

import logging

from bvidfe.analysis.fe_tier import _resolve_log_level, _resolve_max_dof


def test_log_level_default_is_info(monkeypatch):
    monkeypatch.delenv("BVIDFE_LOG_LEVEL", raising=False)
    assert _resolve_log_level() == logging.INFO


def test_log_level_resolves_known_names(monkeypatch):
    for name, expected in [
        ("DEBUG", logging.DEBUG),
        ("info", logging.INFO),  # case-insensitive
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ]:
        monkeypatch.setenv("BVIDFE_LOG_LEVEL", name)
        assert _resolve_log_level() == expected


def test_log_level_typo_falls_back_to_default_with_warning(monkeypatch, capsys):
    monkeypatch.setenv("BVIDFE_LOG_LEVEL", "DUBG")
    level = _resolve_log_level()
    assert level == logging.INFO  # fell back to default
    captured = capsys.readouterr()
    assert "BVIDFE_LOG_LEVEL" in captured.err
    assert "DUBG" in captured.err


def test_max_dof_default(monkeypatch):
    monkeypatch.delenv("BVIDFE_FE3D_MAX_DOF", raising=False)
    assert _resolve_max_dof() == 500000


def test_max_dof_overrides(monkeypatch):
    monkeypatch.setenv("BVIDFE_FE3D_MAX_DOF", "1000000")
    assert _resolve_max_dof() == 1000000


def test_max_dof_non_numeric_falls_back_with_warning(monkeypatch, capsys):
    monkeypatch.setenv("BVIDFE_FE3D_MAX_DOF", "lots")
    assert _resolve_max_dof() == 500000
    captured = capsys.readouterr()
    assert "BVIDFE_FE3D_MAX_DOF" in captured.err
    assert "lots" in captured.err


def test_max_dof_zero_or_negative_falls_back_with_warning(monkeypatch, capsys):
    monkeypatch.setenv("BVIDFE_FE3D_MAX_DOF", "0")
    assert _resolve_max_dof() == 500000
    monkeypatch.setenv("BVIDFE_FE3D_MAX_DOF", "-50")
    assert _resolve_max_dof() == 500000
    assert capsys.readouterr().err.count("BVIDFE_FE3D_MAX_DOF") >= 2

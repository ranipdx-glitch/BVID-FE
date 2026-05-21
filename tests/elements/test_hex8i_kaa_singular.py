"""Regression tests for Hex8i behavior when Kaa is singular.

Issue #108: previously the LinAlgError on `np.linalg.inv(Kaa)` was caught and
silently swallowed, returning the un-condensed Kuu. We now emit a warning on
the `bvidfe.elements` logger by default, and re-raise when
`STRICT_HEX8I_CONDENSATION=True`.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.elements import hex8i as hex8i_module
from bvidfe.elements.hex8i import Hex8iElement


def _unit_cube_nodes() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )


@pytest.fixture(autouse=True)
def _reset_warned_elements():
    """Each test starts with a clean dedup set so warnings emit reliably."""
    hex8i_module._warned_elements.clear()
    yield
    hex8i_module._warned_elements.clear()


def _force_singular_kaa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch np.linalg.inv in the hex8i module so only the Kaa (9x9) inversion fails.

    The other np.linalg.inv calls in stiffness_matrix (e.g. J0_inv on a 3x3) must
    still work. We detect Kaa by its (9, 9) shape.
    """
    real_inv = np.linalg.inv

    def fake_inv(a, *args, **kwargs):
        arr = np.asarray(a)
        if arr.shape == (9, 9):
            raise np.linalg.LinAlgError("forced singular Kaa for test")
        return real_inv(a, *args, **kwargs)

    monkeypatch.setattr(hex8i_module.np.linalg, "inv", fake_inv)


def test_singular_kaa_default_mode_warns_and_returns_kuu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Default mode: warning is emitted, stiffness is finite & well-shaped."""
    _force_singular_kaa(monkeypatch)
    # Make sure we're in default (non-strict) mode.
    monkeypatch.setattr(hex8i_module, "STRICT_HEX8I_CONDENSATION", False)

    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8iElement(_unit_cube_nodes(), m, ply_angle_deg=0.0)

    with caplog.at_level(logging.WARNING, logger="bvidfe.elements"):
        K = elem.stiffness_matrix()

    assert K.shape == (24, 24)
    assert np.all(np.isfinite(K))
    # K is Kuu (un-condensed) — still symmetric.
    assert np.allclose(K, K.T, atol=1e-8)

    # Warning must mention this is a Hex8i issue and surface a cond number.
    warning_records = [
        r for r in caplog.records if r.name == "bvidfe.elements" and r.levelno >= logging.WARNING
    ]
    assert warning_records, "expected a warning on bvidfe.elements logger"
    msg = warning_records[0].getMessage()
    assert "Hex8i" in msg
    assert "Kaa" in msg
    assert "cond(Kaa)" in msg


def test_singular_kaa_strict_mode_raises(monkeypatch: pytest.MonkeyPatch):
    """Strict mode: LinAlgError is re-raised with a chained cause."""
    _force_singular_kaa(monkeypatch)
    monkeypatch.setattr(hex8i_module, "STRICT_HEX8I_CONDENSATION", True)

    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8iElement(_unit_cube_nodes(), m, ply_angle_deg=45.0)

    with pytest.raises(np.linalg.LinAlgError) as excinfo:
        elem.stiffness_matrix()

    # Chained from original LinAlgError.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, np.linalg.LinAlgError)
    # Message should reference Hex8i context.
    assert "Hex8i" in str(excinfo.value)
    assert "ply_angle=45.0" in str(excinfo.value)

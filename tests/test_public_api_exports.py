"""Smoke tests for the public API surface of ``bvidfe``.

These tests guard against accidental drift in ``bvidfe/__init__.py`` —
specifically the names re-exported via ``__all__``. The goal is to fail
CI the moment a top-level export is removed or renamed, instead of
waiting for a downstream user to complain.

Two layers of checks:
    1. Every name in ``bvidfe.__all__`` must resolve as an attribute on
       the ``bvidfe`` module (parametrized).
    2. A minimal end-to-end constructor smoke: build an ``AnalysisConfig``
       and instantiate ``BvidAnalysis(cfg)`` without calling ``.run()``,
       to confirm the public construction path is intact.
"""

from __future__ import annotations

import pytest

import bvidfe


def test_all_attribute_exists():
    """``bvidfe`` must declare an ``__all__`` for the public-export contract."""
    assert hasattr(bvidfe, "__all__"), "bvidfe.__init__ must define __all__"
    assert isinstance(bvidfe.__all__, list)
    assert len(bvidfe.__all__) > 0, "bvidfe.__all__ must not be empty"


@pytest.mark.parametrize("name", bvidfe.__all__)
def test_public_export_resolves(name: str) -> None:
    """Each name in ``bvidfe.__all__`` must be importable from the top level."""
    assert hasattr(bvidfe, name), f"bvidfe.__all__ advertises {name!r} but it is missing"
    obj = getattr(bvidfe, name)
    assert obj is not None, f"bvidfe.{name} resolved to None"


def test_bvid_analysis_constructor_smoke() -> None:
    """End-to-end public-API smoke: build a config and construct the runner.

    Does NOT call ``.run()`` — we only want to confirm that the public
    constructors wired through ``bvidfe.__init__`` still accept the
    documented inputs and produce the documented objects.
    """
    from bvidfe import (
        AnalysisConfig,
        BvidAnalysis,
        ImpactEvent,
        ImpactorGeometry,
        PanelGeometry,
    )

    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness_mm=0.152,
        panel=PanelGeometry(150.0, 100.0),
        loading="compression",
        tier="empirical",
        impact=ImpactEvent(20.0, ImpactorGeometry(), mass_kg=5.5),
    )
    analysis = BvidAnalysis(cfg)
    assert analysis.config is cfg

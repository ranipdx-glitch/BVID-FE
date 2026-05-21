"""Validation tests for :class:`bvidfe.analysis.config.AnalysisConfig`.

Covers the ``tier`` and ``loading`` ``Literal`` aliases added in #82: every
valid combination should construct cleanly, while every invalid string must
raise ``ValueError`` whose message names the offending field.
"""

from __future__ import annotations

import itertools

import pytest

from bvidfe._types import _LOADING_MODES, _TIER_NAMES
from bvidfe.analysis.config import AnalysisConfig, MeshParams
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.impact.mapping import ImpactEvent


def _base_kwargs() -> dict:
    """Minimum kwargs to build a valid :class:`AnalysisConfig`."""
    return dict(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(10.0, 5.0),
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
    )


@pytest.mark.parametrize(
    "tier,loading",
    list(itertools.product(sorted(_TIER_NAMES), sorted(_LOADING_MODES))),
)
def test_valid_tier_and_loading_combinations_construct(tier: str, loading: str) -> None:
    kwargs = _base_kwargs()
    kwargs["tier"] = tier
    kwargs["loading"] = loading
    # fe3d requires a mesh, which __post_init__ supplies via default; ensure
    # explicit construction also works.
    if tier == "fe3d":
        kwargs["mesh"] = MeshParams()
    cfg = AnalysisConfig(**kwargs)
    assert cfg.tier == tier
    assert cfg.loading == loading


@pytest.mark.parametrize("bad_tier", ["fe3D", "FE3D", "empirical ", "", "fea", "fe3d_extra"])
def test_invalid_tier_raises_value_error_with_field_name(bad_tier: str) -> None:
    kwargs = _base_kwargs()
    kwargs["tier"] = bad_tier
    with pytest.raises(ValueError) as excinfo:
        AnalysisConfig(**kwargs)
    msg = str(excinfo.value)
    assert "tier" in msg
    assert repr(bad_tier) in msg


@pytest.mark.parametrize("bad_loading", ["bending", "Compression", "TENSION", "", "compression "])
def test_invalid_loading_raises_value_error_with_field_name(bad_loading: str) -> None:
    kwargs = _base_kwargs()
    kwargs["loading"] = bad_loading
    with pytest.raises(ValueError) as excinfo:
        AnalysisConfig(**kwargs)
    msg = str(excinfo.value)
    assert "loading" in msg
    assert repr(bad_loading) in msg


def test_invalid_tier_error_message_lists_allowed_values() -> None:
    kwargs = _base_kwargs()
    kwargs["tier"] = "fe3D"
    with pytest.raises(ValueError) as excinfo:
        AnalysisConfig(**kwargs)
    msg = str(excinfo.value)
    for allowed in _TIER_NAMES:
        assert allowed in msg

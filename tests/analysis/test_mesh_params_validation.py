"""Tests for MeshParams input validation.

Issue #11: invalid MeshParams (elements_per_ply <= 0, in_plane_size_mm <= 0,
cohesive_zone_factor <= 0) used to silently flow into the fe3d builder and
either produce a degenerate mesh or trip a far-downstream error. The
``__post_init__`` validator now surfaces the bad input at construction time.
"""

import pytest

from bvidfe.analysis.config import MeshParams


def test_mesh_params_default_is_valid():
    mp = MeshParams()
    assert mp.elements_per_ply == 1
    assert mp.in_plane_size_mm == 5.0
    assert mp.cohesive_zone_factor == 1.0


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_mesh_params_rejects_invalid_elements_per_ply(bad):
    """elements_per_ply must be a positive int — float values are rejected
    even when they happen to be > 0 because they cause silent truncation
    in the brick-mesh builder."""
    with pytest.raises(ValueError, match="elements_per_ply"):
        MeshParams(elements_per_ply=bad)


@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_mesh_params_rejects_non_positive_in_plane_size(bad):
    with pytest.raises(ValueError, match="in_plane_size_mm"):
        MeshParams(in_plane_size_mm=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_mesh_params_rejects_non_positive_cohesive_zone_factor(bad):
    with pytest.raises(ValueError, match="cohesive_zone_factor"):
        MeshParams(cohesive_zone_factor=bad)

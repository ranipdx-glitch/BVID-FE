"""Regression tests for ``_T_sigma_z`` lru_cache memoization (issue #86).

The 6x6 Voigt stress-rotation matrix is rebuilt once per element by
``Hex8Element._compute_global_stiffness``. On a typical mesh with 1e4-1e5
elements but only a handful of distinct ply angles, identical-angle
recomputation dominates element-construction cost. These tests pin the
cache behaviour so future refactors do not silently disable it.
"""

import numpy as np

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.elements.hex8 import Hex8Element, _T_sigma_z


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


def test_t_sigma_z_lru_cache_hits_on_repeated_angle():
    """Second call with the same theta must hit the cache, not recompute."""
    _T_sigma_z.cache_clear()
    theta = np.radians(45.0)
    first = _T_sigma_z(theta)
    info_after_first = _T_sigma_z.cache_info()
    second = _T_sigma_z(theta)
    info_after_second = _T_sigma_z.cache_info()

    # First call: a miss; second call: a hit.
    assert info_after_first.misses == 1
    assert info_after_first.hits == 0
    assert info_after_second.hits == 1
    assert info_after_second.misses == 1
    # Cache returns the *same* underlying array, not a fresh copy.
    assert second is first


def test_t_sigma_z_returns_read_only_array():
    """The cached matrix must be read-only so callers cannot corrupt it."""
    _T_sigma_z.cache_clear()
    T = _T_sigma_z(np.radians(30.0))
    assert T.flags.writeable is False


def test_compute_global_stiffness_reuses_cache_across_elements():
    """Many Hex8Elements at the same ply angle must share one cached T."""
    _T_sigma_z.cache_clear()
    m = MATERIAL_LIBRARY["IM7/8552"]
    # Build ten elements at +45 deg and ten at -45 deg — only two cache misses
    # should occur regardless of element count.
    for _ in range(10):
        Hex8Element(_unit_cube_nodes(), m, ply_angle_deg=45.0)
    for _ in range(10):
        Hex8Element(_unit_cube_nodes(), m, ply_angle_deg=-45.0)
    info = _T_sigma_z.cache_info()
    assert info.misses == 2
    assert info.hits == 18

"""Regression tests for the ``_navier_basis_ssss`` lru_cache (issue #85).

In Streamlit energy sweeps the panel geometry and impact location are constant
across iterations; we cache the geometric Navier basis so the ~121-term
``sin^2`` evaluation is not repeated for every laminate / energy point.
"""

import numpy as np
import pytest

from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.impact.olsson import NAVIER_N, _navier_basis_ssss, onset_energy


@pytest.fixture(autouse=True)
def _clear_navier_basis_cache():
    """Isolate each test from cache state set by prior tests."""
    _navier_basis_ssss.cache_clear()
    yield
    _navier_basis_ssss.cache_clear()


def test_navier_basis_cache_hits_on_repeat_call():
    a, b, x0, y0 = 150.0, 100.0, 75.0, 50.0
    arr1 = _navier_basis_ssss(a, b, x0, y0, NAVIER_N)
    arr2 = _navier_basis_ssss(a, b, x0, y0, NAVIER_N)
    info = _navier_basis_ssss.cache_info()
    assert info.hits >= 1
    # Same object returned on cache hit.
    assert arr1 is arr2


def test_navier_basis_is_readonly():
    arr = _navier_basis_ssss(150.0, 100.0, 75.0, 50.0, NAVIER_N)
    assert arr.flags.writeable is False
    with pytest.raises(ValueError):
        arr[0, 0] = 999.0


def test_navier_basis_shape_and_values():
    a, b, x0, y0 = 150.0, 100.0, 75.0, 50.0
    arr = _navier_basis_ssss(a, b, x0, y0, NAVIER_N)
    assert arr.shape == (NAVIER_N, NAVIER_N)
    # Spot-check (m, n) = (1, 1): sin^2(pi/2) * sin^2(pi/2) = 1.
    assert np.isclose(arr[0, 0], 1.0)


def test_onset_energy_energy_sweep_hits_cache():
    """An energy-sweep-style loop should produce cache hits on the basis."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    lam = Laminate(m, [0, 45, -45, 90] * 4, 0.152)
    pan = PanelGeometry(150, 100)
    imp = ImpactorGeometry()
    # Call onset_energy several times with identical geometry+location.
    for _ in range(5):
        onset_energy(lam, pan, imp)
    info = _navier_basis_ssss.cache_info()
    assert info.hits >= 4

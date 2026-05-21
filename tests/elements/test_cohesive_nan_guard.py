"""Regression tests for the cohesive shear traction decomposition NaN guard.

See GitHub issue #109: tiny but nonzero tangential separations (e.g. round-off
noise) used to produce NaN tractions when ``d_t_mag`` was effectively zero but
not exactly zero. The guard is now magnitude-relative.
"""

import numpy as np

from bvidfe.elements.cohesive import CohesiveSurfaceElement


def test_cohesive_pure_normal_separation_returns_zero_shear():
    """Pure normal opening (d_t1 = d_t2 = 0) yields exactly zero shear tractions."""
    elem = CohesiveSurfaceElement(sigma_n_max=60.0, tau_max=90.0, G_Ic=0.28, G_IIc=0.79)
    T = elem.traction(np.array([1e-4, 0.0, 0.0]))
    assert np.isfinite(T).all()
    assert T[1] == 0.0
    assert T[2] == 0.0
    # Normal traction should still be positive (elastic tensile opening)
    assert T[0] > 0


def test_cohesive_near_zero_tangential_no_nan():
    """Tiny tangential separations (rounding noise) must not produce NaN tractions.

    With ``d_t1 = d_t2 = 1e-200`` the squared components underflow to 0,
    so ``d_t_mag`` is exactly 0 even though the components are nonzero —
    the magnitude-relative guard catches this and avoids 0/0 NaN.
    """
    elem = CohesiveSurfaceElement(sigma_n_max=60.0, tau_max=90.0, G_Ic=0.28, G_IIc=0.79)
    T = elem.traction(np.array([1e-4, 1e-200, 1e-200]))
    assert np.isfinite(T).all()
    assert not np.isnan(T).any()
    # Shear tractions are bounded — either clamped to zero or vanishingly small,
    # never NaN/Inf. The point of the magnitude-relative guard is finiteness.
    assert abs(T[1]) < 1e-10
    assert abs(T[2]) < 1e-10


def test_cohesive_healthy_shear_loaded_separation_is_finite_and_proportional():
    """A genuine shear-loaded separation produces finite, direction-aligned tractions."""
    elem = CohesiveSurfaceElement(sigma_n_max=60.0, tau_max=90.0, G_Ic=0.28, G_IIc=0.79)
    # Pick d_t magnitudes inside the elastic regime so tractions are clearly nonzero.
    d_t1, d_t2 = 5.0e-5, 3.0e-5
    T = elem.traction(np.array([0.0, d_t1, d_t2]))
    assert np.isfinite(T).all()
    # T_t1 / T_t2 == d_t1 / d_t2 (shear traction aligns with shear separation)
    assert T[1] != 0.0
    assert T[2] != 0.0
    assert np.isclose(T[1] / T[2], d_t1 / d_t2)
    # Both tangential tractions have the same sign as their respective inputs
    assert T[1] > 0
    assert T[2] > 0

"""CLT validation: Laminate.abd_matrices vs textbook closed-form values.

For a [0/90]_s IM7/8552 laminate the A and D matrices have known
closed-form values from Jones (1999) "Mechanics of Composite Materials".
This test pins the implementation against those values and against a
hand-computed reference so silent drift in the CLT routines (e.g. wrong
ply ordering, missed Poisson term) surfaces immediately.

Reference layup: [0, 90, 90, 0] with t_ply = 0.152 mm (h = 0.608 mm).
Material: IM7/8552 from MATERIAL_LIBRARY (E11 = 165 GPa, E22 = 8.4 GPa,
nu12 = 0.34, G12 = 5.6 GPa).

Numeric values are derived in the agent's audit (see issue #11 work in
the changelog). They match the formula
   A_ij = sum_k Q_bar_ij(theta_k) * t_k
   D_ij = (1/3) * sum_k Q_bar_ij(theta_k) * (z_{k+1}^3 - z_k^3)
to better than 1e-3 relative.
"""

from __future__ import annotations

import numpy as np

from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY


def _balanced_cross_ply():
    return Laminate(
        material=MATERIAL_LIBRARY["IM7/8552"],
        layup_deg=[0.0, 90.0, 90.0, 0.0],
        ply_thickness_mm=0.152,
    )


def test_clt_A11_equals_A22_for_balanced_cross_ply():
    """A11 = A22 by symmetry in [0/90]_s when E11/E22 are isotropic in stack."""
    A, _, _ = _balanced_cross_ply().abd_matrices()
    assert A[0, 0] == A[1, 1]


def test_clt_A_matches_closed_form_im7():
    """Pinned A (extensional) matrix for [0/90]_s IM7/8552, t_ply=0.152 mm."""
    A, _, _ = _balanced_cross_ply().abd_matrices()
    # Reference values (units: N/mm) from Q_bar * t integration.
    A_ref = np.array(
        [
            [53025.66, 1746.73, 0.0],
            [1746.73, 53025.66, 0.0],
            [0.0, 0.0, 3404.80],
        ]
    )
    np.testing.assert_allclose(A, A_ref, rtol=1e-3, atol=0.5)


def test_clt_D_matches_closed_form_im7():
    """Pinned D (bending) matrix for [0/90]_s IM7/8552, t_ply=0.152 mm."""
    _, _, D = _balanced_cross_ply().abd_matrices()
    # Reference values (units: N*mm) from Q_bar * dz^3/3 integration.
    D_ref = np.array(
        [
            [2739.88, 53.81, 0.0],
            [53.81, 527.06, 0.0],
            [0.0, 0.0, 104.89],
        ]
    )
    np.testing.assert_allclose(D, D_ref, rtol=1e-2, atol=0.5)


def test_clt_B_zero_for_symmetric_layup():
    """Symmetric layups must have B = 0 to within numerical precision."""
    _, B, _ = _balanced_cross_ply().abd_matrices()
    np.testing.assert_allclose(B, 0.0, atol=1e-6)


def test_clt_no_extension_shear_coupling_for_balanced():
    """A_16 = A_26 = 0 for balanced layups (no off-axis plies)."""
    A, _, _ = _balanced_cross_ply().abd_matrices()
    np.testing.assert_allclose([A[0, 2], A[1, 2], A[2, 0], A[2, 1]], 0.0, atol=1e-6)

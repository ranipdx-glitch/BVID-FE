"""Hex8 with 3 incompatible bubble modes (Wilson-Taylor / Simo-Rifai style).

After static condensation the element still has 24 DOF but is significantly
less stiff in pure bending modes than the pure Hex8. Useful for coarse meshes
where the fully-integrated Hex8 would lock.

Bubble shape functions:
    M_1(xi, eta, zeta) = 1 - xi^2
    M_2(xi, eta, zeta) = 1 - eta^2
    M_3(xi, eta, zeta) = 1 - zeta^2

Each of the 3 bubbles contributes a 3-DOF internal parameter (one per spatial
direction), giving 9 internal DOFs (alpha). The enhanced strain-displacement
operator Benh (6 x 9) is derived from the M_i gradients mapped with the
Jacobian at the element center (xi=eta=zeta=0) for patch-test consistency.

Condensation:
    K = Kuu - Kua @ inv(Kaa) @ Kua.T
"""

from __future__ import annotations

import numpy as np

from bvidfe.elements.gauss import gauss_points_hex
from bvidfe.elements.hex8 import Hex8Element, _validate_jacobian


class Hex8iElement(Hex8Element):
    """Hex8 enriched with 3 incompatible bubble modes."""

    def _Benh(
        self, xi: float, eta: float, zeta: float, J0_inv: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Enhanced strain-displacement matrix Benh (6, 9) at (xi, eta, zeta).

        M_i gradients in natural coords:
          dM_1/dxi  = -2*xi;  dM_1/deta = 0;    dM_1/dzeta = 0
          dM_2/dxi  = 0;      dM_2/deta = -2*eta; dM_2/dzeta = 0
          dM_3/dxi  = 0;      dM_3/deta = 0;    dM_3/dzeta = -2*zeta

        Mapped to physical coords with J0_inv (evaluated at element center):
          dM_i_phys = J0_inv @ dM_i_natural
        """
        dMn = np.zeros((3, 3))  # rows: xi/eta/zeta natural, cols: bubble index
        dMn[0, 0] = -2 * xi
        dMn[1, 1] = -2 * eta
        dMn[2, 2] = -2 * zeta
        dM_phys = J0_inv @ dMn  # (3, 3)

        Benh = np.zeros((6, 9))
        for k in range(3):
            Mx, My, Mz = dM_phys[0, k], dM_phys[1, k], dM_phys[2, k]
            col = 3 * k
            Benh[0, col + 0] = Mx
            Benh[1, col + 1] = My
            Benh[2, col + 2] = Mz
            Benh[3, col + 1] = Mz
            Benh[3, col + 2] = My
            Benh[4, col + 0] = Mz
            Benh[4, col + 2] = Mx
            Benh[5, col + 0] = My
            Benh[5, col + 1] = Mx
        _, detJ = self.B_matrix(xi, eta, zeta)
        return Benh, detJ

    def stiffness_matrix(self) -> np.ndarray:
        """Condensed 24x24 stiffness with 3 incompatible modes eliminated."""
        gp, wt = gauss_points_hex(order=2)
        C = self._C_global

        # Reference Jacobian at center for patch-test consistency
        J0 = self.jacobian(0.0, 0.0, 0.0)
        _validate_jacobian(np.linalg.det(J0), 0.0, 0.0, 0.0, self.node_coords)
        J0_inv = np.linalg.inv(J0)

        Kuu = np.zeros((24, 24))
        Kua = np.zeros((24, 9))
        Kaa = np.zeros((9, 9))
        for ig in range(gp.shape[0]):
            xi, eta, zeta = gp[ig]
            B, detJ = self.B_matrix(xi, eta, zeta)
            Benh, _ = self._Benh(xi, eta, zeta, J0_inv)
            vol = detJ * wt[ig]
            Kuu += (B.T @ C @ B) * vol
            Kua += (B.T @ C @ Benh) * vol
            Kaa += (Benh.T @ C @ Benh) * vol

        # Static condensation
        try:
            Kaa_inv = np.linalg.inv(Kaa)
        except np.linalg.LinAlgError:
            return Kuu
        K = Kuu - Kua @ Kaa_inv @ Kua.T
        # Enforce exact symmetry
        return 0.5 * (K + K.T)

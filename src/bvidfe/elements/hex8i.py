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

import logging

import numpy as np

from bvidfe.elements.gauss import gauss_points_hex
from bvidfe.elements.hex8 import Hex8Element, _validate_jacobian

# Opt-in strict mode: when True, a singular Kaa during static condensation
# raises LinAlgError instead of silently falling back to the un-condensed Kuu.
# Default (False) preserves historical behavior; tests / callers may flip this
# to fail fast when investigating mesh-quality issues.
STRICT_HEX8I_CONDENSATION: bool = False

_logger = logging.getLogger("bvidfe.elements")

# Dedup key set so a large mesh doesn't spam the same warning per element.
# Keys are (ply_angle_deg_rounded, cond_kaa_rounded).
_warned_elements: set[tuple[float, float]] = set()


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
        except np.linalg.LinAlgError as exc:
            # Diagnostic: compute condition number of Kaa for the log/exception.
            try:
                cond_kaa = float(np.linalg.cond(Kaa))
                if not np.isfinite(cond_kaa):
                    cond_kaa = float("inf")
            except (np.linalg.LinAlgError, ValueError):
                cond_kaa = float("inf")

            ply_angle = getattr(self, "ply_angle_deg", None)
            elem_id = getattr(self, "element_id", None)
            elem_label = (
                f"hex8i element id={elem_id}" if elem_id is not None else "an hex8i element"
            )

            if STRICT_HEX8I_CONDENSATION:
                raise np.linalg.LinAlgError(
                    f"Singular Kaa during Hex8i static condensation "
                    f"({elem_label}, ply_angle={ply_angle}, "
                    f"cond(Kaa)={cond_kaa:.3e}); enrichment cannot be applied."
                ) from exc

            # Dedup so a 100k-element mesh doesn't emit 100k identical warnings.
            ply_key = round(float(ply_angle), 3) if ply_angle is not None else float("nan")
            if np.isfinite(cond_kaa):
                cond_key = round(cond_kaa, -int(np.floor(np.log10(max(cond_kaa, 1.0)))))
            else:
                cond_key = float("inf")
            dedup_key = (ply_key, cond_key)
            if dedup_key not in _warned_elements:
                _warned_elements.add(dedup_key)
                _logger.warning(
                    "Hex8i static condensation: Kaa is singular for %s "
                    "(ply_angle=%s, cond(Kaa)=%.3e). Enrichment disabled for "
                    "this element; check mesh quality if FE3D results are noisy.",
                    elem_label,
                    ply_angle,
                    cond_kaa,
                )
            return Kuu
        K = Kuu - Kua @ Kaa_inv @ Kua.T
        # Enforce exact symmetry
        return 0.5 * (K + K.T)

"""Zero-thickness cohesive surface element with bilinear traction-separation.

Simplified BVID-FE v0.1.0 formulation:
- Constitutive law evaluated at a material point (traction vs local separation).
- Uses a scalar equivalent separation for mixed-mode damage.
- Pristine normal compression stays elastic (no damage in compression).
- Mode I and Mode II use separate peak tractions and fracture toughnesses;
  effective peak is computed via a simplified power-law mixing.

Full finite-element integration over the interface quadrilateral is deferred
to the 3D FE tier (Phase 10) — for v0.1.0 the constitutive model and pointwise
stiffness are sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_COHESIVE_SHEAR_FLOOR = 1e-12  # relative threshold for tangential separation magnitude


@dataclass
class CohesiveSurfaceElement:
    """Bilinear mixed-mode cohesive model at a single material point.

    Parameters
    ----------
    sigma_n_max : peak normal (mode I) traction [MPa].
    tau_max     : peak shear (mode II/III) traction [MPa].
    G_Ic        : mode I fracture toughness [N/mm].
    G_IIc       : mode II fracture toughness [N/mm].
    K_penalty   : initial elastic penalty stiffness [N/mm^3], default 1e6.
    is_precracked : when True, traction is identically zero.
    """

    sigma_n_max: float
    tau_max: float
    G_Ic: float
    G_IIc: float
    K_penalty: float = 1.0e6
    is_precracked: bool = False

    def __post_init__(self) -> None:
        if not self.is_precracked:
            if self.sigma_n_max <= 0 or self.tau_max <= 0:
                raise ValueError("peak tractions must be > 0 for pristine cohesive element")
            if self.G_Ic <= 0 or self.G_IIc <= 0:
                raise ValueError("fracture toughnesses must be > 0")

    # ---- Mode-separated onset and failure separations ----

    def _delta_0_I(self) -> float:
        return self.sigma_n_max / self.K_penalty

    def _delta_f_I(self) -> float:
        return 2.0 * self.G_Ic / self.sigma_n_max

    def _delta_0_II(self) -> float:
        return self.tau_max / self.K_penalty

    def _delta_f_II(self) -> float:
        return 2.0 * self.G_IIc / self.tau_max

    # ---- Core constitutive law ----

    def _scalar_traction(self, delta: float, delta_0: float, delta_f: float) -> float:
        """Bilinear scalar traction-separation with damage."""
        if delta <= 0:
            return self.K_penalty * delta  # compressive: elastic, no damage
        if delta <= delta_0:
            return self.K_penalty * delta
        if delta >= delta_f:
            return 0.0
        # softening regime: d = (delta_f * (delta - delta_0)) / (delta * (delta_f - delta_0))
        d = (delta_f * (delta - delta_0)) / (delta * (delta_f - delta_0))
        return (1.0 - d) * self.K_penalty * delta

    def traction(self, separation: np.ndarray) -> np.ndarray:
        """Traction vector (3,) from separation vector (3,): [delta_n, delta_t1, delta_t2]."""
        if self.is_precracked:
            return np.zeros(3)

        d_n = separation[0]
        d_t1 = separation[1]
        d_t2 = separation[2]
        d_t_mag = math.sqrt(d_t1 * d_t1 + d_t2 * d_t2)

        # Normal component: only tensile opening has damage; compression stays elastic
        if d_n <= 0:
            T_n = self.K_penalty * d_n  # elastic compression
        else:
            T_n = self._scalar_traction(d_n, self._delta_0_I(), self._delta_f_I())

        # Shear magnitude
        T_t_mag = self._scalar_traction(d_t_mag, self._delta_0_II(), self._delta_f_II())

        # Decompose shear back onto tangential directions
        if d_t_mag > _COHESIVE_SHEAR_FLOOR * max(abs(d_t1), abs(d_t2), _COHESIVE_SHEAR_FLOOR):
            T_t1 = T_t_mag * d_t1 / d_t_mag
            T_t2 = T_t_mag * d_t2 / d_t_mag
        else:
            T_t1 = 0.0
            T_t2 = 0.0

        return np.array([T_n, T_t1, T_t2])

    def stiffness_matrix_point(self) -> np.ndarray:
        """Elastic point-tangent (3 x 3). Useful for the FE tier pre-crack assembly."""
        if self.is_precracked:
            return np.zeros((3, 3))
        return self.K_penalty * np.eye(3)

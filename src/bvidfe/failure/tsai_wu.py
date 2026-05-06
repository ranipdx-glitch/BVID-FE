"""3D Tsai-Wu anisotropic failure criterion (Voigt 6-vector, material frame).

Voigt convention used here matches the rest of the codebase (Hex8Element):
    [sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy]
which makes index 3 the 2-3 shear (S23), index 4 the 1-3 shear (S13), and
index 5 the in-plane shear (S12).

Through-thickness strengths come from ``OrthotropicMaterial.Zt_resolved`` /
``Zc_resolved`` / ``S13_resolved``, which fall back to the in-plane
transverse / shear values when the user has not measured the through-thickness
ones (transverse-isotropy assumption for unidirectional CFRP). With those
defaults the formula reduces to the prior plane-stress-style code.
"""

from __future__ import annotations

import math
from typing import Sequence

from bvidfe.core.material import OrthotropicMaterial


def _tsai_wu_coefficients(m: OrthotropicMaterial):
    """Return (F_linear, F_quad) where F_linear is a 6-vector and F_quad is a 6x6 matrix."""
    Xt, Xc, Yt, Yc, S12, S23 = m.Xt, m.Xc, m.Yt, m.Yc, m.S12, m.S23
    Zt, Zc, S13 = m.Zt_resolved, m.Zc_resolved, m.S13_resolved
    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F3 = 1.0 / Zt - 1.0 / Zc
    F = [F1, F2, F3, 0.0, 0.0, 0.0]

    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F33 = 1.0 / (Zt * Zc)
    F44 = 1.0 / (S23**2)  # 2-3 shear (Voigt index 3)
    F55 = 1.0 / (S13**2)  # 1-3 shear (Voigt index 4) — was S12 (a real bug)
    F66 = 1.0 / (S12**2)  # 1-2 in-plane shear (Voigt index 5)
    F12 = -0.5 * math.sqrt(F11 * F22)
    F13 = -0.5 * math.sqrt(F11 * F33)
    F23 = -0.5 * math.sqrt(F22 * F33)

    Q = [[0.0] * 6 for _ in range(6)]
    Q[0][0] = F11
    Q[1][1] = F22
    Q[2][2] = F33
    Q[3][3] = F44
    Q[4][4] = F55
    Q[5][5] = F66
    Q[0][1] = Q[1][0] = F12
    Q[0][2] = Q[2][0] = F13
    Q[1][2] = Q[2][1] = F23
    return F, Q


def tsai_wu_index(m: OrthotropicMaterial, stress: Sequence[float]) -> float:
    """Return the Tsai-Wu failure index F_i s_i + F_ij s_i s_j."""
    F, Q = _tsai_wu_coefficients(m)
    linear = sum(F[i] * stress[i] for i in range(6))
    quad = sum(Q[i][j] * stress[i] * stress[j] for i in range(6) for j in range(6))
    return linear + quad


def tsai_wu_strength_uniaxial(m: OrthotropicMaterial, direction: int, sign: int) -> float:
    """Return the applied uniaxial stress magnitude |sigma| at which the Tsai-Wu
    index equals 1 for sigma_(direction) = sign * |sigma|, other components zero.

    direction is 1-indexed (1, 2, or 3). sign is +1 (tension) or -1 (compression).
    Solve F * (sign*s) + Q * (sign*s)^2 = 1 -> Q * s^2 + sign * F * s - 1 = 0.
    """
    F, Q = _tsai_wu_coefficients(m)
    i = direction - 1
    a = Q[i][i]
    b = sign * F[i]
    c = -1.0
    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError("no real Tsai-Wu strength for this loading")
    # positive root
    return (-b + math.sqrt(disc)) / (2 * a)

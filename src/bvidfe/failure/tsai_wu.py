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

import numpy as np

from bvidfe.core.material import OrthotropicMaterial


def _tsai_wu_coefficients(m: OrthotropicMaterial):
    """Return (F_linear, F_quad) where F_linear is a (6,) array and F_quad is a (6, 6) array."""
    Xt, Xc, Yt, Yc, S12, S23 = m.Xt, m.Xc, m.Yt, m.Yc, m.S12, m.S23
    Zt, Zc, S13 = m.Zt_resolved, m.Zc_resolved, m.S13_resolved
    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F3 = 1.0 / Zt - 1.0 / Zc
    F = np.array([F1, F2, F3, 0.0, 0.0, 0.0], dtype=float)

    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F33 = 1.0 / (Zt * Zc)
    F44 = 1.0 / (S23**2)  # 2-3 shear (Voigt index 3)
    F55 = 1.0 / (S13**2)  # 1-3 shear (Voigt index 4) — was S12 (a real bug)
    F66 = 1.0 / (S12**2)  # 1-2 in-plane shear (Voigt index 5)
    F12 = -0.5 * math.sqrt(F11 * F22)
    F13 = -0.5 * math.sqrt(F11 * F33)
    F23 = -0.5 * math.sqrt(F22 * F33)

    Q = np.zeros((6, 6), dtype=float)
    Q[0, 0] = F11
    Q[1, 1] = F22
    Q[2, 2] = F33
    Q[3, 3] = F44
    Q[4, 4] = F55
    Q[5, 5] = F66
    Q[0, 1] = Q[1, 0] = F12
    Q[0, 2] = Q[2, 0] = F13
    Q[1, 2] = Q[2, 1] = F23
    return F, Q


def tsai_wu_index(m: OrthotropicMaterial, stress: Sequence[float]) -> float:
    """Return the Tsai-Wu (1971) failure index for a Voigt-6 stress state.

    Computes the bilinear-and-quadratic invariant
        index = F_i * sigma_i + F_ij * sigma_i * sigma_j
    using the linear (F) and quadratic (Q) coefficient arrays from
    ``_tsai_wu_coefficients``. Failure is predicted when the index reaches
    1; values above 1 are the fraction-of-overload (margin to failure).
    The Voigt convention is the project-wide
    ``[sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy]`` (i.e. F44
    couples to the 2-3 shear, F55 to the 1-3 shear, F66 to the in-plane
    shear); through-thickness strengths come from
    ``OrthotropicMaterial.Zt_resolved`` / ``Zc_resolved`` / ``S13_resolved``
    so the formula reduces to the plane-stress form when those fields are
    not supplied.

    Parameters
    ----------
    m : OrthotropicMaterial
        Material card with Xt/Xc/Yt/Yc/S12/S23 (mandatory) and optional
        Zt/Zc/S13 (transverse-isotropy fallback applies otherwise).
    stress : sequence of 6 floats
        Voigt-6 stress vector in the material frame; see convention above.

    Returns
    -------
    float
        Dimensionless failure index. >= 1 means failure.
    """
    F, Q = _tsai_wu_coefficients(m)
    s = np.asarray(stress, dtype=float)
    # Bilinear form: sum_i F_i s_i  +  sum_{i,j} Q_{ij} s_i s_j
    # is mathematically identical to the nested-sum form by definition of
    # vector-vector dot product and the bilinear-form contraction; numpy
    # routes both through BLAS so a single Tsai-Wu call drops from ~36
    # Python-level multiplications to two C-level dot products.
    return float(F.dot(s) + s.dot(Q.dot(s)))


def tsai_wu_index_batch(m: OrthotropicMaterial, stresses: np.ndarray) -> np.ndarray:
    """Vectorised Tsai-Wu failure index across a batch of Voigt-6 stresses.

    Equivalent to ``np.array([tsai_wu_index(m, s) for s in stresses])`` but
    evaluates ``F · s + s · Q · s`` once for the whole batch via numpy
    einsum. Used by ``FailureEvaluator`` and intended for future use in
    the ``_solve_failure_strain_analytic`` strain-bisection inner loop.

    Parameters
    ----------
    m : OrthotropicMaterial
        Material card; same fields as in ``tsai_wu_index``.
    stresses : np.ndarray
        Shape ``(..., 6)`` array of Voigt-6 stress vectors. The leading
        axes are preserved in the output.

    Returns
    -------
    np.ndarray
        Shape ``stresses.shape[:-1]``. Each entry is the dimensionless
        Tsai-Wu failure index.
    """
    s = np.asarray(stresses, dtype=float)
    if s.shape[-1] != 6:
        raise ValueError(f"stresses must have last axis 6 (got {s.shape!r})")
    F, Q = _tsai_wu_coefficients(m)
    linear = s @ F  # (...,)
    quad = np.einsum("...i,ij,...j->...", s, Q, s)
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

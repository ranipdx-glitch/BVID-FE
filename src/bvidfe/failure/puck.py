"""Puck (1998) action-plane composite failure criterion.

Implements the inter-fiber-failure (IFF) action-plane search of Puck &
Schürmann (Puck, A.; Schürmann, H., 1998, *Composites Science and
Technology* 58, 1045-1067; refined in Puck & Knops, 2002 and the
World-Wide Failure Exercise II). The fiber-failure (FF) mode is the
simple maximum-stress envelope on sigma_1.

For each candidate fracture-plane angle theta in [-90 deg, +90 deg], the
out-of-plane stresses are transformed via the standard 2-3 plane
rotation:

    sigma_n  =  sigma_2 cos^2 theta + sigma_3 sin^2 theta + 2 tau_23 s c
    tau_nt   = (sigma_3 - sigma_2) s c + tau_23 (c^2 - s^2)
    tau_n1   =  tau_12 cos theta + tau_13 sin theta            (with s = sin t, c = cos t)

The IFF index on that plane is

    sigma_n >= 0   (tensile, Mode A):
        f = sqrt( (tau_nt / R_A)^2 + (tau_n1 / S12)^2
                 + (1 - p+ * Yt/R_A)^2 * (sigma_n/Yt)^2 )
            + (p+ / R_A) * sigma_n

    sigma_n < 0    (compressive, Mode B/C envelope):
        f = sqrt( (tau_nt / R_A)^2 + (tau_n1 / S12)^2
                 + (p- * sigma_n / R_A)^2 )
            + (p- / R_A) * sigma_n

where R_A = Yc / (2 (1 + p-)) is the transverse-transverse action-plane
shear strength and p+/p- are the inclination parameters
``mat.puck_p_nt_plus`` / ``mat.puck_p_nt_minus`` (Puck-Schürmann 1998,
section 4; defaults p+ = 0.30, p- = 0.25 for CFRP/epoxy).

The fiber index is

    f_FF = sigma_1 / Xt  if sigma_1 >= 0 else |sigma_1| / Xc

and the returned scalar is ``max(f_FF, max_theta f_IFF(theta))``.

Voigt convention matches the rest of the codebase
``[sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from bvidfe.core.material import OrthotropicMaterial

# Discrete sweep of candidate fracture-plane angles. 37 evenly-spaced
# points in [-pi/2, +pi/2] gives 5 deg resolution, which is enough to
# resolve the ~53 deg CFRP compression fracture plane within 1% of the
# analytical index without sacrificing batch throughput.
_N_THETA = 37
_THETAS = np.linspace(-np.pi / 2.0, np.pi / 2.0, _N_THETA)
_COS = np.cos(_THETAS)
_SIN = np.sin(_THETAS)
_C2 = _COS * _COS  # cos^2 theta
_S2 = _SIN * _SIN  # sin^2 theta
_SC = _SIN * _COS  # sin*cos
_CS2 = _C2 - _S2  # cos^2 - sin^2 = cos(2 theta)


def _action_plane_R_A(mat: OrthotropicMaterial) -> float:
    """Transverse-transverse shear strength on the action plane (R_perp_perp_A).

    Puck-Schürmann 1998 eq. (35): R_A = Yc / (2 (1 + p-)).
    """
    return mat.Yc / (2.0 * (1.0 + mat.puck_p_nt_minus))


def puck_index(mat: OrthotropicMaterial, stress: Sequence[float]) -> float:
    """Return the maximum Puck failure index for a Voigt-6 stress.

    Performs the action-plane search across ``_N_THETA`` discrete angles
    and returns ``max(fiber_index, max_theta(IFF_index))``. The IFF
    index uses the standard Puck-Schürmann Mode-A (tensile sigma_n) and
    Mode-B/C (compressive sigma_n) envelopes — see module docstring.

    Parameters
    ----------
    mat : OrthotropicMaterial
        Material card supplying Xt/Xc/Yt/Yc/S12 and the Puck inclination
        parameters ``puck_p_nt_plus`` / ``puck_p_nt_minus``.
    stress : sequence of 6 floats
        Voigt-6 stress vector in the material frame, ordered
        ``[sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]``.

    Returns
    -------
    float
        Dimensionless failure index. >= 1 means failure (fiber or IFF
        mode, whichever governs).
    """
    s = np.asarray(stress, dtype=float)
    if s.shape != (6,):
        raise ValueError(f"stress must be shape (6,) (got {s.shape!r})")
    # Re-use the batch routine with a singleton leading axis — the
    # angular sweep is the costly part, the (1, 6) reshape is trivial.
    return float(puck_index_batch(mat, s[None, :])[0])


def puck_index_batch(mat: OrthotropicMaterial, stresses: np.ndarray) -> np.ndarray:
    """Vectorised Puck failure index across a batch of Voigt-6 stresses.

    For an input of shape ``(..., 6)``, returns shape ``stresses.shape[:-1]``.
    Fully vectorised over both the batch axes and the angular sweep —
    the inner action-plane search is a single np.einsum-free broadcast
    so no Python-level loop runs.

    Parameters
    ----------
    mat : OrthotropicMaterial
        Material card; same fields as ``puck_index``.
    stresses : np.ndarray
        Shape ``(..., 6)`` array of Voigt-6 stress vectors.

    Returns
    -------
    np.ndarray
        Shape ``stresses.shape[:-1]``. Each entry is the maximum of the
        fiber-mode index and the inter-fiber-mode index over the
        ``_N_THETA`` candidate fracture planes.
    """
    s = np.asarray(stresses, dtype=float)
    if s.shape[-1] != 6:
        raise ValueError(f"stresses must have last axis 6 (got {s.shape!r})")
    s1 = s[..., 0]
    s2 = s[..., 1]
    s3 = s[..., 2]
    t23 = s[..., 3]
    t13 = s[..., 4]
    t12 = s[..., 5]

    # ---- Fiber-failure (FF) max-stress envelope ----------------------------
    # Tension uses Xt, compression uses Xc; np.where picks the right
    # denominator element-wise.
    Xden = np.where(s1 >= 0.0, mat.Xt, mat.Xc)
    fiber_idx = np.abs(s1) / Xden

    # ---- Inter-fiber-failure (IFF) action-plane sweep ----------------------
    # Broadcast batch (..., ) against angle axis (n_theta,) by adding a
    # trailing 1-axis to every batch stress component. Result shape is
    # (..., n_theta).
    s2_b = s2[..., None]
    s3_b = s3[..., None]
    t23_b = t23[..., None]
    t13_b = t13[..., None]
    t12_b = t12[..., None]

    # Plane-rotation stresses on candidate fracture plane (2-3 rotation by theta).
    sigma_n = s2_b * _C2 + s3_b * _S2 + 2.0 * t23_b * _SC
    tau_nt = (s3_b - s2_b) * _SC + t23_b * _CS2
    tau_n1 = t12_b * _COS + t13_b * _SIN

    R_A = _action_plane_R_A(mat)
    p_plus = mat.puck_p_nt_plus
    p_minus = mat.puck_p_nt_minus
    Yt = mat.Yt
    S12 = mat.S12

    # Tensile-sigma_n branch (Mode A).
    # f_A = sqrt( (tau_nt/R_A)^2 + (tau_n1/S12)^2
    #             + (1 - p+ Yt/R_A)^2 (sigma_n/Yt)^2 )
    #       + (p+ / R_A) sigma_n
    a_factor = 1.0 - p_plus * Yt / R_A
    f_tensile = (
        np.sqrt((tau_nt / R_A) ** 2 + (tau_n1 / S12) ** 2 + (a_factor * sigma_n / Yt) ** 2)
        + (p_plus / R_A) * sigma_n
    )

    # Compressive-sigma_n branch (Mode B / C envelope).
    # f_C = sqrt( (tau_nt/R_A)^2 + (tau_n1/S12)^2 + (p- sigma_n / R_A)^2 )
    #       + (p- / R_A) sigma_n
    f_compressive = (
        np.sqrt((tau_nt / R_A) ** 2 + (tau_n1 / S12) ** 2 + (p_minus * sigma_n / R_A) ** 2)
        + (p_minus / R_A) * sigma_n
    )

    iff_per_theta = np.where(sigma_n >= 0.0, f_tensile, f_compressive)
    # Critical plane is the one that maximises the IFF index.
    iff_idx = np.max(iff_per_theta, axis=-1)

    return np.maximum(fiber_idx, iff_idx)

"""LaRC05 (Hashin-3D reduction) composite failure criterion.

This is a minimal engineering implementation covering the four fundamental
LaRC05 modes (fiber tension, fiber compression, matrix tension, matrix
compression). For BVID CAI/TAI first-ply-failure prediction this is
sufficient; extended LaRC05 features (plane search for matrix cracking,
fiber kinking with non-linear shear) can be added in a future release.

Stress convention: Voigt 6-vector in the material frame,
    [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]
with engineering shear strains.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from bvidfe.core.material import OrthotropicMaterial


def larc05_index(m: OrthotropicMaterial, stress: Sequence[float]) -> float:
    """Return the maximum LaRC05 failure index across the four fundamental modes.

    Implements a minimal Hashin-3D reduction of the LaRC05 criterion (Davila,
    Camanho & Rose, NASA/TM-2005-213530). For a stress state in the material
    frame, the four mode-specific indices are:

      * Fiber tension     (s1 >= 0):  (s1 / Xt)^2
      * Fiber compression (s1  < 0):  (s1 / Xc)^2
      * Matrix tension    (s2 >= 0):  (s2 / Yt)^2 + (t12 / S12)^2 + (t23 / S23)^2
      * Matrix compression(s2  < 0):  (s2 / Yc)^2 + (t12 / S12)^2 + (t23 / S23)^2

    The fiber- and matrix-mode indices are evaluated in parallel and the
    larger is returned. Failure is predicted when the index reaches 1; values
    above 1 indicate the fraction-of-overload (margin-to-failure) for the
    governing mode.

    Parameters
    ----------
    m : OrthotropicMaterial
        Material card supplying Xt/Xc/Yt/Yc/S12/S23. The s3 / t13 components
        of `stress` are accepted (Voigt indexing), but the simplified LaRC05
        reduction implemented here does not couple them into the matrix-mode
        index — extending to the full plane-search formulation is a v0.3.0
        roadmap item.
    stress : sequence of 6 floats
        Voigt-6 stress vector in the material frame, ordered
        ``[s1, s2, s3, t23, t13, t12]`` (i.e. directions 1, 2, 3 followed by
        the three shear components — note the in-plane shear lives at index 5).

    Returns
    -------
    float
        max(fiber_index, matrix_index). Dimensionless; >= 1 indicates failure.
    """
    s1, s2, s3, t23, t13, t12 = stress

    modes: list[float] = []

    # Fiber tension (direction 1)
    if s1 >= 0:
        modes.append((s1 / m.Xt) ** 2)
    else:
        # Fiber compression (direction 1)
        modes.append((s1 / m.Xc) ** 2)

    # Matrix tension (direction 2)
    if s2 >= 0:
        modes.append((s2 / m.Yt) ** 2 + (t12 / m.S12) ** 2 + (t23 / m.S23) ** 2)
    else:
        # Matrix compression (direction 2)
        modes.append((s2 / m.Yc) ** 2 + (t12 / m.S12) ** 2 + (t23 / m.S23) ** 2)

    return max(modes)


def larc05_index_batch(m: OrthotropicMaterial, stresses: np.ndarray) -> np.ndarray:
    """Vectorised LaRC05 failure index across a batch of Voigt-6 stresses.

    Equivalent to ``np.array([larc05_index(m, s) for s in stresses])`` but
    evaluates the four mode branches with ``np.where`` masks instead of
    Python conditionals so a (n, 6) input runs in a single BLAS-routed
    pass. Used by ``FailureEvaluator`` and intended for future use in the
    ``_solve_failure_strain_analytic`` strain-bisection inner loop.

    Parameters
    ----------
    m : OrthotropicMaterial
        Material card; same Xt/Xc/Yt/Yc/S12/S23 used as in ``larc05_index``.
    stresses : np.ndarray
        Shape ``(..., 6)`` array of Voigt-6 stress vectors. The leading
        axes are preserved in the output.

    Returns
    -------
    np.ndarray
        Shape ``stresses.shape[:-1]``. Each entry is the maximum of the
        fiber-mode index and the matrix-mode index for that stress.
    """
    s = np.asarray(stresses, dtype=float)
    if s.shape[-1] != 6:
        raise ValueError(f"stresses must have last axis 6 (got {s.shape!r})")
    s1 = s[..., 0]
    s2 = s[..., 1]
    t23 = s[..., 3]
    t12 = s[..., 5]
    # Fiber mode: tension uses Xt, compression uses Xc — np.where picks the
    # right denominator element-wise without a Python branch.
    Xden = np.where(s1 >= 0, m.Xt, m.Xc)
    fiber_idx = (s1 / Xden) ** 2
    # Matrix mode: same shear contribution either way; tension uses Yt,
    # compression uses Yc.
    Yden = np.where(s2 >= 0, m.Yt, m.Yc)
    matrix_idx = (s2 / Yden) ** 2 + (t12 / m.S12) ** 2 + (t23 / m.S23) ** 2
    return np.maximum(fiber_idx, matrix_idx)

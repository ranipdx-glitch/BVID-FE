"""Classical Lamination Theory (CLT) for BVID-FE composite laminate analysis.

This module provides the ``Laminate`` class for computing the ABD stiffness
matrices (in-plane, coupling, bending), effective engineering constants, and
flexural rigidity of composite laminates.

Units throughout: moduli and stiffness in MPa (N/mm^2), lengths in mm.

References
----------
- Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed. Taylor & Francis.
- Herakovich, C.T. (1998). Mechanics of Fibrous Composites. Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np

from bvidfe.core.material import OrthotropicMaterial

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_ply_thicknesses(raw: Union[float, Sequence[float]], n_plies: int) -> list:
    """Coerce ``raw`` to a per-ply thickness list of length ``n_plies``.

    Accepts a single positive number (uniform thickness) or a sequence of
    positive numbers (per-ply, length must match ``n_plies``). Raises
    ``ValueError`` on any non-positive entry, length mismatch, or unsupported
    type.
    """
    if isinstance(raw, (int, float)):
        t = float(raw)
        if t <= 0.0:
            raise ValueError(f"ply_thickness_mm must be > 0 (got {t}).")
        return [t] * n_plies
    if isinstance(raw, (list, tuple, np.ndarray)):
        ts = [float(x) for x in raw]
        if len(ts) != n_plies:
            raise ValueError(
                f"ply_thickness_mm sequence length ({len(ts)}) must equal "
                f"the number of plies ({n_plies})."
            )
        for i, t in enumerate(ts):
            if t <= 0.0:
                raise ValueError(f"ply_thickness_mm[{i}] must be > 0 (got {t}).")
        return ts
    raise TypeError(
        f"ply_thickness_mm must be a float or sequence of floats " f"(got {type(raw).__name__})."
    )


def _reduced_stiffness(mat: OrthotropicMaterial) -> np.ndarray:
    """Compute the 3x3 on-axis reduced stiffness matrix [Q] for plane stress.

    Parameters
    ----------
    mat : OrthotropicMaterial
        Ply material with fields E11, E22, nu12, G12.

    Returns
    -------
    np.ndarray
        3x3 symmetric matrix [Q] in MPa.
    """
    nu21 = mat.nu12 * mat.E22 / mat.E11
    denom = 1.0 - mat.nu12 * nu21

    Q11 = mat.E11 / denom
    Q12 = mat.nu12 * mat.E22 / denom
    Q22 = mat.E22 / denom
    Q66 = mat.G12

    return np.array(
        [
            [Q11, Q12, 0.0],
            [Q12, Q22, 0.0],
            [0.0, 0.0, Q66],
        ],
        dtype=float,
    )


def _transform_reduced_stiffness(Q: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate the 3x3 reduced stiffness matrix to laminate coordinates.

    Uses the closed-form expressions from Classical Lamination Theory.
    The off-axis (Q-bar) components are computed directly from:
    c = cos(theta), s = sin(theta).

    Parameters
    ----------
    Q : np.ndarray
        3x3 on-axis reduced stiffness matrix.
    angle_rad : float
        Ply orientation angle in radians, measured from the laminate x-axis.

    Returns
    -------
    np.ndarray
        3x3 transformed reduced stiffness matrix [Q-bar] in MPa.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    c2 = c * c
    s2 = s * s
    c4 = c2 * c2
    s4 = s2 * s2
    s2c2 = s2 * c2
    sc3 = s * c * c2
    s3c = s * s2 * c

    Q11 = Q[0, 0]
    Q12 = Q[0, 1]
    Q22 = Q[1, 1]
    Q66 = Q[2, 2]

    Qb11 = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
    Qb12 = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (s4 + c4)
    Qb22 = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
    Qb16 = (Q11 - Q12 - 2.0 * Q66) * sc3 + (Q12 - Q22 + 2.0 * Q66) * s3c
    Qb26 = (Q11 - Q12 - 2.0 * Q66) * s3c + (Q12 - Q22 + 2.0 * Q66) * sc3
    Qb66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (s4 + c4)

    return np.array(
        [
            [Qb11, Qb12, Qb16],
            [Qb12, Qb22, Qb26],
            [Qb16, Qb26, Qb66],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Laminate
# ---------------------------------------------------------------------------


@dataclass
class Laminate:
    """Composite laminate analyzed by Classical Lamination Theory (CLT).

    All plies share the same material; ply thicknesses may be uniform
    (single ``float``) or per-ply (a ``list[float]`` of length ``len(layup_deg)``).
    The stacking sequence is given bottom-to-top; z = 0 is at the laminate
    midplane.

    Parameters
    ----------
    material : OrthotropicMaterial
        Ply material (all plies use the same material).
    layup_deg : list[float]
        Ply fiber angles in degrees, ordered bottom-to-top.
    ply_thickness_mm : float | Sequence[float]
        Either a single uniform ply thickness in mm, or a sequence of per-ply
        thicknesses with length equal to ``len(layup_deg)``. Per-ply
        thicknesses let users model laminates that mix plies of different
        fabric weights or prepreg gauges.

    Examples
    --------
    >>> from bvidfe.core.material import MATERIAL_LIBRARY
    >>> m = MATERIAL_LIBRARY["IM7/8552"]
    >>> lam = Laminate(material=m, layup_deg=[0, 45, -45, 90, 90, -45, 45, 0],
    ...                ply_thickness_mm=0.152)
    >>> A, B, D = lam.abd_matrices()
    >>> import numpy as np; np.allclose(B, 0.0, atol=1e-6)
    True

    Mixed ply thicknesses (e.g. a thin 0/90 surface layer over thicker
    quasi-iso plies):

    >>> lam = Laminate(material=m, layup_deg=[0, 90, 45, -45, -45, 45, 90, 0],
    ...                ply_thickness_mm=[0.10, 0.10, 0.20, 0.20,
    ...                                  0.20, 0.20, 0.10, 0.10])
    >>> abs(lam.thickness_mm - 1.20) < 1e-9
    True
    """

    material: OrthotropicMaterial
    layup_deg: list[float]
    ply_thickness_mm: Union[float, Sequence[float]]

    # Computed on post-init; stored as private attributes.
    _ply_thicknesses: list = field(init=False, repr=False)
    _A: np.ndarray = field(init=False, repr=False)
    _B: np.ndarray = field(init=False, repr=False)
    _D: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs and pre-compute ABD matrices."""
        if not self.layup_deg:
            raise ValueError("layup_deg must contain at least one ply angle.")
        self._ply_thicknesses = _normalise_ply_thicknesses(
            self.ply_thickness_mm, len(self.layup_deg)
        )
        self._compute_abd()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def n_plies(self) -> int:
        """Number of plies in the laminate."""
        return len(self.layup_deg)

    @property
    def ply_thicknesses_mm(self) -> list:
        """Per-ply thickness list (mm) of length ``n_plies``.

        Always returns a list, regardless of whether ``ply_thickness_mm`` was
        provided as a scalar or a sequence. Internal modules that need to
        sum, slice, or index per-ply thicknesses should use this property.
        """
        return list(self._ply_thicknesses)

    @property
    def is_uniform_thickness(self) -> bool:
        """True if every ply has the same thickness."""
        ts = self._ply_thicknesses
        return all(t == ts[0] for t in ts)

    @property
    def thickness_mm(self) -> float:
        """Total laminate thickness (mm) — sum of per-ply thicknesses."""
        return float(sum(self._ply_thicknesses))

    def _z_coords(self) -> np.ndarray:
        """Ply boundary z-coordinates from bottom to top (mm).

        The origin (z = 0) is at the laminate midplane.

        Returns
        -------
        np.ndarray
            Shape (n_plies + 1,) array; z[0] = -h/2, z[-1] = +h/2.
        """
        h = self.thickness_mm
        z = np.empty(self.n_plies + 1, dtype=float)
        z[0] = -h / 2.0
        for k, t in enumerate(self._ply_thicknesses):
            z[k + 1] = z[k] + t
        return z

    # ------------------------------------------------------------------
    # ABD computation
    # ------------------------------------------------------------------

    def _compute_abd(self) -> None:
        """Pre-compute and cache the A, B, D 3x3 stiffness matrices.

        Integrates Q-bar through the thickness using the standard CLT
        summation over ply boundaries:

            A_ij = sum_k  Q_bar_ij,k * (z_{k+1} - z_k)
            B_ij = (1/2) * sum_k  Q_bar_ij,k * (z_{k+1}^2 - z_k^2)
            D_ij = (1/3) * sum_k  Q_bar_ij,k * (z_{k+1}^3 - z_k^3)
        """
        A = np.zeros((3, 3), dtype=float)
        B = np.zeros((3, 3), dtype=float)
        D = np.zeros((3, 3), dtype=float)

        Q = _reduced_stiffness(self.material)
        zc = self._z_coords()

        for k, angle_deg in enumerate(self.layup_deg):
            angle_rad = math.radians(angle_deg)
            Qb = _transform_reduced_stiffness(Q, angle_rad)

            z_bot = zc[k]
            z_top = zc[k + 1]

            dz1 = z_top - z_bot
            dz2 = z_top**2 - z_bot**2
            dz3 = z_top**3 - z_bot**3

            A += Qb * dz1
            B += Qb * (0.5 * dz2)
            D += Qb * (dz3 / 3.0)

        self._A = A
        self._B = B
        self._D = D

    def abd_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the CLT in-plane (A), coupling (B), and bending (D) matrices.

        All matrices are 3x3 and operate on the reduced stress/strain vector
        [x, y, xy] in Voigt notation.

        Returns
        -------
        A : np.ndarray
            3x3 extensional stiffness matrix (N/mm).
        B : np.ndarray
            3x3 coupling stiffness matrix (N).
        D : np.ndarray
            3x3 bending stiffness matrix (N*mm).
        """
        return self._A.copy(), self._B.copy(), self._D.copy()

    # ------------------------------------------------------------------
    # Engineering constants
    # ------------------------------------------------------------------

    def effective_engineering_constants(self) -> tuple[float, float, float, float]:
        """Laminate effective engineering constants from the A-matrix compliance.

        Derived from the extensional compliance a* = A^{-1} / h:

            Ex   = 1 / (a*_11)
            Ey   = 1 / (a*_22)
            Gxy  = 1 / (a*_66)
            nuxy = -a*_12 / a*_11

        Returns
        -------
        Ex : float
            Effective Young's modulus in x-direction (MPa).
        Ey : float
            Effective Young's modulus in y-direction (MPa).
        Gxy : float
            Effective in-plane shear modulus (MPa).
        nuxy : float
            Effective Poisson's ratio nu_xy (dimensionless).
        """
        h = self.thickness_mm
        # Guard against a numerically singular A: np.linalg.inv returns
        # NaN/Inf (no exception) when det(A) ~ 0, which then propagates
        # silently into stress recovery as garbage. Fail loudly instead.
        cond = float(np.linalg.cond(self._A))
        if not np.isfinite(cond) or cond > 1e10:
            raise ValueError(
                f"Laminate A matrix is ill-conditioned (cond={cond:.2e}); "
                f"check layup / ply_thickness_mm consistency."
            )
        a = np.linalg.solve(self._A, np.eye(3))  # 3x3 compliance (mm/N)
        # Normalise by thickness to get extensional compliance per unit modulus
        a_star = a / h  # 1/MPa

        Ex = 1.0 / a_star[0, 0]
        Ey = 1.0 / a_star[1, 1]
        Gxy = 1.0 / a_star[2, 2]
        nuxy = -a_star[0, 1] / a_star[0, 0]

        for name, val in (("Ex", Ex), ("Ey", Ey), ("Gxy", Gxy)):
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(
                    f"effective {name}={val} from an ill-conditioned laminate "
                    f"A matrix; check layup / ply_thickness_mm consistency."
                )
        if not np.isfinite(nuxy):
            raise ValueError(f"effective nuxy={nuxy} from an ill-conditioned laminate A matrix.")

        return Ex, Ey, Gxy, nuxy

    # ------------------------------------------------------------------
    # Flexural rigidity
    # ------------------------------------------------------------------

    def flexural_rigidity_Deff(self) -> float:
        """Effective flexural rigidity (geometric mean of D11 and D22).

        Used by the Olsson impact threshold criterion in Phase 4:

            D_eff = sqrt(D11 * D22)

        Returns
        -------
        float
            Effective bending stiffness D_eff (N*mm).
        """
        return math.sqrt(self._D[0, 0] * self._D[1, 1])

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_uniform_thickness:
            t_str = f"t_ply={self._ply_thicknesses[0]:.3f} mm"
        else:
            t_str = "t_ply=[" + ", ".join(f"{t:.3f}" for t in self._ply_thicknesses) + "] mm"
        return (
            f"Laminate(n_plies={self.n_plies}, "
            f"h={self.thickness_mm:.3f} mm, "
            f"{t_str}, "
            f"material={self.material.name!r}, "
            f"layup={self.layup_deg})"
        )

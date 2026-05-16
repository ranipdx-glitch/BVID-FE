"""8-node isoparametric hexahedral element (C3D8) for 3D composite analysis.

Standard trilinear brick with 2x2x2 Gauss quadrature.

Node ordering (Abaqus/VTK convention):
    Bottom face (zeta = -1): 0, 1, 2, 3 CCW from +z
    Top face    (zeta = +1): 4, 5, 6, 7 CCW
Natural coordinates xi, eta, zeta in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bvidfe.core.material import OrthotropicMaterial
from bvidfe.elements.gauss import gauss_points_hex


@dataclass(frozen=True)
class GeometryTable:
    """Per-Gauss-point geometry table for one element shape (2x2x2 rule).

    For a regular cuboid grid every element of a given ply is a translate of
    every other, so its Jacobian, ``detJ``, ``B`` and shape-function spatial
    gradients are identical. Computing this table once per ply and sharing it
    across all of that ply's elements removes the dominant per-element
    ``B_matrix`` / Jacobian recomputation cost while leaving the numerics
    bit-for-bit identical to the per-element path.

    Attributes
    ----------
    B : np.ndarray
        (n_gp, 6, 24) strain-displacement matrices.
    detJ : np.ndarray
        (n_gp,) Jacobian determinants.
    gradN : np.ndarray
        (n_gp, 3, 8) shape-function spatial gradients ``J_inv @ dN_nat``.
    """

    B: np.ndarray
    detJ: np.ndarray
    gradN: np.ndarray


class DegenerateElementError(ValueError):
    """Raised when an element has a non-positive Jacobian determinant.

    Subclassed from ``ValueError`` so that defensive ``except ValueError``
    handlers still catch it, while letting callers that want to react to
    this specific failure mode catch it precisely (analogous to
    ``CScanSchemaError`` in ``bvidfe.damage.io``).
    """


def _validate_jacobian(
    detJ: float, xi: float, eta: float, zeta: float, node_coords: np.ndarray
) -> None:
    """Raise DegenerateElementError if the Jacobian determinant is non-positive.

    A non-positive ``detJ`` means the element is either inverted (negative
    determinant — node ordering wrong, or the element has folded over itself)
    or singular (zero determinant — two or more nodes coincide). Either case
    silently produces NaN/Inf in ``np.linalg.inv(J)`` and corrupts every
    downstream stiffness assembly. Catch it loudly here so users get a
    reproducible mesh-quality error instead of opaque NaNs in their results.
    """
    if detJ <= 0:
        raise DegenerateElementError(
            f"Hex8 element Jacobian non-positive: detJ={detJ:.3e} at "
            f"natural coords (xi, eta, zeta)=({xi:g}, {eta:g}, {zeta:g}). "
            f"Node coordinates: {node_coords.tolist()}."
        )


# Node natural coordinates (xi, eta, zeta)
_NODE_COORDS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=float,
)


def _T_sigma_z(theta_rad: float) -> np.ndarray:
    """Voigt stress transformation matrix for rotation about the z-axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    T = np.array(
        [
            [c * c, s * s, 0, 0, 0, 2 * s * c],
            [s * s, c * c, 0, 0, 0, -2 * s * c],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [-s * c, s * c, 0, 0, 0, c * c - s * s],
        ],
        dtype=float,
    )
    return T


class Hex8Element:
    """8-node isoparametric hex element with orthotropic material and optional ply rotation."""

    def __init__(
        self,
        node_coords: np.ndarray,
        material: OrthotropicMaterial,
        ply_angle_deg: float = 0.0,
        b_table: "GeometryTable | None" = None,
    ) -> None:
        node_coords = np.asarray(node_coords, dtype=float)
        if node_coords.shape != (8, 3):
            raise ValueError(f"node_coords must be (8,3), got {node_coords.shape}")
        self.node_coords = node_coords
        self.material = material
        self.ply_angle_deg = ply_angle_deg
        self._C_global = self._compute_global_stiffness()
        # Optional precomputed 2x2x2 Gauss-point geometry table shared across
        # all elements with identical geometry (regular cuboid grid → one table
        # per ply). When set, the Gauss-loop routines skip the per-point
        # B_matrix / Jacobian recomputation. None → legacy per-point path.
        self._b_table = b_table

    def _compute_global_stiffness(self) -> np.ndarray:
        C_mat = self.material.get_stiffness_matrix()
        theta = np.radians(self.ply_angle_deg)
        if abs(theta) < 1e-14:
            return C_mat
        T = _T_sigma_z(theta)
        return T @ C_mat @ T.T

    # --- Shape functions and derivatives ---

    def shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Eight trilinear shape functions at (xi, eta, zeta)."""
        N = np.empty(8)
        for i in range(8):
            xi_i, eta_i, zeta_i = _NODE_COORDS[i]
            N[i] = 0.125 * (1 + xi * xi_i) * (1 + eta * eta_i) * (1 + zeta * zeta_i)
        return N

    def shape_derivatives(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Shape function derivatives d N_i / d {xi, eta, zeta}. Returns (3, 8).

        Vectorised over the 8 nodes — ~5x faster than the equivalent Python loop
        because all per-node arithmetic is done by numpy on 8-element arrays.
        """
        return _shape_derivatives(xi, eta, zeta)

    def jacobian(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """3x3 Jacobian matrix: J_ij = sum_k dN_k/d(xi_i) * x_k_j."""
        dN = self.shape_derivatives(xi, eta, zeta)
        return dN @ self.node_coords  # (3,8) @ (8,3) = (3,3)

    def B_matrix(self, xi: float, eta: float, zeta: float) -> tuple[np.ndarray, float]:
        """Strain-displacement matrix B (6, 24) and det(J) at (xi, eta, zeta).

        Voigt strain = [e_xx, e_yy, e_zz, 2*e_yz, 2*e_xz, 2*e_xy] (engineering shear).

        Vectorised: no Python loop over the 8 nodes. The B matrix has a regular
        block pattern — the 6x3 per-node block is determined by (Nx, Ny, Nz) =
        dN_phys[:, k]. We fill it in one shot per row of B using numpy slicing.
        """
        dN_nat = self.shape_derivatives(xi, eta, zeta)  # (3, 8)
        B, detJ, _ = _b_at_point(self.node_coords, dN_nat, xi, eta, zeta)
        return B, detJ

    def geometry_table(self) -> GeometryTable:
        """Return the per-Gauss-point geometry table for 2x2x2 quadrature.

        Uses the precomputed shared ``_b_table`` when present (regular-grid
        fast path — identical for every element of a ply), otherwise builds it
        once from this element's own geometry. The per-point arithmetic is the
        exact same code path as ``B_matrix`` / ``geometric_stiffness_matrix``,
        so results are bit-for-bit identical to the legacy per-point form.
        """
        if self._b_table is not None:
            return self._b_table
        return build_geometry_table(self.node_coords)

    def stiffness_matrix(self) -> np.ndarray:
        """Element stiffness 24x24 via 2x2x2 Gauss quadrature."""
        _, wt = gauss_points_hex(order=2)
        tbl = self.geometry_table()
        K = np.zeros((24, 24))
        C = self._C_global
        for ig in range(tbl.B.shape[0]):
            B = tbl.B[ig]
            K += np.dot(B.T, np.dot(C, B)) * (tbl.detJ[ig] * wt[ig])
        return K

    def stress_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Recover Voigt stress (n_gp, 6) at Gauss points from element DOF vector (24,)."""
        tbl = self.geometry_table()
        out = np.empty((tbl.B.shape[0], 6))
        C = self._C_global
        for ig in range(tbl.B.shape[0]):
            eps = tbl.B[ig] @ u_elem
            out[ig] = C @ eps
        return out

    def strain_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Recover Voigt strain (n_gp, 6) at Gauss points from element DOF vector (24,)."""
        tbl = self.geometry_table()
        out = np.empty((tbl.B.shape[0], 6))
        for ig in range(tbl.B.shape[0]):
            out[ig] = tbl.B[ig] @ u_elem
        return out

    def geometric_stiffness_matrix(self, sigma_bar_3x3: np.ndarray) -> np.ndarray:
        """Element geometric (initial-stress) stiffness for a constant pre-stress.

        For a body in equilibrium under a static stress state ``sigma_bar``,
        the second-order strain perturbation associated with an infinitesimal
        displacement increment ``u`` is the nonlinear Lagrangian term
        ``eps_nl = 1/2 * grad(u)^T @ grad(u)``. Its variation contributes a
        geometric stiffness to the linearised buckling eigenproblem
        ``(K + lambda * K_g) phi = 0``:

            K_g = integral_V grad(N)^T @ sigma_bar @ grad(N) dV

        where ``grad(N)`` is the 3x8 matrix of nodal shape-function spatial
        gradients (i.e. ``J_inv @ dN_natural`` at each Gauss point). Each
        node-pair (i, j) contribution is the scalar ``H_ij = grad(N_i)^T @
        sigma_bar @ grad(N_j)``, expanded to the 3x3 nodal DOF block as
        ``H_ij * I_3``. In Kronecker form, ``K_g (24x24) = kron(H, I_3)``,
        which is what this routine assembles via 2x2x2 Gauss quadrature.
        ``K_g`` is symmetric, linear in ``sigma_bar``, and may be indefinite.

        References: Cook §17.7, Bathe §6.8.

        Parameters
        ----------
        sigma_bar_3x3 : np.ndarray
            (3, 3) symmetric Cauchy stress in the global frame, in MPa
            (consistent units with the elastic K). Typically a uniform
            uniaxial pre-stress for plate-buckling problems.

        Returns
        -------
        np.ndarray
            24x24 element geometric stiffness K_g (units MPa * mm^3 = N*mm),
            symmetric. The eigenvalue ``lambda`` of the generalised
            eigenproblem ``K phi = lambda K_g phi`` is the buckling load
            multiplier on ``sigma_bar``.

        Raises
        ------
        ValueError
            If ``sigma_bar_3x3`` is not (3, 3).
        DegenerateElementError
            If any Gauss point has non-positive Jacobian determinant.
        """
        sigma_bar = np.asarray(sigma_bar_3x3, dtype=float)
        if sigma_bar.shape != (3, 3):
            raise ValueError(f"sigma_bar must be (3,3), got {sigma_bar.shape}")

        _, wt = gauss_points_hex(order=2)
        tbl = self.geometry_table()
        Kg = np.zeros((24, 24))

        for ig in range(tbl.gradN.shape[0]):
            gradN = tbl.gradN[ig]  # (3, 8) — d N_k/d{x,y,z}

            # H_{ij} = gradN_i^T @ sigma @ gradN_j is a scalar;
            # each node-pair (i, j) contributes H_ij * I_3 to the 3x3 DOF block.
            # Vectorised: Hmat = gradN.T @ sigma @ gradN  (8, 8)
            Hmat = gradN.T @ sigma_bar @ gradN  # (8, 8)

            # Expand (8, 8) to (24, 24): each entry becomes I_3 * H_ij
            Kg += np.kron(Hmat, np.eye(3)) * (tbl.detJ[ig] * wt[ig])

        return Kg


def _shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Shape function derivatives d N_i / d {xi, eta, zeta}. Returns (3, 8).

    Module-level so the per-ply geometry-table builder reuses the exact same
    arithmetic as ``Hex8Element.shape_derivatives`` (bit-for-bit identical).
    """
    xi_i = _NODE_COORDS[:, 0]  # (8,)
    eta_i = _NODE_COORDS[:, 1]
    zeta_i = _NODE_COORDS[:, 2]
    one_plus_xi = 1 + xi * xi_i
    one_plus_eta = 1 + eta * eta_i
    one_plus_zeta = 1 + zeta * zeta_i
    dN = np.empty((3, 8))
    dN[0] = 0.125 * xi_i * one_plus_eta * one_plus_zeta
    dN[1] = 0.125 * one_plus_xi * eta_i * one_plus_zeta
    dN[2] = 0.125 * one_plus_xi * one_plus_eta * zeta_i
    return dN


def _b_at_point(
    node_coords: np.ndarray,
    dN_nat: np.ndarray,
    xi: float,
    eta: float,
    zeta: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Build ``(B (6,24), detJ, gradN (3,8))`` at one natural-coordinate point.

    Single source of truth shared by ``Hex8Element.B_matrix`` and
    ``build_geometry_table`` so the per-element and cached-table paths are
    bit-for-bit identical.
    """
    J = dN_nat @ node_coords  # (3,8) @ (8,3) = (3,3)
    detJ = np.linalg.det(J)
    _validate_jacobian(detJ, xi, eta, zeta, node_coords)
    J_inv = np.linalg.inv(J)
    dN_phys = J_inv @ dN_nat  # (3, 8) — d N_k / d x, d y, d z
    Nx = dN_phys[0]  # (8,)
    Ny = dN_phys[1]
    Nz = dN_phys[2]
    B = np.zeros((6, 24))
    # Column offsets: 0, 3, 6, ..., 21 for the x DOF of each node
    B[0, 0::3] = Nx  # e_xx rows — column = 3k + 0
    B[1, 1::3] = Ny  # e_yy rows — column = 3k + 1
    B[2, 2::3] = Nz  # e_zz rows — column = 3k + 2
    B[3, 1::3] = Nz  # 2*e_yz — column = 3k + 1
    B[3, 2::3] = Ny  # 2*e_yz — column = 3k + 2
    B[4, 0::3] = Nz  # 2*e_xz — column = 3k + 0
    B[4, 2::3] = Nx  # 2*e_xz — column = 3k + 2
    B[5, 0::3] = Ny  # 2*e_xy — column = 3k + 0
    B[5, 1::3] = Nx  # 2*e_xy — column = 3k + 1
    return B, detJ, dN_phys


def build_geometry_table(node_coords: np.ndarray) -> GeometryTable:
    """Build the 2x2x2 Gauss-point geometry table for one element shape.

    For a regular cuboid grid this is identical for every element of a ply,
    so callers compute it once per (ply, ply-thickness) and share it via
    ``Hex8Element(..., b_table=table)``. The per-point arithmetic is the same
    code path (`_b_at_point`) used by ``Hex8Element.B_matrix``, so the cached
    and per-element results are bit-for-bit identical.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    gp, _ = gauss_points_hex(order=2)
    n_gp = gp.shape[0]
    B_all = np.empty((n_gp, 6, 24))
    detJ_all = np.empty(n_gp)
    gradN_all = np.empty((n_gp, 3, 8))
    for ig in range(n_gp):
        xi, eta, zeta = gp[ig]
        dN_nat = _shape_derivatives(xi, eta, zeta)
        B, detJ, gradN = _b_at_point(node_coords, dN_nat, xi, eta, zeta)
        B_all[ig] = B
        detJ_all[ig] = detJ
        gradN_all[ig] = gradN
    return GeometryTable(B=B_all, detJ=detJ_all, gradN=gradN_all)

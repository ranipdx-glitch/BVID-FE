"""Displacement-controlled boundary conditions applied via penalty method.

Standard patterns:
- uniaxial_x_bcs: clamp x_min, prescribe u_x at x_max (+ symmetry on y_min,
  z_min) plus a ``boundary``-dependent u_z edge restraint. The sign of
  ``applied_strain`` selects compression (negative) or tension (positive)
  — the in-plane BC topology is identical either way.
- apply_dirichlet_penalty: multiply K[i,i] by penalty and F[i] = penalty * value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp


@dataclass
class BoundaryCondition:
    """A single prescribed-displacement boundary condition on DOF `dof`, value `value`."""

    dof: int
    value: float


def apply_dirichlet_penalty(
    K: sp.spmatrix,
    F: np.ndarray,
    bcs: Sequence[BoundaryCondition],
    penalty: float = 1.0e10,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Apply Dirichlet BCs to (K, F) via the penalty method.

    Returns a new (K_mod, F_mod) without mutating the inputs.

    The diagonal is updated value-only: every DOF is self-coupled in an
    assembled FE stiffness matrix, so ``K[d, d]`` already exists and adding
    ``penalty`` via ``K + sp.diags(p)`` introduces no new sparsity (avoids
    the SparseEfficiencyWarning storm of scalar ``K_csr[d, d] = ...``
    assignment). Numerically identical to the previous per-DOF loop.
    """
    n_dof = K.shape[0]
    F_out = F.copy()
    bc_dofs = np.fromiter((bc.dof for bc in bcs), dtype=int, count=len(bcs))
    bc_values = np.fromiter((bc.value for bc in bcs), dtype=float, count=len(bcs))

    p = np.zeros(n_dof)
    # np.add.at accumulates duplicates, matching the loop's repeated
    # K_csr[d, d] += penalty / F_out[d] += penalty * value for repeated DOFs.
    np.add.at(p, bc_dofs, penalty)
    np.add.at(F_out, bc_dofs, penalty * bc_values)

    K_mod = (K + sp.diags(p, format="csc")).tocsc()
    return K_mod, F_out


def _nodes_on_plane(
    node_coords: np.ndarray, axis: int, coord: float, tol: float = 1e-9
) -> np.ndarray:
    """Return indices of nodes with coordinate along axis close to `coord`."""
    return np.where(np.abs(node_coords[:, axis] - coord) < tol)[0]


def uniaxial_x_bcs(
    node_coords: np.ndarray,
    applied_strain: float,
    boundary: str = "simply_supported",
) -> List[BoundaryCondition]:
    """Build BCs for a uniaxial test along x (compression or tension).

    The sign of ``applied_strain`` is the only difference between the
    compression and tension load cases — the constrained-DOF topology is
    identical, so a single builder serves both:

    - x_min nodes: u_x = 0 (clamped)
    - x_max nodes: u_x = applied_strain * Lx  (negative = compression,
      positive = tension)
    - y_min nodes: u_y = 0 (symmetry)
    - z_min nodes: u_z = 0 (symmetry)

    ``boundary`` adds the same panel-edge out-of-plane (u_z) restraint the
    fe3d buckling path applies, so the GUI boundary selector has a visible
    effect on the FPF/TAI residual (#32):

    - ``"simply_supported"`` : pin u_z on the two loaded edges (x_min, x_max)
    - ``"clamped"``          : pin u_z on all four lateral edges
    - ``"free"``             : no extra edge restraint

    An unrecognised value is treated as ``"simply_supported"`` (matching the
    fe3d buckling path's fallback).
    """
    Lx = node_coords[:, 0].max() - node_coords[:, 0].min()
    xmin_nodes = _nodes_on_plane(node_coords, 0, node_coords[:, 0].min())
    xmax_nodes = _nodes_on_plane(node_coords, 0, node_coords[:, 0].max())
    ymin_nodes = _nodes_on_plane(node_coords, 1, node_coords[:, 1].min())
    zmin_nodes = _nodes_on_plane(node_coords, 2, node_coords[:, 2].min())

    bcs: List[BoundaryCondition] = []
    for n in xmin_nodes:
        bcs.append(BoundaryCondition(dof=3 * int(n) + 0, value=0.0))
    for n in xmax_nodes:
        bcs.append(BoundaryCondition(dof=3 * int(n) + 0, value=applied_strain * Lx))
    for n in ymin_nodes:
        bcs.append(BoundaryCondition(dof=3 * int(n) + 1, value=0.0))
    for n in zmin_nodes:
        bcs.append(BoundaryCondition(dof=3 * int(n) + 2, value=0.0))

    if boundary != "free":
        edge_nodes: set[int] = set()
        edge_nodes.update(int(n) for n in xmin_nodes)
        edge_nodes.update(int(n) for n in xmax_nodes)
        if boundary == "clamped":
            ymax_nodes = _nodes_on_plane(node_coords, 1, node_coords[:, 1].max())
            edge_nodes.update(int(n) for n in ymin_nodes)
            edge_nodes.update(int(n) for n in ymax_nodes)
        for n in sorted(edge_nodes):
            bcs.append(BoundaryCondition(dof=3 * n + 2, value=0.0))
    return bcs

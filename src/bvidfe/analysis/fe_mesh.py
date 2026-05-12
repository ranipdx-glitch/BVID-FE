"""Structured hex mesh builder for the 3D FE tier.

Generates a regular brick mesh of the full panel with per-element ply
assignment and damage-aware stiffness reduction. The stiffness-reduction
approach (per-element factors inside delamination footprints) is a
simplification of zero-thickness cohesive surfaces and is adequate for
linear buckling in v0.2.0. True cohesive surfaces are deferred.

Damage is represented by TWO per-element factors applied component-wise
to the 6x6 elasticity matrix in the global Voigt frame
[xx, yy, zz, yz, xz, xy] = indices [0, 1, 2, 3, 4, 5]:

  - `damage_factors[e]`            — out-of-plane (OOP) factor in (0, 1].
                                     Scales the OOP block (rows/cols 2,
                                     3, 4) and the in-plane / OOP
                                     Poisson cross-coupling. Reduced
                                     inside delamination footprints to
                                     DAMAGE_OOP_FACTOR.

  - `in_plane_damage_factors[e]`   — in-plane factor in (0, 1]. Scales
                                     the in-plane sub-block on rows/cols
                                     {0, 1, 5}. Reduced ONLY inside the
                                     fiber-break core (within
                                     `fiber_break_radius_mm` of any
                                     delamination centroid) to
                                     DAMAGE_FIBER_BREAK_INPLANE_FACTOR;
                                     delamination-only zones leave it
                                     at 1.0.

This separates two physically distinct damage mechanisms. Pure
delamination loses the interlaminar bond, so through-thickness coupling
(E33, G13, G23, and the E1-E3 / E2-E3 Poisson terms) drops sharply, but
the plies themselves are intact and in-plane load-carrying is preserved
(O'Brien 1982; Pavier & Clarke 1995). Inside the fiber-break core under
the impact site, fibers are broken in addition to interlaminar
separation, so in-plane stiffness is also reduced (Camanho & Davila
2007; Maimi et al. 2007).

The previous unified `DAMAGE_STIFFNESS_FACTOR = 0.30` (applied uniformly
to every entry of `_C_global` in `fe_tier._build_elements`) over-penalised
in-plane stiffness in delaminated zones — its docstring already noted
that "the plies themselves are intact (so in-plane load-carrying is
mostly preserved)", but the code did not honour that. The two-factor
model below makes the model match the docstring intent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from bvidfe.analysis.config import AnalysisConfig, MeshParams
from bvidfe.damage.state import DamageState, DelaminationEllipse

# Out-of-plane stiffness fraction at delaminated interfaces. Representative
# of the post-delamination through-thickness / transverse-shear modulus
# loss reported in Bolotin (2001) review and Sun & Tao (1998).
DAMAGE_OOP_FACTOR = 0.05

# In-plane stiffness fraction inside the fiber-break core (matrix-crushed,
# fibers broken). Representative of the fiber-direction damage-saturation
# values in Camanho & Davila 2007 / Maimi et al. 2007 for CFRP.
DAMAGE_FIBER_BREAK_INPLANE_FACTOR = 0.30


def estimate_fe_mesh_size(config: AnalysisConfig) -> dict:
    """Return a dict with n_elements, n_nodes, n_dof for a would-be fe_mesh build.

    Use this to warn the user BEFORE running the analysis."""
    mesh = config.mesh if config.mesh is not None else MeshParams()
    n_plies = len(config.layup_deg)
    nx = max(1, math.ceil(config.panel.Lx_mm / mesh.in_plane_size_mm))
    ny = max(1, math.ceil(config.panel.Ly_mm / mesh.in_plane_size_mm))
    nz = n_plies * mesh.elements_per_ply
    n_elements = nx * ny * nz
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n_dof = 3 * n_nodes
    return {
        "n_elements": n_elements,
        "n_nodes": n_nodes,
        "n_dof": n_dof,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }


@dataclass
class FeMesh:
    """Structured hex mesh with per-element ply/damage metadata."""

    node_coords: np.ndarray  # (n_nodes, 3)
    element_connectivity: np.ndarray  # (n_elements, 8) node indices
    element_dof_maps: List[np.ndarray]  # length n_elements, each (24,)
    ply_indices: np.ndarray  # (n_elements,) int
    ply_angles_deg: np.ndarray  # (n_elements,) float
    damage_factors: np.ndarray  # (n_elements,) out-of-plane factor in (0, 1]
    in_plane_damage_factors: np.ndarray  # (n_elements,) in-plane factor in (0, 1]
    n_nodes: int = field(init=False)
    n_elements: int = field(init=False)
    n_dof: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_nodes = self.node_coords.shape[0]
        self.n_elements = self.element_connectivity.shape[0]
        self.n_dof = 3 * self.n_nodes


def _point_in_ellipse(x: float, y: float, ellipse: DelaminationEllipse) -> bool:
    """Return True if (x, y) is inside the ellipse footprint."""
    c = math.cos(math.radians(-ellipse.orientation_deg))
    s = math.sin(math.radians(-ellipse.orientation_deg))
    dx = x - ellipse.centroid_mm[0]
    dy = y - ellipse.centroid_mm[1]
    xr = c * dx - s * dy
    yr = s * dx + c * dy
    return (xr / ellipse.major_mm) ** 2 + (yr / ellipse.minor_mm) ** 2 <= 1.0


def build_fe_mesh(config: AnalysisConfig, damage: DamageState) -> FeMesh:
    """Build a structured brick mesh for the panel with per-element metadata."""
    panel = config.panel
    layup = config.layup_deg
    h = config.ply_thickness_mm
    n_plies = len(layup)
    mesh = config.mesh if config.mesh is not None else MeshParams()

    Lx, Ly, Lz = panel.Lx_mm, panel.Ly_mm, n_plies * h
    nx = max(1, math.ceil(Lx / mesh.in_plane_size_mm))
    ny = max(1, math.ceil(Ly / mesh.in_plane_size_mm))
    nz = n_plies * mesh.elements_per_ply

    # Nodes on a regular grid
    x_nodes = np.linspace(0.0, Lx, nx + 1)
    y_nodes = np.linspace(0.0, Ly, ny + 1)
    z_nodes = np.linspace(0.0, Lz, nz + 1)
    node_coords = np.array(
        [[x, y, z] for z in z_nodes for y in y_nodes for x in x_nodes],
        dtype=float,
    )

    def _node_id(i: int, j: int, k: int) -> int:
        """Flat index into node_coords. i along x, j along y, k along z."""
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

    # Build element connectivity (Abaqus hex convention)
    connectivity = np.zeros((nx * ny * nz, 8), dtype=int)
    ply_indices = np.zeros(nx * ny * nz, dtype=int)
    ply_angles = np.zeros(nx * ny * nz, dtype=float)
    damage_factors = np.ones(nx * ny * nz, dtype=float)
    in_plane_damage_factors = np.ones(nx * ny * nz, dtype=float)
    element_dof_maps: List[np.ndarray] = []

    elem_idx = 0
    for k in range(nz):
        ply_i = k // mesh.elements_per_ply
        for j in range(ny):
            for i in range(nx):
                nodes_this_element = [
                    _node_id(i, j, k),
                    _node_id(i + 1, j, k),
                    _node_id(i + 1, j + 1, k),
                    _node_id(i, j + 1, k),
                    _node_id(i, j, k + 1),
                    _node_id(i + 1, j, k + 1),
                    _node_id(i + 1, j + 1, k + 1),
                    _node_id(i, j + 1, k + 1),
                ]
                connectivity[elem_idx] = nodes_this_element
                ply_indices[elem_idx] = ply_i
                ply_angles[elem_idx] = layup[ply_i]
                dof_map = np.array(
                    [3 * n + d for n in nodes_this_element for d in range(3)],
                    dtype=int,
                )
                element_dof_maps.append(dof_map)

                # Compute element centroid for damage check
                cx = 0.5 * (x_nodes[i] + x_nodes[i + 1])
                cy = 0.5 * (y_nodes[j] + y_nodes[j + 1])
                cz_top = z_nodes[k + 1]
                cz_bot = z_nodes[k]

                # Check delamination overlap: element straddles an interface at
                # z = (iface + 1) * h if cz_bot < z_iface < cz_top AND
                # (cx, cy) is inside the ellipse. Reduces only the OOP factor;
                # in-plane stiffness is preserved (plies still intact).
                for ell in damage.delaminations:
                    z_iface = (ell.interface_index + 1) * h
                    if cz_bot <= z_iface <= cz_top:
                        if _point_in_ellipse(cx, cy, ell):
                            damage_factors[elem_idx] = DAMAGE_OOP_FACTOR
                            break

                # Fiber-break core: any element within fiber_break_radius of
                # any delamination centroid (all through-thickness layers).
                # Reduces both factors — fibers broken means in-plane loss too.
                if damage.fiber_break_radius_mm > 0:
                    for ell in damage.delaminations:
                        dx = cx - ell.centroid_mm[0]
                        dy = cy - ell.centroid_mm[1]
                        if math.sqrt(dx * dx + dy * dy) <= damage.fiber_break_radius_mm:
                            damage_factors[elem_idx] = DAMAGE_OOP_FACTOR
                            in_plane_damage_factors[elem_idx] = DAMAGE_FIBER_BREAK_INPLANE_FACTOR
                            break

                elem_idx += 1

    return FeMesh(
        node_coords=node_coords,
        element_connectivity=connectivity,
        element_dof_maps=element_dof_maps,
        ply_indices=ply_indices,
        ply_angles_deg=ply_angles,
        damage_factors=damage_factors,
        in_plane_damage_factors=in_plane_damage_factors,
    )

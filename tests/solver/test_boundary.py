import numpy as np
import scipy.sparse as sp

from bvidfe.solver.boundary import (
    BoundaryCondition,
    apply_dirichlet_penalty,
    uniaxial_x_bcs,
)


def test_apply_dirichlet_penalty_enforces_prescribed_value():
    # Small 3-DOF system: u = [u0, u1, u2]
    K = sp.csc_matrix(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float))
    F = np.zeros(3)
    bcs = [BoundaryCondition(dof=0, value=1.5), BoundaryCondition(dof=2, value=-0.5)]
    Kmod, Fmod = apply_dirichlet_penalty(K, F, bcs, penalty=1e10)
    u = sp.linalg.spsolve(Kmod, Fmod)
    assert abs(u[0] - 1.5) < 1e-6
    assert abs(u[2] - (-0.5)) < 1e-6


def test_uniaxial_x_bcs_compression_returns_xmin_and_xmax_bcs():
    # 8 nodes of a unit cube
    node_coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    bcs = uniaxial_x_bcs(node_coords, applied_strain=-0.01)
    # 4 nodes on x_min get u_x=0, 4 nodes on x_max get u_x=-0.01*Lx=-0.01
    xmin_bcs = [b for b in bcs if abs(b.value) < 1e-12]
    xmax_bcs = [b for b in bcs if abs(b.value - (-0.01)) < 1e-6]
    assert len(xmin_bcs) >= 4  # includes symmetry y and z
    assert len(xmax_bcs) == 4


def test_uniaxial_x_bcs_tension_positive_displacement():
    node_coords = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    bcs = uniaxial_x_bcs(node_coords, applied_strain=0.005)
    xmax = [b for b in bcs if abs(b.value - (0.005 * 2)) < 1e-6]
    assert len(xmax) == 4


def _uz_constrained_nodes(bcs):
    """Set of node indices with a u_z=0 constraint (dof % 3 == 2, value 0)."""
    return {b.dof // 3 for b in bcs if b.dof % 3 == 2 and b.value == 0.0}


def test_uniaxial_x_bcs_boundary_adds_nested_uz_edge_restraints():
    """Issue #32: panel.boundary adds the same u_z edge restraint the fe3d
    buckling path uses. On a 3x3x3 grid the constrained-u_z node sets must
    nest strictly free ⊂ simply_supported ⊂ clamped (free = z_min symmetry
    only; simply_supported adds the two loaded x-edges; clamped adds the
    y-edges too)."""
    coords = np.array(
        [[x, y, z] for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0, 2.0) for z in (0.0, 1.0, 2.0)],
        dtype=float,
    )
    free = _uz_constrained_nodes(uniaxial_x_bcs(coords, -0.01, boundary="free"))
    ss = _uz_constrained_nodes(uniaxial_x_bcs(coords, -0.01, boundary="simply_supported"))
    clamped = _uz_constrained_nodes(uniaxial_x_bcs(coords, -0.01, boundary="clamped"))
    assert free < ss < clamped  # strict subset chain
    # free is exactly the z_min symmetry face (9 of the 27 nodes).
    assert free == {i for i, c in enumerate(coords) if c[2] == 0.0}
    # An unrecognised value falls back to simply_supported (mirrors buckling).
    assert _uz_constrained_nodes(uniaxial_x_bcs(coords, -0.01, boundary="bogus")) == ss

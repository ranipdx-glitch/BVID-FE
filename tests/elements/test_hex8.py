import numpy as np
import pytest

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.elements.hex8 import DegenerateElementError, Hex8Element


def _unit_cube_nodes():
    return np.array(
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


def test_shape_functions_partition_of_unity():
    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8Element(_unit_cube_nodes(), m)
    for xi, eta, zeta in [(-0.5, 0.3, 0.1), (0.7, -0.2, 0.8), (0, 0, 0)]:
        N = elem.shape_functions(xi, eta, zeta)
        assert N.shape == (8,)
        assert abs(N.sum() - 1.0) < 1e-12


def test_shape_function_nodes_values():
    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8Element(_unit_cube_nodes(), m)
    # At node 0 corner (-1,-1,-1), N_0 = 1, others = 0
    N = elem.shape_functions(-1, -1, -1)
    assert abs(N[0] - 1.0) < 1e-12
    assert np.allclose(N[1:], 0.0, atol=1e-12)


def test_jacobian_unit_cube_is_half():
    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8Element(_unit_cube_nodes(), m)
    J = elem.jacobian(0, 0, 0)
    assert J.shape == (3, 3)
    # Unit cube (side = 1 mm), natural length = 2, so dx/dxi = 0.5
    assert abs(np.linalg.det(J) - 0.125) < 1e-12  # det(J) = (0.5)^3


def test_B_matrix_shape():
    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8Element(_unit_cube_nodes(), m)
    B, detJ = elem.B_matrix(0, 0, 0)
    assert B.shape == (6, 24)
    assert detJ > 0


def test_stiffness_matrix_shape_and_symmetry():
    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8Element(_unit_cube_nodes(), m)
    K = elem.stiffness_matrix()
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T, atol=1e-6)


def test_stiffness_matrix_positive_definite_after_bc():
    m = MATERIAL_LIBRARY["IM7/8552"]
    elem = Hex8Element(_unit_cube_nodes(), m)
    K = elem.stiffness_matrix()
    # Kill 6 DOF to remove rigid body modes (fix node 0 fully + node 1 yz + node 3 z)
    free = list(range(24))
    fixed = [0, 1, 2, 4, 5, 8]
    for i in sorted(fixed, reverse=True):
        free.remove(i)
    Krr = K[np.ix_(free, free)]
    eigs = np.linalg.eigvalsh(Krr)
    assert eigs.min() > 0


def test_ply_rotation_produces_stiffness_in_laminate_frame():
    m = MATERIAL_LIBRARY["IM7/8552"]
    elem0 = Hex8Element(_unit_cube_nodes(), m, ply_angle_deg=0.0)
    elem90 = Hex8Element(_unit_cube_nodes(), m, ply_angle_deg=90.0)
    # K differs between 0 and 90 because the fiber direction rotates
    K0 = elem0.stiffness_matrix()
    K90 = elem90.stiffness_matrix()
    assert not np.allclose(K0, K90, atol=1.0)


def test_inverted_hex_raises_degenerate_element_error():
    """A hex with reversed top/bottom face has det(J) < 0 — the integration
    would silently produce negative volume and corrupted stiffness without the
    Jacobian guard."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    nodes = _unit_cube_nodes().copy()
    # Swap the top and bottom face — det(J) flips sign
    nodes[[0, 1, 2, 3, 4, 5, 6, 7]] = nodes[[4, 5, 6, 7, 0, 1, 2, 3]]
    elem = Hex8Element(nodes, m)
    with pytest.raises(DegenerateElementError, match="non-positive"):
        elem.B_matrix(0.0, 0.0, 0.0)


def test_singular_hex_raises_degenerate_element_error():
    """A hex with two coincident nodes is singular: det(J) ≈ 0 → raises."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    nodes = _unit_cube_nodes().copy()
    # Collapse top face onto bottom (zero-thickness slab)
    nodes[4:8] = nodes[0:4]
    elem = Hex8Element(nodes, m)
    with pytest.raises(DegenerateElementError, match="non-positive"):
        elem.B_matrix(0.0, 0.0, 0.0)


def test_geometric_stiffness_matrix_also_validates_jacobian():
    """The Kg assembly path goes through its own jacobian/det chain; it must
    also raise on degenerate elements."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    nodes = _unit_cube_nodes().copy()
    nodes[4:8] = nodes[0:4]  # collapsed top → singular J
    elem = Hex8Element(nodes, m)
    sigma_bar = np.diag([1.0, 0.0, 0.0])
    with pytest.raises(DegenerateElementError, match="non-positive"):
        elem.geometric_stiffness_matrix(sigma_bar)


def test_degenerate_element_error_is_a_value_error():
    """Defensive code that catches generic ValueError must still see the new
    error class — we do not want to break existing exception handlers."""
    assert issubclass(DegenerateElementError, ValueError)

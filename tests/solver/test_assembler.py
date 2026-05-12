import numpy as np
import pytest
import scipy.sparse as sp

from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.elements.hex8 import Hex8Element
from bvidfe.solver.assembler import assemble_global_stiffness


def _single_element_system():
    m = MATERIAL_LIBRARY["IM7/8552"]
    nodes = np.array(
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
    elem = Hex8Element(nodes, m)
    dof_map = np.arange(24)  # 24 DOFs for 8 nodes
    return [elem], [dof_map], 24


def _two_element_system():
    """Two unit hex elements sharing the face x=1."""
    m = MATERIAL_LIBRARY["IM7/8552"]
    # 12 nodes total; first cube occupies x in [0,1], second in [1,2]
    n_nodes = 12
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
            [2, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
            [2, 1, 1],
        ],
        dtype=float,
    )
    # Element 1 nodes indices
    e1_nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    # Element 2 nodes indices (shares nodes 1,2,5,6)
    e2_nodes = [1, 8, 9, 2, 5, 10, 11, 6]
    elem1 = Hex8Element(node_coords[e1_nodes], m)
    elem2 = Hex8Element(node_coords[e2_nodes], m)

    # Global DOF index arrays (24 each)
    def _dof(ni):
        return np.array([3 * n + k for n in ni for k in range(3)])

    return [elem1, elem2], [_dof(e1_nodes), _dof(e2_nodes)], 3 * n_nodes


def test_assemble_single_element_shape():
    elems, dofs, n_dof = _single_element_system()
    K = assemble_global_stiffness(elems, dofs, n_dof)
    assert sp.issparse(K)
    assert K.shape == (24, 24)
    assert K.nnz > 0


def test_assemble_symmetric():
    elems, dofs, n_dof = _two_element_system()
    K = assemble_global_stiffness(elems, dofs, n_dof)
    diff = K - K.T
    assert np.allclose(diff.data, 0.0, atol=1e-6)


def test_assemble_six_rigid_body_modes():
    elems, dofs, n_dof = _two_element_system()
    K = assemble_global_stiffness(elems, dofs, n_dof)
    # Convert to dense for eigenvalue analysis (small system)
    Kd = K.toarray()
    eigs = np.linalg.eigvalsh(Kd)
    # 6 rigid-body zero eigenvalues expected (within numerical tolerance)
    near_zero = eigs[np.abs(eigs) < 1e-3]
    assert len(near_zero) == 6


def test_assemble_summation_of_overlapping_dofs():
    """Shared-DOF entries in K must equal the sum of each element's local
    contribution — issue #48. A regression that overwrites instead of
    accumulates (e.g. CSR construction that dedups duplicate (row, col)
    pairs) would still produce a positive-definite K, so the previous
    eigenvalue-only check was insufficient.
    """
    elems, dofs, n_dof = _two_element_system()
    K = assemble_global_stiffness(elems, dofs, n_dof).toarray()

    elem1, elem2 = elems
    K1, K2 = elem1.stiffness_matrix(), elem2.stiffness_matrix()
    dof1, dof2 = dofs

    # Pick a global DOF that is shared by both elements (node 1 -> x DOF = 3).
    shared_global_dof = 3
    loc1 = int(np.where(dof1 == shared_global_dof)[0][0])
    loc2 = int(np.where(dof2 == shared_global_dof)[0][0])

    # Diagonal entry must sum.
    assert K[shared_global_dof, shared_global_dof] == pytest.approx(
        K1[loc1, loc1] + K2[loc2, loc2], rel=1e-12
    )

    # Off-diagonal between two DOFs shared by both elements must also sum.
    # Node 1 x-DOF -> global 3; node 2 x-DOF -> global 6 (both shared).
    other_global_dof = 6
    loc1b = int(np.where(dof1 == other_global_dof)[0][0])
    loc2b = int(np.where(dof2 == other_global_dof)[0][0])
    assert K[shared_global_dof, other_global_dof] == pytest.approx(
        K1[loc1, loc1b] + K2[loc2, loc2b], rel=1e-12
    )

    # Entry for a DOF in element 1 only must NOT include any element-2
    # contribution (sanity-check that the summation is targeted).
    elem1_only_global_dof = 0  # node 0 x-DOF, not in element 2
    loc1c = int(np.where(dof1 == elem1_only_global_dof)[0][0])
    assert K[elem1_only_global_dof, elem1_only_global_dof] == pytest.approx(
        K1[loc1c, loc1c], rel=1e-12
    )

"""Sparse global stiffness assembly from element contributions."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp


def assemble_coo(
    element_dof_maps: Sequence[np.ndarray],
    n_dof: int,
    matrix_fn: Callable[[int], np.ndarray],
) -> sp.csc_matrix:
    """Assemble a global sparse matrix from per-element 24x24 contributions.

    Shared core for both the elastic stiffness (``assemble_global_stiffness``)
    and the geometric stiffness (assembled inline by the buckling path in
    ``analysis.fe_tier``). The two had drifted into independent
    list-of-chunks COO assemblers; this helper centralises the COO build.

    Memory: row/col/data buffers are pre-allocated as a single contiguous
    ``np.empty(n_elem * 576)`` each rather than appended-and-concatenated
    per element. On a problem at the documented ``FE3D_MAX_DOF`` cap
    (~166k elements) this halves the transient memory held alive before
    ``coo_matrix.tocsc()`` — issue #33.

    Parameters
    ----------
    element_dof_maps : sequence of int arrays, each length 24, giving the
        global DOF index for each of the 24 element DOFs.
    n_dof : total number of global DOFs.
    matrix_fn : callable ``(e_idx) -> (24, 24) ndarray`` returning the per-
        element contribution. Indexed by element position in
        ``element_dof_maps``.

    Returns
    -------
    scipy.sparse.csc_matrix of shape ``(n_dof, n_dof)``.
    """
    n_elem = len(element_dof_maps)
    if n_elem == 0:
        return sp.csc_matrix((n_dof, n_dof))

    # 24 * 24 = 576 entries per element. One pre-allocated contiguous
    # buffer per array — no list-of-chunks overhead, no concatenate copy.
    block = 576
    total = n_elem * block
    rows = np.empty(total, dtype=np.int32)
    cols = np.empty(total, dtype=np.int32)
    data = np.empty(total, dtype=np.float64)

    for e_idx, dof_map in enumerate(element_dof_maps):
        dof_arr = np.asarray(dof_map, dtype=np.int32)
        if dof_arr.shape != (24,):
            raise ValueError(f"dof_map must have 24 entries, got {len(dof_map)}")
        Ke = matrix_fn(e_idx)
        if Ke.shape != (24, 24):
            raise ValueError(f"element matrix must be (24,24), got {Ke.shape}")
        start = e_idx * block
        end = start + block
        # Outer product of dof indices → 24x24 row/col index arrays, written
        # directly into the pre-allocated slice (no temporary lists).
        rows[start:end] = np.broadcast_to(dof_arr[:, None], (24, 24)).ravel()
        cols[start:end] = np.broadcast_to(dof_arr[None, :], (24, 24)).ravel()
        data[start:end] = Ke.ravel()

    K = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsc()


def assemble_global_stiffness(
    elements: Sequence,
    element_dof_maps: Sequence[np.ndarray],
    n_dof: int,
) -> sp.csc_matrix:
    """Assemble the global elastic stiffness matrix in CSC format.

    Thin wrapper around :func:`assemble_coo` preserved for backwards
    compatibility — call sites that already exist (``solver.static``,
    ``analysis.fe_tier``, and the test suite) continue to work
    unchanged.

    Parameters
    ----------
    elements : sequence of elements each exposing
        ``.stiffness_matrix() -> (24, 24)``.
    element_dof_maps : sequence of int arrays, each length 24.
    n_dof : total number of global DOFs.
    """
    if len(elements) != len(element_dof_maps):
        raise ValueError(
            f"elements (n={len(elements)}) and element_dof_maps "
            f"(n={len(element_dof_maps)}) length mismatch"
        )
    elements_seq = elements

    def _ke(e_idx: int) -> np.ndarray:
        return elements_seq[e_idx].stiffness_matrix()

    return assemble_coo(element_dof_maps, n_dof, _ke)

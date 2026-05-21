"""Linear buckling eigenvalue solver.

Solves the symmetric generalized eigenproblem:
    K . phi = lambda * K_g . phi

Returns the `n_modes` smallest positive eigenvalues and their mode shapes.
Uses scipy's `eigsh` with shift-invert (sigma=0) for robust small-eigenvalue
extraction in large sparse systems.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def linear_buckling(
    K: sp.spmatrix,
    Kg: sp.spmatrix,
    n_modes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve K phi = lambda Kg phi for the `n_modes` smallest eigenvalues.

    Parameters
    ----------
    K : (n, n) sparse symmetric positive-definite elastic stiffness.
    Kg : (n, n) sparse symmetric geometric stiffness (may be indefinite).
    n_modes : number of smallest eigenvalues to return (default 3).

    Returns
    -------
    eigenvalues : ndarray of shape (n_modes,), sorted ascending.
    mode_shapes : ndarray of shape (n, n_modes) — one mode per column.
    """
    n = K.shape[0]
    if K.shape != Kg.shape:
        raise ValueError(f"K shape {K.shape} != Kg shape {Kg.shape}")
    if n_modes <= 0 or n_modes > n:
        raise ValueError(f"n_modes must satisfy 0 < n_modes <= {n}, got {n_modes}")

    K_csc = K.tocsc()
    Kg_csc = Kg.tocsc()

    # For small systems, the shift-invert machinery can be slow/unstable.
    # Fall back to dense solver if n is small.
    if n < 50:
        K_dense = K_csc.toarray()
        Kg_dense = Kg_csc.toarray()
        eigs_all, modes_all = _dense_generalized_eigh(K_dense, Kg_dense)
        order = np.argsort(eigs_all)
        eigs = eigs_all[order[:n_modes]]
        modes = modes_all[:, order[:n_modes]]
        return eigs, modes

    # Large sparse path: shift-invert with sigma=0.
    # A deterministic v0 keeps ARPACK reproducible across runs. Without it,
    # scipy seeds from numpy's global random state, so test ordering (or any
    # upstream consumer of np.random) can flip ARPACK's convergence path on
    # problems with nearly-degenerate buckling modes.
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n)
    eigs, modes = eigsh(K_csc, k=n_modes, M=Kg_csc, sigma=0.0, which="LM", v0=v0)
    order = np.argsort(eigs)
    return eigs[order], modes[:, order]


def _dense_generalized_eigh(K: np.ndarray, Kg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense fallback using scipy.linalg.eigh(K, Kg). Symmetric generalized."""
    from scipy.linalg import eigh

    eigs, modes = eigh(K, Kg)
    return eigs, modes

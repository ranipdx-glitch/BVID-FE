"""3D finite-element tier for BVID-FE.

v0.2.0: Primary CAI path uses geometric-stiffness-based linear buckling.
First-ply-failure on damaged mesh is retained as a fallback / comparison.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List

import numpy as np
import scipy.sparse as sp

from bvidfe.analysis.config import AnalysisConfig
from bvidfe.analysis.fe_mesh import FeMesh, build_fe_mesh, estimate_fe_mesh_size
from bvidfe.core.laminate import Laminate
from bvidfe.damage.state import DamageState
from bvidfe.elements.hex8 import Hex8Element, build_geometry_table
from bvidfe.failure.larc05 import larc05_index, larc05_index_batch
from bvidfe.failure.tsai_wu import tsai_wu_index, tsai_wu_index_batch
from bvidfe.solver.assembler import assemble_global_stiffness
from bvidfe.solver.boundary import (
    BoundaryCondition,
    apply_dirichlet_penalty,
    uniaxial_x_bcs,
)
from bvidfe.solver.buckling import linear_buckling
from bvidfe.solver.static import solve_linear_static

# Configure a module-level logger that writes to stderr. Users launching
# the GUI from a terminal (or running the CLI / Python API) will see
# one log line per FE stage, making long runs observable.
_log = logging.getLogger("bvidfe.fe3d")


def _resolve_log_level(default: str = "INFO") -> int:
    """Resolve BVIDFE_LOG_LEVEL → numeric level, falling back to ``default``
    with a stderr warning when the value is not a recognised level name.

    Without this guard a typo (``BVIDFE_LOG_LEVEL=DUBG``) raised ValueError
    at module import time and prevented the package from being imported.
    """
    raw = os.environ.get("BVIDFE_LOG_LEVEL", default).upper()
    level = logging.getLevelName(raw)
    if isinstance(level, int):
        return level
    sys.stderr.write(
        f"[bvidfe] BVIDFE_LOG_LEVEL={raw!r} is not a valid log level "
        f"(use DEBUG/INFO/WARNING/ERROR/CRITICAL); defaulting to {default}.\n"
    )
    return logging.getLevelName(default)


def _resolve_max_dof(default: int = 500000) -> int:
    """Resolve BVIDFE_FE3D_MAX_DOF → positive int, falling back to ``default``
    with a stderr warning when the value is not a positive integer."""
    raw = os.environ.get("BVIDFE_FE3D_MAX_DOF")
    if raw is None:
        return default
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError(f"must be > 0 (got {v})")
        return v
    except ValueError as exc:
        sys.stderr.write(
            f"[bvidfe] BVIDFE_FE3D_MAX_DOF={raw!r} is not a positive int "
            f"({exc}); defaulting to {default}.\n"
        )
        return default


if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[bvidfe.fe3d %(asctime)s] %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(_resolve_log_level())


def _t(msg: str, t0: float) -> None:
    """Log a timing line: message + seconds since t0."""
    _log.info("%s (%.2fs)", msg, time.time() - t0)


# Hard cap on fe3d problem size. Beyond ~500k DOFs the pure-Python assembler
# + scipy sparse LU factorization start risking memory exhaustion and native-
# code crashes (scipy calls into BLAS/SuiteSparse which cannot be caught by
# Python exception handling — an OOM there is a SIGSEGV in the parent process).
# Override via BVIDFE_FE3D_MAX_DOF env var at your own risk; an invalid
# value falls back to the default with a stderr warning instead of raising
# at import time.
FE3D_MAX_DOF: int = _resolve_max_dof()


class FE3DSizeError(RuntimeError):
    """Raised when an fe3d problem exceeds the safe-size cap."""


def _guard_problem_size(cfg: AnalysisConfig) -> None:
    """Raise FE3DSizeError if the mesh would exceed the safe-size cap.

    Called at the top of fe3d_cai_buckling / fe3d_cai / fe3d_tai so callers
    get a clean Python exception instead of a native-code crash.
    """
    stats = estimate_fe_mesh_size(cfg)
    if stats["n_dof"] > FE3D_MAX_DOF:
        raise FE3DSizeError(
            f"fe3d problem too large: {stats['n_elements']:,} elements / "
            f"{stats['n_dof']:,} DOFs exceeds the safe-size cap of "
            f"{FE3D_MAX_DOF:,} DOFs. Increase MeshParams.in_plane_size_mm, "
            f"decrease elements_per_ply, or switch to tier='empirical' / "
            f"tier='semi_analytical'. Override the cap via the "
            f"BVIDFE_FE3D_MAX_DOF environment variable at your own risk."
        )


# Voigt indices: [xx, yy, zz, yz, xz, xy] = [0, 1, 2, 3, 4, 5]
# In-plane components live on rows/cols {0, 1, 5}; out-of-plane on {2, 3, 4}.
# After rotation about z (Hex8Element.ply_angle_deg) these blocks remain
# decoupled in C_global, so the mask below applies cleanly in the global
# frame. See fe_mesh.DAMAGE_OOP_FACTOR / DAMAGE_FIBER_BREAK_INPLANE_FACTOR
# docstrings for the physical model.
_INPLANE_VOIGT = np.array([0, 1, 5])


def _build_elements(mesh: FeMesh, lam: Laminate) -> List[Hex8Element]:
    """Build a Hex8Element for each mesh element with component-wise damage scaling.

    Two factors are applied to the 6x6 elasticity matrix:
      - `mesh.damage_factors[e]` scales OOP and in-plane/OOP cross terms
      - `mesh.in_plane_damage_factors[e]` scales the in-plane sub-block
    """
    elements: List[Hex8Element] = []
    # On the regular cuboid grid produced by build_fe_mesh, every element of a
    # given ply is a pure translate of every other, so its 2x2x2 Gauss-point
    # (B, detJ, gradN) table is identical. Compute it once per ply (keyed by
    # ply index AND element geometry — the latter keeps mixed-thickness plies,
    # issue #5/PR#13, correct) and share it across that ply's elements.
    geom_cache: dict[tuple, object] = {}
    for eidx in range(mesh.n_elements):
        node_ids = mesh.element_connectivity[eidx]
        node_coords = mesh.node_coords[node_ids]
        mat = lam.material
        ply_angle = float(mesh.ply_angles_deg[eidx])
        ply_i = int(mesh.ply_indices[eidx])
        dims = node_coords.max(axis=0) - node_coords.min(axis=0)
        cache_key = (
            ply_i,
            round(float(dims[0]), 9),
            round(float(dims[1]), 9),
            round(float(dims[2]), 9),
        )
        b_table = geom_cache.get(cache_key)
        if b_table is None:
            b_table = build_geometry_table(node_coords)
            geom_cache[cache_key] = b_table
        elem = Hex8Element(node_coords, mat, ply_angle_deg=ply_angle, b_table=b_table)
        f_oop = float(mesh.damage_factors[eidx])
        f_ip = float(mesh.in_plane_damage_factors[eidx])
        # Build a 6x6 element-wise scaling mask: start from f_oop everywhere
        # (OOP block + Poisson cross-coupling), then overwrite the 3x3
        # in-plane sub-block on rows/cols {0, 1, 5} with f_ip.
        mask = np.full((6, 6), f_oop)
        mask[np.ix_(_INPLANE_VOIGT, _INPLANE_VOIGT)] = f_ip
        elem._C_global = elem._C_global * mask
        elements.append(elem)
    return elements


def _resolve_material(cfg: AnalysisConfig):
    if isinstance(cfg.material, str):
        from bvidfe.core.material import MATERIAL_LIBRARY

        return MATERIAL_LIBRARY[cfg.material]
    return cfg.material


def _fe3d_preflight(
    cfg: AnalysisConfig,
    damage: DamageState,
    lam: Laminate,
    *,
    label: str = "fe3d",
) -> tuple[FeMesh, List[Hex8Element], float]:
    """Run the size guard, build the damaged mesh, and build elements.

    Factors the duplicated pre-flight prologue used by every fe3d entry
    point (``fe3d_cai_buckling``, ``_fe3d_cai_first_ply_failure``,
    ``fe3d_tai``). The ``label`` differentiates the per-stage log line so
    a stderr trace still distinguishes "fe3d FPF" from "fe3d buckling".

    Returns
    -------
    (mesh, elements, t0)
        ``t0`` is ``time.time()`` captured before the mesh build, suitable
        for relative timing logs in the caller.
    """
    _guard_problem_size(cfg)
    t0 = time.time()
    _log.info("%s: mesh build start", label)
    mesh = build_fe_mesh(cfg, damage)
    _t(f"mesh build done: {mesh.n_elements} elements, {mesh.n_dof} DOFs", t0)
    elements = _build_elements(mesh, lam)
    _t("elements built", t0)
    return mesh, elements, t0


def _solve_failure_strain_analytic(
    cfg: AnalysisConfig,
    mesh: FeMesh,
    elements: List[Hex8Element],
    strain_sign: int,
    criterion: str,
    strain_cap: float = 0.05,
) -> float:
    """Find the applied strain magnitude at which max failure index hits 1,
    using a single FE solve + analytic scaling.

    Linear elasticity: stress field scales linearly with applied strain. So if
    we solve once at a reference strain `eps_ref`, the stress at any strain
    multiplier `c` is just `c * sigma_ref`. Tsai-Wu is quadratic in stress:
        idx(c) = c * (F . sigma_ref) + c^2 * (sigma_ref . F_ij . sigma_ref)
    At the critical strain, idx = 1; positive root of the quadratic.

    LaRC05 is piecewise-quadratic (tensile vs. compressive modes branch on
    the sign of s1 / s2) but each branch is still quadratic in c for a fixed
    stress direction, so we evaluate all four mode branches at a reference
    strain and pick the binding one analytically.

    This replaces the prior 10-12 iteration bisection (each iteration
    reassembled and re-solved the full FE system), giving a ~10x speedup
    for the FPF path with identical results on the quadratic branches.

    The per-element inner Gauss-point loop is vectorised via the batch
    criterion helpers (``larc05_index_batch`` / ``tsai_wu_index_batch``).
    Numerical equivalence with the prior scalar form is locked by
    ``tests/analysis/test_fpf_strain_solve_equivalence.py`` against the
    ``_solve_failure_strain_analytic_scalar_ref`` reference implementation
    kept in this module purely for the equivalence test.
    """
    material = _resolve_material(cfg)

    # One FE solve at reference strain = strain_sign * strain_cap
    bcs = uniaxial_x_bcs(mesh.node_coords, strain_sign * strain_cap, boundary=cfg.panel.boundary)
    u_ref = solve_linear_static(elements, mesh.element_dof_maps, mesh.n_dof, bcs)

    c_crit_min = np.inf
    for eidx, elem in enumerate(elements):
        dof_map = mesh.element_dof_maps[eidx]
        sigma_ref = elem.stress_at_gauss_points(u_ref[dof_map])  # (n_gp, 6)

        if criterion == "larc05":
            idx_ref = larc05_index_batch(material, sigma_ref)  # (n_gp,)
            valid = idx_ref > 0
            if not bool(valid.any()):
                continue
            # LaRC05 modes are sums of squared normalised stresses, so
            # idx(c) = c^2 * idx(1)  ->  c_crit = 1 / sqrt(idx_ref).
            c_crit = np.where(valid, 1.0 / np.sqrt(np.where(valid, idx_ref, 1.0)), np.inf)
            c_crit_elem_min = float(c_crit.min())
        else:
            # Tsai-Wu: idx(c) = a*c + b*c^2. Solve a, b from two samples
            # (c=1 and c=2). System: a + b = idx_ref;  2a + 4b = idx_2.
            idx_ref = tsai_wu_index_batch(material, sigma_ref)  # (n_gp,)
            sigma_2 = 2.0 * sigma_ref
            idx_2 = tsai_wu_index_batch(material, sigma_2)  # (n_gp,)
            b = (idx_2 - 2.0 * idx_ref) / 2.0
            a = idx_ref - b
            valid = idx_ref > 0
            tiny = 1e-14
            # Initialise c_crit = +inf so masked-out points never drive the min.
            c_crit = np.full_like(idx_ref, np.inf, dtype=float)
            # Branch 1: |b| < tiny and |a| > tiny -> c = 1/a.
            mask_lin = valid & (np.abs(b) < tiny) & (np.abs(a) >= tiny)
            # Use np.where to keep divisor non-zero for inactive lanes.
            c_lin = 1.0 / np.where(mask_lin, a, 1.0)
            c_crit = np.where(mask_lin, c_lin, c_crit)
            # Branch 2: |b| >= tiny and disc >= 0 -> c = (-a + sqrt(disc)) / (2b).
            mask_quad = valid & (np.abs(b) >= tiny)
            disc = a * a + 4.0 * b
            mask_quad_real = mask_quad & (disc >= 0)
            sqrt_disc = np.sqrt(np.where(mask_quad_real, disc, 0.0))
            c_quad = (-a + sqrt_disc) / np.where(mask_quad_real, 2.0 * b, 1.0)
            c_crit = np.where(mask_quad_real, c_quad, c_crit)
            # Filter c_crit > 0 (Tsai-Wu can produce a non-physical
            # negative root when the linear and quadratic terms have
            # opposite signs in a way that doesn't bracket failure).
            c_crit = np.where(c_crit > 0, c_crit, np.inf)
            c_crit_elem_min = float(c_crit.min())

        if c_crit_elem_min < c_crit_min:
            c_crit_min = c_crit_elem_min

    if not np.isfinite(c_crit_min) or c_crit_min <= 0:
        return strain_cap  # nothing failed up to strain_cap
    return min(strain_cap, c_crit_min * strain_cap)


def _solve_failure_strain_analytic_scalar_ref(
    cfg: AnalysisConfig,
    mesh: FeMesh,
    elements: List[Hex8Element],
    strain_sign: int,
    criterion: str,
    strain_cap: float = 0.05,
) -> float:
    """Reference (pre-vectorisation) scalar implementation of
    ``_solve_failure_strain_analytic``.

    Kept intact for the equivalence test in
    ``tests/analysis/test_fpf_strain_solve_equivalence.py`` so any future
    edit to the production routine immediately surfaces a numerical
    drift. NOT called from production code; do not use directly.
    """
    material = _resolve_material(cfg)

    bcs = uniaxial_x_bcs(mesh.node_coords, strain_sign * strain_cap, boundary=cfg.panel.boundary)
    u_ref = solve_linear_static(elements, mesh.element_dof_maps, mesh.n_dof, bcs)

    c_crit_min = np.inf
    for eidx, elem in enumerate(elements):
        dof_map = mesh.element_dof_maps[eidx]
        sigma_field_ref = elem.stress_at_gauss_points(u_ref[dof_map])
        for gp in range(sigma_field_ref.shape[0]):
            sigma_ref = sigma_field_ref[gp]
            if criterion == "larc05":
                idx_ref = larc05_index(material, sigma_ref)
            else:
                idx_ref = tsai_wu_index(material, sigma_ref)
            if idx_ref <= 0:
                continue
            if criterion == "larc05":
                c_crit = 1.0 / np.sqrt(idx_ref)
            else:
                sigma_2 = 2.0 * sigma_ref
                idx_2 = tsai_wu_index(material, sigma_2)
                b = (idx_2 - 2.0 * idx_ref) / 2.0
                a = idx_ref - b
                if abs(b) < 1e-14:
                    if abs(a) < 1e-14:
                        continue
                    c_crit = 1.0 / a
                else:
                    disc = a * a + 4.0 * b
                    if disc < 0:
                        continue
                    c_crit = (-a + np.sqrt(disc)) / (2.0 * b)
                if c_crit <= 0:
                    continue
            if c_crit < c_crit_min:
                c_crit_min = c_crit

    if not np.isfinite(c_crit_min) or c_crit_min <= 0:
        return strain_cap
    return min(strain_cap, c_crit_min * strain_cap)


def _effective_modulus(lam: Laminate) -> float:
    """Effective in-plane Young's modulus along x (Ex from CLT)."""
    Ex, _, _, _ = lam.effective_engineering_constants()
    return Ex


def _fe3d_cai_first_ply_failure(
    cfg: AnalysisConfig,
    damage: DamageState,
    lam: Laminate,
    sigma_pristine_MPa: float,
) -> float:
    """3D FE compression-after-impact residual strength via first-ply-failure (MPa).

    Original v0.1.0 implementation — bisects on applied strain until LaRC05 failure
    index reaches 1 on the damaged mesh. Retained as fallback / comparison path.
    """
    mesh, elements, t0 = _fe3d_preflight(cfg, damage, lam, label="fe3d FPF")
    strain_at_failure = _solve_failure_strain_analytic(
        cfg,
        mesh,
        elements,
        strain_sign=-1,
        criterion="larc05",
    )
    _t("FPF analytic solve done", t0)
    E = _effective_modulus(lam)
    sigma = strain_at_failure * E
    _log.info(
        "fe3d FPF done: residual = %.1f MPa (total %.2fs)",
        min(sigma, sigma_pristine_MPa),
        time.time() - t0,
    )
    return min(sigma, sigma_pristine_MPa)


def fe3d_cai_buckling(
    cfg: AnalysisConfig,
    damage: DamageState,
    lam: Laminate,
    sigma_pristine_MPa: float,
    sigma_ref_MPa: float = 1.0,
) -> tuple[float, float, List[str]]:
    """3D FE compression-after-impact via true linear buckling eigensolve.

    Assembles K and K_g under a constant uniaxial pre-stress sigma_ref along x
    (scaled by per-element damage factor), then solves K phi = lambda K_g phi for
    the smallest positive eigenvalue. Critical buckling stress = lambda * sigma_ref.

    Uses the "constant pre-stress" approximation (Cook §17.7 / Bathe §6.8):
    - sigma_0 = sigma_ref_MPa along x everywhere (unit reference compression).
    - Damaged elements carry in_plane_damage_factor * sigma_0 (reduced load
      fraction in fiber-break-core elements; pure-delamination elements carry
      the full uniaxial load because the plies remain intact).
    - K_g is assembled element-by-element from Hex8Element.geometric_stiffness_matrix().
    - A minimal penalty-BC set suppresses rigid-body modes before the eigensolve.

    Returns
    -------
    (sigma_critical_MPa, lambda_crit, notes)
        sigma_critical_MPa : min(lambda_crit * sigma_ref, sigma_pristine_MPa)
        lambda_crit        : smallest positive buckling load factor (0 if solve failed)
        notes              : list of human-readable diagnostic strings emitted
                             during this call (e.g. eigensolver failure /
                             no-positive-eigenvalue fallback). Empty when the
                             solve was clean. Surfaced via
                             ``AnalysisResults.notes``.
    """
    mesh, elements, t0 = _fe3d_preflight(cfg, damage, lam, label="fe3d buckling")

    # Assemble elastic stiffness K
    K = assemble_global_stiffness(elements, mesh.element_dof_maps, mesh.n_dof)
    _t("K assembled", t0)

    # Assemble geometric stiffness K_g under uniform uniaxial pre-stress sigma_ref along x.
    # The pre-stress is sigma_xx (Voigt index 0, an in-plane component), so the
    # damaged-element load fraction is the IN-PLANE damage factor — pure
    # delamination zones still carry full uniaxial load (in_plane_factor ≈ 1.0)
    # because the plies remain intact; only fiber-break-core elements carry less.
    sigma_bar_ref = np.zeros((3, 3))
    sigma_bar_ref[0, 0] = sigma_ref_MPa

    # Vectorised COO assembly of K_g (same pattern as assembler.py)
    rows_chunks: list[np.ndarray] = []
    cols_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    for eidx, elem in enumerate(elements):
        damage_f = float(mesh.in_plane_damage_factors[eidx])
        sigma_elem = sigma_bar_ref * damage_f
        Kg_e = elem.geometric_stiffness_matrix(sigma_elem)
        dof_arr = np.asarray(mesh.element_dof_maps[eidx], dtype=np.int64)
        rows_chunks.append(np.broadcast_to(dof_arr[:, None], (24, 24)).ravel())
        cols_chunks.append(np.broadcast_to(dof_arr[None, :], (24, 24)).ravel())
        data_chunks.append(Kg_e.ravel())
    rows = np.concatenate(rows_chunks) if rows_chunks else np.array([], dtype=np.int64)
    cols = np.concatenate(cols_chunks) if cols_chunks else np.array([], dtype=np.int64)
    data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=float)

    Kg = sp.coo_matrix((data, (rows, cols)), shape=(K.shape[0], K.shape[0])).tocsc()
    _t("Kg assembled", t0)

    # Apply penalty BCs to K (not Kg) — rigid-body suppression plus
    # boundary-dependent lateral restraint so the GUI's boundary selector
    # produces a visible effect on the buckling eigenvalue.
    n_dof = K.shape[0]
    bcs = [
        BoundaryCondition(dof=0, value=0.0),
        BoundaryCondition(dof=1, value=0.0),
        BoundaryCondition(dof=2, value=0.0),
        BoundaryCondition(dof=4, value=0.0),
        BoundaryCondition(dof=5, value=0.0),
        BoundaryCondition(dof=7, value=0.0),
    ]
    # Boundary-dependent out-of-plane (u_z) restraint on the panel edges:
    #   simply_supported : pin u_z on the two loaded edges (x_min, x_max)
    #   clamped          : pin u_z on all four lateral edges
    #   free             : no additional restraint (only rigid-body)
    boundary = cfg.panel.boundary
    if boundary != "free":
        coords = mesh.node_coords
        tol = 1e-6
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        loaded_edge_mask = (np.abs(coords[:, 0] - x_min) < tol) | (
            np.abs(coords[:, 0] - x_max) < tol
        )
        if boundary == "clamped":
            lateral_edge_mask = (
                loaded_edge_mask
                | (np.abs(coords[:, 1] - y_min) < tol)
                | (np.abs(coords[:, 1] - y_max) < tol)
            )
        else:  # simply_supported
            lateral_edge_mask = loaded_edge_mask
        for node_idx in np.where(lateral_edge_mask)[0]:
            # u_z is DOF 3*node_idx + 2
            bcs.append(BoundaryCondition(dof=3 * int(node_idx) + 2, value=0.0))
    F_dummy = np.zeros(n_dof)
    K_mod, _ = apply_dirichlet_penalty(K, F_dummy, bcs, penalty=1.0e10)

    # Solve generalised eigenproblem K phi = lambda K_g phi for smallest positive eig.
    try:
        n_req = min(6, n_dof - 1)
        eigs, _ = linear_buckling(K_mod, Kg, n_modes=n_req)
        _t(f"eigsh returned {len(eigs)} values", t0)
        positive_eigs = [float(e) for e in eigs if e > 1e-6]
        if not positive_eigs:
            _log.info("fe3d buckling: no positive eigenvalue, returning pristine")
            note = (
                "fe3d buckling: eigensolver returned no positive eigenvalue; "
                "fell back to pristine strength (knockdown=1.0 may not reflect "
                "actual damage effect)"
            )
            return sigma_pristine_MPa, 0.0, [note]
        lambda_crit = min(positive_eigs)
    except Exception as exc:  # noqa: BLE001
        _log.warning("fe3d buckling eigensolve failed: %s", exc)
        note = (
            f"fe3d buckling: eigensolver raised {type(exc).__name__}: {exc}; "
            "fell back to pristine strength (knockdown=1.0 does not reflect "
            "actual damage effect)"
        )
        return sigma_pristine_MPa, 0.0, [note]

    sigma_critical = lambda_crit * sigma_ref_MPa
    _log.info(
        "fe3d buckling done: lambda_crit=%.4g sigma_crit=%.1f MPa (total %.2fs)",
        lambda_crit,
        sigma_critical,
        time.time() - t0,
    )
    return min(sigma_critical, sigma_pristine_MPa), lambda_crit, []


def fe3d_cai(
    cfg: AnalysisConfig,
    damage: DamageState,
    lam: Laminate,
    sigma_pristine_MPa: float,
) -> float:
    """3D FE compression-after-impact residual strength (MPa).

    Primary path: true linear buckling eigensolve (fe3d_cai_buckling).
    Fallback: first-ply-failure on damaged mesh (_fe3d_cai_first_ply_failure).
    Returns the smaller of the two — whichever failure mode governs.

    Notes from the buckling solve are discarded by this convenience wrapper;
    callers that need them should invoke fe3d_cai_buckling directly (as
    BvidAnalysis.run does).
    """
    sigma_buckling, _lambda_crit, _notes = fe3d_cai_buckling(cfg, damage, lam, sigma_pristine_MPa)
    sigma_fpf = _fe3d_cai_first_ply_failure(cfg, damage, lam, sigma_pristine_MPa)
    return min(sigma_buckling, sigma_fpf)


def fe3d_tai(
    cfg: AnalysisConfig,
    damage: DamageState,
    lam: Laminate,
    sigma_pristine_MPa: float,
) -> float:
    """3D FE tension-after-impact residual strength (MPa)."""
    mesh, elements, _t0 = _fe3d_preflight(cfg, damage, lam, label="fe3d TAI")
    strain_at_failure = _solve_failure_strain_analytic(
        cfg,
        mesh,
        elements,
        strain_sign=+1,
        criterion="tsai_wu",
    )
    E = _effective_modulus(lam)
    sigma = strain_at_failure * E
    return min(sigma, sigma_pristine_MPa)

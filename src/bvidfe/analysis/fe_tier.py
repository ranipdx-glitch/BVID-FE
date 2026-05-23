"""3D finite-element tier for BVID-FE.

The CAI residual is the minimum of two channels:
- a first-ply-failure analysis on the damaged 3D mesh (this module's
  distinctive value: 3D stress states with per-element damage factors);
- a buckling stress from the Rayleigh-Ritz closed form (delegated to
  :mod:`bvidfe.analysis.semi_analytical`, #129).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List

import numpy as np

from bvidfe._types import CriterionName
from bvidfe.analysis.config import AnalysisConfig
from bvidfe.analysis.fe_mesh import FeMesh, build_fe_mesh, estimate_fe_mesh_size
from bvidfe.core.laminate import Laminate
from bvidfe.damage.state import DamageState
from bvidfe.elements.hex8 import Hex8Element, build_geometry_table
from bvidfe.failure.larc05 import larc05_index, larc05_index_batch
from bvidfe.failure.tsai_wu import tsai_wu_index, tsai_wu_index_batch
from bvidfe.analysis.semi_analytical import (
    find_critical_interface,
    panel_buckling_load,
    sublaminate_buckling_load,
)
from bvidfe.solver.boundary import uniaxial_x_bcs
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
    criterion: CriterionName,
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
    criterion: CriterionName,
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
    criterion: CriterionName = "larc05",
) -> float:
    """3D FE compression-after-impact residual strength via first-ply-failure (MPa).

    Original v0.1.0 implementation — bisects on applied strain until the selected
    failure criterion's index reaches 1 on the damaged mesh. Retained as
    fallback / comparison path. Default ``criterion="larc05"`` preserves the
    historical behaviour; pass ``"tsai_wu"`` to use the polynomial criterion.
    """
    mesh, elements, t0 = _fe3d_preflight(cfg, damage, lam, label="fe3d FPF")
    strain_at_failure = _solve_failure_strain_analytic(
        cfg,
        mesh,
        elements,
        strain_sign=-1,
        criterion=criterion,
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
    """3D FE compression-after-impact buckling channel — closed-form delegation.

    Issue #129: the previous 3D K_g eigensolve under-predicted by ~100x on
    realistic panels because the constant-prestress + 3-2-1 rigid-body BC
    formulation didn't actually enforce a compressive state, and plain Hex8
    locks too aggressively for thin laminates at practical mesh sizes.
    Investigation also showed that even a proper static-prestress
    reformulation did not converge to closed-form within an order of
    magnitude. The Rayleigh-Ritz closed form (Timoshenko & Gere §9.2;
    Reddy 4.4.4) is exact for SSSS orthotropic rectangles, so the buckling
    channel now delegates to it via
    :func:`bvidfe.analysis.semi_analytical.panel_buckling_load`.

    For damaged panels the sublaminate-over-delamination check from
    :func:`bvidfe.analysis.semi_analytical.sublaminate_buckling_load` also
    runs; the minimum of the full-panel and worst-sublaminate buckling
    stress is returned. fe3d's distinctive value is still in the FPF
    channel (3D stress states + per-element damage factors), which runs
    independently.
    """
    boundary = cfg.panel.boundary
    notes: List[str] = []

    t_total = float(sum(lam.ply_thicknesses_mm))
    N_cr_panel = panel_buckling_load(lam, cfg.panel.Lx_mm, cfg.panel.Ly_mm, boundary)
    sigma_critical = N_cr_panel / t_total if t_total > 0 else float("inf")

    if damage.delaminations:
        crit_idx = find_critical_interface(damage, lam)
        if crit_idx is not None:
            ellipses = [e for e in damage.delaminations if e.interface_index == crit_idx]
            critical_ellipse = max(ellipses, key=lambda e: e.area_mm2)
            N_cr_sub = sublaminate_buckling_load(lam, critical_ellipse, boundary=boundary)
            thicknesses = lam.ply_thicknesses_mm
            upper_t = sum(thicknesses[: crit_idx + 1])
            lower_t = sum(thicknesses[crit_idx + 1 :])
            sub_t = min(upper_t, lower_t)
            if sub_t > 0:
                sigma_sublam = N_cr_sub / sub_t
                sigma_critical = min(sigma_critical, sigma_sublam)

    if not np.isfinite(sigma_critical) or sigma_critical <= 0:
        note = (
            "fe3d buckling: closed-form Rayleigh-Ritz returned a degenerate "
            "result; fell back to pristine strength (knockdown=1.0 may not "
            "reflect actual damage effect)"
        )
        return sigma_pristine_MPa, 0.0, [note]

    lambda_crit = sigma_critical / sigma_ref_MPa
    _log.info(
        "fe3d buckling: sigma_crit=%.1f MPa (closed-form Rayleigh-Ritz)",
        sigma_critical,
    )
    return min(sigma_critical, sigma_pristine_MPa), lambda_crit, notes


def fe3d_cai(
    cfg: AnalysisConfig,
    damage: DamageState,
    lam: Laminate,
    sigma_pristine_MPa: float,
) -> float:
    """3D FE compression-after-impact residual strength (MPa).

    The smaller of the buckling channel (``fe3d_cai_buckling``, delegated
    to the Rayleigh-Ritz closed form per #129) and the first-ply-failure
    channel on the damaged mesh (``_fe3d_cai_first_ply_failure``).

    Notes from the buckling channel are discarded by this convenience
    wrapper; callers that need them should invoke ``fe3d_cai_buckling``
    directly (as ``BvidAnalysis.run`` does).
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

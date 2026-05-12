"""fe3d sanity check: pristine in-plane stiffness recovers CLT.

A pristine laminate (no delaminations, no fiber-break core) loaded under
uniaxial extension on a small panel should recover an effective Young's
modulus E_x_FE that agrees with the closed-form CLT value E_x_CLT to
within the discretisation error of the coarse mesh used here. The
agreement validates the entire fe3d stack -- mesh build, element
construction, assembly, BC application, sparse solve, and stress
recovery -- against a known analytical baseline on a problem where the
3D model degenerates exactly to plane stress.

Mesh size is intentionally small (~600-1000 DOFs) so the test runs in
a few seconds and stays well within the BVIDFE_FE3D_MAX_DOF cap.
"""

from __future__ import annotations

import numpy as np
import pytest

from bvidfe.analysis import AnalysisConfig, MeshParams
from bvidfe.analysis.fe_mesh import build_fe_mesh
from bvidfe.analysis.fe_tier import _build_elements
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.damage.state import DamageState
from bvidfe.impact.mapping import ImpactEvent
from bvidfe.solver.boundary import BoundaryCondition
from bvidfe.solver.static import solve_linear_static


def _pristine_cfg():
    return AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0.0, 90.0, 90.0, 0.0],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(20.0, 10.0),
        loading="tension",
        tier="fe3d",
        impact=ImpactEvent(1.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=2.5),
    )


@pytest.fixture(scope="module")
def fe3d_setup():
    """Build a pristine fe3d mesh + element list once for the whole module."""
    cfg = _pristine_cfg()
    lam = Laminate(
        material=MATERIAL_LIBRARY[cfg.material],
        layup_deg=cfg.layup_deg,
        ply_thickness_mm=cfg.ply_thickness_mm,
    )
    mesh = build_fe_mesh(cfg, DamageState(delaminations=[], dent_depth_mm=0.0))
    elements = _build_elements(mesh, lam)
    return cfg, lam, mesh, elements


def test_pristine_mesh_has_no_damage_factors(fe3d_setup):
    """Sanity: a pristine input must produce an entirely undamaged mesh."""
    _cfg, _lam, mesh, _elements = fe3d_setup
    assert np.all(mesh.damage_factors == 1.0)
    assert np.all(mesh.in_plane_damage_factors == 1.0)


def test_pristine_fe3d_recovers_clt_extensional_modulus(fe3d_setup):
    """Apply a small uniaxial strain ex on the right edge and recover the
    average sigma_xx; (avg_sigma_xx / strain) must agree with CLT E_x to
    within the coarse-mesh discretisation tolerance (~10%).

    The asymmetric BC pattern (clamp left edge u_x, release y/z elsewhere
    aside from rigid-body suppression) drops the test from a full plane-
    stress problem to a 1D extension; the CLT prediction (A11 / h) is
    then exact and the FE answer should match to within mesh error.
    """
    cfg, lam, mesh, elements = fe3d_setup

    # CLT reference
    A, _, _ = lam.abd_matrices()
    E_x_clt = A[0, 0] / lam.thickness_mm  # MPa

    # Build BCs: clamp x_min plane in u_x, prescribe u_x = strain * Lx on
    # x_max plane. Suppress rigid-body translation in y and z by pinning
    # one node. Suppress rotation by pinning the orthogonal DOFs at two
    # additional nodes.
    coords = mesh.node_coords
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    Lx = x_max - x_min
    strain = 1.0e-3
    u_right = strain * Lx
    tol = 1e-9

    bcs: list[BoundaryCondition] = []
    # Clamp every node on x_min in the x direction
    left_mask = np.abs(coords[:, 0] - x_min) < tol
    for n in np.where(left_mask)[0]:
        bcs.append(BoundaryCondition(dof=3 * int(n) + 0, value=0.0))
    # Prescribe u_x on every node on x_max
    right_mask = np.abs(coords[:, 0] - x_max) < tol
    for n in np.where(right_mask)[0]:
        bcs.append(BoundaryCondition(dof=3 * int(n) + 0, value=u_right))
    # Anchor the laminate against rigid-body translation in y, z and
    # rotation by pinning a corner node and one of its neighbours.
    bcs.append(BoundaryCondition(dof=1, value=0.0))  # y at node 0
    bcs.append(BoundaryCondition(dof=2, value=0.0))  # z at node 0
    bcs.append(BoundaryCondition(dof=5, value=0.0))  # z at node 1 (locks rotation about y)

    u = solve_linear_static(elements, mesh.element_dof_maps, mesh.n_dof, bcs)

    # Recover sigma_xx at every Gauss point of every element and take the
    # volume-average (Gauss weights for 2x2x2 hex are equal so a simple
    # average suffices for the volume mean of an unstructured-coord brick).
    s_xx_sum = 0.0
    n_gp_total = 0
    for elem, dof_map in zip(elements, mesh.element_dof_maps):
        u_elem = u[dof_map]
        stresses = elem.stress_at_gauss_points(u_elem)  # (8, 6)
        s_xx_sum += float(stresses[:, 0].sum())
        n_gp_total += stresses.shape[0]
    avg_sigma_xx = s_xx_sum / n_gp_total

    # Effective modulus: sigma / strain
    E_x_fe = avg_sigma_xx / strain

    # 10% tolerance on a coarse 5x4x4 brick mesh is generous but
    # appropriate; CLT is exact, FE has Hex8 trilinear discretisation
    # error of ~5% at this resolution.
    assert E_x_fe == pytest.approx(
        E_x_clt, rel=0.10
    ), f"E_x_FE = {E_x_fe:.1f} MPa, E_x_CLT = {E_x_clt:.1f} MPa"


def test_pristine_static_solve_completes_without_warnings(fe3d_setup):
    """A pristine extension solve must run cleanly — no degenerate
    elements, no singular K. Exists as a separate test so a failure
    here points at infrastructure rather than at a numerical drift."""
    cfg, lam, mesh, elements = fe3d_setup
    # Trivial BC: pin three nodes' u_x to suppress rigid-body x-translation
    # and apply zero force everywhere. Should produce u == 0.
    bcs = [
        BoundaryCondition(dof=0, value=0.0),
        BoundaryCondition(dof=1, value=0.0),
        BoundaryCondition(dof=2, value=0.0),
        BoundaryCondition(dof=4, value=0.0),
        BoundaryCondition(dof=5, value=0.0),
        BoundaryCondition(dof=8, value=0.0),
    ]
    u = solve_linear_static(elements, mesh.element_dof_maps, mesh.n_dof, bcs)
    assert np.all(np.isfinite(u))
    assert np.linalg.norm(u) < 1e-6  # zero-load solution

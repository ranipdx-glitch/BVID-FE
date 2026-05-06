import numpy as np

from bvidfe.analysis.config import MeshParams
from bvidfe.analysis.fe_mesh import build_fe_mesh
from bvidfe.damage.state import DamageState, DelaminationEllipse


def _simple_config():
    from bvidfe.analysis.config import AnalysisConfig
    from bvidfe.core.geometry import PanelGeometry
    from bvidfe.impact.mapping import ImpactEvent
    from bvidfe.core.geometry import ImpactorGeometry

    return AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],  # 4 plies
        ply_thickness_mm=0.2,
        panel=PanelGeometry(20.0, 10.0),
        loading="compression",
        tier="fe3d",
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=2, in_plane_size_mm=2.0),
    )


def test_mesh_node_and_element_counts_are_consistent():
    cfg = _simple_config()
    mesh = build_fe_mesh(cfg, DamageState([], dent_depth_mm=0.0))
    # nx = 20/2 = 10, ny = 10/2 = 5, nz = 4 plies * 2 = 8
    assert mesh.n_elements == 10 * 5 * 8
    assert mesh.n_nodes == 11 * 6 * 9
    # DOF count = 3 * nodes
    assert mesh.n_dof == 3 * mesh.n_nodes


def test_mesh_ply_assignment_consistent():
    cfg = _simple_config()
    mesh = build_fe_mesh(cfg, DamageState([], dent_depth_mm=0.0))
    # All elements should have ply_index in [0, n_plies-1]
    assert mesh.ply_indices.min() == 0
    assert mesh.ply_indices.max() == 3


def test_mesh_damage_factor_defaults_to_one_for_pristine():
    cfg = _simple_config()
    mesh = build_fe_mesh(cfg, DamageState([], dent_depth_mm=0.0))
    assert np.all(mesh.damage_factors == 1.0)
    assert np.all(mesh.in_plane_damage_factors == 1.0)


def test_mesh_damage_factor_reduced_inside_ellipse():
    cfg = _simple_config()
    # Damage at interface 1 (between ply 1 and ply 2), large ellipse covering panel center
    ds = DamageState(
        [DelaminationEllipse(1, (10, 5), 8, 4, 0)],
        dent_depth_mm=0.3,
    )
    mesh = build_fe_mesh(cfg, ds)
    # Some elements should have reduced stiffness
    n_damaged = (mesh.damage_factors < 1.0).sum()
    assert n_damaged > 0


def test_mesh_delamination_only_preserves_in_plane_factor():
    """Pure delamination (no fiber-break radius) reduces only the OOP factor."""
    from bvidfe.analysis.fe_mesh import DAMAGE_OOP_FACTOR

    cfg = _simple_config()
    ds = DamageState(
        [DelaminationEllipse(1, (10, 5), 8, 4, 0)],
        dent_depth_mm=0.3,
        fiber_break_radius_mm=0.0,
    )
    mesh = build_fe_mesh(cfg, ds)
    damaged = mesh.damage_factors < 1.0
    assert damaged.any(), "expected at least one damaged element"
    assert np.allclose(mesh.damage_factors[damaged], DAMAGE_OOP_FACTOR)
    # In-plane factor must stay at 1.0 in delamination-only zones
    assert np.all(mesh.in_plane_damage_factors == 1.0)


def test_mesh_fiber_break_core_reduces_in_plane_factor():
    """Inside the fiber-break radius, both factors are reduced."""
    from bvidfe.analysis.fe_mesh import (
        DAMAGE_FIBER_BREAK_INPLANE_FACTOR,
        DAMAGE_OOP_FACTOR,
    )

    cfg = _simple_config()
    ds = DamageState(
        [DelaminationEllipse(1, (10, 5), 8, 4, 0)],
        dent_depth_mm=0.3,
        fiber_break_radius_mm=3.0,  # carve out a fiber-break core at the centroid
    )
    mesh = build_fe_mesh(cfg, ds)
    fiber_break = mesh.in_plane_damage_factors < 1.0
    assert fiber_break.any(), "expected at least one fiber-break-core element"
    assert np.allclose(
        mesh.in_plane_damage_factors[fiber_break],
        DAMAGE_FIBER_BREAK_INPLANE_FACTOR,
    )
    # Fiber-break-core elements must also have OOP reduced
    assert np.all(mesh.damage_factors[fiber_break] == DAMAGE_OOP_FACTOR)


def test_mesh_element_dof_maps_cover_24_dofs_each():
    cfg = _simple_config()
    mesh = build_fe_mesh(cfg, DamageState([], dent_depth_mm=0.0))
    for dof_map in mesh.element_dof_maps:
        assert len(dof_map) == 24
        assert all(0 <= d < mesh.n_dof for d in dof_map)


def test_mesh_node_coordinates_span_panel():
    cfg = _simple_config()
    mesh = build_fe_mesh(cfg, DamageState([], dent_depth_mm=0.0))
    assert mesh.node_coords[:, 0].min() == 0.0
    assert abs(mesh.node_coords[:, 0].max() - 20.0) < 1e-9
    assert abs(mesh.node_coords[:, 1].max() - 10.0) < 1e-9


def test_estimate_fe_mesh_size_returns_sensible_counts():
    from bvidfe.analysis.config import AnalysisConfig, MeshParams
    from bvidfe.analysis.fe_mesh import estimate_fe_mesh_size
    from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
    from bvidfe.impact.mapping import ImpactEvent

    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(20, 10),
        loading="compression",
        tier="fe3d",
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=2, in_plane_size_mm=2.0),
    )
    stats = estimate_fe_mesh_size(cfg)
    assert stats["n_elements"] == 10 * 5 * 8
    assert stats["n_dof"] == 3 * 11 * 6 * 9


def test_default_mesh_params_are_conservative():
    from bvidfe.analysis.config import MeshParams

    mp = MeshParams()
    # At these defaults a 150x100 panel with 8 plies must stay below 10k elements
    # (so non-expert users don't hang their GUI).
    assert mp.elements_per_ply == 1
    assert mp.in_plane_size_mm >= 5.0

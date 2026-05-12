import pytest

pv = pytest.importorskip("pyvista")

pv.OFF_SCREEN = True

from bvidfe.analysis.config import AnalysisConfig, MeshParams  # noqa: E402
from bvidfe.analysis.fe_mesh import build_fe_mesh  # noqa: E402
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry  # noqa: E402
from bvidfe.damage.state import DamageState, DelaminationEllipse  # noqa: E402
from bvidfe.impact.mapping import ImpactEvent  # noqa: E402
from bvidfe.viz.plots_3d import mesh_to_pyvista, plot_mesh_with_damage  # noqa: E402


def _small_cfg():
    return AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0, 90, 0, 90],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(10, 5),
        loading="compression",
        tier="fe3d",
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=2.5),
    )


def test_mesh_to_pyvista_returns_unstructured_grid():
    cfg = _small_cfg()
    mesh = build_fe_mesh(cfg, DamageState([], dent_depth_mm=0.0))
    grid = mesh_to_pyvista(mesh)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_points == mesh.n_nodes
    assert grid.n_cells == mesh.n_elements


def test_mesh_to_pyvista_has_damage_and_ply_arrays():
    cfg = _small_cfg()
    ds = DamageState([DelaminationEllipse(1, (5, 2.5), 3, 1.5, 0)], dent_depth_mm=0.3)
    mesh = build_fe_mesh(cfg, ds)
    grid = mesh_to_pyvista(mesh)
    assert "damage_factor" in grid.cell_data
    assert "ply_index" in grid.cell_data
    # Some cells should be damaged
    assert (grid.cell_data["damage_factor"] < 1.0).sum() > 0


def test_plot_mesh_with_damage_returns_plotter(tmp_path):
    cfg = _small_cfg()
    ds = DamageState([DelaminationEllipse(1, (5, 2.5), 3, 1.5, 0)], dent_depth_mm=0.3)
    mesh = build_fe_mesh(cfg, ds)
    p = plot_mesh_with_damage(mesh)
    assert isinstance(p, pv.Plotter)
    # Save a screenshot to confirm render pipeline works
    out = tmp_path / "mesh.png"
    p.screenshot(str(out))
    p.close()
    assert out.exists() and out.stat().st_size > 1000

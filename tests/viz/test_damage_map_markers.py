"""Regression tests for the impact-location and fiber-break markers in plot_damage_map.

Before this was added, the damage map showed only the delamination ellipses —
users could not see WHERE the impact happened on the panel, which matters
for off-center impacts. The marker was added because inspection of the
rendered tab revealed the gap.
"""

from __future__ import annotations


from bvidfe.core.geometry import PanelGeometry
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.viz.plots_2d import plot_damage_map


def _make_damage(centroid=(100.0, 50.0), fbr=0.0):
    """Build a minimal 3-interface DamageState with given centroid."""
    ellipses = [
        DelaminationEllipse(
            interface_index=i,
            centroid_mm=centroid,
            major_mm=20.0 + 5 * i,
            minor_mm=12.0 + 3 * i,
            orientation_deg=30.0,
        )
        for i in range(3)
    ]
    return DamageState(delaminations=ellipses, dent_depth_mm=0.4, fiber_break_radius_mm=fbr)


def test_plot_damage_map_has_impact_marker_in_legend():
    """The figure must have a legend entry labelled 'impact' anchored at
    the ellipse centroid."""
    panel = PanelGeometry(200, 100)
    damage = _make_damage(centroid=(100.0, 50.0))
    fig = plot_damage_map(damage, panel)

    # Find the legend (should exist)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None, "damage map has no legend"
    labels = [t.get_text() for t in legend.get_texts()]
    assert "impact" in labels, labels


def test_plot_damage_map_marker_at_centroid():
    """The Line2D for the impact 'x' marker must be located at the
    ellipse centroid coordinates."""
    panel = PanelGeometry(300, 200)
    cx, cy = 75.0, 150.0
    damage = _make_damage(centroid=(cx, cy))
    fig = plot_damage_map(damage, panel)

    ax = fig.axes[0]
    xs, ys = [], []
    for line in ax.get_lines():
        xs.extend(line.get_xdata())
        ys.extend(line.get_ydata())
    assert cx in xs, xs
    assert cy in ys, ys


def test_plot_damage_map_renders_fiber_break_core():
    """When fiber_break_radius_mm > 0, a 'fiber break' legend entry
    appears."""
    panel = PanelGeometry(200, 100)
    damage = _make_damage(centroid=(100.0, 50.0), fbr=3.0)
    fig = plot_damage_map(damage, panel)

    ax = fig.axes[0]
    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert "fiber break" in labels, labels


def test_plot_damage_map_no_marker_when_no_damage():
    """With an empty DamageState there should be no impact marker."""
    panel = PanelGeometry(150, 100)
    damage = DamageState(delaminations=[], dent_depth_mm=0.0, fiber_break_radius_mm=0.0)
    fig = plot_damage_map(damage, panel)

    ax = fig.axes[0]
    legend = ax.get_legend()
    # Either no legend or no 'impact' entry
    if legend is not None:
        labels = [t.get_text() for t in legend.get_texts()]
        assert "impact" not in labels

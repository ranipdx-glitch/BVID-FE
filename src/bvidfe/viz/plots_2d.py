"""2D matplotlib plots for BVID-FE: damage map, knockdown curve, tier comparison.

All plot functions accept an optional ``fig`` argument. When ``fig`` is ``None``
(default) they create a fresh ``Figure`` via ``plt.subplots`` — convenient for
standalone scripts and tests that want to call ``fig.savefig(...)``. When
``fig`` is an existing ``Figure`` (typically ``canvas.figure`` from an embedded
``FigureCanvasQTAgg``) the function clears it in place and draws into it. This
second mode is what the Qt GUI tabs need: assigning ``canvas.figure = new_fig``
causes rendering-state leaks on Qt6/macOS because the new figure's ``canvas``
attribute still points to the pyplot-managed Agg canvas, not the Qt one. Drawing
into the canvas's own figure avoids that problem entirely.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bvidfe.core.geometry import PanelGeometry
from bvidfe.damage.state import DamageState
from bvidfe.viz.style import COLORS, ELLIPSE_CMAP, FIGSIZE_LANDSCAPE, FIGSIZE_SQUARE


def _prepare_axes(fig: Optional[Figure], figsize):
    """Return ``(fig, ax)``: if ``fig`` is None make a fresh one, else clear and
    add a single axes. Used by all plot_* functions so the Qt-embedded path
    never relies on ``plt.subplots``."""
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    fig.clear()
    ax = fig.add_subplot(111)
    return fig, ax


def plot_damage_map(
    damage: DamageState,
    panel: PanelGeometry,
    title: str | None = None,
    fig: Optional[Figure] = None,
):
    """Top-down plan view of ellipse delamination footprints + panel outline.

    Ellipses are color-coded by interface index. Returns the matplotlib Figure.
    If ``fig`` is provided it is cleared and reused (safe for embedded Qt
    canvases); otherwise a new Figure is created via ``plt.subplots``.
    """
    fig, ax = _prepare_axes(fig, FIGSIZE_SQUARE)
    ax.set_xlim(0, panel.Lx_mm)
    ax.set_ylim(0, panel.Ly_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title or f"BVID damage map (dent {damage.dent_depth_mm:.2f} mm)")

    # Panel outline
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0),
            panel.Lx_mm,
            panel.Ly_mm,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )

    if damage.delaminations:
        ifaces = sorted({d.interface_index for d in damage.delaminations})
        cmap = plt.get_cmap(ELLIPSE_CMAP)
        norm = plt.Normalize(vmin=min(ifaces), vmax=max(ifaces) + 1)
        for d in damage.delaminations:
            color = cmap(norm(d.interface_index))
            ax.add_patch(
                mpatches.Ellipse(
                    xy=d.centroid_mm,
                    width=2 * d.major_mm,
                    height=2 * d.minor_mm,
                    angle=d.orientation_deg,
                    facecolor=color,
                    alpha=0.35,
                    edgecolor=color,
                    linewidth=1.0,
                )
            )
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Interface index")

        # Impact-centroid marker: all ellipses in an impact-driven DamageState
        # share the same centroid (the impact point). For inspection-driven
        # DamageStates the centroids can differ; we plot each unique one.
        unique_centroids = {d.centroid_mm for d in damage.delaminations}
        for cx, cy in unique_centroids:
            ax.plot(
                cx,
                cy,
                marker="x",
                markersize=12,
                markeredgewidth=2,
                color="black",
                zorder=10,
                label="impact" if (cx, cy) == next(iter(unique_centroids)) else None,
            )
            # Fiber-break core on the back face: small filled red circle
            if damage.fiber_break_radius_mm > 0:
                ax.add_patch(
                    mpatches.Circle(
                        (cx, cy),
                        damage.fiber_break_radius_mm,
                        facecolor="red",
                        alpha=0.6,
                        edgecolor="darkred",
                        linewidth=1.0,
                        zorder=9,
                        label="fiber break" if (cx, cy) == next(iter(unique_centroids)) else None,
                    )
                )
        ax.legend(loc="upper right", framealpha=0.85)
    else:
        ax.text(
            panel.Lx_mm / 2,
            panel.Ly_mm / 2,
            "no damage",
            ha="center",
            va="center",
            color="grey",
            fontsize=14,
        )

    fig.tight_layout()
    return fig


def plot_knockdown_curve(
    energies_J: Sequence[float],
    knockdowns: Sequence[float],
    tier_label: str = "",
    title: str | None = None,
    fig: Optional[Figure] = None,
):
    """Line plot of knockdown vs impact energy.

    If ``fig`` is provided it is cleared and reused; otherwise a new Figure
    is created.
    """
    fig, ax = _prepare_axes(fig, FIGSIZE_LANDSCAPE)
    color = COLORS.get(tier_label, COLORS["knockdown"])
    ax.plot(energies_J, knockdowns, "-o", color=color, label=tier_label or "knockdown")
    ax.set_xlabel("Impact energy [J]")
    ax.set_ylabel("Strength retention (knockdown) [-]")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    if tier_label:
        ax.legend()
    ax.set_title(title or "BVID knockdown curve")
    fig.tight_layout()
    return fig


def plot_tier_comparison(
    energies_J: Sequence[float],
    results_per_tier: Dict[str, Sequence[float]],
    title: str | None = None,
    fig: Optional[Figure] = None,
):
    """Overlaid knockdown curves for multiple tiers.

    If ``fig`` is provided it is cleared and reused; otherwise a new Figure
    is created.
    """
    fig, ax = _prepare_axes(fig, FIGSIZE_LANDSCAPE)
    for tier, kd in results_per_tier.items():
        color = COLORS.get(tier, None)
        ax.plot(energies_J, kd, "-o", color=color, label=tier)
    ax.set_xlabel("Impact energy [J]")
    ax.set_ylabel("Strength retention (knockdown) [-]")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_title(title or "BVID tier comparison")
    fig.tight_layout()
    return fig

"""Analysis configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from bvidfe.core.geometry import PanelGeometry
from bvidfe.core.material import OrthotropicMaterial
from bvidfe.damage.state import DamageState
from bvidfe.impact.mapping import ImpactEvent


@dataclass
class MeshParams:
    """Mesh resolution parameters for the ``tier="fe3d"`` solve.

    Parameters
    ----------
    elements_per_ply : int
        Number of hex elements stacked through the thickness of each ply
        (dimensionless count, ``>= 1``). Total through-thickness elements
        ``nz = n_plies * elements_per_ply``. Default ``1`` (one element
        per ply) is adequate for first-ply-failure screening; increase
        for smoother through-thickness stress gradients at higher cost.
    in_plane_size_mm : float
        Target in-plane element edge length in millimetres (``> 0``). The
        mesh uses ``nx = ceil(Lx_mm / in_plane_size_mm)`` and
        ``ny = ceil(Ly_mm / in_plane_size_mm)`` elements in x and y, so
        smaller values resolve the stress concentration around the damage
        footprint more finely at quadratically higher element count and
        memory. Default ``5.0`` mm. On Streamlit Cloud (1 GB RAM) keep
        this ``>= 5`` mm.
    cohesive_zone_factor : float
        Reserved (``> 0``, default ``1.0``). Intended as a multiplier on
        the cohesive-zone characteristic length once true cohesive
        surfaces land (see README "Limitations"); **currently validated
        but not consumed** by ``build_fe_mesh`` — the present fe3d tier
        uses a component-wise stiffness-reduction model, not cohesive
        elements. Leave at the default.
    """

    elements_per_ply: int = 1
    in_plane_size_mm: float = 5.0
    cohesive_zone_factor: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.elements_per_ply, int) or self.elements_per_ply <= 0:
            raise ValueError(
                f"MeshParams.elements_per_ply must be a positive int "
                f"(got {self.elements_per_ply!r})"
            )
        if not (self.in_plane_size_mm > 0):
            raise ValueError(
                f"MeshParams.in_plane_size_mm must be > 0 " f"(got {self.in_plane_size_mm!r})"
            )
        if not (self.cohesive_zone_factor > 0):
            raise ValueError(
                f"MeshParams.cohesive_zone_factor must be > 0 "
                f"(got {self.cohesive_zone_factor!r})"
            )


@dataclass
class AnalysisConfig:
    """BVID analysis configuration. Provide exactly ONE of `impact` or `damage`."""

    material: Union[str, OrthotropicMaterial]
    layup_deg: List[float]
    ply_thickness_mm: float
    panel: PanelGeometry
    loading: Literal["compression", "tension"] = "compression"
    tier: Literal["empirical", "semi_analytical", "fe3d"] = "empirical"
    impact: Optional[ImpactEvent] = None
    damage: Optional[DamageState] = None
    mesh: Optional[MeshParams] = None

    def __post_init__(self) -> None:
        if (self.impact is None) == (self.damage is None):
            raise ValueError(
                "Provide exactly one of AnalysisConfig.impact or AnalysisConfig.damage"
            )
        if self.tier == "fe3d" and self.mesh is None:
            self.mesh = MeshParams()

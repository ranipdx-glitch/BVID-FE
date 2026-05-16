"""Core composite-mechanics types (geometry, material, laminate)."""

from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY, OrthotropicMaterial

__all__ = [
    "ImpactorGeometry",
    "PanelGeometry",
    "Laminate",
    "MATERIAL_LIBRARY",
    "OrthotropicMaterial",
]

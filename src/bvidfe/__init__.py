"""BVID-FE — Barely Visible Impact Damage analysis for composite laminates.

The most commonly used public types are re-exported here so callers can
``from bvidfe import AnalysisConfig, BvidAnalysis, ...`` without learning
the internal module layout. The deep import paths keep working.
"""

__version__ = "0.2.0.dev0"

from bvidfe.analysis import (
    AnalysisConfig,
    AnalysisResults,
    BvidAnalysis,
    MeshParams,
)
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.material import MATERIAL_LIBRARY, OrthotropicMaterial
from bvidfe.damage.io import load_cscan_json
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.impact.mapping import ImpactEvent

__all__ = [
    "AnalysisConfig",
    "AnalysisResults",
    "BvidAnalysis",
    "MeshParams",
    "ImpactEvent",
    "ImpactorGeometry",
    "PanelGeometry",
    "MATERIAL_LIBRARY",
    "OrthotropicMaterial",
    "DamageState",
    "DelaminationEllipse",
    "load_cscan_json",
    "__version__",
]

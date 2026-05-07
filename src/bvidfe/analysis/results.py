"""Analysis results dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from bvidfe.damage.state import DamageState


@dataclass
class FieldResults:
    """3D FE tier field outputs. Populated only when tier='fe3d'."""

    displacement: np.ndarray
    stress_global: np.ndarray
    strain_global: np.ndarray
    stress_local: np.ndarray
    failure_index: np.ndarray
    buckling_mode_shape: Optional[np.ndarray] = None
    cohesive_damage: Optional[np.ndarray] = None


@dataclass
class AnalysisResults:
    """Outcome of a single ``BvidAnalysis.run()`` invocation.

    The strength fields are defined identically across all three tiers:

    - ``pristine_strength_MPa`` is a thickness-weighted ply-average of the
      lamina-level strengths from the material card (see
      ``bvidfe.analysis.bvid._pristine_strength``). Same denominator for
      every tier.
    - ``residual_strength_MPa`` is the tier-specific damaged strength.
    - ``knockdown = residual_strength_MPa / pristine_strength_MPa``,
      assigned in ``BvidAnalysis.run()``.

    Knockdowns from different tiers are on the same scale (same baseline),
    but capture different failure mechanisms — see the README "Knockdown
    definition and cross-tier comparability" section before comparing
    values across tiers.

    ``notes`` carries free-form runtime diagnostics emitted by the analysis
    backends — primarily silent fallbacks that affect the interpretation of
    ``knockdown`` (e.g. fe3d buckling eigensolve returning pristine because
    no positive eigenvalue was found, or the buckling-plausibility gate
    discarding a tiny eigenvalue and using FPF instead). Empty list when
    the run produced no diagnostic-worthy events.
    """

    residual_strength_MPa: float
    pristine_strength_MPa: float
    knockdown: float
    damage: DamageState
    dpa_mm2: float
    tier_used: str
    config_snapshot: Dict[str, Any]
    buckling_eigenvalues: Optional[List[float]] = None
    critical_sublaminate: Optional[int] = None
    field_results: Optional[FieldResults] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "BVID Analysis Results",
            f"  tier_used              : {self.tier_used}",
            f"  pristine_strength_MPa  : {self.pristine_strength_MPa:.1f}",
            f"  residual_strength_MPa  : {self.residual_strength_MPa:.1f}",
            f"  knockdown              : {self.knockdown:.3f}",
            f"  dpa_mm2                : {self.dpa_mm2:.1f}",
            f"  dent_depth_mm          : {self.damage.dent_depth_mm:.3f}",
            f"  n_delaminations        : {len(self.damage.delaminations)}",
        ]
        if self.notes:
            lines.append("  notes:")
            for note in self.notes:
                lines.append(f"    - {note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "residual_strength_MPa": self.residual_strength_MPa,
            "pristine_strength_MPa": self.pristine_strength_MPa,
            "knockdown": self.knockdown,
            "dpa_mm2": self.dpa_mm2,
            "tier_used": self.tier_used,
            "damage": {
                "dent_depth_mm": self.damage.dent_depth_mm,
                "fiber_break_radius_mm": self.damage.fiber_break_radius_mm,
                "delaminations": [
                    {
                        "interface_index": d.interface_index,
                        "centroid_mm": list(d.centroid_mm),
                        "major_mm": d.major_mm,
                        "minor_mm": d.minor_mm,
                        "orientation_deg": d.orientation_deg,
                    }
                    for d in self.damage.delaminations
                ],
            },
            "buckling_eigenvalues": self.buckling_eigenvalues,
            "critical_sublaminate": self.critical_sublaminate,
            "config_snapshot": self.config_snapshot,
            "notes": list(self.notes),
        }
        return out

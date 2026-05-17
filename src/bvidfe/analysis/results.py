"""Analysis results dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from bvidfe.damage.state import DamageState


@dataclass
class FieldResults:
    """Per-mesh field outputs reserved for the 3D FE tier.

    .. note::
       **Currently always ``None``.** ``BvidAnalysis.run()`` does not yet
       populate ``AnalysisResults.field_results`` for any tier (the fe3d
       tier returns scalar residual strength + buckling eigenvalues only).
       This dataclass documents the *intended* contract for a future
       field-output release; downstream code should treat
       ``result.field_results`` as optional and handle ``None``.

    Attributes
    ----------
    displacement : np.ndarray
        Nodal displacement, shape ``(n_nodes, 3)``, float64, units mm,
        in the global (x, y, z) coordinate system, original (undeformed)
        node ordering.
    stress_global : np.ndarray
        Per-element stress in the global frame, shape
        ``(n_elements, 6)``, units MPa, Voigt order
        ``[σxx, σyy, σzz, σyz, σxz, σxy]``.
    strain_global : np.ndarray
        Per-element strain in the global frame, shape
        ``(n_elements, 6)``, dimensionless, same Voigt order as
        ``stress_global``.
    stress_local : np.ndarray
        Per-element stress rotated into each ply's material (fibre) axes,
        shape ``(n_elements, 6)``, units MPa, Voigt order
        ``[σ11, σ22, σ33, σ23, σ13, σ12]``.
    failure_index : np.ndarray
        Per-element maximum failure index, shape ``(n_elements,)``,
        dimensionless (≥ 1.0 indicates predicted first-ply failure).
    buckling_mode_shape : np.ndarray, optional
        First buckling eigenvector as nodal displacement, shape
        ``(n_nodes, 3)``, normalised (dimensionless). ``None`` when no
        buckling solve ran or it did not converge.
    cohesive_damage : np.ndarray, optional
        Per-interface-element scalar cohesive damage variable in
        ``[0, 1]`` (0 = intact, 1 = fully separated). ``None`` until
        true cohesive surfaces land (see README "Limitations").
    """

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
    #: Machine-readable diagnostic tags, distinct from the human-readable
    #: ``notes``. Lets a script disambiguate an overloaded ``knockdown``
    #: (e.g. ``knockdown == 1.0`` from "pristine-equivalent" vs "fe3d
    #: buckling eigensolve failed") without string-scraping ``notes``.
    #: Populated tags (default ``[]``):
    #:
    #: - ``"fe3d_buckling_unconverged"`` — the fe3d buckling eigensolve
    #:   reported its own failure; residual is from first-ply failure.
    #: - ``"fe3d_buckling_artefact_dropped"`` — buckling solved but the
    #:   result was below 5% of pristine and was discarded as a numerical
    #:   artefact; residual is from first-ply failure.
    #:
    #: The ``impactor_mass_ratio_below_unity`` / ``dpa_panel_area_cap_clipped``
    #: regimes currently surface only via Python ``UserWarning`` + ``notes``;
    #: structured tags for them are future work.
    warnings: List[str] = field(default_factory=list)

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
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
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
            "warnings": list(self.warnings),
        }
        return out

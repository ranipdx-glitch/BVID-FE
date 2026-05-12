"""Analysis results dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from bvidfe.damage.state import DamageState


@dataclass
class FieldResults:
    """Per-node / per-element field outputs from the ``fe3d`` tier.

    All arrays follow the mesh-traversal order produced by
    ``bvidfe.analysis.fe_mesh.build_fe_mesh`` (z varies fastest, then y,
    then x; node index ``i = ix*ny*nz + iy*nz + iz``). Coordinates are
    *global panel* (origin at the lower-left corner of the panel,
    x along ``Lx``, y along ``Ly``, z through-thickness with z=0 at the
    laminate mid-plane). All length units are millimetres; stresses are
    in MPa; strains are dimensionless.

    Note
    ----
    As of v0.2.0-dev the ``BvidAnalysis.run()`` dispatcher never
    populates a ``FieldResults`` instance — ``AnalysisResults.field_results``
    is always ``None``. This class defines the contract for a future
    field-output pass so downstream tooling can be written against a
    stable schema.

    Attributes
    ----------
    displacement : np.ndarray
        Shape ``(n_nodes, 3)``, dtype float64. Nodal displacement vector
        ``(u_x, u_y, u_z)`` in mm, in global panel coordinates.
    stress_global : np.ndarray
        Shape ``(n_elements, 6)``, dtype float64. Element-averaged
        Cauchy stress in Voigt order
        ``(sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy)``,
        units MPa, in global panel coordinates.
    strain_global : np.ndarray
        Shape ``(n_elements, 6)``, dtype float64. Element-averaged
        small-strain tensor in the same Voigt order as ``stress_global``,
        dimensionless, in global panel coordinates.
    stress_local : np.ndarray
        Shape ``(n_elements, 6)``, dtype float64. ``stress_global``
        rotated into each element's *ply-local* frame
        ``(1=fibre, 2=transverse in-plane, 3=through-thickness)``. This
        is the frame in which Tsai-Wu / LaRC05 strengths are defined.
    failure_index : np.ndarray
        Shape ``(n_elements,)``, dtype float64. Per-element scalar Tsai-Wu
        index evaluated on ``stress_local``. Values ``>= 1.0`` indicate
        first-ply failure under the applied load.
    buckling_mode_shape : Optional[np.ndarray]
        Shape ``(n_nodes, 3)``, dtype float64. Eigenvector of the critical
        buckling mode (same node ordering as ``displacement``). ``None``
        when the buckling solve was skipped or returned no positive
        eigenvalue (see ``AnalysisResults.buckling_eigenvalues``).
    cohesive_damage : Optional[np.ndarray]
        Shape ``(n_interfaces, n_y, n_x)``, dtype float64. Per-cohesive-
        element damage variable in ``[0, 1]`` (``0`` = pristine, ``1`` =
        fully delaminated). ``None`` when no cohesive zone was active.
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

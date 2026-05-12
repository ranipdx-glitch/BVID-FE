"""Summary tab: text-only display of AnalysisResults.

In addition to the raw result fields, this tab post-hoc inspects the config
snapshot and appends plain-language notices for common edge cases that the
command-line version emits as ``UserWarning``s on stderr (and are therefore
invisible when the app is launched from Finder). Examples: DPA saturated at
the panel-area cap, the quasi-static mass regime is violated, the panel's
``free`` boundary is using a pragmatic soft-support approximation, and the
known fe3d energy-insensitivity limitation.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from bvidfe.analysis import AnalysisResults


def _density_kg_per_mm3(config_snapshot: dict) -> float | None:
    """Return material density if present in the snapshot.

    config_snapshot['material'] is either a string (library name) or the
    asdict() of an OrthotropicMaterial. In the latter case the 'rho' key
    is already present; in the former we look it up."""
    mat = config_snapshot.get("material")
    if isinstance(mat, dict):
        return mat.get("rho")
    if isinstance(mat, str):
        try:
            from bvidfe.core.material import MATERIAL_LIBRARY

            return MATERIAL_LIBRARY[mat].rho
        except Exception:
            return None
    return None


def _build_limitation_notes(results: AnalysisResults) -> list[str]:
    """Return a list of user-facing limitation / edge-case notices for the
    current run, derived from the config snapshot and result values.

    Each note is one paragraph, already wrapped / formatted. Order matters:
    most-severe first. Empty list means "nothing to call out."
    """
    notes: list[str] = []
    cfg = results.config_snapshot or {}
    panel = cfg.get("panel", {}) or {}
    Lx = float(panel.get("Lx_mm", 0.0) or 0.0)
    Ly = float(panel.get("Ly_mm", 0.0) or 0.0)
    A_panel = Lx * Ly
    boundary = panel.get("boundary", "simply_supported")

    # 1. DPA saturation: the 80% panel-area cap has engaged or is close to
    #    it. Downstream knockdown is insensitive to further energy increase.
    if A_panel > 0 and results.dpa_mm2 >= 0.79 * A_panel:
        pct = 100.0 * results.dpa_mm2 / A_panel
        notes.append(
            f"⚠ DPA saturation: predicted damage area ({results.dpa_mm2:.0f} "
            f"mm², {pct:.1f}% of panel) is at or near the 80% panel-area cap. "
            f"Knockdown will be insensitive to further energy increase for this "
            f"panel geometry. Consider a larger panel, thicker laminate, or "
            f"lower impact energy to keep DPA below saturation."
        )

    # 2. 'free' boundary: the Olsson bending stiffness is approximated as
    #    0.4x SSSS since a truly free plate has no bending restoring force.
    if boundary == "free":
        notes.append(
            "ℹ 'free' panel boundary: BVID-FE uses a 0.4× simply-supported-"
            "equivalent bending stiffness as a soft-support approximation. "
            "A truly unrestrained plate has no bending restoring force at a "
            "central impact and Olsson's quasi-static model does not apply. "
            "Results should be interpreted qualitatively."
        )

    # 3. Quasi-static mass regime: impactor lighter than the plate => Olsson
    #    quasi-static threshold may underpredict damage by 30%+.
    impact = cfg.get("impact") or {}
    mass_kg = float(impact.get("mass_kg", 0.0) or 0.0)
    ply_t = float(cfg.get("ply_thickness_mm", 0.0) or 0.0)
    layup = cfg.get("layup_deg") or []
    h_total = ply_t * len(layup) if ply_t and layup else 0.0
    rho = _density_kg_per_mm3(cfg)
    if rho and mass_kg > 0 and A_panel > 0 and h_total > 0:
        m_plate = rho * A_panel * h_total  # kg
        if m_plate > 0:
            m_ratio = mass_kg / m_plate
            if m_ratio < 1.0:
                notes.append(
                    f"⚠ Small-mass impact regime: impactor mass ({mass_kg:.2f} "
                    f"kg) is comparable to / lighter than the plate's effective "
                    f"mass ({m_plate*1000:.1f} g; ratio = {m_ratio:.2f}). "
                    f"Olsson's quasi-static threshold model is only strictly "
                    f"valid for ratio > 40; below 1 the predicted damage is "
                    f"likely conservative. A mild dynamic-amplification factor "
                    f"is already applied via the mass_kg input."
                )

    # 4. fe3d tier notice: the tier uses a component-wise stiffness-reduction
    #    model (DAMAGE_OOP_FACTOR ≈ 0.05 for delamination, plus an additional
    #    DAMAGE_FIBER_BREAK_INPLANE_FACTOR ≈ 0.30 inside the fiber-break core).
    #    Buckling now scales with damage size, but the absolute values can
    #    still be conservative compared to empirical at small DPA because the
    #    buckling path uses a uniform pre-stress approximation with minimal
    #    lateral BCs. Point users at empirical for screening / sweeps; fe3d
    #    for stress-field context.
    if results.tier_used == "fe3d" and results.knockdown > 0:
        notes.append(
            "ℹ fe3d tier: intended for stress-field context at a single "
            "configuration rather than high-throughput energy sweeps. The "
            "residual-strength prediction now trends monotonically with "
            "impact energy (v0.2.0-dev fix), but absolute values remain "
            "conservative compared to tier='empirical' (Soutis). Use "
            "tier='empirical' for energy sweeps and "
            "tier='semi_analytical' for sublaminate-buckling-dominated "
            "failure modes."
        )

    # 5. Runtime notes from the analysis backend (silent fallbacks etc.).
    #    These describe what the solver actually did this run, as opposed
    #    to the input-driven caveats above.
    for runtime_note in results.notes:
        notes.append(f"⚠ {runtime_note}")

    return notes


class SummaryTab(QWidget):
    """Text-only summary of an AnalysisResults, augmented with edge-case
    notices for DPA saturation, boundary-condition approximations, mass
    regime, and fe3d limitations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text_area = QPlainTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText("Run an analysis to see its summary here.")
        lay = QVBoxLayout(self)
        lay.addWidget(self.text_area)

    def update(self, results: AnalysisResults) -> None:  # type: ignore[override]
        text = results.summary()
        notes = _build_limitation_notes(results)
        if notes:
            text += "\n\n--- Notes ---\n"
            text += "\n\n".join(notes)
        self.text_area.setPlainText(text)

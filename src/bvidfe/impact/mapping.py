"""Forward orchestrator: ImpactEvent -> DamageState.

Pipeline:
  1. compute Olsson onset energy (boundary- and shape-aware)
  2. if below threshold -> empty DamageState
  3. compute target DPA = alpha * (E - Eonset) * 1e3 / (G_IIc * h), then apply
     multiplicative corrections for:
       - dynamic amplification (small impactor-to-plate mass ratios)
       - impactor-tip footprint (hemispherical / flat / conical)
       - impactor-diameter spread
  4. compute dent depth and fiber-break radius
  5. distribute DPA across interfaces via peanut template with union-area scaling
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Tuple

from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.damage.state import DamageState
from bvidfe.impact.dent_model import dent_depth_mm, fiber_break_radius_mm
from bvidfe.impact.olsson import onset_energy
from bvidfe.impact.shape_templates import distribute_damage

# Impactor-shape footprint multiplier on target DPA. Empirical: flat-ended
# impactors spread the contact force over a larger area and produce wider
# delamination footprints; conical impactors concentrate penetration
# rather than delamination. These values are first-order calibration
# constants — override via custom OrthotropicMaterial if you have test data.
_SHAPE_DPA_FACTOR: dict[str, float] = {
    "hemispherical": 1.0,
    "flat": 1.4,
    "conical": 0.7,
}

# Reference impactor diameter (mm) for the footprint-spread correction.
# Matches ASTM D7136 default.
_DIAMETER_REF_MM: float = 16.0
_DIAMETER_DPA_EXPONENT: float = 0.3


def _shape_dpa_factor(shape: str) -> float:
    return _SHAPE_DPA_FACTOR.get(shape, 1.0)


def _diameter_dpa_factor(diameter_mm: float) -> float:
    """Smaller impactors concentrate damage (larger factor); larger
    impactors spread it (smaller factor). Mild (exponent 0.3)."""
    if diameter_mm <= 0:
        return 1.0
    return (_DIAMETER_REF_MM / diameter_mm) ** _DIAMETER_DPA_EXPONENT


# Reference impactor mass (kg) for the small-mass amplification correction.
# 5.5 kg is the ASTM D7136 drop-weight standard and the calibration baseline
# for the empirical knockdown formulas in this package — at this mass the
# DPA-mass correction is exactly unity, so historical test data and the
# validation gate remain unshifted.
_MASS_REF_KG: float = 5.5
_MASS_DPA_EXPONENT: float = 0.10  # very mild: (5.5/m)**0.1


def _dynamic_amplification_factor(
    mass_kg: float, lam: Laminate, panel: PanelGeometry
) -> tuple[float, float]:
    """Mass-ratio-dependent amplification of DPA.

    Returns ``(daf, m_ratio)`` where ``m_ratio = m_impactor / m_plate``
    (both in kg) and::

        daf = (m_ref / m_impactor) ** 0.1        # mild, centered on m_ref = 5.5 kg

    At the ASTM D7136 reference mass (5.5 kg) DAF = 1.000 exactly, so the
    calibration of the empirical knockdown formulas is unperturbed. Lighter
    impactors at the same energy carry higher velocity / shorter contact
    time → slightly more delamination per Joule; heavier impactors → slightly
    less. The exponent 0.1 keeps the effect bounded to about [0.85, 1.17]
    across the practical 1 kg .. 20 kg range.

    We also return the mass ratio ``m_impactor / m_plate_eff`` so the caller
    can emit a "quasi-static validity" warning if the impactor is lighter
    than the plate (m_ratio < 1 — the small-mass regime where the Olsson
    quasi-static threshold can underpredict damage by 30%+).

    Note on units: ``OrthotropicMaterial.rho`` is specified in kg/mm^3
    (e.g. 1.57e-6 for CFRP, matching 1.57 g/cm^3). The docstring in
    ``core/material.py`` says "t/mm^3" for a hypothetical SI-mm-N-s mass
    unit, but the actual numerical values follow the kg/mm^3 convention;
    we treat them as kg/mm^3 to compute a physically meaningful m_ratio.
    """
    rho_kg_per_mm3 = lam.material.rho
    m_plate_eff_kg = rho_kg_per_mm3 * panel.Lx_mm * panel.Ly_mm * lam.thickness_mm
    if m_plate_eff_kg <= 0 or mass_kg <= 0:
        return 1.0, float("inf")
    m_ratio = mass_kg / m_plate_eff_kg
    daf = (_MASS_REF_KG / mass_kg) ** _MASS_DPA_EXPONENT
    return daf, m_ratio


@dataclass
class ImpactEvent:
    """Parameters describing a single impact event."""

    energy_J: float
    impactor: ImpactorGeometry = field(default_factory=ImpactorGeometry)
    mass_kg: float = 5.5
    location_xy_mm: Tuple[float, float] = (0.0, 0.0)


def impact_to_damage(event: ImpactEvent, lam: Laminate, panel: PanelGeometry) -> DamageState:
    """Map an impact event to a full BVID damage state."""
    material = lam.material

    # If location is (0, 0), interpret as panel center for Olsson bending stiffness
    loc = event.location_xy_mm
    if loc == (0.0, 0.0):
        loc = (panel.Lx_mm / 2, panel.Ly_mm / 2)

    E_onset = onset_energy(lam, panel, event.impactor, location_xy_mm=loc)
    if event.energy_J <= E_onset:
        return DamageState([], dent_depth_mm=0.0, fiber_break_radius_mm=0.0)

    # Base Olsson DPA prediction (SI mm/N; energies J -> N*mm via *1e3).
    h = lam.thickness_mm
    dpa_target = material.olsson_alpha * (event.energy_J - E_onset) * 1e3 / (material.G_IIc * h)

    # ---- multiplicative physics-motivated corrections --------------------
    # Dynamic amplification (small impactor mass => more damage per Joule)
    daf, m_ratio = _dynamic_amplification_factor(event.mass_kg, lam, panel)
    # Olsson's quasi-static threshold assumes the impactor is much heavier
    # than the plate (canonical rule: m_ratio > 40). Below ~1, dynamic
    # effects become significant and DPA is likely underpredicted regardless
    # of our mild DAF correction.
    if m_ratio < 1.0:
        warnings.warn(
            f"Impactor mass ({event.mass_kg:.2f} kg) is comparable to or lighter "
            f"than the plate's effective mass (ratio = {m_ratio:.2f}); Olsson's "
            f"quasi-static threshold model may underpredict damage in this "
            f"regime. Predictions should be interpreted qualitatively.",
            UserWarning,
            stacklevel=2,
        )
    # Impactor tip shape footprint (flat spreads damage; conical concentrates)
    shape_factor = _shape_dpa_factor(event.impactor.shape)
    # Impactor diameter spread (smaller D -> more concentrated -> larger DPA factor)
    diam_factor = _diameter_dpa_factor(event.impactor.diameter_mm)

    dpa_target *= daf * shape_factor * diam_factor

    if dpa_target <= 0:
        return DamageState([], dent_depth_mm=0.0, fiber_break_radius_mm=0.0)

    # Cap DPA at 80% of panel area to avoid physically unreasonable footprints
    A_panel = panel.Lx_mm * panel.Ly_mm
    A_cap = 0.8 * A_panel
    if dpa_target > A_cap:
        warnings.warn(
            f"Olsson-predicted DPA ({dpa_target:.0f} mm^2) exceeds 80% of panel "
            f"area ({A_cap:.0f} mm^2). Clipping to {A_cap:.0f} mm^2. The impact "
            f"energy may exceed the panel's capacity to contain damage without "
            f"edge effects; consider a larger panel or lower energy.",
            UserWarning,
            stacklevel=2,
        )
        dpa_target = A_cap

    dent = dent_depth_mm(material, event.energy_J, E_onset, h)
    r_fb = fiber_break_radius_mm(material, event.energy_J)

    # Centroid for distribute_damage is the event location on the panel
    ellipses = distribute_damage(
        layup_deg=lam.layup_deg,
        target_dpa_mm2=dpa_target,
        dent_depth_mm=dent,
        fiber_break_radius_mm=r_fb,
        centroid_mm=loc,
    )

    return DamageState(delaminations=ellipses, dent_depth_mm=dent, fiber_break_radius_mm=r_fb)

"""Regression tests locking in that every *user-facing* input variable actually
produces a measurable change in the downstream knockdown / E_onset / DPA.

These tests were added after a validation sweep (see scripts/validate_inputs.py)
revealed that panel.boundary, impactor.shape, impactor.mass_kg, and
impactor.diameter_mm were silently ignored by the physics pipeline. Wiring them
in required changes in olsson.py, mapping.py, semi_analytical.py, and fe_tier.py
— these tests prevent the inputs from ever silently reverting to inert again.
"""

from __future__ import annotations

import warnings

import pytest

from bvidfe.analysis import AnalysisConfig, BvidAnalysis, MeshParams
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.impact.mapping import ImpactEvent
from bvidfe.impact.olsson import onset_energy

MAT = "IM7/8552"
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_T = 0.152


# Use a *larger* panel (300x200) + moderate energy (10 J) so DPA stays
# well under the 80% cap — otherwise every config saturates and variable
# sensitivity becomes invisible.
TEST_PANEL = PanelGeometry(300, 200)
TEST_ENERGY_J = 10.0


def _cfg(**overrides):
    panel = overrides.pop("panel", TEST_PANEL)
    impact = overrides.pop(
        "impact",
        ImpactEvent(TEST_ENERGY_J, ImpactorGeometry(), mass_kg=5.5),
    )
    kw = dict(
        material=MAT,
        layup_deg=LAYUP,
        ply_thickness_mm=PLY_T,
        panel=panel,
        loading="compression",
        tier="empirical",
        impact=impact,
    )
    kw.update(overrides)
    return AnalysisConfig(**kw)


def _run_kd(cfg) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BvidAnalysis(cfg).run().knockdown


# ---------- panel.boundary ----------


def test_boundary_changes_onset_energy():
    """Per Olsson's formula E_onset = Pc^2 / (2 * k_cb), a STIFFER plate
    (clamped) reaches the fracture threshold load Pc at a LOWER absorbed
    energy because it deflects less. Counterintuitive but correct.

    Soft-support ("free") has the opposite trend — higher E_onset because
    the soft plate can deflect further and absorb more energy before Pc.
    """
    lam = Laminate(MATERIAL_LIBRARY[MAT], LAYUP, PLY_T)
    imp = ImpactorGeometry()
    eo_ss = onset_energy(lam, PanelGeometry(150, 100, "simply_supported"), imp)
    eo_cl = onset_energy(lam, PanelGeometry(150, 100, "clamped"), imp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eo_fr = onset_energy(lam, PanelGeometry(150, 100, "free"), imp)
    # Monotone: clamped (stiffest k_b -> highest k_cb -> lowest E_onset)
    #         < simply_supported
    #         < free (softest k_b -> lowest k_cb -> highest E_onset)
    assert eo_cl < eo_ss < eo_fr, (eo_cl, eo_ss, eo_fr)


def test_boundary_changes_semi_analytical_knockdown():
    """Semi-analytical compression KD must differ for different boundary
    conditions. The combined effect is dominated by the sublaminate
    buckling coefficient (clamped ~ 1.9x SSSS), so clamped panels produce
    a higher residual strength / higher KD than simply-supported."""
    kd_ss = _run_kd(_cfg(panel=PanelGeometry(300, 200, "simply_supported"), tier="semi_analytical"))
    kd_cl = _run_kd(_cfg(panel=PanelGeometry(300, 200, "clamped"), tier="semi_analytical"))
    assert kd_cl != kd_ss, (kd_ss, kd_cl)
    assert kd_cl > kd_ss, (kd_ss, kd_cl)


# ---------- impactor.shape ----------


def test_shape_changes_dpa_and_knockdown():
    """Flat-ended impactors produce larger delamination footprints than
    hemispherical (more spread); conical less (concentrated penetration)."""

    def make(shape):
        return _cfg(impact=ImpactEvent(20.0, ImpactorGeometry(16.0, shape=shape), mass_kg=5.5))

    kd_hemi = _run_kd(make("hemispherical"))
    kd_flat = _run_kd(make("flat"))
    kd_cone = _run_kd(make("conical"))

    # Flat impactor spreads damage -> larger DPA -> lower KD; conical the opposite.
    assert kd_flat < kd_hemi, (kd_hemi, kd_flat)
    assert kd_cone > kd_hemi, (kd_hemi, kd_cone)


# ---------- impactor.mass_kg ----------


def test_mass_changes_dpa_centered_on_reference():
    """At the 5.5 kg reference the mass correction is unity; lighter
    impactors give slightly more damage (higher DPA, lower KD); heavier
    give slightly less."""

    def make(mass):
        return _cfg(impact=ImpactEvent(20.0, ImpactorGeometry(), mass_kg=mass))

    kd_ref = _run_kd(make(5.5))
    kd_light = _run_kd(make(1.0))
    kd_heavy = _run_kd(make(20.0))

    assert kd_light < kd_ref < kd_heavy, (kd_light, kd_ref, kd_heavy)


# ---------- impactor.diameter_mm ----------


def test_diameter_changes_dpa_via_spread_factor():
    """Smaller impactors concentrate damage → larger DPA → lower KD."""

    def make(d):
        return _cfg(impact=ImpactEvent(20.0, ImpactorGeometry(diameter_mm=d), mass_kg=5.5))

    kd_8 = _run_kd(make(8.0))
    kd_16 = _run_kd(make(16.0))
    kd_40 = _run_kd(make(40.0))

    assert kd_8 < kd_16 < kd_40, (kd_8, kd_16, kd_40)


# ---------- sanity: default still matches the validation baseline ----------


def test_fe3d_knockdown_mostly_decreases_with_energy():
    """Regression: the fe3d compression knockdown must trend *downward* with
    rising impact energy. Before DAMAGE_STIFFNESS_FACTOR was raised from 1e-4
    to 0.3 the fe3d residual strength actually *increased* with energy past
    ~15% mesh damage — damaged elements were so null in the stress field
    that the failure criterion never flagged them, and peak stress in the
    undamaged shell dropped as the damage footprint spread wider. The fix
    lets damaged elements carry realistic in-plane stress so the failure
    criterion can flag them.
    """
    from bvidfe.analysis import AnalysisConfig, BvidAnalysis
    from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
    from bvidfe.impact.mapping import ImpactEvent

    def kd_at(E):
        cfg = AnalysisConfig(
            material=MAT,
            layup_deg=LAYUP,
            ply_thickness_mm=PLY_T,
            panel=PanelGeometry(150, 100),
            loading="compression",
            tier="fe3d",
            impact=ImpactEvent(E, ImpactorGeometry(), mass_kg=5.5),
            mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=10.0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return BvidAnalysis(cfg).run().knockdown

    kd_low = kd_at(3.0)
    kd_high = kd_at(20.0)
    # Low energy (less damage) should give higher residual / higher knockdown
    assert kd_low > kd_high, (kd_low, kd_high)


def test_default_mass_gives_unit_correction():
    """The mass correction must be exactly 1.0 at the 5.5 kg reference so
    the validation gate (calibrated against legacy DPA predictions) remains
    unshifted. This is a safety rail against accidentally rebasing the
    calibration."""
    from bvidfe.impact.mapping import _dynamic_amplification_factor

    lam = Laminate(MATERIAL_LIBRARY[MAT], LAYUP, PLY_T)
    panel = PanelGeometry(150, 100)
    daf, _ = _dynamic_amplification_factor(5.5, lam, panel)
    assert daf == pytest.approx(1.0, abs=1e-12)

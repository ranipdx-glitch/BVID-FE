"""Empirical residual-strength models for CAI (Soutis) and TAI (Whitney-Nuismer).

CAI knockdown (Soutis & Curtis 1996):
    sigma_CAI / sigma_0 = 1 / (1 + k_s * (DPA / A_panel)^m)

TAI via equivalent open hole (Whitney & Nuismer 1974, point-stress criterion):
    sigma_N / sigma_0 = 2 / (2 + xi^2 + 3*xi^4 - (Kt_inf - 3)*(5*xi^6 - 7*xi^8))
where xi = R / (R + d0), R = sqrt(DPA/pi), d0 = material characteristic distance,
Kt_inf = infinite-plate stress concentration (3.0 for isotropic; for orthotropic
laminates use ``lekhnitskii_kt_infinity``).
"""

from __future__ import annotations

import math
import warnings

from bvidfe.core.laminate import Laminate
from bvidfe.core.material import OrthotropicMaterial


def lekhnitskii_kt_infinity(lam: Laminate) -> float:
    """Infinite-plate stress concentration factor for an orthotropic laminate.

    Whitney & Nuismer's orthotropic form (Lekhnitskii anisotropic elasticity):

        Kt_inf = 1 + sqrt( 2*(sqrt(Ex/Ey) - nuxy) + Ex/Gxy )

    reduces to the isotropic value 3.0 for a quasi-isotropic laminate. The
    engineering constants are taken from ``lam.effective_engineering_constants``.
    """
    Ex, Ey, Gxy, nuxy = lam.effective_engineering_constants()
    return 1.0 + math.sqrt(2.0 * (math.sqrt(Ex / Ey) - nuxy) + Ex / Gxy)


def soutis_cai(
    m: OrthotropicMaterial,
    dpa_mm2: float,
    A_panel_mm2: float,
    sigma_pristine_MPa: float,
) -> float:
    """Compression-after-impact residual strength via Soutis knockdown.

    Returns ``kd * sigma_pristine_MPa`` where
    ``kd = 1 / (1 + k_s * (DPA/A_panel)^m)``. Reaches ``sigma_pristine_MPa``
    exactly when ``dpa_mm2 == 0``. The returned value is the numerator
    used by ``BvidAnalysis.run()`` to compute ``AnalysisResults.knockdown``;
    the pristine reference is the same for all three tiers.
    """
    if dpa_mm2 <= 0:
        return sigma_pristine_MPa
    kd = 1.0 / (1.0 + m.soutis_k_s * (dpa_mm2 / A_panel_mm2) ** m.soutis_m)
    return kd * sigma_pristine_MPa


def whitney_nuismer_tai(
    m: OrthotropicMaterial,
    dpa_mm2: float,
    sigma_pristine_MPa: float,
    Kt_inf: float | None = None,
) -> float:
    """Tension-after-impact via Whitney-Nuismer point-stress on an equivalent
    circular hole of diameter 2*sqrt(DPA/pi).

    ``Kt_inf`` is the infinite-plate stress concentration factor; callers
    should pass the orthotropic value from ``lekhnitskii_kt_infinity``. When
    left as ``None`` it falls back to the isotropic value 3.0 and a
    ``UserWarning`` is emitted, since that is unconservative for CFRP.

    Used by both the ``empirical`` and ``semi_analytical`` tiers for TAI
    (the semi-analytical TAI path delegates here unchanged), so those two
    tiers report mathematically identical knockdown values for tension.
    """
    if Kt_inf is None:
        warnings.warn(
            "whitney_nuismer_tai called without Kt_inf; falling back to the "
            "isotropic value 3.0. This underpredicts the notch sensitivity of "
            "orthotropic CFRP laminates. Pass lekhnitskii_kt_infinity(lam).",
            UserWarning,
            stacklevel=2,
        )
        Kt_inf = 3.0
    if dpa_mm2 <= 0:
        return sigma_pristine_MPa
    R = math.sqrt(dpa_mm2 / math.pi)
    d0 = m.wn_d0_mm
    xi = R / (R + d0)
    denom = 2.0 + xi**2 + 3 * xi**4 - (Kt_inf - 3.0) * (5 * xi**6 - 7 * xi**8)
    kd = 2.0 / denom
    return kd * sigma_pristine_MPa

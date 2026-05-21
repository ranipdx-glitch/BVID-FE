"""Semi-analytical tier: sublaminate Rayleigh-Ritz buckling + critical interface scoring.

The ellipse is approximated as its enclosing simply-supported rectangle
with full side lengths (2 * major_mm) x (2 * minor_mm) in the panel
frame. For orthotropic simply-supported rectangles under uniaxial
compression, the sine basis is exact and the eigenvalue is closed-form —
we minimize over integer modes (m, n) in [1..5] x [1..5]. The closed-
form formula uses the FULL side lengths (a, b) as in Timoshenko & Gere
§9.2 / Reddy 4.4.4, NOT the ellipse semi-axes.

Sublaminate selection: the plies above the delaminated interface form the
thinner buckling sublaminate (closer to the impact face for interfaces in
the upper half of the laminate). We always use the smaller of the two
sublaminates because it buckles first.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from bvidfe.core.laminate import Laminate
from bvidfe.core.material import OrthotropicMaterial
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.failure.soutis_openhole import (
    lekhnitskii_kt_infinity,
    soutis_cai,
    whitney_nuismer_tai,
)


@dataclass(frozen=True)
class SemiAnalyticalResult:
    """Structured result of :func:`semi_analytical_cai`.

    Attributes
    ----------
    residual_strength_MPa : float
        Compression-after-impact residual strength in MPa. The minimum of
        the Soutis empirical knockdown and the sublaminate buckling stress
        when damage is present; equal to the pristine strength when the
        damage state is empty.
    critical_interface_index : int | None
        Index of the interface that scored highest in
        :func:`find_critical_interface`. ``None`` when the damage state
        contains no delaminations.
    critical_buckling_load_N : float | None
        Sublaminate buckling force per unit width (N/mm) at the critical
        interface. ``None`` when there is no damage or when the buckling
        sublaminate is degenerate (zero plies).
    """

    residual_strength_MPa: float
    critical_interface_index: int | None = None
    critical_buckling_load_N: float | None = None


# Sublaminate buckling coefficient multiplier on the SSSS Rayleigh-Ritz result
# for other panel boundary conditions. The delaminated sublaminate's edge
# condition is tied to how the parent panel is supported — stiffer parent
# boundaries transmit more lateral restraint to the sublaminate. Values are
# ratios of fundamental compression buckling coefficients (k) from
# Timoshenko & Gere (1961) Theory of Elastic Stability §9.2 for square
# plates; they transfer approximately to the rectangular orthotropic case
# used here.
_BOUNDARY_BUCKLING_FACTOR: dict[str, float] = {
    "simply_supported": 1.0,
    "clamped": 1.9,
    "free": 0.5,
}

# Maximum sublaminate aspect ratio (major / minor semi-axis) the closed-form
# Rayleigh-Ritz solution is trusted over. Slender peanut-template ellipses
# (e.g. b ~ 0.01 mm, a ~ 5 mm) drive the (a/b)^4 term to ~1e16, so the
# buckling stress overflows the Soutis empirical bound and the buckling
# channel goes silently inert. We clip the aspect ratio to this value and
# emit a ``DegenerateThinSublaminateWarning`` so the degenerate-thin
# condition is visible rather than masquerading as a clean empirical-tier
# result.
_MAX_SUBLAMINATE_ASPECT: float = 50.0


class DegenerateThinSublaminateWarning(UserWarning):
    """The buckling sublaminate is so slender that its aspect ratio was
    clipped to the trusted domain; the buckling channel may be inactive."""


def _sublaminate_D_matrix(
    material: OrthotropicMaterial,
    sub_layup_deg: list[float],
    ply_thickness_mm: Union[float, Sequence[float]],
) -> np.ndarray:
    """CLT D matrix (3, 3) for a sublaminate. z origin at sublaminate midplane.

    ``ply_thickness_mm`` may be a scalar (uniform) or a sequence of per-ply
    thicknesses with the same length as ``sub_layup_deg``.
    """
    sub_lam = Laminate(material, sub_layup_deg, ply_thickness_mm)
    _, _, D = sub_lam.abd_matrices()
    return D


def sublaminate_buckling_load(
    lam: Laminate,
    ellipse: DelaminationEllipse,
    boundary: str = "simply_supported",
) -> float:
    """Critical buckling force per unit width N_cr (N/mm) for the sublaminate
    above the given delamination interface.

    The delaminated sublaminate is approximated as an orthotropic
    simply-supported rectangle whose full side lengths equal twice the
    ellipse's semi-axes (the enclosing-rectangle simplification:
    ``a = 2 * major_mm``, ``b = 2 * minor_mm``). Under uniaxial compression
    along x, the closed-form Navier / Rayleigh-Ritz solution with a sine-
    basis trial function ``sin(m*pi*x/a) * sin(n*pi*y/b)`` on the domain
    ``0 <= x <= a``, ``0 <= y <= b`` (Timoshenko & Gere §9.2; Reddy *Theory
    and Analysis of Elastic Plates*, Eq. 4.4.4) is

        N_cr(m, n) = (pi^2 / a^2)
                     * [D11 * m^4 + 2*(D12 + 2*D66)*(m*a/b)^2 * n^2
                        + D22 * (a*n/b)^4]
                     / m^2

    where (a, b) are the **full** rectangle side lengths (NOT semi-axes),
    (m, n) are the integer half-wave numbers along x and y, and the D_ij
    are the sublaminate's CLT bending stiffnesses. We minimise over (m, n)
    in [1..5] x [1..5]; the range is bounded by the typical 1-3 mode of
    practical delaminations plus a safety margin. (Note: the sine basis
    above vanishes at x=0, x=a, y=0, y=b, so a/b here must be the full
    plate dimensions for the SSSS boundary conditions to be satisfied —
    same convention as the Navier impact-compliance solver in
    ``impact/olsson.py`` which passes ``pan.Lx_mm``, ``pan.Ly_mm`` directly.)

    The selected sublaminate is the *thinner* (by through-thickness, not
    ply count) of the "above" and "below" stacks at the interface (it
    buckles first), and a boundary-dependent multiplier is applied so the
    parent panel's edge condition transmits appropriate lateral restraint
    to the sublaminate (clamped: 1.9x, free: 0.5x, simply-supported: 1.0x).

    Parameters
    ----------
    lam : Laminate
        Full panel laminate; supplies material, layup, and ply thickness.
    ellipse : DelaminationEllipse
        Delamination at which the sublaminate forms. Its
        ``interface_index`` selects the sublaminate thickness; the
        enclosing rectangle has full side lengths
        ``a = 2 * ellipse.major_mm`` and ``b = 2 * ellipse.minor_mm``
        (``major_mm`` / ``minor_mm`` are the ellipse semi-axes, per
        ``DelaminationEllipse.area_mm2 = pi * major * minor``).
    boundary : str
        One of ``"simply_supported"``, ``"clamped"``, ``"free"``.

    The ellipse aspect ratio ``a / b`` is clipped to ``_MAX_SUBLAMINATE_ASPECT``
    (50) before evaluating the eigenvalue. Slender peanut-template ellipses
    would otherwise drive the ``(a/b)^4`` term to ~1e16 N/mm, overflowing the
    Soutis empirical bound so ``semi_analytical_cai`` silently returns the
    empirical result with the buckling channel inert. When the clip fires a
    :class:`DegenerateThinSublaminateWarning` is emitted so the degenerate
    condition is visible.

    Returns
    -------
    float
        Critical buckling force per unit sublaminate width N_cr in N/mm,
        already multiplied by the boundary factor. Returns ``inf`` when the
        sublaminate is degenerate (zero plies, zero ellipse area).
    """
    i = ellipse.interface_index
    full_layup = lam.layup_deg
    full_thicknesses = lam.ply_thicknesses_mm

    # Choose the geometrically thinner sublaminate between "above" (plies
    # 0..i) and "below" (plies i+1..). Selection is by through-thickness
    # (sum of per-ply thicknesses), not ply count — for non-uniform
    # laminates the side with fewer plies may be the geometrically thicker
    # one, and the *thinner* stack is what buckles first.
    upper_layup = full_layup[: i + 1]
    lower_layup = full_layup[i + 1 :]
    upper_thicknesses = full_thicknesses[: i + 1]
    lower_thicknesses = full_thicknesses[i + 1 :]
    upper_t_total = sum(upper_thicknesses)
    lower_t_total = sum(lower_thicknesses)
    if upper_t_total <= lower_t_total:
        sub_layup = upper_layup
        sub_thicknesses = upper_thicknesses
    else:
        sub_layup = lower_layup
        sub_thicknesses = lower_thicknesses
    if len(sub_layup) == 0:
        return float("inf")

    D = _sublaminate_D_matrix(lam.material, sub_layup, sub_thicknesses)
    D11, D22, D12, D66 = D[0, 0], D[1, 1], D[0, 1], D[2, 2]

    # Rectangle dimensions (panel frame). The Navier sine basis
    # sin(m*pi*x/a) * sin(n*pi*y/b) used in the closed-form eigenvalue
    # vanishes at x=0,a and y=0,b, so (a, b) must be the FULL rectangle
    # side lengths — i.e. twice the ellipse semi-axes — for the SSSS
    # boundary conditions to be satisfied. (Issue #29: previously assigned
    # a = ellipse.major_mm directly, which used semi-axes and overpredicted
    # N_cr by ~4x because N_cr ∝ 1/a^2.)
    a = 2.0 * ellipse.major_mm
    b = 2.0 * ellipse.minor_mm
    if a <= 0 or b <= 0:
        return float("inf")

    # Reformulate in terms of the aspect ratio so a slender (b -> 0) ellipse
    # cannot blow the (a/b)^4 term past the Soutis bound and silently disable
    # the buckling channel. Clip to the trusted domain and surface the clip.
    aspect = a / b
    if aspect > _MAX_SUBLAMINATE_ASPECT:
        warnings.warn(
            f"Sublaminate ellipse aspect ratio (a/b = {aspect:.1f}) exceeds the "
            f"trusted Rayleigh-Ritz domain ({_MAX_SUBLAMINATE_ASPECT:.0f}); "
            f"clipping to {_MAX_SUBLAMINATE_ASPECT:.0f}. The sublaminate-buckling "
            f"channel is degenerate for this thin slice and may not constrain "
            f"the residual strength; the empirical Soutis tier likely governs.",
            DegenerateThinSublaminateWarning,
            stacklevel=2,
        )
        aspect = _MAX_SUBLAMINATE_ASPECT

    # Minimum over (m, n) in 1..5 for uniaxial compression N0_x:
    # N_cr(m,n) = (pi^2 / a^2) * [D11*m^4 + 2*(D12+2*D66)*(m*aspect)^2*n^2 + D22*(aspect*n)^4] / m^2
    pi2 = math.pi * math.pi
    best = float("inf")
    for m_mode in range(1, 6):
        for n_mode in range(1, 6):
            num = (
                D11 * m_mode**4
                + 2.0 * (D12 + 2.0 * D66) * (m_mode * aspect) ** 2 * n_mode**2
                + D22 * (aspect * n_mode) ** 4
            )
            N_mn = (pi2 / a**2) * num / m_mode**2
            if N_mn < best:
                best = N_mn
    boundary_factor = _BOUNDARY_BUCKLING_FACTOR.get(boundary, 1.0)
    return best * boundary_factor


def find_critical_interface(damage: DamageState, lam: Laminate) -> Optional[int]:
    """Return the interface index that would fail first under compression.

    Scoring: max_area_i * max(|z_upper_i|, |z_lower_i|), where z is distance
    from interface to the top/bottom laminate surface. Largest wins.

    For non-uniform laminates the ``z_upper`` / ``z_lower`` distances are
    cumulative sums of the actual per-ply thicknesses rather than
    ``n_plies * uniform_thickness``.
    """
    if not damage.delaminations:
        return None
    thicknesses = lam.ply_thicknesses_mm
    total_h = float(sum(thicknesses))
    # cum_z[i] is the through-thickness distance from the bottom face to the
    # top of ply i — i.e. the z position of interface i (interface k separates
    # ply k from ply k+1).
    cum_z = [0.0] * (len(thicknesses) + 1)
    for k, t in enumerate(thicknesses):
        cum_z[k + 1] = cum_z[k] + t
    per_iface_max_area: dict[int, float] = {}
    for e in damage.delaminations:
        per_iface_max_area[e.interface_index] = max(
            per_iface_max_area.get(e.interface_index, 0.0), e.area_mm2
        )

    best_idx: Optional[int] = None
    best_score = -1.0
    for idx, area in per_iface_max_area.items():
        # Distance from the top/bottom laminate surface to the interface.
        z_upper = cum_z[idx + 1]
        z_lower = total_h - cum_z[idx + 1]
        score = area * max(z_upper, z_lower)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def semi_analytical_cai(
    lam: Laminate,
    damage: DamageState,
    sigma_pristine_MPa: float,
    A_panel_mm2: float,
    boundary: str = "simply_supported",
) -> SemiAnalyticalResult:
    """Semi-analytical compression-after-impact residual strength (MPa).

    Takes the minimum of:
      (a) Soutis empirical knockdown at total DPA, and
      (b) critical sublaminate buckling stress at the most critical interface
          (boundary-aware — clamped parent panels are ~1.9x stiffer, free
          ~0.5x, relative to simply-supported).

    Because of (b), the returned residual stress is always less-than-or-equal
    to the empirical-tier result on the same input — so the resulting
    ``knockdown`` (as computed downstream by ``BvidAnalysis.run()``) is
    guaranteed to be less-than-or-equal to the ``empirical`` tier's.

    Returns
    -------
    SemiAnalyticalResult
        Structured result with ``residual_strength_MPa``,
        ``critical_interface_index``, and ``critical_buckling_load_N``.
        If the damage state is empty, ``residual_strength_MPa`` equals
        ``sigma_pristine_MPa`` and the two interface fields are ``None``.
    """
    if not damage.delaminations:
        return SemiAnalyticalResult(
            residual_strength_MPa=sigma_pristine_MPa,
            critical_interface_index=None,
            critical_buckling_load_N=None,
        )

    # Soutis bound
    dpa = damage.projected_damage_area_mm2
    sigma_soutis = soutis_cai(lam.material, dpa, A_panel_mm2, sigma_pristine_MPa)

    # Sublaminate buckling bound
    crit_idx = find_critical_interface(damage, lam)
    if crit_idx is None:
        return SemiAnalyticalResult(
            residual_strength_MPa=sigma_soutis,
            critical_interface_index=None,
            critical_buckling_load_N=None,
        )

    # Largest ellipse at that interface drives buckling
    ellipses_at_crit = [e for e in damage.delaminations if e.interface_index == crit_idx]
    critical_ellipse = max(ellipses_at_crit, key=lambda e: e.area_mm2)
    N_cr_per_mm = sublaminate_buckling_load(lam, critical_ellipse, boundary=boundary)  # N/mm

    # Sublaminate thickness — sum the actual per-ply thicknesses of whichever
    # half ("above" or "below" the interface) is the buckling sublaminate.
    # Selection is by through-thickness (not ply count) so that for
    # non-uniform laminates the geometrically thinner stack is picked —
    # this must match the selection in ``sublaminate_buckling_load``.
    thicknesses = lam.ply_thicknesses_mm
    upper_t = thicknesses[: crit_idx + 1]
    lower_t = thicknesses[crit_idx + 1 :]
    upper_t_total = sum(upper_t)
    lower_t_total = sum(lower_t)
    sub_t = upper_t if upper_t_total <= lower_t_total else lower_t
    if len(sub_t) == 0:
        return SemiAnalyticalResult(
            residual_strength_MPa=sigma_soutis,
            critical_interface_index=crit_idx,
            critical_buckling_load_N=None,
        )
    h_sub = float(sum(sub_t))
    sigma_buckling = N_cr_per_mm / h_sub if h_sub > 0 else float("inf")

    sigma_cai = min(sigma_soutis, sigma_buckling)
    return SemiAnalyticalResult(
        residual_strength_MPa=sigma_cai,
        critical_interface_index=crit_idx,
        critical_buckling_load_N=N_cr_per_mm,
    )


def semi_analytical_tai(
    lam: Laminate,
    damage: DamageState,
    sigma_pristine_MPa: float,
) -> float:
    """Semi-analytical tension-after-impact residual strength.

    v0.1.0: delegates to Whitney-Nuismer open-hole equivalent (same as empirical tier).
    The full Soutis cohesive-zone notch model with in-situ ply strength is
    deferred to v0.2.0.
    """
    dpa = damage.projected_damage_area_mm2
    Kt_inf = lekhnitskii_kt_infinity(lam)
    return whitney_nuismer_tai(lam.material, dpa, sigma_pristine_MPa, Kt_inf)

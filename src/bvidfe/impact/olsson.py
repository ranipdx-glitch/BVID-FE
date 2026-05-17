"""Olsson quasi-static impact threshold load and onset energy.

Closed-form threshold load from plate bending + delamination fracture balance:

    Pc = pi * sqrt(8 * G_IIc * D_eff / 9)
    E_onset = Pc^2 / (2 * k_cb)
    k_cb = 1 / (1/k_bending + 1/k_contact_linearized_at_Pc)

The bending stiffness ``k_bending`` depends on the plate edge boundary condition
(simply-supported / clamped / free). We compute a Navier-series point-load
compliance for the simply-supported case and apply a closed-form boundary
multiplier for the other cases — this is a first-order correction that keeps
the code simple while making the GUI ``boundary`` selector physically
meaningful. Clamped plates are typically 2-3x stiffer at center than SSSS of
the same dimensions (Timoshenko & Woinowsky-Krieger §4.4); a "free" plate has
only inertial resistance to a point load, so we treat it as a soft pseudo-
support and emit a warning.

The contact stiffness ``k_contact`` depends on the impactor tip shape:
- hemispherical: Hertzian sphere-on-flat (the baseline).
- flat (cylindrical punch): linear force-displacement, k = 2 * E_eff * R.
- conical: P = (2 * E_eff * tan(alpha) / pi) * delta^2 (Love's solution),
  linearized at P.

References:
- Olsson (2001), Composites Part A, 32(9).
- Olsson (2010), Int. J. Solids Struct., 47(21).
- Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, §4.
- Johnson (1985), Contact Mechanics, §3.4 (flat punch), §3.5 (cone).
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np

from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate

NAVIER_N: int = 11  # Navier series truncation (N x N modes)

# Closed-form multipliers on the SSSS center-point bending stiffness for
# other boundary conditions. These are approximate but capture the right
# order-of-magnitude trend so the GUI selector is not inert.
#
# For an orthotropic thin plate under a central point load, exact Galerkin
# solutions give a clamped/SSSS stiffness ratio of ~2-3 for aspect ratios
# in [0.5, 2.0] — we use 2.5 as a representative value. "Free" has no
# restoring bending stiffness from edges; we treat it as 0.4 * SSSS (a
# crude approximation that nonetheless gives users a visible effect).
_BOUNDARY_BENDING_FACTOR: dict[str, float] = {
    "simply_supported": 1.0,
    "clamped": 2.5,
    "free": 0.4,
}

# Typical conical drop-weight tip half-angle (measured from the cone axis).
# A 60-degree included angle gives a 30-degree half-angle.
_CONICAL_HALF_ANGLE_DEG: float = 30.0


@lru_cache(maxsize=64)
def _navier_basis_ssss(
    a: float, b: float, x0: float, y0: float, n_modes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cached SSSS Navier modal basis for a point load at ``(x0, y0)``.

    Returns ``(sin2_m, sin2_n, kx, ky)`` where ``kx = m*pi/a``,
    ``ky = n*pi/b`` and ``sin2_* = sin(k*coord)**2`` for ``m, n`` in
    ``1..n_modes``. These depend only on panel size, impact location and
    mode count — not on the laminate ``D`` matrix — so a parametric sweep
    that holds geometry + location fixed (e.g. an energy sweep) reuses the
    ``~n_modes**2`` trig evaluations instead of recomputing them per point.
    """
    idx = np.arange(1, n_modes + 1, dtype=float)
    kx = idx * np.pi / a
    ky = idx * np.pi / b
    sin2_m = np.sin(kx * x0) ** 2
    sin2_n = np.sin(ky * y0) ** 2
    return sin2_m, sin2_n, kx, ky


def _k_bending_ssss(
    lam: Laminate,
    pan: PanelGeometry,
    x0: float,
    y0: float,
    n_modes: int = NAVIER_N,
) -> float:
    """Navier series point-load stiffness of a simply-supported rectangular
    orthotropic plate at (x0, y0). Returns N/mm.

    w(x0,y0)/P = (4 / (a*b)) * sum_{m,n} sin^2(m*pi*x0/a) sin^2(n*pi*y0/b) / D_mn
    D_mn = D11 * (m*pi/a)^4 + 2*(D12 + 2*D66) * (m*pi/a)^2 * (n*pi/b)^2
           + D22 * (n*pi/b)^4
    k_bending = 1 / (w/P)
    """
    _, _, D = lam.abd_matrices()
    D11, D22, D12, D66 = D[0, 0], D[1, 1], D[0, 1], D[2, 2]
    a, b = pan.Lx_mm, pan.Ly_mm
    if not (0.0 < x0 < a and 0.0 < y0 < b):
        raise ValueError(
            f"impact location ({x0}, {y0}) must lie strictly inside the panel "
            f"(0, {a}) x (0, {b}); SSSS bending compliance is singular at the boundary."
        )
    sin2_m, sin2_n, kx, ky = _navier_basis_ssss(a, b, x0, y0, n_modes)
    kx2 = kx**2
    ky2 = ky**2
    # D_mn = D11*kx^4 + 2*(D12+2*D66)*kx^2*ky^2 + D22*ky^4  (outer over m, n)
    Dmn = (
        D11 * kx2[:, None] ** 2
        + 2.0 * (D12 + 2.0 * D66) * kx2[:, None] * ky2[None, :]
        + D22 * ky2[None, :] ** 2
    )
    numer = sin2_m[:, None] * sin2_n[None, :]
    w_over_P = (4.0 / (a * b)) * float(np.sum(numer / Dmn))
    return 1.0 / w_over_P


def _k_bending(
    lam: Laminate,
    pan: PanelGeometry,
    x0: float,
    y0: float,
    n_modes: int = NAVIER_N,
) -> float:
    """Boundary-aware center-point bending stiffness (N/mm).

    Dispatches to the SSSS Navier solution and then applies a boundary-dependent
    multiplier. "free" is physically ill-defined for a central point impact so
    we warn the caller.
    """
    k_ss = _k_bending_ssss(lam, pan, x0, y0, n_modes=n_modes)
    factor = _BOUNDARY_BENDING_FACTOR.get(pan.boundary, 1.0)
    if pan.boundary == "free":
        warnings.warn(
            "PanelGeometry.boundary='free' — a fully unrestrained plate has no "
            "bending restoring force at a central point load. BVID-FE applies a "
            "0.4x SSSS-equivalent stiffness as a soft-support approximation; "
            "results should be interpreted qualitatively. Use 'clamped' or "
            "'simply_supported' for quantitative predictions.",
            UserWarning,
            stacklevel=3,
        )
    return k_ss * factor


def _k_contact_hemispherical(lam: Laminate, imp: ImpactorGeometry, P: float) -> float:
    """Linear-equivalent Hertzian sphere-on-flat contact stiffness at load P (N).

    Nonlinear Hertz: P = k_nl * delta^(3/2), k_nl = (4/3) * sqrt(R) * E_eff.
    Linearized secant stiffness at P:
        delta = (P / k_nl)^(2/3);   k_lin = P / delta = k_nl^(2/3) * P^(1/3).
    """
    R = imp.diameter_mm / 2.0
    E_eff = _contact_E_eff(lam)
    k_nl = (4.0 / 3.0) * math.sqrt(R) * E_eff
    return (k_nl ** (2.0 / 3.0)) * (P ** (1.0 / 3.0))


def _k_contact_flat(lam: Laminate, imp: ImpactorGeometry, P: float) -> float:
    """Flat-ended cylindrical-punch contact stiffness (N/mm).

    Johnson (1985) §3.4: P = 2 * E_eff * R * delta (linear in delta).
    So the secant stiffness is constant, independent of load.
    """
    R = imp.diameter_mm / 2.0
    E_eff = _contact_E_eff(lam)
    return 2.0 * E_eff * R


def _k_contact_conical(lam: Laminate, imp: ImpactorGeometry, P: float) -> float:
    """Conical-tip contact stiffness at load P (N), linearized (N/mm).

    Johnson (1985) §3.5: P = (2 * E_eff * tan(alpha) / pi) * delta^2
    where alpha is the half-angle from the cone axis. Linearizing:
        k_lin = dP/d(delta)|_P = 2 * (2 E_eff tan(alpha) / pi) * delta
        delta_at_P = sqrt(P * pi / (2 E_eff tan(alpha)))
    -> k_lin = sqrt(8 * E_eff * tan(alpha) * P / pi)
    """
    E_eff = _contact_E_eff(lam)
    tan_alpha = math.tan(math.radians(_CONICAL_HALF_ANGLE_DEG))
    return math.sqrt(8.0 * E_eff * tan_alpha * P / math.pi)


def _contact_E_eff(lam: Laminate) -> float:
    """Effective Hertz-contact modulus between steel impactor and CFRP plate."""
    E_steel = 200e3  # MPa
    nu_steel = 0.3
    E_plate = lam.material.E22
    nu_plate = 0.3
    inv_E = (1 - nu_steel**2) / E_steel + (1 - nu_plate**2) / E_plate
    return 1.0 / inv_E


def _k_contact(lam: Laminate, imp: ImpactorGeometry, P: float) -> float:
    """Shape-aware linear-equivalent contact stiffness (N/mm)."""
    if imp.shape == "hemispherical":
        return _k_contact_hemispherical(lam, imp, P)
    if imp.shape == "flat":
        return _k_contact_flat(lam, imp, P)
    if imp.shape == "conical":
        return _k_contact_conical(lam, imp, P)
    raise ValueError(f"impactor shape {imp.shape!r} not recognized")


def threshold_load(lam: Laminate, pan: PanelGeometry, imp: ImpactorGeometry) -> float:
    """Olsson quasi-static damage-threshold load Pc (Newtons).

    Closed-form prediction from Olsson (2001) "Analytical prediction of large
    mass impact damage in composite laminates", *Composites Part A* 32(9):

        Pc = pi * sqrt(8 * G_IIc * D_eff / 9)

    where ``G_IIc`` is the mode-II interlaminar fracture toughness (N/mm,
    from the material card) and ``D_eff = sqrt(D11 * D22)`` is the
    geometric-mean flexural rigidity of the laminate (N*mm). The formula is
    derived from a plate-bending energy balance at the onset of through-
    thickness shear delamination and is independent of the panel size and
    impactor shape — only the laminate stack matters. Boundary-condition
    and shape effects enter the *onset energy* via ``onset_energy``, not
    here.

    Parameters
    ----------
    lam : Laminate
        Laminate carrying the material card and CLT D matrix.
    pan : PanelGeometry
        Accepted for API symmetry with `onset_energy`; not used.
    imp : ImpactorGeometry
        Accepted for API symmetry with `onset_energy`; not used.

    Returns
    -------
    float
        Threshold load Pc in Newtons. Always positive.
    """
    D_eff = lam.flexural_rigidity_Deff()  # N*mm
    G_IIc = lam.material.G_IIc  # N/mm
    return math.pi * math.sqrt(8 * G_IIc * D_eff / 9.0)


def onset_energy(
    lam: Laminate,
    pan: PanelGeometry,
    imp: ImpactorGeometry,
    location_xy_mm: tuple[float, float] | None = None,
) -> float:
    """Impact energy (J) at which BVID damage onsets.

    Depends on panel boundary condition (via k_bending) and impactor shape
    (via k_contact). See module docstring for the full set of formulas.
    """
    if location_xy_mm is None:
        location_xy_mm = (pan.Lx_mm / 2, pan.Ly_mm / 2)
    x0, y0 = location_xy_mm
    Pc = threshold_load(lam, pan, imp)  # N
    k_b = _k_bending(lam, pan, x0, y0, n_modes=NAVIER_N)  # N/mm (boundary-aware)
    k_c = _k_contact(lam, imp, P=Pc)  # N/mm (shape-aware, linearized at Pc)
    k_cb = 1.0 / (1.0 / k_b + 1.0 / k_c)  # N/mm
    E_mJ = Pc**2 / (2.0 * k_cb)  # N*mm = mJ
    return E_mJ * 1e-3  # J

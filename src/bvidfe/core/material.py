"""Orthotropic composite material model + built-in material library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OrthotropicMaterial:
    """Orthotropic composite ply material with strengths, toughnesses, and
    impact-mapping calibration constants.

    Units: moduli and strengths in MPa (N/mm^2), fracture toughness in N/mm,
    density in kg/mm^3 (e.g. 1.58e-6 for CFRP, matching 1.58 g/cm^3).

    Through-thickness strengths (Zt, Zc) and the 1-3 shear strength (S13)
    are optional. For unidirectional CFRP plies under the transverse-isotropy
    assumption, leaving these as ``None`` falls back to the in-plane
    transverse values via the ``Zt_resolved`` / ``Zc_resolved`` /
    ``S13_resolved`` properties (Zt → Yt, Zc → Yc, S13 → S12). Override only
    when material-specific through-thickness data are available — measured
    Zt is typically 20-40% lower than Yt for void-prone CFRP (Makeev et al.
    2012, *Composites Part A*).
    """

    name: str
    E11: float
    E22: float
    nu12: float
    G12: float
    G13: float
    G23: float
    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S12: float
    S23: float
    G_Ic: float
    G_IIc: float
    rho: float
    # Impact-mapping calibration (defaults chosen for typical CFRP; per-material override):
    olsson_alpha: float = 0.8
    dent_beta: float = 0.05
    dent_gamma: float = 0.5
    fiber_break_eta: float = 0.0
    fiber_break_E_threshold: float = 1e9  # disabled unless explicitly overridden
    soutis_k_s: float = 2.5
    soutis_m: float = 0.5
    wn_d0_mm: float = 1.0
    # Through-thickness strengths and 1-3 shear (None = transverse-isotropy
    # fallback to Yt / Yc / S12 via the *_resolved properties below).
    Zt: Optional[float] = None
    Zc: Optional[float] = None
    S13: Optional[float] = None

    def __post_init__(self) -> None:
        positive_fields = (
            "E11",
            "E22",
            "G12",
            "G13",
            "G23",
            "Xt",
            "Xc",
            "Yt",
            "Yc",
            "S12",
            "S23",
            "G_Ic",
            "G_IIc",
            "rho",
        )
        for k in positive_fields:
            v = getattr(self, k)
            if v <= 0:
                raise ValueError(f"{k} must be > 0 (got {v})")
        for k in ("Zt", "Zc", "S13"):
            v = getattr(self, k)
            if v is not None and v <= 0:
                raise ValueError(f"{k} must be > 0 if provided (got {v})")
        if not -1.0 < self.nu12 < 0.5:
            raise ValueError(f"nu12 out of physical range (got {self.nu12})")
        # rho is kg/mm^3; realistic engineering materials span ~[1e-7, 1e-5]
        # (0.1 .. 10 g/cm^3). A value outside this range almost always means
        # the user followed the old (wrong) "t/mm^3" docstring and is 1000x
        # off, which silently corrupts the impact-mapping mass ratio.
        if not 1e-7 <= self.rho <= 1e-5:
            raise ValueError(
                f"rho={self.rho} kg/mm^3 is outside the realistic range "
                f"[1e-7, 1e-5] (0.1-10 g/cm^3). Note rho is kg/mm^3, "
                f"e.g. 1.58e-6 for CFRP."
            )

    @property
    def nu21(self) -> float:
        return self.nu12 * self.E22 / self.E11

    @property
    def Zt_resolved(self) -> float:
        """Through-thickness tensile strength; falls back to Yt under
        transverse isotropy when not explicitly set."""
        return self.Yt if self.Zt is None else self.Zt

    @property
    def Zc_resolved(self) -> float:
        """Through-thickness compressive strength; falls back to Yc under
        transverse isotropy when not explicitly set."""
        return self.Yc if self.Zc is None else self.Zc

    @property
    def S13_resolved(self) -> float:
        """1-3 plane shear strength; falls back to S12 under transverse
        isotropy when not explicitly set."""
        return self.S12 if self.S13 is None else self.S13

    def get_compliance_matrix(self) -> np.ndarray:
        """Return the 6x6 orthotropic compliance matrix in Voigt notation
        (engineering shear strains), material frame."""
        S = np.zeros((6, 6))
        S[0, 0] = 1.0 / self.E11
        S[1, 1] = 1.0 / self.E22
        S[2, 2] = 1.0 / self.E22
        S[0, 1] = S[1, 0] = -self.nu12 / self.E11
        S[0, 2] = S[2, 0] = -self.nu12 / self.E11
        S[1, 2] = S[2, 1] = -self.nu21 / self.E22
        S[3, 3] = 1.0 / self.G23
        S[4, 4] = 1.0 / self.G13
        S[5, 5] = 1.0 / self.G12
        return S

    def get_stiffness_matrix(self) -> np.ndarray:
        """Return the 6x6 orthotropic stiffness matrix (inverse of compliance)."""
        return np.linalg.inv(self.get_compliance_matrix())


MATERIAL_LIBRARY: dict[str, OrthotropicMaterial] = {
    "AS4/3501-6": OrthotropicMaterial(
        name="AS4/3501-6",
        E11=138000,
        E22=9000,
        nu12=0.30,
        G12=6900,
        G13=6900,
        G23=3450,
        Xt=2280,
        Xc=1440,
        Yt=57,
        Yc=228,
        S12=71,
        S23=50,
        G_Ic=0.26,
        G_IIc=1.0,
        rho=1.58e-6,
    ),
    "IM7/8552": OrthotropicMaterial(
        name="IM7/8552",
        E11=165000,
        E22=8400,
        nu12=0.34,
        G12=5600,
        G13=5600,
        G23=2800,
        Xt=2560,
        Xc=1590,
        Yt=73,
        Yc=185,
        S12=90,
        S23=55,
        G_Ic=0.28,
        G_IIc=0.79,
        rho=1.57e-6,
    ),
    "T700/2510": OrthotropicMaterial(
        name="T700/2510",
        E11=127000,
        E22=8400,
        nu12=0.31,
        G12=4200,
        G13=4200,
        G23=2500,
        Xt=2100,
        Xc=1200,
        Yt=58,
        Yc=175,
        S12=66,
        S23=45,
        G_Ic=0.22,
        G_IIc=0.60,
        rho=1.55e-6,
    ),
    "T800/epoxy": OrthotropicMaterial(
        name="T800/epoxy",
        E11=155000,
        E22=8500,
        nu12=0.33,
        G12=5000,
        G13=5000,
        G23=2600,
        Xt=2700,
        Xc=1680,
        Yt=68,
        Yc=200,
        S12=85,
        S23=52,
        G_Ic=0.25,
        G_IIc=0.70,
        rho=1.58e-6,
    ),
}

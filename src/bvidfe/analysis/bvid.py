"""BvidAnalysis: orchestrates impact-to-damage and tier dispatch."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict
from typing import Union

from bvidfe._types import LoadingMode
from bvidfe.analysis.config import AnalysisConfig
from bvidfe.analysis.fe_tier import (
    _fe3d_cai_first_ply_failure,
    fe3d_cai_buckling,
    fe3d_tai,
)
from bvidfe.analysis.results import AnalysisResults
from bvidfe.analysis.semi_analytical import (
    SemiAnalyticalResult,
    semi_analytical_cai,
    semi_analytical_tai,
)
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY, OrthotropicMaterial
from bvidfe.damage.state import DamageState
from bvidfe.failure.soutis_openhole import (
    lekhnitskii_kt_infinity,
    soutis_cai,
    whitney_nuismer_tai,
)
from bvidfe.impact.mapping import impact_to_damage


def _resolve_material(m: Union[str, OrthotropicMaterial]) -> OrthotropicMaterial:
    if isinstance(m, str):
        return MATERIAL_LIBRARY[m]
    return m


def _pristine_strength(lam: Laminate, loading: LoadingMode) -> float:
    """Thickness-weighted ply-average pristine strength in the loading direction.

    For compression: sum_i t_i * (Xc*cos^2 + Yc*sin^2) / sum_i t_i
    For tension:     sum_i t_i * (Xt*cos^2 + Yt*sin^2) / sum_i t_i

    The per-ply thicknesses ``t_i`` are taken from ``lam.ply_thicknesses_mm``,
    so non-uniform laminates weight each ply by its actual thickness.
    """
    m = lam.material
    total_t = 0.0
    num = 0.0
    for theta, t_i in zip(lam.layup_deg, lam.ply_thicknesses_mm):
        c2 = math.cos(math.radians(theta)) ** 2
        s2 = math.sin(math.radians(theta)) ** 2
        if loading == "compression":
            num += t_i * (m.Xc * c2 + m.Yc * s2)
        else:
            num += t_i * (m.Xt * c2 + m.Yt * s2)
        total_t += t_i
    return num / total_t


def _config_snapshot_dict(cfg: AnalysisConfig) -> dict:
    """Serialise an AnalysisConfig for provenance, handling the material union."""
    d = {}
    for name, value in asdict(cfg).items():
        d[name] = value
    if isinstance(cfg.material, OrthotropicMaterial):
        d["material"] = asdict(cfg.material)
    return deepcopy(d)


class BvidAnalysis:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    def run(self) -> AnalysisResults:
        mat = _resolve_material(self.config.material)
        lam = Laminate(
            material=mat,
            layup_deg=self.config.layup_deg,
            ply_thickness_mm=self.config.ply_thickness_mm,
        )
        damage = self._resolve_damage(lam)
        sigma_0 = _pristine_strength(lam, self.config.loading)
        notes: list[str] = []
        warnings_tags: list[str] = []

        if self.config.tier == "empirical":
            sigma = self._empirical(lam, damage, sigma_0)
            buckling_eigs = None
            critical_interface = None
            field_results = None
        elif self.config.tier == "semi_analytical":
            sa_result = self._semi_analytical(lam, damage, sigma_0)
            sigma = sa_result.residual_strength_MPa
            critical_interface = sa_result.critical_interface_index
            N_cr = sa_result.critical_buckling_load_N
            buckling_eigs = [N_cr] if N_cr is not None else None
            field_results = None
        elif self.config.tier == "fe3d":
            if self.config.loading == "compression":
                # Buckling delegates to the Rayleigh-Ritz closed form
                # (#129); only a degenerate input (returned via the
                # ``buckling_notes`` channel) triggers the pristine
                # fallback. The implausibility guard previously here
                # existed because the old 3D K_g eigensolve mis-predicted
                # by ~100x and we needed to discard its result; with the
                # closed-form delegation a sub-5%-of-pristine answer is
                # the physically expected outcome for slender panels and
                # must be kept.
                sigma_buckling, lambda_crit, buckling_notes = fe3d_cai_buckling(
                    self.config, damage, lam, sigma_0
                )
                notes.extend(buckling_notes)
                if buckling_notes:
                    warnings_tags.append("fe3d_buckling_fallback")
                sigma_fpf = _fe3d_cai_first_ply_failure(self.config, damage, lam, sigma_0)
                sigma = min(sigma_buckling, sigma_fpf)
                buckling_eigs = [lambda_crit] if lambda_crit > 0 else None
            else:
                sigma = fe3d_tai(self.config, damage, lam, sigma_0)
                buckling_eigs = None
            critical_interface = None
            field_results = None
        else:
            raise NotImplementedError(f"tier '{self.config.tier}' is not recognized")

        return AnalysisResults(
            residual_strength_MPa=sigma,
            pristine_strength_MPa=sigma_0,
            knockdown=sigma / sigma_0 if sigma_0 > 0 else 0.0,
            damage=damage,
            dpa_mm2=damage.projected_damage_area_mm2,
            tier_used=self.config.tier,
            config_snapshot=_config_snapshot_dict(self.config),
            buckling_eigenvalues=buckling_eigs,
            critical_sublaminate=critical_interface,
            field_results=field_results,
            notes=notes,
            warnings=warnings_tags,
        )

    def _resolve_damage(self, lam: Laminate) -> DamageState:
        if self.config.damage is not None:
            return self.config.damage
        assert self.config.impact is not None  # asserted in __post_init__
        return impact_to_damage(self.config.impact, lam, self.config.panel)

    def _semi_analytical(
        self, lam: Laminate, damage: DamageState, sigma_0: float
    ) -> SemiAnalyticalResult:
        A_panel = self.config.panel.Lx_mm * self.config.panel.Ly_mm
        if self.config.loading == "compression":
            return semi_analytical_cai(
                lam,
                damage,
                sigma_0,
                A_panel,
                boundary=self.config.panel.boundary,
            )
        # tension: no sublaminate buckling channel — wrap the scalar result
        # so callers consume a uniform ``SemiAnalyticalResult`` shape.
        sigma = semi_analytical_tai(lam, damage, sigma_0)
        return SemiAnalyticalResult(
            residual_strength_MPa=sigma,
            critical_interface_index=None,
            critical_buckling_load_N=None,
        )

    def _empirical(self, lam: Laminate, damage: DamageState, sigma_0: float) -> float:
        A_panel = self.config.panel.Lx_mm * self.config.panel.Ly_mm
        dpa = damage.projected_damage_area_mm2
        mat = lam.material
        if self.config.loading == "compression":
            return soutis_cai(mat, dpa, A_panel, sigma_0)
        Kt_inf = lekhnitskii_kt_infinity(lam)
        return whitney_nuismer_tai(mat, dpa, sigma_0, Kt_inf)

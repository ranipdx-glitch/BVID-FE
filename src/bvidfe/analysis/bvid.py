"""BvidAnalysis: orchestrates impact-to-damage and tier dispatch."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict
from typing import Union

from bvidfe.analysis.config import AnalysisConfig
from bvidfe.analysis.results import AnalysisResults
from bvidfe.analysis.fe_tier import (
    _fe3d_cai_first_ply_failure,
    fe3d_cai_buckling,
    fe3d_tai,
)
from bvidfe.analysis.semi_analytical import (
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


def _pristine_strength(lam: Laminate, loading: str) -> float:
    """Thickness-weighted ply-average pristine strength in the loading direction.

    For compression: sum_i t_i * (Xc*cos^2 + Yc*sin^2) / sum_i t_i
    For tension:     sum_i t_i * (Xt*cos^2 + Yt*sin^2) / sum_i t_i
    """
    m = lam.material
    total_t = 0.0
    num = 0.0
    for theta in lam.layup_deg:
        c2 = math.cos(math.radians(theta)) ** 2
        s2 = math.sin(math.radians(theta)) ** 2
        if loading == "compression":
            num += lam.ply_thickness_mm * (m.Xc * c2 + m.Yc * s2)
        else:
            num += lam.ply_thickness_mm * (m.Xt * c2 + m.Yt * s2)
        total_t += lam.ply_thickness_mm
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
            sigma, critical_interface, N_cr = self._semi_analytical(lam, damage, sigma_0)
            buckling_eigs = [N_cr] if N_cr is not None else None
            field_results = None
        elif self.config.tier == "fe3d":
            if self.config.loading == "compression":
                # Run both paths, but only use the buckling stress if it is
                # physically plausible. v0.2.0-dev's buckling model uses a
                # simplified uniform-pre-stress approximation with 3-point
                # rigid-body BCs; on realistic panels the absolute eigenvalue
                # can be dramatically off from analytical plate buckling. If
                # it's less than 5% of pristine we treat it as a numerical
                # artefact and fall back to FPF. A future release will wire
                # proper in-plane pre-stress BCs into the buckling path.
                sigma_buckling, lambda_crit, buckling_notes = fe3d_cai_buckling(
                    self.config, damage, lam, sigma_0
                )
                notes.extend(buckling_notes)
                if buckling_notes:
                    warnings_tags.append("fe3d_buckling_unconverged")
                sigma_fpf = _fe3d_cai_first_ply_failure(self.config, damage, lam, sigma_0)
                buckling_plausible = sigma_buckling >= 0.05 * sigma_0
                if not buckling_plausible and not buckling_notes:
                    # Buckling solve completed but produced an implausibly small
                    # result — surface that the FPF path is what the user is
                    # actually seeing. (If `buckling_notes` is already populated
                    # the eigensolve reported its own failure; no need to layer
                    # a second note for the same root cause.)
                    notes.append(
                        f"fe3d buckling: result {sigma_buckling:.1f} MPa is below "
                        f"5% of pristine ({sigma_0:.1f} MPa); discarded as a "
                        "numerical artefact and used first-ply-failure instead."
                    )
                    warnings_tags.append("fe3d_buckling_artefact_dropped")
                sigma = min(sigma_buckling, sigma_fpf) if buckling_plausible else sigma_fpf
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

    def _semi_analytical(self, lam: Laminate, damage: DamageState, sigma_0: float):
        A_panel = self.config.panel.Lx_mm * self.config.panel.Ly_mm
        if self.config.loading == "compression":
            return semi_analytical_cai(
                lam,
                damage,
                sigma_0,
                A_panel,
                boundary=self.config.panel.boundary,
            )
        # tension
        sigma = semi_analytical_tai(lam, damage, sigma_0)
        return sigma, None, None

    def _empirical(self, lam: Laminate, damage: DamageState, sigma_0: float) -> float:
        A_panel = self.config.panel.Lx_mm * self.config.panel.Ly_mm
        dpa = damage.projected_damage_area_mm2
        mat = lam.material
        if self.config.loading == "compression":
            return soutis_cai(mat, dpa, A_panel, sigma_0)
        Kt_inf = lekhnitskii_kt_infinity(lam)
        return whitney_nuismer_tai(mat, dpa, sigma_0, Kt_inf)

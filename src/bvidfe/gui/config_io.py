"""Serialise/deserialise AnalysisConfig to/from a JSON-safe dict."""

from __future__ import annotations

from typing import Any, Dict

from bvidfe.analysis import AnalysisConfig, MeshParams
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.damage.io import damage_state_from_dict, damage_state_to_dict
from bvidfe.impact.mapping import ImpactEvent


def config_to_dict(cfg: AnalysisConfig) -> Dict[str, Any]:
    """Convert an AnalysisConfig into a JSON-serialisable dict.

    ``ply_thickness_mm`` is preserved as either a scalar or a list of floats,
    matching whatever shape the original config used.
    """
    if isinstance(cfg.ply_thickness_mm, (list, tuple)):
        ply_thickness_serialised: Any = [float(t) for t in cfg.ply_thickness_mm]
    else:
        ply_thickness_serialised = float(cfg.ply_thickness_mm)
    out: Dict[str, Any] = {
        "material": cfg.material if isinstance(cfg.material, str) else cfg.material.name,
        "layup_deg": list(cfg.layup_deg),
        "ply_thickness_mm": ply_thickness_serialised,
        "panel": {
            "Lx_mm": cfg.panel.Lx_mm,
            "Ly_mm": cfg.panel.Ly_mm,
            "boundary": cfg.panel.boundary,
        },
        "loading": cfg.loading,
        "tier": cfg.tier,
    }
    if cfg.impact is not None:
        out["impact"] = {
            "energy_J": cfg.impact.energy_J,
            "impactor": {
                "diameter_mm": cfg.impact.impactor.diameter_mm,
                "shape": cfg.impact.impactor.shape,
            },
            "mass_kg": cfg.impact.mass_kg,
            "location_xy_mm": list(cfg.impact.location_xy_mm),
        }
    if cfg.damage is not None:
        out["damage"] = damage_state_to_dict(cfg.damage)
    if cfg.mesh is not None:
        out["mesh"] = {
            "elements_per_ply": cfg.mesh.elements_per_ply,
            "in_plane_size_mm": cfg.mesh.in_plane_size_mm,
            "cohesive_zone_factor": cfg.mesh.cohesive_zone_factor,
        }
    return out


def config_from_dict(d: Dict[str, Any]) -> AnalysisConfig:
    """Reconstruct an AnalysisConfig from a dict produced by :func:`config_to_dict`."""
    panel = PanelGeometry(
        Lx_mm=float(d["panel"]["Lx_mm"]),
        Ly_mm=float(d["panel"]["Ly_mm"]),
        boundary=d["panel"].get("boundary", "simply_supported"),
    )
    impact = None
    damage = None
    if "impact" in d and d["impact"] is not None:
        i = d["impact"]
        impact = ImpactEvent(
            energy_J=float(i["energy_J"]),
            impactor=ImpactorGeometry(
                diameter_mm=float(i["impactor"]["diameter_mm"]),
                shape=i["impactor"].get("shape", "hemispherical"),
            ),
            mass_kg=float(i["mass_kg"]),
            location_xy_mm=tuple(i.get("location_xy_mm", (0.0, 0.0))),
        )
    if "damage" in d and d["damage"] is not None:
        damage = damage_state_from_dict(d["damage"])
    mesh = None
    if "mesh" in d and d["mesh"] is not None:
        m = d["mesh"]
        mesh = MeshParams(
            elements_per_ply=int(m.get("elements_per_ply", 4)),
            in_plane_size_mm=float(m.get("in_plane_size_mm", 1.0)),
            cohesive_zone_factor=float(m.get("cohesive_zone_factor", 1.0)),
        )
    raw_t = d["ply_thickness_mm"]
    if isinstance(raw_t, (list, tuple)):
        ply_thickness_mm: Any = [float(t) for t in raw_t]
    else:
        ply_thickness_mm = float(raw_t)
    return AnalysisConfig(
        material=d["material"],
        layup_deg=list(d["layup_deg"]),
        ply_thickness_mm=ply_thickness_mm,
        panel=panel,
        loading=d["loading"],
        tier=d["tier"],
        impact=impact,
        damage=damage,
        mesh=mesh,
    )

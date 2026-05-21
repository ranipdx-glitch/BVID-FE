"""Streamlit web app for BVID-FE residual-strength analysis.

Wraps ``bvidfe.analysis.BvidAnalysis`` in a browser UI: sidebar inputs
build an ``AnalysisConfig``, the main area shows results across tabs
(Summary, Damage Map, Knockdown Curve, 3D Damage, Buckling, Damage
Severity, Sweep).

Run locally:
    streamlit run app.py

Deploy: see DEPLOYMENT_STREAMLIT.md.
"""

from __future__ import annotations

import io
import json
import math
import sys
import warnings
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; required on Streamlit Cloud

import matplotlib.pyplot as plt  # noqa: E402  (matplotlib.use must run first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# Make the src-layout package importable on Streamlit Cloud, which clones
# the repo but does not necessarily editable-install it before the first
# script run.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bvidfe.analysis import AnalysisConfig, BvidAnalysis, MeshParams  # noqa: E402
from bvidfe.analysis.fe_mesh import build_fe_mesh, estimate_fe_mesh_size  # noqa: E402
from bvidfe.analysis.fe_tier import FE3D_MAX_DOF  # noqa: E402
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry  # noqa: E402
from bvidfe.core.material import MATERIAL_LIBRARY  # noqa: E402
from bvidfe.damage.io import CScanSchemaError, damage_state_from_dict  # noqa: E402
from bvidfe.damage.state import DamageState, DelaminationEllipse  # noqa: E402
from bvidfe.impact.mapping import ImpactEvent, impact_to_damage  # noqa: E402
from bvidfe.impact.olsson import onset_energy  # noqa: E402
from bvidfe.core.laminate import Laminate  # noqa: E402
from bvidfe.sweep.parametric_sweep import sweep_energies  # noqa: E402
from bvidfe.viz.plots_2d import plot_damage_map, plot_knockdown_curve  # noqa: E402
from bvidfe.viz.plotly_3d import mesh_damage_figure  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BVID-FE",
    page_icon=":airplane:",
    layout="wide",
)

st.title("BVID-FE — Barely Visible Impact Damage analysis")
st.caption(
    "Predicts residual strength and stiffness of composite laminates "
    "after low-velocity impact. Configure parameters in the sidebar and "
    "click **Run analysis**."
)


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

MATERIAL_NAMES: list[str] = sorted(MATERIAL_LIBRARY.keys())
DEFAULT_MATERIAL = "IM7/8552" if "IM7/8552" in MATERIAL_NAMES else MATERIAL_NAMES[0]
TIERS: list[str] = ["empirical", "semi_analytical", "fe3d"]
BOUNDARIES: list[str] = ["simply_supported", "clamped", "free"]
IMPACTOR_SHAPES: list[str] = ["hemispherical", "flat", "conical"]


MAX_PLIES = 400
MAX_SWEEP_POINTS = 25
DPA_SATURATION_PCT = 79.0

# MRB / NCR disposition support -------------------------------------------
# "Barely visible" impact damage is conventionally bounded by a residual
# surface-dent detectability threshold; 0.30 mm (~0.012 in) is a common
# design damage-tolerance value. Each program MUST override this with its
# own damage-tolerance substantiation — it is exposed as a UI input.
BVID_DENT_THRESHOLD_MM = 0.30

NC_TYPES: list[str] = [
    "Impact damage (BVID/VID)",
    "Delamination / disbond",
    "Porosity / voids",
    "Foreign object / inclusion",
    "Resin-rich / resin-starved",
    "Wrinkle / fiber waviness",
    "Out-of-tolerance geometry",
    "Other",
]

FOUND_DURING: list[str] = [
    "In-process inspection",
    "Final inspection",
    "Assembly / installation",
    "In-service / field inspection",
    "Receiving inspection",
    "Other",
]

_DF_FORMATS = {
    "energy_J": "{:g}",
    "knockdown": "{:.3f}",
    "residual_MPa": "{:.1f}",
    "pristine_MPa": "{:.1f}",
    "dpa_mm2": "{:.1f}",
    "dent_mm": "{:.3f}",
}


# ---------------------------------------------------------------------------
# Presets and shareable URL state
# ---------------------------------------------------------------------------
#
# Presets are canonical sidebar configurations. Picking a preset writes its
# values into ``st.session_state`` BEFORE the widgets render (so the widgets
# pick them up via their ``key=`` arg). A "preset-applied" guard
# (``_last_preset_key``) prevents a re-applied preset from clobbering user
# edits on every rerun — only a *change* in preset selection triggers a
# rewrite of the controlled keys.
#
# URL state uses short keys (``mat``, ``lay``, ``lx``, ``ly``, ``E``, ``d``,
# ``t``, ``ld``) so a shared link stays well under the 2 kB safe ceiling for
# common browsers. On app load, if the URL carries known keys, we hydrate
# session state from them BEFORE widgets render (same mechanism as presets).

PRESET_CUSTOM = "Custom"
PRESETS: dict[str, dict] = {
    PRESET_CUSTOM: {},  # sentinel — no overrides
    "Empirical quick (IM7/8552 quasi-iso, 20 J)": {
        "material_choice": "IM7/8552",
        "layup_text": "0, 45, -45, 90, 90, -45, 45, 0",
        "Lx_mm": 150.0,
        "Ly_mm": 100.0,
        "energy_J": 20.0,
        "diameter_mm": 16.0,
        "tier": "empirical",
        "loading": "compression",
        "input_mode": "Impact event",
    },
    "Semi-analytical buckling (IM7/8552 quasi-iso, 10 J)": {
        "material_choice": "IM7/8552",
        "layup_text": "0, 45, -45, 90, 90, -45, 45, 0",
        "Lx_mm": 150.0,
        "Ly_mm": 100.0,
        "energy_J": 10.0,
        "diameter_mm": 16.0,
        "tier": "semi_analytical",
        "loading": "compression",
        "input_mode": "Impact event",
    },
    "FE3D detailed (T800/epoxy, 15 J, coarse mesh)": {
        "material_choice": "T800/epoxy",
        "layup_text": "0, 45, -45, 90, 90, -45, 45, 0",
        "Lx_mm": 150.0,
        "Ly_mm": 100.0,
        "energy_J": 15.0,
        "diameter_mm": 16.0,
        "tier": "fe3d",
        "loading": "compression",
        "input_mode": "Impact event",
    },
}

PRESET_NAMES: list[str] = list(PRESETS.keys())

# Widget keys we control via preset / URL hydration. Each maps a sidebar
# widget's ``key`` argument to the type its widget expects in session_state.
_URL_TO_STATE: dict[str, tuple[str, type]] = {
    "mat": ("material_choice", str),
    "lay": ("layup_text", str),
    "lx": ("Lx_mm", float),
    "ly": ("Ly_mm", float),
    "E": ("energy_J", float),
    "d": ("diameter_mm", float),
    "t": ("tier", str),
    "ld": ("loading", str),
}


def _coerce_layup_url(text: str) -> str:
    """Sanity-check a URL-supplied layup string before pushing into a widget.

    Falls back to the default if the string parses to nothing valid; the
    widget itself will surface a friendly error on bad user-edited text but
    we don't want a malformed URL to crash before widgets even render.
    """
    try:
        angles = _parse_layup(text)
    except ValueError:
        return text  # let the widget echo the original; user can fix it
    if not angles:
        return text
    return ", ".join(f"{a:g}" for a in angles)


def _apply_state_overrides(overrides: dict) -> None:
    """Write controlled overrides into ``st.session_state`` with type coercion.

    Skips unknown keys and bad casts silently — the URL/preset path must be
    permissive so a stale shared link still loads the app.
    """
    type_map = {state_key: cast for (_, (state_key, cast)) in _URL_TO_STATE.items()}
    # widgets controlled only by presets (no URL key) — keep validation here.
    extra_types = {"input_mode": str}
    for key, value in overrides.items():
        cast = type_map.get(key, extra_types.get(key))
        if cast is None:
            continue
        try:
            coerced = cast(value)
        except (TypeError, ValueError):
            continue
        if key == "layup_text":
            coerced = _coerce_layup_url(str(coerced))
        if key == "material_choice" and coerced not in MATERIAL_NAMES:
            continue
        if key == "tier" and coerced not in TIERS:
            continue
        if key == "loading" and coerced not in ("compression", "tension"):
            continue
        if key == "input_mode" and coerced not in ("Impact event", "C-scan inspection"):
            continue
        st.session_state[key] = coerced


def _hydrate_from_url() -> None:
    """Populate session state from ``st.query_params`` on first script run.

    Only runs once per session (guarded by ``_url_hydrated``) so user edits
    after load are not overwritten by the original URL on every rerun.
    """
    if st.session_state.get("_url_hydrated"):
        return
    st.session_state["_url_hydrated"] = True
    qp = st.query_params
    overrides: dict = {}
    for url_key, (state_key, _cast) in _URL_TO_STATE.items():
        if url_key in qp:
            overrides[state_key] = qp[url_key]
    if overrides:
        _apply_state_overrides(overrides)


def _hydrate_from_preset() -> None:
    """Apply preset overrides when the user picks a non-Custom preset.

    Only fires when the selection *changes* (tracked in ``_last_preset_key``);
    re-runs with the same selection are no-ops, so user edits after applying
    a preset survive subsequent script reruns.
    """
    chosen = st.session_state.get("preset_choice", PRESET_CUSTOM)
    last = st.session_state.get("_last_preset_key")
    if chosen == last:
        return
    st.session_state["_last_preset_key"] = chosen
    if chosen == PRESET_CUSTOM:
        return
    _apply_state_overrides(PRESETS[chosen])


def _sync_query_params() -> None:
    """Write the live sidebar config back to ``st.query_params``.

    Only the minimal-shareable subset (material, layup, panel, impact
    energy, impactor diameter, tier, loading) is encoded — short keys keep
    URLs under the ~2 kB safe ceiling. Values are stringified; Streamlit
    handles URL escaping for us.
    """
    try:
        new_qp = {
            "mat": str(st.session_state.get("material_choice", DEFAULT_MATERIAL)),
            "lay": str(st.session_state.get("layup_text", "")),
            "lx": f"{float(st.session_state.get('Lx_mm', 150.0)):g}",
            "ly": f"{float(st.session_state.get('Ly_mm', 100.0)):g}",
            "E": f"{float(st.session_state.get('energy_J', 30.0)):g}",
            "d": f"{float(st.session_state.get('diameter_mm', 16.0)):g}",
            "t": str(st.session_state.get("tier", "empirical")),
            "ld": str(st.session_state.get("loading", "compression")),
        }
    except (TypeError, ValueError):
        return
    # Only write if the dict actually changes — avoids needless reruns.
    current = {k: st.query_params.get(k) for k in new_qp}
    if current != new_qp:
        try:
            st.query_params.from_dict(new_qp)
        except Exception:
            # Streamlit's query-param API is occasionally fussy on stale
            # sessions; never let URL syncing crash the app.
            pass


def _show_df(df: pd.DataFrame) -> None:
    """Render a results DataFrame with sane float formatting."""
    fmt = {c: f for c, f in _DF_FORMATS.items() if c in df.columns}
    st.dataframe(df.style.format(fmt, na_rep="—"), use_container_width=True)


def _parse_layup(text: str) -> list[float]:
    """Parse a comma- or whitespace-separated layup string into floats.

    Rejects non-finite values (nan/inf), out-of-range angles, and absurdly
    long stacks so a malformed paste can't produce NaN ABD matrices or
    exhaust memory.
    """
    cleaned = text.replace(";", ",").replace("\n", ",")
    tokens = [x.strip() for x in cleaned.split(",") if x.strip()]
    if len(tokens) > MAX_PLIES:
        raise ValueError(f"layup has {len(tokens)} plies; the cap is {MAX_PLIES}")
    angles = [float(t) for t in tokens]
    for a in angles:
        if not math.isfinite(a):
            raise ValueError("ply angles must be finite numbers")
        if not -90.0 <= a <= 90.0:
            raise ValueError("ply angles must be within [-90, 90] degrees")
    return angles


@st.cache_data(show_spinner=False, max_entries=8)
def _run_analysis_cached(config_dict: dict) -> dict:
    """Run BvidAnalysis from a JSON-safe config dict and return a dict.

    Caching key is the config dict itself, so wiggling unrelated UI
    widgets won't re-trigger a 30-second fe3d solve. ``st.cache_data``
    returns a fresh copy on each access, so the contained objects are
    safe for the result tabs to consume. ``max_entries`` bounds memory
    on the 1 GB Streamlit Cloud tier. The display mesh is built here
    (once per unique config) instead of on every rerun in two tabs.
    """
    cfg = _config_from_dict(config_dict)
    result = BvidAnalysis(cfg).run()
    cfg_mesh = cfg if cfg.mesh is not None else replace(cfg, mesh=MeshParams())
    try:
        mesh = build_fe_mesh(cfg_mesh, result.damage)
    except Exception:
        # A display-mesh failure must not sink the numeric result; the
        # 3D/Severity tabs surface the absence with a friendly message.
        mesh = None
    return {
        "result_dict": result.to_dict(),
        "damage": result.damage,
        "config": cfg,
        "config_dict": config_dict,
        "mesh_config": cfg_mesh,
        "mesh": mesh,
        "buckling_eigenvalues": result.buckling_eigenvalues,
        "notes": list(result.notes),
        "summary_text": result.summary(),
    }


@st.cache_data(show_spinner=False, max_entries=16)
def _sweep_cached(config_dict: dict, tier: str, energies: tuple[float, ...]) -> pd.DataFrame:
    """Energy sweep for a given tier, cached and resilient to per-point failures.

    ``on_error="warn"`` makes the library return NaN rows for points that
    fail instead of aborting the whole sweep.
    """
    cfg = replace(_config_from_dict(config_dict), tier=tier)
    return sweep_energies(cfg, list(energies), on_error="warn", progress_callback=None)


@st.cache_data(show_spinner=False, max_entries=32)
def _onset_preview_cached(
    material: str,
    layup: tuple[float, ...],
    ply_thickness_mm: float,
    Lx_mm: float,
    Ly_mm: float,
    boundary: str,
    energy_J: float,
    diameter_mm: float,
    shape: str,
    mass_kg: float,
    loc_xy: tuple[float, float],
) -> str | None:
    """Cached Olsson onset/DPA preview so it isn't recomputed every keystroke."""
    try:
        mat = MATERIAL_LIBRARY[material]
        lam = Laminate(mat, list(layup), ply_thickness_mm)
        pan = PanelGeometry(Lx_mm=Lx_mm, Ly_mm=Ly_mm, boundary=boundary)
        impactor = ImpactorGeometry(diameter_mm=diameter_mm, shape=shape)
        ev = ImpactEvent(
            energy_J=energy_J,
            impactor=impactor,
            mass_kg=mass_kg,
            location_xy_mm=loc_xy,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E_onset = onset_energy(lam, pan, impactor)
            dmg = impact_to_damage(ev, lam, pan)
        A_panel = Lx_mm * Ly_mm
        pct = 100.0 * dmg.projected_damage_area_mm2 / A_panel if A_panel else 0.0
        flag = " ⚠ SATURATED" if pct >= DPA_SATURATION_PCT else ""
        return (
            f"E_onset ≈ **{E_onset:.2f} J** · "
            f"predicted DPA ≈ **{dmg.projected_damage_area_mm2:.0f} mm²** "
            f"({pct:.1f}% of panel){flag}"
        )
    except Exception:
        return None


def _config_from_dict(d: dict) -> AnalysisConfig:
    panel = PanelGeometry(
        Lx_mm=float(d["panel"]["Lx_mm"]),
        Ly_mm=float(d["panel"]["Ly_mm"]),
        boundary=d["panel"].get("boundary", "simply_supported"),
    )
    impact = None
    damage = None
    if d.get("impact") is not None:
        i = d["impact"]
        impact = ImpactEvent(
            energy_J=float(i["energy_J"]),
            impactor=ImpactorGeometry(
                diameter_mm=float(i["impactor"]["diameter_mm"]),
                shape=i["impactor"]["shape"],
            ),
            mass_kg=float(i["mass_kg"]),
            location_xy_mm=tuple(i["location_xy_mm"]),
        )
    if d.get("damage") is not None:
        dd = d["damage"]
        damage = DamageState(
            delaminations=[
                DelaminationEllipse(
                    interface_index=int(e["interface_index"]),
                    centroid_mm=tuple(e["centroid_mm"]),
                    major_mm=float(e["major_mm"]),
                    minor_mm=float(e["minor_mm"]),
                    orientation_deg=float(e["orientation_deg"]),
                )
                for e in dd.get("delaminations", [])
            ],
            dent_depth_mm=float(dd.get("dent_depth_mm", 0.0)),
            fiber_break_radius_mm=float(dd.get("fiber_break_radius_mm", 0.0)),
        )
    mesh = None
    if d.get("mesh") is not None:
        m = d["mesh"]
        mesh = MeshParams(
            elements_per_ply=int(m.get("elements_per_ply", 1)),
            in_plane_size_mm=float(m.get("in_plane_size_mm", 5.0)),
            cohesive_zone_factor=float(m.get("cohesive_zone_factor", 1.0)),
        )
    return AnalysisConfig(
        material=d["material"],
        layup_deg=list(d["layup_deg"]),
        ply_thickness_mm=float(d["ply_thickness_mm"]),
        panel=panel,
        loading=d["loading"],
        tier=d["tier"],
        impact=impact,
        damage=damage,
        mesh=mesh,
    )


# ---------------------------------------------------------------------------
# NCR (Nonconformance Report) export — MRB disposition support
# ---------------------------------------------------------------------------


def _recommend_disposition(result_dict: dict, dent_threshold_mm: float) -> dict:
    """Heuristic MRB disposition *recommendation* from BVID-FE outputs.

    This is structured decision support only: it bands the predicted
    strength retention, raises the governing flags, and cites the criteria
    a board must check. It does NOT issue an approved disposition — a
    qualified Material Review Board must review, modify, and sign.
    """
    kd = float(result_dict["knockdown"])
    dent = float(result_dict["damage"]["dent_depth_mm"])
    fbr = float(result_dict["damage"]["fiber_break_radius_mm"])
    n_delam = len(result_dict["damage"]["delaminations"])
    dpa = float(result_dict["dpa_mm2"])

    flags: list[str] = []
    if dent >= dent_threshold_mm:
        flags.append(
            f"Residual dent {dent:.3f} mm ≥ detectability threshold "
            f"{dent_threshold_mm:.3f} mm — damage is visible (VID, not "
            "BVID); disposition under the visible-damage damage-tolerance "
            "branch."
        )
    if fbr > 0.0:
        flags.append(
            f"Fiber breakage present (r ≈ {fbr:.2f} mm) — primary load "
            "path severed; Use-As-Is is not appropriate without structural "
            "substantiation."
        )
    if n_delam > 0:
        flags.append(
            f"{n_delam} delamination(s) modelled — assess sublaminate "
            "stability and growth under spectrum/limit load."
        )

    if fbr > 0.0:
        path = "Repair"
        band = "fiber breakage present"
    elif kd >= 0.97 and dent < dent_threshold_mm and n_delam == 0:
        path = "Use-As-Is (UAI)"
        band = f"knockdown {kd:.3f} ≥ 0.97, no delamination, sub-threshold dent"
    elif kd >= 0.85:
        path = "Use-As-Is (UAI) with engineering substantiation, or Repair"
        band = f"0.85 ≤ knockdown {kd:.3f} < 0.97"
    elif kd >= 0.70:
        path = "Repair"
        band = f"0.70 ≤ knockdown {kd:.3f} < 0.85"
    else:
        path = "Repair (structural) or Scrap"
        band = f"knockdown {kd:.3f} < 0.70"

    rationale = (
        f"Predicted residual strength retains {kd * 100:.1f}% of pristine "
        f"({result_dict['residual_strength_MPa']:.1f} MPa vs "
        f"{result_dict['pristine_strength_MPa']:.1f} MPa) at projected "
        f"damage area {dpa:.0f} mm² (analysis tier: "
        f"{result_dict['tier_used']}). Banding: {band}. The board must "
        "confirm a positive Margin of Safety against design ULTIMATE load "
        "(and LIMIT load with no detrimental deformation) for the as-found "
        "damage, including no-growth / arrested-growth substantiation over "
        "the service inspection interval."
    )

    criteria = [
        "Margin of Safety ≥ 0 vs design ultimate load for the as-found "
        "damage state (governing design load case).",
        "Damage-tolerance / no-growth substantiation per CMH-17 Vol. 3 "
        "guidance and the program structural substantiation document.",
        "BVID detectability basis (residual dent depth + NDI/tap-test) "
        f"against the {dent_threshold_mm:.3f} mm design threshold.",
        "Approved Structural Repair Manual (SRM) limits if a Repair path "
        "is selected; engineering disposition required if outside SRM.",
        "Control of nonconforming output per AS9100 §8.7 and the site MRB " "procedure.",
    ]

    return {
        "recommended_path": path,
        "rationale": rationale,
        "criteria": criteria,
        "flags": flags,
        "knockdown": kd,
        "status": "PROPOSED — REQUIRES MRB REVIEW AND APPROVAL",
    }


def _build_ncr(form: dict, cfg, result_dict: dict | None, rec: dict | None) -> tuple[str, dict]:
    """Assemble a structured NCR as (Markdown document, machine-readable dict)."""
    layup_str = "[" + "/".join(f"{a:g}" for a in cfg.layup_deg) + "]"

    analysis: dict | None = None
    if result_dict is not None:
        analysis = {
            "tool": "BVID-FE",
            "tier_used": result_dict["tier_used"],
            "pristine_strength_MPa": result_dict["pristine_strength_MPa"],
            "residual_strength_MPa": result_dict["residual_strength_MPa"],
            "knockdown": result_dict["knockdown"],
            "projected_damage_area_mm2": result_dict["dpa_mm2"],
            "dent_depth_mm": result_dict["damage"]["dent_depth_mm"],
            "fiber_break_radius_mm": result_dict["damage"]["fiber_break_radius_mm"],
            "n_delaminations": len(result_dict["damage"]["delaminations"]),
            "notes": list(result_dict.get("notes", [])),
            "warnings": list(result_dict.get("warnings", [])),
        }

    ncr = {
        "ncr_number": form["ncr_number"],
        "revision": form["revision"],
        "originated_by": form["originator"],
        "date": form["date"],
        "program": form["program"],
        "part_number": form["part_number"],
        "part_name": form["part_name"],
        "serial_or_lot": form["serial_lot"],
        "work_order": form["work_order"],
        "quantity_affected": form["quantity"],
        "found_at_station": form["station"],
        "found_during": form["found_during"],
        "nonconformance_type": form["nc_type"],
        "nonconformance_description": form["description"],
        "as_found_dimensions": form["dimensions"],
        "material_configuration": {
            "material": cfg.material,
            "layup_deg": list(cfg.layup_deg),
            "layup_string": layup_str,
            "ply_thickness_mm": cfg.ply_thickness_mm,
            "panel_Lx_mm": cfg.panel.Lx_mm,
            "panel_Ly_mm": cfg.panel.Ly_mm,
            "panel_boundary": cfg.panel.boundary,
        },
        "engineering_analysis": analysis,
        "proposed_disposition": rec,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    lines: list[str] = []
    lines.append(f"# Nonconformance Report — {form['ncr_number']}")
    lines.append("")
    lines.append(
        "> **Decision-support document.** The disposition below is a "
        "structured *recommendation* generated from BVID-FE analysis. It "
        "is NOT an approved disposition. A qualified Material Review Board "
        "must review, modify as required, and approve."
    )
    lines.append("")
    lines.append("## 1. Identification")
    lines.append("")
    lines.append(f"- **NCR number:** {form['ncr_number']}  (rev {form['revision']})")
    lines.append(f"- **Originated by:** {form['originator']}")
    lines.append(f"- **Date:** {form['date']}")
    lines.append(f"- **Program / project:** {form['program']}")
    lines.append(f"- **Part number:** {form['part_number']}")
    lines.append(f"- **Part name:** {form['part_name']}")
    lines.append(f"- **Serial / lot:** {form['serial_lot']}")
    lines.append(f"- **Work order:** {form['work_order']}")
    lines.append(f"- **Quantity affected:** {form['quantity']}")
    lines.append(f"- **Found at / station:** {form['station']}")
    lines.append(f"- **Found during:** {form['found_during']}")
    lines.append("")
    lines.append("## 2. Nonconformance")
    lines.append("")
    lines.append(f"- **Type:** {form['nc_type']}")
    lines.append(f"- **As-found dimensions:** {form['dimensions']}")
    lines.append("")
    lines.append(form["description"] or "_No description provided._")
    lines.append("")
    lines.append("## 3. Material / configuration")
    lines.append("")
    lines.append(f"- **Material:** {cfg.material}")
    lines.append(f"- **Layup:** {layup_str} ({len(cfg.layup_deg)} plies)")
    lines.append(f"- **Ply thickness:** {cfg.ply_thickness_mm:g} mm")
    lines.append(
        f"- **Panel:** {cfg.panel.Lx_mm:g} × {cfg.panel.Ly_mm:g} mm, " f"{cfg.panel.boundary}"
    )
    lines.append("")
    lines.append("## 4. Engineering analysis (BVID-FE)")
    lines.append("")
    if analysis is None:
        lines.append(
            "_Engineering analysis pending — run a BVID-FE analysis and "
            "regenerate this NCR to attach residual-strength results._"
        )
    else:
        lines.append(f"- **Analysis tier:** {analysis['tier_used']}")
        lines.append(f"- **Pristine strength:** {analysis['pristine_strength_MPa']:.1f} MPa")
        lines.append(f"- **Residual strength:** {analysis['residual_strength_MPa']:.1f} MPa")
        lines.append(f"- **Knockdown:** {analysis['knockdown']:.3f}")
        lines.append(
            f"- **Projected damage area:** " f"{analysis['projected_damage_area_mm2']:.0f} mm²"
        )
        lines.append(f"- **Dent depth:** {analysis['dent_depth_mm']:.3f} mm")
        lines.append(f"- **Fiber-break radius:** {analysis['fiber_break_radius_mm']:.2f} mm")
        lines.append(f"- **Delaminations modelled:** {analysis['n_delaminations']}")
        if analysis["notes"]:
            lines.append("- **Runtime notes:**")
            for n in analysis["notes"]:
                lines.append(f"  - {n}")
        if analysis["warnings"]:
            lines.append("- **Warnings:**")
            for w in analysis["warnings"]:
                lines.append(f"  - {w}")
    lines.append("")
    lines.append("## 5. Proposed disposition (requires MRB review)")
    lines.append("")
    if rec is None:
        lines.append("_No disposition recommendation — attach a BVID-FE analysis " "first._")
    else:
        lines.append(f"- **Status:** {rec['status']}")
        lines.append(f"- **Recommended path:** {rec['recommended_path']}")
        lines.append("")
        lines.append(f"**Rationale.** {rec['rationale']}")
        lines.append("")
        if rec["flags"]:
            lines.append("**Flags raised:**")
            for fl in rec["flags"]:
                lines.append(f"- {fl}")
            lines.append("")
        lines.append("**Criteria the board must verify:**")
        for c in rec["criteria"]:
            lines.append(f"- {c}")
        lines.append("")
    lines.append("## 6. MRB review & approval")
    lines.append("")
    lines.append("| Role | Name | Disposition concurrence | Signature | Date |")
    lines.append("|------|------|-------------------------|-----------|------|")
    lines.append("| Engineering (stress) |  |  |  |  |")
    lines.append("| Quality |  |  |  |  |")
    lines.append("| Materials & Processes |  |  |  |  |")
    lines.append("| MRB chair |  |  |  |  |")
    lines.append("")
    lines.append(f"_Generated {ncr['generated_utc']} UTC by BVID-FE._")
    lines.append("")

    return "\n".join(lines), ncr


# ---------------------------------------------------------------------------
# Sidebar — AnalysisConfig builder
# ---------------------------------------------------------------------------

# Hydrate session_state from URL on first run, then from preset whenever the
# preset selection changes. Both must run BEFORE the widgets render so the
# widgets pick up the new values via their ``key=`` arg.
_hydrate_from_url()
_hydrate_from_preset()

with st.sidebar:
    st.markdown("**Configuration preset**")
    st.selectbox(
        "Preset",
        PRESET_NAMES,
        index=(
            PRESET_NAMES.index(st.session_state.get("preset_choice", PRESET_CUSTOM))
            if st.session_state.get("preset_choice", PRESET_CUSTOM) in PRESET_NAMES
            else 0
        ),
        key="preset_choice",
        help=(
            "Pick a canonical configuration to populate the sidebar. "
            "Choose **Custom** to keep your current settings. Selecting a "
            "preset overwrites the affected widgets; subsequent edits stick."
        ),
        on_change=_hydrate_from_preset,
    )

    expert_mode = st.toggle(
        "Expert mode",
        value=False,
        help=(
            "**Off (default)** — simplified sidebar: material, layup, panel, "
            "impact event, tier, loading.\n\n"
            "**On** — full controls: ply thickness, boundary condition, "
            "impactor shape/mass, mesh density for fe3d."
        ),
    )

    st.markdown("**Material & layup**")
    if "material_choice" not in st.session_state:
        st.session_state["material_choice"] = DEFAULT_MATERIAL
    material_choice = st.selectbox(
        "Material",
        MATERIAL_NAMES,
        key="material_choice",
        help="Built-in carbon/epoxy preset from the BVID-FE material library.",
    )

    if expert_mode:
        ply_thickness_mm = st.number_input(
            "Ply thickness [mm]",
            min_value=0.01,
            max_value=2.0,
            value=0.152,
            step=0.001,
            format="%.4f",
            help="Cured ply thickness. Default 0.152 mm matches AS4/3501-6.",
        )
    else:
        ply_thickness_mm = 0.152

    if "layup_text" not in st.session_state:
        st.session_state["layup_text"] = "0, 45, -45, 90, 90, -45, 45, 0"
    layup_text = st.text_area(
        "Layup [deg]",
        key="layup_text",
        height=80,
        help="Comma-separated ply angles in degrees, mid-plane to outer surface.",
    )
    try:
        layup_deg = _parse_layup(layup_text)
        if not layup_deg:
            st.error("Layup is empty — enter at least one ply angle.")
    except ValueError as exc:
        layup_deg = []
        st.error(f"Invalid layup: {exc}")

    st.markdown("**Panel geometry**")
    if "Lx_mm" not in st.session_state:
        st.session_state["Lx_mm"] = 150.0
    if "Ly_mm" not in st.session_state:
        st.session_state["Ly_mm"] = 100.0
    col_lx, col_ly = st.columns(2)
    Lx_mm = col_lx.number_input("Lx [mm]", min_value=10.0, max_value=2000.0, step=10.0, key="Lx_mm")
    Ly_mm = col_ly.number_input("Ly [mm]", min_value=10.0, max_value=2000.0, step=10.0, key="Ly_mm")
    if expert_mode:
        boundary = st.selectbox(
            "Boundary condition",
            BOUNDARIES,
            index=0,
            help=(
                "Edge constraint applied by the Olsson onset model and the "
                "semi-analytical buckling solve. 'free' uses a 0.4× SSSS "
                "approximation — interpret qualitatively."
            ),
        )
    else:
        boundary = "simply_supported"

    st.markdown("**Input mode**")
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Impact event"
    input_mode = st.radio(
        "Damage source",
        ["Impact event", "C-scan inspection"],
        horizontal=False,
        key="input_mode",
        help=(
            "**Impact event** — predict damage from impact energy via "
            "Olsson's quasi-static threshold model.\n\n"
            "**C-scan inspection** — load measured delamination ellipses "
            "directly from a JSON C-scan file."
        ),
    )

    impact_event: ImpactEvent | None = None
    damage_state: DamageState | None = None

    if input_mode == "Impact event":
        st.markdown("**Impact event**")
        if "energy_J" not in st.session_state:
            st.session_state["energy_J"] = 30.0
        if "diameter_mm" not in st.session_state:
            st.session_state["diameter_mm"] = 16.0
        energy_J = st.number_input(
            "Energy [J]",
            min_value=0.1,
            max_value=200.0,
            step=1.0,
            key="energy_J",
            help="Kinetic energy of the impactor at contact.",
        )
        diameter_mm = st.number_input(
            "Impactor diameter [mm]",
            min_value=1.0,
            max_value=100.0,
            step=1.0,
            key="diameter_mm",
        )
        if expert_mode:
            impactor_shape = st.selectbox("Impactor shape", IMPACTOR_SHAPES, index=0)
            mass_kg = st.number_input(
                "Impactor mass [kg]",
                min_value=0.01,
                max_value=100.0,
                value=5.5,
                step=0.1,
            )
            col_lx_loc, col_ly_loc = st.columns(2)
            loc_x = col_lx_loc.number_input("Location X [mm]", value=0.0, step=1.0)
            loc_y = col_ly_loc.number_input("Location Y [mm]", value=0.0, step=1.0)
        else:
            impactor_shape = "hemispherical"
            mass_kg = 5.5
            loc_x = 0.0
            loc_y = 0.0
        impact_event = ImpactEvent(
            energy_J=energy_J,
            impactor=ImpactorGeometry(diameter_mm=diameter_mm, shape=impactor_shape),
            mass_kg=mass_kg,
            location_xy_mm=(loc_x, loc_y),
        )
    else:
        st.markdown("**C-scan inspection**")
        uploaded = st.file_uploader(
            "C-scan JSON",
            type=["json"],
            help="JSON conforming to docs/cscan_schema.md.",
        )
        dent_depth_mm = st.number_input(
            "Dent depth [mm]", min_value=0.0, max_value=10.0, value=0.0, step=0.05
        )
        fb_radius_mm = st.number_input(
            "Fiber-break radius [mm]",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
        )
        if uploaded is not None:
            try:
                # Parse in memory — never write an attacker-controlled
                # filename to disk (path-traversal write primitive).
                data = json.loads(uploaded.getvalue())
                damage_state = damage_state_from_dict(data)
                # Override dent / fiber-break with the sidebar inputs if the
                # user has typed values; otherwise keep the file's values.
                if dent_depth_mm > 0:
                    damage_state = replace(damage_state, dent_depth_mm=dent_depth_mm)
                if fb_radius_mm > 0:
                    damage_state = replace(damage_state, fiber_break_radius_mm=fb_radius_mm)
                st.success(f"Loaded {len(damage_state.delaminations)} delamination(s)")
            except (CScanSchemaError, json.JSONDecodeError, ValueError) as exc:
                st.error(f"Invalid C-scan JSON: {exc}")
                damage_state = None
        else:
            st.info(
                "Upload a C-scan JSON to populate the damage state. Until "
                "then the run button is disabled."
            )
            damage_state = None

    st.markdown("**Analysis**")
    if "tier" not in st.session_state:
        st.session_state["tier"] = TIERS[0]
    if "loading" not in st.session_state:
        st.session_state["loading"] = "compression"
    tier = st.selectbox(
        "Tier",
        TIERS,
        key="tier",
        help=(
            "**empirical** — Soutis closed-form CAI, milliseconds.\n\n"
            "**semi_analytical** — Rayleigh-Ritz sublaminate buckling + "
            "Soutis envelope, seconds.\n\n"
            "**fe3d** — 3D hex FE with LaRC05/Tsai-Wu, minutes; capped at "
            f"{FE3D_MAX_DOF:,} DOF."
        ),
    )
    loading = st.radio(
        "Loading mode",
        ["compression", "tension"],
        horizontal=True,
        key="loading",
        help="CAI = compression-after-impact, TAI = tension-after-impact.",
    )

    mesh_params: MeshParams | None = None
    if tier == "fe3d":
        with st.expander("Mesh parameters (fe3d only)", expanded=expert_mode):
            elements_per_ply = st.number_input(
                "Elements per ply", min_value=1, max_value=10, value=1, step=1
            )
            in_plane_size_mm = st.number_input(
                "In-plane size [mm]",
                min_value=0.5,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help=(
                    "Smaller values = finer mesh, more memory. On the cloud "
                    "tier (1 GB RAM) keep this >= 5 mm."
                ),
            )
            mesh_params = MeshParams(
                elements_per_ply=int(elements_per_ply),
                in_plane_size_mm=float(in_plane_size_mm),
            )

    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)


# Sync the live sidebar config into the URL so the page link is shareable.
# Runs once per script execution, after all widgets have committed their
# values to ``st.session_state``.
_sync_query_params()


# ---------------------------------------------------------------------------
# Validate config and live-preview E_onset before running
# ---------------------------------------------------------------------------

panel = None
config_dict: dict | None = None
config_valid = bool(layup_deg) and Lx_mm > 0 and Ly_mm > 0
if input_mode == "C-scan inspection" and damage_state is None:
    config_valid = False

if config_valid:
    panel = PanelGeometry(Lx_mm=float(Lx_mm), Ly_mm=float(Ly_mm), boundary=boundary)
    config_dict = {
        "material": material_choice,
        "layup_deg": layup_deg,
        "ply_thickness_mm": ply_thickness_mm,
        "panel": {
            "Lx_mm": panel.Lx_mm,
            "Ly_mm": panel.Ly_mm,
            "boundary": panel.boundary,
        },
        "loading": loading,
        "tier": tier,
        "impact": (
            None
            if impact_event is None
            else {
                "energy_J": impact_event.energy_J,
                "impactor": {
                    "diameter_mm": impact_event.impactor.diameter_mm,
                    "shape": impact_event.impactor.shape,
                },
                "mass_kg": impact_event.mass_kg,
                "location_xy_mm": list(impact_event.location_xy_mm),
            }
        ),
        "damage": (
            None
            if damage_state is None
            else {
                "dent_depth_mm": damage_state.dent_depth_mm,
                "fiber_break_radius_mm": damage_state.fiber_break_radius_mm,
                "delaminations": [
                    {
                        "interface_index": e.interface_index,
                        "centroid_mm": list(e.centroid_mm),
                        "major_mm": e.major_mm,
                        "minor_mm": e.minor_mm,
                        "orientation_deg": e.orientation_deg,
                    }
                    for e in damage_state.delaminations
                ],
            }
        ),
        "mesh": (
            None
            if mesh_params is None
            else {
                "elements_per_ply": mesh_params.elements_per_ply,
                "in_plane_size_mm": mesh_params.in_plane_size_mm,
            }
        ),
    }


# Persistent result across reruns.
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


def _live_onset_preview() -> str | None:
    """Olsson onset energy + DPA preview for the impact-driven path."""
    if input_mode != "Impact event" or panel is None or impact_event is None:
        return None
    return _onset_preview_cached(
        material_choice,
        tuple(layup_deg),
        float(ply_thickness_mm),
        float(panel.Lx_mm),
        float(panel.Ly_mm),
        panel.boundary,
        float(impact_event.energy_J),
        float(impact_event.impactor.diameter_mm),
        impact_event.impactor.shape,
        float(impact_event.mass_kg),
        tuple(float(v) for v in impact_event.location_xy_mm),
    )


preview = _live_onset_preview()
if preview:
    st.info(preview)


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

if run_clicked:
    if not config_valid or config_dict is None:
        st.error(
            "Configuration is incomplete — fix the highlighted sidebar inputs " "before running."
        )
    else:
        # fe3d mesh-size guard: estimate DOF count before the long-running
        # solve so we can refuse impossibly large meshes early.
        if tier == "fe3d":
            try:
                cfg_check = _config_from_dict(config_dict)
                stats = estimate_fe_mesh_size(cfg_check)
            except Exception as exc:
                st.error(f"Failed to estimate mesh size: {exc}")
                stats = None
            if stats and stats["n_dof"] > FE3D_MAX_DOF:
                st.error(
                    f"fe3d mesh has {stats['n_elements']:,} elements "
                    f"({stats['n_dof']:,} DOFs), exceeding the safe cap "
                    f"of {FE3D_MAX_DOF:,}. Increase **In-plane size** or "
                    f"reduce **Elements per ply**."
                )
                st.stop()
        with st.spinner("Running BVID analysis…"):
            try:
                payload = _run_analysis_cached(config_dict)
                st.session_state["last_result"] = payload
            except Exception as exc:
                st.session_state["last_result"] = None
                st.error(f"Analysis failed: {exc}")


# ---------------------------------------------------------------------------
# Results tabs
# ---------------------------------------------------------------------------

payload = st.session_state.get("last_result")

if payload is not None and config_dict is not None and config_dict != payload["config_dict"]:
    st.warning(
        "Sidebar inputs changed since the last run — the results below are "
        "stale. Click **Run analysis** to refresh."
    )

tabs = st.tabs(
    [
        "Summary",
        "Damage Map",
        "Knockdown Curve",
        "3D Damage",
        "Buckling",
        "Damage Severity",
        "Sweep",
        "Export NCR",
    ]
)


# --- Summary tab ----------------------------------------------------------

with tabs[0]:
    st.subheader("Summary")
    if payload is None:
        st.info("Run an analysis to see its summary here.")
    else:
        result_dict = payload["result_dict"]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Pristine strength",
            f"{result_dict['pristine_strength_MPa']:.1f} MPa",
        )
        col2.metric(
            "Residual strength",
            f"{result_dict['residual_strength_MPa']:.1f} MPa",
        )
        col3.metric("Knockdown", f"{result_dict['knockdown']:.3f}")
        st.text(payload["summary_text"])
        if payload["notes"]:
            st.warning("Runtime notes:\n\n" + "\n\n".join(payload["notes"]))

        st.download_button(
            "Download results JSON",
            data=json.dumps(result_dict, indent=2),
            file_name="bvidfe_results.json",
            mime="application/json",
        )


# --- Damage Map tab -------------------------------------------------------

with tabs[1]:
    st.subheader("Damage map")
    if payload is None:
        st.info("Run an analysis to see the damage map.")
    else:
        try:
            cfg = payload["config"]
            fig = plot_damage_map(payload["damage"], cfg.panel)
            st.pyplot(fig, use_container_width=False)
        except Exception as exc:
            st.error(f"Could not render the damage map: {exc}")


# --- Knockdown Curve tab --------------------------------------------------

with tabs[2]:
    st.subheader("Knockdown vs impact energy")
    if payload is None:
        st.info(
            "Run an impact-driven analysis to seed this view, or use the "
            "Sweep tab to compute a full curve."
        )
    elif payload["config"].impact is None:
        st.info(
            "Knockdown curves require an impact-driven configuration. "
            "Switch the sidebar to **Impact event** mode."
        )
    else:
        try:
            cfg = payload["config"]
            # Quick empirical sweep around the current energy — fast
            # (sub-second) and cached so it doesn't rerun every script pass.
            base_E = cfg.impact.energy_J
            sweep_E = tuple(np.linspace(max(1.0, base_E * 0.2), base_E * 2.0, 7).tolist())
            with st.spinner("Computing empirical knockdown sweep…"):
                df = _sweep_cached(payload["config_dict"], "empirical", sweep_E)
            ok = df.dropna(subset=["knockdown"])
            fig = plot_knockdown_curve(
                ok["energy_J"].tolist(),
                ok["knockdown"].tolist(),
                tier_label="empirical",
            )
            st.pyplot(fig, use_container_width=False)
            _show_df(df)
        except Exception as exc:
            st.error(f"Could not compute the knockdown curve: {exc}")


# --- 3D Damage tab --------------------------------------------------------

with tabs[3]:
    st.subheader("3D damaged mesh")
    if payload is None:
        st.info("Run an analysis to see the 3D damaged mesh.")
    else:
        mesh = payload["mesh"]
        if mesh is None:
            st.warning("The display mesh could not be built for this configuration.")
        else:
            try:
                st.plotly_chart(
                    mesh_damage_figure(mesh, title="Damage factor (1 = pristine, 0 = damaged)"),
                    use_container_width=True,
                )
                st.caption(
                    f"Mesh: {mesh.n_elements:,} elements, {mesh.n_nodes:,} nodes. "
                    "Hot-colored regions sit inside a delamination or fiber-break "
                    "footprint."
                )
            except Exception as exc:
                st.error(f"Could not render the 3D mesh: {exc}")


# --- Buckling tab ---------------------------------------------------------

with tabs[4]:
    st.subheader("Buckling eigenvalues")
    if payload is None:
        st.info("Run an analysis to see buckling eigenvalues.")
    else:
        eigs = payload["buckling_eigenvalues"]
        tier_used = payload["result_dict"]["tier_used"]
        if not eigs:
            st.info(
                f"Tier '{tier_used}' does not produce buckling eigenvalues. "
                "Switch to **semi_analytical** or **fe3d** to populate this "
                "tab."
            )
        else:
            df_eigs = pd.DataFrame(
                {
                    "mode": list(range(1, len(eigs) + 1)),
                    "buckling load factor [-]": eigs,
                }
            )
            st.bar_chart(df_eigs.set_index("mode"))
            st.dataframe(
                df_eigs.style.format({"buckling load factor [-]": "{:.4g}"}),
                use_container_width=True,
            )


# --- Damage Severity tab --------------------------------------------------

with tabs[5]:
    st.subheader("Damage severity (through-thickness sum)")
    if payload is None:
        st.info("Run an analysis to see the damage-severity heatmap.")
    else:
        mesh = payload["mesh"]
        if mesh is None:
            st.warning("The display mesh could not be built for this configuration.")
        else:
            try:
                cfg_mesh = payload["mesh_config"]
                in_plane_size = cfg_mesh.mesh.in_plane_size_mm
                nx = max(1, math.ceil(cfg_mesh.panel.Lx_mm / in_plane_size))
                ny = max(1, math.ceil(cfg_mesh.panel.Ly_mm / in_plane_size))
                nz = len(cfg_mesh.layup_deg) * cfg_mesh.mesh.elements_per_ply
                severity = (1.0 - mesh.damage_factors).reshape(nz, ny, nx).sum(axis=0)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(
                    severity,
                    origin="lower",
                    extent=(0, cfg_mesh.panel.Lx_mm, 0, cfg_mesh.panel.Ly_mm),
                    cmap="hot_r",
                    vmin=0.0,
                    vmax=max(1.0, float(severity.max())),
                )
                ax.set_xlabel("x [mm]")
                ax.set_ylabel("y [mm]")
                ax.set_title(
                    f"Damage severity — tier={payload['result_dict']['tier_used']}  "
                    f"KD={payload['result_dict']['knockdown']:.3f}"
                )
                ax.set_aspect("equal")
                fig.colorbar(im, ax=ax, label="Stacked OOP loss")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
            except Exception as exc:
                st.error(f"Could not render the damage-severity heatmap: {exc}")


# --- Sweep tab ------------------------------------------------------------

with tabs[6]:
    st.subheader("Parametric energy sweep")
    if not config_valid or config_dict is None:
        st.info("Fix the sidebar inputs before running a sweep.")
    elif input_mode != "Impact event":
        st.info("Energy sweeps require **Impact event** mode in the sidebar.")
    else:
        default_E = ", ".join(f"{v:g}" for v in [5, 10, 20, 30, 40])
        energies_text = st.text_input("Energies [J] (comma-separated)", value=default_E)
        sweep_tier = st.selectbox(
            "Tier for sweep",
            TIERS,
            index=0,
            help=(
                "Use **empirical** for fast scans; **fe3d** sweeps can take "
                "minutes per point and may exhaust cloud memory."
            ),
        )
        sweep_clicked = st.button("Run sweep", type="primary")
        if sweep_clicked:
            try:
                energies = [float(x.strip()) for x in energies_text.split(",") if x.strip()]
            except ValueError:
                st.error("Energies must be a comma-separated list of numbers.")
                energies = []
            if energies and not all(math.isfinite(e) and e > 0 for e in energies):
                st.error("Energies must be finite and positive.")
                energies = []
            if len(energies) > MAX_SWEEP_POINTS:
                st.error(
                    f"Sweep is capped at {MAX_SWEEP_POINTS} energy points "
                    f"(got {len(energies)}). Shorten the list."
                )
                energies = []
            if energies and sweep_tier == "fe3d":
                try:
                    stats = estimate_fe_mesh_size(
                        replace(_config_from_dict(config_dict), tier="fe3d")
                    )
                except Exception as exc:
                    st.error(f"Failed to estimate fe3d mesh size: {exc}")
                    energies = []
                else:
                    if stats["n_dof"] > FE3D_MAX_DOF:
                        st.error(
                            f"fe3d mesh has {stats['n_elements']:,} elements "
                            f"({stats['n_dof']:,} DOFs), exceeding the safe cap "
                            f"of {FE3D_MAX_DOF:,}. Increase **In-plane size** or "
                            f"reduce **Elements per ply** in the sidebar."
                        )
                        energies = []
            if energies:
                try:
                    with st.spinner(f"Running {len(energies)}-point {sweep_tier} sweep…"):
                        df = _sweep_cached(config_dict, sweep_tier, tuple(energies))
                except Exception as exc:
                    st.error(f"Sweep failed: {exc}")
                else:
                    if df["knockdown"].isna().all():
                        st.error(
                            "Every sweep point failed — check the sidebar "
                            "configuration and the energy values."
                        )
                    else:
                        failed = df.loc[df["knockdown"].isna(), "energy_J"].tolist()
                        if failed:
                            st.warning(
                                "Some sweep points failed and are blank: "
                                + ", ".join(f"{e:g} J" for e in failed)
                            )
                        _show_df(df)
                        ok = df.dropna(subset=["knockdown"])
                        fig = plot_knockdown_curve(
                            ok["energy_J"].tolist(),
                            ok["knockdown"].tolist(),
                            tier_label=sweep_tier,
                        )
                        st.pyplot(fig, use_container_width=False)
                        csv_buf = io.StringIO()
                        df.to_csv(csv_buf, index=False)
                        st.download_button(
                            "Download sweep CSV",
                            data=csv_buf.getvalue(),
                            file_name=f"bvidfe_sweep_{sweep_tier}.csv",
                            mime="text/csv",
                        )


# --- Export NCR tab -------------------------------------------------------

with tabs[7]:
    st.subheader("Create Nonconformance Report (NCR)")
    st.caption(
        "For an engineer in the field: log the nonconformance, attach the "
        "BVID-FE residual-strength analysis, and export a structured NCR "
        "with a **proposed** MRB disposition. The recommendation is "
        "decision support only — a qualified Material Review Board must "
        "review, modify, and approve it."
    )

    # The NCR needs a material/layup/panel configuration. Prefer the last
    # analysed config so results attach; fall back to the live sidebar
    # config so the NC can be logged before analysis.
    ncr_payload = st.session_state.get("last_result")
    if ncr_payload is not None:
        ncr_cfg = ncr_payload["config"]
        ncr_result = ncr_payload["result_dict"]
    elif config_dict is not None:
        ncr_cfg = _config_from_dict(config_dict)
        ncr_result = None
    else:
        ncr_cfg = None
        ncr_result = None

    if ncr_cfg is None:
        st.info(
            "Complete the sidebar configuration (material, layup, panel) " "before creating an NCR."
        )
    else:
        if ncr_result is None:
            st.warning(
                "No BVID-FE analysis attached — the NCR will record the "
                "nonconformance with engineering analysis marked pending. "
                "Run an analysis to attach residual strength and a "
                "disposition recommendation."
            )

        today = date.today()
        c1, c2, c3 = st.columns(3)
        with c1:
            ncr_number = st.text_input(
                "NCR number",
                value=f"NCR-{today:%Y%m%d}-001",
                key="ncr_number",
            )
            revision = st.text_input("Revision", value="-", key="ncr_rev")
            originator = st.text_input("Originated by (engineer)", value="", key="ncr_originator")
            ncr_date = st.date_input("Date", value=today, key="ncr_date")
        with c2:
            program = st.text_input("Program / project", value="", key="ncr_program")
            part_number = st.text_input("Part number", value="", key="ncr_pn")
            part_name = st.text_input("Part name", value="", key="ncr_part_name")
            serial_lot = st.text_input("Serial / lot", value="", key="ncr_serial")
        with c3:
            work_order = st.text_input("Work order", value="", key="ncr_wo")
            quantity = st.text_input("Quantity affected", value="1", key="ncr_qty")
            station = st.text_input("Found at / station", value="", key="ncr_station")
            found_during = st.selectbox(
                "Found during", FOUND_DURING, index=3, key="ncr_found_during"
            )

        nc_type = st.selectbox("Nonconformance type", NC_TYPES, index=0, key="ncr_nc_type")
        dimensions = st.text_input(
            "As-found dimensions / location",
            value="",
            placeholder="e.g. 25 mm dia. dent, 0.4 mm deep, 120 mm from edge",
            key="ncr_dims",
        )
        description = st.text_area(
            "Nonconformance description",
            value="",
            height=120,
            placeholder=(
                "Describe what was found, how, and any inspection (visual, "
                "tap-test, ultrasonic C-scan) performed."
            ),
            key="ncr_desc",
        )
        dent_threshold_mm = st.number_input(
            "BVID detectability threshold [mm]",
            min_value=0.05,
            max_value=5.0,
            value=BVID_DENT_THRESHOLD_MM,
            step=0.05,
            help=(
                "Residual dent depth above which damage is treated as "
                "visible (VID). Override with your program's "
                "damage-tolerance value."
            ),
            key="ncr_dent_thr",
        )

        form = {
            "ncr_number": ncr_number.strip() or f"NCR-{today:%Y%m%d}-001",
            "revision": revision.strip() or "-",
            "originator": originator.strip() or "—",
            "date": ncr_date.isoformat(),
            "program": program.strip() or "—",
            "part_number": part_number.strip() or "—",
            "part_name": part_name.strip() or "—",
            "serial_lot": serial_lot.strip() or "—",
            "work_order": work_order.strip() or "—",
            "quantity": quantity.strip() or "—",
            "station": station.strip() or "—",
            "found_during": found_during,
            "nc_type": nc_type,
            "dimensions": dimensions.strip() or "—",
            "description": description.strip(),
        }

        rec = (
            None
            if ncr_result is None
            else _recommend_disposition(ncr_result, float(dent_threshold_mm))
        )
        ncr_md, ncr_dict = _build_ncr(form, ncr_cfg, ncr_result, rec)

        if rec is not None:
            st.success(
                f"Proposed disposition path: **{rec['recommended_path']}** " f"— {rec['status']}"
            )
            if rec["flags"]:
                st.warning("Flags:\n\n" + "\n\n".join(f"- {f}" for f in rec["flags"]))

        with st.expander("Preview NCR", expanded=True):
            st.markdown(ncr_md)

        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in form["ncr_number"])
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download NCR (Markdown)",
                data=ncr_md,
                file_name=f"{safe_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Download NCR (JSON)",
                data=json.dumps(ncr_dict, indent=2),
                file_name=f"{safe_id}.json",
                mime="application/json",
                use_container_width=True,
            )

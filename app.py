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
def _knockdown_sweep_cached(config_dict: dict, energies: tuple[float, ...]) -> pd.DataFrame:
    """Empirical knockdown sweep, cached so it doesn't rerun every script pass."""
    cfg = replace(_config_from_dict(config_dict), tier="empirical")
    return sweep_energies(cfg, list(energies), progress_callback=None)


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
        flag = " ⚠ SATURATED" if pct >= 79.0 else ""
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
# Sidebar — AnalysisConfig builder
# ---------------------------------------------------------------------------

with st.sidebar:
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
    material_choice = st.selectbox(
        "Material",
        MATERIAL_NAMES,
        index=MATERIAL_NAMES.index(DEFAULT_MATERIAL),
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

    layup_text = st.text_area(
        "Layup [deg]",
        value="0, 45, -45, 90, 90, -45, 45, 0",
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
    col_lx, col_ly = st.columns(2)
    Lx_mm = col_lx.number_input("Lx [mm]", min_value=10.0, max_value=2000.0, value=150.0, step=10.0)
    Ly_mm = col_ly.number_input("Ly [mm]", min_value=10.0, max_value=2000.0, value=100.0, step=10.0)
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
    input_mode = st.radio(
        "Damage source",
        ["Impact event", "C-scan inspection"],
        horizontal=False,
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
        energy_J = st.number_input(
            "Energy [J]",
            min_value=0.1,
            max_value=200.0,
            value=30.0,
            step=1.0,
            help="Kinetic energy of the impactor at contact.",
        )
        diameter_mm = st.number_input(
            "Impactor diameter [mm]",
            min_value=1.0,
            max_value=100.0,
            value=16.0,
            step=1.0,
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
    tier = st.selectbox(
        "Tier",
        TIERS,
        index=0,
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
                "cohesive_zone_factor": mesh_params.cohesive_zone_factor,
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
                df = _knockdown_sweep_cached(payload["config_dict"], sweep_E)
            fig = plot_knockdown_curve(
                df["energy_J"].tolist(),
                df["knockdown"].tolist(),
                tier_label="empirical",
            )
            st.pyplot(fig, use_container_width=False)
            st.dataframe(df, use_container_width=True)
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
                    "eigenvalue": eigs,
                }
            )
            st.bar_chart(df_eigs.set_index("mode"))
            st.dataframe(df_eigs, use_container_width=True)


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
            if energies:
                base_cfg = _config_from_dict(config_dict)
                base_cfg = replace(base_cfg, tier=sweep_tier)
                progress = st.progress(0.0)
                rows = []
                for i, E in enumerate(energies):
                    cfg_i = replace(
                        base_cfg,
                        impact=replace(base_cfg.impact, energy_J=E),
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = BvidAnalysis(cfg_i).run()
                    rows.append(
                        {
                            "energy_J": E,
                            "knockdown": res.knockdown,
                            "residual_MPa": res.residual_strength_MPa,
                            "dpa_mm2": res.dpa_mm2,
                        }
                    )
                    progress.progress((i + 1) / len(energies))
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                fig = plot_knockdown_curve(
                    df["energy_J"].tolist(),
                    df["knockdown"].tolist(),
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

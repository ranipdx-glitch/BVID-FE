"""Streamlit Community Cloud entry point for BVID-FE.

This module wraps the existing ``BvidAnalysis`` orchestrator (the same
engine the desktop PyQt GUI and the CLI use) in a Streamlit UI so the
project can run as a hosted web app.

Design notes
------------
* No physics is reimplemented. The sidebar collects the same inputs the
  desktop GUI's panels collect, builds an ``AnalysisConfig``, and calls
  ``BvidAnalysis(cfg).run()`` once per click. The result tabs reuse the
  existing matplotlib plot helpers in ``bvidfe.viz.plots_2d``.
* PyQt6, pyvista, and pyvistaqt are intentionally NOT imported here —
  they don't run reliably on Streamlit Cloud (no display server) and
  the cloud free tier's 1 GB memory cap rules out the larger 3D scenes.
* The 3D PyVista plots from the desktop GUI are mapped to matplotlib-
  only equivalents (damage map, ortho projections). Users who need the
  full 3D experience should run the desktop GUI (``bvidfe-gui``) or the
  CLI locally.
* fe3d is allowed but a runtime warning is shown when the predicted DOF
  count would push the cloud free tier past its memory limit. The
  ``BVIDFE_FE3D_MAX_DOF`` env var is honoured if you set it as a
  Streamlit Cloud secret.

Run locally
-----------
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Add the local src/ directory to sys.path so this script works on
# Streamlit Cloud whether or not the package is `pip install`ed. (When
# `requirements.txt` includes a `.` entry the package is installed and
# this is a no-op; the fallback covers the case where users edit the
# app and skip the editable-install line.)
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from bvidfe.analysis import AnalysisConfig, BvidAnalysis, MeshParams  # noqa: E402
from bvidfe.analysis.fe_mesh import estimate_fe_mesh_size  # noqa: E402
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry  # noqa: E402
from bvidfe.core.material import MATERIAL_LIBRARY  # noqa: E402
from bvidfe.damage.io import CScanSchemaError, damage_state_from_dict  # noqa: E402
from bvidfe.damage.state import DamageState  # noqa: E402
from bvidfe.gui.config_io import config_to_dict  # noqa: E402
from bvidfe.impact.mapping import ImpactEvent  # noqa: E402
from bvidfe.sweep.parametric_sweep import sweep_energies  # noqa: E402
from bvidfe.viz.plots_2d import plot_damage_map  # noqa: E402

st.set_page_config(
    page_title="BVID-FE — Composite impact damage",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("BVID-FE")
st.caption(
    "Residual strength + stiffness for composite laminates with "
    "Barely Visible Impact Damage. Cloud version of the BVID-FE library; "
    "for batch / scripted use see the `bvidfe` CLI and PyQt GUI on the desktop."
)


# ---------------------------------------------------------------------------
# Sidebar — inputs (mirrors the desktop GUI's seven input panels)
# ---------------------------------------------------------------------------


def _parse_layup(text: str) -> list[float]:
    """Same parsing rule as ``bvidfe.cli._parse_layup``."""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _build_damage_from_upload(uploaded) -> DamageState | None:
    """Convert an st.file_uploader handle to a DamageState.

    Surfaces ``CScanSchemaError`` as a Streamlit error message so a bad
    upload doesn't crash the app — matches the desktop GUI's import
    behaviour added in issue #10.
    """
    if uploaded is None:
        return None
    try:
        data = json.loads(uploaded.read())
        return damage_state_from_dict(data)
    except CScanSchemaError as exc:
        st.error(f"Invalid C-scan JSON: {exc}")
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse JSON: {exc.msg} (line {exc.lineno}, col {exc.colno}).")
    return None


with st.sidebar:
    st.header("Inputs")

    st.subheader("Material + layup")
    material = st.selectbox(
        "Material",
        sorted(MATERIAL_LIBRARY.keys()),
        index=sorted(MATERIAL_LIBRARY.keys()).index("IM7/8552"),
        help="Built-in CFRP presets. Custom materials require the Python API.",
    )
    layup_str = st.text_input(
        "Layup angles (deg, comma-separated)",
        value="0,45,-45,90,90,-45,45,0",
    )
    ply_thickness_mm = st.number_input(
        "Ply thickness (mm)",
        value=0.152,
        min_value=0.01,
        max_value=2.0,
        step=0.001,
        format="%.3f",
    )

    st.subheader("Panel")
    Lx = st.number_input("Lx (mm)", value=150.0, min_value=10.0, max_value=2000.0, step=10.0)
    Ly = st.number_input("Ly (mm)", value=100.0, min_value=10.0, max_value=2000.0, step=10.0)
    boundary = st.selectbox(
        "Boundary",
        ["simply_supported", "clamped", "free"],
        index=0,
        help="Affects Olsson bending stiffness, sublaminate buckling factor, and fe3d edge BCs.",
    )

    st.subheader("Loading + tier")
    loading = st.selectbox("Loading", ["compression", "tension"])
    tier = st.selectbox(
        "Tier",
        ["empirical", "semi_analytical", "fe3d"],
        index=0,
        help=(
            "empirical: Soutis CAI / Whitney-Nuismer TAI, sub-second.\n"
            "semi_analytical: adds Rayleigh-Ritz sublaminate buckling, ~1 s.\n"
            "fe3d: 3D FE first-ply-failure + linear buckling. Tens of "
            "seconds; memory-bounded on the cloud free tier."
        ),
    )

    st.subheader("Damage source")
    input_mode = st.radio(
        "Drive damage from:",
        ["Impact event", "Inspection (C-scan)"],
        horizontal=True,
    )

    impact: ImpactEvent | None = None
    damage: DamageState | None = None

    if input_mode == "Impact event":
        E = st.number_input(
            "Impact energy (J)", value=20.0, min_value=0.1, max_value=200.0, step=0.5
        )
        d_imp = st.number_input(
            "Impactor diameter (mm)", value=16.0, min_value=1.0, max_value=100.0, step=0.5
        )
        m_imp = st.number_input(
            "Impactor mass (kg)", value=5.5, min_value=0.1, max_value=100.0, step=0.1
        )
        impactor_shape = st.selectbox(
            "Impactor shape",
            ["hemispherical", "flat", "conical"],
            help="Affects Hertz contact stiffness and DPA spread factor.",
        )
        impact = ImpactEvent(
            energy_J=E,
            impactor=ImpactorGeometry(diameter_mm=d_imp, shape=impactor_shape),
            mass_kg=m_imp,
        )
    else:
        st.caption(
            "Upload a C-scan JSON (see `docs/cscan_schema.md`). The damage state "
            "will be applied without running the impact-mapping pipeline."
        )
        cscan_file = st.file_uploader("C-scan JSON", type=["json"])
        damage = _build_damage_from_upload(cscan_file)
        if cscan_file is not None and damage is None:
            st.stop()  # error already rendered

    mesh: MeshParams | None = None
    if tier == "fe3d":
        st.subheader("Mesh (fe3d)")
        elements_per_ply = st.number_input(
            "Elements per ply", value=1, min_value=1, max_value=4, step=1
        )
        in_plane_size_mm = st.number_input(
            "In-plane element size (mm)",
            value=5.0,
            min_value=1.0,
            max_value=20.0,
            step=0.5,
        )
        mesh = MeshParams(
            elements_per_ply=int(elements_per_ply),
            in_plane_size_mm=float(in_plane_size_mm),
        )

    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Pre-flight: assemble config, sanity-check fe3d size before running
# ---------------------------------------------------------------------------


def _build_cfg() -> AnalysisConfig:
    return AnalysisConfig(
        material=material,
        layup_deg=_parse_layup(layup_str),
        ply_thickness_mm=ply_thickness_mm,
        panel=PanelGeometry(Lx_mm=Lx, Ly_mm=Ly, boundary=boundary),
        loading=loading,
        tier=tier,
        impact=impact,
        damage=damage,
        mesh=mesh,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        cfg = _build_cfg()
    except (ValueError, AssertionError) as exc:
        st.error(f"Invalid config: {exc}")
        st.stop()

    # fe3d size pre-flight: warn if the predicted DOF count is large for
    # Streamlit Cloud's 1 GB memory cap. Use the same estimator the
    # desktop GUI uses.
    if tier == "fe3d":
        stats = estimate_fe_mesh_size(cfg)
        if stats["n_dof"] > 100_000:
            st.warning(
                f"This fe3d configuration produces {stats['n_elements']:,} elements "
                f"/ {stats['n_dof']:,} DOFs. Streamlit Cloud's free tier has only "
                f"~1 GB memory and may OOM-kill the worker. Consider increasing "
                f"`In-plane element size` or running the empirical / "
                f"semi_analytical tier first."
            )

    with st.spinner(f"Running {tier} analysis…"):
        try:
            result = BvidAnalysis(cfg).run()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
            st.stop()

    st.session_state["last_result"] = result
    st.session_state["last_cfg"] = cfg


# ---------------------------------------------------------------------------
# Result display (mirrors the desktop GUI's six result tabs)
# ---------------------------------------------------------------------------


if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    cfg = st.session_state["last_cfg"]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Knockdown", f"{result.knockdown:.3f}")
    summary_cols[1].metric("Residual strength", f"{result.residual_strength_MPa:.1f} MPa")
    summary_cols[2].metric("Pristine strength", f"{result.pristine_strength_MPa:.1f} MPa")
    summary_cols[3].metric("Projected damage area", f"{result.dpa_mm2:.0f} mm²")

    if result.notes:
        for note in result.notes:
            st.warning(f"⚠ {note}")

    tab_summary, tab_map, tab_kd, tab_buck, tab_export = st.tabs(
        ["Summary", "Damage Map", "Knockdown Curve", "Buckling", "Export"]
    )

    with tab_summary:
        st.code(result.summary())
        st.subheader("Configuration snapshot")
        st.json(config_to_dict(cfg), expanded=False)

    with tab_map:
        st.caption(
            "Delamination ellipses, impact location (×), and fiber-break "
            "core (red) on the panel outline."
        )
        fig = plot_damage_map(result.damage, cfg.panel)
        st.pyplot(fig, clear_figure=True)

    with tab_kd:
        st.caption(
            "Energy sweep at the current config. The fe3d tier is excluded "
            "from the cloud sweep — it would saturate the worker for any "
            "non-trivial point count."
        )
        if cfg.impact is None:
            st.info("Knockdown curve requires an impact-driven config.")
        else:
            n_points = st.slider("Sweep points", 4, 16, value=8)
            sweep_tier = st.selectbox(
                "Sweep tier",
                ["empirical", "semi_analytical"],
                index=0,
                key="sweep_tier",
            )
            with st.spinner("Running sweep…"):
                e_cur = cfg.impact.energy_J
                energies = list(np.linspace(2.0, max(5.0, 1.5 * e_cur), n_points))
                from dataclasses import replace

                sweep_cfg = replace(cfg, tier=sweep_tier, mesh=None)
                df = sweep_energies(
                    sweep_cfg, energies, on_error="skip"
                )
            st.line_chart(df.set_index("energy_J")[["knockdown"]])
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv_bytes,
                file_name="bvidfe_sweep.csv",
                mime="text/csv",
            )

    with tab_buck:
        if result.buckling_eigenvalues:
            fig, ax = plt.subplots(figsize=(6, 3))
            modes = list(range(1, len(result.buckling_eigenvalues) + 1))
            ax.bar(modes, result.buckling_eigenvalues, color="#3a7ca5")
            ax.set_xlabel("Mode index")
            ax.set_ylabel("Eigenvalue (load multiplier or N/mm)")
            ax.set_title("Buckling eigenvalues")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        else:
            st.info(
                "No buckling eigenvalues for this configuration "
                "(empirical tier or pristine input)."
            )
        if result.critical_sublaminate is not None:
            st.metric("Critical sublaminate interface", result.critical_sublaminate)

    with tab_export:
        st.caption(
            "Download the result and config as JSON. These are the same "
            "files the desktop GUI's File menu produces."
        )
        st.download_button(
            "Results JSON",
            data=json.dumps(result.to_dict(), default=str, indent=2),
            file_name="bvidfe_results.json",
            mime="application/json",
        )
        st.download_button(
            "Config JSON",
            data=json.dumps(config_to_dict(cfg), indent=2),
            file_name="bvidfe_config.json",
            mime="application/json",
        )
else:
    st.info(
        "Configure inputs in the sidebar and click **Run analysis**. "
        "Empirical and semi-analytical tiers return in under a second; "
        "fe3d takes 10-60 s depending on mesh size."
    )

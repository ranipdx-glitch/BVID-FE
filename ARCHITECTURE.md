# BVID-FE Architecture

## Module Dependency Diagram

```
core ──► impact ──► damage ──► elements ──► solver ──► failure ──► analysis ──► (viz, sweep, cli, gui)
 │                    ▲                                                ▲
 │                    └──────────────────────────────────────────────┘
 └──────────────────────────────── viz ─────────────────────────────┘
```

`core` has no internal dependencies. Each layer depends only on layers to its left.
`damage/state` is consumed by both the `impact` mapping stage and the `analysis` orchestrator.
The `cli` and `gui` packages sit at the right edge and are the two user-facing
entry points; both go through `analysis.BvidAnalysis` and never reach into the
inner layers directly.

## Module Catalog

| Module | Submodule | Role |
|--------|-----------|------|
| `core/` | `material.py` | `OrthotropicMaterial` dataclass + `MaterialLibrary` with four built-in presets (AS4/3501-6, IM7/8552, T700/2510, T800/epoxy). Ported from WrinkleFE. |
| | `laminate.py` | `Laminate`, `Ply`, `LoadState`; Classical Lamination Theory ABD matrices and effective engineering constants. Ported from WrinkleFE. |
| | `geometry.py` | `PanelGeometry`, `ImpactorGeometry`, `BoundaryKind` (clamped / simply-supported / free). |
| `impact/` | `olsson.py` | Olsson quasi-static damage-threshold load `P_c` and onset energy `E_onset` from plate bending + fracture energy balance. |
| | `shape_templates.py` | Layup-dependent per-interface "peanut" templates that distribute the total DPA into per-interface elliptical delaminations. |
| | `dent_model.py` | Thickness-normalized empirical dent-depth model producing `dent_depth_mm` from impact energy. |
| | `mapping.py` | Orchestrates the impact-driven workflow: `ImpactEvent` → `DamageState`. |
| `damage/` | `state.py` | `DamageState` dataclass + `DelaminationEllipse`; shapely-union projected damage area computation. |
| | `io.py` | C-scan JSON/CSV import, validation, and manual-entry helpers per `docs/cscan_schema.md`. |
| `elements/` | `hex8.py` | Standard 8-node isoparametric hexahedral element: shape functions, B-matrix, 24x24 stiffness. |
| | `hex8i.py` | Incompatible-modes hex element (Wilson modes) for improved bending accuracy. |
| | `gauss.py` | Gauss-Legendre quadrature points and weights: `gauss_points_1d`, `gauss_points_hex`. |
| | `cohesive.py` | 8-node zero-thickness cohesive surface element with bilinear traction-separation law (stiffness-reduction approximation in v0.1.0). |
| `solver/` | `static.py` | `StaticSolver` — sparse-direct linear static solve (SciPy). |
| | `assembler.py` | COO → CSC global stiffness matrix assembly from element contributions. |
| | `boundary.py` | `BoundaryCondition`, `BoundaryHandler` — penalty-method Dirichlet BCs for compression and tension loading. |
| | `buckling.py` | Generic linear eigenvalue buckling utility (`scipy.sparse.linalg.eigsh`). No longer called from the `fe3d` tier as of #129 — kept as a utility for future plate/shell-element buckling work. |
| `failure/` | `larc05.py` | Minimal LaRC05 composite failure criterion (Hashin-3D reduction). |
| | `tsai_wu.py` | Full 3D Tsai-Wu failure criterion with interaction terms. |
| | `soutis_openhole.py` | Soutis open-hole-equivalent CAI model + Whitney-Nuismer TAI (point-stress and average-stress). |
| | `evaluator.py` | `FailureEvaluator` — applies criteria across all elements; produces `LaminateFailureReport`. |
| `analysis/` | `config.py` | `AnalysisConfig` + `MeshParams` dataclasses (material, layup, panel, loading, tier, impact or damage; mesh resolution for fe3d). `MeshParams` validates positive elements_per_ply / in_plane_size_mm / cohesive_zone_factor at construction. |
| | `bvid.py` | `BvidAnalysis(AnalysisConfig).run()` — main orchestrator; dispatches to tier; merges fe3d buckling fallbacks into `AnalysisResults.notes`. |
| | `results.py` | `AnalysisResults` dataclass (`residual_strength_MPa`, `pristine_strength_MPa`, `knockdown`, `notes`, `field_results`, ...) with `summary()` and `to_dict()`. |
| | `semi_analytical.py` | Semi-analytical tier implementation (Rayleigh-Ritz sublaminate buckling + Soutis CAI; Whitney-Nuismer TAI). |
| | `fe_tier.py` | 3D FE tier implementation: damaged mesh build + K assembly for FPF (`_fe3d_cai_first_ply_failure`) for CAI/TAI; buckling channel (`fe3d_cai_buckling`) delegates to the Rayleigh-Ritz closed form in `semi_analytical.py` (#129). Component-wise damage scaling via Voigt mask. |
| | `fe_mesh.py` | Damaged hexahedral mesh construction from `DamageState`. Carries two per-element factors: `damage_factors` (out-of-plane, reduced inside delamination footprints) and `in_plane_damage_factors` (reduced only inside the fiber-break core). |
| `viz/` | `plots_2d.py` | Damage-map ellipse overlay (with impact location + fiber-break-core markers), knockdown curves, per-tier comparison charts (matplotlib). |
| | `plots_3d.py` | 3D PyVista plots: delamination surface, buckling mode shape, stress contour. Standalone scripts only — VTK/Qt embedding is not used (see CHANGELOG). |
| | `style.py` | Publication styling constants (fonts, DPI, color maps). |
| `sweep/` | `parametric_sweep.py` | `sweep_energies`, `sweep_layups`, `sweep_thicknesses` with `on_error` (`raise`/`skip`/`warn`) and optional `progress_callback` — CSV output, partial-result preserving. |
| `gui/` | `main_window.py` | `BvidMainWindow` — seven input panels + six result tabs + File menu; runs `BvidAnalysis` via `AnalysisWorker` / `SweepWorker` / `TierComparisonWorker` to keep the Qt main thread responsive. |
| | `panels/*.py` | Material, panel-geometry, input-mode, impact, damage-table, analysis-tier, and sweep input panels. |
| | `tabs/*.py` | Summary (with input + runtime caveat notes), Damage Map, Knockdown Curve (with auto background sweep), Damage View (orthographic projections), Buckling (tier-specific indicator), Damage Severity (through-thickness OOP-stiffness-loss heatmap). |
| | `workers.py` | `AnalysisWorker`, `SweepWorker`, `TierComparisonWorker` — QThread subclasses; daemon-thread heartbeats, deleteLater cleanup, runtime-note propagation. |
| | `config_io.py` | `config_to_dict` / `config_from_dict` for the File-menu Save/Load Config feature. |
| | `app.py` | `bvidfe-gui` entry point. |
| `cli.py` | — | `bvidfe` entry point; argparse with `choices=` for `--material` / `--tier` / `--loading`, custom positive-float / existing-path types, `--quick` (bare scalar) and `--quick-json` (single-line object) output modes. |

## Public API

```python
# High-level orchestrator
from bvidfe.analysis import AnalysisConfig, BvidAnalysis, AnalysisResults

# Geometry and event dataclasses
from bvidfe.core.geometry import PanelGeometry, ImpactorGeometry
from bvidfe.impact.mapping import ImpactEvent

# Damage state (inspection-driven path)
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.damage.io import load_cscan_json

# Lower-level access (advanced use)
from bvidfe.impact.olsson import threshold_load
from bvidfe.failure.soutis_openhole import soutis_cai_knockdown, whitney_nuismer_tai
from bvidfe.sweep.parametric_sweep import sweep_energies, sweep_layups, sweep_thicknesses
```

## Data-Flow Diagram

```
  ┌─────────────────────┐     ┌──────────────────────┐
  │   ImpactEvent        │     │   C-scan JSON / dict  │
  │  (energy, impactor,  │     │  (delaminations,      │
  │   mass, location)    │     │   dent, fiber break)  │
  └────────┬────────────┘     └──────────┬────────────┘
           │  impact/mapping.py           │  damage/io.py
           │  Olsson threshold            │  load_cscan_json()
           │  shape_templates DPA         │
           │  dent_model                  │
           └───────────┬──────────────────┘
                       ▼
              ┌─────────────────┐
              │   DamageState   │
              │  delaminations  │
              │  dent_depth_mm  │
              │  dpa_mm2        │
              └────────┬────────┘
                       │  analysis/bvid.py  BvidAnalysis.run()
                       │
          ┌────────────┼────────────────────┐
          ▼            ▼                    ▼
    empirical    semi_analytical          fe3d
    (Soutis +    (Rayleigh-Ritz +    (hex mesh +
    WN, ~ms)      Soutis, ~s)         LaRC05/TW,
                                       ~minutes)
          │            │                    │
          └────────────┴────────────────────┘
                       ▼
              ┌─────────────────────┐
              │   AnalysisResults   │
              │  residual_strength  │
              │  pristine_strength  │
              │  knockdown          │
              │  damage (echoed)    │
              │  buckling_eigenvals │
              │  field_results      │
              └─────────────────────┘
```

## Roadmap

### Shipped in v0.2.0-dev
- **PyQt6 GUI** with seven input panels and six result tabs (Summary, Damage Map, Knockdown Curve, Damage View, Buckling, Damage Severity).
- **Rayleigh-Ritz closed-form buckling CAI** in the `fe3d` tier (`fe3d_cai_buckling`, delegated to `semi_analytical.panel_buckling_load` per #129); FPF on the damaged 3D mesh retained alongside as the other channel, with `min(buckling, FPF)` governing.
- **PyInstaller** packaging for standalone macOS and Windows apps via the GitHub Actions release workflow.
- **Component-wise fe3d damage scaling** (in-plane vs out-of-plane) replacing the prior uniform `DAMAGE_STIFFNESS_FACTOR`.
- **Through-thickness strengths and 1-3 shear strength** (`Zt`, `Zc`, `S13`) on `OrthotropicMaterial` with transverse-isotropy fallback to `Yt` / `Yc` / `S12`.
- **Hex8 Jacobian validation** raising `DegenerateElementError` on non-positive `det(J)`.
- **Runtime notes** on `AnalysisResults` surfacing the fe3d buckling-channel fallback when the Rayleigh-Ritz closed form returns a degenerate result (tagged `fe3d_buckling_fallback`).
- **Tier-comparison worker** moving the menu's 16-analysis sweep off the Qt main thread.
- **Robust CLI / GUI input validation**: `choices=` materials, positive-float numeric flags, existing-path C-scan, distinct dialogs for missing-field / invalid-value / malformed-JSON config loads, surfaced damage-table row skipping.
- **Sweep `on_error` / `progress_callback`** preserving partial results on per-iteration failure.

### Planned (post-v0.2.0)
- **True cohesive surfaces**: replace stiffness-reduction approximation with full bilinear traction-separation law in the `fe3d` tier (resolves the residual energy-insensitivity at small DPA).
- **Reinstated 3D buckling FE path** with proper load-introduction BCs and plate/shell elements (DKQ or MITC4) so the buckling channel can capture damage-zone heterogeneity beyond what the closed-form delegation (#129) can express.
- **Validated datasets**: Soutis (1996), Caprino (1984), Sanchez-Saez (2005), NASA COCOMAT — digitized and integrated into `validation/`.
- **Calibrated material constants**: `olsson_alpha`, `soutis_k_s`, `dent_beta`, and related parameters refined against specific material test data.
- **fe3d in CI validation gate**: extend `validate_bvid_public.py` to run all three tiers with looser tolerances on fe3d.

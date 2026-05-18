# BVID-FE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-309%20passing-brightgreen.svg)](https://github.com/elhajjar1/BVID-FE/actions)

A Python library for predicting residual strength and stiffness of fiber-reinforced composite laminates containing Barely Visible Impact Damage (BVID).

## Why This Tool?

Low-velocity impacts on composite structures can create internal delaminations that are invisible to the naked eye yet significantly degrade compression and tension strength. Engineers certifying composite airframes and pressure vessels need fast, reliable estimates of how much strength is lost and how the knockdown depends on layup, panel size, and impact energy. BVID-FE provides three modeling tiers — from a 30-millisecond empirical lookup to a full 3D finite element solution — so the right level of fidelity can be chosen for each stage of the design process.

BVID-FE is the third in a family of defect-specific composite tools, joining **PorosityFE** (porosity defects) and **WrinkleFE** (fiber waviness). The three tools share material models, laminate theory, failure criteria, and documentation conventions.

## Features

- **Two workflow paths** converging on a shared `DamageState`:
  - *Impact-driven*: Olsson quasi-static threshold + peanut-template DPA distribution + empirical dent model
  - *Inspection-driven*: C-scan JSON import per the documented schema (`docs/cscan_schema.md`)
- **Three modeling tiers** for residual strength after BVID:
  - *Empirical*: Soutis CAI knockdown + Whitney-Nuismer TAI (seconds)
  - *Semi-analytical*: Rayleigh-Ritz sublaminate buckling + Soutis post-buckling envelope; Whitney-Nuismer for TAI (seconds)
  - *3D FE*: First-ply-failure on a damaged hexahedral mesh; LaRC05 for CAI, Tsai-Wu for TAI (minutes)
- **CAI and TAI loading modes** (Compression-After-Impact and Tension-After-Impact)
- **Per-interface ellipse damage model** using `DelaminationEllipse` with shapely-union projected damage area
- **Four material presets**: AS4/3501-6, IM7/8552, T700/2510, T800/epoxy
- **CLI** for single-run and batch use
- **PyQt6 desktop GUI** with seven input panels and six result tabs — Summary (with inline fe3d caveat), Damage Map, Knockdown Curve (auto-populates via a background empirical sweep after every single run), Damage View (matplotlib orthographic top/side/front projections + text summary), Buckling Eigenvalues (bar chart), and Damage Severity (through-thickness sum of per-element `1 − damage_factor`; see [Physics Models](#damage-severity-heatmap)). Threaded workers with heartbeat progress, mesh-size guards on the fe3d tier, and a File menu (Save/Load Config, Export Results JSON, Export Damage Map PNG)
- **Parametric sweeps** over impact energy, layup, or ply thickness with CSV output
- **2D plots**: damage map, knockdown curves, tier comparison charts
- **3D PyVista plots**: delamination surfaces, buckling mode shape, stress contour
- **PyInstaller packaging** for standalone macOS and Windows apps via GitHub Actions release workflow

## Installation

```bash
git clone https://github.com/elhajjar1/BVID-FE.git
cd BVID-FE
pip install -e ".[all]"
pytest -v
```

## Quick Start

### Desktop GUI

```bash
bvidfe-gui
```

Launches the PyQt6 desktop application: configure material + layup, panel size, impact event or damage state, pick a tier, click **Run analysis**. Results appear in the Summary and Damage Map tabs. File menu → Save/Load Config to persist setups, Export Results JSON/PNG for reporting.

### Command-line interface

```bash
bvidfe --material IM7/8552 \
       --layup "0,45,-45,90,90,-45,45,0" \
       --thickness 0.152 \
       --panel 150x100 \
       --loading compression \
       --energy 30
```

Argument units and formats:

- `--thickness` — ply thickness in millimeters. Either a single positive
  number for a uniform laminate, **or** a comma-separated list of per-ply
  thicknesses with length matching `--layup` for laminates that mix plies
  of different fabric weights / prepreg gauges (e.g.
  `--thickness 0.10,0.10,0.20,0.20,0.20,0.20,0.10,0.10` for an 8-ply stack
  with thinner outer layers).
- `--panel` — panel dimensions as `Lx_mm x Ly_mm` (e.g. `150x100`), in
  millimeters. Use a lowercase `x` separator with no spaces; `Lx` is the
  in-plane x-dimension (loading direction for compression/tension along x)
  and `Ly` is the in-plane y-dimension.
- `--energy` — impact energy in joules.
- `--impactor-diameter` — impactor diameter in millimeters (default 16.0).
- `--mass` — impactor mass in kilograms (default 5.5).
- `--layup` — comma-separated ply angles in degrees, ordered bottom-to-top.

### Python API — impact-driven path

```python
from bvidfe.analysis import AnalysisConfig, BvidAnalysis
from bvidfe.core.geometry import PanelGeometry, ImpactorGeometry
from bvidfe.impact.mapping import ImpactEvent

cfg = AnalysisConfig(
    material="IM7/8552",
    layup_deg=[0, 45, -45, 90] * 4,
    ply_thickness_mm=0.152,
    panel=PanelGeometry(Lx_mm=150, Ly_mm=100),
    impact=ImpactEvent(
        energy_J=30,
        impactor=ImpactorGeometry(diameter_mm=16),
        mass_kg=5.5,
    ),
    loading="compression",
    tier="semi_analytical",
)
result = BvidAnalysis(cfg).run()
print(result.summary())
# -> residual CAI strength, knockdown, DPA, dent depth, per-interface delaminations
```

`ply_thickness_mm` accepts either a single ``float`` (uniform laminate) or
a list/tuple of per-ply thicknesses with one entry per ply in
``layup_deg``. Per-ply thicknesses let users model laminates that mix
plies of different fabric weights or prepreg gauges:

```python
cfg = AnalysisConfig(
    material="IM7/8552",
    layup_deg=[0, 90, 45, -45, -45, 45, 90, 0],
    # Thinner 0/90 surface plies over thicker quasi-iso interior:
    ply_thickness_mm=[0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.10, 0.10],
    panel=PanelGeometry(Lx_mm=150, Ly_mm=100),
    impact=ImpactEvent(energy_J=30),
    loading="compression",
    tier="semi_analytical",
)
```

### Inspection-driven path (C-scan import)

```python
import json
from bvidfe.damage.io import load_cscan_json
from bvidfe.analysis import AnalysisConfig, BvidAnalysis
from bvidfe.core.geometry import PanelGeometry

with open("scan.json") as f:
    damage_state = load_cscan_json(json.load(f))

cfg = AnalysisConfig(
    material="AS4/3501-6",
    layup_deg=[0, 45, -45, 90] * 4,
    ply_thickness_mm=0.125,
    panel=PanelGeometry(Lx_mm=200, Ly_mm=150),
    damage=damage_state,
    loading="compression",
    tier="empirical",
)
result = BvidAnalysis(cfg).run()
print(f"Knockdown: {result.knockdown:.3f}")
```

### Parametric sweep

```python
from bvidfe.sweep.parametric_sweep import sweep_energies

results_df = sweep_energies(
    energies_J=[10, 20, 30, 40, 50],
    base_config=cfg,
    output_csv="knockdown_vs_energy.csv",
)
```

## Physics Models

### Empirical tier

Soutis open-hole-equivalent CAI model relates the delamination-affected stress concentration around the projected damage area to a net-section failure. Whitney-Nuismer point-stress and average-stress criteria are used for TAI. Both models are closed-form and run in milliseconds.

### Semi-analytical tier

The damaged sublaminate above the largest delamination is treated as a plate with reduced in-plane stiffness. A Rayleigh-Ritz energy method solves for the sublaminate buckling load, and the Soutis post-buckling envelope predicts the far-field CAI stress at overall failure. Whitney-Nuismer is retained for TAI. Sublaminate eigenvalues are available in `AnalysisResults.buckling_eigenvalues`.

### 3D FE tier

A structured hexahedral mesh is built for the damaged laminate. Delaminated interfaces are approximated by a **component-wise stiffness-reduction model** (true cohesive surfaces deferred to a future release): each damaged element carries an out-of-plane factor (`DAMAGE_OOP_FACTOR ≈ 0.05`) that scales the through-thickness and transverse-shear stiffness, while in-plane stiffness is preserved (the plies themselves remain intact). Inside the fiber-break core under the impact site, in-plane stiffness is also reduced (`DAMAGE_FIBER_BREAK_INPLANE_FACTOR ≈ 0.30`) to represent fiber bundle fracture. First-ply-failure is evaluated at all Gauss points using LaRC05 (CAI) and Tsai-Wu (TAI). For CAI, a true linear buckling eigensolve runs alongside FPF and the lower of the two governs.

### Knockdown definition and cross-tier comparability

`AnalysisResults.knockdown` is computed in exactly one place — `BvidAnalysis.run()` (`src/bvidfe/analysis/bvid.py`):

```python
knockdown = residual_strength_MPa / pristine_strength_MPa
```

**The denominator is identical across all three tiers.** `_pristine_strength()` (`src/bvidfe/analysis/bvid.py`) is a thickness-weighted ply-average of the lamina-level strengths from the material card:

- CAI: `Σ tᵢ (Xc·cos²θᵢ + Yc·sin²θᵢ) / Σ tᵢ`
- TAI: `Σ tᵢ (Xt·cos²θᵢ + Yt·sin²θᵢ) / Σ tᵢ`

**The numerator (residual strength) is what differs between tiers:**

| Tier | CAI residual stress | TAI residual stress |
|---|---|---|
| `empirical` | Soutis: `σ₀ / (1 + k_s·(DPA/A_panel)^m)` | Whitney-Nuismer point-stress on equivalent hole |
| `semi_analytical` | `min(Soutis, σ_buckling_sublam)` — adds Rayleigh-Ritz sublaminate buckling floor | Delegates to Whitney-Nuismer (mathematically identical to `empirical`) |
| `fe3d` | `min(λ_crit·σ_ref, FPF_LaRC05)`, capped at σ₀ | FPF Tsai-Wu on damaged mesh, capped at σ₀ |

**What this means for users:**

- All three tiers report knockdown on the **same scale** (ratio relative to the same pristine baseline), so values are *qualitatively comparable*.
- They are **not** numerically interchangeable: each tier captures different failure mechanisms.
  - For **TAI**, `empirical` and `semi_analytical` are mathematically identical; `fe3d` differs.
  - For **CAI**, `semi_analytical ≤ empirical` always (the buckling floor only lowers the residual). `fe3d` is independent and dominated by stress concentration at the damage boundary rather than damage magnitude — see the flat-vs-energy caveat in [Limitations](#limitations).
- For **energy-scaling studies**, prefer `empirical` (Soutis scales with DPA) or `semi_analytical` (Rayleigh-Ritz scales with ellipse size). `fe3d` is intended for stress-field context and through-thickness damage visualization, not energy-dependent knockdown curves.
- A few silent fallbacks affect interpretation:
  - `fe3d` buckling: if no positive eigenvalue is found (or the eigenvalue is < 5% of σ₀), the buckling result is discarded and FPF — or, in pure-buckling failure, σ₀ — is reported instead. The reason is now surfaced as a string in `AnalysisResults.notes` (also rendered in the GUI Summary tab and in `result.summary()` / `result.to_dict()`), so a `knockdown` of 1.0 from `fe3d` is distinguishable from "buckling solve unconverged" by inspecting `result.notes`.
  - DPA is globally capped at 80% of panel area (`src/bvidfe/impact/mapping.py`); above this damage threshold all three tiers saturate.

### Damage severity heatmap

The "Damage Severity" tab in the GUI is **not** a simple count of damaged interfaces, nor is it a continuous physical damage variable (e.g. a Kachanov-style scalar). It is a through-thickness accumulation of the per-element stiffness-reduction metric used by the `fe3d` mesh:

1. Each hex element in the damaged mesh carries an **out-of-plane** stiffness factor `damage_factor`: `1.0` if pristine, or `DAMAGE_OOP_FACTOR ≈ 0.05` if it is intersected by a delamination interface inside an ellipse footprint, or sits inside the fiber-break core. (Fiber-break-core elements additionally carry a reduced **in-plane** factor `in_plane_damage_factor = DAMAGE_FIBER_BREAK_INPLANE_FACTOR ≈ 0.30`; pure-delamination elements leave the in-plane factor at `1.0`.) The factors are **categorical per element** (geometric overlap tests in `bvidfe.analysis.fe_mesh.build_fe_mesh`), not a continuum damage variable.
2. The heatmap metric is `1 − damage_factor` (the OOP factor) — `0.0` for pristine, `0.95` for fully delaminated.
3. For each in-plane column `(x, y)` the metric is **summed over the through-thickness elements** to produce the heatmap value (`bvidfe/gui/tabs/stress_field_tab.py`).

The colorbar is therefore in units of "stacked OOP-stiffness loss contributions": `0` means no delamination at any interface in that column, and the maximum (`≈ n_plies − 1` for a single fully-delaminated interface; higher if multiple interfaces are delaminated and `elements_per_ply > 1`) means every element through the thickness sits inside a damaged region. It is a visualization aid analogous to a C-scan operator's depth-projected damage map, not a quantitative continuum-damage field. The fiber-break-core in-plane reduction is a separate channel (`mesh.in_plane_damage_factors`) and is not currently overlaid on the heatmap.

## Limitations

- Material calibration constants (`olsson_alpha`, `soutis_k_s`, `dent_beta`, and related parameters) are reasonable defaults for typical CFRP systems. Precise values need to be calibrated against material-specific coupon test data before use in certification.
- LaRC05 is implemented as a minimal Hashin-3D reduction. Full plane-search fiber-kinking is deferred to a future release.
- The `fe3d` tier uses component-wise stiffness reduction at delaminated interfaces (in-plane preserved, out-of-plane reduced) instead of true cohesive surfaces with bilinear traction-separation laws. Cohesive surfaces are deferred to a future release.
- **The `fe3d` tier's knockdown is partially insensitive to impact energy** above the Olsson threshold. Linear buckling (`fe3d_cai_buckling`) now responds to delamination size, but the FPF fallback strain is controlled by stress concentration at the healthy/damaged boundary rather than damage magnitude. For energy-dependent knockdown curves prefer `tier="empirical"` (Soutis scales with DPA) or `tier="semi_analytical"` (Rayleigh-Ritz scales with ellipse size). The `fe3d` tier is most useful for stress-field context, through-thickness damage visualization, and buckling-driven failure modes. Full energy-monotonicity (in-plane pre-stress BCs + cohesive surfaces) is v0.3.0 scope.
- No validated datasets included; comparison against published Soutis, Caprino, and NASA datasets is on the roadmap.

## Citation

If you use BVID-FE in your research, please cite:

```bibtex
@software{elhajjar2026bvidfe,
  author    = {Elhajjar, Rani},
  title     = {{BVID-FE}: Barely Visible Impact Damage residual-strength analysis
               for composite laminates},
  year      = {2026},
  version   = {0.1.0-alpha},
  publisher = {GitHub},
  url       = {https://github.com/elhajjar1/BVID-FE},
  note      = {University of Wisconsin-Milwaukee}
}
```

## References

- Olsson, R. (2001). Analytical prediction of large mass impact damage in composite laminates. *Composites Part A*, 32(9), 1207-1215.
- Soutis, C. (1996). Compressive strength of unidirectional composites: measurement and prediction. *ASTM STP*, 1242, 168-176.
- Whitney, J.M. & Nuismer, R.J. (1974). Stress fracture criteria for laminated composites containing stress concentrations. *Journal of Composite Materials*, 8(3), 253-265.
- Tsai, S.W. & Wu, E.M. (1971). A general theory of strength for anisotropic materials. *Journal of Composite Materials*, 5(1), 58-80.
- Davila, C.G., Camanho, P.P., & Rose, C.A. (2005). Failure criteria for FRP laminates. NASA/TM-2005-213530 (LaRC05).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and submitting pull requests.

## License

MIT License. See [LICENSE](LICENSE) for details.

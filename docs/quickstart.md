# Quickstart

The fastest path from a fresh clone to a residual-strength estimate.

## 1. Install

```bash
git clone https://github.com/ranipdx-glitch/BVID-FE.git
cd BVID-FE
pip install -e ".[dev]"
```

To also build this documentation site locally, install the `docs` extras:

```bash
pip install -e ".[docs]"
mkdocs serve     # http://127.0.0.1:8000
```

## 2. Try it in Jupyter

The fastest tour of the physics is
[`examples/quickstart.ipynb`](https://github.com/ranipdx-glitch/BVID-FE/blob/main/examples/quickstart.ipynb)
— a runnable notebook that walks through:

- the Olsson quasi-static damage threshold,
- the peanut-template DPA distribution across ply interfaces,
- the Soutis CAI knockdown formula,
- the Whitney-Nuismer TAI point-stress criterion,

with LaTeX derivations and inline Plotly damage map / knockdown sweep / 3D
damage mesh visualisations.

```bash
pip install jupyter
jupyter notebook examples/quickstart.ipynb
```

## 3. Run from Python

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
```

## 4. Run from the command line

```bash
bvidfe --material IM7/8552 \
       --layup "0,45,-45,90,90,-45,45,0" \
       --thickness 0.152 \
       --panel 150x100 \
       --loading compression \
       --energy 30
```

All lengths are in **millimetres**. `--panel` is `<Lx>x<Ly>` (literal `x`
separator, no spaces — e.g. `150x100` is `Lx = 150` mm, `Ly = 100` mm).
`--thickness` is the per-ply thickness in mm; `--layup` is a comma-separated
list of ply angles in degrees; `--energy` is the impact energy in joules.

## 5. Streamlit web app

```bash
streamlit run app.py
```

Opens the BVID-FE web UI at <http://localhost:8501>. Configure the material,
layup, panel, impact (or C-scan), pick a tier, click **Run analysis**. Results
appear across the Summary / Damage Map / 3D Damage / Knockdown / Buckling /
Damage Severity / Sweep tabs.

See [Deployment](deployment.md) for publishing this app to a public URL on
Streamlit Community Cloud.

## More examples

The repository's [`examples/`](https://github.com/ranipdx-glitch/BVID-FE/tree/main/examples)
folder contains five end-to-end scripts:

- `01_empirical_quick.py` — minimal one-shot empirical analysis.
- `02_tier_comparison.py` — all three tiers on the same damage state.
- `03_energy_sweep.py` — parametric sweep + plot.
- `04_inspection_driven.py` — C-scan JSON loading.
- `05_tier_comparison_sweep.py` — empirical vs. semi-analytical across
  impact energies.

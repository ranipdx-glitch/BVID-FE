# BVID-FE

A Python library for predicting residual strength and stiffness of fiber-reinforced composite laminates containing Barely Visible Impact Damage (BVID).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/ranipdx-glitch/BVID-FE/actions/workflows/tests.yml/badge.svg)](https://github.com/ranipdx-glitch/BVID-FE/actions/workflows/tests.yml)

## Why this tool?

Low-velocity impacts on composite structures can create internal delaminations
that are invisible to the naked eye yet significantly degrade compression and
tension strength. Engineers certifying composite airframes and pressure vessels
need fast, reliable estimates of how much strength is lost and how the
knockdown depends on layup, panel size, and impact energy. BVID-FE provides
three modeling tiers — from a 30-millisecond empirical lookup to a full 3D
finite-element solution — so the right level of fidelity can be chosen for
each stage of the design process.

BVID-FE is the third in a family of defect-specific composite tools, joining
**PorosityFE** (porosity defects) and **WrinkleFE** (fiber waviness). The three
tools share material models, laminate theory, failure criteria, and
documentation conventions.

## Features

- **Two workflow paths** converging on a shared `DamageState`:
    - *Impact-driven*: Olsson quasi-static threshold + peanut-template DPA
      distribution + empirical dent model
    - *Inspection-driven*: C-scan JSON import per the documented schema
      (see [C-Scan Schema](cscan_schema.md))
- **Three modeling tiers** for residual strength after BVID:
    - *Empirical*: Soutis CAI knockdown + Whitney-Nuismer TAI (seconds)
    - *Semi-analytical*: Rayleigh-Ritz sublaminate buckling + Soutis
      post-buckling envelope; Whitney-Nuismer for TAI (seconds)
    - *3D FE*: First-ply-failure on a damaged hexahedral mesh; LaRC05 for
      CAI, Tsai-Wu for TAI (minutes)
- **CAI and TAI loading modes** (Compression-After-Impact and
  Tension-After-Impact)
- **Per-interface ellipse damage model** using `DelaminationEllipse` with
  shapely-union projected damage area
- **Four material presets**: AS4/3501-6, IM7/8552, T700/2510, T800/epoxy
- **CLI**, **Streamlit web app**, and **Python API**
- **Parametric sweeps** over impact energy, layup, or ply thickness with
  CSV output
- **2D / 3D visualisations** of damage and stress fields

## Installation

```bash
git clone https://github.com/ranipdx-glitch/BVID-FE.git
cd BVID-FE
pip install -e ".[dev]"
pytest -v
```

## Where to next

- [Quickstart](quickstart.md) — run your first analysis in under a minute.
- [Python API](python_api.md) — script BVID-FE workflows beyond the GUI/CLI.
- [C-Scan Schema](cscan_schema.md) — load inspection data from JSON.
- [Physics Models](physics_models.md) — what each tier solves.
- [Deployment](deployment.md) — publish the Streamlit app to a public URL.

## JSON Schemas

Auto-generated JSON Schemas (draft 2020-12) for the public data contracts are
published with this site:

- [`analysis_config.json`](schemas/analysis_config.json) — `AnalysisConfig`
- [`analysis_results.json`](schemas/analysis_results.json) — `AnalysisResults`

The schemas are regenerated from the in-tree dataclasses by
`scripts/generate_schemas.py`.

## Citation

If you use BVID-FE in your research, please cite the project (see the
[CITATION.cff](https://github.com/ranipdx-glitch/BVID-FE/blob/main/CITATION.cff)
at the repository root).

## License

MIT License. See
[LICENSE](https://github.com/ranipdx-glitch/BVID-FE/blob/main/LICENSE) for
details.

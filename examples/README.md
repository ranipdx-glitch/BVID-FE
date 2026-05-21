# BVID-FE Examples

Short, self-contained scripts that exercise the main BVID-FE workflows.
Each one runs standalone from the repository root:

```bash
python examples/01_empirical_quick.py
python examples/02_tier_comparison.py
python examples/03_energy_sweep.py
python examples/04_inspection_driven.py
python examples/05_tier_comparison_sweep.py
```

Outputs (PNGs, CSVs) are written to `examples/output/` which is
gitignored.

## What each example shows

| Script | Demonstrates |
|---|---|
| `01_empirical_quick.py` | Minimal API call: 30-J impact → CAI knockdown in three lines of setup |
| `02_tier_comparison.py` | All three tiers (empirical / semi_analytical / fe3d) on the same damage, with a bar-chart summary |
| `03_energy_sweep.py` | `sweep_energies()` → DataFrame → CSV + knockdown-vs-energy plot |
| `04_inspection_driven.py` | Loading a C-scan JSON and running the damage-driven path (no impact event) |
| `05_tier_comparison_sweep.py` | Empirical vs. semi_analytical over a 12-point energy sweep — overlaid knockdown curves and a CSV with both series. Python-API equivalent of the GUI's File → Compare Tiers action. |
| `quickstart.ipynb` | Jupyter walkthrough of `01_empirical_quick.py` with LaTeX derivations (Olsson, Soutis, Whitney-Nuismer) and inline Plotly damage map / knockdown sweep / 3D damage mesh. Outputs are stripped on commit — run `jupyter notebook examples/quickstart.ipynb` to populate them. |

## Runtime expectations (arm64 Mac, default mesh)

- empirical / semi_analytical: sub-second per run
- fe3d: ~10 s per run (single analysis); ~50 s for the 5-point energy sweep

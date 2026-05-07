"""BVID-FE validation harness.

Runs each case in a published (or synthetic) dataset through the selected
BVID-FE tier and reports Mean Absolute Percentage Error (MAE%) on residual
strength. Used both interactively (prints a table) and as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from bvidfe.analysis import AnalysisConfig, BvidAnalysis
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.impact.mapping import ImpactEvent


DATASET_DIR = Path(__file__).parent / "datasets"

# Per-tier slack on the dataset's target_mae_pct. ``empirical`` and
# ``semi_analytical`` are calibrated closed-form models; ``fe3d`` is the
# stress-field tool whose absolute-strength prediction has known caveats
# (energy-flatness, simplified buckling BCs) — see CHANGELOG. The looser
# fe3d gate keeps CI from flapping on regressions that are still inside
# the documented model uncertainty.
_TIER_GATE_MULTIPLIER = {
    "empirical": 1.25,
    "semi_analytical": 1.25,
    "fe3d": 2.0,
}


@dataclass
class DatasetCase:
    """A single BVID test record suitable for validation."""

    material: str
    layup_deg: List[float]
    ply_thickness_mm: float
    panel_Lx_mm: float
    panel_Ly_mm: float
    impactor_diameter_mm: float
    impactor_mass_kg: float
    impact_energy_J: float
    loading: str  # "compression" or "tension"
    measured_strength_MPa: float
    measured_dent_mm: Optional[float] = None
    measured_dpa_mm2: Optional[float] = None
    note: str = ""


def case_from_dict(d: dict) -> DatasetCase:
    return DatasetCase(
        material=d["material"],
        layup_deg=list(d["layup_deg"]),
        ply_thickness_mm=float(d["ply_thickness_mm"]),
        panel_Lx_mm=float(d["panel_Lx_mm"]),
        panel_Ly_mm=float(d["panel_Ly_mm"]),
        impactor_diameter_mm=float(d.get("impactor_diameter_mm", 16.0)),
        impactor_mass_kg=float(d.get("impactor_mass_kg", 5.5)),
        impact_energy_J=float(d["impact_energy_J"]),
        loading=d.get("loading", "compression"),
        measured_strength_MPa=float(d["measured_strength_MPa"]),
        measured_dent_mm=d.get("measured_dent_mm"),
        measured_dpa_mm2=d.get("measured_dpa_mm2"),
        note=d.get("note", ""),
    )


def load_dataset(path: Path) -> tuple[str, List[DatasetCase], dict]:
    """Load a dataset JSON. Returns (name, cases, meta-dict)."""
    obj = json.loads(path.read_text())
    name = obj.get("name", path.stem)
    target_mae = float(obj.get("target_mae_pct", 100.0))
    cases = [case_from_dict(c) for c in obj["cases"]]
    return name, cases, {"target_mae_pct": target_mae}


def run_case(case: DatasetCase, tier: str) -> dict:
    """Run a single case at the given tier. Returns a row dict."""
    cfg = AnalysisConfig(
        material=case.material,
        layup_deg=list(case.layup_deg),
        ply_thickness_mm=case.ply_thickness_mm,
        panel=PanelGeometry(case.panel_Lx_mm, case.panel_Ly_mm),
        loading=case.loading,
        tier=tier,
        impact=ImpactEvent(
            energy_J=case.impact_energy_J,
            impactor=ImpactorGeometry(diameter_mm=case.impactor_diameter_mm),
            mass_kg=case.impactor_mass_kg,
        ),
    )
    result = BvidAnalysis(cfg).run()
    err_pct = (
        100.0
        * abs(result.residual_strength_MPa - case.measured_strength_MPa)
        / case.measured_strength_MPa
    )
    return {
        "material": case.material,
        "energy_J": case.impact_energy_J,
        "loading": case.loading,
        "predicted_MPa": result.residual_strength_MPa,
        "measured_MPa": case.measured_strength_MPa,
        "error_pct": err_pct,
        "predicted_knockdown": result.knockdown,
        "predicted_dpa_mm2": result.dpa_mm2,
        "note": case.note,
    }


def run_dataset(cases: List[DatasetCase], tier: str) -> pd.DataFrame:
    return pd.DataFrame([run_case(c, tier) for c in cases])


def _discover_datasets() -> List[Path]:
    if not DATASET_DIR.exists():
        return []
    return sorted(DATASET_DIR.glob("*.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run BVID-FE validation against digitized datasets."
    )
    parser.add_argument(
        "--tier",
        default="empirical",
        choices=["empirical", "semi_analytical", "fe3d"],
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset stem name (without .json). If omitted, run all.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if MAE exceeds the tier-specific multiplier "
        "of the dataset target_mae_pct (1.25 for empirical / "
        "semi_analytical, 2.0 for fe3d).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="If set, run only the first N cases of each dataset. Useful for "
        "the fe3d CI step where the full dataset would push CI runtime "
        "past 60 s.",
    )
    args = parser.parse_args(argv)

    paths = _discover_datasets()
    if args.dataset is not None:
        paths = [p for p in paths if p.stem == args.dataset]
    if not paths:
        print("No datasets found.", file=sys.stderr)
        return 1

    any_gate_fail = False
    multiplier = _TIER_GATE_MULTIPLIER.get(args.tier, 1.25)
    for path in paths:
        name, cases, meta = load_dataset(path)
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        df = run_dataset(cases, tier=args.tier)
        mae = float(df["error_pct"].mean())
        max_err = float(df["error_pct"].max())
        target = meta["target_mae_pct"]
        print(f"\n=== {name} ({args.tier}) — {len(cases)} cases ===")
        print(df.to_string(index=False))
        print(
            f"MAE = {mae:.2f}%   max error = {max_err:.2f}%   "
            f"target = {target:.1f}%   gate-multiplier = {multiplier:g}"
        )
        if args.gate and mae > multiplier * target:
            print(
                f"FAIL: {name} ({args.tier}) MAE {mae:.2f}% exceeds "
                f"{multiplier:g} * {target:.1f}% target",
                file=sys.stderr,
            )
            any_gate_fail = True

    return 1 if any_gate_fail else 0


if __name__ == "__main__":
    sys.exit(main())

"""BVID-FE command-line interface.

Runs a BvidAnalysis from command-line arguments and prints the result as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Sequence

import bvidfe
from bvidfe.analysis import AnalysisConfig, BvidAnalysis
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.impact.mapping import ImpactEvent


def _parse_panel(spec: str) -> PanelGeometry:
    try:
        a, b = spec.lower().split("x")
        Lx, Ly = float(a), float(b)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--panel must be '<Lx>x<Ly>' (got {spec!r})") from exc
    if Lx <= 0 or Ly <= 0:
        raise argparse.ArgumentTypeError(
            f"--panel dimensions must be positive (got {Lx}x{Ly})"
        )
    return PanelGeometry(Lx_mm=Lx, Ly_mm=Ly)


def _parse_layup(spec: str) -> List[float]:
    try:
        return [float(x) for x in spec.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--layup must be a comma-separated list of ply angles in degrees"
        ) from exc


def _positive_float(spec: str) -> float:
    """argparse type that rejects non-positive numbers."""
    try:
        v = float(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number (got {spec!r})") from exc
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {v})")
    return v


def _existing_path(spec: str):
    """argparse type that requires a readable file at the given path."""
    from pathlib import Path

    p = Path(spec)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file not found: {spec}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {spec}")
    return p


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bvidfe",
        description="BVID-FE: Barely Visible Impact Damage residual-strength analysis.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"bvidfe {bvidfe.__version__}",
    )
    # Defer the MATERIAL_LIBRARY import + choices population to parser-build
    # time so --material rejects typos early with the standard argparse
    # 'invalid choice' message listing the four presets, instead of failing
    # downstream with a raw KeyError.
    from bvidfe.core.material import MATERIAL_LIBRARY

    p.add_argument(
        "--material",
        choices=sorted(MATERIAL_LIBRARY.keys()),
        help="Material preset name (e.g. IM7/8552)",
    )
    p.add_argument(
        "--layup",
        type=_parse_layup,
        help="Comma-separated ply angles in degrees, e.g. 0,45,-45,90",
    )
    p.add_argument("--thickness", type=_positive_float, help="Ply thickness in millimeters")
    p.add_argument(
        "--panel",
        type=_parse_panel,
        help="Panel dimensions as LxY in millimeters, e.g. 150x100",
    )
    p.add_argument("--loading", choices=["compression", "tension"])
    p.add_argument("--tier", default="empirical", choices=["empirical", "semi_analytical", "fe3d"])
    p.add_argument(
        "--energy",
        type=_positive_float,
        help="Impact energy in Joules (impact-driven workflow). Mutually exclusive with --cscan.",
    )
    p.add_argument(
        "--cscan",
        type=_existing_path,
        help="Path to a C-scan JSON file (inspection-driven workflow). "
        "Mutually exclusive with --energy. See docs/cscan_schema.md for the format.",
    )
    p.add_argument(
        "--impactor-diameter",
        type=_positive_float,
        default=16.0,
        help="Impactor diameter in mm (default 16.0)",
    )
    p.add_argument(
        "--mass", type=_positive_float, default=5.5, help="Impactor mass in kg (default 5.5)"
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Print only the knockdown scalar (residual / pristine) to stdout instead of the full JSON. "
        "Useful for shell pipelines: e.g. `bvidfe ... --quick | xargs -I {} ...`.",
    )
    p.add_argument(
        "--quick-json",
        action="store_true",
        help="Like --quick but emits a single-line JSON object {knockdown, "
        "residual_strength_MPa, pristine_strength_MPa, tier_used} instead of "
        "a bare scalar. Recommended for scripted callers that want the "
        "value plus tier provenance without parsing the full --json output.",
    )
    p.add_argument(
        "--list-materials",
        action="store_true",
        help="List available material presets with key properties and exit.",
    )
    return p


def _list_materials() -> None:
    from bvidfe.core.material import MATERIAL_LIBRARY

    print(f"{'Name':<18} {'E11':>8} {'E22':>7} {'Xt':>7} {'Xc':>7} {'Yt':>5} {'Yc':>5}")
    print(
        f"{'':-<18} {'-' * 8:>8} {'-' * 7:>7} {'-' * 7:>7} {'-' * 7:>7} {'-' * 5:>5} {'-' * 5:>5}"
    )
    for name, m in MATERIAL_LIBRARY.items():
        print(
            f"{name:<18} {m.E11:>8.0f} {m.E22:>7.0f} {m.Xt:>7.0f} {m.Xc:>7.0f} {m.Yt:>5.0f} {m.Yc:>5.0f}"
        )
    print("\nUnits: MPa. Use --material <Name> to select.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_materials:
        _list_materials()
        return 0
    # Base required args (always needed)
    missing = [
        n
        for n in ("material", "layup", "thickness", "panel", "loading")
        if getattr(args, n) is None
    ]
    if missing:
        parser.error(f"missing required arguments: {', '.join('--' + m for m in missing)}")
    # Exactly one of --energy (impact-driven) or --cscan (inspection-driven)
    if args.energy is None and args.cscan is None:
        parser.error("must provide either --energy (impact-driven) or --cscan (inspection-driven)")
    if args.energy is not None and args.cscan is not None:
        parser.error("--energy and --cscan are mutually exclusive")

    if args.energy is not None:
        cfg = AnalysisConfig(
            material=args.material,
            layup_deg=args.layup,
            ply_thickness_mm=args.thickness,
            panel=args.panel,
            loading=args.loading,
            tier=args.tier,
            impact=ImpactEvent(
                energy_J=args.energy,
                impactor=ImpactorGeometry(diameter_mm=args.impactor_diameter),
                mass_kg=args.mass,
            ),
        )
    else:
        from bvidfe.damage.io import load_cscan_json

        # args.cscan is already a validated Path (see _existing_path).
        damage = load_cscan_json(args.cscan)
        cfg = AnalysisConfig(
            material=args.material,
            layup_deg=args.layup,
            ply_thickness_mm=args.thickness,
            panel=args.panel,
            loading=args.loading,
            tier=args.tier,
            damage=damage,
        )
    if args.quick and args.quick_json:
        parser.error("--quick and --quick-json are mutually exclusive")
    result = BvidAnalysis(cfg).run()
    if args.quick_json:
        json.dump(
            {
                "knockdown": result.knockdown,
                "residual_strength_MPa": result.residual_strength_MPa,
                "pristine_strength_MPa": result.pristine_strength_MPa,
                "tier_used": result.tier_used,
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
    elif args.quick:
        print(f"{result.knockdown:.6f}")
    else:
        json.dump(result.to_dict(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

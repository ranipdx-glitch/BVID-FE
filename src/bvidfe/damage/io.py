"""C-scan / NDE JSON I/O for BVID damage states.

Schema: `docs/cscan_schema.md`.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Union

from bvidfe.damage.state import DamageState, DelaminationEllipse

SCHEMA_VERSION = "1.0"


class CScanSchemaError(ValueError):
    """Raised when a C-scan input does not conform to the BVID-FE schema."""


_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "dent_depth_mm", "fiber_break_radius_mm", "delaminations"}
)
_DELAMINATION_KEYS = frozenset(
    {"interface_index", "centroid_mm", "major_mm", "minor_mm", "orientation_deg"}
)


def damage_state_to_dict(ds: DamageState) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dent_depth_mm": ds.dent_depth_mm,
        "fiber_break_radius_mm": ds.fiber_break_radius_mm,
        "delaminations": [
            {
                "interface_index": e.interface_index,
                "centroid_mm": list(e.centroid_mm),
                "major_mm": e.major_mm,
                "minor_mm": e.minor_mm,
                "orientation_deg": e.orientation_deg,
            }
            for e in ds.delaminations
        ],
    }


def _require_finite_number(value: Any, label: str) -> float:
    """Validate ``value`` is a finite real number (rejects str, NaN, Inf)
    and return it as a float. Raises ``CScanSchemaError`` on every failure
    mode so callers see a uniform error type."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CScanSchemaError(f"{label} must be a number (got {type(value).__name__}: {value!r})")
    f = float(value)
    if not math.isfinite(f):
        raise CScanSchemaError(f"{label} must be finite (got {f})")
    return f


def _validate_dict(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise CScanSchemaError("top-level JSON must be an object")
    if "schema_version" not in data:
        raise CScanSchemaError("missing required field: schema_version")
    if data["schema_version"] != SCHEMA_VERSION:
        raise CScanSchemaError(
            f"unsupported schema_version {data['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    # Warn (but accept) on unknown top-level fields so old C-scan files keep
    # loading after we add new optional fields, but typos in known fields
    # ("dent_dept_mm") still get the user's attention.
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        warnings.warn(
            f"C-scan JSON contains unknown top-level field(s): {sorted(unknown)}; " "ignoring",
            stacklevel=3,
        )
    if "dent_depth_mm" not in data:
        raise CScanSchemaError("missing required field: dent_depth_mm")
    dent = _require_finite_number(data["dent_depth_mm"], "dent_depth_mm")
    if dent < 0:
        raise CScanSchemaError(f"dent_depth_mm must be >= 0 (got {dent})")
    if "fiber_break_radius_mm" in data:
        fbr = _require_finite_number(data["fiber_break_radius_mm"], "fiber_break_radius_mm")
        if fbr < 0:
            raise CScanSchemaError(f"fiber_break_radius_mm must be >= 0 (got {fbr})")
    if "delaminations" not in data or not isinstance(data["delaminations"], list):
        raise CScanSchemaError("delaminations must be a list")
    for i, d in enumerate(data["delaminations"]):
        if not isinstance(d, dict):
            raise CScanSchemaError(
                f"delaminations[{i}] must be an object " f"(got {type(d).__name__}: {d!r})"
            )
        for k in _DELAMINATION_KEYS:
            if k not in d:
                raise CScanSchemaError(f"delaminations[{i}] missing field {k!r}")
        unknown_keys = set(d) - _DELAMINATION_KEYS
        if unknown_keys:
            warnings.warn(
                f"delaminations[{i}] contains unknown field(s): "
                f"{sorted(unknown_keys)}; ignoring",
                stacklevel=3,
            )
        major = _require_finite_number(d["major_mm"], f"delaminations[{i}].major_mm")
        minor = _require_finite_number(d["minor_mm"], f"delaminations[{i}].minor_mm")
        if major <= 0 or minor <= 0:
            raise CScanSchemaError(
                f"delaminations[{i}] has non-positive axis (major={major}, minor={minor})"
            )
        iface = d["interface_index"]
        if isinstance(iface, bool) or not isinstance(iface, int):
            raise CScanSchemaError(
                f"delaminations[{i}].interface_index must be an int "
                f"(got {type(iface).__name__}: {iface!r})"
            )
        if iface < 0:
            raise CScanSchemaError(f"delaminations[{i}].interface_index must be >= 0")
        _require_finite_number(d["orientation_deg"], f"delaminations[{i}].orientation_deg")
        c = d["centroid_mm"]
        if not (isinstance(c, (list, tuple)) and len(c) == 2):
            raise CScanSchemaError(f"delaminations[{i}].centroid_mm must be [x, y]")
        _require_finite_number(c[0], f"delaminations[{i}].centroid_mm[0]")
        _require_finite_number(c[1], f"delaminations[{i}].centroid_mm[1]")


def damage_state_from_dict(data: Dict[str, Any]) -> DamageState:
    _validate_dict(data)
    try:
        dels = [
            DelaminationEllipse(
                interface_index=int(d["interface_index"]),
                centroid_mm=(float(d["centroid_mm"][0]), float(d["centroid_mm"][1])),
                major_mm=float(d["major_mm"]),
                minor_mm=float(d["minor_mm"]),
                orientation_deg=float(d["orientation_deg"]),
            )
            for d in data["delaminations"]
        ]
        return DamageState(
            delaminations=dels,
            dent_depth_mm=float(data["dent_depth_mm"]),
            fiber_break_radius_mm=float(data.get("fiber_break_radius_mm", 0.0)),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise CScanSchemaError(f"invalid damage record: {exc}") from exc


def save_cscan_json(ds: DamageState, path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(damage_state_to_dict(ds), indent=2))


def load_cscan_json(path: Union[str, Path]) -> DamageState:
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise CScanSchemaError(f"invalid JSON: {exc}") from exc
    return damage_state_from_dict(data)

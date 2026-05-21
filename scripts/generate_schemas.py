#!/usr/bin/env python3
"""Generate JSON Schemas (draft 2020-12) for BVID-FE public data contracts.

Currently emits:

* ``docs/schemas/analysis_config.json`` — ``AnalysisConfig``
* ``docs/schemas/analysis_results.json`` — ``AnalysisResults``

The generator walks ``__dataclass_fields__`` and ``__annotations__`` of the
target dataclasses and builds a JSON Schema document by hand. We deliberately
avoid Pydantic v2 as a hard dependency (it's heavy and would compile a large
C extension into the install path of every BVID-FE user). The conversion is
deterministic so it can be diff-checked with ``--check``.

Usage::

    python scripts/generate_schemas.py            # write schemas
    python scripts/generate_schemas.py --check    # fail if regen would diff

The ``--check`` mode is intended for use in a future CI step; the issue
explicitly asks us to expose the flag without wiring it into CI yet.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any, Dict, List

# We add ``src/`` to sys.path so the script can be run from a fresh clone
# without first ``pip install -e .``.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Importing after sys.path tweak.
from bvidfe.analysis.config import AnalysisConfig, MeshParams  # noqa: E402
from bvidfe.analysis.results import AnalysisResults, FieldResults  # noqa: E402
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry  # noqa: E402
from bvidfe.core.material import OrthotropicMaterial  # noqa: E402
from bvidfe.damage.state import DamageState, DelaminationEllipse  # noqa: E402
from bvidfe.impact.mapping import ImpactEvent  # noqa: E402

SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"
SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_BASE = "https://ranipdx-glitch.github.io/BVID-FE/schemas/"


# ---------------------------------------------------------------------------
# Type -> JSON Schema conversion
# ---------------------------------------------------------------------------


def _is_optional(tp: Any) -> bool:
    """``Optional[X]`` is ``Union[X, None]``; return True if NoneType is a member."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        return type(None) in typing.get_args(tp)
    return False


def _strip_optional(tp: Any) -> Any:
    """Drop ``NoneType`` from a Union, leaving either a bare type or a Union."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = tuple(a for a in typing.get_args(tp) if a is not type(None))
        if len(args) == 1:
            return args[0]
        return typing.Union[args]  # type: ignore[return-value]
    return tp


# Dataclasses we want to inline as ``$ref``s rather than re-emit at every use.
_KNOWN_DATACLASSES: Dict[type, str] = {
    PanelGeometry: "PanelGeometry",
    ImpactorGeometry: "ImpactorGeometry",
    ImpactEvent: "ImpactEvent",
    DelaminationEllipse: "DelaminationEllipse",
    DamageState: "DamageState",
    MeshParams: "MeshParams",
    FieldResults: "FieldResults",
    OrthotropicMaterial: "OrthotropicMaterial",
}


def _type_to_schema(tp: Any, defs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment.

    ``defs`` is the ``$defs`` dict for the current top-level schema. Nested
    dataclasses are emitted into ``defs`` on first encounter and referenced
    via ``$ref`` thereafter.
    """
    # Optional[X] -> the inner X (we set ``required`` outside this helper).
    if _is_optional(tp):
        inner = _strip_optional(tp)
        sub = _type_to_schema(inner, defs)
        # Allow null as a valid value for optional fields.
        if "type" in sub and isinstance(sub["type"], str):
            sub = {"type": [sub["type"], "null"], **{k: v for k, v in sub.items() if k != "type"}}
        else:
            sub = {"anyOf": [sub, {"type": "null"}]}
        return sub

    origin = typing.get_origin(tp)

    # Literal[...] -> enum
    if origin is typing.Literal:
        values = list(typing.get_args(tp))
        # Infer JSON type from the first literal value.
        sample = values[0]
        if isinstance(sample, bool):
            jt = "boolean"
        elif isinstance(sample, int):
            jt = "integer"
        elif isinstance(sample, float):
            jt = "number"
        else:
            jt = "string"
        return {"type": jt, "enum": values}

    # Union[X, Y] -> anyOf
    if origin in (typing.Union, types.UnionType):
        return {"anyOf": [_type_to_schema(a, defs) for a in typing.get_args(tp)]}

    # list / List / Sequence / tuple
    if origin in (list, typing.List, typing.Sequence, tuple, typing.Tuple):
        args = typing.get_args(tp)
        if not args:
            return {"type": "array"}
        if origin in (tuple, typing.Tuple) and len(args) > 1 and args[-1] is not Ellipsis:
            # Fixed-length tuple -> prefixItems
            return {
                "type": "array",
                "prefixItems": [_type_to_schema(a, defs) for a in args],
                "minItems": len(args),
                "maxItems": len(args),
            }
        return {"type": "array", "items": _type_to_schema(args[0], defs)}

    # dict / Dict / Mapping
    if origin in (dict, typing.Dict, typing.Mapping):
        args = typing.get_args(tp)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _type_to_schema(args[1], defs)}
        return {"type": "object"}

    # Bare types and known dataclasses
    if isinstance(tp, type):
        if tp in _KNOWN_DATACLASSES:
            name = _KNOWN_DATACLASSES[tp]
            if name not in defs:
                # Insert a placeholder so recursive types don't infinite-loop.
                defs[name] = {}
                defs[name] = _dataclass_schema_body(tp, defs)
            return {"$ref": f"#/$defs/{name}"}
        if tp is str:
            return {"type": "string"}
        if tp is bool:
            return {"type": "boolean"}
        if tp is int:
            return {"type": "integer"}
        if tp is float:
            return {"type": "number"}
        if tp is type(None):
            return {"type": "null"}
        # numpy arrays etc. — emit a permissive placeholder with a description.
        try:
            import numpy as np  # local import to avoid hard dependency at module load

            if issubclass(tp, np.ndarray):
                return {
                    "type": "array",
                    "description": f"numpy.ndarray ({tp.__name__}) — serialised as a nested list.",
                }
        except Exception:  # pragma: no cover - numpy is always installed in this repo
            pass
        # Fallback for unknown classes.
        return {"description": f"Python type {tp.__name__} (no JSON Schema mapping)"}

    # ForwardRef, TypeVar, etc. -> permissive placeholder.
    return {"description": f"Unresolved annotation {tp!r}"}


def _dataclass_schema_body(cls: type, defs: Dict[str, Any]) -> Dict[str, Any]:
    """Build the JSON Schema *body* (without ``$schema`` / ``$id``) for a dataclass."""
    hints = typing.get_type_hints(cls)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for f in dataclasses.fields(cls):
        ann = hints.get(f.name, f.type)
        properties[f.name] = _type_to_schema(ann, defs)
        # A field is required when it has no default AND no default_factory.
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            required.append(f.name)
    body: Dict[str, Any] = {
        "type": "object",
        "title": cls.__name__,
        "description": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
        "properties": properties,
    }
    if required:
        body["required"] = required
    body["additionalProperties"] = False
    return body


def dataclass_to_jsonschema(cls: type, schema_id: str) -> Dict[str, Any]:
    """Top-level JSON Schema (draft 2020-12) for a dataclass."""
    defs: Dict[str, Any] = {}
    body = _dataclass_schema_body(cls, defs)
    schema: Dict[str, Any] = {
        "$schema": SCHEMA_URI,
        "$id": schema_id,
        **body,
    }
    if defs:
        schema["$defs"] = dict(sorted(defs.items()))
    return schema


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


TARGETS = [
    (AnalysisConfig, SCHEMA_BASE + "analysis_config.json", "analysis_config.json"),
    (AnalysisResults, SCHEMA_BASE + "analysis_results.json", "analysis_results.json"),
]


def _serialise(schema: Dict[str, Any]) -> str:
    """Deterministic JSON encoding — sorted keys, 2-space indent, trailing newline."""
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail with exit code 1 if regenerating would change any file on disk.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCHEMA_DIR,
        help=f"Destination directory (default: {SCHEMA_DIR.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    diffs: List[str] = []
    for cls, schema_id, filename in TARGETS:
        schema = dataclass_to_jsonschema(cls, schema_id)
        rendered = _serialise(schema)
        path = args.out_dir / filename
        if args.check:
            existing = path.read_text() if path.exists() else ""
            if existing != rendered:
                diffs.append(str(path.relative_to(REPO_ROOT)))
        else:
            path.write_text(rendered)
            print(f"wrote {path.relative_to(REPO_ROOT)} ({len(rendered):,} bytes)")

    if args.check:
        if diffs:
            print(
                "generate_schemas.py --check: regeneration would change these files:",
                file=sys.stderr,
            )
            for d in diffs:
                print(f"  {d}", file=sys.stderr)
            print(
                "Run `python scripts/generate_schemas.py` to refresh them.",
                file=sys.stderr,
            )
            return 1
        print("generate_schemas.py --check: schemas are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

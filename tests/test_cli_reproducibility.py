"""CLI numeric snapshot tests.

The BVID-FE CLI emits a JSON result document, but no existing test pins
the exact numeric signature of canonical runs. A dependency bump (numpy,
scipy) or an accidental formula tweak can shift the ``knockdown`` by a
few percent and pass CI unnoticed. This module locks down the numeric
output of three canonical CLI invocations.

Tolerance
---------
The empirical and semi_analytical tiers are purely deterministic
closed-form / direct-solve evaluations: rerunning the same command
three times produced bit-identical floats. We assert with
``rel=1e-6`` so a tweak that shifts ``knockdown`` by even 0.0001%
trips the test, while still leaving headroom for benign cross-platform
floating-point drift in the lower bits.

The 3D FE tier (``fe3d``) is intentionally excluded — a single
invocation already exceeds the 5-second budget for the snapshot suite,
and the iterative eigensolve introduces low-bit drift that would
require a much looser tolerance (~1e-4) to be useful. Cross-tier
agreement is covered by ``tests/test_cross_tier_consistency.py``.

Regenerating snapshots
----------------------
When a formula change is intentional, re-capture the snapshots by
running each CLI invocation below with ``--quick-json`` and pasting
the new numeric fields into ``CANONICAL_RUNS``. Example::

    python -m bvidfe.cli --tier empirical --material IM7/8552 \\
        --layup 0,45,-45,90,90,-45,45,0,0,45,-45,90,90,-45,45,0 \\
        --thickness 0.152 --panel 200x150 --loading compression \\
        --energy 20 --quick-json

Always rerun ``pytest tests/test_cli_reproducibility.py`` after
updating, and document the formula change in the PR description.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest


# Each entry: (id, CLI args, expected JSON snapshot fields).
# Snapshots captured 2026-05-20 against bvidfe 0.2.0.dev0 with
# numpy 2.x and scipy 1.14.x on linux x86_64.
CANONICAL_RUNS: list[tuple[str, list[str], dict[str, Any]]] = [
    (
        "empirical_quasi_iso_quad_20J",
        [
            "--tier",
            "empirical",
            "--material",
            "IM7/8552",
            "--layup",
            "0,45,-45,90,90,-45,45,0,0,45,-45,90,90,-45,45,0",
            "--thickness",
            "0.152",
            "--panel",
            "200x150",
            "--loading",
            "compression",
            "--energy",
            "20",
        ],
        {
            "knockdown": 0.43923234565804536,
            "residual_strength_MPa": 389.81870677151534,
            "pristine_strength_MPa": 887.5000000000002,
            "dpa_mm2": 7823.814020954099,
            "tier_used": "empirical",
        },
    ),
    (
        "semi_analytical_quasi_iso_quad_10J",
        [
            "--tier",
            "semi_analytical",
            "--material",
            "IM7/8552",
            "--layup",
            "0,45,-45,90,90,-45,45,0,0,45,-45,90,90,-45,45,0",
            "--thickness",
            "0.152",
            "--panel",
            "200x150",
            "--loading",
            "compression",
            "--energy",
            "10",
        ],
        {
            # Recaptured after rebase onto main with the #29 (full-plate
            # SSSS buckling dims) and #18 (sublaminate selection by
            # through-thickness) fixes — semi_analytical knockdown
            # tightened from 0.221 to 0.055.
            "knockdown": 0.05512833900538707,
            "residual_strength_MPa": 48.92640086728104,
            "pristine_strength_MPa": 887.5000000000002,
            "dpa_mm2": 3659.9324669198636,
            "tier_used": "semi_analytical",
        },
    ),
    (
        "empirical_cross_ply_tension_15J",
        [
            "--tier",
            "empirical",
            "--material",
            "IM7/8552",
            "--layup",
            "0,90,0,90,90,0,90,0",
            "--thickness",
            "0.2",
            "--panel",
            "150x100",
            "--loading",
            "tension",
            "--energy",
            "15",
        ],
        {
            "knockdown": 0.21852316445726744,
            "residual_strength_MPa": 287.6857460079925,
            "pristine_strength_MPa": 1316.4999999999998,
            "dpa_mm2": 8901.845339242213,
            "tier_used": "empirical",
        },
    ),
]


# Fields snapshot-asserted with pytest.approx; the categorical
# ``tier_used`` is compared exactly.
NUMERIC_FIELDS = (
    "knockdown",
    "residual_strength_MPa",
    "pristine_strength_MPa",
    "dpa_mm2",
)


@pytest.mark.parametrize(
    ("case_id", "cli_args", "expected"),
    CANONICAL_RUNS,
    ids=[c[0] for c in CANONICAL_RUNS],
)
def test_cli_numeric_snapshot(
    case_id: str,
    cli_args: list[str],
    expected: dict[str, Any],
) -> None:
    """Assert the CLI's JSON output matches the captured numeric snapshot.

    Invoked via ``python -m bvidfe.cli`` (not the installed ``bvidfe``
    console script) so the test works in environments where the package
    is importable but the console entry point is not on PATH.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bvidfe.cli", *cli_args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed for {case_id}: stderr={result.stderr!r}"

    data = json.loads(result.stdout)

    assert data["tier_used"] == expected["tier_used"], (
        f"{case_id}: tier_used mismatch (got {data['tier_used']!r}, "
        f"expected {expected['tier_used']!r})"
    )
    for field in NUMERIC_FIELDS:
        assert field in data, f"{case_id}: missing field {field!r} in CLI output"
        assert data[field] == pytest.approx(expected[field], rel=1e-6), (
            f"{case_id}: {field} drifted "
            f"(got {data[field]!r}, expected {expected[field]!r}). "
            "If this drift is intentional, re-capture the snapshot — see "
            "the module docstring for the regeneration recipe."
        )

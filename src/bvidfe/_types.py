"""Shared Literal type aliases and their runtime-validation sets.

This module centralises the small set of string-enum-style options that flow
through ``AnalysisConfig`` and the analysis pipeline so they can be referenced
from a single place. Caller modules (e.g. ``bvid.py``, ``fe_tier.py``,
``failure/evaluator.py``) will migrate to these aliases in follow-up PRs;
this PR only wires them through :class:`bvidfe.analysis.config.AnalysisConfig`.

The frozensets ``_TIER_NAMES``, ``_LOADING_MODES``, and ``_CRITERION_NAMES``
mirror each ``Literal`` so callers have an O(1) runtime-validation set that
stays in sync with the static type. Keep the two in lockstep when adding new
values.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Literal aliases
# ---------------------------------------------------------------------------

#: Identifier for the analysis tier selected on ``AnalysisConfig.tier``.
TierName = Literal["empirical", "semi_analytical", "fe3d"]

#: In-plane loading mode selected on ``AnalysisConfig.loading``.
LoadingMode = Literal["compression", "tension"]

#: Failure-criterion name. ``failure/evaluator.py`` currently defines its own
#: ``CriterionName`` alias; this duplicate exists so future callers can import
#: from a single place once the evaluator is migrated in a follow-up PR.
CriterionName = Literal["tsai_wu", "larc05"]


# ---------------------------------------------------------------------------
# Runtime-validation sets (keep in lockstep with the Literals above)
# ---------------------------------------------------------------------------

_TIER_NAMES: frozenset[str] = frozenset({"empirical", "semi_analytical", "fe3d"})
_LOADING_MODES: frozenset[str] = frozenset({"compression", "tension"})
_CRITERION_NAMES: frozenset[str] = frozenset({"tsai_wu", "larc05"})


__all__ = [
    "TierName",
    "LoadingMode",
    "CriterionName",
    "_TIER_NAMES",
    "_LOADING_MODES",
    "_CRITERION_NAMES",
]

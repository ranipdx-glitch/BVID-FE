"""Parametric sweep utilities producing pandas DataFrames and CSV output.

All three sweep entry points (`sweep_energies`, `sweep_layups`,
`sweep_thicknesses`) accept two optional parameters:

  * ``on_error`` — ``"raise"`` (default), ``"skip"``, or ``"warn"``.
    With ``"raise"`` a per-iteration failure aborts the whole sweep and
    no CSV is written (the legacy behaviour). With ``"skip"`` the failed
    row is filled with ``NaN`` numeric fields plus an ``error`` column
    containing the exception message, and the sweep continues; partial
    results still reach the DataFrame and the CSV. ``"warn"`` is identical
    to ``"skip"`` but additionally emits a ``UserWarning`` per failure.

  * ``progress_callback`` — optional ``Callable[[int, int], None]``
    invoked as ``progress_callback(i_done, n_total)`` after each iteration
    completes (success or failure). Used by the GUI ``SweepWorker`` to
    drive its progress bar and by long-running scripts to report progress
    to stderr / a logger.

Together these make sweeps robust to a single bad input (a degenerate
mesh, a Tsai-Wu invalid combination, an out-of-range Olsson regime) and
preserve the partial CSV that previously evaporated when the sweep
aborted mid-run.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence

import pandas as pd

from bvidfe.analysis import AnalysisConfig, BvidAnalysis
from bvidfe.impact.mapping import DPACapClipWarning, SmallMassQuasiStaticWarning

_ProgressCallback = Callable[[int, int], None]
_ON_ERROR_VALUES = ("raise", "skip", "warn")


_DEDUP_WARNING_CATEGORIES = (SmallMassQuasiStaticWarning, DPACapClipWarning)


@contextmanager
def _dedupe_impact_warnings() -> Iterator[None]:
    """Emit each impact-mapping diagnostic at most once for the whole sweep.

    ``impact_to_damage`` raises ``SmallMassQuasiStaticWarning`` /
    ``DPACapClipWarning`` once per sweep point. Python's ``"once"`` /
    ``"default"`` filter actions key on the *message text*, which here
    embeds per-point values (the predicted DPA, the mass ratio), so the
    filter never recognises them as duplicates and the diagnostic floods
    stderr once per iteration. Dedupe by *category* explicitly instead.
    Non-impact warnings (e.g. the ``on_error="warn"`` per-failure
    ``UserWarning``) pass through unchanged.
    """
    seen: set[type] = set()
    with warnings.catch_warnings():
        outer_show = warnings.showwarning
        # See every warning so we can decide per category; the original
        # showwarning is still used for the first of each / pass-through.
        warnings.simplefilter("always")

        def _show(message, category, filename, lineno, file=None, line=None):
            if issubclass(category, _DEDUP_WARNING_CATEGORIES):
                if category in seen:
                    return
                seen.add(category)
            outer_show(message, category, filename, lineno, file, line)

        warnings.showwarning = _show
        yield


def _run_one(cfg: AnalysisConfig) -> dict:
    """Run a single analysis and return a dict of key fields."""
    result = BvidAnalysis(cfg).run()
    return {
        "knockdown": result.knockdown,
        "residual_MPa": result.residual_strength_MPa,
        "pristine_MPa": result.pristine_strength_MPa,
        "dpa_mm2": result.dpa_mm2,
        "dent_mm": result.damage.dent_depth_mm,
        "n_delaminations": len(result.damage.delaminations),
        "tier_used": result.tier_used,
    }


def _nan_row() -> dict:
    """Row schema matching _run_one with NaN numerics — used for failed
    iterations under on_error in {'skip', 'warn'}."""
    return {
        "knockdown": math.nan,
        "residual_MPa": math.nan,
        "pristine_MPa": math.nan,
        "dpa_mm2": math.nan,
        "dent_mm": math.nan,
        "n_delaminations": 0,
        "tier_used": "",
    }


def _try_run_one(cfg: AnalysisConfig, on_error: str, label: str) -> dict:
    """Invoke ``_run_one`` and translate any exception to a NaN-filled row
    according to ``on_error``. ``label`` is used in the warning text."""
    if on_error not in _ON_ERROR_VALUES:
        raise ValueError(f"on_error must be one of {_ON_ERROR_VALUES} (got {on_error!r})")
    try:
        return _run_one(cfg)
    except Exception as exc:  # noqa: BLE001
        if on_error == "raise":
            raise
        row = _nan_row()
        row["error"] = f"{type(exc).__name__}: {exc}"
        if on_error == "warn":
            warnings.warn(
                f"sweep iteration {label} failed: {row['error']}",
                stacklevel=3,
            )
        return row


# Fixed result schema. The swept independent variable goes first; `error`
# is always present (NaN/empty on success) so downstream `df["error"]`
# never KeyErrors and column order is identical across runs (#38).
_RESULT_COLS = (
    "knockdown",
    "residual_MPa",
    "pristine_MPa",
    "dpa_mm2",
    "dent_mm",
    "n_delaminations",
    "tier_used",
    "error",
)


def _finalize(rows: List[dict], swept_col: str) -> pd.DataFrame:
    """Build the DataFrame with a deterministic column order regardless of
    how many iterations failed: ``[swept_col, *_RESULT_COLS]`` with
    ``error`` always present (#38)."""
    return pd.DataFrame(rows).reindex(columns=[swept_col, *_RESULT_COLS])


def _write_csv(df: pd.DataFrame, csv_path: Optional[Path]) -> None:
    if csv_path is not None:
        df.to_csv(Path(csv_path), index=False)


def _emit_progress(cb: Optional[_ProgressCallback], i: int, n: int) -> None:
    if cb is not None:
        cb(i, n)


def sweep_energies(
    base_cfg: AnalysisConfig,
    energies_J: Sequence[float],
    csv_path: Optional[Path | str] = None,
    *,
    on_error: str = "raise",
    progress_callback: Optional[_ProgressCallback] = None,
) -> pd.DataFrame:
    """Sweep impact energies; base_cfg must have `impact` set.

    See module docstring for ``on_error`` and ``progress_callback`` semantics.
    """
    if base_cfg.impact is None:
        raise ValueError("sweep_energies requires base_cfg.impact to be set")
    rows: List[dict] = []
    n = len(energies_J)
    # Only the impact energy (→ damage) varies here; the mesh-defining inputs
    # are constant, so build_fe_mesh reuses the cached mesh skeleton across
    # iterations and only re-derives the per-element damage factors (#23).
    with _dedupe_impact_warnings():
        for i, E in enumerate(energies_J):
            new_impact = replace(base_cfg.impact, energy_J=float(E))
            cfg = replace(base_cfg, impact=new_impact)
            row = _try_run_one(cfg, on_error, f"energy_J={float(E):g}")
            row["energy_J"] = float(E)
            rows.append(row)
            _emit_progress(progress_callback, i + 1, n)
    df = _finalize(rows, "energy_J")
    _write_csv(df, Path(csv_path) if csv_path else None)
    return df


def sweep_layups(
    base_cfg: AnalysisConfig,
    layups: Sequence[Sequence[float]],
    csv_path: Optional[Path | str] = None,
    *,
    on_error: str = "raise",
    progress_callback: Optional[_ProgressCallback] = None,
) -> pd.DataFrame:
    """Sweep layup sequences."""
    rows: List[dict] = []
    n = len(layups)
    with _dedupe_impact_warnings():
        for i, layup in enumerate(layups):
            cfg = replace(base_cfg, layup_deg=list(layup))
            layup_str = "/".join(f"{a:g}" for a in layup)
            row = _try_run_one(cfg, on_error, f"layup={layup_str}")
            row["layup"] = layup_str
            rows.append(row)
            _emit_progress(progress_callback, i + 1, n)
    df = _finalize(rows, "layup")
    _write_csv(df, Path(csv_path) if csv_path else None)
    return df


def sweep_thicknesses(
    base_cfg: AnalysisConfig,
    ply_thicknesses_mm: Sequence[float],
    csv_path: Optional[Path | str] = None,
    *,
    on_error: str = "raise",
    progress_callback: Optional[_ProgressCallback] = None,
) -> pd.DataFrame:
    """Sweep ply thickness values."""
    rows: List[dict] = []
    n = len(ply_thicknesses_mm)
    with _dedupe_impact_warnings():
        for i, t in enumerate(ply_thicknesses_mm):
            cfg = replace(base_cfg, ply_thickness_mm=float(t))
            row = _try_run_one(cfg, on_error, f"ply_thickness_mm={float(t):g}")
            row["ply_thickness_mm"] = float(t)
            rows.append(row)
            _emit_progress(progress_callback, i + 1, n)
    df = _finalize(rows, "ply_thickness_mm")
    _write_csv(df, Path(csv_path) if csv_path else None)
    return df

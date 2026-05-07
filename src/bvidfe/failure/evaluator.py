"""Failure-criterion evaluator across an element x Gauss-point stress field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from bvidfe.core.material import OrthotropicMaterial
from bvidfe.failure.larc05 import larc05_index, larc05_index_batch
from bvidfe.failure.tsai_wu import tsai_wu_index, tsai_wu_index_batch

CriterionName = Literal["tsai_wu", "larc05"]


@dataclass
class LaminateFailureReport:
    """Summary of a failure evaluation across a laminate stress field."""

    max_index: float
    critical_element: int
    critical_gauss_point: int
    criterion: str


class FailureEvaluator:
    """Applies a failure criterion across every (element, gauss-point) stress."""

    def __init__(self, material: OrthotropicMaterial, criterion: CriterionName = "tsai_wu"):
        if criterion not in ("tsai_wu", "larc05"):
            raise ValueError(f"unknown criterion {criterion!r}")
        self.material = material
        self.criterion = criterion

    def _index(self, stress):
        if self.criterion == "tsai_wu":
            return tsai_wu_index(self.material, stress)
        return larc05_index(self.material, stress)

    def _index_batch(self, stresses: np.ndarray) -> np.ndarray:
        if self.criterion == "tsai_wu":
            return tsai_wu_index_batch(self.material, stresses)
        return larc05_index_batch(self.material, stresses)

    def evaluate(self, stress_field: np.ndarray) -> LaminateFailureReport:
        """Return the highest failure index across an (n_elem, n_gp, 6) field.

        Vectorised: a single ``_index_batch`` call replaces the prior nested
        Python loop. Numerical equivalence to the scalar form is locked by
        ``tests/failure/test_evaluator.py::test_evaluate_matches_scalar_loop``.
        """
        assert stress_field.ndim == 3 and stress_field.shape[2] == 6
        idx_grid = self._index_batch(stress_field)  # shape (n_elem, n_gp)
        flat_argmax = int(np.argmax(idx_grid))
        crit_e, crit_g = np.unravel_index(flat_argmax, idx_grid.shape)
        return LaminateFailureReport(
            max_index=float(idx_grid[crit_e, crit_g]),
            critical_element=int(crit_e),
            critical_gauss_point=int(crit_g),
            criterion=self.criterion,
        )

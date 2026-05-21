"""Failure-criterion evaluator across an element x Gauss-point stress field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from bvidfe.core.material import OrthotropicMaterial
from bvidfe.failure.larc05 import larc05_index, larc05_index_batch
from bvidfe.failure.tsai_wu import tsai_wu_index, tsai_wu_index_batch

CriterionName = Literal["tsai_wu", "larc05"]

# Registry mapping criterion name → batch index function. New criteria are
# added by extending this dict (and the ``CriterionName`` Literal) rather
# than by adding another string-comparison branch in the evaluator.
_CRITERION_REGISTRY: dict[
    CriterionName, Callable[[OrthotropicMaterial, np.ndarray], np.ndarray]
] = {
    "tsai_wu": tsai_wu_index_batch,
    "larc05": larc05_index_batch,
}

# Scalar variants used by ``FailureEvaluator._index`` (per-point evaluation).
# Kept parallel to ``_CRITERION_REGISTRY`` so any extension touches both.
_CRITERION_SCALAR_REGISTRY: dict[
    CriterionName, Callable[[OrthotropicMaterial, np.ndarray], float]
] = {
    "tsai_wu": tsai_wu_index,
    "larc05": larc05_index,
}


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
        if criterion not in _CRITERION_REGISTRY:
            valid = sorted(_CRITERION_REGISTRY.keys())
            raise ValueError(f"unknown criterion {criterion!r}; valid options are {valid}")
        self.material = material
        self.criterion = criterion
        # Cache function pointers at construction time so ``evaluate`` and
        # ``_index`` no longer branch on raw strings.
        self._evaluate_fn = _CRITERION_REGISTRY[criterion]
        self._scalar_fn = _CRITERION_SCALAR_REGISTRY[criterion]

    def _index(self, stress):
        return self._scalar_fn(self.material, stress)

    def _index_batch(self, stresses: np.ndarray) -> np.ndarray:
        return self._evaluate_fn(self.material, stresses)

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

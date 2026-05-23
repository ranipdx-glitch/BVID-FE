---
name: add-failure-criterion
description: Add a new failure criterion (e.g. Hashin, Christensen, max-stress) to the BVID-FE failure module. Use when the user wants to add, register, or wire up a failure index alongside the existing Tsai-Wu / LaRC05 / Puck criteria.
---

# Add a new failure criterion

Failure criteria plug into `FailureEvaluator` via a pair of registries.
A criterion is "added" only when it shows up in **four** places that
must move in lockstep:

1. The implementation module under `src/bvidfe/failure/`
2. `_CRITERION_REGISTRY` (batch) in `src/bvidfe/failure/evaluator.py`
3. `_CRITERION_SCALAR_REGISTRY` (scalar) in the same file
4. `CriterionName` Literal **and** `_CRITERION_NAMES` frozenset in
   `src/bvidfe/_types.py`

Forgetting any one of these causes a confusing runtime error far
from the actual omission. Treat it as one atomic change.

## Pre-flight

Confirm with the user:

1. The criterion name (the registry key, e.g. `"hashin"`). Use
   `snake_case`, lowercase. This is also the string users pass to
   `FailureEvaluator(criterion=...)`.
2. The literature source for the formulation.
3. Whether the criterion is direction-dependent or has any
   in-plane-only assumptions worth documenting.

## Required public interface

The new module must expose **two** functions with these exact
signatures, mirroring `larc05.py` / `puck.py` / `tsai_wu.py`:

```python
def <name>_index(material: OrthotropicMaterial, stress: np.ndarray) -> float:
    """Per-point failure index. stress shape: (6,) in Voigt order."""

def <name>_index_batch(material: OrthotropicMaterial, stresses: np.ndarray) -> np.ndarray:
    """Vectorised. stresses shape: (n_elem, n_gp, 6). Returns (n_elem, n_gp)."""
```

The batch form is **not optional** — `FailureEvaluator.evaluate` calls
it directly and the scalar-loop equivalent is what
`tests/failure/test_evaluator.py::test_evaluate_matches_scalar_loop`
checks. The two functions must agree numerically; the test will fail
otherwise.

Voigt stress order in this codebase: `[σ11, σ22, σ33, σ23, σ13, σ12]`.
Confirm against any other criterion file before assuming.

## Steps

### 1. Implement the criterion

Create `src/bvidfe/failure/<name>.py`. Use the closest existing
criterion as a structural template:

- `tsai_wu.py` — quadratic interaction, simplest reference
- `larc05.py` — fibre/matrix/kinking modes, decomposed
- `puck.py` — action-plane search, most complex

Vectorise the batch form properly (numpy operations on the full
`(n_elem, n_gp, 6)` array, not a Python loop) — the evaluator
delegates to it inside the hot path.

### 2. Register in `failure/evaluator.py`

Add the import and the two registry entries:

```python
from bvidfe.failure.<name> import <name>_index, <name>_index_batch

_CRITERION_REGISTRY[CriterionName] = {
    ...,
    "<name>": <name>_index_batch,
}

_CRITERION_SCALAR_REGISTRY[CriterionName] = {
    ...,
    "<name>": <name>_index,
}
```

Add the entry to **both** registries. The evaluator caches both
function pointers in `__init__`.

### 3. Extend the type alias in `_types.py`

In `src/bvidfe/_types.py`, add the new string to:

```python
CriterionName = Literal["tsai_wu", "larc05", "puck", "<name>"]

_CRITERION_NAMES: frozenset[str] = frozenset({"tsai_wu", "larc05", "puck", "<name>"})
```

The module comment explicitly calls out that the Literal and the
frozenset must move together — keep them in lockstep.

### 4. Tests in `tests/failure/`

Create `tests/failure/test_<name>.py`. At minimum:

- A pristine-stress sanity check (index < 1 for a clearly safe state)
- A failure-stress sanity check (index ≥ 1 at the documented strength)
- A scalar-vs-batch equivalence test on a random stress field
- A check that `FailureEvaluator(material, criterion="<name>")`
  constructs and `.evaluate(...)` runs

Then add a parametrize entry to `tests/failure/test_evaluator.py` if
it parametrizes over all criteria — grep for the existing names to
confirm.

### 5. Validate

```bash
ruff check src tests
black src tests
pytest tests/failure -v
```

The full failure suite must stay green; the scalar-vs-batch test in
`test_evaluator.py` is the canonical guard against batch/scalar drift.

## Things this skill must never do

- Never add the criterion to only one of the two registries.
- Never extend `CriterionName` without also extending `_CRITERION_NAMES`.
- Never assume Voigt order — confirm it against an existing criterion
  file in this repo (it is `[σ11, σ22, σ33, σ23, σ13, σ12]`).
- Never skip the scalar-vs-batch test — it is what catches the most
  common implementation bug.

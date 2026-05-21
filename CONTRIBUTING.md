# Contributing to BVID-FE

Thank you for your interest in contributing to BVID-FE!

## Reporting Bugs

Open an [issue](https://github.com/ranipdx-glitch/BVID-FE/issues) with:
- Python version and OS
- Steps to reproduce the problem
- Expected vs actual behavior
- Error messages or tracebacks

## Suggesting Features

Open an issue describing:
- The use case (what composite damage analysis problem you are solving)
- Expected behavior and outputs
- Any relevant references or equations

## Development Setup

```bash
git clone https://github.com/ranipdx-glitch/BVID-FE.git
cd BVID-FE
pip install -e ".[all]"
pytest tests/ -v
```

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Single subsystem
pytest tests/impact/ -v
pytest tests/failure/ -v
pytest tests/analysis/ -v
```

## Code Style

- Format with **black** and lint with **ruff** (pre-commit hooks enforce both)
- Add docstrings to all public functions and classes
- Include units in variable names and docstrings (MPa, mm, rad, J)
- Use SI units throughout (millimetres, MPa, kg)

## Pre-commit setup

Install pre-commit hooks once after cloning so black, ruff, and basic file
hygiene checks run automatically before each commit:

```bash
pip install pre-commit
pre-commit install
```

### Before committing

If you haven't installed pre-commit, run these three commands locally before
pushing — CI runs `black --check` and **will fail** if any file is not
already formatted, so you must auto-format with `black` (not just verify
with `black --check`):

```bash
ruff check src tests
black src tests        # auto-format in place; do NOT use --check here
pytest -q
```

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for any new functionality before implementing it
4. Ensure all 149+ tests pass (`pytest tests/ -v`)
5. Submit a pull request with a clear description of what was changed and why

## Examples

Runnable examples live in `examples/`. The script set (`01_empirical_quick.py` ... `05_tier_comparison_sweep.py`) covers the headless workflows; `examples/quickstart.ipynb` is an annotated Jupyter notebook that mirrors `01_empirical_quick.py` with LaTeX derivations of the Olsson, Soutis, and Whitney-Nuismer models plus inline Plotly visualisations. **Strip notebook outputs before committing** with `nbstripout examples/quickstart.ipynb` (`pip install nbstripout`); commits with output cells will be flagged in review.

## Adding New Materials

Add a new entry to `MATERIAL_PRESETS` in `src/bvidfe/core/material.py` using the `OrthotropicMaterial` dataclass. Include all orthotropic stiffness constants (MPa), strength values (MPa), interlaminar fracture toughnesses (N/mm), and density (kg/mm^3). Include a literature source reference in the docstring. Add tests in `tests/core/test_material.py`.

## Adding New Failure Criteria

Create a new file in `src/bvidfe/failure/` subclassing `FailureCriterion` from `failure/base.py`. Register it in `FailureEvaluator` in `failure/evaluator.py`. Add tests in `tests/failure/`.

## Adding New Modeling Tiers

1. Implement the tier function in `src/bvidfe/analysis/my_tier.py`
2. Add the tier string to the `Literal` type in `AnalysisConfig` (`analysis/config.py`)
3. Add the dispatch branch in `BvidAnalysis.run()` (`analysis/bvid.py`)
4. Add end-to-end tests in `tests/analysis/test_my_tier_path.py`

## Claude Code on the web

This repo ships a `SessionStart` hook at `.claude/hooks/session_start.sh` (wired
up via `.claude/settings.json`) that runs `black --check src tests` and
`ruff check src tests` at the start of every Claude Code session and reports
any drift. It is informational only (always exits 0) and prevents agents from
delegating commits while formatting is broken. If `black`/`ruff` are not
installed yet, the hook prints a friendly note and exits cleanly.

## Releasing

BVID-FE publishes to PyPI via **OIDC Trusted Publishing** — there is no
long-lived API token stored in the repo. The release flow is driven by
`.github/workflows/publish.yml`, which triggers on any tag matching `v*`.

### One-time PyPI setup (maintainer only)

Trusted Publishing must be configured **once** on PyPI for this repo. See
[PyPI's Trusted Publishing docs](https://docs.pypi.org/trusted-publishers/)
for the canonical instructions. Register a pending or existing publisher
with:

- **Repository owner:** `ranipdx-glitch`
- **Repository name:** `BVID-FE`
- **Workflow filename:** `publish.yml`
- **Environment name:** `release`

A matching GitHub Actions environment named `release` must also exist on
the repo (Settings → Environments → New environment → `release`). The
workflow declares `environment: release`, so it will queue indefinitely
until that environment is created — which is the intended fail-safe.

### Cut-a-release flow

1. Bump `[project].version` in `pyproject.toml` (e.g. `0.2.0` →
   `0.2.1` for a patch, `0.3.0` for a feature release).
2. Update `CHANGELOG.md`: move the `## [Unreleased]` notes into a
   new `## [X.Y.Z] - YYYY-MM-DD` section (today's date) and leave a
   fresh empty `## [Unreleased]` above it.
3. Commit those two changes on `main`.
4. Tag the commit with `git tag vX.Y.Z` (note the leading `v`) and
   push with `git push origin vX.Y.Z`.
5. The `Publish to PyPI` workflow takes over: it verifies the tag
   matches `pyproject.toml`, builds an sdist + wheel, runs
   `twine check`, and uploads to PyPI through OIDC. No manual token
   handling.

If the tag/version check fails, delete the tag (`git push --delete
origin vX.Y.Z` and `git tag -d vX.Y.Z`), fix the version mismatch,
and re-tag.

## Code of Conduct

Be respectful. Focus feedback on code and ideas, not people. Contributions at any experience level are welcome.

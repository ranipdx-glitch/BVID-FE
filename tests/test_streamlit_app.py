"""Streamlit ``app.py`` importability smoke test.

A typo or broken import in ``app.py`` is invisible to the rest of the
test suite — nothing in ``tests/`` imports the file — so the failure
only surfaces when someone visits the deployed Streamlit URL. This
module imports ``app`` from the repository root and fails fast if that
fails.

The test sets ``STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`` so the
import does not attempt any usage-stat network handshake. ``app.py``
runs Streamlit API calls at module top level (``st.set_page_config``,
``st.title``, sidebar widgets); these are tolerated by Streamlit in
"bare mode" (just a noisy warning) and do not raise.

If Streamlit is somehow unavailable in the runtime environment the test
is skipped rather than errored — the package depends on ``streamlit>=1.32``
but defensive skipping keeps this smoke test useful in stripped-down
environments.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")


def test_app_module_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import app`` must succeed without raising.

    Catches typos in import statements, syntax errors, and breakage in
    any module ``app.py`` pulls in at import time (``bvidfe.analysis``,
    ``bvidfe.viz``, ``bvidfe.sweep``, ...).
    """
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    # Telemetry off too — belt-and-braces for offline CI environments.
    monkeypatch.setenv("STREAMLIT_BROWSER_SERVER_ADDRESS", "localhost")

    # Make the repo root importable so ``import app`` resolves to
    # ``<repo>/app.py``. The repo is src-layout, so the root is not
    # automatically on sys.path the way ``src/`` is.
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    # Force a fresh import — otherwise a previously cached module would
    # mask an import-time regression introduced after pytest started.
    sys.modules.pop("app", None)

    app_module = importlib.import_module("app")

    # Spot-check a couple of public names that the Streamlit page relies
    # on; if app.py is imported but its globals never got built, those
    # assertions catch a half-initialised module (e.g. early ``return``).
    assert hasattr(app_module, "MATERIAL_NAMES"), "app.py did not finish initialising"
    assert app_module.MATERIAL_NAMES, "MATERIAL_NAMES should not be empty"
    assert hasattr(app_module, "_parse_layup"), "expected _parse_layup helper in app.py"

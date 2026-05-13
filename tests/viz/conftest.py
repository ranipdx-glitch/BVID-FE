"""Belt-and-suspenders Xvfb startup for the pyvista-backed viz tests.

The CI workflow already starts ``Xvfb :99`` explicitly before pytest and
exports ``DISPLAY=:99`` (see ``.github/workflows/tests.yml``). This
session-scoped autouse fixture is an additional safety net:

- On Linux, it calls ``pyvista.start_xvfb()`` so a developer or a CI
  variant that hasn't pre-started Xvfb still gets a usable display
  before VTK's ``vtkRenderingOpenGL2`` module probes GLX (which
  SIGABRTs the process if the X socket isn't reachable).
- ``pyvista.start_xvfb()`` is a no-op-equivalent when Xvfb is already
  running on display :99: the second Xvfb invocation silently fails to
  bind, the existing display continues to serve, and pyvista sets
  ``DISPLAY=:99`` (already set by CI).
- On macOS and Windows the fixture is skipped — those platforms render
  through native frameworks and don't need an X server.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_xvfb_for_vtk():
    """Make sure VTK has an X display before any pyvista test imports it."""
    if not sys.platform.startswith("linux"):
        yield
        return
    try:
        import pyvista
    except ImportError:
        yield
        return
    try:
        pyvista.start_xvfb()
    except Exception:
        # If Xvfb isn't installed or the helper raises for any reason,
        # don't fail the whole session — the underlying tests will report
        # their own failures and a developer can install xvfb locally.
        pass
    yield

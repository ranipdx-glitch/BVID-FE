"""Belt-and-suspenders Xvfb startup for the pyvista-backed viz tests.

The CI workflow already starts ``Xvfb :99`` explicitly before pytest and
exports ``DISPLAY=:99`` (see ``.github/workflows/tests.yml``). This
session-scoped autouse fixture is an additional safety net for local
developers (and any CI variant) that haven't pre-started Xvfb:

- If ``DISPLAY`` is already set (CI's normal path), do nothing — the
  pre-existing X server is what we want VTK to talk to. Calling
  ``pyvista.start_xvfb()`` on top of an already-running Xvfb on :99
  was observed to race and SIGABRT the python process on
  ubuntu-latest 3.12, presumably because pyvista's helper spawns a
  second Xvfb that fights the first one for the display socket.
- If ``DISPLAY`` is unset and we're on Linux, ask pyvista to start
  Xvfb so VTK's ``vtkRenderingOpenGL2`` module load doesn't abort.
- On macOS and Windows, do nothing — those platforms render through
  native frameworks and don't need an X server.
"""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_xvfb_for_vtk():
    """Make sure VTK has an X display before any pyvista test imports it."""
    if not sys.platform.startswith("linux"):
        yield
        return
    if os.environ.get("DISPLAY"):
        # CI workflow (or a developer's existing session) already provides
        # an X display — don't fight over it.
        yield
        return
    try:
        import pyvista

        pyvista.start_xvfb()
    except Exception:
        # If Xvfb isn't installed or the helper raises for any reason,
        # don't fail the whole session — the underlying tests will report
        # their own failures and a developer can install xvfb locally.
        pass
    yield

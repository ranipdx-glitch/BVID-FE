"""Skip the entire tests/gui/ tree if a Qt binding or pytest-qt is missing.

The base ``[dev]`` extra installs pytest + pytest-qt's siblings but
deliberately omits PyQt6 (which lives in the ``[gui]`` extra). Without a
Qt binding the GUI tests have no ``qtbot`` fixture, so we ignore the
whole directory at collection time rather than letting individual tests
ERROR. To exercise these tests install with ``pip install -e ".[gui]"``
or ``[all]``.
"""

from __future__ import annotations

collect_ignore_glob: list[str] = []

try:
    import PyQt6  # noqa: F401
    import pytestqt  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]

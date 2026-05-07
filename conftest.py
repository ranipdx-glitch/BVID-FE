"""Top-level pytest config: ensure GUI tests run headlessly.

Setting ``QT_QPA_PLATFORM=offscreen`` before any PyQt import makes the
GUI test suite (``tests/gui/``) work on CI runners that have no display
server. Individual GUI test modules still set this defensively, but
keeping it here means new GUI tests don't have to remember to.

``setdefault`` is intentional: a developer running locally with a real
display ``QT_QPA_PLATFORM=xcb pytest tests/gui/`` should still get a
visible window for debugging.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

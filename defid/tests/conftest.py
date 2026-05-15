"""Fix sys.path so the installed defid src-layout package takes precedence
over the workspace-root namespace package that pytest creates."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_workspace_root = str(Path(__file__).parents[2])  # /Users/stuartwells/test


def _fix_defid_import() -> None:
    # Move workspace root to after all /src entries.
    if _workspace_root in sys.path:
        sys.path.remove(_workspace_root)
        last_src = max(
            (i for i, p in enumerate(sys.path) if p.endswith("/src")),
            default=len(sys.path) - 1,
        )
        sys.path.insert(last_src + 1, _workspace_root)
    # Evict only the top-level 'defid' namespace package (not sub-modules currently loading).
    if "defid" in sys.modules:
        mod = sys.modules["defid"]
        if getattr(mod, "__spec__", None) is not None and getattr(mod.__spec__, "origin", ...) is None:
            # It's a namespace package — safe to evict
            del sys.modules["defid"]
    importlib.invalidate_caches()


_fix_defid_import()

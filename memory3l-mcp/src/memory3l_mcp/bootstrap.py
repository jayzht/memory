"""
Make the ``memory3l`` audit core importable, or say exactly how to fix it.

The MCP server is a distribution wrapper: it owns protocol, packaging and
storage-location policy, and delegates every audit answer to ``memory3l``.  That
split is deliberate -- the invariants are the product and must have one
implementation, so this package never re-implements them.

Why resolution is explicit rather than a plain ``import``
---------------------------------------------------------
``memory3l`` is not on PyPI yet.  A wrapper that just did ``import memory3l``
would fail with a bare ``ModuleNotFoundError`` for every user who ran
``uvx memory3l-mcp`` from an arbitrary directory, which is the opposite of
plug-and-play.  So the lookup is ordered and the failure mode is a sentence
naming the one thing to set:

1. already importable (installed normally, or PYTHONPATH covers it);
2. ``$MEMORY3L_ROOT`` -- the explicit answer, and the one the error recommends;
3. walking up from the current directory (running inside the checkout);
4. walking up from this file (an installed copy adjacent to the source tree).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["ensure_memory3l", "memory3l_root", "MemoryCoreNotFound"]

_MARKER = Path("memory3l") / "audit.py"


class MemoryCoreNotFound(RuntimeError):
    """The ``memory3l`` audit core could not be located on this machine."""


def _looks_like_root(path: Path) -> bool:
    return (path / _MARKER).is_file()


def _ancestors(start: Path):
    """Yield ``start`` and every parent, nearest first."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        yield candidate


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("MEMORY3L_ROOT")
    if env:
        roots.append(Path(env).expanduser())
    roots.extend(_ancestors(Path.cwd()))
    # .../memory3l-mcp/src/memory3l_mcp/bootstrap.py -> repo root
    roots.extend(_ancestors(Path(__file__).parent))
    return roots


def memory3l_root() -> Path | None:
    """The first directory that actually contains ``memory3l/audit.py``."""
    for candidate in _candidate_roots():
        if _looks_like_root(candidate):
            return candidate
    return None


def ensure_memory3l() -> Path:
    """
    Import ``memory3l``, adding a discovered root to ``sys.path`` if needed.

    Returns the root that was used, so the caller can report it.  Raises
    :class:`MemoryCoreNotFound` with an actionable message when nothing matched.
    """
    try:
        import memory3l  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return Path(memory3l.__file__).resolve().parent.parent

    root = memory3l_root()
    if root is None:
        raise MemoryCoreNotFound(
            "memory3l-mcp: cannot find the `memory3l` audit core (looked for "
            "memory3l/audit.py in $MEMORY3L_ROOT, the current directory and its "
            "parents, and next to this file).\n"
            "Install it from the source checkout, or point at the checkout:\n"
            "    export MEMORY3L_ROOT=/path/to/Memory"
        )
    sys.path.insert(0, str(root))
    try:
        import memory3l  # noqa: F401
    except ModuleNotFoundError as error:  # pragma: no cover - defensive
        raise MemoryCoreNotFound(
            f"memory3l-mcp: found {root} but importing memory3l from it failed: {error}"
        ) from error
    return root

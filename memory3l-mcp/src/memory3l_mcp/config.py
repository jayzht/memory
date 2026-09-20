"""
Where the audited ledger lives, resolved so a fresh machine just works.

Plug-and-play fails on state, not on protocol.  A server that requires the user
to already know a database path produces "it installed but there is nothing in
it", which reads as broken.  So the order is: explicit argument, then
environment, then a per-user default that is *created on demand*.

Nothing here requires the file to exist.  The cold store creates the directory,
the file and the schema on open, so serving an empty ledger is a normal state
that answers ``episodes: []`` rather than a crash.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DEFAULT_DIR_ENV", "DB_ENV", "resolve_db_path", "is_explicit"]

DEFAULT_DIR_ENV = "MEMORY3L_HOME"
DB_ENV = "MEMORY3L_DB"
DEFAULT_DIRNAME = ".memory3l"
DEFAULT_FILENAME = "memory3l.db"


def _default_dir() -> Path:
    override = os.environ.get(DEFAULT_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_DIRNAME


def resolve_db_path(explicit: str | None = None) -> Path:
    """
    Resolve the ledger path: argument, then ``$MEMORY3L_DB``, then the default.

    A directory component is created if missing.  The file itself is left to the
    cold store, which creates it together with the schema.
    """
    if explicit:
        path = Path(explicit).expanduser()
    elif os.environ.get(DB_ENV):
        path = Path(os.environ[DB_ENV]).expanduser()
    else:
        path = _default_dir() / DEFAULT_FILENAME
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_explicit() -> bool:
    """Whether the location was chosen by the user rather than defaulted."""
    return bool(os.environ.get(DB_ENV) or os.environ.get(DEFAULT_DIR_ENV))

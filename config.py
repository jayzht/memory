"""
Backwards-compatible alias for :mod:`memory3l.config`.

The configuration lives inside the package now, so that an installed ``memory3l``
is self-contained: it used to sit at the project root, which meant a wheel could
not import ``MemoryManager`` at all (``import config`` found nothing).

Why an alias and not a re-export
--------------------------------
The root scripts (``evaluation.py``, ``gate_sweep.py``, ``web_ui.py``,
``chat_debug.py``) do ``import config`` **and mutate it** -- CLI flags write back
into the module.  ``from memory3l.config import *`` would create a *second* module
object: the script would set an attribute on the copy while the package read the
original, and the flag would silently have no effect.  Rebinding ``sys.modules``
makes ``import config`` and ``import memory3l.config`` the same object, so a write
through either name is visible through both.

Package modules import it relatively (``from . import config``), so they do not
depend on this file being importable: outside a source checkout it is simply absent.
"""

from __future__ import annotations

import sys as _sys

from memory3l import config as _config

_sys.modules[__name__] = _config

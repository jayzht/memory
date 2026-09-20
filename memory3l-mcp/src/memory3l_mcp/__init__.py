"""
memory3l over MCP: auditable-memory checks as model-callable tools.

The audit core lives in the ``memory3l`` package; this distribution owns the
protocol surface, packaging and storage-location policy.
"""

from __future__ import annotations

__all__ = ["__version__", "main"]

__version__ = "0.1.0"


def main(argv=None) -> int:
    """Entry point, re-exported so ``python -m memory3l_mcp`` and the console
    script share one implementation."""
    from .server import main as _main

    return _main(argv)

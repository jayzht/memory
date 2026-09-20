"""
Install the bundled Agent Skills bundle where clients actually look for skills.

Why this is a command and not just a file in the repo
-----------------------------------------------------
The tools teach an agent *what* it can ask; the skill teaches it *when* to ask and
how not to overstate the answer.  Shipping the skill inside the wheel and stopping
there would leave every user to discover the cross-client directory themselves, so
the last step is one command.

The destination is the cross-client convention (``~/.agents/skills/``), which the
`Agent Skills`_ specification documents and DSH, Claude Code and others scan.  A
skill installed there is visible to every compliant client at once, with no
per-client adaptation.  ``DSH_AGENTS_HOME`` is honoured when set, because DSH
resolves its user skills root from it.

.. _Agent Skills: https://agentskills.io/specification
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

__all__ = ["BUNDLED_SKILL_DIR", "DEFAULT_SKILL", "install_skill", "main"]

BUNDLED_SKILL_DIR = Path(__file__).resolve().parent / "skills"
DEFAULT_SKILL = "memory3l"


def _skills_root() -> Path:
    """The cross-client user skills directory."""
    home = os.environ.get("DSH_AGENTS_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".agents"
    return base / "skills"


def _differs(left: Path, right: Path) -> bool:
    """Whether two skill trees differ, by content rather than by timestamp."""
    if not right.is_dir():
        return True
    comparison = filecmp.dircmp(str(left), str(right))
    if comparison.left_only or comparison.right_only or comparison.diff_files:
        return True
    return any(_differs(Path(left, d), Path(right, d)) for d in comparison.common_dirs)


def install_skill(
    target_root: str | os.PathLike[str] | None = None,
    name: str = DEFAULT_SKILL,
    force: bool = False,
) -> tuple[Path, bool]:
    """
    Copy the bundled skill into the client skills directory.

    Returns ``(destination, written)``.  An existing skill is never clobbered
    silently: identical content is a no-op, and different content requires
    ``force``.  Overwriting a file the user may have edited is a worse failure than
    making them pass a flag.
    """
    source = BUNDLED_SKILL_DIR / name
    if not source.is_dir():
        raise FileNotFoundError(
            f"memory3l-mcp: the bundled skill {name!r} is missing from this install "
            f"(looked in {BUNDLED_SKILL_DIR}). A wheel built without the skills/ "
            "directory is broken, not empty."
        )

    root = Path(target_root).expanduser() if target_root else _skills_root()
    destination = root / name

    if destination.exists() and not force:
        if not _differs(source, destination):
            return destination, False
        raise FileExistsError(
            f"{destination} already exists with different content. "
            "Re-run with --force to overwrite it (your edits would be lost)."
        )

    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination, True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="memory3l-mcp-install-skill",
        description=(
            "Install the memory3l Agent Skill into the cross-client skills "
            "directory so every compliant agent can see it."
        ),
    )
    parser.add_argument(
        "--target",
        metavar="DIR",
        help="skills root to install into (default: $DSH_AGENTS_HOME/skills or ~/.agents/skills)",
    )
    parser.add_argument("--name", default=DEFAULT_SKILL, help="skill directory name")
    parser.add_argument("--force", action="store_true", help="overwrite an existing, differing skill")
    args = parser.parse_args(argv)

    try:
        destination, written = install_skill(args.target, args.name, args.force)
    except (FileNotFoundError, FileExistsError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"{'installed' if written else 'already up to date'}: {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the overlay script from repo root; forwards all CLI args."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "docs" / "Visualization" / "Overlay" / "overlay_robot_on_person_cv.py"


def main() -> None:
    if not _SCRIPT.is_file():
        print(f"Error: expected script at {_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    raise SystemExit(subprocess.call([sys.executable, str(_SCRIPT), *sys.argv[1:]]))


if __name__ == "__main__":
    main()

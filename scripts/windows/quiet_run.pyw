"""Run an HKRL console command from pythonw without creating a console window."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a child command is required")
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("ab") as log:
        try:
            return subprocess.run(
                args.command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            ).returncode
        except OSError as exc:
            log.write(f"{exc}\n".encode("utf-8", "replace"))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())

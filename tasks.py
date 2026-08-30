#!/usr/bin/env python3
"""Cross-platform task runner — the Makefile's equivalent for Windows.

    python tasks.py test
    python tasks.py env-demo

Exists because `make` is not present on a default Windows install, and the
"clone it and run it" promise in the README has to hold on every platform a
reviewer might be using. The Makefile stays for Linux and CI; both delegate to
the same underlying commands so they cannot drift apart.
"""

from __future__ import annotations

import subprocess
import sys

PY = sys.executable

TASKS: dict[str, list[list[str]]] = {
    "install": [[PY, "-m", "pip", "install", "-e", "."]],
    "dev": [[PY, "-m", "pip", "install", "-e", ".[dev,dashboard]"]],
    "test": [[PY, "-m", "pytest"]],
    "lint": [[PY, "-m", "ruff", "check", "src", "tests"]],
    "fmt": [
        [PY, "-m", "ruff", "format", "src", "tests"],
        [PY, "-m", "ruff", "check", "--fix", "src", "tests"],
    ],
    "env-demo": [[PY, "-m", "hrapay.env.demo", "--episodes", "5"]],
    "calibrate": [[PY, "-m", "hrapay.rewards.calibrator"]],
    "calibrate-dry": [[PY, "-m", "hrapay.rewards.calibrator", "--dry-run"]],
    "train": [
        [PY, "-m", "hrapay.train", "--agent", "flat", "--steps", "60000"],
        [PY, "-m", "hrapay.train", "--agent", "bdq", "--steps", "60000"],
    ],
    "train-flat": [[PY, "-m", "hrapay.train", "--agent", "flat", "--steps", "60000"]],
    "train-bdq": [[PY, "-m", "hrapay.train", "--agent", "bdq", "--steps", "60000"]],
    "eval": [[PY, "-m", "hrapay.eval.cli", "--episodes", "1000", "--seeds", "3"]],
    "priors": [[PY, "-m", "hrapay.env.demo", "--episodes", "1", "--refresh-priors"]],
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(__doc__)
        print("Available tasks:")
        for name in TASKS:
            print(f"  {name}")
        return 0

    name = sys.argv[1]
    if name not in TASKS:
        print(f"Unknown task: {name}\nAvailable: {', '.join(TASKS)}", file=sys.stderr)
        return 2

    for cmd in TASKS[name]:
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd)  # noqa: S603
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

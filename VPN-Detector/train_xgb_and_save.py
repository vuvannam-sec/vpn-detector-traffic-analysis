"""Compatibility entrypoint for the original project command.

New usage should prefer:

    python train.py --window 15
"""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "train.py",
                "--window",
                "15",
                "--output",
                "models/xgb_15s.joblib",
            ]
        )
    )

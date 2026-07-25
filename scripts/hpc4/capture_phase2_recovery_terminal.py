#!/usr/bin/env python3
"""Capture terminal sacct rows for recovery 1648125 in its exact project namespace.

The frozen supplementary live-scontrol receipt must already verify.  Its
RUNNING/RUNNING/PENDING states are submission/live evidence only; this command
still requires all three independent sacct task rows to be COMPLETED/0:0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart_reward.phase2_recovery_aggregate import (
    capture_phase2_recovery_scheduler_evidence_with_digest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        help=(
            "exact new /project/sigroup/smart-reward-model/runs/"
            "phase2-recovery-pilot/recovery-1648125-terminal.json path"
        ),
    )
    arguments = parser.parse_args()
    _, digest = capture_phase2_recovery_scheduler_evidence_with_digest(arguments.output)
    print(
        json.dumps(
            {
                "status": "captured",
                "output": str(arguments.output),
                "sha256": digest,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

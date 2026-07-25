#!/usr/bin/env python3
"""Capture or verify terminal Slurm evidence for post-recovery pilot tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from smart_reward.phase2_post_recovery_control import (
    capture_post_recovery_terminal_evidence,
    verify_post_recovery_terminal_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("array_job_id")
    capture.add_argument("output", type=Path)
    capture.add_argument("--pilot-phase", choices=("calibration", "freeze"), required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("evidence", type=Path)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--array-job-id", required=True)
    verify.add_argument("--pilot-phase", choices=("calibration", "freeze"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "capture":
        payload = capture_post_recovery_terminal_evidence(
            arguments.array_job_id,
            arguments.output,
            pilot_phase=arguments.pilot_phase,
        )
        digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
        result = {
            "status": "captured",
            "array_job_id": payload["array_job_id"],
            "pilot_phase": payload["pilot_phase"],
            "output": str(arguments.output),
            "sha256": digest,
        }
    else:
        payload = verify_post_recovery_terminal_evidence(
            arguments.evidence,
            expected_sha256=arguments.expected_sha256,
            expected_array_job_id=arguments.array_job_id,
            expected_pilot_phase=arguments.pilot_phase,
        )
        result = {
            "status": "verified",
            "array_job_id": payload["array_job_id"],
            "pilot_phase": payload["pilot_phase"],
            "sha256": arguments.expected_sha256,
        }
    print(
        json.dumps(
            result,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

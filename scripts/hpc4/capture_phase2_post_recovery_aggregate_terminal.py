#!/usr/bin/env python3
"""Terminalize one durably published post-recovery gpu-l20 aggregate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from smart_reward.phase2_post_recovery_control import (
    capture_post_recovery_aggregate_terminal_evidence,
    verify_post_recovery_aggregate_success_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("aggregate", type=Path)
    capture.add_argument("--attempt-job-id", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("aggregate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "capture":
        result = capture_post_recovery_aggregate_terminal_evidence(
            arguments.aggregate,
            attempt_job_id=arguments.attempt_job_id,
        )
        status = "captured"
    else:
        result = verify_post_recovery_aggregate_success_receipt(arguments.aggregate)
        status = "verified"
    print(
        json.dumps(
            {
                "status": status,
                "aggregate": str(arguments.aggregate),
                "aggregate_sha256": result["aggregate_sha256"],
                "publication_receipt_sha256": result["publication_receipt_sha256"],
                "terminal_evidence_sha256": result["terminal_evidence_sha256"],
                "success_receipt_sha256": result["receipt_sha256"],
                "aggregation_slurm_job_id": result["aggregation_slurm_job_id"],
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

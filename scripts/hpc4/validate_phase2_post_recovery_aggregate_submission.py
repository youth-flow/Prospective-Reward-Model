#!/usr/bin/env python3
"""Validate the running gpu-l20 aggregate against its held submission ledger."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from submit_phase2_post_recovery_aggregate_attempt import (
    verify_aggregate_submission_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--intent-sha256", required=True)
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-export-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = verify_aggregate_submission_registry(
        arguments.registry,
        expected_intent_sha256=arguments.intent_sha256,
        expected_attempt_index=arguments.attempt_index,
        expected_job_id=arguments.job_id,
        expected_project_root=arguments.project_root,
        expected_repository_root=arguments.repo_root,
        expected_output=arguments.output,
        expected_workload_export_sha256=arguments.workload_export_sha256,
    )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
                "intent_path": os.fspath(result["intent_path"]),
                "intent_sha256": result["intent_sha256"],
                "attempt_path": os.fspath(result["attempt_path"]),
                "attempt_sha256": result["attempt_sha256"],
                "attempt_index": result["attempt_index"],
                "slurm_job_id": result["slurm_job_id"],
                "failure_entries": result["failure_entries"],
                "failure_chain": result["failure_chain_raw"].decode("utf-8"),
                "failure_chain_sha256": result["failure_chain_sha256"],
                "script_path": os.fspath(result["script_path"]),
                "script_sha256": result["script_sha256"],
                "script_size_bytes": result["script_size_bytes"],
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

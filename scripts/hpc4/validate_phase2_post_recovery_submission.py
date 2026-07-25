#!/usr/bin/env python3
"""Verify that a post-recovery task belongs to the one registered array."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from submit_phase2_post_recovery_array_once import verify_submission_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pilot-phase", choices=("calibration", "freeze"), required=True)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--base-config-hash", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--optimizer-schedule-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--overlay-sha256", required=True)
    parser.add_argument("--base-file-sha256", required=True)
    parser.add_argument("--export-spec-sha256", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--submitter-user", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = verify_submission_registry(
        arguments.registry,
        project_root=arguments.project_root,
        repo_root=arguments.repo_root,
        pilot_phase=arguments.pilot_phase,
        design_sha256=arguments.design_sha256,
        base_config_hash=arguments.base_config_hash,
        authorization_sha256=arguments.authorization_sha256,
        optimizer_schedule_sha256=arguments.optimizer_schedule_sha256,
        git_commit=arguments.git_commit,
        image_sha256=arguments.image_sha256,
        inventory_sha256=arguments.inventory_sha256,
        overlay_sha256=arguments.overlay_sha256,
        base_file_sha256=arguments.base_file_sha256,
        export_spec_sha256=arguments.export_spec_sha256,
        array_job_id=arguments.array_job_id,
        submitter_user=arguments.submitter_user,
    )
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

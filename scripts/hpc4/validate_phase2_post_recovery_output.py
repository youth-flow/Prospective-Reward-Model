#!/usr/bin/env python3
"""Write strict per-seed post-recovery calibration verification receipts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from smart_reward.phase2_post_recovery_output import (
    verify_and_write_post_recovery_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("authorization", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("phase2_output_verification", type=Path)
    parser.add_argument("post_recovery_output_verification", type=Path)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--base-config-hash", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--hf-inventory-sha256", required=True)
    parser.add_argument("--artifact-metadata-sha256", required=True)
    parser.add_argument("--slurm-job-id-raw", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--array-task-id", required=True, type=int)
    parser.add_argument("--pilot-phase", choices=("calibration", "freeze"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = verify_and_write_post_recovery_output(
        overlay_path=arguments.overlay,
        authorization_path=arguments.authorization,
        authorization_sha256=arguments.authorization_sha256,
        result_path=arguments.result,
        diagnostics_path=arguments.diagnostics,
        phase2_output_verification_path=arguments.phase2_output_verification,
        post_recovery_output_verification_path=(arguments.post_recovery_output_verification),
        seed=arguments.seed,
        expected_design_sha256=arguments.design_sha256,
        expected_base_config_hash=arguments.base_config_hash,
        expected_git_commit=arguments.git_commit,
        expected_image_sha256=arguments.image_sha256,
        expected_hf_inventory_sha256=arguments.hf_inventory_sha256,
        expected_artifact_metadata_sha256=arguments.artifact_metadata_sha256,
        expected_slurm_job_id_raw=arguments.slurm_job_id_raw,
        expected_array_job_id=arguments.array_job_id,
        expected_array_task_id=arguments.array_task_id,
        expected_pilot_phase=arguments.pilot_phase,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "seed": result["seed"],
                "result_sha256": result["result_sha256"],
                "five_head_adopted_schedule_verified": result[
                    "five_head_adopted_schedule_verified"
                ],
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate a real recovery authorization and its post-recovery overlay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from smart_reward.phase2_post_recovery_control import (
    verify_recovery_authorization_config_binding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("authorization", type=Path)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--expected-stage",
        choices=("pilot", "budgeted_end_to_end", "confirmatory"),
        default="pilot",
    )
    parser.add_argument(
        "--legacy-r2-replay",
        action="store_true",
        help="validate a historical R2 replay instead of active Gate-F",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    verification_options: dict[str, object] = {
        "expected_sha256": arguments.expected_sha256,
        "expected_stage": arguments.expected_stage,
        "require_r3_gate_f": not arguments.legacy_r2_replay,
    }
    if arguments.project_root is not None:
        verification_options["project_root"] = arguments.project_root
    binding = verify_recovery_authorization_config_binding(
        arguments.authorization,
        arguments.overlay,
        **verification_options,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "authorization_sha256": binding["authorization_sha256"],
                "phase2_design_sha256": binding["phase2_design_sha256"],
                "base_config_hash": binding["base_config_hash"],
                "optimizer_schedule_sha256": binding["optimizer_schedule_sha256"],
                "authorized_next_action": binding["authorization"]["authorized_next_action"],
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

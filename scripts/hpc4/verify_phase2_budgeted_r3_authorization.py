#!/usr/bin/env python3
"""Deeply verify the sole R3 authorization admissible to GateE."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from smart_reward.phase2_config import (
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    load_phase2_config_bundle,
    validate_post_recovery_authorization_reference,
)
from smart_reward.phase2_r3_post_recovery_authorization import (
    verify_r3_final_authorization,
)
from smart_reward.phase2_r3_post_recovery_contract import (
    R3_AUTHORIZED_NEXT_ACTION,
    R3_FINAL_AUTHORIZATION_RELATIVE,
    R3_FINAL_AUTHORIZATION_ROLE,
    R3_FINAL_AUTHORIZATION_SCHEMA,
    R3_OPTIMIZER_SCHEDULE_SHA256,
    R3_PRODUCTION_PROJECT_ROOT,
)
from smart_reward.phase2_training import compile_phase2_training_settings


def verify_budgeted_r3_authorization(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    project_root: str | os.PathLike[str] = R3_PRODUCTION_PROJECT_ROOT,
) -> dict[str, object]:
    root = Path(project_root)
    source = Path(path)
    if not source.is_absolute():
        source = source.absolute()
    if source != root / R3_FINAL_AUTHORIZATION_RELATIVE:
        raise ValueError("GateE requires the exact fixed R3 combined authorization path")
    authorization = verify_r3_final_authorization(
        source,
        expected_sha256=expected_sha256,
        project_root=root,
    )
    if (
        authorization.get("schema_version") != R3_FINAL_AUTHORIZATION_SCHEMA
        or authorization.get("role") != R3_FINAL_AUTHORIZATION_ROLE
        or authorization.get("optimizer_schedule_sha256") != R3_OPTIMIZER_SCHEDULE_SHA256
        or authorization.get("gate_r_passed") is not True
        or authorization.get("gate_c_passed") is not True
        or authorization.get("fresh_calibration_authorized") is not True
        or authorization.get("authorized_next_action") != R3_AUTHORIZED_NEXT_ACTION
        or authorization.get("recovery_or_control_outputs_reusable") is not False
        or authorization.get("validation_or_heldout_access_authorized") is not False
        or authorization.get("policy_or_final_utility_access_authorized") is not False
        or authorization.get("formal_efficacy_claim_authorized") is not False
    ):
        raise ValueError("R3 combined authorization exceeds or misses the GateE boundary")
    transport = authorization.get("transport_boundary")
    if (
        not isinstance(transport, dict)
        or not transport
        or any(value is not False for value in transport.values())
    ):
        raise ValueError("R3 combined authorization transport boundary is open")
    return authorization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("authorization", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--project-root", type=Path, default=R3_PRODUCTION_PROJECT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    authorization = verify_budgeted_r3_authorization(
        arguments.authorization,
        expected_sha256=arguments.expected_sha256,
        project_root=arguments.project_root,
    )
    bundle = load_phase2_config_bundle(arguments.config)
    validate_post_recovery_authorization_reference(
        bundle.config["recovery_success_reference"],
        authorization_payload_sha256=arguments.expected_sha256,
        authorization_payload=authorization,
    )
    settings = compile_phase2_training_settings(
        {"config": bundle.config, "base_config": bundle.base_config}
    )
    protocol = settings.convergence.optimizer_protocol
    if (
        settings.stage != PHASE2_BUDGETED_END_TO_END_STAGE
        or settings.formal_eligibility is not False
        or settings.seeds != PHASE2_BUDGETED_END_TO_END_SEEDS
        or protocol is None
        or protocol.mode != "adopted"
        or protocol.schedule_sha256 != R3_OPTIMIZER_SCHEDULE_SHA256
        or protocol.source_recovery_authorization_sha256 != arguments.expected_sha256
    ):
        raise ValueError("budgeted config is not bound to the verified R3 authorization")
    print(
        json.dumps(
            {
                "schema_version": "phase2-budgeted-r3-authorization-check/v1",
                "status": "passed",
                "authorization_sha256": arguments.expected_sha256,
                "authorization_semantic_sha256": authorization["authorization_sha256"],
                "recovery_design_sha256": authorization["recovery_design_sha256"],
                "optimizer_schedule_sha256": authorization["optimizer_schedule_sha256"],
                "gate_r_passed": True,
                "gate_c_passed": True,
                "fresh_calibration_authorized": True,
                "trained_outputs_reused": False,
                "phase2_design_sha256": bundle.design_identity,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

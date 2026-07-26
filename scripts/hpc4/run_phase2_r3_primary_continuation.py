#!/usr/bin/env python3
"""Run one R3 continuation segment derived only from sealed predecessor history."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prepare_phase2_r3_continuation_submission import reopen_continuation_plan

from smart_reward.phase2_checkpoint import CheckpointSignal
from smart_reward.phase2_r3_config import load_r3_science_config
from smart_reward.phase2_r3_gate0 import verify_live_r3_gate0_in_container
from smart_reward.phase2_r3_gate1 import verify_live_r3_gate1_in_container
from smart_reward.phase2_r3_identity import (
    admit_primary_segment,
    authorize_gate_p,
    create_r3_primary_design,
    rehydrate_primary_segment_admission,
    validate_continuation_evidence,
)
from smart_reward.phase2_r3_inputs import materialize_r3_train_only_from_parent
from smart_reward.phase2_r3_orchestrator import run_r3_primary_task_segment
from smart_reward.phase2_r3_primary import capture_slurm_segment_runtime
from smart_reward.phase2_r3_profile_artifacts import (
    reopen_verified_gate_p_operational_bundle,
)
from smart_reward.phase2_r3_terminal import (
    publish_primary_segment_runtime_closure,
    reopen_primary_segment_runtime_closure,
    reopen_profile_allocation_intent,
    reopen_profile_slurm_runtime_receipt,
    revalidate_continuable_primary_terminal,
    revalidate_successful_profile_terminal,
)

_TASK_SEED_MAP = {0: 20260801, 1: 20260802, 2: 20260803}
_DIGEST_LENGTH = 64


def _digest(value: str, *, name: str) -> str:
    if (
        len(value) != _DIGEST_LENGTH
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _task_seed_from_environment() -> tuple[int, int]:
    text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if text not in {"0", "1", "2"}:
        raise RuntimeError("formal continuation requires SLURM_ARRAY_TASK_ID in {0,1,2}")
    task_id = int(text)
    return task_id, _TASK_SEED_MAP[task_id]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--science-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--parent-registry-file-sha256", required=True)
    parser.add_argument("--gate0-file-sha256", required=True)
    parser.add_argument("--gate1-file-sha256", required=True)
    parser.add_argument("--source-test-receipt-file-sha256", required=True)
    parser.add_argument("--operational-bundle", type=Path, required=True)
    parser.add_argument("--operational-bundle-file-sha256", required=True)
    parser.add_argument("--profile-allocation-intent", type=Path, required=True)
    parser.add_argument("--profile-allocation-intent-file-sha256", required=True)
    parser.add_argument("--profile-runtime-receipt", type=Path, required=True)
    parser.add_argument("--profile-runtime-receipt-file-sha256", required=True)
    parser.add_argument(
        "--profile-terminal-evidence-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--profile-terminal-manifest-file-sha256", required=True)
    parser.add_argument("--profile-terminal-raw-sacct-sha256", required=True)
    parser.add_argument("--continuation-plan", type=Path, required=True)
    parser.add_argument("--continuation-plan-file-sha256", required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--runtime-closure", type=Path, required=True)
    return parser


def _emit(value: dict[str, object]) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    task_id, seed = _task_seed_from_environment()
    plan = reopen_continuation_plan(
        arguments.continuation_plan,
        expected_file_sha256=_digest(
            arguments.continuation_plan_file_sha256,
            name="continuation plan file SHA-256",
        ),
    )
    route = plan["task_routes"][task_id]
    if route["action"] != "continue":
        raise RuntimeError("completed task must never enter the continuation runner")

    science = load_r3_science_config(arguments.science_config)
    gate0 = verify_live_r3_gate0_in_container(
        expected_file_sha256=_digest(
            arguments.gate0_file_sha256,
            name="Gate-0 file SHA-256",
        )
    )
    gate1 = verify_live_r3_gate1_in_container(
        expected_file_sha256=_digest(
            arguments.gate1_file_sha256,
            name="Gate-1 file SHA-256",
        ),
        expected_source_test_receipt_file_sha256=_digest(
            arguments.source_test_receipt_file_sha256,
            name="source-test receipt file SHA-256",
        ),
    )
    bundle = reopen_verified_gate_p_operational_bundle(
        arguments.operational_bundle,
        expected_file_sha256=_digest(
            arguments.operational_bundle_file_sha256,
            name="operational bundle file SHA-256",
        ),
    )
    if (
        plan["operational_bundle_file_sha256"] != bundle.file_sha256
        or plan["operational_bundle_semantic_sha256"] != bundle.bundle_semantic_sha256
        or plan["resource_plan_sha256"] != bundle.resource_plan_sha256
    ):
        raise ValueError("continuation plan belongs to another Gate-P bundle")
    profile_intent = reopen_profile_allocation_intent(
        arguments.profile_allocation_intent,
        expected_file_sha256=_digest(
            arguments.profile_allocation_intent_file_sha256,
            name="profile allocation intent file SHA-256",
        ),
    )
    profile_runtime = reopen_profile_slurm_runtime_receipt(
        arguments.profile_runtime_receipt,
        expected_file_sha256=_digest(
            arguments.profile_runtime_receipt_file_sha256,
            name="profile runtime receipt file SHA-256",
        ),
        operational_bundle=bundle,
        allocation_intent=profile_intent,
    )
    profile_terminal = revalidate_successful_profile_terminal(
        bundle,
        runtime_receipt=profile_runtime,
        evidence_directory=arguments.profile_terminal_evidence_directory,
        expected_manifest_file_sha256=_digest(
            arguments.profile_terminal_manifest_file_sha256,
            name="profile terminal manifest file SHA-256",
        ),
        expected_raw_sacct_sha256=_digest(
            arguments.profile_terminal_raw_sacct_sha256,
            name="profile terminal raw sacct SHA-256",
        ),
    )
    authorization = authorize_gate_p(
        operational_bundle=bundle,
        successful_terminal=profile_terminal,
    )
    design = create_r3_primary_design(
        science=science,
        gate0_capability=gate0,
        gate1_capabilities=gate1,
        profile_authorization=authorization,
        operational_bundle=bundle,
    )
    materialized = materialize_r3_train_only_from_parent(
        project_root=arguments.project_root,
        parent_registry_path=arguments.parent_registry,
        expected_parent_registry_file_sha256=_digest(
            arguments.parent_registry_file_sha256,
            name="parent registry file SHA-256",
        ),
        source_config_path=arguments.source_config,
        science_bundle=science,
        seed=seed,
        device="cuda",
    )

    continuation_evidence = None
    for expected_segment, entry in enumerate(route["history"], start=1):
        if entry["terminal_kind"] != "continuable":
            raise ValueError("continuation history cannot contain a completed predecessor")
        closure = reopen_primary_segment_runtime_closure(
            entry["runtime_closure_path"],
            expected_file_sha256=entry["runtime_closure_file_sha256"],
            operational_bundle=bundle,
        )
        if (
            closure.closure_sha256 != entry["runtime_closure_sha256"]
            or closure.admission_payload["segment_index"] != expected_segment
            or closure.admission_payload["task_id"] != task_id
            or closure.admission_payload["seed"] != seed
        ):
            raise ValueError("continuation history closure identity is invalid")
        predecessor = rehydrate_primary_segment_admission(
            closure.admission_payload,
            design=design,
            materialization_capability=materialized.capability,
            continuation_evidence=continuation_evidence,
        )
        terminal = revalidate_continuable_primary_terminal(
            bundle,
            runtime_closure=closure,
            evidence_directory=entry["terminal_evidence_directory"],
            expected_manifest_file_sha256=entry["terminal_manifest_file_sha256"],
            expected_raw_sacct_sha256=entry["terminal_raw_sacct_sha256"],
        )
        if (
            terminal.terminal_sha256 != entry["terminal_sha256"]
            or closure.outcome_payload["continuation_checkpoint"] != entry["selected_checkpoint"]
        ):
            raise ValueError("continuation history terminal/checkpoint identity is invalid")
        continuation_evidence = validate_continuation_evidence(
            predecessor=predecessor,
            continuable_terminal=terminal,
        )
    if continuation_evidence is None:
        raise RuntimeError("continuation route did not yield sealed predecessor evidence")
    admission = admit_primary_segment(
        design=design,
        materialization_capability=materialized.capability,
        task_id=task_id,
        seed=seed,
        segment_index=route["next_segment_index"],
        continuation_evidence=continuation_evidence,
    )
    runtime = capture_slurm_segment_runtime(
        admission,
        requested_walltime_seconds=bundle.requested_walltime_seconds_per_segment,
    )
    with CheckpointSignal() as checkpoint_signal:
        outcome = run_r3_primary_task_segment(
            admission,
            runtime=runtime,
            task_root=arguments.task_root,
            checkpoint_signal=checkpoint_signal,
            operational_policy=bundle,
        )
    closure = publish_primary_segment_runtime_closure(
        arguments.runtime_closure,
        admission=admission,
        runtime=runtime,
        outcome=outcome,
        operational_bundle=bundle,
    )
    _emit(
        {
            "status": "r3_primary_continuation_closed_pending_scheduler_terminal",
            "segment_outcome_status": outcome.status,
            "task_id": task_id,
            "seed": seed,
            "segment_index": admission.segment_index,
            "continuation_plan_sha256": plan["continuation_plan_sha256"],
            "continuation_evidence_sha256": continuation_evidence.evidence_sha256,
            "admission_sha256": admission.admission_sha256,
            "runtime_sha256": runtime.runtime_sha256,
            "segment_outcome_sha256": outcome.outcome_sha256,
            "runtime_closure_sha256": closure.closure_sha256,
            "runtime_closure_file_sha256": closure.file_sha256,
            "external_scheduler_terminal_required": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

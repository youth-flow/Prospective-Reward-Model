#!/usr/bin/env python3
"""Produce or verify the final three-seed R3 Gate-R authorization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from smart_reward.phase2_r3_authorization import (
    build_r3_success_authorization,
    publish_r3_success_authorization,
    verify_r3_success_authorization,
)
from smart_reward.phase2_r3_config import load_r3_science_config
from smart_reward.phase2_r3_execution_evidence import (
    identity_receipt_path,
    reopen_primary_identity_receipt,
    reopen_segment_evidence_receipt,
    segment_evidence_receipt_path,
)
from smart_reward.phase2_r3_gate0 import verify_live_r3_gate0_in_container
from smart_reward.phase2_r3_gate1 import verify_live_r3_gate1_in_container
from smart_reward.phase2_r3_identity import authorize_gate_p, create_r3_primary_design
from smart_reward.phase2_r3_profile_artifacts import (
    reopen_verified_gate_p_operational_bundle,
)
from smart_reward.phase2_r3_terminal import (
    CompletedPrimaryTerminalCapability,
    ContinuablePrimaryTerminalCapability,
    reopen_primary_segment_runtime_closure,
    reopen_profile_allocation_intent,
    reopen_profile_slurm_runtime_receipt,
    revalidate_completed_primary_terminal,
    revalidate_continuable_primary_terminal,
    revalidate_successful_profile_terminal,
)

_CONTINUATION_PLAN_SCRIPT = Path(__file__).with_name("prepare_phase2_r3_continuation_submission.py")
_SUBMISSION_LEDGER_SCRIPT = Path(__file__).with_name("submit_phase2_r3_once.py")
_PROJECT_ROOT = Path("/project/sigroup/smart-reward-model")


def _continuation_plan_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_phase2_r3_continuation_plan_for_authorization",
        _CONTINUATION_PLAN_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the R3 continuation-plan validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _submission_ledger_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_phase2_r3_submission_ledger_for_authorization",
        _SUBMISSION_LEDGER_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the R3 submission-ledger validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_submission_ledger_lineage(
    *,
    base: Mapping[str, object],
    lineage: tuple[dict[str, object], ...],
) -> None:
    ledger_module = _submission_ledger_module()
    observed_jobs: set[str] = set()
    base_ledger = ledger_module.reopen_submission_ledger(
        _PROJECT_ROOT,
        plan_semantic_sha256=base["submission_plan_sha256"],
    )
    base_intent = base_ledger["intent"]
    base_submission = base_ledger["submission"]
    expected_base_plan_relative = (
        Path(str(lineage[0]["base_primary_submission_plan_path"]))
        .relative_to(_PROJECT_ROOT)
        .as_posix()
    )
    if (
        base_intent["plan_kind"] != "primary"
        or base_intent["plan_path"] != expected_base_plan_relative
        or base_intent["plan_file_sha256"] != lineage[0]["base_primary_submission_plan_file_sha256"]
        or base_intent["array_task_ids"] != [0, 1, 2]
        or base_intent["dependency_array_job_ids"] != []
        or base_intent["git_commit"] != base["git_commit"]
        or base_intent["container_image_file_sha256"] != base["container_image_file_sha256"]
    ):
        raise ValueError("base primary submission ledger has an invalid task/dependency set")
    base_job = str(base_submission["array_job_id"])
    observed_jobs.add(base_job)
    first_routes = lineage[0]["task_routes"]
    if not isinstance(first_routes, list) or any(
        route["history"][0]["array_job_id"] != base_job for route in first_routes
    ):
        raise ValueError("segment-1 terminals do not belong to the ledgered primary array")

    for index, plan in enumerate(lineage[:-1]):
        if plan["continuation_array_required"] is not True:
            raise ValueError("nonfinal continuation plan did not authorize its successor wave")
        successor = lineage[index + 1]
        expected_plan = Path(str(successor["previous_continuation_plan_path"]))
        try:
            expected_plan_relative = expected_plan.relative_to(_PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(
                "continuation plan lineage escapes the retained project root"
            ) from error
        ledger = ledger_module.reopen_submission_ledger(
            _PROJECT_ROOT,
            plan_semantic_sha256=plan["continuation_plan_sha256"],
        )
        intent = ledger["intent"]
        submission = ledger["submission"]
        if (
            intent["plan_kind"] != "continuation"
            or intent["plan_path"] != expected_plan_relative
            or intent["plan_file_sha256"] != successor["previous_continuation_plan_file_sha256"]
            or intent["array_task_ids"] != plan["active_array_task_ids"]
            or intent["dependency_array_job_ids"] != plan["dependency_array_job_ids"]
            or intent["git_commit"] != base["git_commit"]
            or intent["container_image_file_sha256"] != base["container_image_file_sha256"]
        ):
            raise ValueError("continuation submission ledger differs from its sealed plan")
        job_id = str(submission["array_job_id"])
        if job_id in observed_jobs:
            raise ValueError("R3 submission lineage reuses one Slurm array allocation")
        observed_jobs.add(job_id)
        successor_routes = lineage[index + 1]["task_routes"]
        if not isinstance(successor_routes, list):
            raise TypeError("successor continuation routes lost their list type")
        for task_id in plan["active_array_task_ids"]:
            route = successor_routes[task_id]
            if route["history"][-1]["array_job_id"] != job_id:
                raise ValueError("continuation terminal does not belong to its ledgered array")
    final = lineage[-1]
    final_ledger = (
        _PROJECT_ROOT
        / "runs"
        / "phase2-recovery-r3"
        / "submission-ledgers"
        / str(final["continuation_plan_sha256"])
    )
    if (
        final["all_tasks_complete"] is not True
        or final_ledger.exists()
        or final_ledger.is_symlink()
    ):
        raise ValueError("completed continuation plan must never have a submission ledger")


def _completed_terminal_histories(
    plan_path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[
    tuple[ContinuablePrimaryTerminalCapability | CompletedPrimaryTerminalCapability, ...],
    ...,
]:
    module = _continuation_plan_module()
    lineage = module.reopen_continuation_plan_lineage(
        plan_path,
        expected_file_sha256=expected_file_sha256,
    )
    plan = lineage[-1]
    if plan["all_tasks_complete"] is not True or plan["continuation_array_required"] is not False:
        raise ValueError("R3 success authorization requires a terminal all-tasks-complete plan")
    base = module.reopen_base_primary_submission_plan(
        Path(plan["base_primary_submission_plan_path"]),
        expected_file_sha256=plan["base_primary_submission_plan_file_sha256"],
    )
    if (
        base["submission_plan_sha256"] != plan["base_primary_submission_plan_sha256"]
        or base["operational_bundle_file_sha256"] != plan["operational_bundle_file_sha256"]
    ):
        raise ValueError("completion lineage differs from its base primary plan")
    _validate_submission_ledger_lineage(base=base, lineage=lineage)
    observed_commit = subprocess.run(
        ["git", "-C", "/home/yyangjo/Smart-Reward-Model", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != base["git_commit"]:
        raise ValueError("live production Git commit differs from the base primary plan")
    bundle = reopen_verified_gate_p_operational_bundle(
        plan["operational_bundle_path"],
        expected_file_sha256=plan["operational_bundle_file_sha256"],
    )
    if (
        bundle.bundle_semantic_sha256 != plan["operational_bundle_semantic_sha256"]
        or bundle.resource_plan_sha256 != plan["resource_plan_sha256"]
    ):
        raise ValueError("R3 completion plan differs from its Gate-P operational bundle")
    science = load_r3_science_config(base["science_config_path"])
    if science.file_sha256 != base["science_config_file_sha256"]:
        raise ValueError("retained science config differs from the base primary plan")
    gate0 = verify_live_r3_gate0_in_container(
        expected_file_sha256=base["gate0_file_sha256"],
    )
    gate1 = verify_live_r3_gate1_in_container(
        expected_file_sha256=base["gate1_file_sha256"],
        expected_source_test_receipt_file_sha256=(base["source_test_receipt_file_sha256"]),
    )
    profile_intent = reopen_profile_allocation_intent(
        base["profile_allocation_intent_path"],
        expected_file_sha256=base["profile_allocation_intent_file_sha256"],
    )
    profile_runtime = reopen_profile_slurm_runtime_receipt(
        base["profile_runtime_receipt_path"],
        expected_file_sha256=base["profile_runtime_receipt_file_sha256"],
        operational_bundle=bundle,
        allocation_intent=profile_intent,
    )
    profile_terminal = revalidate_successful_profile_terminal(
        bundle,
        runtime_receipt=profile_runtime,
        evidence_directory=base["profile_terminal_evidence_directory"],
        expected_manifest_file_sha256=(base["profile_terminal_manifest_file_sha256"]),
        expected_raw_sacct_sha256=base["profile_terminal_raw_sacct_sha256"],
    )
    profile_authorization = authorize_gate_p(
        operational_bundle=bundle,
        successful_terminal=profile_terminal,
    )
    design = create_r3_primary_design(
        science=science,
        gate0_capability=gate0,
        gate1_capabilities=gate1,
        profile_authorization=profile_authorization,
        operational_bundle=bundle,
    )

    routes = plan["task_routes"]
    if not isinstance(routes, list) or len(routes) != 3:
        raise ValueError("R3 completion plan must contain exactly three task routes")
    histories: list[
        tuple[ContinuablePrimaryTerminalCapability | CompletedPrimaryTerminalCapability, ...]
    ] = []
    for task_id, raw_route in enumerate(routes):
        if not isinstance(raw_route, Mapping):
            raise TypeError("R3 completion-plan task route must be a mapping")
        route = dict(raw_route)
        if (
            route["task_id"] != task_id
            or route["action"] != "complete"
            or route["next_segment_index"] is not None
        ):
            raise ValueError("R3 completion-plan route is not terminally complete")
        raw_history = route["history"]
        if not isinstance(raw_history, list) or not raw_history:
            raise ValueError("R3 completion-plan route has no terminal history")
        history: list[
            ContinuablePrimaryTerminalCapability | CompletedPrimaryTerminalCapability
        ] = []
        identity_path = identity_receipt_path(
            _PROJECT_ROOT,
            task_id=task_id,
        )
        identity_file_sha256 = hashlib.sha256(identity_path.read_bytes()).hexdigest()
        identity = reopen_primary_identity_receipt(
            _PROJECT_ROOT,
            task_id=task_id,
            expected_file_sha256=identity_file_sha256,
            expected_design=design,
        )
        if (
            identity["base_primary_submission_plan"]["file_sha256"]  # type: ignore[index]
            != plan["base_primary_submission_plan_file_sha256"]
            or identity["base_primary_submission_plan"]["submission_plan_sha256"]  # type: ignore[index]
            != plan["base_primary_submission_plan_sha256"]
        ):
            raise ValueError("retained seed identity differs from the completion lineage")
        previous_manifest: list[dict[str, object]] | None = None
        for raw_entry in raw_history:
            if not isinstance(raw_entry, Mapping):
                raise TypeError("R3 completion-plan history entry must be a mapping")
            entry = dict(raw_entry)
            closure = reopen_primary_segment_runtime_closure(
                entry["runtime_closure_path"],
                expected_file_sha256=entry["runtime_closure_file_sha256"],
                operational_bundle=bundle,
            )
            common = {
                "runtime_closure": closure,
                "evidence_directory": entry["terminal_evidence_directory"],
                "expected_manifest_file_sha256": entry["terminal_manifest_file_sha256"],
                "expected_raw_sacct_sha256": entry["terminal_raw_sacct_sha256"],
            }
            capability: ContinuablePrimaryTerminalCapability | CompletedPrimaryTerminalCapability
            if entry["terminal_kind"] == "continuable":
                capability = revalidate_continuable_primary_terminal(bundle, **common)
            elif entry["terminal_kind"] == "completed":
                capability = revalidate_completed_primary_terminal(bundle, **common)
            else:
                raise ValueError("R3 completion-plan history has an unknown terminal kind")
            if (
                capability.terminal_sha256 != entry["terminal_sha256"]
                or closure.closure_sha256 != entry["runtime_closure_sha256"]
            ):
                raise ValueError("R3 completion-plan history differs from revalidated evidence")
            evidence_path = segment_evidence_receipt_path(
                _PROJECT_ROOT,
                task_id=task_id,
                segment_index=int(entry["segment_index"]),
            )
            evidence_file_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            evidence = reopen_segment_evidence_receipt(
                _PROJECT_ROOT,
                task_id=task_id,
                segment_index=int(entry["segment_index"]),
                expected_file_sha256=evidence_file_sha256,
                runtime_closure=closure,
                require_exact_current_manifest=(entry["terminal_kind"] == "completed"),
            )
            manifest = evidence["immutable_file_manifest"]
            if not isinstance(manifest, list):
                raise TypeError("validated segment evidence manifest lost its list type")
            if previous_manifest is not None:
                current_by_path = {
                    item["relative_path"]: item for item in manifest if isinstance(item, Mapping)
                }
                if any(
                    current_by_path.get(item["relative_path"]) != item for item in previous_manifest
                ):
                    raise ValueError("segment evidence manifest dropped or changed prior bytes")
            previous_manifest = manifest
            history.append(capability)
        histories.append(tuple(history))
    return tuple(histories)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    produce = commands.add_parser(
        "produce",
        help="publish the exact production authorization from a completed continuation plan",
    )
    produce.add_argument("--completion-plan", type=Path, required=True)
    produce.add_argument("--completion-plan-file-sha256", required=True)

    verify = commands.add_parser(
        "verify",
        help="revalidate an existing authorization and every referenced scheduler segment",
    )
    verify.add_argument("--authorization", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--completion-plan", type=Path, required=True)
    verify.add_argument("--completion-plan-file-sha256", required=True)
    return parser


def _emit(value: Mapping[str, object]) -> None:
    print(
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "produce":
        histories = _completed_terminal_histories(
            arguments.completion_plan,
            expected_file_sha256=arguments.completion_plan_file_sha256,
        )
        artifact = publish_r3_success_authorization(histories)
        payload = verify_r3_success_authorization(
            artifact.artifact_path,
            expected_sha256=artifact.file_sha256,
        )
        _emit(
            {
                "status": "r3_gate_r_three_seed_success_capability_published",
                "authorization_path": str(artifact.artifact_path),
                "authorization_file_sha256": artifact.file_sha256,
                "authorization_sha256": payload["authorization_sha256"],
                "terminal_set_sha256": payload["terminal_set_sha256"],
            }
        )
        return 0

    histories = _completed_terminal_histories(
        arguments.completion_plan,
        expected_file_sha256=arguments.completion_plan_file_sha256,
    )
    payload = verify_r3_success_authorization(
        arguments.authorization,
        expected_sha256=arguments.expected_sha256,
    )
    if build_r3_success_authorization(histories) != payload:
        raise ValueError(
            "R3 authorization differs from the recursively validated completion lineage"
        )
    _emit(
        {
            "status": "r3_gate_r_three_seed_success_capability_verified",
            "authorization_file_sha256": arguments.expected_sha256,
            "authorization_sha256": payload["authorization_sha256"],
            "terminal_set_sha256": payload["terminal_set_sha256"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

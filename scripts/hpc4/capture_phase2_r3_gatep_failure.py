#!/usr/bin/env python3
"""Publish and re-open append-only non-authorizing Gate-P failure lineage.

Raw scheduler bytes should first be captured with the existing locked helper:

``phase2_r3_terminalize_stdlib.py capture --job-selector ... --output ...
--user "$USER"``.

This CLI never submits work and never issues Gate-P or primary authority.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from smart_reward.phase2_r3_gatep_failure import (
    FAILURE_STAGES,
    derive_gate_p_campaign_identity,
    plan_next_gate_p_attempt,
    publish_gate_p_attempt_lineage,
    publish_gate_p_failure_receipt,
    reopen_gate_p_attempt_lineage,
    reopen_gate_p_failure_receipt,
)


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


def _add_source_bindings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--gate0-file-sha256", required=True)
    parser.add_argument("--gate1-file-sha256", required=True)
    parser.add_argument("--source-test-receipt-file-sha256", required=True)
    parser.add_argument("--science-config-file-sha256", required=True)
    parser.add_argument("--container-file-sha256", required=True)


def _source_bindings(arguments: argparse.Namespace) -> dict[str, str]:
    return {
        "source_git_commit": arguments.source_git_commit,
        "gate0_file_sha256": arguments.gate0_file_sha256,
        "gate1_file_sha256": arguments.gate1_file_sha256,
        "source_test_receipt_file_sha256": (arguments.source_test_receipt_file_sha256),
        "science_config_file_sha256": arguments.science_config_file_sha256,
        "container_file_sha256": arguments.container_file_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    derive = commands.add_parser(
        "derive-campaign",
        help="derive the exact current Gate-P campaign identity directory name",
    )
    _add_source_bindings(derive)

    failure = commands.add_parser(
        "publish-failure",
        help="publish one immutable failed-attempt receipt at its fixed path",
    )
    failure.add_argument("--project-root", type=Path, required=True)
    failure.add_argument("--attempt-root", type=Path, required=True)
    failure.add_argument("--raw-sacct", type=Path, required=True)
    failure.add_argument("--stdout", type=Path, required=True)
    failure.add_argument("--stderr", type=Path, required=True)
    failure.add_argument("--job-id", required=True)
    failure.add_argument("--failure-stage", choices=sorted(FAILURE_STAGES), required=True)
    failure.add_argument("--captured-at-utc")
    _add_source_bindings(failure)

    inspect_failure = commands.add_parser(
        "inspect-failure",
        help="deeply re-open a failure receipt and every byte it binds",
    )
    inspect_failure.add_argument("--project-root", type=Path, required=True)
    inspect_failure.add_argument("--receipt", type=Path, required=True)
    inspect_failure.add_argument("--receipt-file-sha256", required=True)

    plan = commands.add_parser(
        "plan-next",
        help="derive and validate the next identity/index before creating its root",
    )
    plan.add_argument("--project-root", type=Path, required=True)
    plan.add_argument("--predecessor-receipt", type=Path, required=True)
    plan.add_argument("--predecessor-receipt-file-sha256", required=True)
    _add_source_bindings(plan)

    lineage = commands.add_parser(
        "publish-lineage",
        help="publish the immediate predecessor edge before the next submission",
    )
    lineage.add_argument("--project-root", type=Path, required=True)
    lineage.add_argument("--attempt-root", type=Path, required=True)
    lineage.add_argument("--predecessor-receipt", type=Path, required=True)
    lineage.add_argument("--predecessor-receipt-file-sha256", required=True)
    _add_source_bindings(lineage)

    inspect_lineage = commands.add_parser(
        "inspect-lineage",
        help="deeply re-open a pre-submit lineage and its predecessor receipt",
    )
    inspect_lineage.add_argument("--project-root", type=Path, required=True)
    inspect_lineage.add_argument("--lineage", type=Path, required=True)
    inspect_lineage.add_argument("--lineage-file-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "derive-campaign":
        campaign = derive_gate_p_campaign_identity(**_source_bindings(arguments))
        _emit(
            {
                "status": "r3_gate_p_campaign_identity_derived_non_authorizing",
                **campaign,
            }
        )
        return 0
    if arguments.command == "publish-failure":
        receipt = publish_gate_p_failure_receipt(
            project_root=arguments.project_root,
            attempt_root=arguments.attempt_root,
            raw_sacct_path=arguments.raw_sacct,
            stdout_path=arguments.stdout,
            stderr_path=arguments.stderr,
            job_id=arguments.job_id,
            failure_stage=arguments.failure_stage,
            captured_at_utc=arguments.captured_at_utc,
            **_source_bindings(arguments),
        )
        status = "r3_gate_p_failure_receipt_published_non_authorizing"
        _emit(
            {
                "status": status,
                "receipt": str(receipt.artifact_path),
                "receipt_file_sha256": receipt.file_sha256,
                "receipt_sha256": receipt.receipt_sha256,
                "campaign_identity_sha256": receipt.campaign_identity_sha256,
                "attempt_index": receipt.attempt_index,
                "job_id": receipt.job_id,
                "source_binding_evidence": receipt.payload["source_binding_evidence"],
            }
        )
        return 0
    if arguments.command == "inspect-failure":
        receipt = reopen_gate_p_failure_receipt(
            arguments.receipt,
            project_root=arguments.project_root,
            expected_file_sha256=arguments.receipt_file_sha256,
        )
        _emit(
            {
                "status": "r3_gate_p_failure_receipt_revalidated_non_authorizing",
                "receipt_file_sha256": receipt.file_sha256,
                "receipt_sha256": receipt.receipt_sha256,
                "campaign_identity_sha256": receipt.campaign_identity_sha256,
                "attempt_index": receipt.attempt_index,
                "job_id": receipt.job_id,
                "source_binding_evidence": receipt.payload["source_binding_evidence"],
            }
        )
        return 0
    if arguments.command == "plan-next":
        predecessor = reopen_gate_p_failure_receipt(
            arguments.predecessor_receipt,
            project_root=arguments.project_root,
            expected_file_sha256=arguments.predecessor_receipt_file_sha256,
        )
        plan = plan_next_gate_p_attempt(
            project_root=arguments.project_root,
            predecessor_receipt=predecessor,
            **_source_bindings(arguments),
        )
        _emit(
            {
                "status": "r3_gate_p_next_attempt_planned_non_authorizing",
                **plan,
            }
        )
        return 0
    if arguments.command == "publish-lineage":
        predecessor = reopen_gate_p_failure_receipt(
            arguments.predecessor_receipt,
            project_root=arguments.project_root,
            expected_file_sha256=arguments.predecessor_receipt_file_sha256,
        )
        lineage = publish_gate_p_attempt_lineage(
            project_root=arguments.project_root,
            attempt_root=arguments.attempt_root,
            predecessor_receipt=predecessor,
            **_source_bindings(arguments),
        )
        _emit(
            {
                "status": "r3_gate_p_attempt_lineage_published_non_authorizing",
                "lineage": str(lineage.artifact_path),
                "lineage_file_sha256": lineage.file_sha256,
                "lineage_sha256": lineage.lineage_sha256,
                "campaign_identity_sha256": lineage.campaign_identity_sha256,
                "attempt_index": lineage.attempt_index,
                "predecessor_receipt_file_sha256": (lineage.predecessor_file_sha256),
                "predecessor_receipt_sha256": (lineage.predecessor_receipt_sha256),
            }
        )
        return 0
    lineage = reopen_gate_p_attempt_lineage(
        arguments.lineage,
        project_root=arguments.project_root,
        expected_file_sha256=arguments.lineage_file_sha256,
    )
    _emit(
        {
            "status": "r3_gate_p_attempt_lineage_revalidated_non_authorizing",
            "lineage_file_sha256": lineage.file_sha256,
            "lineage_sha256": lineage.lineage_sha256,
            "campaign_identity_sha256": lineage.campaign_identity_sha256,
            "attempt_index": lineage.attempt_index,
            "predecessor_receipt_file_sha256": lineage.predecessor_file_sha256,
            "predecessor_receipt_sha256": lineage.predecessor_receipt_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

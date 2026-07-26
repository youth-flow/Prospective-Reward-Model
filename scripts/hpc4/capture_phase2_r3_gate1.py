#!/usr/bin/env python3
"""Capture or verify the clean-source and live-container parts of R3 Gate 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart_reward.phase2_r3_gate1 import (
    capture_live_r3_gate1_evidence,
    capture_r3_gate1_source_test_receipt,
    inspect_r3_gate1_bundle,
    inspect_r3_source_test_receipt,
    verify_live_r3_gate1_bundle,
)


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


def _inspection_payload(value: object, *, status: str) -> dict[str, object]:
    payload = {
        "status": status,
        "schema_version": value.schema_version,
        "artifact_sha256": value.artifact_sha256,
        "file_sha256": value.file_sha256,
        "source_artifact_sha256": value.source_artifact_sha256,
        "source_commit": value.source_commit,
        "formal_authorization": value.formal_authorization,
    }
    container = getattr(value, "container_artifact_sha256", None)
    if container is not None:
        payload["container_artifact_sha256"] = container
        payload["formal_path_count"] = value.formal_path_count
    else:
        payload["verification_suite_sha256"] = value.verification_suite_sha256
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "source-test",
        help="test the fixed clean repository and publish to fixed persistence",
    )

    capture = commands.add_parser(
        "capture-live",
        help="capture live HPC4 source/container verification against a source receipt",
    )
    capture.add_argument("--container", type=Path, required=True)
    capture.add_argument("--source-test-receipt-file-sha256", required=True)

    verify = commands.add_parser(
        "verify-live",
        help="reverify the caller-pinned live Gate-1 artifact",
    )
    verify.add_argument("--container", type=Path, required=True)
    verify.add_argument("--gate1-file-sha256", required=True)
    verify.add_argument("--source-test-receipt-file-sha256", required=True)

    inspect_source = commands.add_parser(
        "inspect-source-test",
        help="strict non-authorizing inspection of a copied source-test receipt",
    )
    inspect_source.add_argument("artifact", type=Path)
    inspect_source.add_argument("--expected-file-sha256")

    inspect_gate1 = commands.add_parser(
        "inspect-live",
        help="strict non-authorizing inspection of a copied Gate-1 artifact",
    )
    inspect_gate1.add_argument("artifact", type=Path)
    inspect_gate1.add_argument("--expected-file-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "source-test":
        inspection = capture_r3_gate1_source_test_receipt()
        _emit(
            _inspection_payload(
                inspection,
                status="r3_source_test_receipt_published_non_authorizing",
            )
        )
        return 0
    if arguments.command == "capture-live":
        inspection = capture_live_r3_gate1_evidence(
            container=arguments.container,
            expected_source_test_receipt_file_sha256=(arguments.source_test_receipt_file_sha256),
        )
        _emit(
            _inspection_payload(
                inspection,
                status="r3_gate1_live_evidence_published_pending_reverification",
            )
        )
        return 0
    if arguments.command == "verify-live":
        capabilities = verify_live_r3_gate1_bundle(
            container=arguments.container,
            expected_file_sha256=arguments.gate1_file_sha256,
            expected_source_test_receipt_file_sha256=(arguments.source_test_receipt_file_sha256),
        )
        _emit(
            {
                "status": "r3_gate1_live_capabilities_issued",
                "gate1_artifact_sha256": capabilities.gate1.artifact_sha256,
                "gate1_file_sha256": capabilities.gate1.file_sha256,
                "source_artifact_sha256": capabilities.source.artifact_sha256,
                "container_artifact_sha256": capabilities.container.artifact_sha256,
                "live_reverification_sha256": (capabilities.gate1.live_reverification_sha256),
            }
        )
        return 0
    if arguments.command == "inspect-source-test":
        inspection = inspect_r3_source_test_receipt(
            arguments.artifact,
            expected_file_sha256=arguments.expected_file_sha256,
        )
        _emit(
            _inspection_payload(
                inspection,
                status="r3_source_test_receipt_inspected_non_authorizing",
            )
        )
        return 0
    inspection = inspect_r3_gate1_bundle(
        arguments.artifact,
        expected_file_sha256=arguments.expected_file_sha256,
    )
    _emit(
        _inspection_payload(
            inspection,
            status="r3_gate1_evidence_inspected_non_authorizing",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

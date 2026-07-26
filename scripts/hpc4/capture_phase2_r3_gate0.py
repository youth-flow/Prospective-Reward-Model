#!/usr/bin/env python3
"""Capture or verify the immutable R2-failure parent required by R3 Gate 0.

Formal ``capture`` and ``verify`` have no output/root override.  They operate
only on the fixed HPC4 production namespace and return a typed capability.
``inspect`` is intentionally non-authorizing and may be used on a copied
artifact for offline review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart_reward.phase2_r3_gate0 import (
    capture_live_r3_gate0_bundle,
    inspect_r3_gate0_bundle,
    verify_live_r3_gate0_bundle,
)


def _capability_payload(value: object) -> dict[str, object]:
    return {
        "status": "validated_live_hpc4_gate0",
        "schema_version": value.schema_version,
        "role": value.role,
        "artifact_sha256": value.artifact_sha256,
        "file_sha256": value.file_sha256,
        "production_relative": value.production_relative,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture and publish live HPC4 Gate 0")
    capture.add_argument("--container", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="reverify published live HPC4 Gate 0")
    verify.add_argument("--container", type=Path, required=True)

    inspect = subparsers.add_parser(
        "inspect",
        help="strict offline inspection; never emits an authorization capability",
    )
    inspect.add_argument("artifact", type=Path)

    arguments = parser.parse_args(argv)
    if arguments.command == "capture":
        result = _capability_payload(capture_live_r3_gate0_bundle(container=arguments.container))
    elif arguments.command == "verify":
        result = _capability_payload(verify_live_r3_gate0_bundle(container=arguments.container))
    else:
        report = inspect_r3_gate0_bundle(arguments.artifact)
        result = {
            "status": "validated_offline_non_authorizing_inspection",
            "schema_version": report.schema_version,
            "artifact_sha256": report.artifact_sha256,
            "file_sha256": report.file_sha256,
            "formal_authorization": report.formal_authorization,
        }
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

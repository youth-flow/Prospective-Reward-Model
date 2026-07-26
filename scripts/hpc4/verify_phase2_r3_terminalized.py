#!/usr/bin/env python3
"""Deeply revalidate one already-published R3 scheduler terminal bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from smart_reward.phase2_r3_profile_artifacts import (
    reopen_verified_gate_p_operational_bundle,
)
from smart_reward.phase2_r3_terminal import (
    reopen_primary_segment_runtime_closure,
    reopen_profile_allocation_intent,
    reopen_profile_slurm_runtime_receipt,
    revalidate_completed_primary_terminal,
    revalidate_continuable_primary_terminal,
    revalidate_successful_profile_terminal,
)

_EXPECTED_ENTRIES = {"raw-sacct.psv", "parsed-sacct.json", "terminal-manifest.json"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_evidence_directory(path: Path) -> tuple[str, str]:
    if not path.is_dir() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("terminal evidence directory must be canonical and non-symlink")
    entries = {child.name for child in path.iterdir()}
    if entries != _EXPECTED_ENTRIES:
        raise ValueError("terminal evidence directory does not have the exact closed entry set")
    for child in path.iterdir():
        if not child.is_file() or child.is_symlink():
            raise ValueError("terminal evidence entry must be a regular non-symlink file")
    return _sha256(path / "terminal-manifest.json"), _sha256(path / "raw-sacct.psv")


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("gatep", "primary-continuable", "primary-completed"),
    )
    parser.add_argument("--operational-bundle", type=Path, required=True)
    parser.add_argument("--operational-bundle-file-sha256", required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--allocation-intent", type=Path)
    parser.add_argument("--allocation-intent-file-sha256")
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument("--runtime-receipt-file-sha256")
    parser.add_argument("--runtime-closure", type=Path)
    parser.add_argument("--runtime-closure-file-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest_sha256, raw_sha256 = _exact_evidence_directory(
        arguments.evidence_directory,
    )
    bundle = reopen_verified_gate_p_operational_bundle(
        arguments.operational_bundle,
        expected_file_sha256=arguments.operational_bundle_file_sha256,
    )
    if arguments.mode == "gatep":
        if any(
            value is None
            for value in (
                arguments.allocation_intent,
                arguments.allocation_intent_file_sha256,
                arguments.runtime_receipt,
                arguments.runtime_receipt_file_sha256,
            )
        ) or any(
            value is not None
            for value in (
                arguments.runtime_closure,
                arguments.runtime_closure_file_sha256,
            )
        ):
            raise ValueError("Gate-P revalidation arguments are incomplete or mixed")
        intent = reopen_profile_allocation_intent(
            arguments.allocation_intent,
            expected_file_sha256=arguments.allocation_intent_file_sha256,
        )
        receipt = reopen_profile_slurm_runtime_receipt(
            arguments.runtime_receipt,
            expected_file_sha256=arguments.runtime_receipt_file_sha256,
            operational_bundle=bundle,
            allocation_intent=intent,
        )
        terminal = revalidate_successful_profile_terminal(
            bundle,
            runtime_receipt=receipt,
            evidence_directory=arguments.evidence_directory,
            expected_manifest_file_sha256=manifest_sha256,
            expected_raw_sacct_sha256=raw_sha256,
        )
        status = "r3_gate_p_scheduler_terminal_revalidated"
    else:
        if any(
            value is None
            for value in (
                arguments.runtime_closure,
                arguments.runtime_closure_file_sha256,
            )
        ) or any(
            value is not None
            for value in (
                arguments.allocation_intent,
                arguments.allocation_intent_file_sha256,
                arguments.runtime_receipt,
                arguments.runtime_receipt_file_sha256,
            )
        ):
            raise ValueError("Gate-R revalidation arguments are incomplete or mixed")
        closure = reopen_primary_segment_runtime_closure(
            arguments.runtime_closure,
            expected_file_sha256=arguments.runtime_closure_file_sha256,
            operational_bundle=bundle,
        )
        if arguments.mode == "primary-continuable":
            terminal = revalidate_continuable_primary_terminal(
                bundle,
                runtime_closure=closure,
                evidence_directory=arguments.evidence_directory,
                expected_manifest_file_sha256=manifest_sha256,
                expected_raw_sacct_sha256=raw_sha256,
            )
            status = "r3_primary_continuable_scheduler_terminal_revalidated"
        else:
            terminal = revalidate_completed_primary_terminal(
                bundle,
                runtime_closure=closure,
                evidence_directory=arguments.evidence_directory,
                expected_manifest_file_sha256=manifest_sha256,
                expected_raw_sacct_sha256=raw_sha256,
            )
            status = "r3_primary_completed_scheduler_terminal_revalidated"
    _emit(
        {
            "status": status,
            "manifest_file_sha256": terminal.manifest_file_sha256,
            "raw_sacct_sha256": raw_sha256,
            "terminal_sha256": terminal.terminal_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

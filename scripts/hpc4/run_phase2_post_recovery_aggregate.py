#!/usr/bin/env python3
"""Run a strict post-recovery pilot aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from smart_reward.phase2_post_recovery_aggregate import (
    write_phase2_post_recovery_aggregate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("results", type=Path, nargs=3)
    parser.add_argument("--publication-output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--terminal-evidence", type=Path, required=True)
    parser.add_argument("--terminal-evidence-sha256", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--submission-intent-sha256", required=True)
    parser.add_argument("--submission-ledger-sha256", required=True)
    parser.add_argument("--submission-intent-reference", type=Path, required=True)
    parser.add_argument("--submission-ledger-reference", type=Path, required=True)
    parser.add_argument("--aggregator-git-commit", required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--hf-inventory-sha256", required=True)
    parser.add_argument(
        "--reference-base",
        type=Path,
        help=(
            "directory of the final published aggregate; use this when the "
            "output itself is first staged elsewhere"
        ),
    )
    parser.add_argument(
        "--phase2-overlay-reference",
        type=Path,
        help="final immutable evidence-copy path for the committed Phase-2 overlay",
    )
    parser.add_argument("--beta-source-aggregate", type=Path)
    parser.add_argument("--horizon-parent-aggregate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = write_phase2_post_recovery_aggregate(
        arguments.overlay,
        arguments.results,
        arguments.output,
        authorization_path=arguments.authorization,
        authorization_sha256=arguments.authorization_sha256,
        terminal_evidence_path=arguments.terminal_evidence,
        terminal_evidence_sha256=arguments.terminal_evidence_sha256,
        array_job_id=arguments.array_job_id,
        submission_intent_sha256=arguments.submission_intent_sha256,
        submission_ledger_sha256=arguments.submission_ledger_sha256,
        submission_intent_reference_path=arguments.submission_intent_reference,
        submission_ledger_reference_path=arguments.submission_ledger_reference,
        aggregator_git_commit=arguments.aggregator_git_commit,
        producer_git_commit=arguments.producer_git_commit,
        image_sha256=arguments.image_sha256,
        hf_inventory_sha256=arguments.hf_inventory_sha256,
        reference_base=arguments.reference_base,
        require_production_output_path=True,
        publication_output_path=arguments.publication_output,
        phase2_overlay_reference_path=arguments.phase2_overlay_reference,
        beta_source_aggregate_path=arguments.beta_source_aggregate,
        horizon_parent_aggregate_path=arguments.horizon_parent_aggregate,
    )
    selection = payload["selection"]
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": hashlib.sha256(arguments.output.read_bytes()).hexdigest(),
                "phase2_design_sha256": payload["phase2_design_sha256"],
                "selection_accepted": selection["selection_accepted"],
                "next_action": selection["next_action"],
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

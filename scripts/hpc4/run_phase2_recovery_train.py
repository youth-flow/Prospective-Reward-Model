#!/usr/bin/env python3
"""CLI wrapper for the isolated one-shot recovery trainer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_recovery import (
    run_phase2_recovery_train_only,
    write_recovery_failure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay")
    parser.add_argument("registry")
    parser.add_argument("artifact")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    design_sha: str | None = None
    registry_sha: str | None = None
    try:
        bundle = load_phase2_config_bundle(args.overlay)
        design_sha = bundle.design_identity
        control = bundle.config["recovery_control"]
        registry_sha = control["parent_failure_registry_sha256"]
        payload = run_phase2_recovery_train_only(
            args.overlay,
            registry=args.registry,
            artifact_dir=args.artifact,
            current_run_manifest=args.manifest,
            output_json=args.output,
            seed=args.seed,
            device=args.device,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        failure = os.environ.get("PRORM_FAILURE_EVIDENCE")
        if failure:
            write_recovery_failure(
                failure,
                error=error,
                seed=args.seed,
                recovery_design_sha256=design_sha,
                registry_sha256=registry_sha,
            )
        raise
    print(
        f"recovery train-only SUCCESS seed={payload['seed']} "
        f"design={payload['recovery_design_sha256']} output={Path(args.output).name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

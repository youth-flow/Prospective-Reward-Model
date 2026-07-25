#!/usr/bin/env python3
"""Materialize the real authorization-bound calibration overlay once."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path

import yaml

from smart_reward.config import config_hash, load_config
from smart_reward.phase2_config import (
    PHASE2_POST_RECOVERY_SCHEMA_VERSION,
    PHASE2_RECOVERY_SUCCESS_PROJECTION_SCHEMA,
    PHASE2_RECOVERY_SUCCESS_REFERENCE_SCHEMA,
    load_phase2_config_bundle,
    phase2_design_identity,
    validate_phase2_config,
)
from smart_reward.phase2_post_recovery_control import (
    POST_RECOVERY_DESIGN_NAME,
    verify_recovery_authorization_file,
)
from smart_reward.phase2_training import compile_phase2_training_settings

_OUTPUT_RELATIVE = Path("configs/common_beta_post_recovery_calibration.yaml")
_TEMPLATE_RELATIVE = Path("configs/common_beta_pilot.yaml")
_BASE_RELATIVE = Path("configs/common_beta_pilot_base.yaml")
_RECOVERY_RELATIVE = Path("configs/common_beta_recovery_pilot.yaml")
_PROJECTION_FIELDS = (
    "recovery_design_sha256",
    "optimizer_schedule_sha256",
    "source_array_job_id",
    "execution_revision",
    "ordered_seeds",
    "recovery_status",
    "full_calibration_authorized",
    "authorized_information",
    "recovery_outputs_reusable",
    "validation_or_heldout_access_authorized",
    "policy_or_final_utility_access_authorized",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("authorization", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser


def _require_real_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical real directory")
    return absolute


def _committed_clean_repo(repo_root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repo_root),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    if status:
        raise ValueError("overlay materialization requires a clean committed worktree")
    for relative in (_TEMPLATE_RELATIVE, _BASE_RELATIVE, _RECOVERY_RELATIVE):
        subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repo_root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative.as_posix(),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        committed = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repo_root),
                "cat-file",
                "blob",
                f"{head}:{relative.as_posix()}",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        worktree = (repo_root / relative).read_bytes()
        if committed != worktree:
            raise ValueError(f"template bytes differ from HEAD: {relative}")
    return head


def _candidate(
    *,
    template: dict[str, object],
    recovery: dict[str, object],
    authorization: dict[str, object],
    authorization_sha256: str,
) -> dict[str, object]:
    candidate = copy.deepcopy(template)
    candidate["schema_version"] = PHASE2_POST_RECOVERY_SCHEMA_VERSION
    design = candidate["design"]
    reward = candidate["reward_model"]
    if not isinstance(design, dict) or not isinstance(reward, dict):
        raise TypeError("pilot template does not have the expected complete overlay shape")
    design["name"] = POST_RECOVERY_DESIGN_NAME
    projection = {
        "schema_version": PHASE2_RECOVERY_SUCCESS_PROJECTION_SCHEMA,
        "source_schema_version": authorization["schema_version"],
        **{field: copy.deepcopy(authorization[field]) for field in _PROJECTION_FIELDS},
    }
    candidate["recovery_success_reference"] = {
        "schema_version": PHASE2_RECOVERY_SUCCESS_REFERENCE_SCHEMA,
        "artifact_sha256": authorization_sha256,
        "authorization_projection": projection,
    }

    recovery_reward = recovery["reward_model"]
    if not isinstance(recovery_reward, dict):
        raise TypeError("recovery overlay reward_model is malformed")
    recovery_protocol = copy.deepcopy(recovery_reward["optimizer_protocol"])
    if not isinstance(recovery_protocol, dict):
        raise TypeError("recovery optimizer protocol is malformed")
    if recovery_protocol.pop("one_time_recovery", None) is not True:
        raise ValueError("source recovery protocol is not the locked one-time protocol")
    recovery_protocol["schema_version"] = "deterministic-adamw-lr-decay/v1"
    recovery_protocol["role"] = "frozen_post_recovery_phase2_optimizer"
    recovery_protocol["source_recovery_authorization_sha256"] = authorization_sha256
    reward["optimizer_protocol"] = recovery_protocol
    recovery_convergence = recovery_reward["adaptive_convergence"]
    if not isinstance(recovery_convergence, dict):
        raise TypeError("recovery adaptive convergence block is malformed")
    reward["adaptive_convergence"] = copy.deepcopy(recovery_convergence)
    identifiability = reward["identifiability"]
    if not isinstance(identifiability, dict):
        raise TypeError("pilot identifiability block is malformed")
    identifiability["algorithmic_tie_break"] = (
        "exact_zero_initialized_deterministic_adamw_lr_decay_path"
    )
    return candidate


def _write_exclusive(path: Path, raw: bytes) -> None:
    parent = _require_real_directory(path.parent, name="overlay output parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    try:
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except PermissionError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repo_root = _require_real_directory(arguments.repo_root, name="repo root")
    git_commit = _committed_clean_repo(repo_root)
    output = repo_root / _OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration overlay: {output}")
    authorization_path = arguments.authorization.absolute()
    authorization = verify_recovery_authorization_file(
        authorization_path,
        expected_sha256=arguments.expected_sha256,
    )
    template = load_config(repo_root / _TEMPLATE_RELATIVE)
    base = load_config(repo_root / _BASE_RELATIVE)
    recovery = load_config(repo_root / _RECOVERY_RELATIVE)
    candidate = _candidate(
        template=template,
        recovery=recovery,
        authorization=authorization,
        authorization_sha256=arguments.expected_sha256,
    )
    normalized = validate_phase2_config(candidate, base_config=base)
    candidate_identity = phase2_design_identity(normalized)
    settings = compile_phase2_training_settings({"config": normalized, "base_config": base})
    protocol = settings.convergence.optimizer_protocol
    if (
        settings.convergence.max_steps != 12760
        or settings.convergence.check_interval != 20
        or settings.convergence.consecutive_checks != 3
        or protocol is None
        or protocol.mode != "adopted"
        or protocol.source_recovery_authorization_sha256 != arguments.expected_sha256
        or protocol.schedule_sha256 != authorization["optimizer_schedule_sha256"]
        or protocol.to_dict().get("scope") != "every_phase2_first_order_convergence_trainer"
    ):
        raise ValueError("candidate did not compile the one adopted schedule for all five trainers")
    raw = yaml.safe_dump(
        candidate,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")
    _write_exclusive(output, raw)
    try:
        bundle = load_phase2_config_bundle(output)
        if (
            bundle.config != normalized
            or bundle.design_identity != candidate_identity
            or config_hash(bundle.base_config) != candidate["design"]["source_config_hash"]
        ):
            raise ValueError("published overlay did not round-trip to its validated identity")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "candidate_materialized",
                "output": os.fspath(output),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "phase2_design_sha256": candidate_identity,
                "base_config_hash": config_hash(bundle.base_config),
                "authorization_sha256": arguments.expected_sha256,
                "git_commit_used_for_templates": git_commit,
                "next_action": "review_then_commit_push_and_sync_before_submission",
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

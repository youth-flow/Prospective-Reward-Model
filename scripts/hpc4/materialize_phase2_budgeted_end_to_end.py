#!/usr/bin/env python3
"""Materialize the fixed-three, non-formal Phase-2 end-to-end identity.

This entrypoint is intentionally independent from the formal confirmatory
materializer and campaign control plane.  It consumes exactly one terminally
proven production-v3 accepted freeze, then creates a fresh exploratory base and
overlay.  The accepted freeze is the single source for both the global beta and
the response horizon.

The two generated files are candidates only.  They must be reviewed, committed,
pushed, and synchronized before a separate budgeted execution entrypoint may
consume them.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from smart_reward.config import config_hash
from smart_reward.phase2_config import (
    PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
    PHASE2_BUDGETED_END_TO_END_CONFIG,
    PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    PHASE2_POST_RECOVERY_SCHEMA_VERSION,
    load_phase2_config_bundle,
    phase2_design_identity,
    validate_phase2_config,
    validate_post_recovery_authorization_reference,
)
from smart_reward.phase2_pilot_aggregate import (
    verify_beta_source_aggregate,
    verify_horizon_parent_aggregate,
)
from smart_reward.phase2_post_recovery_control import (
    OPTIMIZER_SCHEDULE_SHA256,
    verify_post_recovery_aggregate_success_receipt,
    verify_recovery_authorization_file,
)
from smart_reward.phase2_training import compile_phase2_training_settings

_PROJECT_ROOT = Path("/project/sigroup/smart-reward-model")
_AGGREGATE_ROOT = _PROJECT_ROOT / "aggregates"
_AUTHORIZATION_PATH = (
    _PROJECT_ROOT / "runs" / "phase2-recovery-pilot" / "recovery-success-authorization.json"
)
_BUDGETED_BASE_RELATIVE = Path(PHASE2_BUDGETED_END_TO_END_BASE_CONFIG)
_BUDGETED_OVERLAY_RELATIVE = Path(PHASE2_BUDGETED_END_TO_END_CONFIG)
_BUDGETED_RECEIPT_RELATIVE = Path(
    "configs/.common_beta_post_recovery_budgeted_end_to_end.materialized.json"
)
_BUDGETED_EVIDENCE_ROLE = "budgeted_end_to_end_fixed_three_exploratory_only"
_MATERIALIZATION_RECEIPT_SCHEMA = "budgeted-end-to-end-fixed-three-materialization-receipt/v1"
if PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE != _BUDGETED_EVIDENCE_ROLE:
    raise RuntimeError("core config and budgeted materializer evidence roles differ")


def _strict_json(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(f"{name} bytes differ from its terminal success receipt")

    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite constant {item!r}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_repo(repo_root: Path) -> Path:
    absolute = repo_root.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError("repo root must be a canonical real directory")
    return absolute


def _canonical_regular_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical regular non-symlink file")
    return absolute


def _clean_committed_source(
    repo_root: Path,
    source_overlay: Path,
    source_base: Path,
    *,
    allowed_untracked: Sequence[Path] = (),
) -> str:
    source_overlay = _canonical_regular_file(source_overlay, name="source overlay")
    source_base = _canonical_regular_file(source_base, name="source base")
    for path in (source_overlay, source_base):
        try:
            path.relative_to(repo_root)
        except ValueError as error:
            raise ValueError("source overlay and base must be inside the repo root") from error

    commit = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    for diff_arguments in (
        ["diff", "--quiet", "--ignore-submodules", "--"],
        ["diff", "--cached", "--quiet", "--ignore-submodules", "--"],
    ):
        status = subprocess.run(
            ["git", "-C", os.fspath(repo_root), *diff_arguments],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if status.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(
                status.returncode,
                status.args,
                output=status.stdout,
                stderr=status.stderr,
            )
        if status.returncode == 1:
            raise ValueError("materialization requires a clean committed worktree")

    allowed_relative: set[str] = set()
    for path in allowed_untracked:
        absolute = path.absolute()
        try:
            relative = absolute.relative_to(repo_root).as_posix()
        except ValueError as error:
            raise ValueError("allowed untracked paths must be inside the repo root") from error
        allowed_relative.add(relative)
    untracked_raw = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repo_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    untracked = {item.decode("utf-8") for item in untracked_raw.split(b"\0") if item}
    if not untracked.issubset(allowed_relative):
        raise ValueError("materialization requires a clean committed worktree")

    for path in (source_overlay, source_base):
        relative = path.relative_to(repo_root).as_posix()
        subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repo_root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        committed = subprocess.run(
            ["git", "-C", os.fspath(repo_root), "cat-file", "blob", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        if committed != path.read_bytes():
            raise ValueError(f"source bytes differ from HEAD: {relative}")
    return commit


def _production_aggregate(path: Path) -> tuple[Path, dict[str, object]]:
    absolute = _canonical_regular_file(path, name="accepted freeze aggregate")
    if absolute.parent.resolve(strict=True) != _AGGREGATE_ROOT.resolve(strict=True):
        raise ValueError("accepted freeze must be a canonical production aggregate")
    receipt = verify_post_recovery_aggregate_success_receipt(absolute)
    if not isinstance(receipt, dict):
        raise TypeError("accepted freeze terminal success receipt is malformed")
    receipt_sha256 = receipt.get("aggregate_sha256")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise ValueError("accepted freeze terminal receipt has an invalid aggregate SHA256")
    return absolute, receipt


def _source_matches_aggregate(
    source: Mapping[str, object],
    aggregate: Mapping[str, object],
) -> None:
    if (
        aggregate.get("schema_version") != "common-beta-pilot-selection-aggregate/v3"
        or aggregate.get("pilot_phase") != "freeze"
        or aggregate.get("formal_eligibility") is not False
        or aggregate.get("supports_formal_claim") is not False
        or aggregate.get("evidence_role") != "target_free_design_selection_only"
        or aggregate.get("phase2_design_sha256") != phase2_design_identity(source)
    ):
        raise ValueError(
            "accepted freeze is not a production-v3 target-free aggregate "
            "for the supplied source identity"
        )
    design = _mapping(source.get("design"), name="source design")
    if (
        design.get("stage") != "pilot"
        or design.get("pilot_phase") != "freeze"
        or design.get("formal_eligibility") is not False
        or design.get("evidence_role") != "pilot_design_selection_only"
    ):
        raise ValueError("source identity is not the formally excluded freeze pilot")
    boundary = _mapping(
        aggregate.get("information_boundary"),
        name="accepted freeze information_boundary",
    )
    if not boundary or any(value is not False for value in boundary.values()):
        raise ValueError("accepted freeze crossed the target-free information boundary")


def _accepted_freeze(
    aggregate: Mapping[str, object],
    source: Mapping[str, object],
) -> tuple[float, int]:
    selection = _mapping(
        aggregate.get("selection"),
        name="accepted freeze selection",
    )
    horizon = _mapping(
        aggregate.get("horizon"),
        name="accepted freeze horizon",
    )
    beta = _finite_positive(
        selection.get("frozen_global_beta"),
        name="accepted frozen global beta",
    )
    next_beta = _finite_positive(
        selection.get("next_global_beta"),
        name="accepted freeze next global beta",
    )
    beta_grid_index = selection.get("beta_grid_index")
    if (
        aggregate.get("pilot_phase") != "freeze"
        or selection.get("schema_version") != "pilot-freeze-selection/v1"
        or selection.get("all_seeds_and_arms_used_same_beta") is not True
        or selection.get("all_pre_oracle_safety_gates_passed") is not True
        or selection.get("all_length_gates_passed") is not True
        or selection.get("all_non_length_safety_gates_passed") is not True
        or selection.get("selection_accepted") is not True
        or selection.get("accepted_for_confirmatory_identity") is not True
        or selection.get("next_action") != "freeze_confirmatory_design_identity"
        or horizon.get("all_seed_length_gates_passed") is not True
        or isinstance(beta_grid_index, bool)
        or not isinstance(beta_grid_index, int)
        or beta_grid_index < 0
        or next_beta != beta
    ):
        raise ValueError(
            "only a fully accepted production-v3 freeze may authorize "
            "budgeted end-to-end materialization"
        )

    policy = _mapping(source.get("policy"), name="source policy")
    objective = _mapping(source.get("objective"), name="source objective")
    common = _mapping(
        objective.get("common_beta"),
        name="source objective.common_beta",
    )
    evaluation = _mapping(source.get("evaluation"), name="source evaluation")
    maximum = _mapping(
        evaluation.get("max_length"),
        name="source evaluation.max_length",
    )
    source_tokens = policy.get("max_response_tokens")
    source_index = maximum.get("horizon_grid_index")
    if (
        isinstance(source_tokens, bool)
        or not isinstance(source_tokens, int)
        or isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or horizon.get("candidate_horizon_tokens") != source_tokens
        or horizon.get("horizon_grid_index") != source_index
        or selection.get("next_horizon_tokens") != source_tokens
        or common.get("frozen_global_beta") != beta
    ):
        raise ValueError("accepted freeze does not bind the source beta and response horizon")
    return beta, beta_grid_index


def _expected_freeze_overlay_relative(beta_grid_index: int) -> Path:
    if isinstance(beta_grid_index, bool) or not isinstance(beta_grid_index, int):
        raise ValueError("accepted freeze beta grid index must be an integer")
    if beta_grid_index < 0:
        raise ValueError("accepted freeze beta grid index must be non-negative")
    name = (
        "common_beta_post_recovery_freeze.yaml"
        if beta_grid_index == 0
        else f"common_beta_post_recovery_freeze_retry_{beta_grid_index}.yaml"
    )
    return Path("configs") / name


def _budgeted_projection(
    freeze: Mapping[str, object],
    base: Mapping[str, object],
    *,
    freeze_sha256: str,
    frozen_beta: float,
) -> tuple[dict[str, object], dict[str, object]]:
    candidate_base = copy.deepcopy(dict(base))
    base_run = _mapping(candidate_base.get("run"), name="base.run")
    base_run["name"] = "budgeted-end-to-end-materialization"
    base_run["seeds"] = list(PHASE2_BUDGETED_END_TO_END_SEEDS)

    candidate = copy.deepcopy(dict(freeze))
    design = _mapping(candidate.get("design"), name="design")
    design.update(
        {
            "name": "common-beta-post-recovery-budgeted-end-to-end-v1",
            "stage": PHASE2_BUDGETED_END_TO_END_STAGE,
            "pilot_phase": None,
            "formal_eligibility": False,
            "evidence_role": _BUDGETED_EVIDENCE_ROLE,
            "source_config": _BUDGETED_BASE_RELATIVE.as_posix(),
            "source_config_hash": config_hash(candidate_base),
        }
    )

    run = _mapping(candidate.get("run"), name="run")
    run.update(
        {
            "seeds": list(PHASE2_BUDGETED_END_TO_END_SEEDS),
            "confirmatory": False,
            "formal_eligibility": False,
            "excluded_from_confirmatory_evidence": True,
        }
    )

    reward = _mapping(candidate.get("reward_model"), name="reward_model")
    identifiability = _mapping(
        reward.get("identifiability"),
        name="reward_model.identifiability",
    )
    identifiability.update(
        {
            "role": "budgeted_end_to_end_exploratory_frozen_identifiability_audit",
            "require_full_column_rank": False,
            "confirmatory_freeze_requirement": (
                "satisfied_by_accepted_freeze_budgeted_end_to_end_identity"
            ),
        }
    )

    objective = _mapping(candidate.get("objective"), name="objective")
    common = _mapping(
        objective.get("common_beta"),
        name="objective.common_beta",
    )
    common.update(
        {
            "calibration_split": "excluded_pilot",
            "calibration_source": (
                "accepted_freeze_global_beta_in_budgeted_end_to_end_design_identity"
            ),
            "rule": "single_accepted_freeze_global_beta_scalar",
            "frozen_global_beta": frozen_beta,
            "beta_source_aggregate_sha256": freeze_sha256,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": None,
            "primary_execution_role": "budgeted_end_to_end_exploratory_primary",
            "sensitivity_execution_role": "not_executed_in_budgeted_end_to_end",
            "sensitivity_executed_separately": False,
            "sensitivity_eligible_for_primary_claim": False,
        }
    )
    tangent = _mapping(
        objective.get("full_tangent"),
        name="objective.full_tangent",
    )
    ridge = _mapping(
        tangent.get("ridge"),
        name="objective.full_tangent.ridge",
    )
    ridge.update(
        {
            "primary_execution_role": "budgeted_end_to_end_exploratory_primary",
            "sensitivity_execution_role": "not_executed_in_budgeted_end_to_end",
            "sensitivity_executed_separately": False,
            "sensitivity_eligible_for_primary_claim": False,
        }
    )

    evaluation = _mapping(candidate.get("evaluation"), name="evaluation")
    decision = _mapping(
        evaluation.get("decision_gates"),
        name="evaluation.decision_gates",
    )
    decision.update(
        {
            "application": "budgeted_end_to_end_exploratory_summary_only",
            "supports_formal_claim": False,
        }
    )
    maximum = _mapping(
        evaluation.get("max_length"),
        name="evaluation.max_length",
    )
    maximum.update(
        {
            "role": "budgeted_end_to_end_pre_oracle_safety_gate",
            "measure_only": False,
            "formal_gate": False,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": freeze_sha256,
            "post_pilot_requirement": ("satisfied_by_accepted_freeze_budgeted_end_to_end_identity"),
        }
    )
    return candidate, candidate_base


def _verify_authorization_and_optimizer(
    config: Mapping[str, object],
    base: Mapping[str, object],
    *,
    authorization_path: Path,
    authorization_sha256: str,
) -> dict[str, object]:
    authorization = verify_recovery_authorization_file(
        authorization_path,
        expected_sha256=authorization_sha256,
    )
    validate_post_recovery_authorization_reference(
        config["recovery_success_reference"],
        authorization_payload_sha256=authorization_sha256,
        authorization_payload=authorization,
    )
    settings = compile_phase2_training_settings({"config": dict(config), "base_config": dict(base)})
    protocol = settings.convergence.optimizer_protocol
    if (
        settings.stage != PHASE2_BUDGETED_END_TO_END_STAGE
        or settings.formal_eligibility is not False
        or settings.seeds != PHASE2_BUDGETED_END_TO_END_SEEDS
        or settings.convergence.max_steps != 12760
        or settings.convergence.check_interval != 20
        or settings.convergence.consecutive_checks != 3
        or protocol is None
        or protocol.mode != "adopted"
        or protocol.source_recovery_authorization_sha256 != authorization_sha256
        or protocol.schedule_sha256 != OPTIMIZER_SCHEDULE_SHA256
        or protocol.to_dict().get("scope") != "every_phase2_first_order_convergence_trainer"
    ):
        raise ValueError(
            "budgeted identity did not retain the authorization-bound optimizer schedule"
        )
    return {
        "authorization": authorization,
        "optimizer_schedule_sha256": protocol.schedule_sha256,
        "stage": settings.stage,
        "formal_eligibility": settings.formal_eligibility,
        "ordered_seeds": list(settings.seeds),
    }


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


def _fsync_directory(path: Path) -> None:
    directory = _require_real_directory(path, name="output parent")
    try:
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except PermissionError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _yaml_bytes(value: Mapping[str, object]) -> bytes:
    return yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _matching_published_file(path: Path, raw: bytes) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or path.resolve(strict=True) != path.absolute()
    ):
        raise FileExistsError(f"refusing conflicting non-regular publication target: {path}")
    if path.read_bytes() != raw:
        raise FileExistsError(f"refusing to overwrite conflicting publication target: {path}")
    return True


def _publish_bytes_no_replace(
    path: Path,
    raw: bytes,
    *,
    staging_directory: Path,
) -> bool:
    """Atomically link one fully-fsynced file, resuming only exact prior bytes."""

    if _matching_published_file(path, raw):
        return False
    parent = _require_real_directory(path.parent, name="publication parent")
    staging = _require_real_directory(staging_directory, name="publication staging directory")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=staging,
        prefix=".prorm-budgeted-materialization-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o640)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _matching_published_file(path, raw):
                return False
            raise
        _fsync_directory(parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_freeze_overlay", type=Path)
    parser.add_argument("accepted_freeze_aggregate", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, default=_AUTHORIZATION_PATH)
    parser.add_argument("--authorization-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repo_root = _canonical_repo(arguments.repo_root)
    base_output = repo_root / _BUDGETED_BASE_RELATIVE
    overlay_output = repo_root / _BUDGETED_OVERLAY_RELATIVE
    receipt_output = repo_root / _BUDGETED_RECEIPT_RELATIVE

    source_overlay = _canonical_regular_file(
        arguments.source_freeze_overlay,
        name="source freeze overlay",
    )
    try:
        source_overlay.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("source freeze overlay must be inside the repo root") from error
    source_bundle = load_phase2_config_bundle(source_overlay)
    source = source_bundle.config
    if source.get("schema_version") != PHASE2_POST_RECOVERY_SCHEMA_VERSION:
        raise ValueError("source overlay is not the post-recovery schema")
    git_commit = _clean_committed_source(
        repo_root,
        source_overlay,
        source_bundle.base_config_path,
        allowed_untracked=(base_output, overlay_output, receipt_output),
    )

    predecessor, predecessor_receipt = _production_aggregate(arguments.accepted_freeze_aggregate)
    predecessor_sha256 = str(predecessor_receipt["aggregate_sha256"])
    aggregate = _strict_json(
        predecessor,
        name="accepted freeze aggregate",
        expected_sha256=predecessor_sha256,
    )
    _source_matches_aggregate(source, aggregate)
    frozen_beta, beta_grid_index = _accepted_freeze(aggregate, source)
    expected_source = repo_root / _expected_freeze_overlay_relative(beta_grid_index)
    if source_overlay != expected_source:
        raise ValueError("source overlay path differs from its exact accepted-freeze identity")

    candidate, candidate_base = _budgeted_projection(
        source,
        source_bundle.base_config,
        freeze_sha256=predecessor_sha256,
        frozen_beta=frozen_beta,
    )
    normalized = validate_phase2_config(candidate, base_config=candidate_base)
    beta_binding = verify_beta_source_aggregate(normalized, predecessor)
    horizon_binding = verify_horizon_parent_aggregate(normalized, predecessor)
    if (
        beta_binding is None
        or horizon_binding is None
        or beta_binding.get("sha256") != predecessor_sha256
        or horizon_binding.get("sha256") != predecessor_sha256
    ):
        raise ValueError(
            "budgeted identity did not bind one accepted freeze as both beta "
            "source and horizon parent"
        )
    authorization_binding = _verify_authorization_and_optimizer(
        normalized,
        candidate_base,
        authorization_path=arguments.authorization.absolute(),
        authorization_sha256=arguments.authorization_sha256,
    )

    base_raw = _yaml_bytes(candidate_base)
    overlay_raw = _yaml_bytes(candidate)
    design_sha256 = phase2_design_identity(normalized)
    receipt_raw = _canonical_json_bytes(
        {
            "schema_version": _MATERIALIZATION_RECEIPT_SCHEMA,
            "stage": PHASE2_BUDGETED_END_TO_END_STAGE,
            "formal_claim_eligible": False,
            "git_commit_used_for_source": git_commit,
            "base_relative_path": _BUDGETED_BASE_RELATIVE.as_posix(),
            "base_file_sha256": hashlib.sha256(base_raw).hexdigest(),
            "overlay_relative_path": _BUDGETED_OVERLAY_RELATIVE.as_posix(),
            "overlay_file_sha256": hashlib.sha256(overlay_raw).hexdigest(),
            "phase2_design_sha256": design_sha256,
            "accepted_freeze_aggregate_sha256": predecessor_sha256,
            "authorization_sha256": arguments.authorization_sha256,
        }
    )
    if _matching_published_file(receipt_output, receipt_raw):
        if not _matching_published_file(base_output, base_raw) or not _matching_published_file(
            overlay_output, overlay_raw
        ):
            raise RuntimeError("complete materialization receipt lacks its exact identity files")
        raise FileExistsError("refusing to overwrite completed budgeted end-to-end identity")

    staging_directory = repo_root / ".git"
    resumed_publications: list[str] = []
    if not _publish_bytes_no_replace(
        base_output,
        base_raw,
        staging_directory=staging_directory,
    ):
        resumed_publications.append(_BUDGETED_BASE_RELATIVE.as_posix())
    if not _publish_bytes_no_replace(
        overlay_output,
        overlay_raw,
        staging_directory=staging_directory,
    ):
        resumed_publications.append(_BUDGETED_OVERLAY_RELATIVE.as_posix())
    round_trip = load_phase2_config_bundle(overlay_output)
    if (
        round_trip.config != normalized
        or round_trip.design_identity != design_sha256
        or config_hash(round_trip.base_config) != config_hash(candidate_base)
        or round_trip.base_config_path != base_output
    ):
        raise ValueError("published budgeted identity did not round-trip to its validated config")
    _publish_bytes_no_replace(
        receipt_output,
        receipt_raw,
        staging_directory=staging_directory,
    )

    report = {
        "action": "budgeted_end_to_end_identity_materialized",
        "stage": PHASE2_BUDGETED_END_TO_END_STAGE,
        "formal": False,
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "evidence_role": _BUDGETED_EVIDENCE_ROLE,
        "base_output": os.fspath(base_output),
        "base_file_sha256": hashlib.sha256(base_raw).hexdigest(),
        "overlay_output": os.fspath(overlay_output),
        "overlay_file_sha256": hashlib.sha256(overlay_raw).hexdigest(),
        "materialization_receipt": os.fspath(receipt_output),
        "materialization_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "phase2_design_sha256": round_trip.design_identity,
        "accepted_freeze_aggregate_sha256": predecessor_sha256,
        "beta_source_aggregate_sha256": predecessor_sha256,
        "horizon_parent_aggregate_sha256": predecessor_sha256,
        "frozen_global_beta": frozen_beta,
        "ordered_seeds": list(PHASE2_BUDGETED_END_TO_END_SEEDS),
        "seed_count": len(PHASE2_BUDGETED_END_TO_END_SEEDS),
        "git_commit_used_for_source": git_commit,
        "optimizer_schedule_sha256": authorization_binding["optimizer_schedule_sha256"],
        "authorization_sha256": arguments.authorization_sha256,
        "materialization_mode": "recoverable_atomic_link_no_overwrite",
        "resumed_publications": resumed_publications,
        "reward_heads_reused": False,
        "optimizer_state_reused": False,
        "pilot_outputs_reused_as_model_inputs": False,
        "next_action": ("review_register_commit_push_and_sync_before_budgeted_submission"),
    }
    print(
        json.dumps(
            report,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

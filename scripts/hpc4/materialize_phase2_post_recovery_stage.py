#!/usr/bin/env python3
"""Materialize the next authorization-bound post-recovery Phase-2 identity.

Only target-free v3 pilot decisions may reach this entrypoint.  It supports
the predeclared state machine:

* failed length gate -> next-horizon calibration;
* accepted calibration -> first frozen-beta rehearsal;
* each non-length freeze failure -> the immediately next exact doubled-beta retry;
* accepted freeze -> exact-30 confirmatory overlay and base.

Every output is a new, no-overwrite file.  The caller must review, register in
``configs/identities.json``, commit, push, and sync it before submission.
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
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from smart_reward.config import config_hash
from smart_reward.phase2_config import (
    PHASE2_CONFIRMATORY_SEEDS,
    PHASE2_POST_RECOVERY_SCHEMA_VERSION,
    load_phase2_config_bundle,
    phase2_design_identity,
    validate_phase2_config,
)
from smart_reward.phase2_pilot_aggregate import (
    verify_beta_source_aggregate,
    verify_horizon_parent_aggregate,
)
from smart_reward.phase2_post_recovery_control import (
    OPTIMIZER_SCHEDULE_SHA256,
    verify_post_recovery_aggregate_success_receipt,
    verify_recovery_authorization_config_binding,
)

_PROJECT_ROOT = Path("/project/sigroup/smart-reward-model")
_AGGREGATE_ROOT = _PROJECT_ROOT / "aggregates"
_AUTHORIZATION_PATH = (
    _PROJECT_ROOT / "runs" / "phase2-recovery-pilot" / "recovery-success-authorization.json"
)
_CONFIRMATORY_BASE_RELATIVE = Path("configs/common_beta_post_recovery_confirmatory_base.yaml")
_CONFIRMATORY_OVERLAY_RELATIVE = Path("configs/common_beta_post_recovery_confirmatory.yaml")


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"{name} contains non-finite constant {item!r}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _finite(value: object, *, name: str) -> float:
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


def _clean_commit(
    repo_root: Path,
    source_overlay: Path,
    source_base: Path,
) -> str:
    commit = subprocess.run(
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
        raise ValueError("materialization requires a clean committed worktree")
    for path in (source_overlay, source_base):
        relative = path.relative_to(repo_root).as_posix()
        subprocess.run(
            ["git", "-C", os.fspath(repo_root), "ls-files", "--error-unmatch", "--", relative],
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


def _source_matches_aggregate(
    source: Mapping[str, object],
    aggregate: Mapping[str, object],
) -> None:
    if (
        aggregate.get("schema_version") != "common-beta-pilot-selection-aggregate/v3"
        or aggregate.get("formal_eligibility") is not False
        or aggregate.get("supports_formal_claim") is not False
        or aggregate.get("phase2_design_sha256") != phase2_design_identity(source)
    ):
        raise ValueError("predecessor v3 does not bind the supplied source identity")
    boundary = _mapping(
        aggregate.get("information_boundary"),
        name="predecessor information_boundary",
    )
    if any(value is not False for value in boundary.values()):
        raise ValueError("predecessor crossed the target-free information boundary")


def _source_semantic_overlay_relative(
    source: Mapping[str, object],
    aggregate: Mapping[str, object],
) -> Path:
    design = _mapping(source.get("design"), name="source design")
    phase = design.get("pilot_phase")
    if phase == "calibration":
        evaluation = _mapping(source.get("evaluation"), name="source evaluation")
        maximum = _mapping(
            evaluation.get("max_length"),
            name="source evaluation.max_length",
        )
        index = maximum.get("horizon_grid_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("source calibration horizon grid index is invalid")
        name = (
            "common_beta_post_recovery_calibration.yaml"
            if index == 0
            else f"common_beta_post_recovery_calibration_horizon_{index}.yaml"
        )
    elif phase == "freeze":
        selection = _mapping(
            aggregate.get("selection"),
            name="source freeze aggregate selection",
        )
        index = selection.get("beta_grid_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("source freeze beta grid index is invalid")
        name = (
            "common_beta_post_recovery_freeze.yaml"
            if index == 0
            else f"common_beta_post_recovery_freeze_retry_{index}.yaml"
        )
    else:
        raise ValueError("source post-recovery identity is not a pilot phase")
    return Path("configs") / name


def _calibration_projection(
    source: Mapping[str, object],
    *,
    next_tokens: int,
    next_index: int,
    parent_sha256: str,
) -> dict[str, object]:
    candidate = copy.deepcopy(dict(source))
    design = _mapping(candidate["design"], name="design")
    design.update(
        {
            "name": f"common-beta-post-recovery-calibration-horizon-{next_index}-v1",
            "stage": "pilot",
            "pilot_phase": "calibration",
            "formal_eligibility": False,
            "evidence_role": "pilot_design_selection_only",
        }
    )
    common = _mapping(
        _mapping(candidate["objective"], name="objective")["common_beta"],
        name="objective.common_beta",
    )
    common.update(
        {
            "calibration_split": "train",
            "calibration_source": "transformed_operational_oracle",
            "rule": (
                "pilot_seed_candidate_from_oracle_train_fisher_quadratic_for_future_global_beta"
            ),
            "frozen_global_beta": None,
            "beta_source_aggregate_sha256": None,
            "sensitivity_k_cal": [0.001, 0.01],
            "sensitivity_frozen_global_beta_multipliers": None,
            "primary_execution_role": "pilot_global_beta_calibration_candidate",
            "sensitivity_execution_role": ("required_separate_global_beta_candidate_sensitivity"),
        }
    )
    policy = _mapping(candidate["policy"], name="policy")
    policy["max_response_tokens"] = next_tokens
    evaluation = _mapping(candidate["evaluation"], name="evaluation")
    decision = _mapping(evaluation["decision_gates"], name="evaluation.decision_gates")
    decision["application"] = "pilot_calibration_target_free_selection"
    maximum = _mapping(evaluation["max_length"], name="evaluation.max_length")
    maximum.update(
        {
            "candidate_horizon_tokens": next_tokens,
            "horizon_grid_index": next_index,
            "parent_pilot_aggregate_sha256": parent_sha256,
            "previous_horizon_failed_length_gate": True,
            "role": "pilot_horizon_selection_input",
            "measure_only": True,
            "formal_gate": False,
            "formal_threshold": 0.05,
            "post_pilot_requirement": "issue_new_pilot_freeze_design_identity",
        }
    )
    return candidate


def _freeze_projection(
    source: Mapping[str, object],
    *,
    beta: float,
    beta_source_sha256: str,
    horizon_parent_sha256: str,
    beta_grid_index: int,
) -> dict[str, object]:
    candidate = copy.deepcopy(dict(source))
    design = _mapping(candidate["design"], name="design")
    design.update(
        {
            "name": (
                "common-beta-post-recovery-freeze-v1"
                if beta_grid_index == 0
                else f"common-beta-post-recovery-freeze-retry-{beta_grid_index}-v1"
            ),
            "stage": "pilot",
            "pilot_phase": "freeze",
            "formal_eligibility": False,
            "evidence_role": "pilot_design_selection_only",
        }
    )
    common = _mapping(
        _mapping(candidate["objective"], name="objective")["common_beta"],
        name="objective.common_beta",
    )
    common.update(
        {
            "calibration_split": "excluded_pilot_calibration",
            "calibration_source": (
                "frozen_calibration_aggregate_candidate_in_pilot_freeze_design_identity"
            ),
            "rule": "pilot_fixed_global_beta_target_free_safety_rehearsal",
            "frozen_global_beta": beta,
            "beta_source_aggregate_sha256": beta_source_sha256,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": None,
            "primary_execution_role": "pilot_frozen_global_beta_safety_rehearsal",
            "sensitivity_execution_role": "new_pilot_freeze_design_identity_double_beta_grid",
        }
    )
    evaluation = _mapping(candidate["evaluation"], name="evaluation")
    decision = _mapping(evaluation["decision_gates"], name="evaluation.decision_gates")
    decision["application"] = "pilot_freeze_target_free_safety_selection"
    maximum = _mapping(evaluation["max_length"], name="evaluation.max_length")
    maximum.update(
        {
            "role": "pilot_frozen_global_beta_safety_selection",
            "measure_only": True,
            "formal_gate": False,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": horizon_parent_sha256,
            "post_pilot_requirement": (
                "freeze_confirmatory_identity_if_passed_else_double_beta_new_identity"
            ),
        }
    )
    return candidate


def _confirmatory_projection(
    freeze: Mapping[str, object],
    base: Mapping[str, object],
    *,
    freeze_sha256: str,
    frozen_beta: float,
) -> tuple[dict[str, object], dict[str, object]]:
    candidate_base = copy.deepcopy(dict(base))
    base_run = _mapping(candidate_base["run"], name="base.run")
    base_run["name"] = "common-beta-post-recovery-confirmatory-materialization"
    base_run["seeds"] = list(PHASE2_CONFIRMATORY_SEEDS)

    candidate = copy.deepcopy(dict(freeze))
    design = _mapping(candidate["design"], name="design")
    design.update(
        {
            "name": "common-beta-post-recovery-confirmatory-v1",
            "stage": "confirmatory",
            "pilot_phase": None,
            "formal_eligibility": True,
            "evidence_role": "confirmatory_evidence",
            "source_config": _CONFIRMATORY_BASE_RELATIVE.as_posix(),
            "source_config_hash": config_hash(candidate_base),
        }
    )
    run = _mapping(candidate["run"], name="run")
    run.update(
        {
            "seeds": list(PHASE2_CONFIRMATORY_SEEDS),
            "confirmatory": True,
            "formal_eligibility": True,
            "excluded_from_confirmatory_evidence": False,
        }
    )
    identifiability = _mapping(
        _mapping(candidate["reward_model"], name="reward_model")["identifiability"],
        name="reward_model.identifiability",
    )
    identifiability.update(
        {
            "role": "confirmatory_frozen_identifiability_contract",
            "confirmatory_freeze_requirement": "satisfied_by_current_confirmatory_identity",
        }
    )
    common = _mapping(
        _mapping(candidate["objective"], name="objective")["common_beta"],
        name="objective.common_beta",
    )
    common.update(
        {
            "calibration_split": "excluded_pilot",
            "calibration_source": "frozen_pilot_global_beta_in_confirmatory_design_identity",
            "rule": "single_pilot_frozen_global_beta_scalar",
            "frozen_global_beta": frozen_beta,
            "beta_source_aggregate_sha256": freeze_sha256,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": [0.5, 2.0],
            "primary_execution_role": "confirmatory_primary",
            "sensitivity_execution_role": (
                "required_separate_frozen_global_beta_multiplier_sensitivity"
            ),
        }
    )
    ridge = _mapping(
        _mapping(
            _mapping(candidate["objective"], name="objective")["full_tangent"],
            name="objective.full_tangent",
        )["ridge"],
        name="objective.full_tangent.ridge",
    )
    ridge["primary_execution_role"] = "confirmatory_primary"
    ridge["sensitivity_execution_role"] = "required_separate_confirmatory_sensitivity"
    evaluation = _mapping(candidate["evaluation"], name="evaluation")
    decision = _mapping(evaluation["decision_gates"], name="evaluation.decision_gates")
    decision.update(
        {
            "application": "confirmatory_evidence_decision",
            "supports_formal_claim": True,
        }
    )
    maximum = _mapping(evaluation["max_length"], name="evaluation.max_length")
    maximum.update(
        {
            "role": "confirmatory_truncation_safety_gate",
            "measure_only": False,
            "formal_gate": True,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": freeze_sha256,
            "post_pilot_requirement": "satisfied_by_new_confirmatory_design_identity",
        }
    )
    return candidate, candidate_base


def _write_yaml_exclusive(path: Path, value: Mapping[str, object]) -> bytes:
    raw = yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return raw


def _production_aggregate(path: Path) -> Path:
    absolute = path.absolute()
    if (
        not absolute.is_file()
        or absolute.is_symlink()
        or absolute.parent.resolve(strict=True) != _AGGREGATE_ROOT
    ):
        raise ValueError("predecessor must be a canonical production aggregate")
    verify_post_recovery_aggregate_success_receipt(absolute)
    return absolute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("next-pilot", "confirmatory"),
    )
    parser.add_argument("source_overlay", type=Path)
    parser.add_argument("predecessor_aggregate", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, default=_AUTHORIZATION_PATH)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--horizon-parent-aggregate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repo_root = _canonical_repo(arguments.repo_root)
    source_overlay = arguments.source_overlay.absolute()
    source_overlay.relative_to(repo_root)
    source_bundle = load_phase2_config_bundle(source_overlay)
    commit = _clean_commit(
        repo_root,
        source_overlay,
        source_bundle.base_config_path,
    )
    source = source_bundle.config
    if source["schema_version"] != PHASE2_POST_RECOVERY_SCHEMA_VERSION:
        raise ValueError("source overlay is not the post-recovery schema")

    predecessor = _production_aggregate(arguments.predecessor_aggregate)
    predecessor_sha = _sha256(predecessor)
    payload = _strict_json(predecessor, name="predecessor v3 aggregate")
    _source_matches_aggregate(source, payload)
    source_relative = _source_semantic_overlay_relative(source, payload)
    if source_overlay != repo_root / source_relative:
        raise ValueError("source overlay path differs from its exact semantic identity")
    selection = _mapping(payload.get("selection"), name="predecessor selection")
    horizon = _mapping(payload.get("horizon"), name="predecessor horizon")
    phase = payload.get("pilot_phase")

    output_paths: list[Path] = []
    if arguments.action == "next-pilot":
        candidate_base = source_bundle.base_config
        new_base_output: Path | None = None
        if selection.get("next_action") == "issue_new_calibration_identity_at_next_horizon":
            next_tokens = selection.get("next_horizon_tokens")
            current_index = horizon.get("horizon_grid_index")
            if (
                isinstance(next_tokens, bool)
                or not isinstance(next_tokens, int)
                or isinstance(current_index, bool)
                or not isinstance(current_index, int)
                or horizon.get("all_seed_length_gates_passed") is not False
            ):
                raise ValueError("predecessor does not authorize horizon escalation")
            candidate = _calibration_projection(
                source,
                next_tokens=next_tokens,
                next_index=current_index + 1,
                parent_sha256=predecessor_sha,
            )
            candidate_base = copy.deepcopy(source_bundle.base_config)
            _mapping(candidate_base["run"], name="base.run")["name"] = (
                f"common-beta-post-recovery-calibration-horizon-{current_index + 1}"
            )
            _mapping(candidate_base["policy"], name="base.policy")["max_response_tokens"] = (
                next_tokens
            )
            base_relative = Path(
                "configs/"
                f"common_beta_post_recovery_calibration_horizon_{current_index + 1}_base.yaml"
            )
            _mapping(candidate["design"], name="design").update(
                {
                    "source_config": base_relative.as_posix(),
                    "source_config_hash": config_hash(candidate_base),
                }
            )
            new_base_output = repo_root / base_relative
            beta_source: Path | None = None
            horizon_parent = predecessor
            relative_output = Path(
                f"configs/common_beta_post_recovery_calibration_horizon_{current_index + 1}.yaml"
            )
        elif selection.get("next_action") == "issue_pilot_freeze_identity_at_recommended_beta":
            if phase != "calibration" or selection.get("horizon_accepted") is not True:
                raise ValueError("predecessor does not authorize the first freeze")
            beta = _finite(
                selection.get("recommended_pilot_freeze_beta"),
                name="recommended freeze beta",
            )
            candidate = _freeze_projection(
                source,
                beta=beta,
                beta_source_sha256=predecessor_sha,
                horizon_parent_sha256=predecessor_sha,
                beta_grid_index=0,
            )
            beta_source = predecessor
            horizon_parent = predecessor
            relative_output = Path("configs/common_beta_post_recovery_freeze.yaml")
        elif selection.get("next_action") == "issue_new_pilot_freeze_identity_at_double_beta":
            previous_index = selection.get("beta_grid_index")
            if (
                phase != "freeze"
                or selection.get("selection_accepted") is not False
                or selection.get("all_length_gates_passed") is not True
                or selection.get("all_non_length_safety_gates_passed") is not False
                or isinstance(previous_index, bool)
                or not isinstance(previous_index, int)
                or previous_index < 0
            ):
                raise ValueError("predecessor does not authorize the immediately next freeze retry")
            if arguments.horizon_parent_aggregate is None:
                raise ValueError("freeze retry requires its accepted calibration horizon parent")
            horizon_parent = _production_aggregate(arguments.horizon_parent_aggregate)
            beta = _finite(selection.get("next_global_beta"), name="doubled retry beta")
            current = _finite(selection.get("frozen_global_beta"), name="failed freeze beta")
            if not math.isclose(beta, 2.0 * current, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("freeze retry beta is not exactly doubled")
            candidate = _freeze_projection(
                source,
                beta=beta,
                beta_source_sha256=predecessor_sha,
                horizon_parent_sha256=_sha256(horizon_parent),
                beta_grid_index=previous_index + 1,
            )
            beta_source = predecessor
            relative_output = Path(
                f"configs/common_beta_post_recovery_freeze_retry_{previous_index + 1}.yaml"
            )
        else:
            raise ValueError("predecessor does not authorize another pilot identity")

        normalized = validate_phase2_config(candidate, base_config=candidate_base)
        verify_beta_source_aggregate(normalized, beta_source)
        verify_horizon_parent_aggregate(normalized, horizon_parent)
        output = repo_root / relative_output
        if (
            output.exists()
            or output.is_symlink()
            or (
                new_base_output is not None
                and (new_base_output.exists() or new_base_output.is_symlink())
            )
        ):
            raise FileExistsError(f"refusing to overwrite post-recovery identity: {output}")
        if new_base_output is not None:
            base_raw = _write_yaml_exclusive(new_base_output, candidate_base)
            output_paths.append(new_base_output)
        try:
            raw = _write_yaml_exclusive(output, candidate)
            output_paths.append(output)
            binding = verify_recovery_authorization_config_binding(
                arguments.authorization,
                output,
                expected_sha256=arguments.authorization_sha256,
                expected_pilot_phase=str(normalized["design"]["pilot_phase"]),
            )
            if binding["optimizer_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256:
                raise ValueError("new pilot did not retain the authorized optimizer schedule")
        except BaseException:
            for path in reversed(output_paths):
                path.unlink(missing_ok=True)
            raise
        report = {
            "action": "pilot_identity_materialized",
            "output": os.fspath(output),
            "output_file_sha256": hashlib.sha256(raw).hexdigest(),
            "phase2_design_sha256": phase2_design_identity(normalized),
            "predecessor_aggregate_sha256": predecessor_sha,
        }
        if new_base_output is not None:
            report["base_output"] = os.fspath(new_base_output)
            report["base_file_sha256"] = hashlib.sha256(base_raw).hexdigest()
    else:
        if (
            phase != "freeze"
            or selection.get("selection_accepted") is not True
            or selection.get("accepted_for_confirmatory_identity") is not True
            or selection.get("next_action") != "freeze_confirmatory_design_identity"
        ):
            raise ValueError("only an accepted freeze may authorize confirmatory materialization")
        frozen_beta = _finite(selection.get("frozen_global_beta"), name="accepted frozen beta")
        base = source_bundle.base_config
        candidate, candidate_base = _confirmatory_projection(
            source,
            base,
            freeze_sha256=predecessor_sha,
            frozen_beta=frozen_beta,
        )
        normalized = validate_phase2_config(candidate, base_config=candidate_base)
        verify_beta_source_aggregate(normalized, predecessor)
        verify_horizon_parent_aggregate(normalized, predecessor)
        base_output = repo_root / _CONFIRMATORY_BASE_RELATIVE
        overlay_output = repo_root / _CONFIRMATORY_OVERLAY_RELATIVE
        if any(path.exists() or path.is_symlink() for path in (base_output, overlay_output)):
            raise FileExistsError("refusing to overwrite confirmatory identity files")
        base_raw = _write_yaml_exclusive(base_output, candidate_base)
        output_paths.append(base_output)
        try:
            overlay_raw = _write_yaml_exclusive(overlay_output, candidate)
            output_paths.append(overlay_output)
            binding = verify_recovery_authorization_config_binding(
                arguments.authorization,
                overlay_output,
                expected_sha256=arguments.authorization_sha256,
                expected_stage="confirmatory",
            )
            if binding["optimizer_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256 or normalized[
                "run"
            ]["seeds"] != list(PHASE2_CONFIRMATORY_SEEDS):
                raise ValueError("confirmatory identity lost an authorization binding")
        except BaseException:
            for path in reversed(output_paths):
                path.unlink(missing_ok=True)
            raise
        report = {
            "action": "confirmatory_identity_materialized",
            "base_output": os.fspath(base_output),
            "base_file_sha256": hashlib.sha256(base_raw).hexdigest(),
            "overlay_output": os.fspath(overlay_output),
            "overlay_file_sha256": hashlib.sha256(overlay_raw).hexdigest(),
            "phase2_design_sha256": phase2_design_identity(normalized),
            "accepted_freeze_aggregate_sha256": predecessor_sha,
            "frozen_global_beta": frozen_beta,
            "seed_count": len(PHASE2_CONFIRMATORY_SEEDS),
        }

    report.update(
        {
            "git_commit_used_for_source": commit,
            "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
            "authorization_sha256": arguments.authorization_sha256,
            "materialization_mode": "fresh_no_overwrite",
            "reward_heads_reused": False,
            "optimizer_state_reused": False,
            "pilot_outputs_reused_as_model_inputs": False,
            "next_action": "review_register_commit_push_and_sync_before_submission",
        }
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

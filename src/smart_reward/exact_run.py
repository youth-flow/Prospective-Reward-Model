"""Train and evaluate the two frozen exact-delta reward learners."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    exact_delta_artifact_metadata_sha256,
    load_exact_delta_artifact,
)
from .config import PROTOCOL, config_hash, validate_config
from .evaluation import (
    GeometrySettings,
    evaluate_local_policy,
    evaluate_reference_policy,
    solve_natural_direction,
    validate_natural_direction,
)
from .exact import (
    ExactDeltaExperiment,
    ExactSplitData,
    MLETrainingConfig,
    ProTrainingConfig,
    RewardFitResult,
    evaluate_reward_head,
    fit_mle_reward,
    fit_pro_reward,
    validate_pro_reward_fit,
)
from .runtime import producer_identity, sha256_file

EXACT_COMPARISON_SCHEMA = "exact-delta-reward-comparison/v1"
# Validation may run on a different L20 or on CPU. Preserve the source metric
# only when independent float64 reproduction agrees to tight cross-device
# roundoff; scientific solver gates remain unchanged and are checked separately.
_REUSE_ABSOLUTE_TOLERANCE = 1.0e-10
_REUSE_RELATIVE_TOLERANCE = 1.0e-7


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_producer_identity(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    required = {"git_commit", "image_sha256", "hf_inventory_sha256"}
    if set(value) != required:
        raise ValueError(f"{name} must contain complete immutable identities")
    result = dict(value)
    git_commit = result["git_commit"]
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError(f"{name}.git_commit must be a lowercase Git commit")
    _validate_sha256(result["image_sha256"], name=f"{name}.image_sha256")
    _validate_sha256(
        result["hf_inventory_sha256"],
        name=f"{name}.hf_inventory_sha256",
    )
    return result


def _to_device(split: ExactSplitData, device: torch.device) -> ExactSplitData:
    return ExactSplitData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores.to(device),
        reward_features=split.reward_features.to(device),
        true_rewards=split.true_rewards.to(device),
    )


def _experiment_to_device(
    experiment: ExactDeltaExperiment,
    device: torch.device,
) -> ExactDeltaExperiment:
    return ExactDeltaExperiment(
        train=_to_device(experiment.train, device),
        validation=_to_device(experiment.validation, device),
        test=_to_device(experiment.test, device),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _canonical_component_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_reused_numeric_component(
    source: object,
    recomputed: object,
    *,
    name: str,
) -> float:
    maximum_absolute_difference = 0.0

    def compare(left: object, right: object, path: str) -> None:
        nonlocal maximum_absolute_difference
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                raise ValueError(f"{path} structure mismatch")
            for key in sorted(left):
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                raise ValueError(f"{path} structure mismatch")
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if left is not right:
                raise ValueError(f"{path} value mismatch")
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            left_value = float(left)
            right_value = float(right)
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                raise ValueError(f"{path} must be finite")
            difference = abs(left_value - right_value)
            maximum_absolute_difference = max(maximum_absolute_difference, difference)
            if not math.isclose(
                left_value,
                right_value,
                rel_tol=_REUSE_RELATIVE_TOLERANCE,
                abs_tol=_REUSE_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    f"{path} numerical mismatch: source={left_value:.17g}, "
                    f"recomputed={right_value:.17g}"
                )
            return
        if left != right:
            raise ValueError(f"{path} value mismatch")

    compare(source, recomputed, name)
    return maximum_absolute_difference


def _validated_source_direction(
    source: Mapping[str, Any],
    *,
    method: str,
    split: ExactSplitData,
    rewards: torch.Tensor,
    settings: GeometrySettings,
) -> tuple[torch.Tensor, dict[str, Any]]:
    directions = source.get("policy_directions")
    if not isinstance(directions, dict):
        raise ValueError("reused source policy directions are missing")
    values = directions.get(method)
    if not isinstance(values, list) or len(values) != split.policy_dimension:
        raise ValueError(f"reused {method} direction has the wrong dimension")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in values
    ):
        raise ValueError(f"reused {method} direction must contain finite numbers")
    direction = torch.tensor(
        values,
        device=split.policy_scores.device,
        dtype=torch.float64,
    )
    relative_residual = validate_natural_direction(
        split,
        rewards,
        direction,
        settings,
    )
    return direction, {
        "source_component_sha256": _canonical_component_sha256(values),
        "validation": {"relative_residual": relative_residual},
    }


def _validated_reusable_pro_fit(
    path: Path,
    train: ExactSplitData,
    config: ProTrainingConfig,
    *,
    expected_config_hash: str,
    expected_artifact_identity: str,
    expected_seed: int,
) -> tuple[RewardFitResult, dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        source = json.load(stream)
    if not isinstance(source, dict) or source.get("schema") != EXACT_COMPARISON_SCHEMA:
        raise ValueError("reused Pro-RM source has an unsupported schema")
    if source.get("protocol") != PROTOCOL:
        raise ValueError("reused Pro-RM source protocol mismatch")
    if source.get("config_sha256") != expected_config_hash:
        raise ValueError("reused Pro-RM source config hash mismatch")
    if source.get("artifact_metadata_sha256") != expected_artifact_identity:
        raise ValueError("reused Pro-RM source artifact identity mismatch")
    if source.get("seed") != expected_seed:
        raise ValueError("reused Pro-RM source seed mismatch")
    source_producer = _validate_producer_identity(
        source.get("producer"),
        name="reused Pro-RM source producer",
    )
    methods = source.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("reused Pro-RM source methods are missing")
    record = methods.get("Pro-RM")
    if not isinstance(record, dict) or record.get("converged") is not True:
        raise ValueError("reused Pro-RM source did not converge")
    weights = record.get("head_weight")
    if not isinstance(weights, list) or len(weights) != train.reward_dimension:
        raise ValueError("reused Pro-RM head has the wrong dimension")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in weights
    ):
        raise ValueError("reused Pro-RM head must contain finite numbers")
    iterations = record.get("iterations")
    inner_pcg_calls = record.get("inner_pcg_calls")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("reused Pro-RM iterations must be a positive integer")
    if (
        isinstance(inner_pcg_calls, bool)
        or not isinstance(inner_pcg_calls, int)
        or inner_pcg_calls < 0
    ):
        raise ValueError("reused Pro-RM inner_pcg_calls must be a non-negative integer")
    recorded_relative_residual = record.get("relative_residual")
    if (
        isinstance(recorded_relative_residual, bool)
        or not isinstance(recorded_relative_residual, (int, float))
        or not math.isfinite(float(recorded_relative_residual))
        or not 0.0 <= float(recorded_relative_residual) <= config.outer_tolerance
    ):
        raise ValueError("reused Pro-RM recorded residual did not pass its gate")
    validated = validate_pro_reward_fit(
        train,
        torch.tensor(weights, dtype=torch.float64, device=train.reward_features.device),
        config,
        source_iterations=iterations,
        source_inner_pcg_calls=inner_pcg_calls,
    )
    recorded_head = _validate_sha256(
        record.get("head_sha256"),
        name="reused Pro-RM head_sha256",
    )
    if recorded_head != validated.head_sha256:
        raise ValueError("reused Pro-RM head SHA-256 mismatch")
    recorded_numbers: dict[str, float] = {}
    for name in ("objective", "gradient_norm", "effective_inner_tolerance"):
        item = record.get(name)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
        ):
            raise ValueError(f"reused Pro-RM {name} is invalid")
        recorded_numbers[name] = float(item)
    reused = RewardFitResult(
        method="Pro-RM",
        weight=validated.weight,
        objective=recorded_numbers["objective"],
        gradient_norm=recorded_numbers["gradient_norm"],
        converged=True,
        iterations=iterations,
        head_sha256=recorded_head,
        inner_pcg_calls=inner_pcg_calls,
        relative_residual=float(recorded_relative_residual),
        effective_inner_tolerance=recorded_numbers["effective_inner_tolerance"],
    )
    return (
        reused,
        {
            "mode": "validated_reuse",
            "source_result_sha256": sha256_file(path),
            "source_component_sha256": _canonical_component_sha256(record),
            "source_producer": source_producer,
            "validation": {
                "head_sha256": validated.head_sha256,
                "relative_residual": validated.relative_residual,
                "gradient_norm": validated.gradient_norm,
                "objective": validated.objective,
                "effective_inner_tolerance": validated.effective_inner_tolerance,
            },
        },
        source,
    )


def run_exact_reward_comparison(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    reuse_pro_from: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Fit MLE-RM and Pro-RM before reading either held-out split."""

    normalized = validate_config(config)
    if normalized.get("protocol") != PROTOCOL:
        raise ValueError(f"reward comparison requires protocol {PROTOCOL}")
    if seed not in normalized["run"]["seeds"]:
        raise ValueError("seed must be one of the configured experiment seeds")
    digest = config_hash(normalized)
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    experiment = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    experiment = _experiment_to_device(experiment, target_device)

    reward_config = normalized["reward_model"]
    mle_config = reward_config["mle"]
    pro_config = reward_config["pro"]
    geometry_config = normalized["geometry"]
    print("reward_fit method=MLE-RM status=running", flush=True)
    mle = fit_mle_reward(
        experiment.train,
        MLETrainingConfig(
            max_iterations=int(mle_config["max_iterations"]),
            history_size=int(mle_config["history_size"]),
            gradient_tolerance=float(mle_config["gradient_tolerance"]),
            change_tolerance=float(mle_config["change_tolerance"]),
            microbatch_size=int(mle_config["microbatch_size"]),
        ),
    )
    print(
        f"reward_fit method=MLE-RM status=complete iterations={mle.iterations}",
        flush=True,
    )
    effective_pro_config = ProTrainingConfig(
        relative_damping=float(geometry_config["damping_relative_to_mean_diagonal"]),
        fisher_estimator=geometry_config["fisher_estimator"],
        inner_max_iterations=int(geometry_config["cg_max_iterations"]),
        inner_tolerance=float(geometry_config["cg_tolerance"]),
        outer_max_iterations=int(pro_config["max_iterations"]),
        outer_tolerance=float(pro_config["tolerance"]),
        residual_recompute_interval=int(pro_config["residual_recompute_interval"]),
    )
    if reuse_pro_from is None:
        print("reward_fit method=Pro-RM status=running", flush=True)
        pro = fit_pro_reward(experiment.train, effective_pro_config)
        pro_provenance: dict[str, Any] = {"mode": "computed"}
        reuse_source: dict[str, Any] | None = None
    else:
        print("reward_fit method=Pro-RM status=validating-reuse", flush=True)
        pro, pro_provenance, reuse_source = _validated_reusable_pro_fit(
            Path(reuse_pro_from),
            experiment.train,
            effective_pro_config,
            expected_config_hash=digest,
            expected_artifact_identity=artifact_identity,
            expected_seed=seed,
        )
    print(
        f"reward_fit method=Pro-RM status=complete iterations={pro.iterations}",
        flush=True,
    )
    fits = {"MLE-RM": mle, "Pro-RM": pro}
    # Held-out targets are first read after both train-only fits have finished.
    evaluations: dict[str, Any] = {}
    evaluation_provenance: dict[str, Any] = {"MLE-RM": {"mode": "computed"}}
    for method, fit in fits.items():
        recomputed = {
            split_name: evaluate_reward_head(getattr(experiment, split_name), fit.weight).to_dict()
            for split_name in ("train", "validation", "test")
        }
        if method != "Pro-RM" or reuse_source is None:
            evaluations[method] = recomputed
            evaluation_provenance[method] = {"mode": "computed"}
            continue
        source_evaluations = reuse_source.get("evaluation")
        if not isinstance(source_evaluations, dict):
            raise ValueError("reused source reward evaluation is missing")
        source_component = source_evaluations.get(method)
        maximum_difference = _validate_reused_numeric_component(
            source_component,
            recomputed,
            name="reused Pro-RM reward evaluation",
        )
        evaluations[method] = source_component
        evaluation_provenance[method] = {
            "mode": "validated_reuse",
            "source_result_sha256": pro_provenance["source_result_sha256"],
            "source_component_sha256": _canonical_component_sha256(source_component),
            "source_producer": pro_provenance["source_producer"],
            "validation": {"maximum_absolute_difference": maximum_difference},
        }
    settings = GeometrySettings(
        fisher_estimator=geometry_config["fisher_estimator"],
        relative_damping=float(geometry_config["damping_relative_to_mean_diagonal"]),
        cg_tolerance=float(geometry_config["cg_tolerance"]),
        cg_max_iterations=int(geometry_config["cg_max_iterations"]),
        residual_recompute_interval=int(geometry_config["residual_recompute_interval"]),
    )
    train_rewards = {
        "mle_rm": experiment.train.reward_features.to(dtype=torch.float64) @ mle.weight,
        "pro_rm": experiment.train.reward_features.to(dtype=torch.float64) @ pro.weight,
        "oracle": experiment.train.true_rewards.to(dtype=torch.float64),
    }
    train_directions: dict[str, torch.Tensor] = {}
    direction_provenance: dict[str, Any] = {}
    for method, rewards in train_rewards.items():
        if reuse_source is not None and method in {"pro_rm", "oracle"}:
            print(f"natural_direction method={method} status=validating-reuse", flush=True)
            direction, validation = _validated_source_direction(
                reuse_source,
                method=method,
                split=experiment.train,
                rewards=rewards,
                settings=settings,
            )
            train_directions[method] = direction
            direction_provenance[method] = {
                "mode": "validated_reuse",
                "source_result_sha256": pro_provenance["source_result_sha256"],
                "source_producer": pro_provenance["source_producer"],
                **validation,
            }
        else:
            print(f"natural_direction method={method} status=running", flush=True)
            train_directions[method] = solve_natural_direction(
                experiment.train,
                rewards,
                settings,
            )
            direction_provenance[method] = {"mode": "computed"}
        print(f"natural_direction method={method} status=complete", flush=True)
    local_policy_evaluation: dict[str, Any] = {}
    reusable_local_methods = {"pi0", "pro_rm", "oracle"} if reuse_source is not None else set()
    local_validation_differences = {method: 0.0 for method in reusable_local_methods}
    source_local = reuse_source.get("local_policy_evaluation") if reuse_source is not None else None
    if reuse_source is not None and not isinstance(source_local, dict):
        raise ValueError("reused source local-policy evaluation is missing")
    for beta in normalized["policy_update"]["beta_grid"]:
        beta_key = str(beta)
        recomputed_local = {
            "pi0": evaluate_reference_policy(
                experiment.test,
                beta=float(beta),
                settings=settings,
            ).to_dict(),
            **{
                method: evaluate_local_policy(
                    experiment.test,
                    direction,
                    beta=float(beta),
                    settings=settings,
                ).to_dict()
                for method, direction in train_directions.items()
            },
        }
        local_policy_evaluation[beta_key] = {"mle_rm": recomputed_local["mle_rm"]}
        for method in ("pi0", "pro_rm", "oracle"):
            if method not in reusable_local_methods:
                local_policy_evaluation[beta_key][method] = recomputed_local[method]
                continue
            assert isinstance(source_local, dict)
            source_beta = source_local.get(beta_key)
            if not isinstance(source_beta, dict):
                raise ValueError(f"reused source local-policy beta {beta_key} is missing")
            source_component = source_beta.get(method)
            difference = _validate_reused_numeric_component(
                source_component,
                recomputed_local[method],
                name=f"reused local-policy evaluation beta={beta_key} method={method}",
            )
            local_validation_differences[method] = max(
                local_validation_differences[method],
                difference,
            )
            local_policy_evaluation[beta_key][method] = source_component
    local_policy_provenance: dict[str, Any] = {"mle_rm": {"mode": "computed"}}
    for method in ("pi0", "pro_rm", "oracle"):
        if method not in reusable_local_methods:
            local_policy_provenance[method] = {"mode": "computed"}
            continue
        assert isinstance(source_local, dict)
        component = {
            str(beta): source_local[str(beta)][method]
            for beta in normalized["policy_update"]["beta_grid"]
        }
        local_policy_provenance[method] = {
            "mode": "validated_reuse",
            "source_result_sha256": pro_provenance["source_result_sha256"],
            "source_component_sha256": _canonical_component_sha256(component),
            "source_producer": pro_provenance["source_producer"],
            "validation": {
                "maximum_absolute_difference": local_validation_differences[method],
            },
        }
    result: dict[str, Any] = {
        "schema": EXACT_COMPARISON_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": digest,
        "artifact_metadata_sha256": artifact_identity,
        "seed": seed,
        "validation_usage": "diagnostics_only",
        "training_target": "exact_delta_r_star",
        "policy_geometry": dict(geometry_config),
        "pro_solver": dict(pro_config),
        "fit_provenance": {
            "MLE-RM": {"mode": "computed"},
            "Pro-RM": pro_provenance,
        },
        "component_provenance": {
            "reward_fit": {
                "MLE-RM": {"mode": "computed"},
                "Pro-RM": pro_provenance,
            },
            "reward_evaluation": evaluation_provenance,
            "natural_direction": direction_provenance,
            "local_policy_evaluation": local_policy_provenance,
        },
        "methods": {method: fit.to_dict() for method, fit in fits.items()},
        "policy_directions": {
            method: direction.detach().cpu().tolist()
            for method, direction in train_directions.items()
        },
        "evaluation": evaluations,
        "local_policy_evaluation": local_policy_evaluation,
        "dimensions": {
            "train_prompts": experiment.train.num_prompts,
            "candidates_per_prompt": experiment.train.num_candidates,
            "train_nodes": experiment.train.num_prompts * experiment.train.num_candidates,
            "train_edges": experiment.train.num_edges,
            "independent_train_contrasts": experiment.train.num_prompts
            * (experiment.train.num_candidates - 1),
            "reward_head": experiment.train.reward_dimension,
            "policy_tangent": experiment.train.policy_dimension,
        },
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), result)
    return result


def _validate_component_provenance_record(
    value: object,
    *,
    name: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    mode = value.get("mode")
    if mode == "computed":
        if value != {"mode": "computed"}:
            raise ValueError(f"{name} computed provenance has unexpected fields")
        return
    if mode != "validated_reuse":
        raise ValueError(f"{name} provenance mode is invalid")
    _validate_sha256(
        value.get("source_result_sha256"),
        name=f"{name} source_result_sha256",
    )
    _validate_sha256(
        value.get("source_component_sha256"),
        name=f"{name} source_component_sha256",
    )
    _validate_producer_identity(
        value.get("source_producer"),
        name=f"{name} source producer",
    )
    validation = value.get("validation")
    if not isinstance(validation, dict) or not validation:
        raise ValueError(f"{name} validation is missing")
    for key, item in validation.items():
        if key.endswith("_sha256"):
            _validate_sha256(item, name=f"{name} validation {key}")
            continue
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
        ):
            raise ValueError(f"{name} validation {key} is invalid")


def load_exact_reward_comparison(
    path: str | os.PathLike[str],
    *,
    expected_config_hash: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Load the strict result fields consumed by the NGD-LoRA stage."""

    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema") != EXACT_COMPARISON_SCHEMA:
        raise ValueError("unsupported exact reward comparison")
    if set(value.get("methods", {})) != {"MLE-RM", "Pro-RM"}:
        raise ValueError("comparison must contain exactly MLE-RM and Pro-RM")
    if expected_config_hash is not None and value.get("config_sha256") != expected_config_hash:
        raise ValueError("comparison config hash mismatch")
    if expected_seed is not None and value.get("seed") != expected_seed:
        raise ValueError("comparison seed mismatch")
    _validate_sha256(value.get("config_sha256"), name="comparison config_sha256")
    _validate_sha256(
        value.get("artifact_metadata_sha256"),
        name="comparison artifact_metadata_sha256",
    )
    fit_provenance = value.get("fit_provenance")
    if fit_provenance is not None:
        if not isinstance(fit_provenance, dict) or set(fit_provenance) != {
            "MLE-RM",
            "Pro-RM",
        }:
            raise ValueError("comparison fit provenance is incomplete")
        if fit_provenance["MLE-RM"] != {"mode": "computed"}:
            raise ValueError("comparison MLE-RM provenance is invalid")
        pro_provenance = fit_provenance["Pro-RM"]
        if not isinstance(pro_provenance, dict):
            raise ValueError("comparison Pro-RM provenance is invalid")
        mode = pro_provenance.get("mode")
        if mode == "computed":
            if pro_provenance != {"mode": "computed"}:
                raise ValueError("computed Pro-RM provenance has unexpected fields")
        elif mode == "validated_reuse":
            _validate_sha256(
                pro_provenance.get("source_result_sha256"),
                name="comparison Pro-RM source_result_sha256",
            )
            _validate_sha256(
                pro_provenance.get("source_component_sha256"),
                name="comparison Pro-RM source_component_sha256",
            )
            _validate_producer_identity(
                pro_provenance.get("source_producer"),
                name="comparison Pro-RM source producer",
            )
            validation = pro_provenance.get("validation")
            if not isinstance(validation, dict):
                raise ValueError("comparison Pro-RM reuse validation is missing")
            _validate_sha256(
                validation.get("head_sha256"),
                name="comparison Pro-RM validation head_sha256",
            )
            for name in (
                "relative_residual",
                "gradient_norm",
                "objective",
                "effective_inner_tolerance",
            ):
                item = validation.get(name)
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or float(item) < 0.0
                ):
                    raise ValueError(f"comparison Pro-RM validation {name} is invalid")
        else:
            raise ValueError("comparison Pro-RM provenance mode is invalid")
    component_provenance = value.get("component_provenance")
    if component_provenance is not None:
        expected_components = {
            "reward_fit": {"MLE-RM", "Pro-RM"},
            "reward_evaluation": {"MLE-RM", "Pro-RM"},
            "natural_direction": {"mle_rm", "pro_rm", "oracle"},
            "local_policy_evaluation": {"pi0", "mle_rm", "pro_rm", "oracle"},
        }
        if not isinstance(component_provenance, dict) or set(component_provenance) != set(
            expected_components
        ):
            raise ValueError("comparison component provenance is incomplete")
        for substage, expected_methods in expected_components.items():
            records = component_provenance[substage]
            if not isinstance(records, dict) or set(records) != expected_methods:
                raise ValueError(f"comparison {substage} provenance is incomplete")
            for method, record in records.items():
                _validate_component_provenance_record(
                    record,
                    name=f"comparison {substage}.{method}",
                )
    for method in ("MLE-RM", "Pro-RM"):
        record = value["methods"][method]
        if not isinstance(record, dict) or not record.get("converged"):
            raise ValueError(f"{method} fit did not converge")
        weights = record.get("head_weight")
        if not isinstance(weights, list) or not weights:
            raise ValueError(f"{method} head_weight is missing")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in weights):
            raise ValueError(f"{method} head_weight must contain numbers")
        if any(not math.isfinite(float(item)) for item in weights):
            raise ValueError(f"{method} head_weight must be finite")
        _validate_sha256(record.get("head_sha256"), name=f"{method} head_sha256")
    policy_dimension = value.get("dimensions", {}).get("policy_tangent")
    if isinstance(policy_dimension, bool) or not isinstance(policy_dimension, int):
        raise ValueError("comparison policy tangent dimension is missing")
    directions = value.get("policy_directions")
    if not isinstance(directions, dict) or set(directions) != {"mle_rm", "pro_rm", "oracle"}:
        raise ValueError("comparison policy directions are incomplete")
    for method, direction in directions.items():
        if not isinstance(direction, list) or len(direction) != policy_dimension:
            raise ValueError(f"{method} policy direction has the wrong dimension")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in direction
        ):
            raise ValueError(f"{method} policy direction must contain finite numbers")
    return value


__all__ = [
    "EXACT_COMPARISON_SCHEMA",
    "load_exact_reward_comparison",
    "run_exact_reward_comparison",
]

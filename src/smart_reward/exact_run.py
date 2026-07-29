"""Train and evaluate the two frozen exact-delta reward learners."""

from __future__ import annotations

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
)
from .exact import (
    ExactDeltaExperiment,
    ExactSplitData,
    MLETrainingConfig,
    ProTrainingConfig,
    evaluate_reward_head,
    fit_mle_reward,
    fit_pro_reward,
)

EXACT_COMPARISON_SCHEMA = "exact-delta-reward-comparison/v1"


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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


def run_exact_reward_comparison(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
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
    pro = fit_pro_reward(
        experiment.train,
        ProTrainingConfig(
            relative_damping=float(geometry_config["damping_relative_to_mean_diagonal"]),
            fisher_estimator=geometry_config["fisher_estimator"],
            inner_max_iterations=int(geometry_config["cg_max_iterations"]),
            inner_tolerance=float(geometry_config["cg_tolerance"]),
            outer_max_iterations=int(pro_config["max_iterations"]),
            outer_tolerance=float(pro_config["tolerance"]),
            residual_recompute_interval=int(pro_config["residual_recompute_interval"]),
        ),
    )
    fits = {"MLE-RM": mle, "Pro-RM": pro}
    # Held-out targets are first read after both train-only fits have finished.
    evaluations = {
        method: {
            split_name: evaluate_reward_head(getattr(experiment, split_name), fit.weight).to_dict()
            for split_name in ("train", "validation", "test")
        }
        for method, fit in fits.items()
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
    train_directions = {
        method: solve_natural_direction(experiment.train, rewards, settings)
        for method, rewards in train_rewards.items()
    }
    local_policy_evaluation = {
        str(beta): {
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
        for beta in normalized["policy_update"]["beta_grid"]
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
    }
    _atomic_json(Path(output), result)
    return result


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

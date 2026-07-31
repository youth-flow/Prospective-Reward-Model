"""Reward fitting and one-step Fisher-corrected TRPO update construction."""

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

from .artifacts import exact_delta_artifact_metadata_sha256, load_exact_delta_artifact
from .config import TRPO_PROTOCOL, config_hash, validate_config
from .evaluation import (
    GeometrySettings,
    evaluate_trpo_local_policy,
    evaluate_trpo_reference_policy,
    solve_natural_direction,
)
from .exact import (
    ExactDeltaExperiment,
    ExactSplitData,
    MLETrainingConfig,
    ProTrainingConfig,
    empirical_fisher_score_rows,
    evaluate_reward_head,
    fit_mle_reward,
    fit_pro_reward,
    validate_mle_reward_fit,
)
from .exact_run import load_exact_reward_comparison
from .fisher_crossfit import load_fisher_selection
from .linear import DampedEmpiricalFisher
from .policy_update import scale_direction_to_quadratic_kl
from .runtime import producer_identity, sha256_file

SCHEMA = "prorm-fisher-trpo-reward-comparison/v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
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
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _component_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_device(split: ExactSplitData, device: torch.device) -> ExactSplitData:
    return ExactSplitData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores.to(device),
        reward_features=split.reward_features.to(device),
        true_rewards=split.true_rewards.to(device),
    )


def _experiment_to_device(
    experiment: ExactDeltaExperiment, device: torch.device
) -> ExactDeltaExperiment:
    return ExactDeltaExperiment(
        train=_to_device(experiment.train, device),
        validation=_to_device(experiment.validation, device),
        test=_to_device(experiment.test, device),
    )


def _mle_config(config: Mapping[str, Any]) -> MLETrainingConfig:
    mle = config["reward_model"]["mle"]
    return MLETrainingConfig(
        max_iterations=int(mle["max_iterations"]),
        history_size=int(mle["history_size"]),
        gradient_tolerance=float(mle["gradient_tolerance"]),
        change_tolerance=float(mle["change_tolerance"]),
        microbatch_size=int(mle["microbatch_size"]),
    )


def _reuse_mle(
    source_path: Path,
    train: ExactSplitData,
    config: MLETrainingConfig,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    source = load_exact_reward_comparison(source_path, expected_seed=seed)
    record = source["methods"]["MLE-RM"]
    weight = torch.tensor(
        record["head_weight"],
        device=train.reward_features.device,
        dtype=torch.float64,
    )
    validated = validate_mle_reward_fit(
        train,
        weight,
        config,
        source_iterations=int(record["iterations"]),
    )
    if validated.head_sha256 != record["head_sha256"]:
        raise ValueError("reused MLE-RM head digest changed during validation")
    producer = source.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("reused MLE-RM source producer is missing")
    provenance = {
        "mode": "validated_reuse",
        "source_result_sha256": sha256_file(source_path),
        "source_component_sha256": _component_sha256(record),
        "source_producer": dict(producer),
        "validation": {
            "head_sha256": validated.head_sha256,
            "objective": validated.objective,
            "gradient_norm": validated.gradient_norm,
        },
    }
    return validated, provenance


def run_trpo_reward_comparison(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    fisher_selection: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    reuse_mle_from: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Fit affected reward components and construct all matched-KL updates."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError(f"TRPO reward comparison requires protocol {TRPO_PROTOCOL}")
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
        raise RuntimeError("CUDA was requested but is unavailable")
    experiment = _experiment_to_device(experiment, target_device)
    selection_path = Path(fisher_selection)
    selection = load_fisher_selection(
        selection_path,
        expected_config_sha256=digest,
    )
    relative_damping = float(selection["selected_relative_damping"])

    mle_config = _mle_config(normalized)
    if reuse_mle_from is None:
        print("reward_fit method=MLE-RM status=running", flush=True)
        mle = fit_mle_reward(experiment.train, mle_config)
        mle_provenance: dict[str, Any] = {"mode": "computed"}
    else:
        print("reward_fit method=MLE-RM status=validating-reuse", flush=True)
        mle, mle_provenance = _reuse_mle(
            Path(reuse_mle_from),
            experiment.train,
            mle_config,
            seed=seed,
        )
    if not mle.converged:
        raise RuntimeError("MLE-RM did not pass its convergence gate")

    geometry = normalized["geometry"]
    pro_config_raw = normalized["reward_model"]["pro"]
    pro_config = ProTrainingConfig(
        relative_damping=relative_damping,
        fisher_estimator=geometry["fisher_estimator"],
        inner_max_iterations=int(geometry["cg_max_iterations"]),
        inner_tolerance=float(geometry["cg_tolerance"]),
        outer_max_iterations=int(pro_config_raw["max_iterations"]),
        outer_tolerance=float(pro_config_raw["tolerance"]),
        residual_recompute_interval=int(pro_config_raw["residual_recompute_interval"]),
    )
    print("reward_fit method=Pro-RM status=running", flush=True)
    pro = fit_pro_reward(experiment.train, pro_config)
    if not pro.converged:
        raise RuntimeError("Pro-RM did not pass its convergence gate")

    fits = {"MLE-RM": mle, "Pro-RM": pro}
    evaluations = {
        method: {
            split_name: evaluate_reward_head(
                getattr(experiment, split_name),
                fit.weight,
            ).to_dict()
            for split_name in ("train", "validation", "test")
        }
        for method, fit in fits.items()
    }
    settings = GeometrySettings(
        fisher_estimator=geometry["fisher_estimator"],
        relative_damping=relative_damping,
        cg_tolerance=float(geometry["cg_tolerance"]),
        cg_max_iterations=int(geometry["cg_max_iterations"]),
        residual_recompute_interval=int(geometry["residual_recompute_interval"]),
    )
    train_rewards = {
        "mle_rm": experiment.train.reward_features.to(dtype=torch.float64) @ mle.weight,
        "pro_rm": experiment.train.reward_features.to(dtype=torch.float64) @ pro.weight,
        "oracle": experiment.train.true_rewards.to(dtype=torch.float64),
    }
    directions = {
        method: solve_natural_direction(experiment.train, rewards, settings)
        for method, rewards in train_rewards.items()
    }
    raw_rows = empirical_fisher_score_rows(
        experiment.train.policy_scores.to(dtype=torch.float64),
        "raw_second_moment",
    )
    raw_fisher = DampedEmpiricalFisher(raw_rows, damping=0.0)
    policy_updates: dict[str, Any] = {}
    local_evaluation: dict[str, Any] = {}
    for method, direction in directions.items():
        policy_updates[method] = {}
        for target_raw in normalized["policy_update"]["kl_targets"]:
            target = float(target_raw)
            key = str(target)
            update, scale, realized = scale_direction_to_quadratic_kl(
                direction,
                raw_fisher.matvec,
                kl_target=target,
            )
            policy_updates[method][key] = {
                "update": update.detach().cpu().tolist(),
                "step_scale": scale,
                "train_quadratic_forward_kl": realized,
                "update_norm": float(torch.linalg.vector_norm(update).item()),
            }
            local_evaluation.setdefault(key, {})[method] = evaluate_trpo_local_policy(
                experiment.test,
                update,
                kl_target=target,
                settings=settings,
            ).to_dict()
    for target_raw in normalized["policy_update"]["kl_targets"]:
        target = float(target_raw)
        local_evaluation[str(target)]["pi0"] = evaluate_trpo_reference_policy(
            experiment.test,
            kl_target=target,
            settings=settings,
        ).to_dict()

    payload = {
        "schema": SCHEMA,
        "protocol": TRPO_PROTOCOL,
        "config_sha256": digest,
        "artifact_metadata_sha256": artifact_identity,
        "fisher_selection_sha256": sha256_file(selection_path),
        "selected_relative_damping": relative_damping,
        "seed": seed,
        "validation_usage": "crossfit_and_kl_calibration_only",
        "training_target": "exact_delta_r_star",
        "fit_provenance": {
            "MLE-RM": mle_provenance,
            "Pro-RM": {"mode": "computed"},
        },
        "methods": {method: fit.to_dict() for method, fit in fits.items()},
        "evaluation": evaluations,
        "policy_directions": {
            method: direction.detach().cpu().tolist() for method, direction in directions.items()
        },
        "policy_updates": policy_updates,
        "local_policy_evaluation": local_evaluation,
        "dimensions": {
            "train_prompts": experiment.train.num_prompts,
            "candidates_per_prompt": experiment.train.num_candidates,
            "train_nodes": experiment.train.num_prompts * experiment.train.num_candidates,
            "train_edges": experiment.train.num_edges,
            "reward_head": experiment.train.reward_dimension,
            "policy_tangent": experiment.train.policy_dimension,
        },
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def load_trpo_reward_comparison(
    path: str | os.PathLike[str],
    *,
    expected_config_sha256: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("protocol") != TRPO_PROTOCOL
    ):
        raise ValueError("unsupported Fisher-TRPO reward comparison")
    if expected_config_sha256 is not None and value.get("config_sha256") != (
        expected_config_sha256
    ):
        raise ValueError("TRPO reward comparison config mismatch")
    if expected_seed is not None and value.get("seed") != expected_seed:
        raise ValueError("TRPO reward comparison seed mismatch")
    selected = value.get("selected_relative_damping")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, (int, float))
        or not math.isfinite(float(selected))
        or float(selected) <= 0.0
    ):
        raise ValueError("TRPO reward comparison damping is invalid")
    methods = value.get("methods")
    if not isinstance(methods, dict) or set(methods) != {"MLE-RM", "Pro-RM"}:
        raise ValueError("TRPO reward methods are incomplete")
    updates = value.get("policy_updates")
    if not isinstance(updates, dict) or set(updates) != {"mle_rm", "pro_rm", "oracle"}:
        raise ValueError("TRPO policy updates are incomplete")
    tangent = value.get("dimensions", {}).get("policy_tangent")
    if isinstance(tangent, bool) or not isinstance(tangent, int) or tangent < 1:
        raise ValueError("TRPO policy tangent dimension is missing")
    for method_updates in updates.values():
        if not isinstance(method_updates, dict):
            raise ValueError("TRPO method updates are malformed")
        for record in method_updates.values():
            vector = record.get("update") if isinstance(record, dict) else None
            if (
                not isinstance(vector, list)
                or len(vector) != tangent
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in vector
                )
            ):
                raise ValueError("TRPO update vector is invalid")
    return value


__all__ = ["SCHEMA", "load_trpo_reward_comparison", "run_trpo_reward_comparison"]

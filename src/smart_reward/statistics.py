"""Three-seed descriptive aggregation for the main experiment."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PROTOCOL, config_hash, validate_config
from .runtime import producer_identity, sha256_file


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("aggregate values must be non-empty and finite")
    return {
        "per_seed": list(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "num_seeds": len(values),
    }


def _paired_difference(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired metric vectors must have equal length")
    return _summary([first - second for first, second in zip(left, right, strict=True)])


def aggregate_results(
    config: Mapping[str, object],
    reward_result_paths: Sequence[str | Path],
    rollout_result_paths: Sequence[str | Path],
) -> dict[str, Any]:
    normalized = validate_config(config)
    digest = config_hash(normalized)
    expected_seeds = list(normalized["run"]["seeds"])
    reward_payloads = [_read(path) for path in reward_result_paths]
    rollout_payloads = [_read(path) for path in rollout_result_paths]
    rewards = {payload["seed"]: payload for payload in reward_payloads}
    rollouts = {payload["seed"]: payload for payload in rollout_payloads}
    reward_paths = {
        payload["seed"]: Path(path)
        for payload, path in zip(reward_payloads, reward_result_paths, strict=True)
    }
    rollout_paths = {
        payload["seed"]: Path(path)
        for payload, path in zip(rollout_payloads, rollout_result_paths, strict=True)
    }
    if len(rewards) != len(reward_payloads) or len(rollouts) != len(rollout_payloads):
        raise ValueError("aggregate inputs contain duplicate seeds")
    if sorted(rewards) != sorted(expected_seeds) or sorted(rollouts) != sorted(expected_seeds):
        raise ValueError("reward and rollout inputs must cover every configured seed exactly once")
    for payload in (*rewards.values(), *rollouts.values()):
        if payload.get("config_sha256") != digest or payload.get("protocol") != PROTOCOL:
            raise ValueError("aggregate input protocol/config mismatch")
        if payload.get("producer") != producer_identity():
            raise ValueError("aggregate input producer identity mismatch")
    reward_metrics: dict[str, Any] = {}
    for method in ("MLE-RM", "Pro-RM"):
        reward_metrics[method] = {
            metric: _summary(
                [
                    float(rewards[seed]["evaluation"][method]["test"][metric])
                    for seed in expected_seeds
                ]
            )
            for metric in normalized["evaluation"]["reward_fit_metrics"]
        }
    reward_paired = {
        metric: _paired_difference(
            reward_metrics["Pro-RM"][metric]["per_seed"],
            reward_metrics["MLE-RM"][metric]["per_seed"],
        )
        for metric in normalized["evaluation"]["reward_fit_metrics"]
    }
    local: dict[str, Any] = {}
    rollout: dict[str, Any] = {}
    local_paired: dict[str, Any] = {}
    rollout_paired: dict[str, Any] = {}
    for beta in normalized["policy_update"]["beta_grid"]:
        key = str(float(beta))
        local[key] = {
            method: {
                metric: _summary(
                    [
                        float(rewards[seed]["local_policy_evaluation"][key][method][metric])
                        for seed in expected_seeds
                    ]
                )
                for metric in normalized["evaluation"]["local_policy_metrics"]
            }
            for method in ("pi0", "mle_rm", "pro_rm", "oracle")
        }
        rollout[key] = {
            method: {
                metric: _summary(
                    [
                        float(rollouts[seed]["metrics"][key][method][metric])
                        for seed in expected_seeds
                    ]
                )
                for metric in normalized["evaluation"]["rollout"]["metrics"]
            }
            for method in ("pi0", "mle_rm", "pro_rm", "oracle")
        }
        local_paired[key] = {
            metric: _paired_difference(
                local[key]["pro_rm"][metric]["per_seed"],
                local[key]["mle_rm"][metric]["per_seed"],
            )
            for metric in normalized["evaluation"]["local_policy_metrics"]
        }
        rollout_paired[key] = {
            metric: _paired_difference(
                rollout[key]["pro_rm"][metric]["per_seed"],
                rollout[key]["mle_rm"][metric]["per_seed"],
            )
            for metric in normalized["evaluation"]["rollout"]["metrics"]
        }
    return {
        "schema": "prorm-three-seed-aggregate/v1",
        "protocol": PROTOCOL,
        "config_sha256": digest,
        "seeds": expected_seeds,
        "reward_fit": reward_metrics,
        "reward_fit_pro_minus_mle": reward_paired,
        "local_policy": local,
        "local_policy_pro_minus_mle": local_paired,
        "rollout_policy": rollout,
        "rollout_policy_pro_minus_mle": rollout_paired,
        "inference_scope": "descriptive_three_seed_experiment",
        "producer": producer_identity(),
        "inputs": {
            "reward_results": {
                str(seed): sha256_file(reward_paths[seed]) for seed in expected_seeds
            },
            "rollout_results": {
                str(seed): sha256_file(rollout_paths[seed]) for seed in expected_seeds
            },
        },
    }


__all__ = ["aggregate_results"]

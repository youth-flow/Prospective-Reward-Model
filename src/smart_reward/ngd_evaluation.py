"""Fisher-corrected common-beta NGD evaluation on the frozen candidate pool."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .artifacts import exact_delta_artifact_metadata_sha256, load_exact_delta_artifact
from .config import TRPO_PROTOCOL, config_hash, validate_config
from .evaluation import GeometrySettings
from .exact import (
    ExactSplitData,
    empirical_fisher_score_rows,
    evaluate_reward_head,
    policy_reward_moment,
)
from .linear import DampedEmpiricalFisher
from .pcg import pcg
from .runtime import producer_identity, sha256_file, validate_seed
from .trpo_run import load_trpo_reward_comparison

PROTOCOL = "prorm_fisher_corrected_common_beta_ngd_v3"
SEED_SCHEMA = "prorm-fisher-corrected-ngd-evaluation/v3"
AGGREGATE_SCHEMA = "prorm-fisher-corrected-ngd-aggregate/v3"
AUDIT_SCHEMA = "prorm-fisher-corrected-ngd-integrity-audit/v3"
BETAS = (0.1, 0.5, 1.0, 2.0, 4.0)
POLICIES = ("pi0", "mle", "pro", "oracle", "tabular")
_DIRECTION_KEYS = {"mle": "mle_rm", "pro": "pro_rm", "oracle": "oracle"}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _geometry_settings(config: Mapping[str, Any], relative_damping: float) -> GeometrySettings:
    geometry = config["geometry"]
    return GeometrySettings(
        fisher_estimator=str(geometry["fisher_estimator"]),
        relative_damping=relative_damping,
        cg_tolerance=float(geometry["cg_tolerance"]),
        cg_max_iterations=int(geometry["cg_max_iterations"]),
        residual_recompute_interval=int(geometry["residual_recompute_interval"]),
    )


def _direction(
    record: Mapping[str, Any], key: str, dimension: int, device: torch.device
) -> torch.Tensor:
    raw = record.get(key)
    if not isinstance(raw, list) or len(raw) != dimension:
        raise ValueError(f"source policy direction {key!r} has the wrong dimension")
    result = torch.tensor(raw, dtype=torch.float64, device=device)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"source policy direction {key!r} is not finite")
    return result


def _centered_reward_mse(split: ExactSplitData, weight: torch.Tensor) -> float:
    predicted = split.reward_features.to(dtype=torch.float64) @ weight.to(
        device=split.reward_features.device, dtype=torch.float64
    )
    target = split.true_rewards.to(dtype=torch.float64)
    predicted = predicted - predicted.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    return float((predicted - target).square().mean().item())


def _policy_metrics(
    probabilities: torch.Tensor,
    log_probabilities: torch.Tensor,
    log_tabular_probabilities: torch.Tensor,
    rewards: torch.Tensor,
    *,
    beta: float,
    j_tabular: float,
) -> dict[str, float]:
    candidates = probabilities.shape[1]
    log_reference = -math.log(candidates)
    reward = (probabilities * rewards).sum(dim=1).mean()
    kl = (probabilities * (log_probabilities - log_reference)).sum(dim=1).mean()
    objective = reward - beta * kl
    kl_to_tabular = (
        probabilities * (log_probabilities - log_tabular_probabilities)
    ).sum(dim=1).mean()
    return {
        "R": float(reward.item()),
        "K": float(kl.item()),
        "J": float(objective.item()),
        "delta_J": float(j_tabular - objective.item()),
        "beta_KL": float(beta * kl_to_tabular.item()),
    }


@torch.no_grad()
def evaluate_candidate_pool(
    split: ExactSplitData,
    directions: Mapping[str, torch.Tensor],
    *,
    beta: float,
) -> dict[str, Any]:
    """Evaluate the five frozen candidate-pool policies for one common beta."""

    if float(beta) not in BETAS:
        raise ValueError(f"beta must be one of the frozen values {BETAS!r}")
    scores = split.policy_scores.to(dtype=torch.float64)
    rewards = split.true_rewards.to(dtype=torch.float64)
    candidates = split.num_candidates
    reference = torch.full_like(rewards, 1.0 / candidates)
    tabular_log_probabilities = torch.log_softmax(rewards / beta, dim=1)
    tabular = tabular_log_probabilities.exp()
    policy_probabilities: dict[str, torch.Tensor] = {"pi0": reference}
    policy_log_probabilities: dict[str, torch.Tensor] = {
        "pi0": torch.full_like(rewards, -math.log(candidates))
    }
    for policy, direction_key in _DIRECTION_KEYS.items():
        direction = directions[direction_key].to(device=scores.device, dtype=torch.float64)
        logits = torch.einsum("pmd,d->pm", scores, direction) / beta
        policy_log_probabilities[policy] = torch.log_softmax(logits, dim=1)
        policy_probabilities[policy] = policy_log_probabilities[policy].exp()
    policy_probabilities["tabular"] = tabular
    policy_log_probabilities["tabular"] = tabular_log_probabilities
    j_close_tensor = beta * (
        torch.logsumexp(rewards / beta, dim=1) - math.log(candidates)
    ).mean()
    j_close = float(j_close_tensor.item())
    metrics = {
        policy: _policy_metrics(
            policy_probabilities[policy],
            policy_log_probabilities[policy],
            tabular_log_probabilities,
            rewards,
            beta=beta,
            j_tabular=j_close,
        )
        for policy in POLICIES
    }
    j_identity = abs(metrics["tabular"]["J"] - j_close)
    gibbs_identity = max(
        abs(record["delta_J"] - record["beta_KL"]) for record in metrics.values()
    )
    scale = 1.0 + abs(j_close)
    if j_identity > 1.0e-10 * scale or gibbs_identity > 1.0e-10 * scale:
        raise RuntimeError(
            "candidate-pool Gibbs identity failed: "
            f"J residual={j_identity:.3e}, gap residual={gibbs_identity:.3e}"
        )
    return {
        "beta": beta,
        "J_close": j_close,
        "policies": metrics,
        "identity_residuals": {
            "abs_J_tabular_minus_J_close": j_identity,
            "max_abs_delta_J_minus_beta_KL": gibbs_identity,
        },
    }


def run_ngd_evaluation(
    config: Mapping[str, Any],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Reuse validated Fisher/TRPO ancestors and evaluate common-beta NGD."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError(f"source config must use protocol {TRPO_PROTOCOL!r}")
    seed = validate_seed(seed)
    if seed not in normalized["run"]["seeds"]:
        raise ValueError("seed is not declared in the source config")
    source_config_sha256 = config_hash(normalized)
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=source_config_sha256,
        expected_seed=seed,
    )
    source_path = Path(reward_result)
    source = load_trpo_reward_comparison(
        source_path,
        expected_config_sha256=source_config_sha256,
        expected_seed=seed,
    )
    if source.get("artifact_metadata_sha256") != artifact_identity:
        raise ValueError("source reward result and materialized artifact identities differ")
    relative_damping = float(source["selected_relative_damping"])
    expected_candidates = [
        float(value)
        for value in normalized["geometry"]["damping_selection"]["relative_candidates"]
    ]
    if relative_damping not in expected_candidates:
        raise ValueError("selected Fisher damping is not a declared cross-fit candidate")

    target_device = torch.device(device)
    experiment = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=source_config_sha256,
        expected_seed=seed,
    )
    train = ExactSplitData(
        prompt_ids=experiment.train.prompt_ids,
        policy_scores=experiment.train.policy_scores.to(target_device),
        reward_features=experiment.train.reward_features.to(target_device),
        true_rewards=experiment.train.true_rewards.to(target_device),
    )
    test = ExactSplitData(
        prompt_ids=experiment.test.prompt_ids,
        policy_scores=experiment.test.policy_scores.to(target_device),
        reward_features=experiment.test.reward_features.to(target_device),
        true_rewards=experiment.test.true_rewards.to(target_device),
    )
    dimension = test.policy_dimension
    raw_directions = source.get("policy_directions")
    if not isinstance(raw_directions, Mapping) or set(raw_directions) != set(
        _DIRECTION_KEYS.values()
    ):
        raise ValueError("source reward result does not contain exactly three policy directions")
    directions = {
        key: _direction(raw_directions, key, dimension, target_device)
        for key in _DIRECTION_KEYS.values()
    }

    settings = _geometry_settings(normalized, relative_damping)
    rows = empirical_fisher_score_rows(
        train.policy_scores.to(dtype=torch.float64), settings.fisher_estimator
    )
    raw_fisher = DampedEmpiricalFisher(rows, damping=0.0)
    damping = relative_damping * float(raw_fisher.diagonal().mean().item())
    damped_fisher = DampedEmpiricalFisher(rows, damping=damping)

    policy_metrics = {
        str(beta): evaluate_candidate_pool(test, directions, beta=beta) for beta in BETAS
    }
    oracle_test_moment = policy_reward_moment(
        test.policy_scores.to(dtype=torch.float64),
        test.true_rewards.to(dtype=torch.float64),
    )
    reward_metrics: dict[str, dict[str, Any]] = {}
    for label, source_method in (
        ("mle", "MLE-RM"),
        ("pro", "Pro-RM"),
    ):
        method = source["methods"][source_method]
        weight = torch.tensor(method["head_weight"], dtype=torch.float64, device=target_device)
        evaluation = evaluate_reward_head(test, weight)
        predicted_test_rewards = test.reward_features.to(dtype=torch.float64) @ weight
        predicted_test_moment = policy_reward_moment(
            test.policy_scores.to(dtype=torch.float64), predicted_test_rewards
        )
        moment_error = predicted_test_moment - oracle_test_moment
        solve = pcg(
            damped_fisher.matvec,
            moment_error,
            inverse_diagonal=damped_fisher.pcg_inverse_diagonal(),
            max_iterations=settings.cg_max_iterations,
            tolerance=settings.cg_tolerance,
            residual_recompute_interval=settings.residual_recompute_interval,
        )
        if not solve.converged:
            raise RuntimeError(
                f"{label} test moment-error Fisher solve did not converge: "
                f"iterations={solve.iterations}, residual={solve.relative_residual:.3e}"
            )
        quadratic = float(torch.dot(moment_error, solve.solution).item())
        if quadratic < -1.0e-10:
            raise RuntimeError("inverse-Fisher moment error produced a negative quadratic form")
        approximate_regret = {
            str(beta): max(0.0, quadratic) / (2.0 * beta) for beta in BETAS
        }
        exact_regret = {
            str(beta): float(policy_metrics[str(beta)]["policies"][label]["delta_J"])
            for beta in BETAS
        }
        reward_metrics[label] = {
            "NLL": evaluation.soft_btl_nll,
            "MSE": _centered_reward_mse(test, weight),
            "approximate_regret": approximate_regret,
            "exact_regret": exact_regret,
            "approximation_gap": {
                str(beta): exact_regret[str(beta)] - approximate_regret[str(beta)]
                for beta in BETAS
            },
            "moment_error_inverse_fisher_quadratic": max(0.0, quadratic),
            "moment_error_pcg_relative_residual": solve.relative_residual,
        }
    payload: dict[str, Any] = {
        "schema": SEED_SCHEMA,
        "protocol": PROTOCOL,
        "seed": seed,
        "betas": list(BETAS),
        "test_usage": "formal_evaluation_only_no_hyperparameter_selection",
        "reference_policy": "uniform_over_frozen_candidate_pool",
        "fisher": {
            "estimator": settings.fisher_estimator,
            "selected_relative_damping": relative_damping,
            "absolute_train_damping": damping,
            "geometry_split": "train",
            "selection": "train_prompt_crossfit_reused_from_source_and_frozen",
        },
        "reward": reward_metrics,
        "policy": policy_metrics,
        "definitions": {
            "reward_MSE": "mean_{p,i}(((rhat_pi-mean_i rhat_pi)-(r*_pi-mean_i r*_pi))^2)",
            "approximate_regret": (
                "(A_rhat,test-A_r*,test)^T F_lambda,train^-1 "
                "(A_rhat,test-A_r*,test)/(2 beta)"
            ),
            "exact_regret": "J_tabular-J_m",
            "approximation_gap": "exact_regret-approximate_regret",
            "R": "mean_p sum_i pi(i|x_p) r*_pi",
            "K": "mean_p KL(pi(.|x_p) || pi0(.|x_p))",
            "J": "R-beta*K",
            "delta_J": "J_tabular-J",
            "beta_KL": "beta*mean_p KL(pi(.|x_p) || pi_tabular(.|x_p))",
            "J_close": "mean_p beta*log(sum_i pi0(i|x_p)*exp(r*_pi/beta))",
        },
        "provenance_bridge": {
            "source_protocol": source["protocol"],
            "source_config_sha256": source_config_sha256,
            "source_artifact_metadata_sha256": artifact_identity,
            "source_reward_result_sha256": sha256_file(source_path),
            "source_fisher_selection_sha256": source["fisher_selection_sha256"],
            "source_producer": source.get("producer", {}),
            "reuse_scope": [
                "fresh_test_materialization",
                "train_only_crossfit_fisher_selection",
                "reward_heads",
                "beta_free_natural_directions",
            ],
            "recomputed_scope": [
                "test_reward_moments",
                "train_fisher_inverse_moment_error",
                "reward_evaluation",
                "five_policy_candidate_pool_evaluation",
            ],
        },
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if len(values) < 2 or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("aggregate requires at least two finite values")
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
    }


def _summarize_tree(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        keys = set(first)
        if any(not isinstance(value, Mapping) or set(value) != keys for value in values):
            raise ValueError("seed result metric trees do not match")
        return {key: _summarize_tree([value[key] for value in values]) for key in sorted(keys)}
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return _numeric_summary([float(value) for value in values])
    if any(value != first for value in values[1:]):
        raise ValueError("non-numeric seed result fields do not match")
    return first


def aggregate_ngd_evaluations(
    config: Mapping[str, Any],
    result_paths: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Strictly aggregate exactly one common-beta result per declared seed."""

    normalized = validate_config(config)
    expected_seeds = list(normalized["run"]["seeds"])
    records_with_paths: list[tuple[dict[str, Any], Path]] = []
    for path in result_paths:
        result_path = Path(path)
        with result_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict) or value.get("schema") != SEED_SCHEMA:
            raise ValueError(f"unsupported NGD evaluation result: {path}")
        records_with_paths.append((value, result_path))
    records = [record for record, _ in records_with_paths]
    seeds = [record.get("seed") for record in records]
    if len(seeds) != len(set(seeds)) or sorted(seeds) != sorted(expected_seeds):
        raise ValueError("NGD results must cover every declared seed exactly once")
    records_with_paths.sort(key=lambda item: expected_seeds.index(item[0]["seed"]))
    records = [record for record, _ in records_with_paths]
    source_hash = config_hash(normalized)
    for record in records:
        if record.get("protocol") != PROTOCOL or record.get("betas") != list(BETAS):
            raise ValueError("NGD result protocol or beta grid mismatch")
        bridge = record.get("provenance_bridge", {})
        if bridge.get("source_config_sha256") != source_hash:
            raise ValueError("NGD result source config mismatch")

    policy_values = [record["policy"] for record in records]
    reward_values = [record["reward"] for record in records]
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "source_config_sha256": source_hash,
        "seeds": expected_seeds,
        "betas": list(BETAS),
        "reward": _summarize_tree(reward_values),
        "policy": _summarize_tree(policy_values),
        "input_sha256": {
            str(record["seed"]): sha256_file(Path(path))
            for record, path in records_with_paths
        },
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), aggregate)
    return aggregate


def audit_ngd_run(
    config: Mapping[str, Any],
    run_root: str | os.PathLike[str],
    source_run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Fail closed on coverage, identities, algebra, and immutable source hashes."""

    normalized = validate_config(config)
    expected_seeds = list(normalized["run"]["seeds"])
    source_hash = config_hash(normalized)
    root = Path(run_root)
    source_root = Path(source_run_root)
    aggregate_path = root / "aggregate.json"
    with aggregate_path.open("r", encoding="utf-8") as stream:
        aggregate = json.load(stream)
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("protocol") != PROTOCOL
        or aggregate.get("source_config_sha256") != source_hash
        or aggregate.get("seeds") != expected_seeds
        or aggregate.get("betas") != list(BETAS)
    ):
        raise ValueError("aggregate identity or coverage mismatch")

    checks: list[dict[str, Any]] = []
    for seed in expected_seeds:
        result_path = root / f"seed-{seed}" / "evaluation.json"
        if aggregate["input_sha256"].get(str(seed)) != sha256_file(result_path):
            raise ValueError(f"aggregate input hash mismatch for seed {seed}")
        with result_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if (
            result.get("schema") != SEED_SCHEMA
            or result.get("protocol") != PROTOCOL
            or result.get("seed") != seed
            or result.get("betas") != list(BETAS)
            or result.get("test_usage")
            != "formal_evaluation_only_no_hyperparameter_selection"
        ):
            raise ValueError(f"seed result identity or test-usage mismatch for seed {seed}")
        if set(result.get("reward", {})) != {"mle", "pro"}:
            raise ValueError(f"reward metric coverage mismatch for seed {seed}")
        for method in ("mle", "pro"):
            reward = result["reward"][method]
            scalars = [
                reward.get("NLL"),
                reward.get("MSE"),
                reward.get("moment_error_inverse_fisher_quadratic"),
                reward.get("moment_error_pcg_relative_residual"),
            ]
            approximate = reward.get("approximate_regret", {})
            exact = reward.get("exact_regret", {})
            gaps = reward.get("approximation_gap", {})
            beta_keys = {str(beta) for beta in BETAS}
            if set(approximate) != beta_keys or set(exact) != beta_keys or set(gaps) != beta_keys:
                raise ValueError(f"regret metric coverage mismatch for seed {seed}")
            scalars.extend(approximate.values())
            scalars.extend(exact.values())
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < -1.0e-12
                for value in scalars
            ):
                raise ValueError(f"invalid reward metric for seed {seed}")
            for beta in BETAS:
                key = str(beta)
                policy_exact = result["policy"][key]["policies"][method]["delta_J"]
                if abs(float(exact[key]) - float(policy_exact)) > 1.0e-12:
                    raise ValueError(f"exact-regret linkage failed for seed {seed}")
                expected_gap = float(exact[key]) - float(approximate[key])
                if abs(float(gaps[key]) - expected_gap) > 1.0e-12:
                    raise ValueError(f"approximation-gap identity failed for seed {seed}")
        policy = result.get("policy", {})
        if set(policy) != {str(beta) for beta in BETAS}:
            raise ValueError(f"policy beta coverage mismatch for seed {seed}")
        max_identity_residual = 0.0
        for beta in BETAS:
            beta_result = policy[str(beta)]
            if set(beta_result.get("policies", {})) != set(POLICIES):
                raise ValueError(f"five-policy coverage mismatch for seed {seed}, beta {beta}")
            for policy_name, metrics in beta_result["policies"].items():
                if set(metrics) != {"R", "K", "J", "delta_J", "beta_KL"}:
                    raise ValueError(
                        f"policy metric schema mismatch for seed {seed}, beta {beta}"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in metrics.values()
                ):
                    raise ValueError(f"non-finite policy metric for {policy_name}")
                if metrics["K"] < -1.0e-12 or metrics["delta_J"] < -1.0e-10:
                    raise ValueError(f"negative KL or objective gap for {policy_name}")
            if abs(beta_result["policies"]["pi0"]["K"]) > 1.0e-12:
                raise ValueError(f"pi0 KL is nonzero for seed {seed}, beta {beta}")
            residuals = beta_result.get("identity_residuals", {})
            if set(residuals) != {
                "abs_J_tabular_minus_J_close",
                "max_abs_delta_J_minus_beta_KL",
            }:
                raise ValueError(f"identity audit schema mismatch for seed {seed}")
            max_identity_residual = max(
                max_identity_residual, *(float(value) for value in residuals.values())
            )
            if max_identity_residual > 1.0e-9:
                raise ValueError(f"Gibbs identity failed for seed {seed}")

        bridge = result.get("provenance_bridge", {})
        source_artifact = source_root / f"seed-{seed}" / "artifact"
        source_reward = source_root / f"seed-{seed}" / "reward_result.json"
        if (
            bridge.get("source_config_sha256") != source_hash
            or bridge.get("source_artifact_metadata_sha256")
            != exact_delta_artifact_metadata_sha256(
                source_artifact,
                expected_config_hash=source_hash,
                expected_seed=seed,
            )
            or bridge.get("source_reward_result_sha256") != sha256_file(source_reward)
        ):
            raise ValueError(f"provenance bridge mismatch for seed {seed}")
        checks.append(
            {
                "seed": seed,
                "result_sha256": sha256_file(result_path),
                "source_artifact_metadata_sha256": bridge[
                    "source_artifact_metadata_sha256"
                ],
                "source_reward_result_sha256": bridge["source_reward_result_sha256"],
                "max_identity_residual": max_identity_residual,
                "status": "passed",
            }
        )

    payload = {
        "schema": AUDIT_SCHEMA,
        "protocol": PROTOCOL,
        "status": "passed",
        "source_config_sha256": source_hash,
        "aggregate_sha256": sha256_file(aggregate_path),
        "seeds": expected_seeds,
        "betas": list(BETAS),
        "checks": checks,
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


__all__ = [
    "AGGREGATE_SCHEMA",
    "AUDIT_SCHEMA",
    "BETAS",
    "POLICIES",
    "PROTOCOL",
    "SEED_SCHEMA",
    "aggregate_ngd_evaluations",
    "audit_ngd_run",
    "evaluate_candidate_pool",
    "run_ngd_evaluation",
]

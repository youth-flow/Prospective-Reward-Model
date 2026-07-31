"""Strict YAML configuration for the exact-delta ProRM experiment."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL = "prorm_exact_delta_v1"
TRPO_PROTOCOL = "prorm_fisher_trpo_v1"
_REVISION = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
_WALLTIME = re.compile(r"([0-9]+):([0-5][0-9]):([0-5][0-9])")


class ConfigError(ValueError):
    """The YAML file does not satisfy the experiment schema."""


def _map(value: object, path: str, required: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{path} must be a mapping with string keys")
    actual = set(value)
    if actual != required:
        raise ConfigError(
            f"{path} keys mismatch: missing={sorted(required - actual)!r}, "
            f"unknown={sorted(actual - required)!r}"
        )
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ConfigError(f"{path} must be {qualifier}")
    return result


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be boolean")
    return value


def _walltime(value: object, path: str) -> str:
    text = _string(value, path)
    match = _WALLTIME.fullmatch(text)
    if match is None or all(int(part) == 0 for part in match.groups()):
        raise ConfigError(f"{path} must be a positive HH:MM:SS walltime")
    return text


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ConfigError(f"{path} must be a sequence")
    if not value:
        raise ConfigError(f"{path} must not be empty")
    return value


def _unique_ints(value: object, path: str, minimum: int = 0) -> list[int]:
    result = [
        _integer(item, f"{path}[{index}]", minimum)
        for index, item in enumerate(_sequence(value, path))
    ]
    if len(result) != len(set(result)):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def _unique_strings(value: object, path: str) -> list[str]:
    result = [
        _string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))
    ]
    if len(result) != len(set(result)):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def _model(value: object, path: str) -> Mapping[str, object]:
    result = _map(value, path, {"model", "revision", "dtype"})
    _string(result["model"], f"{path}.model")
    revision = _string(result["revision"], f"{path}.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ConfigError(f"{path}.revision must be an immutable 40- or 64-character hash")
    if _string(result["dtype"], f"{path}.dtype") not in {"float32", "bfloat16"}:
        raise ConfigError(f"{path}.dtype must be float32 or bfloat16")
    return result


def _validate_legacy_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Return a detached validated configuration with no implicit defaults."""

    root = _map(
        config,
        "config",
        {
            "protocol",
            "run",
            "data",
            "policy",
            "oracle",
            "reward_model",
            "geometry",
            "policy_update",
            "evaluation",
            "execution",
        },
    )
    if _string(root["protocol"], "protocol") != PROTOCOL:
        raise ConfigError(f"protocol must equal {PROTOCOL!r}")

    run = _map(
        root["run"], "run", {"name", "seeds", "prompt_split_seed", "num_prompts", "split_sizes"}
    )
    _string(run["name"], "run.name")
    seeds = _unique_ints(run["seeds"], "run.seeds")
    if len(seeds) < 1:
        raise ConfigError("run.seeds must not be empty")
    _integer(run["prompt_split_seed"], "run.prompt_split_seed")
    total = _integer(run["num_prompts"], "run.num_prompts", 3)
    splits = _map(run["split_sizes"], "run.split_sizes", {"train", "validation", "test"})
    split_values = {name: _integer(splits[name], f"run.split_sizes.{name}", 1) for name in splits}
    if sum(split_values.values()) != total:
        raise ConfigError("run.split_sizes must sum to run.num_prompts")

    data = _map(
        root["data"],
        "data",
        {
            "prompt_dataset",
            "prompt_revision",
            "num_candidates",
            "edge_construction",
            "prompt_overlength",
        },
    )
    _string(data["prompt_dataset"], "data.prompt_dataset")
    if _REVISION.fullmatch(_string(data["prompt_revision"], "data.prompt_revision")) is None:
        raise ConfigError("data.prompt_revision must be immutable")
    _integer(data["num_candidates"], "data.num_candidates", 2)
    if data["edge_construction"] != "all_unordered_pairs":
        raise ConfigError("data.edge_construction must be all_unordered_pairs")
    if data["prompt_overlength"] != "exclude_before_sampling":
        raise ConfigError("data.prompt_overlength must be exclude_before_sampling")

    policy = _map(
        root["policy"],
        "policy",
        {
            "model",
            "revision",
            "dtype",
            "max_prompt_tokens",
            "max_response_tokens",
            "sampling",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_layers",
            "lora_modules",
            "trainable_tangent_parameters",
        },
    )
    _model({key: policy[key] for key in ("model", "revision", "dtype")}, "policy")
    _integer(policy["max_prompt_tokens"], "policy.max_prompt_tokens", 1)
    _integer(policy["max_response_tokens"], "policy.max_response_tokens", 1)
    sampling = _map(
        policy["sampling"],
        "policy.sampling",
        {"do_sample", "temperature", "top_p", "top_k", "min_new_tokens", "repetition_penalty"},
    )
    if not _bool(sampling["do_sample"], "policy.sampling.do_sample"):
        raise ConfigError("policy.sampling.do_sample must be true")
    _number(sampling["temperature"], "policy.sampling.temperature", positive=True)
    top_p = _number(sampling["top_p"], "policy.sampling.top_p", positive=True)
    if top_p > 1.0:
        raise ConfigError("policy.sampling.top_p must be <= 1")
    _integer(sampling["top_k"], "policy.sampling.top_k")
    _integer(sampling["min_new_tokens"], "policy.sampling.min_new_tokens")
    _number(sampling["repetition_penalty"], "policy.sampling.repetition_penalty", positive=True)
    _integer(policy["lora_rank"], "policy.lora_rank", 1)
    _number(policy["lora_alpha"], "policy.lora_alpha", positive=True)
    dropout = _number(policy["lora_dropout"], "policy.lora_dropout")
    if not 0.0 <= dropout < 1.0:
        raise ConfigError("policy.lora_dropout must lie in [0, 1)")
    _unique_ints(policy["lora_layers"], "policy.lora_layers")
    _unique_strings(policy["lora_modules"], "policy.lora_modules")
    if policy["trainable_tangent_parameters"] != "lora_B_only":
        raise ConfigError("policy.trainable_tangent_parameters must be lora_B_only")

    oracle = _map(
        root["oracle"],
        "oracle",
        {"model", "revision", "dtype", "transform", "robust_scale_floor", "batch_size"},
    )
    _model({key: oracle[key] for key in ("model", "revision", "dtype")}, "oracle")
    if oracle["transform"] != "train_median_scaled_mad_affine":
        raise ConfigError("oracle.transform must be train_median_scaled_mad_affine")
    _number(oracle["robust_scale_floor"], "oracle.robust_scale_floor", positive=True)
    _integer(oracle["batch_size"], "oracle.batch_size", 1)

    reward = _map(
        root["reward_model"],
        "reward_model",
        {
            "model",
            "revision",
            "dtype",
            "parameterization",
            "feature_pooling",
            "linear_head_bias",
            "mle",
            "pro",
        },
    )
    _model({key: reward[key] for key in ("model", "revision", "dtype")}, "reward_model")
    if (reward["model"], reward["revision"], reward["dtype"]) != (
        policy["model"],
        policy["revision"],
        policy["dtype"],
    ):
        raise ConfigError("reward_model must share the frozen policy backbone")
    if (
        reward["parameterization"] != "frozen_backbone_linear_head"
        or reward["feature_pooling"] != "last_response_token"
    ):
        raise ConfigError("reward_model must be a frozen-backbone last-response-token linear head")
    if _bool(reward["linear_head_bias"], "reward_model.linear_head_bias"):
        raise ConfigError("reward_model.linear_head_bias must be false")
    mle = _map(
        reward["mle"],
        "reward_model.mle",
        {
            "objective",
            "optimizer",
            "max_iterations",
            "history_size",
            "gradient_tolerance",
            "change_tolerance",
            "microbatch_size",
        },
    )
    if mle["objective"] != "exact_soft_btl_nll" or mle["optimizer"] != "lbfgs":
        raise ConfigError("MLE must use exact_soft_btl_nll with lbfgs")
    for key in ("max_iterations", "history_size", "microbatch_size"):
        _integer(mle[key], f"reward_model.mle.{key}", 1)
    for key in ("gradient_tolerance", "change_tolerance"):
        _number(mle[key], f"reward_model.mle.{key}", positive=True)
    pro = _map(
        reward["pro"],
        "reward_model.pro",
        {"objective", "solver", "max_iterations", "tolerance", "residual_recompute_interval"},
    )
    if (
        pro["objective"] != "fisher_weighted_reward_moment_error"
        or pro["solver"] != "nested_matrix_free_cg"
    ):
        raise ConfigError("Pro-RM must use the exact quadratic with nested_matrix_free_cg")
    _integer(pro["max_iterations"], "reward_model.pro.max_iterations", 1)
    _number(pro["tolerance"], "reward_model.pro.tolerance", positive=True)
    _integer(pro["residual_recompute_interval"], "reward_model.pro.residual_recompute_interval", 1)

    geometry = _map(
        root["geometry"],
        "geometry",
        {
            "fisher_estimator",
            "damping_relative_to_mean_diagonal",
            "solve_dtype",
            "cg_tolerance",
            "cg_max_iterations",
            "residual_recompute_interval",
        },
    )
    if geometry["fisher_estimator"] not in {
        "raw_second_moment",
        "prompt_centered_sample_covariance",
    }:
        raise ConfigError("geometry.fisher_estimator is unsupported")
    _number(
        geometry["damping_relative_to_mean_diagonal"],
        "geometry.damping_relative_to_mean_diagonal",
        positive=True,
    )
    if geometry["solve_dtype"] != "float64":
        raise ConfigError("geometry.solve_dtype must be float64")
    _number(geometry["cg_tolerance"], "geometry.cg_tolerance", positive=True)
    _integer(geometry["cg_max_iterations"], "geometry.cg_max_iterations", 1)
    _integer(geometry["residual_recompute_interval"], "geometry.residual_recompute_interval", 1)

    update = _map(
        root["policy_update"],
        "policy_update",
        {"method", "reward_sources", "beta_grid", "kl_orientation", "save_adapters"},
    )
    if update["method"] != "one_step_damped_ngd_lora_b":
        raise ConfigError("policy_update.method must be one_step_damped_ngd_lora_b")
    if _unique_strings(update["reward_sources"], "policy_update.reward_sources") != [
        "mle_rm",
        "pro_rm",
        "oracle",
    ]:
        raise ConfigError("policy_update.reward_sources must be [mle_rm, pro_rm, oracle]")
    betas = [
        _number(item, f"policy_update.beta_grid[{index}]", positive=True)
        for index, item in enumerate(_sequence(update["beta_grid"], "policy_update.beta_grid"))
    ]
    if len(betas) != len(set(betas)) or betas != sorted(betas):
        raise ConfigError("policy_update.beta_grid must be unique and increasing")
    if update["kl_orientation"] != "updated_to_reference":
        raise ConfigError("policy_update.kl_orientation must be updated_to_reference")
    _bool(update["save_adapters"], "policy_update.save_adapters")

    evaluation = _map(
        root["evaluation"],
        "evaluation",
        {"validation_usage", "reward_fit_metrics", "local_policy_metrics", "rollout"},
    )
    if evaluation["validation_usage"] != "diagnostics_only":
        raise ConfigError("evaluation.validation_usage must be diagnostics_only")
    expected_reward_metrics = [
        "pair_kl",
        "soft_btl_nll",
        "probability_mse",
        "pairwise_accuracy",
        "centered_reward_nmse",
    ]
    if (
        _unique_strings(evaluation["reward_fit_metrics"], "evaluation.reward_fit_metrics")
        != expected_reward_metrics
    ):
        raise ConfigError(f"evaluation.reward_fit_metrics must equal {expected_reward_metrics!r}")
    expected_local = [
        "local_regret",
        "fisher_cosine",
        "local_target_utility",
        "tabular_optimal_utility",
        "tabular_regret",
    ]
    if (
        _unique_strings(evaluation["local_policy_metrics"], "evaluation.local_policy_metrics")
        != expected_local
    ):
        raise ConfigError(f"evaluation.local_policy_metrics must equal {expected_local!r}")
    rollout = _map(
        evaluation["rollout"],
        "evaluation.rollout",
        {"prompts", "responses_per_prompt", "metrics"},
    )
    if _integer(rollout["prompts"], "evaluation.rollout.prompts", 1) > split_values["test"]:
        raise ConfigError("evaluation.rollout.prompts cannot exceed test split size")
    _integer(rollout["responses_per_prompt"], "evaluation.rollout.responses_per_prompt", 1)
    expected_rollout = [
        "oracle_reward",
        "reward_improvement",
        "forward_kl",
        "regularized_utility",
        "utility_improvement",
        "oracle_ngd_regret",
    ]
    if _unique_strings(rollout["metrics"], "evaluation.rollout.metrics") != expected_rollout:
        raise ConfigError(f"evaluation.rollout.metrics must equal {expected_rollout!r}")

    execution = _map(
        root["execution"],
        "execution",
        {
            "materialization_prompt_batch_size",
            "materialization_checkpoint_prompts",
            "rollout_prompt_batch_size",
            "rollout_checkpoint_prompts",
            "rollout_max_parallel_policies",
            "materialization_walltime",
            "reward_walltime",
            "adapter_walltime",
            "rollout_walltime",
            "rollout_aggregate_walltime",
            "three_seed_aggregate_walltime",
        },
    )
    materialization_batch = _integer(
        execution["materialization_prompt_batch_size"],
        "execution.materialization_prompt_batch_size",
        1,
    )
    materialization_checkpoint = _integer(
        execution["materialization_checkpoint_prompts"],
        "execution.materialization_checkpoint_prompts",
        materialization_batch,
    )
    rollout_batch = _integer(
        execution["rollout_prompt_batch_size"],
        "execution.rollout_prompt_batch_size",
        1,
    )
    rollout_checkpoint = _integer(
        execution["rollout_checkpoint_prompts"],
        "execution.rollout_checkpoint_prompts",
        rollout_batch,
    )
    if materialization_checkpoint % materialization_batch:
        raise ConfigError(
            "execution.materialization_checkpoint_prompts must be divisible by batch size"
        )
    if rollout_checkpoint % rollout_batch:
        raise ConfigError("execution.rollout_checkpoint_prompts must be divisible by batch size")
    if materialization_checkpoint > total:
        raise ConfigError("execution.materialization_checkpoint_prompts exceeds num_prompts")
    if rollout_checkpoint > int(rollout["prompts"]):
        raise ConfigError("execution.rollout_checkpoint_prompts exceeds rollout prompts")
    if (
        _integer(
            execution["rollout_max_parallel_policies"],
            "execution.rollout_max_parallel_policies",
            1,
        )
        > 10
    ):
        raise ConfigError("execution.rollout_max_parallel_policies cannot exceed 10")
    for name in (
        "materialization_walltime",
        "reward_walltime",
        "adapter_walltime",
        "rollout_walltime",
        "rollout_aggregate_walltime",
        "three_seed_aggregate_walltime",
    ):
        _walltime(execution[name], f"execution.{name}")
    return copy.deepcopy(dict(root))


def _validate_trpo_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Validate the Fisher-corrected, matched-KL TRPO protocol.

    The legacy validator remains the single source of truth for unchanged
    model, data, reward, and execution fields.  A synthetic legacy view is
    validated first, then every v2-only field is checked without defaults.
    """

    root = _map(
        config,
        "config",
        {
            "protocol",
            "run",
            "data",
            "policy",
            "oracle",
            "reward_model",
            "geometry",
            "policy_update",
            "evaluation",
            "execution",
        },
    )
    if _string(root["protocol"], "protocol") != TRPO_PROTOCOL:
        raise ConfigError(f"protocol must equal {TRPO_PROTOCOL!r}")

    run = _map(
        root["run"],
        "run",
        {
            "name",
            "seeds",
            "prompt_split_seed",
            "num_prompts",
            "split_sizes",
            "split_offsets",
        },
    )
    splits = _map(run["split_sizes"], "run.split_sizes", {"train", "validation", "test"})
    offsets = _map(run["split_offsets"], "run.split_offsets", {"train", "validation", "test"})
    split_sizes = {name: _integer(splits[name], f"run.split_sizes.{name}", 1) for name in splits}
    split_offsets = {name: _integer(offsets[name], f"run.split_offsets.{name}") for name in offsets}
    intervals = sorted(
        (
            split_offsets[name],
            split_offsets[name] + split_sizes[name],
            name,
        )
        for name in ("train", "validation", "test")
    )
    for interval_index in range(len(intervals) - 1):
        _, previous_end, previous_name = intervals[interval_index]
        next_start, _, next_name = intervals[interval_index + 1]
        if previous_end > next_start:
            raise ConfigError(f"run.split_offsets overlap: {previous_name!r} and {next_name!r}")
    if split_offsets["train"] != 0:
        raise ConfigError("run.split_offsets.train must be zero")
    if split_offsets["validation"] != split_sizes["train"]:
        raise ConfigError("validation must immediately follow the frozen train split")
    if split_offsets["test"] != int(run["num_prompts"]):
        raise ConfigError(
            "fresh test must start immediately after the complete legacy prompt inventory"
        )

    geometry = _map(
        root["geometry"],
        "geometry",
        {
            "fisher_estimator",
            "damping_selection",
            "solve_dtype",
            "cg_tolerance",
            "cg_max_iterations",
            "residual_recompute_interval",
        },
    )
    selection = _map(
        geometry["damping_selection"],
        "geometry.damping_selection",
        {
            "method",
            "folds",
            "relative_candidates",
            "selection_metric",
            "eligibility",
            "tie_break",
            "shared_across",
        },
    )
    if selection["method"] != "train_prompt_crossfit":
        raise ConfigError("geometry.damping_selection.method must be train_prompt_crossfit")
    _integer(selection["folds"], "geometry.damping_selection.folds", 2)
    candidates = [
        _number(
            item,
            f"geometry.damping_selection.relative_candidates[{index}]",
            positive=True,
        )
        for index, item in enumerate(
            _sequence(
                selection["relative_candidates"],
                "geometry.damping_selection.relative_candidates",
            )
        )
    ]
    if len(candidates) != len(set(candidates)) or candidates != sorted(candidates):
        raise ConfigError(
            "geometry.damping_selection.relative_candidates must be unique and increasing"
        )
    if selection["selection_metric"] != "heldout_oracle_reward_improvement":
        raise ConfigError(
            "geometry.damping_selection.selection_metric must be heldout_oracle_reward_improvement"
        )
    if selection["eligibility"] != "positive_mean_each_seed":
        raise ConfigError("geometry.damping_selection.eligibility must be positive_mean_each_seed")
    if selection["tie_break"] != "largest_in_best_one_standard_error_set":
        raise ConfigError(
            "geometry.damping_selection.tie_break must be largest_in_best_one_standard_error_set"
        )
    if _unique_strings(
        selection["shared_across"],
        "geometry.damping_selection.shared_across",
    ) != ["seeds", "reward_sources"]:
        raise ConfigError(
            "geometry.damping_selection.shared_across must be [seeds, reward_sources]"
        )

    update = _map(
        root["policy_update"],
        "policy_update",
        {
            "method",
            "reward_sources",
            "kl_targets",
            "primary_kl_target",
            "kl_orientation",
            "quadratic_scaling_fisher",
            "calibration",
            "save_adapters",
        },
    )
    if update["method"] != "one_step_damped_trpo_lora_b":
        raise ConfigError("policy_update.method must be one_step_damped_trpo_lora_b")
    if _unique_strings(update["reward_sources"], "policy_update.reward_sources") != [
        "mle_rm",
        "pro_rm",
        "oracle",
    ]:
        raise ConfigError("policy_update.reward_sources must be [mle_rm, pro_rm, oracle]")
    targets = [
        _number(item, f"policy_update.kl_targets[{index}]", positive=True)
        for index, item in enumerate(_sequence(update["kl_targets"], "policy_update.kl_targets"))
    ]
    if len(targets) != len(set(targets)) or targets != sorted(targets):
        raise ConfigError("policy_update.kl_targets must be unique and increasing")
    primary = _number(
        update["primary_kl_target"],
        "policy_update.primary_kl_target",
        positive=True,
    )
    if primary not in targets:
        raise ConfigError("policy_update.primary_kl_target must occur in kl_targets")
    if update["kl_orientation"] != "updated_to_reference":
        raise ConfigError("policy_update.kl_orientation must be updated_to_reference")
    if update["quadratic_scaling_fisher"] != "raw_undamped":
        raise ConfigError("policy_update.quadratic_scaling_fisher must be raw_undamped")
    calibration = _map(
        update["calibration"],
        "policy_update.calibration",
        {
            "split",
            "estimator",
            "max_attempts",
            "point_relative_interval",
            "confidence_level",
            "upper_confidence_multiplier",
            "responses_per_prompt",
            "confidence_interval",
            "search",
            "max_scale_change_per_attempt",
        },
    )
    if calibration["split"] != "validation":
        raise ConfigError("policy_update.calibration.split must be validation")
    if calibration["estimator"] != "rao_blackwellized_updated_policy_forward_kl":
        raise ConfigError(
            "policy_update.calibration.estimator must be "
            "rao_blackwellized_updated_policy_forward_kl"
        )
    _integer(calibration["max_attempts"], "policy_update.calibration.max_attempts", 1)
    interval = [
        _number(
            item,
            f"policy_update.calibration.point_relative_interval[{index}]",
            positive=True,
        )
        for index, item in enumerate(
            _sequence(
                calibration["point_relative_interval"],
                "policy_update.calibration.point_relative_interval",
            )
        )
    ]
    if len(interval) != 2 or not interval[0] < 1.0 < interval[1]:
        raise ConfigError("policy_update.calibration.point_relative_interval must straddle one")
    confidence = _number(
        calibration["confidence_level"],
        "policy_update.calibration.confidence_level",
        positive=True,
    )
    if confidence >= 1.0:
        raise ConfigError("policy_update.calibration.confidence_level must be < 1")
    if (
        _number(
            calibration["upper_confidence_multiplier"],
            "policy_update.calibration.upper_confidence_multiplier",
            positive=True,
        )
        <= interval[1]
    ):
        raise ConfigError(
            "policy_update.calibration.upper_confidence_multiplier must exceed "
            "the point interval upper bound"
        )
    _integer(
        calibration["responses_per_prompt"],
        "policy_update.calibration.responses_per_prompt",
        2,
    )
    if calibration["confidence_interval"] != "normal_prompt_clustered":
        raise ConfigError(
            "policy_update.calibration.confidence_interval must be normal_prompt_clustered"
        )
    if calibration["search"] != "deterministic_quadratic_ratio":
        raise ConfigError("policy_update.calibration.search must be deterministic_quadratic_ratio")
    if (
        _number(
            calibration["max_scale_change_per_attempt"],
            "policy_update.calibration.max_scale_change_per_attempt",
            positive=True,
        )
        <= 1.0
    ):
        raise ConfigError("policy_update.calibration.max_scale_change_per_attempt must exceed one")
    _bool(update["save_adapters"], "policy_update.save_adapters")

    evaluation = _map(
        root["evaluation"],
        "evaluation",
        {"validation_usage", "reward_fit_metrics", "local_policy_metrics", "rollout"},
    )
    if evaluation["validation_usage"] != "crossfit_and_kl_calibration_only":
        raise ConfigError("evaluation.validation_usage must be crossfit_and_kl_calibration_only")
    expected_local = [
        "fisher_cosine",
        "local_reward_improvement",
        "quadratic_forward_kl",
        "finite_pool_reward_improvement",
        "finite_pool_forward_kl",
    ]
    if (
        _unique_strings(evaluation["local_policy_metrics"], "evaluation.local_policy_metrics")
        != expected_local
    ):
        raise ConfigError(f"evaluation.local_policy_metrics must equal {expected_local!r}")
    rollout = _map(
        evaluation["rollout"],
        "evaluation.rollout",
        {"prompts", "responses_per_prompt", "metrics"},
    )
    if _integer(rollout["prompts"], "evaluation.rollout.prompts", 1) > split_sizes["test"]:
        raise ConfigError("evaluation.rollout.prompts cannot exceed test split size")
    _integer(rollout["responses_per_prompt"], "evaluation.rollout.responses_per_prompt", 1)
    expected_rollout = [
        "oracle_reward",
        "reward_improvement",
        "forward_kl",
        "kl_target_error",
    ]
    if _unique_strings(rollout["metrics"], "evaluation.rollout.metrics") != expected_rollout:
        raise ConfigError(f"evaluation.rollout.metrics must equal {expected_rollout!r}")

    execution = _map(
        root["execution"],
        "execution",
        {
            "materialization_prompt_batch_size",
            "materialization_checkpoint_prompts",
            "crossfit_walltime",
            "reward_walltime",
            "adapter_walltime",
            "kl_calibration_walltime",
            "rollout_prompt_batch_size",
            "rollout_checkpoint_prompts",
            "rollout_max_parallel_policies",
            "calibration_max_parallel_policies",
            "materialization_walltime",
            "rollout_walltime",
            "rollout_aggregate_walltime",
            "three_seed_aggregate_walltime",
        },
    )
    _walltime(execution["crossfit_walltime"], "execution.crossfit_walltime")
    _walltime(
        execution["kl_calibration_walltime"],
        "execution.kl_calibration_walltime",
    )
    if (
        _integer(
            execution["calibration_max_parallel_policies"],
            "execution.calibration_max_parallel_policies",
            1,
        )
        > 8
    ):
        raise ConfigError("execution.calibration_max_parallel_policies cannot exceed 8")
    if (split_sizes["train"] + split_sizes["validation"]) % int(
        execution["materialization_checkpoint_prompts"]
    ):
        raise ConfigError(
            "materialization checkpoints must align with the train/validation reuse boundary"
        )

    legacy = copy.deepcopy(dict(root))
    legacy["protocol"] = PROTOCOL
    legacy["run"] = {key: value for key, value in legacy["run"].items() if key != "split_offsets"}
    legacy["geometry"] = {
        "fisher_estimator": geometry["fisher_estimator"],
        "damping_relative_to_mean_diagonal": 1.0,
        "solve_dtype": geometry["solve_dtype"],
        "cg_tolerance": geometry["cg_tolerance"],
        "cg_max_iterations": geometry["cg_max_iterations"],
        "residual_recompute_interval": geometry["residual_recompute_interval"],
    }
    legacy["policy_update"] = {
        "method": "one_step_damped_ngd_lora_b",
        "reward_sources": list(update["reward_sources"]),
        "beta_grid": [1.0, 2.0, 4.0],
        "kl_orientation": update["kl_orientation"],
        "save_adapters": update["save_adapters"],
    }
    legacy["evaluation"] = {
        "validation_usage": "diagnostics_only",
        "reward_fit_metrics": list(evaluation["reward_fit_metrics"]),
        "local_policy_metrics": [
            "local_regret",
            "fisher_cosine",
            "local_target_utility",
            "tabular_optimal_utility",
            "tabular_regret",
        ],
        "rollout": {
            "prompts": rollout["prompts"],
            "responses_per_prompt": rollout["responses_per_prompt"],
            "metrics": [
                "oracle_reward",
                "reward_improvement",
                "forward_kl",
                "regularized_utility",
                "utility_improvement",
                "oracle_ngd_regret",
            ],
        },
    }
    legacy["execution"] = {
        key: value
        for key, value in execution.items()
        if key
        not in {
            "crossfit_walltime",
            "kl_calibration_walltime",
            "calibration_max_parallel_policies",
        }
    }
    _validate_legacy_config(legacy)
    return copy.deepcopy(dict(root))


def validate_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Return a detached, closed-schema configuration for a supported protocol."""

    protocol = config.get("protocol") if isinstance(config, Mapping) else None
    if protocol == PROTOCOL:
        return _validate_legacy_config(config)
    if protocol == TRPO_PROTOCOL:
        return _validate_trpo_config(config)
    raise ConfigError(f"unsupported protocol: {protocol!r}")


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        yaml = importlib.import_module("yaml")
    except ImportError as error:
        raise ConfigError("PyYAML is required to read configuration files") from error
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return validate_config(value)


def config_hash(config: Mapping[str, object]) -> str:
    normalized = validate_config(config)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "PROTOCOL",
    "TRPO_PROTOCOL",
    "ConfigError",
    "config_hash",
    "load_config",
    "validate_config",
]

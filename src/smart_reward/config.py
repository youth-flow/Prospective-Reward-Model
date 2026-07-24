"""Strict experiment configuration loading and canonical hashing.

Configuration is deliberately validated as an exact, recursively closed
schema.  A misspelled option must stop a run instead of being silently ignored
by whichever training component happens to consume the file.  PyYAML is
imported only when a YAML file is actually read; it is a lightweight base
dependency, while Transformers/PEFT/Datasets remain optional LLM dependencies.
"""

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


class ConfigError(ValueError):
    """Raised when a configuration cannot be loaded or violates the protocol."""


class MissingConfigDependencyError(ConfigError):
    """Raised when YAML loading is requested without PyYAML installed."""


_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    non_string_keys = [key for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise ConfigError(f"{path} must contain only string keys")
    return value


def _keys(
    value: object,
    *,
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, object]:
    result = _mapping(value, path)
    optional = optional or set()
    actual = set(result)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys {sorted(missing)!r}")
        if unknown:
            details.append(f"unknown keys {sorted(unknown)!r}")
        raise ConfigError(f"{path}: {', '.join(details)}")
    return result


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _integer(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path} must be at most {maximum}")
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ConfigError(f"{path} must be {operator} {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ConfigError(f"{path} must be {operator} {maximum}")
    return result


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ConfigError(f"{path} must be a sequence")
    return value


def _nonempty_unique_strings(value: object, path: str) -> list[str]:
    items = [_string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))]
    if not items:
        raise ConfigError(f"{path} must not be empty")
    if len(set(items)) != len(items):
        raise ConfigError(f"{path} must not contain duplicates")
    return items


def _nonempty_unique_integers(
    value: object,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[int]:
    items = [
        _integer(item, f"{path}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(_sequence(value, path))
    ]
    if not items:
        raise ConfigError(f"{path} must not be empty")
    if len(set(items)) != len(items):
        raise ConfigError(f"{path} must not contain duplicates")
    return items


def _pinned_revision(value: object, path: str) -> str:
    revision = _string(value, path)
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ConfigError(f"{path} must be an immutable 40- or 64-character hexadecimal revision")
    return revision


def _validate_run(value: object) -> None:
    run = _keys(
        value,
        path="run",
        required={"name", "num_prompts", "split_sizes"},
        optional={"seed", "seeds"},
    )
    run_name = _string(run["name"], "run.name")
    if _RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise ConfigError(
            "run.name must be a filesystem-safe identifier containing only "
            "ASCII letters, digits, '.', '_', or '-'"
        )
    num_prompts = _integer(run["num_prompts"], "run.num_prompts", minimum=1)
    has_seed = "seed" in run
    has_seeds = "seeds" in run
    if has_seed == has_seeds:
        raise ConfigError("run must specify exactly one of seed or seeds")
    if has_seed:
        _integer(run["seed"], "run.seed", minimum=0, maximum=2**63 - 1)
    else:
        seeds = _nonempty_unique_integers(run["seeds"], "run.seeds", minimum=0, maximum=2**63 - 1)
        if not seeds:  # Kept explicit for type checkers and future refactors.
            raise ConfigError("run.seeds must not be empty")

    split_sizes = _keys(
        run["split_sizes"],
        path="run.split_sizes",
        required={"train", "validation", "test"},
    )
    sizes = {
        name: _integer(split_sizes[name], f"run.split_sizes.{name}", minimum=1)
        for name in ("train", "validation", "test")
    }
    if sum(sizes.values()) != num_prompts:
        raise ConfigError(
            "run.split_sizes must sum exactly to run.num_prompts "
            f"({sum(sizes.values())} != {num_prompts})"
        )
    if sizes["test"] < 2:
        raise ConfigError("run.split_sizes.test must be at least 2 for prompt-level uncertainty")


def _validate_data(value: object) -> None:
    data = _keys(
        value,
        path="data",
        required={"prompt_dataset", "prompt_revision", "num_candidates"},
        optional={"prompt_eligibility"},
    )
    _string(data["prompt_dataset"], "data.prompt_dataset")
    _pinned_revision(data["prompt_revision"], "data.prompt_revision")
    num_candidates = _integer(data["num_candidates"], "data.num_candidates", minimum=2)
    if num_candidates != 4:
        raise ConfigError("data.num_candidates must equal 4 for the locked Phase-1 graph")
    if "prompt_eligibility" in data:
        eligibility = _string(data["prompt_eligibility"], "data.prompt_eligibility")
        if eligibility != "policy_chat_template_tokens_lte_max_prompt_tokens_before_seeded_shuffle":
            raise ConfigError(
                "data.prompt_eligibility must freeze policy-token-length filtering "
                "before the seeded prompt split"
            )


def _validate_sampling(value: object) -> None:
    sampling = _keys(
        value,
        path="policy.sampling",
        required={
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "min_new_tokens",
            "repetition_penalty",
        },
    )
    if not _boolean(sampling["do_sample"], "policy.sampling.do_sample"):
        raise ConfigError("policy.sampling.do_sample must be true for candidate diversity")
    temperature = _number(
        sampling["temperature"],
        "policy.sampling.temperature",
        minimum=0.0,
        minimum_inclusive=False,
    )
    top_p = _number(
        sampling["top_p"],
        "policy.sampling.top_p",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    top_k = _integer(sampling["top_k"], "policy.sampling.top_k", minimum=0)
    min_new_tokens = _integer(
        sampling["min_new_tokens"], "policy.sampling.min_new_tokens", minimum=0
    )
    repetition_penalty = _number(
        sampling["repetition_penalty"],
        "policy.sampling.repetition_penalty",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if (temperature, top_p, top_k, min_new_tokens, repetition_penalty) != (
        1.0,
        1.0,
        0,
        0,
        1.0,
    ):
        raise ConfigError(
            "policy.sampling must equal the unwarped on-policy contract: "
            "temperature=1, top_p=1, top_k=0, min_new_tokens=0, "
            "repetition_penalty=1"
        )


def _validate_fixed_lora_a(value: object) -> None:
    contract = _keys(
        value,
        path="policy.fixed_lora_a",
        required={
            "mode",
            "initialization_seed",
            "expected_sha256",
            "source_seed",
            "source_named_stream",
            "source_config",
            "source_config_hash",
            "source_artifact_metadata_sha256",
            "source_seed_excluded_from_phase2",
        },
    )
    if contract["mode"] != "frozen_global":
        raise ConfigError("policy.fixed_lora_a.mode must equal 'frozen_global'")
    _integer(
        contract["initialization_seed"],
        "policy.fixed_lora_a.initialization_seed",
        minimum=0,
        maximum=2**63 - 1,
    )
    expected_sha256 = _string(
        contract["expected_sha256"],
        "policy.fixed_lora_a.expected_sha256",
    )
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ConfigError(
            "policy.fixed_lora_a.expected_sha256 must be a lowercase 64-character SHA-256 digest"
        )
    _integer(
        contract["source_seed"],
        "policy.fixed_lora_a.source_seed",
        minimum=0,
        maximum=2**63 - 1,
    )
    if contract["source_named_stream"] != "policy_lora_a":
        raise ConfigError("policy.fixed_lora_a.source_named_stream must equal 'policy_lora_a'")
    _string(contract["source_config"], "policy.fixed_lora_a.source_config")
    source_config_hash = _string(
        contract["source_config_hash"],
        "policy.fixed_lora_a.source_config_hash",
    )
    if _SHA256_PATTERN.fullmatch(source_config_hash) is None:
        raise ConfigError(
            "policy.fixed_lora_a.source_config_hash must be a lowercase 64-character SHA-256 digest"
        )
    source_metadata_hash = _string(
        contract["source_artifact_metadata_sha256"],
        "policy.fixed_lora_a.source_artifact_metadata_sha256",
    )
    if _SHA256_PATTERN.fullmatch(source_metadata_hash) is None:
        raise ConfigError(
            "policy.fixed_lora_a.source_artifact_metadata_sha256 must be a lowercase "
            "64-character SHA-256 digest"
        )
    if not _boolean(
        contract["source_seed_excluded_from_phase2"],
        "policy.fixed_lora_a.source_seed_excluded_from_phase2",
    ):
        raise ConfigError("policy.fixed_lora_a.source_seed_excluded_from_phase2 must be true")


def _validate_policy(value: object, normalized: dict[str, Any]) -> None:
    policy = _keys(
        value,
        path="policy",
        required={
            "model",
            "revision",
            "dtype",
            "max_prompt_tokens",
            "max_response_tokens",
            "sampling",
            "lora_rank",
            "lora_alpha",
            "lora_layers",
            "lora_modules",
        },
        optional={
            "lora_dropout",
            "trainable_tangent_parameters",
            "fixed_lora_a",
        },
    )
    _string(policy["model"], "policy.model")
    _pinned_revision(policy["revision"], "policy.revision")
    dtype = _string(policy["dtype"], "policy.dtype")
    if dtype != "float32":
        raise ConfigError("policy.dtype must equal float32 for the locked Phase-1 geometry")
    _integer(policy["max_prompt_tokens"], "policy.max_prompt_tokens", minimum=1)
    _integer(policy["max_response_tokens"], "policy.max_response_tokens", minimum=1)
    _validate_sampling(policy["sampling"])

    rank = _integer(policy["lora_rank"], "policy.lora_rank", minimum=1)
    alpha = _integer(policy["lora_alpha"], "policy.lora_alpha", minimum=1)
    if alpha != rank:
        raise ConfigError("fixed-A LoRA requires policy.lora_alpha == policy.lora_rank")
    _nonempty_unique_integers(policy["lora_layers"], "policy.lora_layers")
    _nonempty_unique_strings(policy["lora_modules"], "policy.lora_modules")

    dropout = _number(policy.get("lora_dropout", 0.0), "policy.lora_dropout")
    if dropout != 0.0:
        raise ConfigError("fixed-A LoRA requires policy.lora_dropout == 0")
    tangent = policy.get("trainable_tangent_parameters", "lora_B_only")
    if tangent != "lora_B_only":
        raise ConfigError(
            "fixed-A LoRA requires policy.trainable_tangent_parameters == 'lora_B_only'"
        )
    if "fixed_lora_a" in policy:
        _validate_fixed_lora_a(policy["fixed_lora_a"])

    # The smoke config predates the explicit spelling of these locked values.
    # Materializing them makes semantically identical runs hash identically.
    normalized_policy = dict(normalized["policy"])
    normalized_policy.setdefault("lora_dropout", 0.0)
    normalized_policy.setdefault("trainable_tangent_parameters", "lora_B_only")
    normalized["policy"] = normalized_policy


def _validate_oracle_transform_calibration(
    value: object,
    *,
    scale_floor: float,
) -> None:
    calibration = _keys(
        value,
        path="oracle.transform_calibration",
        required={
            "mode",
            "b",
            "tau",
            "aggregation_rule",
            "source_split",
            "source_config",
            "source_config_hash",
            "source_seeds_excluded_from_phase2",
            "source_artifacts",
        },
    )
    if calibration["mode"] != "frozen_global":
        raise ConfigError("oracle.transform_calibration.mode must equal 'frozen_global'")
    _number(calibration["b"], "oracle.transform_calibration.b")
    _number(
        calibration["tau"],
        "oracle.transform_calibration.tau",
        minimum=scale_floor,
    )
    if calibration["aggregation_rule"] != "componentwise_median":
        raise ConfigError(
            "oracle.transform_calibration.aggregation_rule must equal 'componentwise_median'"
        )
    if calibration["source_split"] != "train":
        raise ConfigError("oracle.transform_calibration.source_split must equal 'train'")
    _string(
        calibration["source_config"],
        "oracle.transform_calibration.source_config",
    )
    source_config_hash = _string(
        calibration["source_config_hash"],
        "oracle.transform_calibration.source_config_hash",
    )
    if _SHA256_PATTERN.fullmatch(source_config_hash) is None:
        raise ConfigError(
            "oracle.transform_calibration.source_config_hash must be a lowercase "
            "64-character SHA-256 digest"
        )
    if not _boolean(
        calibration["source_seeds_excluded_from_phase2"],
        "oracle.transform_calibration.source_seeds_excluded_from_phase2",
    ):
        raise ConfigError(
            "oracle.transform_calibration.source_seeds_excluded_from_phase2 must be true"
        )

    source_artifacts = _sequence(
        calibration["source_artifacts"],
        "oracle.transform_calibration.source_artifacts",
    )
    if not source_artifacts:
        raise ConfigError("oracle.transform_calibration.source_artifacts must not be empty")
    seeds: list[int] = []
    metadata_hashes: list[str] = []
    for index, raw_artifact in enumerate(source_artifacts):
        path = f"oracle.transform_calibration.source_artifacts[{index}]"
        artifact = _keys(
            raw_artifact,
            path=path,
            required={"seed", "metadata_sha256"},
        )
        seeds.append(
            _integer(
                artifact["seed"],
                f"{path}.seed",
                minimum=0,
                maximum=2**63 - 1,
            )
        )
        digest = _string(artifact["metadata_sha256"], f"{path}.metadata_sha256")
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ConfigError(
                f"{path}.metadata_sha256 must be a lowercase 64-character SHA-256 digest"
            )
        metadata_hashes.append(digest)
    if len(seeds) != len(set(seeds)):
        raise ConfigError("oracle.transform_calibration.source_artifacts must have unique seeds")
    if len(metadata_hashes) != len(set(metadata_hashes)):
        raise ConfigError(
            "oracle.transform_calibration.source_artifacts must have unique metadata hashes"
        )


def _validate_oracle(value: object) -> float:
    oracle = _keys(
        value,
        path="oracle",
        required={
            "model",
            "revision",
            "dtype",
            "transform",
            "robust_scale",
            "robust_scale_floor",
            "probability_floor",
        },
        optional={"transform_calibration"},
    )
    _string(oracle["model"], "oracle.model")
    _pinned_revision(oracle["revision"], "oracle.revision")
    dtype = _string(oracle["dtype"], "oracle.dtype")
    if dtype != "float32":
        raise ConfigError("oracle.dtype must equal float32 for the locked Phase-1 experiment")
    if oracle["transform"] != "robust_center_scale_then_tanh":
        raise ConfigError("oracle.transform must be 'robust_center_scale_then_tanh'")
    if oracle["robust_scale"] != "scaled_mad":
        raise ConfigError("oracle.robust_scale must be 'scaled_mad'")
    scale_floor = _number(
        oracle["robust_scale_floor"],
        "oracle.robust_scale_floor",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if scale_floor != 1.0e-6:
        raise ConfigError("oracle.robust_scale_floor must equal the locked value 1e-6")
    if "transform_calibration" in oracle:
        _validate_oracle_transform_calibration(
            oracle["transform_calibration"],
            scale_floor=scale_floor,
        )
    probability_floor = _number(
        oracle["probability_floor"],
        "oracle.probability_floor",
        minimum=0.0,
        maximum=0.5,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    if probability_floor != 0.25:
        raise ConfigError("oracle.probability_floor must equal the locked value 0.25")
    return probability_floor


def _validate_annotations(value: object, probability_floor: float) -> None:
    annotations = _keys(
        value,
        path="annotations",
        required={"scheme", "gamma"},
    )
    if annotations["scheme"] != "geometric_randomized_truncation":
        raise ConfigError("annotations.scheme must be 'geometric_randomized_truncation'")
    gamma = _number(
        annotations["gamma"],
        "annotations.gamma",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    if gamma != 0.9:
        raise ConfigError("annotations.gamma must equal the locked training-schema value 0.9")
    lower_bound = 1.0 - probability_floor
    if gamma <= lower_bound:
        raise ConfigError(
            "annotations.gamma must be strictly greater than "
            f"1 - oracle.probability_floor ({lower_bound})"
        )


def _validate_objective(value: object) -> None:
    objective = _keys(
        value,
        path="objective",
        required={
            "beta",
            "damping_relative_to_mean_fisher_diagonal",
            "pcg_dtype",
            "pcg_tolerance",
            "pcg_max_iterations",
        },
        optional={"damping_sensitivity_multipliers"},
    )
    _number(objective["beta"], "objective.beta", minimum=0.0, minimum_inclusive=False)
    _number(
        objective["damping_relative_to_mean_fisher_diagonal"],
        "objective.damping_relative_to_mean_fisher_diagonal",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _number(
        objective["pcg_tolerance"],
        "objective.pcg_tolerance",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if objective["pcg_dtype"] != "float64":
        raise ConfigError("objective.pcg_dtype must be 'float64'")
    _integer(objective["pcg_max_iterations"], "objective.pcg_max_iterations", minimum=1)
    if "damping_sensitivity_multipliers" in objective:
        multipliers = _sequence(
            objective["damping_sensitivity_multipliers"],
            "objective.damping_sensitivity_multipliers",
        )
        if not multipliers:
            raise ConfigError("objective.damping_sensitivity_multipliers must not be empty")
        parsed = [
            _number(
                item,
                f"objective.damping_sensitivity_multipliers[{index}]",
                minimum=0.0,
                minimum_inclusive=False,
            )
            for index, item in enumerate(multipliers)
        ]
        if len(set(parsed)) != len(parsed):
            raise ConfigError(
                "objective.damping_sensitivity_multipliers must not contain duplicates"
            )


def _validate_reward_model(value: object) -> None:
    reward = _keys(
        value,
        path="reward_model",
        required={
            "model",
            "revision",
            "dtype",
            "parameterization",
            "feature_pooling",
            "linear_head_bias",
            "outer_steps",
            "refresh_dual_every_steps",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "microbatch_size",
        },
        optional={"max_grad_norm"},
    )
    _string(reward["model"], "reward_model.model")
    _pinned_revision(reward["revision"], "reward_model.revision")
    dtype = _string(reward["dtype"], "reward_model.dtype")
    if dtype != "float32":
        raise ConfigError("reward_model.dtype must equal float32 for the locked comparison")
    if reward["parameterization"] != "frozen_backbone_linear_head":
        raise ConfigError("reward_model.parameterization must be 'frozen_backbone_linear_head'")
    if reward["feature_pooling"] != "last_response_token":
        raise ConfigError("reward_model.feature_pooling must be 'last_response_token'")
    if _boolean(reward["linear_head_bias"], "reward_model.linear_head_bias"):
        raise ConfigError("reward_model.linear_head_bias must be false")
    _integer(reward["outer_steps"], "reward_model.outer_steps", minimum=1)
    refresh = _integer(
        reward["refresh_dual_every_steps"],
        "reward_model.refresh_dual_every_steps",
        minimum=1,
    )
    if refresh != 1:
        raise ConfigError("reward_model.refresh_dual_every_steps must equal 1")
    optimizer = _string(reward["optimizer"], "reward_model.optimizer")
    if optimizer != "adamw":
        raise ConfigError("reward_model.optimizer must equal 'adamw' for the locked comparison")
    _number(
        reward["learning_rate"],
        "reward_model.learning_rate",
        minimum=0.0,
        minimum_inclusive=False,
    )
    weight_decay = _number(reward["weight_decay"], "reward_model.weight_decay", minimum=0.0)
    if weight_decay != 0.0:
        raise ConfigError("reward_model.weight_decay must equal 0 for the locked objective")
    _integer(reward["microbatch_size"], "reward_model.microbatch_size", minimum=1)
    if "max_grad_norm" in reward:
        _number(
            reward["max_grad_norm"],
            "reward_model.max_grad_norm",
            minimum=0.0,
            minimum_inclusive=False,
        )


def _validate_evaluation(value: object) -> None:
    evaluation = _keys(
        value,
        path="evaluation",
        required={
            "heldout_moment",
            "kl_budget",
            "kl_relative_tolerance",
            "kl_probe_candidates",
            "rollout_candidates_per_prompt",
            "report_pairwise_accuracy",
            "paired_bootstrap_resamples",
            "paired_bootstrap_seed",
        },
    )
    if evaluation["heldout_moment"] != "per_prompt_unbiased_covariance":
        raise ConfigError("evaluation.heldout_moment must be 'per_prompt_unbiased_covariance'")
    _number(
        evaluation["kl_budget"],
        "evaluation.kl_budget",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _number(
        evaluation["kl_relative_tolerance"],
        "evaluation.kl_relative_tolerance",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    _integer(
        evaluation["kl_probe_candidates"],
        "evaluation.kl_probe_candidates",
        minimum=1,
    )
    _integer(
        evaluation["rollout_candidates_per_prompt"],
        "evaluation.rollout_candidates_per_prompt",
        minimum=1,
    )
    if not _boolean(
        evaluation["report_pairwise_accuracy"],
        "evaluation.report_pairwise_accuracy",
    ):
        raise ConfigError("evaluation.report_pairwise_accuracy must be true")
    _integer(
        evaluation["paired_bootstrap_resamples"],
        "evaluation.paired_bootstrap_resamples",
        minimum=1,
    )
    _integer(
        evaluation["paired_bootstrap_seed"],
        "evaluation.paired_bootstrap_seed",
        minimum=0,
        maximum=2**63 - 1,
    )


def validate_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Validate and return an independent, normalized configuration mapping.

    The returned mapping is safe to hash or include in a run manifest.  Two
    locked fixed-A defaults omitted by the original smoke configuration are
    made explicit: zero LoRA dropout and ``lora_B_only`` tangent parameters.
    """

    root = _keys(
        config,
        path="config",
        required={
            "run",
            "data",
            "policy",
            "oracle",
            "annotations",
            "objective",
            "reward_model",
            "evaluation",
        },
    )
    normalized = copy.deepcopy(dict(root))
    _validate_run(root["run"])
    _validate_data(root["data"])
    _validate_policy(root["policy"], normalized)
    probability_floor = _validate_oracle(root["oracle"])
    _validate_annotations(root["annotations"], probability_floor)
    _validate_objective(root["objective"])
    _validate_reward_model(root["reward_model"])
    _validate_evaluation(root["evaluation"])
    policy = root["policy"]
    reward = root["reward_model"]
    for field in ("model", "revision", "dtype"):
        if reward[field] != policy[field]:
            raise ConfigError(
                f"frozen_backbone_linear_head requires reward_model.{field} to equal policy.{field}"
            )
    if root["evaluation"]["rollout_candidates_per_prompt"] != root["data"]["num_candidates"]:
        raise ConfigError("evaluation.rollout_candidates_per_prompt must equal data.num_candidates")
    train_candidates = root["run"]["split_sizes"]["train"] * root["data"]["num_candidates"]
    minimum_pcg_iterations = train_candidates + 1
    if root["objective"]["pcg_max_iterations"] < minimum_pcg_iterations:
        raise ConfigError(
            "objective.pcg_max_iterations must be at least the number of train "
            f"Fisher nodes plus one ({minimum_pcg_iterations})"
        )
    if root["evaluation"]["kl_probe_candidates"] > train_candidates:
        raise ConfigError(
            "evaluation.kl_probe_candidates cannot exceed the number of train candidates"
        )
    return normalized


def _load_yaml_module() -> Any:
    try:
        return importlib.import_module("yaml")
    except ModuleNotFoundError as error:
        if error.name != "yaml":
            raise
        raise MissingConfigDependencyError(
            "reading YAML configuration requires PyYAML; install the project with "
            "`pip install -e .` or install `pyyaml>=6,<7`"
        ) from error


def _parse_yaml(text: str, *, source: Path) -> object:
    yaml = _load_yaml_module()

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise ConfigError(f"{source}: YAML mapping keys must be hashable") from error
            if duplicate:
                raise ConfigError(f"{source}: duplicate YAML key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    try:
        return yaml.load(text, Loader=UniqueKeySafeLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as error:
        raise ConfigError(f"failed to parse YAML configuration {source}: {error}") from error


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one UTF-8 YAML file and return its validated normalized mapping."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read configuration {source}: {error}") from error
    parsed = _parse_yaml(text, source=source)
    if parsed is None:
        raise ConfigError(f"configuration {source} is empty")
    try:
        return validate_config(parsed)
    except ConfigError as error:
        raise ConfigError(f"{source}: {error}") from error


load_yaml_config = load_config


def _canonicalize(value: object, path: str = "config") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{path} contains a non-finite number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConfigError(f"{path} contains a non-string mapping key")
        result: dict[str, object] = {}
        for key in sorted(value):
            result[key] = _canonicalize(value[key], _path(path, key))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ConfigError(f"{path} contains non-JSON value of type {type(value).__name__}")


def canonical_json(config: Mapping[str, object]) -> str:
    """Serialize JSON deterministically for hashing, independent of key order."""

    return json.dumps(
        _canonicalize(config),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


canonical_config_json = canonical_json


def config_hash(config: Mapping[str, object]) -> str:
    """Return the lowercase SHA-256 digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


canonical_config_hash = config_hash


__all__ = [
    "ConfigError",
    "MissingConfigDependencyError",
    "canonical_config_hash",
    "canonical_config_json",
    "canonical_json",
    "config_hash",
    "load_config",
    "load_yaml_config",
    "validate_config",
]

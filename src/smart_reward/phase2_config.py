"""Strict configuration contract for common-beta pilot and confirmatory designs.

The completed Phase-1 campaign and the common-beta campaign answer different
questions.  This module therefore does not extend the Phase-1 schema with
optional fields.  It defines a recursively closed Phase-2 schema, binds the
frozen model/data geometry to the materialization source declared by each
overlay, and rejects combinations that would silently turn a pilot into
confirmatory evidence or the experiment back into a learner-wise matched-KL
comparison.

The module performs configuration parsing and validation only.  It deliberately
does not import Transformers, allocate tensors, or execute a model.
"""

from __future__ import annotations

import copy
import importlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import (
    ConfigError,
    MissingConfigDependencyError,
    config_hash,
    load_config,
    validate_config,
)

PHASE2_SCHEMA_VERSION = "prorm-common-beta-config/v2"
PHASE2_RECOVERY_SCHEMA_VERSION = "prorm-common-beta-recovery-config/v1"
PHASE1_MAIN_CONFIG = "configs/main.yaml"
PHASE1_MAIN_CONFIG_HASH = "ae5d628ee47ff74a1fa2b89478c40b4fdd289935d8cf58dcbcf98b42f69a0df6"
PHASE1_MAIN_SEEDS = frozenset({20260722, 20260723, 20260724, 20260725, 20260726})
PHASE2_PILOT_CONFIG = "configs/common_beta_pilot.yaml"
PHASE2_PILOT_BASE_CONFIG = "configs/common_beta_pilot_base.yaml"
PHASE2_RECOVERY_PILOT_CONFIG = "configs/common_beta_recovery_pilot.yaml"
PHASE2_PILOT_SEEDS = frozenset({20260801, 20260802, 20260803})
_PHASE2_RECOVERY_PILOT_SEEDS = (20260801, 20260802, 20260803)
PHASE2_CONFIRMATORY_SEEDS = tuple(range(20260901, 20260931))
PHASE2_CONFIRMATORY_NUM_SEEDS = len(PHASE2_CONFIRMATORY_SEEDS)
# Backward-compatible import name.  The contract is exact, not a lower bound.
PHASE2_MIN_CONFIRMATORY_SEEDS = PHASE2_CONFIRMATORY_NUM_SEEDS
PHASE2_CONFIRMATORY_EXCLUDED_SEEDS = PHASE1_MAIN_SEEDS | PHASE2_PILOT_SEEDS
PHASE2_FROZEN_ORACLE_B = -4.500244140625
PHASE2_FROZEN_ORACLE_TAU = 2.7715682983398438
PHASE2_FIXED_LORA_A_INITIALIZATION_SEED = 946081152281754541
PHASE2_FIXED_LORA_A_SHA256 = "a2b5804109396f76b96cde98d1e2060f175a47724b1ca9fef317c7a10cb9a838"
PHASE2_FIXED_LORA_A_SOURCE_SEED = 20260722
PHASE2_FIXED_LORA_A_SOURCE_METADATA_SHA256 = (
    "8ec7ca893a7c93d4bbfac0b71b829b37ec63af3ca0c7a9496465ed60815b155f"
)
PHASE2_FROZEN_ORACLE_SOURCE_ARTIFACTS = (
    (
        20260722,
        "8ec7ca893a7c93d4bbfac0b71b829b37ec63af3ca0c7a9496465ed60815b155f",
    ),
    (
        20260723,
        "4dcd84ba5be64ac7ddc0c8f6caee43ed857e2d735273382e26c388184d2991c8",
    ),
    (
        20260724,
        "84b9d89fa76555e34c7e3fa840a2873e32b91f1378991a2a3f2cf9fa42c7451d",
    ),
    (
        20260725,
        "5061881856516d61dcb20bac8a9240aacbf6300f1d6d136fb1f15258e2b0078a",
    ),
    (
        20260726,
        "fdf8b717f7360ba3019ef56af56d07bfe6ee00d6d9aff2836fbf7f249f201b97",
    ),
)

_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_HEX_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_POLICY_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_ORACLE_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
_ORACLE_REVISION = "8c14a4e9e6321deaf572544339b16b8d6bbe8886"
_PROMPT_DATASET = "allenai/multipref"
_PROMPT_REVISION = "12910233a0238a997ebe425656e9dfed7b0ff031"
_POLICY_ARMS = ("zero_b", "bt_mle", "prorm_plus", "oracle_step")
_COMMON_BETA_ARMS = ("bt_mle", "prorm_plus", "oracle_step")
_DESIGN_STAGES = ("pilot", "confirmatory")
_PILOT_PHASES = ("calibration", "freeze")
_HORIZON_SEQUENCE = (256, 512, 1024)
_RECOVERY_MAXIMUM_STEPS = 12760
_RECOVERY_LEGACY_BOUNDARY_STEPS = 5760
_RECOVERY_LR_STAGES = (
    (1, 5760, 1.0e-3),
    (5761, 6760, 3.0e-4),
    (6761, 8760, 1.0e-4),
    (8761, 10760, 3.0e-5),
    (10761, 12760, 1.0e-5),
)
_RECOVERY_TIE_BREAK = "exact_zero_initialized_deterministic_adamw_lr_decay_path"


def _recovery_schedule_payload() -> dict[str, object]:
    return {
        "update_indexing": "one_indexed_inclusive",
        "application": "set_learning_rate_immediately_before_optimizer_update",
        "stages": [
            {
                "first_update": first_update,
                "last_update": last_update,
                "learning_rate": learning_rate,
            }
            for first_update, last_update, learning_rate in _RECOVERY_LR_STAGES
        ],
    }


PHASE2_RECOVERY_LR_SCHEDULE_SHA256 = config_hash(_recovery_schedule_payload())


@dataclass(frozen=True)
class Phase2ConfigBundle:
    """A validated Phase-2 design together with its materialization config."""

    config: dict[str, Any]
    base_config: dict[str, Any]
    design_identity: str
    source_path: Path
    base_config_path: Path


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
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
    minimum_inclusive: bool = True,
    maximum: float | None = None,
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


def _locked_string(value: object, path: str, expected: str) -> str:
    result = _string(value, path)
    if result != expected:
        raise ConfigError(f"{path} must equal {expected!r}")
    return result


def _locked_boolean(value: object, path: str, expected: bool) -> bool:
    result = _boolean(value, path)
    if result is not expected:
        raise ConfigError(f"{path} must be {str(expected).lower()}")
    return result


def _locked_integer(value: object, path: str, expected: int) -> int:
    result = _integer(value, path)
    if result != expected:
        raise ConfigError(f"{path} must equal {expected}")
    return result


def _locked_number(value: object, path: str, expected: float) -> float:
    result = _number(value, path)
    if result != expected:
        raise ConfigError(f"{path} must equal {expected}")
    return result


def _unique_strings(value: object, path: str) -> list[str]:
    result = [
        _string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))
    ]
    if not result:
        raise ConfigError(f"{path} must not be empty")
    if len(result) != len(set(result)):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def _unique_integers(value: object, path: str) -> list[int]:
    result = [
        _integer(item, f"{path}[{index}]", minimum=0, maximum=2**63 - 1)
        for index, item in enumerate(_sequence(value, path))
    ]
    if not result:
        raise ConfigError(f"{path} must not be empty")
    if len(result) != len(set(result)):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def _locked_number_sequence(
    value: object,
    path: str,
    expected: tuple[float, ...],
) -> tuple[float, ...]:
    parsed = tuple(
        _number(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))
    )
    if parsed != expected:
        raise ConfigError(f"{path} must equal {list(expected)!r} in that order")
    return parsed


def _locked_string_sequence(
    value: object,
    path: str,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    parsed = tuple(_unique_strings(value, path))
    if parsed != expected:
        raise ConfigError(f"{path} must equal {list(expected)!r} in that order")
    return parsed


def _declared_config_path(value: object, path: str) -> str:
    result = _string(value, path)
    pure = PurePosixPath(result)
    if (
        pure.is_absolute()
        or "\\" in result
        or ".." in pure.parts
        or pure.parts[:1] != ("configs",)
        or pure.suffix not in {".yaml", ".yml"}
        or str(pure) != result
    ):
        raise ConfigError(f"{path} must be a normalized relative POSIX YAML path below configs/")
    return result


def _validate_design(value: object) -> tuple[str, str | None, str, str]:
    design = _keys(
        value,
        path="design",
        required={
            "name",
            "stage",
            "pilot_phase",
            "formal_eligibility",
            "evidence_role",
            "pilot_results_permanently_excluded_from_confirmatory",
            "source_config",
            "source_config_hash",
            "predecessor_config",
            "predecessor_config_hash",
            "predecessor_evidence_role",
            "estimand",
        },
    )
    name = _string(design["name"], "design.name")
    if _RUN_NAME_PATTERN.fullmatch(name) is None:
        raise ConfigError(
            "design.name must be a filesystem-safe identifier containing only "
            "ASCII letters, digits, '.', '_', or '-'"
        )
    if name == "controlled-main":
        raise ConfigError("design.name must differ from the completed Phase-1 design")
    stage = _string(design["stage"], "design.stage")
    if stage not in _DESIGN_STAGES:
        raise ConfigError(f"design.stage must be one of {_DESIGN_STAGES!r}")
    formal_eligibility = _boolean(
        design["formal_eligibility"],
        "design.formal_eligibility",
    )
    evidence_role = _string(design["evidence_role"], "design.evidence_role")
    pilot_phase_value = design["pilot_phase"]
    if stage == "pilot":
        pilot_phase = _string(pilot_phase_value, "design.pilot_phase")
        if pilot_phase not in _PILOT_PHASES:
            raise ConfigError(f"design.pilot_phase must be one of {_PILOT_PHASES!r}")
        if "confirmatory" in name.lower():
            raise ConfigError("a pilot design.name must not claim to be confirmatory")
        if formal_eligibility:
            raise ConfigError("pilot design.formal_eligibility must be false")
        if evidence_role != "pilot_design_selection_only":
            raise ConfigError("pilot design.evidence_role must equal 'pilot_design_selection_only'")
    else:
        if pilot_phase_value is not None:
            raise ConfigError("confirmatory design.pilot_phase must be null")
        pilot_phase = None
        if not formal_eligibility:
            raise ConfigError("confirmatory design.formal_eligibility must be true")
        if evidence_role != "confirmatory_evidence":
            raise ConfigError(
                "confirmatory design.evidence_role must equal 'confirmatory_evidence'"
            )
    _locked_boolean(
        design["pilot_results_permanently_excluded_from_confirmatory"],
        "design.pilot_results_permanently_excluded_from_confirmatory",
        True,
    )
    source_config = _declared_config_path(design["source_config"], "design.source_config")
    if source_config == PHASE1_MAIN_CONFIG:
        raise ConfigError("design.source_config must not reuse the Phase-1 main config")
    source_digest = _string(design["source_config_hash"], "design.source_config_hash")
    if _HEX_DIGEST_PATTERN.fullmatch(source_digest) is None:
        raise ConfigError("design.source_config_hash must be 64 lowercase hexadecimal characters")
    _locked_string(
        design["predecessor_config"],
        "design.predecessor_config",
        PHASE1_MAIN_CONFIG,
    )
    digest = _string(design["predecessor_config_hash"], "design.predecessor_config_hash")
    if _HEX_DIGEST_PATTERN.fullmatch(digest) is None:
        raise ConfigError(
            "design.predecessor_config_hash must be 64 lowercase hexadecimal characters"
        )
    if digest != PHASE1_MAIN_CONFIG_HASH:
        raise ConfigError(
            "design.predecessor_config_hash must bind the accepted Phase-1 main design"
        )
    _locked_string(
        design["predecessor_evidence_role"],
        "design.predecessor_evidence_role",
        "exploratory_audit_only",
    )
    _locked_string(
        design["estimand"],
        "design.estimand",
        "fixed_beta_downstream_policy_regret",
    )
    return stage, pilot_phase, source_config, source_digest


def _validate_run(value: object, *, stage: str) -> tuple[int, int]:
    run = _keys(
        value,
        path="run",
        required={
            "seeds",
            "num_prompts",
            "split_sizes",
            "confirmatory",
            "formal_eligibility",
            "excluded_from_confirmatory_evidence",
        },
    )
    seeds = _unique_integers(run["seeds"], "run.seeds")
    confirmatory = _boolean(run["confirmatory"], "run.confirmatory")
    formal_eligibility = _boolean(run["formal_eligibility"], "run.formal_eligibility")
    excluded = _boolean(
        run["excluded_from_confirmatory_evidence"],
        "run.excluded_from_confirmatory_evidence",
    )
    if stage == "pilot":
        if confirmatory or formal_eligibility or not excluded:
            raise ConfigError(
                "pilot runs require confirmatory=false, formal_eligibility=false, "
                "and excluded_from_confirmatory_evidence=true"
            )
        if frozenset(seeds) != PHASE2_PILOT_SEEDS or len(seeds) != len(PHASE2_PILOT_SEEDS):
            raise ConfigError(
                "pilot run.seeds must be exactly the permanently excluded seeds "
                f"{sorted(PHASE2_PILOT_SEEDS)!r}"
            )
    else:
        if not confirmatory or not formal_eligibility or excluded:
            raise ConfigError(
                "confirmatory runs require confirmatory=true, formal_eligibility=true, "
                "and excluded_from_confirmatory_evidence=false"
            )
        if tuple(seeds) != PHASE2_CONFIRMATORY_SEEDS:
            raise ConfigError(
                "confirmatory run.seeds must equal the exact preregistered ordered "
                f"{PHASE2_CONFIRMATORY_NUM_SEEDS}-seed list "
                f"{list(PHASE2_CONFIRMATORY_SEEDS)!r}"
            )
        overlap = sorted(set(seeds) & PHASE2_CONFIRMATORY_EXCLUDED_SEEDS)
        if overlap:
            raise ConfigError(
                "confirmatory run.seeds must exclude all Phase-1 and pilot seeds; "
                f"overlap={overlap!r}"
            )

    num_prompts = _locked_integer(run["num_prompts"], "run.num_prompts", 2048)
    split = _keys(
        run["split_sizes"],
        path="run.split_sizes",
        required={"train", "validation", "test"},
    )
    sizes = {
        name: _integer(split[name], f"run.split_sizes.{name}", minimum=1)
        for name in ("train", "validation", "test")
    }
    if sizes != {"train": 1536, "validation": 256, "test": 256}:
        raise ConfigError("run.split_sizes must preserve the bound 1536/256/256 prompt geometry")
    if sum(sizes.values()) != num_prompts:
        raise ConfigError("run.split_sizes must sum exactly to run.num_prompts")
    return sizes["train"], len(seeds)


def _validate_data(value: object) -> int:
    data = _keys(
        value,
        path="data",
        required={
            "prompt_dataset",
            "prompt_revision",
            "prompt_eligibility",
            "num_candidates",
        },
    )
    _locked_string(data["prompt_dataset"], "data.prompt_dataset", _PROMPT_DATASET)
    _locked_string(data["prompt_revision"], "data.prompt_revision", _PROMPT_REVISION)
    _locked_string(
        data["prompt_eligibility"],
        "data.prompt_eligibility",
        "policy_chat_template_tokens_lte_max_prompt_tokens_before_seeded_shuffle",
    )
    return _locked_integer(data["num_candidates"], "data.num_candidates", 4)


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
    _locked_boolean(sampling["do_sample"], "policy.sampling.do_sample", True)
    _locked_number(sampling["temperature"], "policy.sampling.temperature", 1.0)
    _locked_number(sampling["top_p"], "policy.sampling.top_p", 1.0)
    _locked_integer(sampling["top_k"], "policy.sampling.top_k", 0)
    _locked_integer(sampling["min_new_tokens"], "policy.sampling.min_new_tokens", 0)
    _locked_number(
        sampling["repetition_penalty"],
        "policy.sampling.repetition_penalty",
        1.0,
    )


def _validate_policy(value: object, *, stage: str) -> int:
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
            "lora_dropout",
            "lora_layers",
            "lora_modules",
            "trainable_tangent_parameters",
            "fixed_lora_a",
        },
    )
    _locked_string(policy["model"], "policy.model", _POLICY_MODEL)
    _locked_string(policy["revision"], "policy.revision", _POLICY_REVISION)
    _locked_string(policy["dtype"], "policy.dtype", "float32")
    _locked_integer(policy["max_prompt_tokens"], "policy.max_prompt_tokens", 1024)
    max_response_tokens = _integer(
        policy["max_response_tokens"],
        "policy.max_response_tokens",
        minimum=1,
        maximum=4096,
    )
    _validate_sampling(policy["sampling"])
    rank = _locked_integer(policy["lora_rank"], "policy.lora_rank", 4)
    alpha = _locked_integer(policy["lora_alpha"], "policy.lora_alpha", 4)
    if rank != alpha:
        raise ConfigError("fixed-A LoRA requires policy.lora_alpha == policy.lora_rank")
    _locked_number(policy["lora_dropout"], "policy.lora_dropout", 0.0)
    layers = _unique_integers(policy["lora_layers"], "policy.lora_layers")
    if layers != [20, 21, 22, 23]:
        raise ConfigError("policy.lora_layers must preserve the bound last-four-layer tangent")
    modules = _unique_strings(policy["lora_modules"], "policy.lora_modules")
    if modules != ["q_proj", "v_proj"]:
        raise ConfigError("policy.lora_modules must equal ['q_proj', 'v_proj']")
    _locked_string(
        policy["trainable_tangent_parameters"],
        "policy.trainable_tangent_parameters",
        "lora_B_only",
    )
    fixed_a = _keys(
        policy["fixed_lora_a"],
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
    _locked_string(
        fixed_a["mode"],
        "policy.fixed_lora_a.mode",
        "frozen_global",
    )
    _locked_integer(
        fixed_a["initialization_seed"],
        "policy.fixed_lora_a.initialization_seed",
        PHASE2_FIXED_LORA_A_INITIALIZATION_SEED,
    )
    _locked_string(
        fixed_a["expected_sha256"],
        "policy.fixed_lora_a.expected_sha256",
        PHASE2_FIXED_LORA_A_SHA256,
    )
    _locked_integer(
        fixed_a["source_seed"],
        "policy.fixed_lora_a.source_seed",
        PHASE2_FIXED_LORA_A_SOURCE_SEED,
    )
    _locked_string(
        fixed_a["source_named_stream"],
        "policy.fixed_lora_a.source_named_stream",
        "policy_lora_a",
    )
    _locked_string(
        fixed_a["source_config"],
        "policy.fixed_lora_a.source_config",
        PHASE1_MAIN_CONFIG,
    )
    _locked_string(
        fixed_a["source_config_hash"],
        "policy.fixed_lora_a.source_config_hash",
        PHASE1_MAIN_CONFIG_HASH,
    )
    _locked_string(
        fixed_a["source_artifact_metadata_sha256"],
        "policy.fixed_lora_a.source_artifact_metadata_sha256",
        PHASE2_FIXED_LORA_A_SOURCE_METADATA_SHA256,
    )
    _locked_boolean(
        fixed_a["source_seed_excluded_from_phase2"],
        "policy.fixed_lora_a.source_seed_excluded_from_phase2",
        True,
    )
    return max_response_tokens


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
            "transform_calibration",
        },
    )
    _locked_string(oracle["model"], "oracle.model", _ORACLE_MODEL)
    _locked_string(oracle["revision"], "oracle.revision", _ORACLE_REVISION)
    _locked_string(oracle["dtype"], "oracle.dtype", "float32")
    _locked_string(
        oracle["transform"],
        "oracle.transform",
        "robust_center_scale_then_tanh",
    )
    _locked_string(oracle["robust_scale"], "oracle.robust_scale", "scaled_mad")
    _locked_number(oracle["robust_scale_floor"], "oracle.robust_scale_floor", 1.0e-6)
    calibration = _keys(
        oracle["transform_calibration"],
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
    _locked_string(
        calibration["mode"],
        "oracle.transform_calibration.mode",
        "frozen_global",
    )
    _locked_number(
        calibration["b"],
        "oracle.transform_calibration.b",
        PHASE2_FROZEN_ORACLE_B,
    )
    _locked_number(
        calibration["tau"],
        "oracle.transform_calibration.tau",
        PHASE2_FROZEN_ORACLE_TAU,
    )
    _locked_string(
        calibration["aggregation_rule"],
        "oracle.transform_calibration.aggregation_rule",
        "componentwise_median",
    )
    _locked_string(
        calibration["source_split"],
        "oracle.transform_calibration.source_split",
        "train",
    )
    _locked_string(
        calibration["source_config"],
        "oracle.transform_calibration.source_config",
        PHASE1_MAIN_CONFIG,
    )
    _locked_string(
        calibration["source_config_hash"],
        "oracle.transform_calibration.source_config_hash",
        PHASE1_MAIN_CONFIG_HASH,
    )
    _locked_boolean(
        calibration["source_seeds_excluded_from_phase2"],
        "oracle.transform_calibration.source_seeds_excluded_from_phase2",
        True,
    )
    source_artifacts = _sequence(
        calibration["source_artifacts"],
        "oracle.transform_calibration.source_artifacts",
    )
    parsed_artifacts: list[tuple[int, str]] = []
    for index, raw_artifact in enumerate(source_artifacts):
        path = f"oracle.transform_calibration.source_artifacts[{index}]"
        artifact = _keys(
            raw_artifact,
            path=path,
            required={"seed", "metadata_sha256"},
        )
        seed = _integer(
            artifact["seed"],
            f"{path}.seed",
            minimum=0,
            maximum=2**63 - 1,
        )
        metadata_sha256 = _string(
            artifact["metadata_sha256"],
            f"{path}.metadata_sha256",
        )
        if _HEX_DIGEST_PATTERN.fullmatch(metadata_sha256) is None:
            raise ConfigError(f"{path}.metadata_sha256 must be a lowercase SHA-256 digest")
        parsed_artifacts.append((seed, metadata_sha256))
    if tuple(parsed_artifacts) != PHASE2_FROZEN_ORACLE_SOURCE_ARTIFACTS:
        raise ConfigError(
            "oracle.transform_calibration.source_artifacts must equal the five "
            "ordered, Phase-2-excluded Phase-1 metadata identities"
        )
    if frozenset(seed for seed, _ in parsed_artifacts) != PHASE1_MAIN_SEEDS:
        raise ConfigError(
            "oracle.transform_calibration source seeds must equal the Phase-1 main seeds"
        )
    return _locked_number(oracle["probability_floor"], "oracle.probability_floor", 0.25)


def _validate_annotations(value: object, probability_floor: float) -> None:
    annotations = _keys(
        value,
        path="annotations",
        required={
            "scheme",
            "gamma",
            "independent_replicates_per_edge",
            "replicate_reduction",
            "prohibit_clipping",
            "bt_label_use",
            "replicate_rng",
            "decision_gates",
        },
    )
    _locked_string(
        annotations["scheme"],
        "annotations.scheme",
        "geometric_randomized_truncation",
    )
    gamma = _locked_number(annotations["gamma"], "annotations.gamma", 0.9)
    if gamma <= 1.0 - probability_floor:
        raise ConfigError(
            "annotations.gamma must exceed 1 - oracle.probability_floor for finite variance"
        )
    _locked_integer(
        annotations["independent_replicates_per_edge"],
        "annotations.independent_replicates_per_edge",
        4,
    )
    _locked_string(
        annotations["replicate_reduction"],
        "annotations.replicate_reduction",
        "arithmetic_mean",
    )
    _locked_boolean(
        annotations["prohibit_clipping"],
        "annotations.prohibit_clipping",
        True,
    )
    _locked_string(
        annotations["bt_label_use"],
        "annotations.bt_label_use",
        "all_underlying_bernoulli_labels",
    )
    _locked_string(
        annotations["replicate_rng"],
        "annotations.replicate_rng",
        "single_named_generator_sequential_independent_draws_with_preserved_boundaries",
    )
    decisions = _keys(
        annotations["decision_gates"],
        path="annotations.decision_gates",
        required={"action", "require_all"},
    )
    _locked_string(
        decisions["action"],
        "annotations.decision_gates.action",
        "fail_closed",
    )
    _locked_string_sequence(
        decisions["require_all"],
        "annotations.decision_gates.require_all",
        (
            "exactly_four_replicate_boundaries",
            "single_generator_initial_final_state_and_draw_count",
            "replicate_tensor_hashes_preserve_boundaries",
            "no_label_clipping",
            "bt_uses_all_raw_bernoulli_labels",
            "prorm_uses_arithmetic_mean_of_four_unclipped_estimators",
        ),
    )


def _validate_recovery_optimizer_protocol(value: object) -> None:
    protocol = _keys(
        value,
        path="reward_model.optimizer_protocol",
        required={
            "schema_version",
            "one_time_recovery",
            "scope",
            "initialization",
            "learning_rate_schedule",
            "legacy_constant_lr_boundary_snapshot_steps",
            "state_transition",
            "adamw",
            "reward_head_dtype",
            "first_order_audit_dtype",
            "microbatch_order",
            "optimizer_state_reset_at_lr_milestone",
            "one_optimizer_update_per_step",
            "tie_break",
            "validation_or_test_selection",
        },
    )
    _locked_string(
        protocol["schema_version"],
        "reward_model.optimizer_protocol.schema_version",
        "deterministic-adamw-lr-decay-recovery/v1",
    )
    _locked_boolean(
        protocol["one_time_recovery"],
        "reward_model.optimizer_protocol.one_time_recovery",
        True,
    )
    _locked_string(
        protocol["scope"],
        "reward_model.optimizer_protocol.scope",
        "every_phase2_first_order_convergence_trainer",
    )
    _locked_string(
        protocol["initialization"],
        "reward_model.optimizer_protocol.initialization",
        "exact_zero_head_and_fresh_optimizer_state",
    )
    schedule = _keys(
        protocol["learning_rate_schedule"],
        path="reward_model.optimizer_protocol.learning_rate_schedule",
        required={
            "update_indexing",
            "application",
            "stages",
            "schedule_sha256",
        },
    )
    _locked_string(
        schedule["update_indexing"],
        "reward_model.optimizer_protocol.learning_rate_schedule.update_indexing",
        "one_indexed_inclusive",
    )
    _locked_string(
        schedule["application"],
        "reward_model.optimizer_protocol.learning_rate_schedule.application",
        "set_learning_rate_immediately_before_optimizer_update",
    )
    stages = _sequence(
        schedule["stages"],
        "reward_model.optimizer_protocol.learning_rate_schedule.stages",
    )
    if len(stages) != len(_RECOVERY_LR_STAGES):
        raise ConfigError(
            "reward_model.optimizer_protocol.learning_rate_schedule.stages "
            f"must contain exactly {len(_RECOVERY_LR_STAGES)} stages"
        )
    normalized_stages: list[dict[str, object]] = []
    for index, (stage_value, expected) in enumerate(zip(stages, _RECOVERY_LR_STAGES, strict=True)):
        path = f"reward_model.optimizer_protocol.learning_rate_schedule.stages[{index}]"
        stage = _keys(
            stage_value,
            path=path,
            required={"first_update", "last_update", "learning_rate"},
        )
        first_update = _locked_integer(
            stage["first_update"],
            f"{path}.first_update",
            expected[0],
        )
        last_update = _locked_integer(
            stage["last_update"],
            f"{path}.last_update",
            expected[1],
        )
        learning_rate = _locked_number(
            stage["learning_rate"],
            f"{path}.learning_rate",
            expected[2],
        )
        normalized_stages.append(
            {
                "first_update": first_update,
                "last_update": last_update,
                "learning_rate": learning_rate,
            }
        )
    schedule_payload = {
        "update_indexing": schedule["update_indexing"],
        "application": schedule["application"],
        "stages": normalized_stages,
    }
    actual_schedule_sha256 = config_hash(schedule_payload)
    declared_schedule_sha256 = _string(
        schedule["schedule_sha256"],
        "reward_model.optimizer_protocol.learning_rate_schedule.schedule_sha256",
    )
    if _HEX_DIGEST_PATTERN.fullmatch(declared_schedule_sha256) is None:
        raise ConfigError(
            "reward_model.optimizer_protocol.learning_rate_schedule.schedule_sha256 "
            "must be a lowercase SHA256 digest"
        )
    if declared_schedule_sha256 != actual_schedule_sha256:
        raise ConfigError(
            "reward_model.optimizer_protocol.learning_rate_schedule.schedule_sha256 "
            "does not bind the declared schedule"
        )
    if actual_schedule_sha256 != PHASE2_RECOVERY_LR_SCHEDULE_SHA256:
        raise ConfigError("recovery learning-rate schedule does not match the locked protocol")
    _locked_integer(
        protocol["legacy_constant_lr_boundary_snapshot_steps"],
        "reward_model.optimizer_protocol.legacy_constant_lr_boundary_snapshot_steps",
        _RECOVERY_LEGACY_BOUNDARY_STEPS,
    )
    _locked_string(
        protocol["state_transition"],
        "reward_model.optimizer_protocol.state_transition",
        "preserve_all_adamw_moments_across_learning_rate_boundaries",
    )
    adamw = _keys(
        protocol["adamw"],
        path="reward_model.optimizer_protocol.adamw",
        required={
            "betas",
            "eps",
            "amsgrad",
            "maximize",
            "foreach",
            "fused",
            "capturable",
            "differentiable",
        },
    )
    _locked_number_sequence(
        adamw["betas"],
        "reward_model.optimizer_protocol.adamw.betas",
        (0.9, 0.999),
    )
    _locked_number(
        adamw["eps"],
        "reward_model.optimizer_protocol.adamw.eps",
        1.0e-8,
    )
    for field in (
        "amsgrad",
        "maximize",
        "foreach",
        "fused",
        "capturable",
        "differentiable",
    ):
        _locked_boolean(
            adamw[field],
            f"reward_model.optimizer_protocol.adamw.{field}",
            False,
        )
    _locked_string(
        protocol["reward_head_dtype"],
        "reward_model.optimizer_protocol.reward_head_dtype",
        "float32",
    )
    _locked_string(
        protocol["first_order_audit_dtype"],
        "reward_model.optimizer_protocol.first_order_audit_dtype",
        "float64",
    )
    _locked_string(
        protocol["microbatch_order"],
        "reward_model.optimizer_protocol.microbatch_order",
        "canonical_edge_order_contiguous_ascending_no_shuffle",
    )
    _locked_boolean(
        protocol["optimizer_state_reset_at_lr_milestone"],
        "reward_model.optimizer_protocol.optimizer_state_reset_at_lr_milestone",
        False,
    )
    _locked_boolean(
        protocol["one_optimizer_update_per_step"],
        "reward_model.optimizer_protocol.one_optimizer_update_per_step",
        True,
    )
    _locked_string(
        protocol["tie_break"],
        "reward_model.optimizer_protocol.tie_break",
        _RECOVERY_TIE_BREAK,
    )
    _locked_boolean(
        protocol["validation_or_test_selection"],
        "reward_model.optimizer_protocol.validation_or_test_selection",
        False,
    )


def _validate_reward_model(
    value: object,
    *,
    stage: str,
    recovery_protocol: bool,
) -> None:
    required = {
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
        "max_grad_norm",
        "adaptive_convergence",
        "identifiability",
    }
    if recovery_protocol:
        required.add("optimizer_protocol")
    reward = _keys(
        value,
        path="reward_model",
        required=required,
    )
    _locked_string(reward["model"], "reward_model.model", _POLICY_MODEL)
    _locked_string(reward["revision"], "reward_model.revision", _POLICY_REVISION)
    _locked_string(reward["dtype"], "reward_model.dtype", "float32")
    _locked_string(
        reward["parameterization"],
        "reward_model.parameterization",
        "frozen_backbone_linear_head",
    )
    _locked_string(
        reward["feature_pooling"],
        "reward_model.feature_pooling",
        "last_response_token",
    )
    _locked_boolean(reward["linear_head_bias"], "reward_model.linear_head_bias", False)
    _locked_integer(reward["outer_steps"], "reward_model.outer_steps", 720)
    _locked_integer(
        reward["refresh_dual_every_steps"],
        "reward_model.refresh_dual_every_steps",
        1,
    )
    _locked_string(reward["optimizer"], "reward_model.optimizer", "adamw")
    _locked_number(reward["learning_rate"], "reward_model.learning_rate", 1.0e-3)
    _locked_number(reward["weight_decay"], "reward_model.weight_decay", 0.0)
    _locked_integer(reward["microbatch_size"], "reward_model.microbatch_size", 64)
    _locked_number(reward["max_grad_norm"], "reward_model.max_grad_norm", 1.0)
    convergence = _keys(
        reward["adaptive_convergence"],
        path="reward_model.adaptive_convergence",
        required={
            "relative_gradient_ratio_tolerance",
            "minimum_steps",
            "maximum_steps",
            "check_interval_steps",
            "consecutive_passing_checks",
            "compute_matched_checkpoint_steps",
            "gradient_measurement",
            "denominator",
            "denominator_floor",
            "prorm_pcg_audit_initialization",
            "fail_closed",
            "solution_tie_break",
            "unique_solution_claim",
            "validation_or_test_selection",
            "primary_heads_required_to_converge",
        },
    )
    _locked_number(
        convergence["relative_gradient_ratio_tolerance"],
        "reward_model.adaptive_convergence.relative_gradient_ratio_tolerance",
        1.0e-3,
    )
    minimum_steps = _locked_integer(
        convergence["minimum_steps"],
        "reward_model.adaptive_convergence.minimum_steps",
        100,
    )
    expected_maximum_steps = _RECOVERY_MAXIMUM_STEPS if recovery_protocol else 5760
    maximum_steps = _locked_integer(
        convergence["maximum_steps"],
        "reward_model.adaptive_convergence.maximum_steps",
        expected_maximum_steps,
    )
    interval = _locked_integer(
        convergence["check_interval_steps"],
        "reward_model.adaptive_convergence.check_interval_steps",
        20,
    )
    _locked_integer(
        convergence["consecutive_passing_checks"],
        "reward_model.adaptive_convergence.consecutive_passing_checks",
        3,
    )
    checkpoint = _locked_integer(
        convergence["compute_matched_checkpoint_steps"],
        "reward_model.adaptive_convergence.compute_matched_checkpoint_steps",
        720,
    )
    if minimum_steps % interval != 0 or maximum_steps % interval != 0 or checkpoint % interval != 0:
        raise ConfigError(
            "adaptive convergence min/max/checkpoint steps must be scheduled check intervals"
        )
    if checkpoint != reward["outer_steps"]:
        raise ConfigError(
            "reward_model.outer_steps is retained only as the bound 720-step "
            "compute-matched checkpoint"
        )
    _locked_string(
        convergence["gradient_measurement"],
        "reward_model.adaptive_convergence.gradient_measurement",
        "post_update_full_data_unclipped",
    )
    _locked_string(
        convergence["denominator"],
        "reward_model.adaptive_convergence.denominator",
        "exact_zero_initialization_gradient_l2_norm",
    )
    _locked_number(
        convergence["denominator_floor"],
        "reward_model.adaptive_convergence.denominator_floor",
        1.0e-30,
    )
    _locked_string(
        convergence["prorm_pcg_audit_initialization"],
        "reward_model.adaptive_convergence.prorm_pcg_audit_initialization",
        "cold_start_zero",
    )
    _locked_boolean(
        convergence["fail_closed"],
        "reward_model.adaptive_convergence.fail_closed",
        True,
    )
    expected_tie_break = (
        _RECOVERY_TIE_BREAK if recovery_protocol else "zero_initialized_adamw_implicit_bias"
    )
    _locked_string(
        convergence["solution_tie_break"],
        "reward_model.adaptive_convergence.solution_tie_break",
        expected_tie_break,
    )
    _locked_boolean(
        convergence["unique_solution_claim"],
        "reward_model.adaptive_convergence.unique_solution_claim",
        False,
    )
    _locked_boolean(
        convergence["validation_or_test_selection"],
        "reward_model.adaptive_convergence.validation_or_test_selection",
        False,
    )
    _locked_string_sequence(
        convergence["primary_heads_required_to_converge"],
        "reward_model.adaptive_convergence.primary_heads_required_to_converge",
        ("bt_mle", "prorm_plus"),
    )
    identifiability = _keys(
        reward["identifiability"],
        path="reward_model.identifiability",
        required={
            "design_matrix",
            "split",
            "relative_rank_tolerance",
            "role",
            "require_full_column_rank",
            "algorithmic_tie_break",
            "minimum_norm_claim",
            "confirmatory_freeze_requirement",
        },
    )
    _locked_string(
        identifiability["design_matrix"],
        "reward_model.identifiability.design_matrix",
        "reward_feature_difference_design_matrix",
    )
    _locked_string(
        identifiability["split"],
        "reward_model.identifiability.split",
        "train",
    )
    _locked_number(
        identifiability["relative_rank_tolerance"],
        "reward_model.identifiability.relative_rank_tolerance",
        1.0e-10,
    )
    expected_rank_role = (
        "pilot_measure_only" if stage == "pilot" else "confirmatory_frozen_identifiability_contract"
    )
    _locked_string(
        identifiability["role"],
        "reward_model.identifiability.role",
        expected_rank_role,
    )
    require_full_rank = _boolean(
        identifiability["require_full_column_rank"],
        "reward_model.identifiability.require_full_column_rank",
    )
    if stage == "pilot" and require_full_rank:
        raise ConfigError(
            "pilot reward-model rank is measure-only and cannot require full column rank"
        )
    _locked_string(
        identifiability["algorithmic_tie_break"],
        "reward_model.identifiability.algorithmic_tie_break",
        expected_tie_break,
    )
    _locked_boolean(
        identifiability["minimum_norm_claim"],
        "reward_model.identifiability.minimum_norm_claim",
        False,
    )
    expected_freeze = (
        "decide_gate_from_train_only_pilot_then_issue_new_identity"
        if stage == "pilot"
        else "satisfied_by_current_confirmatory_identity"
    )
    _locked_string(
        identifiability["confirmatory_freeze_requirement"],
        "reward_model.identifiability.confirmatory_freeze_requirement",
        expected_freeze,
    )
    if recovery_protocol:
        _validate_recovery_optimizer_protocol(reward["optimizer_protocol"])


def _validate_common_beta(
    value: object,
    *,
    stage: str,
    pilot_phase: str | None,
) -> tuple[str, ...]:
    common = _keys(
        value,
        path="objective.common_beta",
        required={
            "calibration_split",
            "calibration_source",
            "rule",
            "frozen_global_beta",
            "beta_source_aggregate_sha256",
            "primary_k_cal",
            "sensitivity_k_cal",
            "sensitivity_frozen_global_beta_multipliers",
            "primary_execution_role",
            "sensitivity_execution_role",
            "sensitivity_executed_separately",
            "sensitivity_eligible_for_primary_claim",
            "shared_by",
            "learner_specific_line_search",
            "learner_specific_norm_rescaling",
            "post_evaluation_retuning",
        },
    )
    if stage == "pilot" and pilot_phase == "calibration":
        expected_calibration_split = "train"
        expected_calibration_source = "transformed_operational_oracle"
        expected_rule = (
            "pilot_seed_candidate_from_oracle_train_fisher_quadratic_for_future_global_beta"
        )
    elif stage == "pilot" and pilot_phase == "freeze":
        expected_calibration_split = "excluded_pilot_calibration"
        expected_calibration_source = (
            "frozen_calibration_aggregate_candidate_in_pilot_freeze_design_identity"
        )
        expected_rule = "pilot_fixed_global_beta_target_free_safety_rehearsal"
    else:
        expected_calibration_split = "excluded_pilot"
        expected_calibration_source = "frozen_pilot_global_beta_in_confirmatory_design_identity"
        expected_rule = "single_pilot_frozen_global_beta_scalar"
    _locked_string(
        common["calibration_split"],
        "objective.common_beta.calibration_split",
        expected_calibration_split,
    )
    _locked_string(
        common["calibration_source"],
        "objective.common_beta.calibration_source",
        expected_calibration_source,
    )
    _locked_string(
        common["rule"],
        "objective.common_beta.rule",
        expected_rule,
    )
    frozen_global_beta = common["frozen_global_beta"]
    beta_source_aggregate_sha256 = common["beta_source_aggregate_sha256"]
    if stage == "pilot" and pilot_phase == "calibration":
        if frozen_global_beta is not None:
            raise ConfigError(
                "pilot calibration objective.common_beta.frozen_global_beta must be null; "
                "this stage only produces per-seed candidates"
            )
        if beta_source_aggregate_sha256 is not None:
            raise ConfigError(
                "pilot calibration objective.common_beta.beta_source_aggregate_sha256 must be null"
            )
    else:
        _number(
            frozen_global_beta,
            "objective.common_beta.frozen_global_beta",
            minimum=0.0,
            minimum_inclusive=False,
        )
        digest = _string(
            beta_source_aggregate_sha256,
            "objective.common_beta.beta_source_aggregate_sha256",
        )
        if _HEX_DIGEST_PATTERN.fullmatch(digest) is None:
            raise ConfigError(
                "objective.common_beta.beta_source_aggregate_sha256 must be a "
                "64-character lowercase SHA256 digest"
            )
    primary = _locked_number(
        common["primary_k_cal"],
        "objective.common_beta.primary_k_cal",
        0.003,
    )
    if stage == "pilot" and pilot_phase == "calibration":
        sensitivity = _locked_number_sequence(
            common["sensitivity_k_cal"],
            "objective.common_beta.sensitivity_k_cal",
            (0.001, 0.01),
        )
        if not sensitivity[0] < primary < sensitivity[1]:
            raise ConfigError(
                "objective.common_beta.primary_k_cal must be bracketed by its "
                "pilot sensitivity values"
            )
        if common["sensitivity_frozen_global_beta_multipliers"] is not None:
            raise ConfigError(
                "pilot calibration objective.common_beta."
                "sensitivity_frozen_global_beta_multipliers must be null"
            )
    elif stage == "pilot":
        if common["sensitivity_k_cal"] is not None:
            raise ConfigError(
                "pilot freeze objective.common_beta.sensitivity_k_cal must be null; "
                "current-seed curvature cannot change the frozen candidate"
            )
        if common["sensitivity_frozen_global_beta_multipliers"] is not None:
            raise ConfigError(
                "pilot freeze objective.common_beta."
                "sensitivity_frozen_global_beta_multipliers must be null; "
                "a failed safety rehearsal requires a new identity at beta*=2"
            )
    else:
        if common["sensitivity_k_cal"] is not None:
            raise ConfigError(
                "confirmatory objective.common_beta.sensitivity_k_cal must be null; "
                "formal sensitivities may only multiply the frozen global beta"
            )
        _locked_number_sequence(
            common["sensitivity_frozen_global_beta_multipliers"],
            "objective.common_beta.sensitivity_frozen_global_beta_multipliers",
            (0.5, 2.0),
        )
    if stage == "pilot" and pilot_phase == "calibration":
        expected_primary_role = "pilot_global_beta_calibration_candidate"
        expected_sensitivity_role = "required_separate_global_beta_candidate_sensitivity"
    elif stage == "pilot":
        expected_primary_role = "pilot_frozen_global_beta_safety_rehearsal"
        expected_sensitivity_role = "new_pilot_freeze_design_identity_double_beta_grid"
    else:
        expected_primary_role = "confirmatory_primary"
        expected_sensitivity_role = "required_separate_frozen_global_beta_multiplier_sensitivity"
    _locked_string(
        common["primary_execution_role"],
        "objective.common_beta.primary_execution_role",
        expected_primary_role,
    )
    _locked_string(
        common["sensitivity_execution_role"],
        "objective.common_beta.sensitivity_execution_role",
        expected_sensitivity_role,
    )
    _locked_boolean(
        common["sensitivity_executed_separately"],
        "objective.common_beta.sensitivity_executed_separately",
        True,
    )
    _locked_boolean(
        common["sensitivity_eligible_for_primary_claim"],
        "objective.common_beta.sensitivity_eligible_for_primary_claim",
        False,
    )
    shared_by = tuple(_unique_strings(common["shared_by"], "objective.common_beta.shared_by"))
    if shared_by != _COMMON_BETA_ARMS:
        raise ConfigError(
            "objective.common_beta.shared_by must list BT-MLE, ProRM+, and oracle_step "
            "exactly once in the locked order"
        )
    _locked_boolean(
        common["learner_specific_line_search"],
        "objective.common_beta.learner_specific_line_search",
        False,
    )
    _locked_boolean(
        common["learner_specific_norm_rescaling"],
        "objective.common_beta.learner_specific_norm_rescaling",
        False,
    )
    _locked_boolean(
        common["post_evaluation_retuning"],
        "objective.common_beta.post_evaluation_retuning",
        False,
    )
    return shared_by


def _validate_full_tangent(
    value: object,
    train_fisher_nodes: int,
    *,
    stage: str,
) -> None:
    tangent = _keys(
        value,
        path="objective.full_tangent",
        required={"kind", "ridge"},
    )
    _locked_string(
        tangent["kind"],
        "objective.full_tangent.kind",
        "full_fixed_a_lora_b",
    )
    ridge = _keys(
        tangent["ridge"],
        path="objective.full_tangent.ridge",
        required={
            "enabled",
            "rule",
            "relative_coefficient",
            "sensitivity_multipliers",
            "primary_execution_role",
            "sensitivity_execution_role",
            "sensitivity_executed_separately",
            "sensitivity_eligible_for_primary_claim",
            "solver_dtype",
            "pcg_tolerance",
            "pcg_max_iterations",
        },
    )
    _locked_boolean(ridge["enabled"], "objective.full_tangent.ridge.enabled", True)
    _locked_string(
        ridge["rule"],
        "objective.full_tangent.ridge.rule",
        "relative_to_mean_fisher_diagonal",
    )
    coefficient = _locked_number(
        ridge["relative_coefficient"],
        "objective.full_tangent.ridge.relative_coefficient",
        0.001,
    )
    if coefficient <= 0.0:
        raise ConfigError("the rank-deficient full tangent requires strictly positive ridge")
    _locked_number_sequence(
        ridge["sensitivity_multipliers"],
        "objective.full_tangent.ridge.sensitivity_multipliers",
        (0.1, 1.0, 10.0),
    )
    expected_primary_role = (
        "pilot_candidate_primary" if stage == "pilot" else "confirmatory_primary"
    )
    expected_sensitivity_role = (
        "required_separate_pilot_sensitivity"
        if stage == "pilot"
        else "required_separate_confirmatory_sensitivity"
    )
    _locked_string(
        ridge["primary_execution_role"],
        "objective.full_tangent.ridge.primary_execution_role",
        expected_primary_role,
    )
    _locked_string(
        ridge["sensitivity_execution_role"],
        "objective.full_tangent.ridge.sensitivity_execution_role",
        expected_sensitivity_role,
    )
    _locked_boolean(
        ridge["sensitivity_executed_separately"],
        "objective.full_tangent.ridge.sensitivity_executed_separately",
        True,
    )
    _locked_boolean(
        ridge["sensitivity_eligible_for_primary_claim"],
        "objective.full_tangent.ridge.sensitivity_eligible_for_primary_claim",
        False,
    )
    _locked_string(
        ridge["solver_dtype"],
        "objective.full_tangent.ridge.solver_dtype",
        "float64",
    )
    _locked_number(
        ridge["pcg_tolerance"],
        "objective.full_tangent.ridge.pcg_tolerance",
        1.0e-5,
    )
    max_iterations = _integer(
        ridge["pcg_max_iterations"],
        "objective.full_tangent.ridge.pcg_max_iterations",
        minimum=1,
    )
    if max_iterations < train_fisher_nodes + 1:
        raise ConfigError(
            "objective.full_tangent.ridge.pcg_max_iterations must cover the train "
            f"Fisher rank bound plus one ({train_fisher_nodes + 1})"
        )


def _validate_objective(
    value: object,
    train_fisher_nodes: int,
    *,
    stage: str,
    pilot_phase: str | None,
) -> tuple[str, ...]:
    objective = _keys(
        value,
        path="objective",
        required={"common_beta", "full_tangent"},
    )
    shared_by = _validate_common_beta(
        objective["common_beta"],
        stage=stage,
        pilot_phase=pilot_phase,
    )
    _validate_full_tangent(
        objective["full_tangent"],
        train_fisher_nodes,
        stage=stage,
    )
    return shared_by


def _validate_positive_controls(value: object, train_fisher_nodes: int) -> None:
    controls = _keys(
        value,
        path="positive_controls",
        required={
            "direct_oracle_geometry",
            "exact_margin",
            "exact_soft_label_bt",
            "low_dimensional_tangent",
            "numeric_gate_tolerances",
            "decision_gates",
        },
    )
    direct = _keys(
        controls["direct_oracle_geometry"],
        path="positive_controls.direct_oracle_geometry",
        required={"enabled", "construction", "role", "eligible_for_primary_claim"},
    )
    _locked_boolean(
        direct["enabled"],
        "positive_controls.direct_oracle_geometry.enabled",
        True,
    )
    _locked_string(
        direct["construction"],
        "positive_controls.direct_oracle_geometry.construction",
        "complete_pair_u_statistic_equals_node_covariance",
    )
    _locked_string(
        direct["role"],
        "positive_controls.direct_oracle_geometry.role",
        "algebraic_identity_positive_control",
    )
    _locked_boolean(
        direct["eligible_for_primary_claim"],
        "positive_controls.direct_oracle_geometry.eligible_for_primary_claim",
        False,
    )
    exact = _keys(
        controls["exact_margin"],
        path="positive_controls.exact_margin",
        required={"enabled", "h_source", "role", "eligible_for_primary_claim"},
    )
    _locked_boolean(exact["enabled"], "positive_controls.exact_margin.enabled", True)
    _locked_string(
        exact["h_source"],
        "positive_controls.exact_margin.h_source",
        "transformed_oracle_reward_difference",
    )
    _locked_string(
        exact["role"],
        "positive_controls.exact_margin.role",
        "zero_noise_mechanism_positive_control",
    )
    _locked_boolean(
        exact["eligible_for_primary_claim"],
        "positive_controls.exact_margin.eligible_for_primary_claim",
        False,
    )
    exact_soft_bt = _keys(
        controls["exact_soft_label_bt"],
        path="positive_controls.exact_soft_label_bt",
        required={
            "enabled",
            "role",
            "noise_free",
            "input",
            "eligible_for_primary_claim",
        },
    )
    _locked_boolean(
        exact_soft_bt["enabled"],
        "positive_controls.exact_soft_label_bt.enabled",
        True,
    )
    _locked_string(
        exact_soft_bt["role"],
        "positive_controls.exact_soft_label_bt.role",
        "noise_free_positive_control_and_secondary_misspecification_diagnostic",
    )
    _locked_boolean(
        exact_soft_bt["noise_free"],
        "positive_controls.exact_soft_label_bt.noise_free",
        True,
    )
    _locked_string(
        exact_soft_bt["input"],
        "positive_controls.exact_soft_label_bt.input",
        "sigmoid_of_train_transformed_oracle_margin",
    )
    _locked_boolean(
        exact_soft_bt["eligible_for_primary_claim"],
        "positive_controls.exact_soft_label_bt.eligible_for_primary_claim",
        False,
    )

    low_dimensional = _keys(
        controls["low_dimensional_tangent"],
        path="positive_controls.low_dimensional_tangent",
        required={
            "enabled",
            "construction",
            "dimension",
            "seed_namespace",
            "regularization",
            "relative_eigenvalue_tolerance",
            "eligible_for_primary_claim",
        },
    )
    _locked_boolean(
        low_dimensional["enabled"],
        "positive_controls.low_dimensional_tangent.enabled",
        True,
    )
    _locked_string(
        low_dimensional["construction"],
        "positive_controls.low_dimensional_tangent.construction",
        "seeded_orthonormal_projection",
    )
    dimension = _locked_integer(
        low_dimensional["dimension"],
        "positive_controls.low_dimensional_tangent.dimension",
        256,
    )
    if dimension >= train_fisher_nodes:
        raise ConfigError(
            "positive_controls.low_dimensional_tangent.dimension must be smaller "
            "than the train Fisher node count"
        )
    _locked_string(
        low_dimensional["seed_namespace"],
        "positive_controls.low_dimensional_tangent.seed_namespace",
        "prorm-common-beta-low-dimensional-tangent-v1",
    )
    _locked_string(
        low_dimensional["regularization"],
        "positive_controls.low_dimensional_tangent.regularization",
        "moore_penrose_pseudoinverse",
    )
    tolerance = _number(
        low_dimensional["relative_eigenvalue_tolerance"],
        "positive_controls.low_dimensional_tangent.relative_eigenvalue_tolerance",
        minimum=0.0,
        minimum_inclusive=False,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if tolerance != 1.0e-10:
        raise ConfigError(
            "positive_controls.low_dimensional_tangent.relative_eigenvalue_tolerance "
            "must equal 1e-10"
        )
    _locked_boolean(
        low_dimensional["eligible_for_primary_claim"],
        "positive_controls.low_dimensional_tangent.eligible_for_primary_claim",
        False,
    )

    tolerances = _keys(
        controls["numeric_gate_tolerances"],
        path="positive_controls.numeric_gate_tolerances",
        required={
            "direct_identity_absolute_error",
            "direct_identity_relative_error",
            "objective_binding_relative_error",
            "objective_binding_absolute_error",
            "outer_relative_gradient_ratio",
            "low_dimensional_orthonormality_max_absolute_error",
            "low_dimensional_pseudoinverse_relative_residual",
            "low_dimensional_scatter_max_absolute_error",
            "low_dimensional_score_identity_max_absolute_error",
        },
    )
    locked_tolerances = {
        "direct_identity_absolute_error": 1.0e-10,
        "direct_identity_relative_error": 1.0e-10,
        "objective_binding_relative_error": 2.0e-5,
        "objective_binding_absolute_error": 2.0e-7,
        "outer_relative_gradient_ratio": 1.0e-3,
        "low_dimensional_orthonormality_max_absolute_error": 1.0e-10,
        "low_dimensional_pseudoinverse_relative_residual": 1.0e-6,
        "low_dimensional_scatter_max_absolute_error": 1.0e-4,
        "low_dimensional_score_identity_max_absolute_error": 1.0e-4,
    }
    for name, expected in locked_tolerances.items():
        _locked_number(
            tolerances[name],
            f"positive_controls.numeric_gate_tolerances.{name}",
            expected,
        )
    if tolerances["outer_relative_gradient_ratio"] != 1.0e-3:
        raise ConfigError(
            "positive-control outer gradient tolerance must match adaptive convergence"
        )

    decision = _keys(
        controls["decision_gates"],
        path="positive_controls.decision_gates",
        required={"action", "unit", "require_all"},
    )
    _locked_string(
        decision["action"],
        "positive_controls.decision_gates.action",
        "fail_closed",
    )
    _locked_string(
        decision["unit"],
        "positive_controls.decision_gates.unit",
        "per_seed",
    )
    _locked_string_sequence(
        decision["require_all"],
        "positive_controls.decision_gates.require_all",
        (
            "direct_oracle_moment_identity",
            "exact_margin_objective_decrease",
            "exact_margin_first_order_convergence",
            "low_dimensional_exact_rank",
            "low_dimensional_orthonormality",
            "low_dimensional_pseudoinverse_residual",
            "low_dimensional_scatter_identity",
            "low_dimensional_score_identity",
            "all_required_pcg_solves_converged",
        ),
    )


def _validate_arms(value: object, shared_by: tuple[str, ...]) -> None:
    arms = _keys(value, path="arms", required=set(_POLICY_ARMS))
    contracts = {
        "zero_b": ("zero", False, "reference"),
        "bt_mle": ("bt_train_natural_direction", True, "learner"),
        "prorm_plus": ("prorm_plus_train_natural_direction", True, "learner"),
        "oracle_step": ("oracle_train_natural_direction", True, "positive_control"),
    }
    configured_common_beta_arms: list[str] = []
    for name in _POLICY_ARMS:
        arm = _keys(
            arms[name],
            path=f"arms.{name}",
            required={"direction", "uses_common_beta", "role"},
        )
        expected_direction, expected_common_beta, expected_role = contracts[name]
        _locked_string(arm["direction"], f"arms.{name}.direction", expected_direction)
        uses_common_beta = _locked_boolean(
            arm["uses_common_beta"],
            f"arms.{name}.uses_common_beta",
            expected_common_beta,
        )
        _locked_string(arm["role"], f"arms.{name}.role", expected_role)
        if uses_common_beta:
            configured_common_beta_arms.append(name)
    if tuple(configured_common_beta_arms) != shared_by:
        raise ConfigError(
            "arms using common beta must exactly match objective.common_beta.shared_by"
        )


def _validate_evaluation(
    value: object,
    num_candidates: int,
    *,
    stage: str,
    pilot_phase: str | None,
    max_response_tokens: int,
) -> None:
    evaluation = _keys(
        value,
        path="evaluation",
        required={
            "rollout_candidates_per_prompt",
            "primary_endpoint",
            "endpoints",
            "primary_kl",
            "secondary_kl",
            "safety",
            "experimental_unit",
            "paired_bootstrap_resamples",
            "paired_bootstrap_seed",
            "seed_level_interval",
            "decision_gates",
            "max_length",
            "report_response_length_diagnostics",
        },
    )
    candidates = _integer(
        evaluation["rollout_candidates_per_prompt"],
        "evaluation.rollout_candidates_per_prompt",
        minimum=1,
    )
    if candidates != num_candidates:
        raise ConfigError("evaluation.rollout_candidates_per_prompt must equal data.num_candidates")
    _locked_string(
        evaluation["primary_endpoint"],
        "evaluation.primary_endpoint",
        "operational_oracle_reward_minus_beta_common_on_policy_kl",
    )
    endpoints = _keys(
        evaluation["endpoints"],
        path="evaluation.endpoints",
        required={"heldout_local_regret", "finite_policy_utility"},
    )
    heldout = _keys(
        endpoints["heldout_local_regret"],
        path="evaluation.endpoints.heldout_local_regret",
        required={"split", "metric", "contrast", "direction"},
    )
    _locked_string(
        heldout["split"],
        "evaluation.endpoints.heldout_local_regret.split",
        "test",
    )
    _locked_string(
        heldout["metric"],
        "evaluation.endpoints.heldout_local_regret.metric",
        "local_regret_at_frozen_global_beta",
    )
    _locked_string(
        heldout["contrast"],
        "evaluation.endpoints.heldout_local_regret.contrast",
        "bt_mle_minus_prorm_plus",
    )
    _locked_string(
        heldout["direction"],
        "evaluation.endpoints.heldout_local_regret.direction",
        "higher_is_better",
    )
    finite = _keys(
        endpoints["finite_policy_utility"],
        path="evaluation.endpoints.finite_policy_utility",
        required={"split", "metric", "contrast", "direction"},
    )
    _locked_string(
        finite["split"],
        "evaluation.endpoints.finite_policy_utility.split",
        "test",
    )
    _locked_string(
        finite["metric"],
        "evaluation.endpoints.finite_policy_utility.metric",
        "operational_oracle_reward_minus_beta_common_on_policy_kl",
    )
    _locked_string(
        finite["contrast"],
        "evaluation.endpoints.finite_policy_utility.contrast",
        "prorm_plus_minus_bt_mle",
    )
    _locked_string(
        finite["direction"],
        "evaluation.endpoints.finite_policy_utility.direction",
        "higher_is_better",
    )
    primary = _keys(
        evaluation["primary_kl"],
        path="evaluation.primary_kl",
        required={"orientation", "trajectories", "token_reduction", "candidate_reduction"},
    )
    _locked_string(
        primary["orientation"],
        "evaluation.primary_kl.orientation",
        "updated_to_reference",
    )
    _locked_string(
        primary["trajectories"],
        "evaluation.primary_kl.trajectories",
        "updated_policy",
    )
    _locked_string(
        primary["token_reduction"],
        "evaluation.primary_kl.token_reduction",
        "sum_per_sequence",
    )
    _locked_string(
        primary["candidate_reduction"],
        "evaluation.primary_kl.candidate_reduction",
        "mean_within_prompt",
    )
    secondary = _keys(
        evaluation["secondary_kl"],
        path="evaluation.secondary_kl",
        required={"orientation", "trajectories", "role"},
    )
    _locked_string(
        secondary["orientation"],
        "evaluation.secondary_kl.orientation",
        "reference_to_updated",
    )
    _locked_string(
        secondary["trajectories"],
        "evaluation.secondary_kl.trajectories",
        "fixed_reference",
    )
    _locked_string(
        secondary["role"],
        "evaluation.secondary_kl.role",
        "diagnostic_only",
    )
    safety = _keys(
        evaluation["safety"],
        path="evaluation.safety",
        required={
            "mean_policy_to_reference_kl_cap",
            "prompt_mean_p95_kl_cap",
            "prompt_mean_p99_kl_cap",
            "prompt_mean_maximum_kl_cap",
            "per_sequence_maximum_kl_cap",
            "action",
            "retune_beta_on_violation",
        },
    )
    for field, expected in (
        ("mean_policy_to_reference_kl_cap", 0.02),
        ("prompt_mean_p95_kl_cap", 0.02),
        ("prompt_mean_p99_kl_cap", 0.05),
        ("prompt_mean_maximum_kl_cap", 0.10),
        ("per_sequence_maximum_kl_cap", 0.20),
    ):
        _locked_number(safety[field], f"evaluation.safety.{field}", expected)
    _locked_string(safety["action"], "evaluation.safety.action", "fail_closed")
    _locked_boolean(
        safety["retune_beta_on_violation"],
        "evaluation.safety.retune_beta_on_violation",
        False,
    )
    _locked_string(evaluation["experimental_unit"], "evaluation.experimental_unit", "seed")
    _locked_integer(
        evaluation["paired_bootstrap_resamples"],
        "evaluation.paired_bootstrap_resamples",
        10000,
    )
    _integer(
        evaluation["paired_bootstrap_seed"],
        "evaluation.paired_bootstrap_seed",
        minimum=0,
        maximum=2**63 - 1,
    )
    interval = _keys(
        evaluation["seed_level_interval"],
        path="evaluation.seed_level_interval",
        required={
            "method",
            "confidence_level",
            "interval_sidedness",
            "effective_component_one_sided_alpha",
            "lower_bound_rule",
            "prompt_rows_used_as_independent_replicates",
            "estimand",
            "test_structure",
            "component_null",
            "component_alternative",
            "multiplicity_adjustment",
        },
    )
    _locked_string(
        interval["method"],
        "evaluation.seed_level_interval.method",
        "paired_seed_percentile_bootstrap",
    )
    _locked_number(
        interval["confidence_level"],
        "evaluation.seed_level_interval.confidence_level",
        0.95,
    )
    _locked_string(
        interval["interval_sidedness"],
        "evaluation.seed_level_interval.interval_sidedness",
        "two_sided",
    )
    _locked_number(
        interval["effective_component_one_sided_alpha"],
        "evaluation.seed_level_interval.effective_component_one_sided_alpha",
        0.025,
    )
    _locked_string(
        interval["lower_bound_rule"],
        "evaluation.seed_level_interval.lower_bound_rule",
        "strictly_greater_than_zero",
    )
    _locked_boolean(
        interval["prompt_rows_used_as_independent_replicates"],
        "evaluation.seed_level_interval.prompt_rows_used_as_independent_replicates",
        False,
    )
    _locked_string(
        interval["estimand"],
        "evaluation.seed_level_interval.estimand",
        (
            "rng_expectation_of_paired_contrast_conditioned_on_frozen_prompt_pool_"
            "models_oracle_and_design"
        ),
    )
    _locked_string(
        interval["test_structure"],
        "evaluation.seed_level_interval.test_structure",
        "intersection_union_single_conjunctive_claim",
    )
    _locked_string(
        interval["component_null"],
        "evaluation.seed_level_interval.component_null",
        "mean_paired_contrast_lte_zero",
    )
    _locked_string(
        interval["component_alternative"],
        "evaluation.seed_level_interval.component_alternative",
        "mean_paired_contrast_gt_zero",
    )
    _locked_string(
        interval["multiplicity_adjustment"],
        "evaluation.seed_level_interval.multiplicity_adjustment",
        "none_for_intersection_union_conjunctive_claim",
    )
    decisions = _keys(
        evaluation["decision_gates"],
        path="evaluation.decision_gates",
        required={"application", "supports_formal_claim", "require_all"},
    )
    if stage == "pilot" and pilot_phase == "calibration":
        expected_application = "pilot_calibration_target_free_selection"
    elif stage == "pilot":
        expected_application = "pilot_freeze_target_free_safety_selection"
    else:
        expected_application = "confirmatory_evidence_decision"
    _locked_string(
        decisions["application"],
        "evaluation.decision_gates.application",
        expected_application,
    )
    _locked_boolean(
        decisions["supports_formal_claim"],
        "evaluation.decision_gates.supports_formal_claim",
        stage == "confirmatory",
    )
    _locked_string_sequence(
        decisions["require_all"],
        "evaluation.decision_gates.require_all",
        (
            "heldout_bt_minus_prorm_plus_interval_lower_positive",
            "finite_prorm_plus_minus_bt_mle_interval_lower_positive",
            "finite_prorm_plus_minus_zero_b_interval_lower_positive",
            "finite_oracle_step_minus_zero_b_interval_lower_positive",
            "all_optimization_gates_pass",
            "all_positive_control_gates_pass",
            "all_kl_safety_gates_pass",
        ),
    )
    max_length = _keys(
        evaluation["max_length"],
        path="evaluation.max_length",
        required={
            "candidate_horizon_tokens",
            "role",
            "measure_only",
            "formal_gate",
            "formal_threshold",
            "allowed_horizon_sequence",
            "horizon_grid_index",
            "parent_pilot_aggregate_sha256",
            "previous_horizon_failed_length_gate",
            "post_pilot_requirement",
        },
    )
    horizon = _integer(
        max_length["candidate_horizon_tokens"],
        "evaluation.max_length.candidate_horizon_tokens",
        minimum=1,
    )
    if horizon != max_response_tokens:
        raise ConfigError(
            "evaluation.max_length.candidate_horizon_tokens must equal policy.max_response_tokens"
        )
    _locked_number_sequence(
        max_length["allowed_horizon_sequence"],
        "evaluation.max_length.allowed_horizon_sequence",
        _HORIZON_SEQUENCE,
    )
    horizon_grid_index = _integer(
        max_length["horizon_grid_index"],
        "evaluation.max_length.horizon_grid_index",
        minimum=0,
        maximum=len(_HORIZON_SEQUENCE) - 1,
    )
    if horizon != _HORIZON_SEQUENCE[horizon_grid_index]:
        raise ConfigError(
            "evaluation.max_length.candidate_horizon_tokens must equal "
            "allowed_horizon_sequence[horizon_grid_index]"
        )
    parent_aggregate = max_length["parent_pilot_aggregate_sha256"]
    previous_failed = _boolean(
        max_length["previous_horizon_failed_length_gate"],
        "evaluation.max_length.previous_horizon_failed_length_gate",
    )
    if horizon_grid_index == 0:
        if previous_failed:
            raise ConfigError(
                "the initial 256-token horizon cannot claim a previous length-gate failure"
            )
    else:
        if not previous_failed:
            raise ConfigError(
                "an escalated horizon requires previous_horizon_failed_length_gate=true"
            )
        digest = _string(
            parent_aggregate,
            "evaluation.max_length.parent_pilot_aggregate_sha256",
        )
        if _HEX_DIGEST_PATTERN.fullmatch(digest) is None:
            raise ConfigError(
                "an escalated horizon requires a lowercase SHA256 parent pilot aggregate"
            )
    if stage == "pilot" and pilot_phase == "calibration":
        if horizon_grid_index == 0 and parent_aggregate is not None:
            raise ConfigError(
                "initial pilot calibration parent_pilot_aggregate_sha256 must be null"
            )
        _locked_string(
            max_length["role"],
            "evaluation.max_length.role",
            "pilot_horizon_selection_input",
        )
        _locked_boolean(
            max_length["measure_only"],
            "evaluation.max_length.measure_only",
            True,
        )
        _locked_boolean(
            max_length["formal_gate"],
            "evaluation.max_length.formal_gate",
            False,
        )
        _locked_number(
            max_length["formal_threshold"],
            "evaluation.max_length.formal_threshold",
            0.05,
        )
        _locked_string(
            max_length["post_pilot_requirement"],
            "evaluation.max_length.post_pilot_requirement",
            "issue_new_pilot_freeze_design_identity",
        )
    elif stage == "pilot":
        digest = _string(
            parent_aggregate,
            "evaluation.max_length.parent_pilot_aggregate_sha256",
        )
        if _HEX_DIGEST_PATTERN.fullmatch(digest) is None:
            raise ConfigError(
                "pilot freeze requires its calibration aggregate SHA256 as horizon parent"
            )
        _locked_string(
            max_length["role"],
            "evaluation.max_length.role",
            "pilot_frozen_global_beta_safety_selection",
        )
        _locked_boolean(
            max_length["measure_only"],
            "evaluation.max_length.measure_only",
            True,
        )
        _locked_boolean(
            max_length["formal_gate"],
            "evaluation.max_length.formal_gate",
            False,
        )
        _locked_number(
            max_length["formal_threshold"],
            "evaluation.max_length.formal_threshold",
            0.05,
        )
        _locked_string(
            max_length["post_pilot_requirement"],
            "evaluation.max_length.post_pilot_requirement",
            "freeze_confirmatory_identity_if_passed_else_double_beta_new_identity",
        )
    else:
        digest = _string(
            parent_aggregate,
            "evaluation.max_length.parent_pilot_aggregate_sha256",
        )
        if _HEX_DIGEST_PATTERN.fullmatch(digest) is None:
            raise ConfigError(
                "confirmatory execution requires its accepted freeze aggregate "
                "SHA256 as horizon parent"
            )
        _locked_string(
            max_length["role"],
            "evaluation.max_length.role",
            "confirmatory_truncation_safety_gate",
        )
        _locked_boolean(
            max_length["measure_only"],
            "evaluation.max_length.measure_only",
            False,
        )
        _locked_boolean(
            max_length["formal_gate"],
            "evaluation.max_length.formal_gate",
            True,
        )
        _number(
            max_length["formal_threshold"],
            "evaluation.max_length.formal_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        _locked_string(
            max_length["post_pilot_requirement"],
            "evaluation.max_length.post_pilot_requirement",
            "satisfied_by_new_confirmatory_design_identity",
        )
    _locked_boolean(
        evaluation["report_response_length_diagnostics"],
        "evaluation.report_response_length_diagnostics",
        True,
    )


def _validate_secondary_experiments(value: object) -> None:
    secondary = _keys(
        value,
        path="secondary_experiments",
        required={"all_six_pairs"},
    )
    all_six = _keys(
        secondary["all_six_pairs"],
        path="secondary_experiments.all_six_pairs",
        required={
            "enabled",
            "construction",
            "execution_role",
            "executed_in_primary_four_arm_run",
            "eligible_for_primary_claim",
        },
    )
    _locked_boolean(
        all_six["enabled"],
        "secondary_experiments.all_six_pairs.enabled",
        True,
    )
    _locked_string(
        all_six["construction"],
        "secondary_experiments.all_six_pairs.construction",
        "all_six_prompt_pairs_u_statistic",
    )
    _locked_string(
        all_six["execution_role"],
        "secondary_experiments.all_six_pairs.execution_role",
        "separate_secondary_efficiency_experiment",
    )
    _locked_boolean(
        all_six["executed_in_primary_four_arm_run"],
        "secondary_experiments.all_six_pairs.executed_in_primary_four_arm_run",
        False,
    )
    _locked_boolean(
        all_six["eligible_for_primary_claim"],
        "secondary_experiments.all_six_pairs.eligible_for_primary_claim",
        False,
    )


def _validate_recovery_control(value: object) -> None:
    recovery = _keys(
        value,
        path="recovery_control",
        required={
            "schema_version",
            "parent_failure_registry",
            "parent_failure_registry_sha256",
            "optimizer_diagnostic_path",
            "optimizer_diagnostic_sha256",
            "optimizer_diagnostic_source_git_commit",
            "optimizer_diagnostic_source_job_id",
            "optimizer_diagnostic_role",
            "parent_phase2_design_sha256",
            "parent_source_job_array_id",
            "parent_seeds",
            "parent_terminal_status",
            "parent_failure_aggregate_present",
            "parent_failure_evidence",
            "artifact_reuse",
            "artifact_producer_identity_separate_from_recovery_training_identity",
            "execution_scope",
            "policy_rollout_allowed",
            "validation_or_test_access_allowed",
            "final_oracle_allowed",
            "downstream_utility_allowed",
            "one_shot_no_further_adaptation",
            "failure_action",
        },
    )
    exact_strings = {
        "schema_version": "prorm-phase2-recovery-control/v1",
        "parent_failure_registry": "configs/phase2_recovery_parent_failures.json",
        "parent_failure_registry_sha256": (
            "7be4ee90b1f494d32f96214f407a57cbee54be86a77dacc1206d2acd527857dc"
        ),
        "optimizer_diagnostic_path": (
            "diagnostics/bt-convergence/seed-20260801-commit-791c2da.json"
        ),
        "optimizer_diagnostic_sha256": (
            "bd7c3d80c26500ee273b14bb1ea8bc3428f71fdb319a49c792bf4de567e2c6a9"
        ),
        "optimizer_diagnostic_source_git_commit": ("791c2daac7f1601f6798d5878bef1770ca9d5ebf"),
        "optimizer_diagnostic_source_job_id": "1647982",
        "optimizer_diagnostic_role": ("train_only_nonconfirmatory_schedule_selection"),
        "parent_phase2_design_sha256": (
            "0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca"
        ),
        "parent_source_job_array_id": "1647491",
        "parent_terminal_status": "FAILED",
        "parent_failure_evidence": (
            "exact_three_seed_registry_binds_each_failed_terminal_and_phase2_run_log"
        ),
        "artifact_reuse": "immutable_parent_materialization_only",
        "execution_scope": "train_only",
        "failure_action": "hard_fail_no_second_recovery",
    }
    for field, expected in exact_strings.items():
        _locked_string(
            recovery[field],
            f"recovery_control.{field}",
            expected,
        )
    parent_seeds = tuple(
        _unique_integers(
            recovery["parent_seeds"],
            "recovery_control.parent_seeds",
        )
    )
    if parent_seeds != _PHASE2_RECOVERY_PILOT_SEEDS:
        raise ConfigError(
            "recovery_control.parent_seeds must equal [20260801, 20260802, 20260803] in that order"
        )
    for field, expected in {
        "parent_failure_aggregate_present": False,
        "artifact_producer_identity_separate_from_recovery_training_identity": True,
        "policy_rollout_allowed": False,
        "validation_or_test_access_allowed": False,
        "final_oracle_allowed": False,
        "downstream_utility_allowed": False,
        "one_shot_no_further_adaptation": True,
    }.items():
        _locked_boolean(
            recovery[field],
            f"recovery_control.{field}",
            expected,
        )


def _validate_recovery_experiment_scope(root: Mapping[str, object]) -> None:
    """Lock the one-time recovery schema to its sole calibration-pilot use."""

    design = _mapping(root["design"], "design")
    _locked_string(
        design.get("stage"),
        "design.stage",
        "pilot",
    )
    _locked_string(
        design.get("pilot_phase"),
        "design.pilot_phase",
        "calibration",
    )
    _locked_string(
        design.get("name"),
        "design.name",
        "common-beta-recovery-pilot-v1",
    )

    run = _mapping(root["run"], "run")
    run_seeds = tuple(_unique_integers(run.get("seeds"), "run.seeds"))
    if run_seeds != _PHASE2_RECOVERY_PILOT_SEEDS:
        raise ConfigError(
            "recovery run.seeds must equal [20260801, 20260802, 20260803] in that order"
        )

    recovery_control = _mapping(root["recovery_control"], "recovery_control")
    parent_seeds = tuple(
        _unique_integers(
            recovery_control.get("parent_seeds"),
            "recovery_control.parent_seeds",
        )
    )
    if parent_seeds != run_seeds:
        raise ConfigError(
            "recovery_control.parent_seeds must exactly match recovery run.seeds in order"
        )


def _at(value: Mapping[str, object], path: str) -> object:
    result: object = value
    for component in path.split("."):
        result = _mapping(result, path)[component]
    return result


def _validate_base_binding(
    phase2: Mapping[str, object],
    base_config: Mapping[str, object],
) -> None:
    base = validate_config(base_config)
    actual_hash = config_hash(base)
    design = _mapping(phase2["design"], "design")
    declared_source = _declared_config_path(
        design["source_config"],
        "design.source_config",
    )
    declared_hash = _string(
        design["source_config_hash"],
        "design.source_config_hash",
    )
    if actual_hash != declared_hash:
        raise ConfigError(
            "base_config semantic identity does not match "
            f"design.source_config_hash for {declared_source}: "
            f"{actual_hash} != {declared_hash}"
        )

    identical_paths = (
        "run.seeds",
        "run.num_prompts",
        "run.split_sizes",
        "data",
        "policy.model",
        "policy.revision",
        "policy.dtype",
        "policy.max_prompt_tokens",
        "policy.max_response_tokens",
        "policy.sampling",
        "policy.lora_rank",
        "policy.lora_alpha",
        "policy.lora_dropout",
        "policy.lora_layers",
        "policy.lora_modules",
        "policy.trainable_tangent_parameters",
        "policy.fixed_lora_a",
        "oracle",
        "reward_model.model",
        "reward_model.revision",
        "reward_model.dtype",
        "reward_model.parameterization",
        "reward_model.feature_pooling",
        "reward_model.linear_head_bias",
        "reward_model.outer_steps",
        "reward_model.refresh_dual_every_steps",
        "reward_model.optimizer",
        "reward_model.learning_rate",
        "reward_model.weight_decay",
        "reward_model.microbatch_size",
        "reward_model.max_grad_norm",
    )
    for path in identical_paths:
        if _at(phase2, path) != _at(base, path):
            raise ConfigError(f"Phase-2 field {path} must remain bound to {declared_source}")

    phase2_annotations = _mapping(phase2["annotations"], "annotations")
    base_annotations = _mapping(base["annotations"], "annotations")
    for field in ("scheme", "gamma"):
        if phase2_annotations[field] != base_annotations[field]:
            raise ConfigError(f"Phase-2 annotations.{field} must remain bound to {declared_source}")

    ridge = _mapping(
        _mapping(
            _mapping(phase2["objective"], "objective")["full_tangent"],
            "objective.full_tangent",
        )["ridge"],
        "objective.full_tangent.ridge",
    )
    base_objective = _mapping(base["objective"], "objective")
    paired_ridge_fields = {
        "relative_coefficient": "damping_relative_to_mean_fisher_diagonal",
        "solver_dtype": "pcg_dtype",
        "pcg_tolerance": "pcg_tolerance",
        "pcg_max_iterations": "pcg_max_iterations",
    }
    for phase2_field, base_field in paired_ridge_fields.items():
        if ridge[phase2_field] != base_objective[base_field]:
            raise ConfigError(
                f"Phase-2 full-tangent ridge field {phase2_field} must remain bound "
                f"to objective.{base_field} in {declared_source}"
            )
    if ridge["sensitivity_multipliers"] != base_objective["damping_sensitivity_multipliers"]:
        raise ConfigError(
            f"Phase-2 full-tangent ridge sensitivity must remain bound to {declared_source}"
        )

    phase2_evaluation = _mapping(phase2["evaluation"], "evaluation")
    base_evaluation = _mapping(base["evaluation"], "evaluation")
    for field in (
        "rollout_candidates_per_prompt",
        "paired_bootstrap_resamples",
        "paired_bootstrap_seed",
    ):
        if phase2_evaluation[field] != base_evaluation[field]:
            raise ConfigError(f"Phase-2 evaluation.{field} must remain bound to {declared_source}")


def validate_phase2_config(
    config: Mapping[str, object],
    *,
    base_config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate and return an independent common-beta configuration mapping.

    ``base_config`` is optional for in-memory validation and mandatory by
    construction in :func:`load_phase2_config`.  When supplied, its canonical
    hash and every frozen identity/geometry field are checked against the
    Phase-2 mapping.
    """

    schema_hint = config.get("schema_version") if isinstance(config, Mapping) else None
    recovery_schema_hint = schema_hint == PHASE2_RECOVERY_SCHEMA_VERSION
    required_root = {
        "schema_version",
        "design",
        "run",
        "data",
        "policy",
        "oracle",
        "annotations",
        "reward_model",
        "objective",
        "positive_controls",
        "arms",
        "evaluation",
        "secondary_experiments",
    }
    if recovery_schema_hint:
        required_root.add("recovery_control")
    root = _keys(
        config,
        path="config",
        required=required_root,
    )
    schema_version = _string(root["schema_version"], "schema_version")
    if schema_version not in {
        PHASE2_SCHEMA_VERSION,
        PHASE2_RECOVERY_SCHEMA_VERSION,
    }:
        raise ConfigError(
            "schema_version must equal either "
            f"{PHASE2_SCHEMA_VERSION!r} or {PHASE2_RECOVERY_SCHEMA_VERSION!r}"
        )
    recovery_protocol = schema_version == PHASE2_RECOVERY_SCHEMA_VERSION
    if recovery_protocol:
        _validate_recovery_control(root["recovery_control"])
        _validate_recovery_experiment_scope(root)
    stage, pilot_phase, _, _ = _validate_design(root["design"])
    train_prompts, _ = _validate_run(root["run"], stage=stage)
    num_candidates = _validate_data(root["data"])
    max_response_tokens = _validate_policy(root["policy"], stage=stage)
    probability_floor = _validate_oracle(root["oracle"])
    _validate_annotations(root["annotations"], probability_floor)
    _validate_reward_model(
        root["reward_model"],
        stage=stage,
        recovery_protocol=recovery_protocol,
    )
    train_fisher_nodes = train_prompts * num_candidates
    shared_by = _validate_objective(
        root["objective"],
        train_fisher_nodes,
        stage=stage,
        pilot_phase=pilot_phase,
    )
    _validate_positive_controls(root["positive_controls"], train_fisher_nodes)
    _validate_arms(root["arms"], shared_by)
    _validate_evaluation(
        root["evaluation"],
        num_candidates,
        stage=stage,
        pilot_phase=pilot_phase,
        max_response_tokens=max_response_tokens,
    )
    if stage == "confirmatory":
        objective = _mapping(root["objective"], "objective")
        common_beta = _mapping(objective["common_beta"], "objective.common_beta")
        evaluation = _mapping(root["evaluation"], "evaluation")
        max_length = _mapping(evaluation["max_length"], "evaluation.max_length")
        if (
            max_length["parent_pilot_aggregate_sha256"]
            != common_beta["beta_source_aggregate_sha256"]
        ):
            raise ConfigError(
                "a confirmatory design must bind the accepted freeze aggregate as "
                "both evaluation.max_length.parent_pilot_aggregate_sha256 and "
                "objective.common_beta.beta_source_aggregate_sha256"
            )
    _validate_secondary_experiments(root["secondary_experiments"])

    normalized = copy.deepcopy(dict(root))
    if base_config is not None:
        _validate_base_binding(normalized, base_config)
    return normalized


def _load_yaml_module() -> Any:
    try:
        return importlib.import_module("yaml")
    except ModuleNotFoundError as error:
        if error.name != "yaml":
            raise
        raise MissingConfigDependencyError(
            "reading Phase-2 YAML configuration requires PyYAML; install the project "
            "with `pip install -e .` or install `pyyaml>=6,<7`"
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
        raise ConfigError(
            f"failed to parse Phase-2 YAML configuration {source}: {error}"
        ) from error


def _default_base_path(source: Path, declared_source_config: str) -> Path:
    """Resolve the overlay-declared repository-relative materialization path."""

    relative = PurePosixPath(declared_source_config)
    repository_root = source.parent.parent if source.parent.name == "configs" else source.parent
    return repository_root.joinpath(*relative.parts)


def load_phase2_config_bundle(
    path: str | Path,
    *,
    base_config_path: str | Path | None = None,
) -> Phase2ConfigBundle:
    """Load the overlay and expose its validated eight-section source config."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read Phase-2 configuration {source}: {error}") from error
    parsed = _parse_yaml(text, source=source)
    if parsed is None:
        raise ConfigError(f"Phase-2 configuration {source} is empty")
    try:
        preliminary = validate_phase2_config(parsed)
        design = _mapping(preliminary["design"], "design")
        source_config = _declared_config_path(
            design["source_config"],
            "design.source_config",
        )
        declared_base_path = _default_base_path(source, source_config)
        base_path = declared_base_path if base_config_path is None else Path(base_config_path)
        if base_path.resolve() != declared_base_path.resolve():
            raise ConfigError(
                "base_config_path must resolve to the exact overlay-declared "
                f"design.source_config path {declared_base_path}"
            )
        base = load_config(base_path)
        validated = validate_phase2_config(preliminary, base_config=base)
        return Phase2ConfigBundle(
            config=validated,
            base_config=copy.deepcopy(base),
            design_identity=phase2_design_identity(validated),
            source_path=source,
            base_config_path=base_path,
        )
    except ConfigError as error:
        raise ConfigError(f"{source}: {error}") from error


def load_phase2_config(
    path: str | Path,
    *,
    base_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load Phase-2 YAML and verify its materialization-source binding."""

    return load_phase2_config_bundle(
        path,
        base_config_path=base_config_path,
    ).config


def phase2_design_identity(config: Mapping[str, object]) -> str:
    """Return the canonical semantic identity of one validated Phase-2 design."""

    digest = config_hash(validate_phase2_config(config))
    if digest == PHASE1_MAIN_CONFIG_HASH:
        raise ConfigError("Phase-2 design identity must differ from its Phase-1 predecessor")
    return digest


phase2_config_hash = phase2_design_identity


__all__ = [
    "PHASE1_MAIN_CONFIG",
    "PHASE1_MAIN_CONFIG_HASH",
    "PHASE1_MAIN_SEEDS",
    "PHASE2_CONFIRMATORY_EXCLUDED_SEEDS",
    "PHASE2_FIXED_LORA_A_INITIALIZATION_SEED",
    "PHASE2_FIXED_LORA_A_SHA256",
    "PHASE2_FIXED_LORA_A_SOURCE_METADATA_SHA256",
    "PHASE2_FIXED_LORA_A_SOURCE_SEED",
    "PHASE2_CONFIRMATORY_NUM_SEEDS",
    "PHASE2_CONFIRMATORY_SEEDS",
    "PHASE2_FROZEN_ORACLE_B",
    "PHASE2_FROZEN_ORACLE_SOURCE_ARTIFACTS",
    "PHASE2_FROZEN_ORACLE_TAU",
    "PHASE2_MIN_CONFIRMATORY_SEEDS",
    "PHASE2_PILOT_BASE_CONFIG",
    "PHASE2_PILOT_CONFIG",
    "PHASE2_PILOT_SEEDS",
    "PHASE2_RECOVERY_LR_SCHEDULE_SHA256",
    "PHASE2_RECOVERY_PILOT_CONFIG",
    "PHASE2_RECOVERY_SCHEMA_VERSION",
    "PHASE2_SCHEMA_VERSION",
    "Phase2ConfigBundle",
    "load_phase2_config",
    "load_phase2_config_bundle",
    "phase2_config_hash",
    "phase2_design_identity",
    "validate_phase2_config",
]

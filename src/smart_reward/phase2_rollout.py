"""Common-beta finite-policy rollouts with a strict train/test boundary.

This module is the Phase-2 counterpart of :mod:`smart_reward.phase1_rollout`.
It deliberately does *not* reuse the learner-specific matched-KL line search.
Instead it implements the prospective-reward estimand literally:

1. rescore only the saved training candidates with the pinned oracle chat
   template and the frozen Phase-1 robust transform;
2. form the train-oracle natural direction, deriving a per-seed beta candidate
   only in the pilot while confirmatory runs bind the pre-frozen global beta;
3. deploy zero-B, BT-MLE, ProRM+, and the train-oracle direction without any
   learner-specific normalization;
4. estimate per-sequence ``KL(pi_updated || pi_0)`` on each updated policy's
   own sampled histories; and
5. score those rollouts only after every policy quantity is frozen.

The orchestration is dependency injected through a train-only head trainer and
context-managed oracle/policy sessions.  This makes the information boundary,
the ordering of repeated-label construction, and the memory schedule unit
testable without loading Transformers.  A concrete backend can wrap the pinned
Qwen/Skywork loaders already used by Phase 1; a policy and an oracle must never
be resident at the same time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, Protocol

import torch

from . import phase1 as _phase1
from .common_beta import (
    CommonBetaCalibration,
    CommonBetaDirection,
    MeasuredKLSafety,
    assess_measured_kl_safety,
    bind_frozen_common_beta,
    calibrate_common_beta,
    deploy_with_common_beta,
    summarize_downstream_utility,
)
from .config import config_hash
from .contracts import BT_MLE, CANONICAL_LEARNERS, PRORM_PLUS
from .data import CandidateNode
from .experiment import TrainingTensorData
from .linear import DampedEmpiricalFisher, resolve_fisher_solve_dtype
from .oracle import RobustOracleTransform
from .paths import relative_posix_reference
from .phase1_rollout import (
    _publish_staged_pair,
    _sha256_file,
    _stage_json,
    _stage_jsonl,
)
from .phase2_heldout import (
    DeferredHeldoutInputs,
    FrozenHeldoutEvaluationState,
    heldout_evaluation_sha256,
    score_and_evaluate_deferred_heldout,
)
from .prompts import PromptRecord
from .repro import collect_execution_identity
from .rollout import (
    PolicyDirectionResult,
    policy_direction_from_head,
    policy_direction_from_node_rewards,
)
from .scores import ParameterLayout
from .seeding import SeedBundle

PHASE2_RESULT_SCHEMA = "common-beta-finite-policy/v2"
PHASE2_ROLLOUT_SCHEMA = "common-beta-trajectory/v2"
PHASE2_PILOT_RESULT_SCHEMA = "common-beta-pilot-diagnostics/v2"
PHASE2_PILOT_DIAGNOSTIC_SCHEMA = "common-beta-pilot-diagnostic-row/v2"
PHASE2_DESIGN_SCHEMA = "common-beta-design/v4"
PHASE2_ARM_ORDER = ("zero_b", BT_MLE, PRORM_PLUS, "oracle_step")
KL_ORIENTATION = "pi_updated_to_pi0"
KL_HISTORY_SOURCE = "updated_policy"
PILOT_COMMON_BETA_RULE = (
    "pilot_seed_candidate_from_oracle_train_fisher_quadratic_for_future_global_beta"
)
PILOT_FREEZE_COMMON_BETA_RULE = "pilot_fixed_global_beta_target_free_safety_rehearsal"
CONFIRMATORY_COMMON_BETA_RULE = "single_pilot_frozen_global_beta_scalar"
MEAN_POLICY_TO_REFERENCE_KL_CAP = 0.02
PROMPT_MEAN_P95_KL_CAP = 0.02
PROMPT_MEAN_P99_KL_CAP = 0.05
PROMPT_MEAN_MAXIMUM_KL_CAP = 0.10
PER_SEQUENCE_MAXIMUM_KL_CAP = 0.20
REACHED_MAX_LENGTH_RATE_CAP = 0.05
_HEX_DIGITS = frozenset("0123456789abcdef")


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _validate_seed(value: object, *, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be an integer in [0, 2**63 - 1]")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(bytes(tensor.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Phase2Design:
    """Frozen Phase-2 choices, independent of the source artifact config."""

    stage: Literal["pilot", "confirmatory"] = "pilot"
    formal_eligibility: bool = False
    pilot_phase: Literal["calibration", "freeze"] | None = "calibration"
    common_beta_rule: str = PILOT_COMMON_BETA_RULE
    common_beta_calibration_split: str = "train"
    common_beta_source: str = "transformed_operational_oracle"
    frozen_global_beta: float | None = None
    beta_source_aggregate_sha256: str | None = None
    target_oracle_quadratic_kl: float = 0.003
    measured_kl_safety_cap: float = MEAN_POLICY_TO_REFERENCE_KL_CAP
    prompt_mean_p95_kl_cap: float = PROMPT_MEAN_P95_KL_CAP
    prompt_mean_p99_kl_cap: float = PROMPT_MEAN_P99_KL_CAP
    prompt_mean_maximum_kl_cap: float = PROMPT_MEAN_MAXIMUM_KL_CAP
    per_sequence_maximum_kl_cap: float = PER_SEQUENCE_MAXIMUM_KL_CAP
    max_response_tokens: int = 256
    allowed_horizon_sequence: tuple[int, ...] = (256, 512, 1024)
    horizon_grid_index: int = 0
    parent_pilot_aggregate_sha256: str | None = None
    previous_horizon_failed_length_gate: bool = False
    rollout_candidates_per_prompt: int = 4
    relative_damping: float = 0.001
    pcg_dtype: str = "float64"
    pcg_max_iterations: int = 8192
    pcg_tolerance: float = 1.0e-5
    oracle_batch_size: int = 16
    kl_token_chunk_size: int = 4
    k_cal_sensitivity_values: tuple[float, ...] | None = (0.001, 0.01)
    frozen_global_beta_sensitivity_multipliers: tuple[float, ...] | None = None
    ridge_sensitivity_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0)
    max_length_formal_gate: bool = False
    max_length_formal_threshold: float = REACHED_MAX_LENGTH_RATE_CAP

    def __post_init__(self) -> None:
        if self.stage not in {"pilot", "confirmatory"}:
            raise ValueError("stage must be 'pilot' or 'confirmatory'")
        if not isinstance(self.formal_eligibility, bool):
            raise TypeError("formal_eligibility must be bool")
        if self.stage == "pilot" and self.formal_eligibility:
            raise ValueError("a pilot runtime cannot be formally eligible")
        if self.stage == "confirmatory" and not self.formal_eligibility:
            raise ValueError("a confirmatory runtime must be formally eligible")
        if self.stage == "pilot":
            if self.pilot_phase not in {"calibration", "freeze"}:
                raise ValueError("pilot pilot_phase must be 'calibration' or 'freeze'")
        elif self.pilot_phase is not None:
            raise ValueError("confirmatory pilot_phase must be None")
        if self.pilot_phase == "calibration":
            expected_common_beta_rule = PILOT_COMMON_BETA_RULE
            expected_calibration_split = "train"
            expected_common_beta_source = "transformed_operational_oracle"
        elif self.pilot_phase == "freeze":
            expected_common_beta_rule = PILOT_FREEZE_COMMON_BETA_RULE
            expected_calibration_split = "excluded_pilot_calibration"
            expected_common_beta_source = (
                "frozen_calibration_aggregate_candidate_in_pilot_freeze_design_identity"
            )
        else:
            expected_common_beta_rule = CONFIRMATORY_COMMON_BETA_RULE
            expected_calibration_split = "excluded_pilot"
            expected_common_beta_source = "frozen_pilot_global_beta_in_confirmatory_design_identity"
        if self.common_beta_rule != expected_common_beta_rule:
            raise ValueError(
                f"{self.stage} common_beta_rule must equal {expected_common_beta_rule!r}"
            )
        if self.common_beta_calibration_split != expected_calibration_split:
            raise ValueError(
                f"{self.stage} common_beta_calibration_split must equal "
                f"{expected_calibration_split!r}"
            )
        if self.common_beta_source != expected_common_beta_source:
            raise ValueError(
                f"{self.stage} common_beta_source must equal {expected_common_beta_source!r}"
            )
        if self.pilot_phase == "calibration":
            if self.frozen_global_beta is not None:
                raise ValueError("pilot calibration frozen_global_beta must be None")
            if self.beta_source_aggregate_sha256 is not None:
                raise ValueError("pilot calibration beta_source_aggregate_sha256 must be None")
        else:
            _positive_float(
                self.frozen_global_beta,
                name="frozen_global_beta",
            )
            _validate_digest(
                self.beta_source_aggregate_sha256,
                name="beta_source_aggregate_sha256",
            )
        _positive_float(
            self.target_oracle_quadratic_kl,
            name="target_oracle_quadratic_kl",
        )
        safety_thresholds = {
            "measured_kl_safety_cap": (
                self.measured_kl_safety_cap,
                MEAN_POLICY_TO_REFERENCE_KL_CAP,
            ),
            "prompt_mean_p95_kl_cap": (
                self.prompt_mean_p95_kl_cap,
                PROMPT_MEAN_P95_KL_CAP,
            ),
            "prompt_mean_p99_kl_cap": (
                self.prompt_mean_p99_kl_cap,
                PROMPT_MEAN_P99_KL_CAP,
            ),
            "prompt_mean_maximum_kl_cap": (
                self.prompt_mean_maximum_kl_cap,
                PROMPT_MEAN_MAXIMUM_KL_CAP,
            ),
            "per_sequence_maximum_kl_cap": (
                self.per_sequence_maximum_kl_cap,
                PER_SEQUENCE_MAXIMUM_KL_CAP,
            ),
        }
        for name, (value, expected) in safety_thresholds.items():
            observed = _positive_float(value, name=name)
            if observed != expected:
                raise ValueError(f"{name} must equal the preregistered threshold {expected}")
        _positive_integer(self.max_response_tokens, name="max_response_tokens")
        if self.allowed_horizon_sequence != (256, 512, 1024):
            raise ValueError(
                "allowed_horizon_sequence must equal the preregistered sequence (256, 512, 1024)"
            )
        if (
            isinstance(self.horizon_grid_index, bool)
            or not isinstance(self.horizon_grid_index, int)
            or self.horizon_grid_index < 0
            or self.horizon_grid_index >= len(self.allowed_horizon_sequence)
        ):
            raise ValueError("horizon_grid_index must index allowed_horizon_sequence")
        if self.max_response_tokens != self.allowed_horizon_sequence[self.horizon_grid_index]:
            raise ValueError(
                "max_response_tokens must equal allowed_horizon_sequence[horizon_grid_index]"
            )
        if not isinstance(self.previous_horizon_failed_length_gate, bool):
            raise TypeError("previous_horizon_failed_length_gate must be bool")
        if self.horizon_grid_index == 0:
            if self.previous_horizon_failed_length_gate:
                raise ValueError(
                    "the initial response horizon cannot claim a previous length-gate failure"
                )
            if self.pilot_phase == "calibration":
                if self.parent_pilot_aggregate_sha256 is not None:
                    raise ValueError(
                        "initial pilot calibration cannot name a parent pilot aggregate"
                    )
            else:
                _validate_digest(
                    self.parent_pilot_aggregate_sha256,
                    name="parent_pilot_aggregate_sha256",
                )
        else:
            _validate_digest(
                self.parent_pilot_aggregate_sha256,
                name="parent_pilot_aggregate_sha256",
            )
            if not self.previous_horizon_failed_length_gate:
                raise ValueError(
                    "an escalated response horizon requires a failed previous length gate"
                )
        if self.pilot_phase is None and (
            self.parent_pilot_aggregate_sha256 != self.beta_source_aggregate_sha256
        ):
            raise ValueError(
                "confirmatory designs must bind the accepted freeze aggregate as "
                "both horizon parent and beta source"
            )
        _positive_integer(
            self.rollout_candidates_per_prompt,
            name="rollout_candidates_per_prompt",
        )
        _positive_float(self.relative_damping, name="relative_damping")
        resolve_fisher_solve_dtype(self.pcg_dtype)
        _positive_integer(self.pcg_max_iterations, name="pcg_max_iterations")
        _positive_float(self.pcg_tolerance, name="pcg_tolerance")
        _positive_integer(self.oracle_batch_size, name="oracle_batch_size")
        _positive_integer(self.kl_token_chunk_size, name="kl_token_chunk_size")
        if self.pilot_phase == "calibration":
            if self.k_cal_sensitivity_values != (0.001, 0.01):
                raise ValueError("pilot calibration K_cal sensitivities must equal (0.001, 0.01)")
            if self.frozen_global_beta_sensitivity_multipliers is not None:
                raise ValueError("pilot calibration frozen-global-beta sensitivities must be None")
        elif self.pilot_phase == "freeze":
            if self.k_cal_sensitivity_values is not None:
                raise ValueError("pilot freeze K_cal sensitivities must be None")
            if self.frozen_global_beta_sensitivity_multipliers is not None:
                raise ValueError("pilot freeze frozen-global-beta sensitivities must be None")
        else:
            if self.k_cal_sensitivity_values is not None:
                raise ValueError("confirmatory K_cal sensitivities must be None")
            if self.frozen_global_beta_sensitivity_multipliers != (0.5, 2.0):
                raise ValueError(
                    "confirmatory frozen-global-beta multipliers must equal (0.5, 2.0)"
                )
        if self.ridge_sensitivity_multipliers != (0.1, 1.0, 10.0):
            raise ValueError("Phase-2 ridge sensitivities must equal (0.1, 1.0, 10.0)")
        if not isinstance(self.max_length_formal_gate, bool):
            raise TypeError("max_length_formal_gate must be bool")
        threshold = _positive_float(
            self.max_length_formal_threshold,
            name="max_length_formal_threshold",
        )
        if threshold != REACHED_MAX_LENGTH_RATE_CAP:
            raise ValueError(
                "max_length_formal_threshold must equal the preregistered threshold "
                f"{REACHED_MAX_LENGTH_RATE_CAP}"
            )
        if self.stage == "pilot":
            if self.max_length_formal_gate:
                raise ValueError("pilot max-length evidence must be measure-only")
        else:
            if not self.max_length_formal_gate:
                raise ValueError("confirmatory runtime must enforce a max-length gate")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PHASE2_DESIGN_SCHEMA,
            "stage": self.stage,
            "formal_eligibility": self.formal_eligibility,
            "pilot_phase": self.pilot_phase,
            "common_beta_rule": self.common_beta_rule,
            "common_beta_calibration_split": self.common_beta_calibration_split,
            "common_beta_source": self.common_beta_source,
            "frozen_global_beta": self.frozen_global_beta,
            "beta_source_aggregate_sha256": self.beta_source_aggregate_sha256,
            "current_seed_oracle_curvature_role": (
                "pilot_beta_candidate_calibration"
                if self.pilot_phase == "calibration"
                else "predicted_kl_diagnostic_only_no_step_size_authority"
            ),
            "target_oracle_quadratic_kl": self.target_oracle_quadratic_kl,
            "measured_kl_safety_cap": self.measured_kl_safety_cap,
            "prompt_mean_p95_kl_cap": self.prompt_mean_p95_kl_cap,
            "prompt_mean_p99_kl_cap": self.prompt_mean_p99_kl_cap,
            "prompt_mean_maximum_kl_cap": self.prompt_mean_maximum_kl_cap,
            "per_sequence_maximum_kl_cap": self.per_sequence_maximum_kl_cap,
            "max_response_tokens": self.max_response_tokens,
            "allowed_horizon_sequence": list(self.allowed_horizon_sequence),
            "horizon_grid_index": self.horizon_grid_index,
            "parent_pilot_aggregate_sha256": self.parent_pilot_aggregate_sha256,
            "previous_horizon_failed_length_gate": (self.previous_horizon_failed_length_gate),
            "rollout_candidates_per_prompt": self.rollout_candidates_per_prompt,
            "relative_damping": self.relative_damping,
            "pcg_dtype": self.pcg_dtype,
            "pcg_max_iterations": self.pcg_max_iterations,
            "pcg_tolerance": self.pcg_tolerance,
            "oracle_batch_size": self.oracle_batch_size,
            "kl_token_chunk_size": self.kl_token_chunk_size,
            "sensitivity_scope": {
                "pilot_k_cal_candidates": (
                    None
                    if self.k_cal_sensitivity_values is None
                    else list(self.k_cal_sensitivity_values)
                ),
                "frozen_global_beta_multipliers": (
                    None
                    if self.frozen_global_beta_sensitivity_multipliers is None
                    else list(self.frozen_global_beta_sensitivity_multipliers)
                ),
                "sensitivity_step_rule": (
                    "recalibrate_pilot_seed_candidate_from_k_cal"
                    if self.pilot_phase == "calibration"
                    else (
                        "deploy_config_frozen_beta_without_seed_curvature_calibration"
                        if self.pilot_phase == "freeze"
                        else (
                            "multiply_config_frozen_global_beta_without_seed_curvature_calibration"
                        )
                    )
                ),
                "ridge_multipliers_configured": list(self.ridge_sensitivity_multipliers),
                "executed_by_this_runner_invocation": False,
                "result_role": "primary_only",
            },
            "arm_order": list(PHASE2_ARM_ORDER),
            "kl_orientation": KL_ORIENTATION,
            "kl_history_source": KL_HISTORY_SOURCE,
            "learner_specific_line_search": False,
            "calibration_split": "train_only",
            "max_length_gate": {
                "formal_gate": self.max_length_formal_gate,
                "formal_threshold": self.max_length_formal_threshold,
                "measure_only": self.stage == "pilot",
            },
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_phase2_config(cls, config: Mapping[str, object]) -> Phase2Design:
        """Extract runtime choices from an already validated Phase-2 overlay."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a validated Phase-2 mapping")
        try:
            policy = config["policy"]
            design = config["design"]
            data = config["data"]
            objective = config["objective"]
            evaluation = config["evaluation"]
            reward_model = config["reward_model"]
            if not all(
                isinstance(value, Mapping)
                for value in (policy, data, objective, evaluation, reward_model, design)
            ):
                raise TypeError
            common_beta = objective["common_beta"]
            full_tangent = objective["full_tangent"]
            safety = evaluation["safety"]
            max_length = evaluation["max_length"]
            if not all(
                isinstance(value, Mapping)
                for value in (common_beta, full_tangent, safety, max_length)
            ):
                raise TypeError
            ridge = full_tangent["ridge"]
            if not isinstance(ridge, Mapping):
                raise TypeError
            return cls(
                stage=str(design["stage"]),
                formal_eligibility=bool(design["formal_eligibility"]),
                pilot_phase=(None if design["pilot_phase"] is None else str(design["pilot_phase"])),
                common_beta_rule=str(common_beta["rule"]),
                common_beta_calibration_split=str(common_beta["calibration_split"]),
                common_beta_source=str(common_beta["calibration_source"]),
                frozen_global_beta=(
                    None
                    if common_beta["frozen_global_beta"] is None
                    else float(common_beta["frozen_global_beta"])
                ),
                beta_source_aggregate_sha256=(
                    None
                    if common_beta["beta_source_aggregate_sha256"] is None
                    else str(common_beta["beta_source_aggregate_sha256"])
                ),
                target_oracle_quadratic_kl=float(common_beta["primary_k_cal"]),
                measured_kl_safety_cap=float(safety["mean_policy_to_reference_kl_cap"]),
                prompt_mean_p95_kl_cap=float(safety["prompt_mean_p95_kl_cap"]),
                prompt_mean_p99_kl_cap=float(safety["prompt_mean_p99_kl_cap"]),
                prompt_mean_maximum_kl_cap=float(safety["prompt_mean_maximum_kl_cap"]),
                per_sequence_maximum_kl_cap=float(safety["per_sequence_maximum_kl_cap"]),
                max_response_tokens=int(policy["max_response_tokens"]),
                allowed_horizon_sequence=tuple(
                    int(value) for value in max_length["allowed_horizon_sequence"]
                ),
                horizon_grid_index=int(max_length["horizon_grid_index"]),
                parent_pilot_aggregate_sha256=(
                    None
                    if max_length["parent_pilot_aggregate_sha256"] is None
                    else str(max_length["parent_pilot_aggregate_sha256"])
                ),
                previous_horizon_failed_length_gate=bool(
                    max_length["previous_horizon_failed_length_gate"]
                ),
                rollout_candidates_per_prompt=int(evaluation["rollout_candidates_per_prompt"]),
                relative_damping=float(ridge["relative_coefficient"]),
                pcg_dtype=str(ridge["solver_dtype"]),
                pcg_max_iterations=int(ridge["pcg_max_iterations"]),
                pcg_tolerance=float(ridge["pcg_tolerance"]),
                oracle_batch_size=min(16, int(reward_model["microbatch_size"])),
                kl_token_chunk_size=4,
                k_cal_sensitivity_values=(
                    None
                    if common_beta["sensitivity_k_cal"] is None
                    else tuple(float(value) for value in common_beta["sensitivity_k_cal"])
                ),
                frozen_global_beta_sensitivity_multipliers=(
                    None
                    if common_beta["sensitivity_frozen_global_beta_multipliers"] is None
                    else tuple(
                        float(value)
                        for value in common_beta["sensitivity_frozen_global_beta_multipliers"]
                    )
                ),
                ridge_sensitivity_multipliers=tuple(
                    float(value) for value in ridge["sensitivity_multipliers"]
                ),
                max_length_formal_gate=bool(max_length["formal_gate"]),
                max_length_formal_threshold=(float(max_length["formal_threshold"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "config does not expose the validated common-beta runtime contract"
            ) from error


@dataclass(frozen=True, slots=True)
class Phase2PreparedInputs:
    """Validated source objects with no held-out reward tensor."""

    source_config: Mapping[str, object]
    source_config_hash: str
    phase2_config_hash: str
    seed: int
    train: TrainingTensorData
    train_candidates: tuple[CandidateNode, ...]
    test_prompts: tuple[PromptRecord, ...]
    heldout: DeferredHeldoutInputs
    oracle_transform: RobustOracleTransform
    policy_layout: ParameterLayout
    policy_a_sha256: str
    policy_chat_template_sha256: str
    oracle_chat_template_sha256: str
    artifact_dir: Path
    artifact_metadata_sha256: str
    run_manifest: Path
    run_manifest_sha256: str
    environment_identity: Mapping[str, object]
    materialization_prompt_semantics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.source_config, Mapping):
            raise TypeError("source_config must be a mapping")
        _validate_digest(self.source_config_hash, name="source_config_hash")
        if config_hash(self.source_config) != self.source_config_hash:
            raise ValueError("source_config bytes do not match source_config_hash")
        _validate_digest(self.phase2_config_hash, name="phase2_config_hash")
        _validate_seed(self.seed)
        if not isinstance(self.train, TrainingTensorData):
            raise TypeError("train must be TrainingTensorData")
        if not isinstance(self.train_candidates, tuple) or not all(
            isinstance(candidate, CandidateNode) for candidate in self.train_candidates
        ):
            raise TypeError("train_candidates must be a tuple of CandidateNode objects")
        expected_train_count = self.train.num_prompts * self.train.num_candidates
        if len(self.train_candidates) != expected_train_count:
            raise ValueError("train candidate count does not match train tensor geometry")
        expected_prompt_ids = tuple(
            prompt_id
            for prompt_id in self.train.prompt_ids
            for _ in range(self.train.num_candidates)
        )
        if tuple(candidate.prompt_id for candidate in self.train_candidates) != (
            expected_prompt_ids
        ):
            raise ValueError("train_candidates must be prompt-major in train tensor order")
        if not isinstance(self.test_prompts, tuple) or len(self.test_prompts) < 2:
            raise ValueError("test_prompts must contain at least two prompts")
        if not all(
            isinstance(prompt, PromptRecord) and prompt.split == "test"
            for prompt in self.test_prompts
        ):
            raise ValueError("test_prompts must contain only test PromptRecord objects")
        test_ids = tuple(prompt.prompt_id for prompt in self.test_prompts)
        if len(set(test_ids)) != len(test_ids):
            raise ValueError("test prompt IDs must be unique")
        if set(test_ids).intersection(self.train.prompt_ids):
            raise ValueError("train and test prompt IDs must be disjoint")
        if not isinstance(self.heldout, DeferredHeldoutInputs):
            raise TypeError("heldout must be DeferredHeldoutInputs")
        self.heldout.verify_integrity()
        if tuple(self.heldout.test.prompt_ids) != test_ids:
            raise ValueError("policy test prompts and deferred held-out test geometry must align")
        train_ids = set(self.train.prompt_ids)
        if train_ids.intersection(self.heldout.validation.prompt_ids) or train_ids.intersection(
            self.heldout.test.prompt_ids
        ):
            raise ValueError("train and deferred held-out prompt IDs must be disjoint")
        if (
            self.heldout.validation.num_candidates != self.train.num_candidates
            or self.heldout.test.num_candidates != self.train.num_candidates
            or self.heldout.validation.policy_dimension != self.train.policy_dimension
            or self.heldout.test.policy_dimension != self.train.policy_dimension
            or self.heldout.validation.reward_dimension != self.train.reward_dimension
            or self.heldout.test.reward_dimension != self.train.reward_dimension
        ):
            raise ValueError("deferred held-out geometry must match train dimensions")
        if not isinstance(self.oracle_transform, RobustOracleTransform):
            raise TypeError("oracle_transform must be RobustOracleTransform")
        if not isinstance(self.policy_layout, ParameterLayout):
            raise TypeError("policy_layout must be ParameterLayout")
        if self.policy_layout.dimension != self.train.policy_dimension:
            raise ValueError("policy layout dimension does not match train policy scores")
        for name, value in (
            ("policy_a_sha256", self.policy_a_sha256),
            ("policy_chat_template_sha256", self.policy_chat_template_sha256),
            ("oracle_chat_template_sha256", self.oracle_chat_template_sha256),
            ("artifact_metadata_sha256", self.artifact_metadata_sha256),
            ("run_manifest_sha256", self.run_manifest_sha256),
        ):
            _validate_digest(value, name=name)
        if not isinstance(self.artifact_dir, Path) or not isinstance(self.run_manifest, Path):
            raise TypeError("artifact_dir and run_manifest must be pathlib.Path objects")
        if not isinstance(self.environment_identity, Mapping):
            raise TypeError("environment_identity must be a mapping")
        if not isinstance(self.environment_identity.get("formal"), bool):
            raise ValueError("environment_identity must declare a boolean formal field")
        semantics = self.materialization_prompt_semantics
        expected_semantics_keys = {
            "schema_version",
            "policy_chat_template_sha256",
            "encoding",
            "add_generation_prompt",
            "truncation",
            "fail_closed_above_max_prompt_tokens",
            "max_prompt_tokens",
            "num_prompts",
            "minimum_policy_chat_token_count",
            "maximum_policy_chat_token_count",
            "mean_policy_chat_token_count",
            "over_limit_prompt_count",
            "truncated_prompt_count",
            "raw_prompt_preserved_count",
            "records_sha256",
            "candidate_prefixes_verified",
            "records",
        }
        if not isinstance(semantics, Mapping) or set(semantics) != expected_semantics_keys:
            raise ValueError("materialization_prompt_semantics has an invalid schema")
        if (
            semantics["schema_version"] != _phase1._POLICY_PROMPT_SEMANTICS_SCHEMA
            or semantics["policy_chat_template_sha256"] != self.policy_chat_template_sha256
            or semantics["encoding"] != "policy_tokenizer_apply_chat_template"
            or semantics["add_generation_prompt"] is not True
            or semantics["truncation"] is not False
            or semantics["fail_closed_above_max_prompt_tokens"] is not True
            or semantics["over_limit_prompt_count"] != 0
            or semantics["truncated_prompt_count"] != 0
            or semantics["candidate_prefixes_verified"] is not True
        ):
            raise ValueError("materialization prompt semantics violate the full-prompt contract")
        source_policy = self.source_config.get("policy")
        if not isinstance(source_policy, Mapping):
            raise ValueError("source_config.policy must be a mapping")
        max_prompt_tokens = source_policy.get("max_prompt_tokens")
        records = semantics["records"]
        if (
            semantics["max_prompt_tokens"] != max_prompt_tokens
            or isinstance(records, (str, bytes, bytearray))
            or not isinstance(records, Sequence)
            or semantics["num_prompts"] != len(records)
            or semantics["raw_prompt_preserved_count"] != len(records)
            or len(records)
            != (
                self.train.num_prompts
                + len(self.heldout.validation.prompt_ids)
                + len(self.heldout.test.prompt_ids)
            )
        ):
            raise ValueError("materialization prompt semantics do not cover the source artifact")
        _validate_digest(semantics["records_sha256"], name="prompt semantics records_sha256")
        if _phase1._prompt_semantics_records_sha256(records) != semantics["records_sha256"]:
            raise ValueError("materialization prompt semantics record identity changed")
        expected_record_ids = (
            *self.train.prompt_ids,
            *self.heldout.validation.prompt_ids,
            *self.heldout.test.prompt_ids,
        )
        if tuple(record.get("prompt_id") for record in records) != expected_record_ids:
            raise ValueError("materialization prompt semantics records are not in split order")
        candidates_by_prompt: dict[str | int, list[CandidateNode]] = {}
        for candidate in (
            *self.train_candidates,
            *self.heldout.validation.candidates,
            *self.heldout.test.candidates,
        ):
            candidates_by_prompt.setdefault(candidate.prompt_id, []).append(candidate)
        test_text = {prompt.prompt_id: _phase1._prompt_text(prompt) for prompt in self.test_prompts}
        token_counts: list[int] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("materialization prompt semantics record must be a mapping")
            prompt_id = record["prompt_id"]
            nodes = candidates_by_prompt.get(prompt_id)
            if not nodes:
                raise ValueError("materialization prompt record has no candidate nodes")
            prompt_text = nodes[0].prompt
            if any(node.prompt != prompt_text for node in nodes):
                raise ValueError("candidate nodes disagree on their complete raw prompt")
            if prompt_id in test_text and prompt_text != test_text[prompt_id]:
                raise ValueError("test PromptRecord differs from materialized candidate prompt")
            if record.get("raw_prompt_sha256") != _phase1._prompt_text_sha256(prompt_text):
                raise ValueError("materialization prompt record raw-text identity is invalid")
            count = record.get("policy_chat_token_count")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 < count <= max_prompt_tokens
            ):
                raise ValueError("materialization prompt record token count is invalid")
            if (
                record.get("max_prompt_tokens") != max_prompt_tokens
                or record.get("truncated") is not False
                or record.get("raw_prompt_preserved") is not True
            ):
                raise ValueError("materialization prompt record violates full-prompt semantics")
            for node in nodes:
                active = tuple(index for index, value in enumerate(node.response_mask) if value)
                if not active or active[0] != count:
                    raise ValueError("candidate response boundary differs from prompt evidence")
                if _phase1._prompt_token_ids_sha256(node.token_ids[:count]) != record.get(
                    "policy_prompt_token_ids_sha256"
                ):
                    raise ValueError("candidate token prefix differs from prompt evidence")
            token_counts.append(count)
        if (
            semantics["minimum_policy_chat_token_count"] != min(token_counts)
            or semantics["maximum_policy_chat_token_count"] != max(token_counts)
            or float(semantics["mean_policy_chat_token_count"])
            != sum(token_counts) / len(token_counts)
        ):
            raise ValueError("materialization prompt token-count summary is inconsistent")


@dataclass(frozen=True, slots=True)
class Phase2HeadTrainingResult:
    """Frozen BT/ProRM+ heads produced after the train-oracle rescore.

    ``training_design_sha256`` identifies the complete label construction and
    optimization design (for example R=4 independent gamma=0.9 estimators).
    The result deliberately cannot carry a comparison JSON from the old
    single-label-stream Phase-1 design.
    """

    heads: Mapping[str, tuple[float, ...]]
    training_design_sha256: str
    training_arm: str
    audit: Mapping[str, object]
    test_data_accessed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.heads, Mapping) or set(self.heads) != set(CANONICAL_LEARNERS):
            raise ValueError(f"heads must contain exactly {CANONICAL_LEARNERS!r}")
        for name, head in self.heads.items():
            if (
                not isinstance(head, tuple)
                or not head
                or not all(
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in head
                )
            ):
                raise ValueError(f"heads[{name!r}] must be a finite non-empty tuple")
        _validate_digest(
            self.training_design_sha256,
            name="training_design_sha256",
        )
        if not isinstance(self.training_arm, str) or not self.training_arm:
            raise ValueError("training_arm must be a non-empty string")
        if not isinstance(self.audit, Mapping):
            raise TypeError("audit must be a mapping")
        # Enforce JSON safety and finiteness now, before a model can be loaded.
        try:
            json.dumps(
                dict(self.audit),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("head-training audit must be strict JSON data") from error
        if self.test_data_accessed is not False:
            raise ValueError("Phase-2 head training must not access test data")

    @property
    def heads_sha256(self) -> str:
        return _canonical_sha256({name: list(self.heads[name]) for name in CANONICAL_LEARNERS})


@dataclass(frozen=True, slots=True)
class Phase2ArmDeployment:
    """One direct common-beta LoRA-B displacement."""

    arm_name: str
    beta_common: float
    displacement: torch.Tensor
    direction_evidence: Mapping[str, object] | None
    common_beta_evidence: Mapping[str, object] | None

    def __post_init__(self) -> None:
        if self.arm_name not in PHASE2_ARM_ORDER:
            raise ValueError(f"unknown Phase-2 arm {self.arm_name!r}")
        _positive_float(self.beta_common, name="beta_common")
        if (
            not isinstance(self.displacement, torch.Tensor)
            or self.displacement.ndim != 1
            or not self.displacement.is_floating_point()
            or self.displacement.requires_grad
            or not bool(torch.isfinite(self.displacement).all())
        ):
            raise ValueError("displacement must be a finite detached floating vector")
        if self.arm_name == "zero_b":
            if bool(torch.count_nonzero(self.displacement)):
                raise ValueError("zero_b displacement must be exactly zero")
            if self.direction_evidence is not None or self.common_beta_evidence is not None:
                raise ValueError("zero_b must not claim a learned direction")
        elif self.direction_evidence is None or self.common_beta_evidence is None:
            raise ValueError("updated arms require direction and common-beta evidence")


@dataclass(frozen=True, slots=True)
class Phase2Trajectory:
    """One sampled updated-policy trajectory before oracle scoring."""

    arm_name: str
    prompt_id: str
    candidate_index: int
    prompt: str
    raw_prompt_sha256: str
    policy_chat_token_count: int
    policy_prompt_token_ids_sha256: str
    max_prompt_tokens: int
    prompt_truncated: bool
    raw_prompt_preserved: bool
    response: str
    token_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    terminated_by_eos: bool
    reached_max_length: bool
    prompt_rollout_seed: int

    def __post_init__(self) -> None:
        if self.arm_name not in PHASE2_ARM_ORDER:
            raise ValueError("trajectory arm_name is invalid")
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("trajectory prompt_id must be non-empty")
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("trajectory prompt must be non-empty")
        _validate_digest(self.raw_prompt_sha256, name="raw_prompt_sha256")
        if self.raw_prompt_sha256 != _phase1._prompt_text_sha256(self.prompt):
            raise ValueError("trajectory raw prompt SHA256 does not match prompt text")
        _positive_integer(
            self.policy_chat_token_count,
            name="policy_chat_token_count",
        )
        _validate_digest(
            self.policy_prompt_token_ids_sha256,
            name="policy_prompt_token_ids_sha256",
        )
        _positive_integer(self.max_prompt_tokens, name="max_prompt_tokens")
        if self.policy_chat_token_count > self.max_prompt_tokens:
            raise ValueError("trajectory policy prompt exceeds max_prompt_tokens")
        if self.prompt_truncated is not False or self.raw_prompt_preserved is not True:
            raise ValueError("trajectory must prove that the complete raw prompt was preserved")
        if not isinstance(self.response, str):
            raise TypeError("trajectory response must be a string")
        if (
            not isinstance(self.token_ids, tuple)
            or not self.token_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.token_ids
            )
        ):
            raise ValueError("token_ids must be a non-empty tuple of token IDs")
        if (
            not isinstance(self.response_mask, tuple)
            or len(self.response_mask) != len(self.token_ids)
            or any(value not in (0, 1) for value in self.response_mask)
            or not any(self.response_mask)
        ):
            raise ValueError("response_mask must be a non-empty binary token mask")
        active = tuple(index for index, value in enumerate(self.response_mask) if value)
        if active != tuple(range(active[0], active[-1] + 1)):
            raise ValueError("response_mask must select one contiguous response span")
        if active[0] != self.policy_chat_token_count:
            raise ValueError("trajectory prompt token count does not match response-mask boundary")
        if (
            _phase1._prompt_token_ids_sha256(self.token_ids[: self.policy_chat_token_count])
            != self.policy_prompt_token_ids_sha256
        ):
            raise ValueError("trajectory policy prompt token-prefix SHA256 mismatch")
        if not isinstance(self.terminated_by_eos, bool) or not isinstance(
            self.reached_max_length, bool
        ):
            raise TypeError("termination flags must be boolean")
        if self.terminated_by_eos and self.reached_max_length:
            raise ValueError("a trajectory cannot terminate by EOS and hit the length cap")
        _validate_seed(self.prompt_rollout_seed, name="prompt_rollout_seed")

    @property
    def response_token_count(self) -> int:
        return sum(self.response_mask)

    def to_unscored_dict(self, *, beta_common: float) -> dict[str, object]:
        return {
            "schema_version": PHASE2_ROLLOUT_SCHEMA,
            "arm": self.arm_name,
            "policy_source": (
                "zero_b_reference"
                if self.arm_name == "zero_b"
                else "direct_common_beta_displacement"
            ),
            "beta_common": beta_common,
            "prompt_id": self.prompt_id,
            "candidate_index": self.candidate_index,
            "prompt": self.prompt,
            "prompt_semantics": {
                "schema_version": _phase1._POLICY_PROMPT_SEMANTICS_SCHEMA,
                "raw_prompt_sha256": self.raw_prompt_sha256,
                "policy_chat_token_count": self.policy_chat_token_count,
                "policy_prompt_token_ids_sha256": self.policy_prompt_token_ids_sha256,
                "max_prompt_tokens": self.max_prompt_tokens,
                "truncated": self.prompt_truncated,
                "raw_prompt_preserved": self.raw_prompt_preserved,
            },
            "response": self.response,
            "token_ids": list(self.token_ids),
            "response_mask": list(self.response_mask),
            "response_token_count": self.response_token_count,
            "terminated_by_eos": self.terminated_by_eos,
            "reached_max_length": self.reached_max_length,
            "prompt_rollout_seed": self.prompt_rollout_seed,
        }

    def to_pilot_diagnostic_dict(
        self,
        *,
        beta_common: float,
        pilot_phase: Literal["calibration", "freeze"],
        on_policy_kl: float,
    ) -> dict[str, object]:
        """Return the target-free row allowed to leave a pilot job."""

        beta = _positive_float(beta_common, name="beta_common")
        if pilot_phase not in {"calibration", "freeze"}:
            raise ValueError("pilot_phase must be 'calibration' or 'freeze'")
        return {
            "schema_version": PHASE2_PILOT_DIAGNOSTIC_SCHEMA,
            "pilot_phase": pilot_phase,
            "arm": self.arm_name,
            "beta_common": beta,
            "beta_role": (
                "seed_calibration_candidate"
                if pilot_phase == "calibration"
                else "frozen_global_beta_candidate"
            ),
            "prompt_id": self.prompt_id,
            "candidate_index": self.candidate_index,
            "response_token_count": self.response_token_count,
            "terminated_by_eos": self.terminated_by_eos,
            "reached_max_length": self.reached_max_length,
            "prompt_rollout_seed": self.prompt_rollout_seed,
            "kl_orientation": KL_ORIENTATION,
            "kl_history_source": KL_HISTORY_SOURCE,
            "on_policy_kl_pi_updated_to_pi0": on_policy_kl,
            "contains_prompt_text": False,
            "contains_response_text": False,
            "contains_token_ids": False,
            "contains_oracle_outcome": False,
        }


@dataclass(frozen=True, slots=True)
class Phase2PolicyRollout:
    """Policy-session output with an explicit KL estimand contract."""

    arm_name: str
    trajectories: tuple[Phase2Trajectory, ...]
    per_sequence_kl_updated_to_reference: torch.Tensor
    kl_orientation: str = KL_ORIENTATION
    history_source: str = KL_HISTORY_SOURCE

    def __post_init__(self) -> None:
        if self.arm_name not in PHASE2_ARM_ORDER:
            raise ValueError("rollout arm_name is invalid")
        if not self.trajectories or any(
            trajectory.arm_name != self.arm_name for trajectory in self.trajectories
        ):
            raise ValueError("rollout trajectories must be non-empty and share arm_name")
        kl = self.per_sequence_kl_updated_to_reference
        if (
            not isinstance(kl, torch.Tensor)
            or kl.shape != (len(self.trajectories),)
            or not kl.is_floating_point()
            or kl.requires_grad
            or not bool(torch.isfinite(kl).all())
            or bool((kl < 0.0).any())
        ):
            raise ValueError("per-sequence KL must be a finite non-negative detached vector")
        if self.kl_orientation != KL_ORIENTATION:
            raise ValueError("Phase-2 requires KL(pi_updated || pi0), not the reverse orientation")
        if self.history_source != KL_HISTORY_SOURCE:
            raise ValueError("Phase-2 KL histories must be sampled from the updated policy")
        if self.arm_name == "zero_b" and bool(torch.count_nonzero(kl)):
            raise ValueError("zero-B updated-to-reference KL must be exactly zero")


class Phase2OracleSession(Protocol):
    """Transient oracle interface; raw logits never cross this boundary."""

    def score_transformed(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        transform: RobustOracleTransform,
        batch_size: int,
    ) -> torch.Tensor: ...


class Phase2PolicySession(Protocol):
    """Transient policy interface for direct common-beta deployments."""

    def rollout(
        self,
        deployment: Phase2ArmDeployment,
        test_prompts: Sequence[PromptRecord],
        *,
        candidates_per_prompt: int,
        max_response_tokens: int,
        rollout_seed: int,
        kl_token_chunk_size: int,
    ) -> Phase2PolicyRollout: ...


class Phase2HeadTrainer(Protocol):
    """Train-only label construction and reward-head optimization boundary."""

    def train_heads(
        self,
        train: TrainingTensorData,
        train_oracle_rewards: torch.Tensor,
        *,
        seed: int,
    ) -> Phase2HeadTrainingResult: ...


class Phase2RuntimeBackend(Protocol):
    """Factory for mutually exclusive oracle and policy model sessions."""

    def oracle_session(
        self,
        *,
        expected_chat_template_sha256: str,
    ) -> AbstractContextManager[Phase2OracleSession]: ...

    def policy_session(
        self,
        *,
        seed: int,
        expected_a_sha256: str,
        expected_layout: ParameterLayout,
        expected_chat_template_sha256: str,
    ) -> AbstractContextManager[Phase2PolicySession]: ...


class Phase2KLSafetyError(RuntimeError):
    """Raised before final oracle scoring when a confirmatory update exceeds the cap."""

    def __init__(self, safety: MeasuredKLSafety) -> None:
        self.safety = safety
        super().__init__(
            "Phase-2 measured KL safety cap exceeded; beta remains frozen and "
            f"no result was published: violations={safety.violations!r}"
        )


@dataclass(frozen=True, slots=True)
class Phase2PreOracleSafetyGate:
    """Outcome-blind finite-policy gate evaluated before any final oracle."""

    design_stage: Literal["pilot", "confirmatory"]
    pilot_phase: Literal["calibration", "freeze"] | None
    mean_kl_safety: MeasuredKLSafety
    thresholds: tuple[tuple[str, float], ...]
    observed_by_arm: tuple[
        tuple[str, tuple[tuple[str, float], ...]],
        ...,
    ]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "phase2-pre-oracle-safety-gate/v1",
            "design_stage": self.design_stage,
            "pilot_phase": self.pilot_phase,
            "measure_only": self.design_stage == "pilot",
            "formal_gate": self.design_stage == "confirmatory",
            "thresholds": dict(self.thresholds),
            "observed_by_arm": {
                arm_name: dict(observed) for arm_name, observed in self.observed_by_arm
            },
            "violations": list(self.violations),
            "passed": self.passed,
            "beta_retuned": False,
            "on_violation": (
                "publish_target_free_diagnostics_without_final_oracle"
                if self.design_stage == "pilot"
                else "fail_before_final_oracle_and_heldout"
            ),
        }


class Phase2PreOracleSafetyError(Phase2KLSafetyError):
    """Fail a confirmatory seed before revealing any final-oracle outcome."""

    def __init__(self, gate: Phase2PreOracleSafetyGate) -> None:
        self.pre_oracle_safety = gate
        self.safety = gate.mean_kl_safety
        RuntimeError.__init__(
            self,
            "Phase-2 pre-oracle safety gate failed; beta remains frozen and no "
            f"outcome was revealed or published: violations={gate.violations!r}",
        )


def _score_training_oracle(
    inputs: Phase2PreparedInputs,
    backend: Phase2RuntimeBackend,
    *,
    design: Phase2Design,
) -> torch.Tensor:
    prompts = tuple(candidate.prompt for candidate in inputs.train_candidates)
    responses = tuple(candidate.response for candidate in inputs.train_candidates)
    with backend.oracle_session(
        expected_chat_template_sha256=inputs.oracle_chat_template_sha256
    ) as oracle:
        transformed = oracle.score_transformed(
            prompts,
            responses,
            transform=inputs.oracle_transform,
            batch_size=design.oracle_batch_size,
        )
    if (
        not isinstance(transformed, torch.Tensor)
        or transformed.shape != (len(inputs.train_candidates),)
        or not transformed.is_floating_point()
        or transformed.requires_grad
        or not bool(torch.isfinite(transformed).all())
    ):
        raise ValueError("oracle train rescore must return one finite detached value per node")
    return (
        transformed.detach()
        .to(device=inputs.train.policy_scores.device, dtype=inputs.train.policy_scores.dtype)
        .reshape(inputs.train.num_prompts, inputs.train.num_candidates)
        .clone()
    )


def _compute_common_beta_deployments(
    inputs: Phase2PreparedInputs,
    train_oracle_rewards: torch.Tensor,
    head_training: Phase2HeadTrainingResult,
    *,
    design: Phase2Design,
) -> tuple[
    CommonBetaCalibration,
    dict[str, PolicyDirectionResult],
    dict[str, CommonBetaDirection],
]:
    solve_dtype = resolve_fisher_solve_dtype(design.pcg_dtype)
    common_kwargs = {
        "relative_damping": design.relative_damping,
        "beta": 1.0,
        "pcg_dtype": design.pcg_dtype,
        "pcg_max_iterations": design.pcg_max_iterations,
        "pcg_tolerance": design.pcg_tolerance,
        "require_pcg_convergence": True,
    }
    for name, head in head_training.heads.items():
        if len(head) != inputs.train.reward_dimension:
            raise ValueError(
                f"trained head {name!r} has length {len(head)}, "
                f"expected {inputs.train.reward_dimension}"
            )
    directions: dict[str, PolicyDirectionResult] = {
        learner: policy_direction_from_head(
            inputs.train,
            head_training.heads[learner],
            **common_kwargs,
        )
        for learner in CANONICAL_LEARNERS
    }
    directions["oracle_step"] = policy_direction_from_node_rewards(
        inputs.train,
        train_oracle_rewards,
        **common_kwargs,
    )
    flat_scores = inputs.train.policy_scores.to(dtype=solve_dtype).reshape(
        -1, inputs.train.policy_dimension
    )
    undamped_fisher = DampedEmpiricalFisher(flat_scores, damping=0.0)
    if design.pilot_phase == "calibration":
        calibration = calibrate_common_beta(
            directions["oracle_step"].direction,
            undamped_fisher.matvec,
            target_oracle_quadratic_kl=design.target_oracle_quadratic_kl,
        )
    else:
        if design.frozen_global_beta is None:
            raise RuntimeError("fixed-beta design lost its frozen global beta")
        calibration = bind_frozen_common_beta(
            directions["oracle_step"].direction,
            undamped_fisher.matvec,
            frozen_global_beta=design.frozen_global_beta,
            reference_target_oracle_quadratic_kl=design.target_oracle_quadratic_kl,
        )
    deployed = deploy_with_common_beta(
        {name: result.direction for name, result in directions.items()},
        undamped_fisher.matvec,
        calibration=calibration,
    )
    return calibration, directions, deployed


def _arm_deployments(
    inputs: Phase2PreparedInputs,
    calibration: CommonBetaCalibration,
    directions: Mapping[str, PolicyDirectionResult],
    deployed: Mapping[str, CommonBetaDirection],
) -> dict[str, Phase2ArmDeployment]:
    beta = calibration.beta_common
    zero = torch.zeros(
        inputs.train.policy_dimension,
        dtype=inputs.train.policy_scores.dtype,
        device=inputs.train.policy_scores.device,
    )
    result = {
        "zero_b": Phase2ArmDeployment(
            arm_name="zero_b",
            beta_common=beta,
            displacement=zero,
            direction_evidence=None,
            common_beta_evidence=None,
        )
    }
    for name in (*CANONICAL_LEARNERS, "oracle_step"):
        result[name] = Phase2ArmDeployment(
            arm_name=name,
            beta_common=beta,
            displacement=deployed[name].displacement.detach().clone(),
            direction_evidence=directions[name].to_dict(),
            common_beta_evidence=deployed[name].to_dict(),
        )
    return result


def _freeze_heldout_evaluation_state(
    inputs: Phase2PreparedInputs,
    head_training: Phase2HeadTrainingResult,
    calibration: CommonBetaCalibration,
    deployments: Mapping[str, Phase2ArmDeployment],
    *,
    design: Phase2Design,
) -> FrozenHeldoutEvaluationState:
    """Seal every train-derived policy quantity before held-out oracle reveal."""

    if tuple(deployments) != PHASE2_ARM_ORDER:
        raise ValueError("deployments must follow the frozen Phase-2 arm order")
    deployment_identity: dict[str, object] = {
        "arm_order": list(PHASE2_ARM_ORDER),
        "learner_specific_rescaling": False,
        "arms": {},
    }
    arm_identity = deployment_identity["arms"]
    if not isinstance(arm_identity, dict):
        raise RuntimeError("internal deployment identity assembly failed")
    for arm_name in PHASE2_ARM_ORDER:
        deployment = deployments[arm_name]
        if deployment.beta_common != calibration.beta_common:
            raise RuntimeError("deployment beta changed before held-out state freeze")
        arm_identity[arm_name] = {
            "beta_common": deployment.beta_common,
            "displacement_sha256": _tensor_sha256(deployment.displacement),
            "direction_evidence_sha256": (
                None
                if deployment.direction_evidence is None
                else _canonical_sha256(dict(deployment.direction_evidence))
            ),
            "common_beta_evidence_sha256": (
                None
                if deployment.common_beta_evidence is None
                else _canonical_sha256(dict(deployment.common_beta_evidence))
            ),
        }
    return FrozenHeldoutEvaluationState(
        source_config_hash=inputs.source_config_hash,
        phase2_design_sha256=inputs.phase2_config_hash,
        phase2_runtime_contract_sha256=design.sha256,
        seed=inputs.seed,
        heads={
            learner: tuple(float(value) for value in head_training.heads[learner])
            for learner in CANONICAL_LEARNERS
        },
        heads_sha256=head_training.heads_sha256,
        training_design_sha256=head_training.training_design_sha256,
        beta_common=calibration.beta_common,
        deployment_identity=deployment_identity,
    )


def _validate_rollout_order(
    rollout: Phase2PolicyRollout,
    *,
    test_prompts: Sequence[PromptRecord],
    candidates_per_prompt: int,
    materialization_records: Mapping[str, Mapping[str, object]],
) -> None:
    expected = tuple(
        (prompt.prompt_id, candidate_index)
        for prompt in test_prompts
        for candidate_index in range(candidates_per_prompt)
    )
    observed = tuple(
        (trajectory.prompt_id, trajectory.candidate_index) for trajectory in rollout.trajectories
    )
    if observed != expected:
        raise ValueError(
            f"{rollout.arm_name} trajectories must be prompt-major and candidate-index ordered"
        )
    prompt_text = {prompt.prompt_id: _phase1._prompt_text(prompt) for prompt in test_prompts}
    if any(
        trajectory.prompt != prompt_text[trajectory.prompt_id]
        for trajectory in rollout.trajectories
    ):
        raise ValueError("trajectory prompt text does not match the frozen test prompt")
    for trajectory in rollout.trajectories:
        try:
            record = materialization_records[trajectory.prompt_id]
        except KeyError as error:
            raise ValueError(
                "trajectory prompt has no materialization prompt-semantics record"
            ) from error
        if (
            trajectory.raw_prompt_sha256 != record["raw_prompt_sha256"]
            or trajectory.policy_chat_token_count != record["policy_chat_token_count"]
            or trajectory.policy_prompt_token_ids_sha256 != record["policy_prompt_token_ids_sha256"]
            or trajectory.max_prompt_tokens != record["max_prompt_tokens"]
            or trajectory.prompt_truncated is not record["truncated"]
            or trajectory.raw_prompt_preserved is not record["raw_prompt_preserved"]
        ):
            raise ValueError("rollout policy prompt semantics differ from materialization evidence")


def _materialization_prompt_records(
    inputs: Phase2PreparedInputs,
) -> dict[str, Mapping[str, object]]:
    records = inputs.materialization_prompt_semantics["records"]
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise RuntimeError("materialization prompt semantics records changed after preparation")
    result: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("materialization prompt semantics record changed after preparation")
        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in result:
            raise RuntimeError("materialization prompt semantics IDs are invalid or duplicated")
        result[prompt_id] = record
    return result


def _prompt_semantics_result_summary(
    inputs: Phase2PreparedInputs,
    rollouts: Mapping[str, Phase2PolicyRollout],
) -> dict[str, object]:
    materialization = {
        key: value
        for key, value in inputs.materialization_prompt_semantics.items()
        if key != "records"
    }
    reference = rollouts["zero_b"].trajectories
    first_per_prompt: dict[str, Phase2Trajectory] = {}
    for trajectory in reference:
        first_per_prompt.setdefault(trajectory.prompt_id, trajectory)
    token_counts = [trajectory.policy_chat_token_count for trajectory in first_per_prompt.values()]
    return {
        "schema_version": "phase2-full-prompt-continuity/v1",
        "materialization": materialization,
        "rollout": {
            "schema_version": _phase1._POLICY_PROMPT_SEMANTICS_SCHEMA,
            "num_prompts": len(token_counts),
            "max_prompt_tokens": reference[0].max_prompt_tokens,
            "minimum_policy_chat_token_count": min(token_counts),
            "maximum_policy_chat_token_count": max(token_counts),
            "mean_policy_chat_token_count": sum(token_counts) / len(token_counts),
            "over_limit_prompt_count": 0,
            "truncated_prompt_count": 0,
            "raw_prompt_preserved_count": len(token_counts),
            "matches_materialization_token_prefix_evidence": True,
            "same_evidence_across_policy_arms": True,
        },
        "oracle": {
            "input_text": "same_raw_prompt_plus_assistant_response",
            "rerendered_with_independent_oracle_chat_template": True,
            "policy_chat_tokens_reused_by_oracle": False,
            "policy_and_oracle_chat_template_sha256_distinct": (
                inputs.policy_chat_template_sha256 != inputs.oracle_chat_template_sha256
            ),
            "policy_chat_template_sha256": inputs.policy_chat_template_sha256,
            "oracle_chat_template_sha256": inputs.oracle_chat_template_sha256,
        },
    }


def _rollout_policy_arms(
    inputs: Phase2PreparedInputs,
    backend: Phase2RuntimeBackend,
    deployments: Mapping[str, Phase2ArmDeployment],
    *,
    design: Phase2Design,
) -> tuple[dict[str, Phase2PolicyRollout], MeasuredKLSafety]:
    rollouts: dict[str, Phase2PolicyRollout] = {}
    materialization_records = _materialization_prompt_records(inputs)
    rollout_seed = SeedBundle.from_base_seed(inputs.seed).rollout
    with backend.policy_session(
        seed=inputs.seed,
        expected_a_sha256=inputs.policy_a_sha256,
        expected_layout=inputs.policy_layout,
        expected_chat_template_sha256=inputs.policy_chat_template_sha256,
    ) as policy:
        for arm_name in PHASE2_ARM_ORDER:
            deployment = deployments[arm_name]
            if deployment.beta_common != deployments["zero_b"].beta_common:
                raise RuntimeError("every Phase-2 arm must receive the exact same beta_common")
            rollout = policy.rollout(
                deployment,
                inputs.test_prompts,
                candidates_per_prompt=design.rollout_candidates_per_prompt,
                max_response_tokens=design.max_response_tokens,
                rollout_seed=rollout_seed,
                kl_token_chunk_size=design.kl_token_chunk_size,
            )
            if rollout.arm_name != arm_name:
                raise ValueError("policy backend returned an arm out of the frozen order")
            _validate_rollout_order(
                rollout,
                test_prompts=inputs.test_prompts,
                candidates_per_prompt=design.rollout_candidates_per_prompt,
                materialization_records=materialization_records,
            )
            if rollouts:
                reference_seeds = tuple(
                    trajectory.prompt_rollout_seed for trajectory in rollouts["zero_b"].trajectories
                )
                observed_seeds = tuple(
                    trajectory.prompt_rollout_seed for trajectory in rollout.trajectories
                )
                if observed_seeds != reference_seeds:
                    raise ValueError(
                        "all Phase-2 arms must reset identical per-prompt rollout seeds"
                    )
                reference_semantics = tuple(
                    (
                        trajectory.raw_prompt_sha256,
                        trajectory.policy_chat_token_count,
                        trajectory.policy_prompt_token_ids_sha256,
                        trajectory.max_prompt_tokens,
                        trajectory.prompt_truncated,
                        trajectory.raw_prompt_preserved,
                    )
                    for trajectory in rollouts["zero_b"].trajectories
                )
                observed_semantics = tuple(
                    (
                        trajectory.raw_prompt_sha256,
                        trajectory.policy_chat_token_count,
                        trajectory.policy_prompt_token_ids_sha256,
                        trajectory.max_prompt_tokens,
                        trajectory.prompt_truncated,
                        trajectory.raw_prompt_preserved,
                    )
                    for trajectory in rollout.trajectories
                )
                if observed_semantics != reference_semantics:
                    raise ValueError(
                        "all Phase-2 arms must use identical complete policy-prompt encodings"
                    )
            rollouts[arm_name] = rollout
    measured = {
        arm_name: float(
            rollout.per_sequence_kl_updated_to_reference.detach()
            .to(device="cpu", dtype=torch.float64)
            .mean()
            .item()
        )
        for arm_name, rollout in rollouts.items()
    }
    safety = assess_measured_kl_safety(
        measured,
        cap=design.measured_kl_safety_cap,
    )
    return rollouts, safety


def _score_final_rollouts(
    inputs: Phase2PreparedInputs,
    backend: Phase2RuntimeBackend,
    rollouts: Mapping[str, Phase2PolicyRollout],
    frozen_heldout_state: FrozenHeldoutEvaluationState,
    *,
    design: Phase2Design,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    ordered = tuple(
        trajectory
        for arm_name in PHASE2_ARM_ORDER
        for trajectory in rollouts[arm_name].trajectories
    )
    with backend.oracle_session(
        expected_chat_template_sha256=inputs.oracle_chat_template_sha256
    ) as oracle:
        rewards = oracle.score_transformed(
            tuple(trajectory.prompt for trajectory in ordered),
            tuple(trajectory.response for trajectory in ordered),
            transform=inputs.oracle_transform,
            batch_size=design.oracle_batch_size,
        )
        if (
            not isinstance(rewards, torch.Tensor)
            or rewards.shape != (len(ordered),)
            or not rewards.is_floating_point()
            or rewards.requires_grad
            or not bool(torch.isfinite(rewards).all())
        ):
            raise ValueError("final oracle must return one finite detached reward per trajectory")
        heldout = score_and_evaluate_deferred_heldout(
            oracle,
            inputs.heldout,
            frozen_heldout_state,
            transform=inputs.oracle_transform,
            oracle_chat_template_sha256=inputs.oracle_chat_template_sha256,
            batch_size=design.oracle_batch_size,
            relative_damping=design.relative_damping,
            pcg_dtype=design.pcg_dtype,
            pcg_max_iterations=design.pcg_max_iterations,
            pcg_tolerance=design.pcg_tolerance,
        )
    per_arm = len(inputs.test_prompts) * design.rollout_candidates_per_prompt
    per_arm_rewards = {
        arm_name: rewards[index * per_arm : (index + 1) * per_arm]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
        for index, arm_name in enumerate(PHASE2_ARM_ORDER)
    }
    return per_arm_rewards, heldout


def _length_summary(trajectories: Sequence[Phase2Trajectory]) -> dict[str, object]:
    lengths = torch.tensor(
        [trajectory.response_token_count for trajectory in trajectories],
        dtype=torch.float64,
    )
    return {
        "num_trajectories": len(trajectories),
        "terminated_by_eos_count": sum(trajectory.terminated_by_eos for trajectory in trajectories),
        "terminated_by_eos_rate": (
            sum(trajectory.terminated_by_eos for trajectory in trajectories) / len(trajectories)
        ),
        "reached_max_length_count": sum(
            trajectory.reached_max_length for trajectory in trajectories
        ),
        "reached_max_length_rate": (
            sum(trajectory.reached_max_length for trajectory in trajectories) / len(trajectories)
        ),
        "response_token_count": {
            "mean": float(lengths.mean().item()),
            "minimum": int(lengths.min().item()),
            "maximum": int(lengths.max().item()),
        },
    }


def _kl_tail_summary(
    per_sequence_kl: torch.Tensor,
    *,
    formal_gate_applied: bool = False,
) -> dict[str, object]:
    if (
        not isinstance(per_sequence_kl, torch.Tensor)
        or per_sequence_kl.ndim != 2
        or not per_sequence_kl.is_floating_point()
        or not bool(torch.isfinite(per_sequence_kl).all())
        or bool((per_sequence_kl < 0.0).any())
    ):
        raise ValueError("per_sequence_kl must be a finite non-negative prompt-by-candidate tensor")
    values = per_sequence_kl.to(dtype=torch.float64)
    prompt_means = values.mean(dim=1)
    quantile_levels = torch.tensor(
        [0.5, 0.9, 0.95, 0.99],
        dtype=prompt_means.dtype,
        device=prompt_means.device,
    )
    quantiles = torch.quantile(prompt_means, quantile_levels)
    return {
        "schema_version": "on-policy-kl-tail-summary/v1",
        "unit": "prompt_mean_over_candidates",
        "num_prompts": int(prompt_means.numel()),
        "candidates_per_prompt": int(values.shape[1]),
        "mean": float(prompt_means.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "maximum": float(prompt_means.max().item()),
        "per_sequence_maximum": float(values.max().item()),
        "pilot_selection_role": "locality_tail_measurement",
        "formal_gate_applied": formal_gate_applied,
    }


def assess_phase2_pre_oracle_safety(
    rollouts: Mapping[str, Phase2PolicyRollout],
    *,
    design: Phase2Design,
) -> Phase2PreOracleSafetyGate:
    """Evaluate frozen mean/tail/length limits without reading any outcome."""

    if not isinstance(design, Phase2Design):
        raise TypeError("design must be Phase2Design")
    if not isinstance(rollouts, Mapping) or tuple(rollouts) != PHASE2_ARM_ORDER:
        raise ValueError("rollouts must contain the four Phase-2 arms in frozen order")

    thresholds = (
        ("mean_policy_to_reference_kl_cap", design.measured_kl_safety_cap),
        ("prompt_mean_p95_kl_cap", design.prompt_mean_p95_kl_cap),
        ("prompt_mean_p99_kl_cap", design.prompt_mean_p99_kl_cap),
        ("prompt_mean_maximum_kl_cap", design.prompt_mean_maximum_kl_cap),
        ("per_sequence_maximum_kl_cap", design.per_sequence_maximum_kl_cap),
        ("reached_max_length_rate_cap", design.max_length_formal_threshold),
    )
    threshold_by_metric = dict(thresholds)
    measured_means: dict[str, float] = {}
    observed_by_arm: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    violations: list[str] = []
    for arm_name in PHASE2_ARM_ORDER:
        rollout = rollouts[arm_name]
        if rollout.arm_name != arm_name:
            raise ValueError("rollout arm identity does not match the frozen order")
        trajectory_count = len(rollout.trajectories)
        candidates = design.rollout_candidates_per_prompt
        if trajectory_count % candidates != 0:
            raise ValueError("rollout geometry is not divisible by candidates_per_prompt")
        prompts = trajectory_count // candidates
        if prompts < 1:
            raise ValueError("pre-oracle safety requires at least one rollout prompt")
        kl = (
            rollout.per_sequence_kl_updated_to_reference.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(prompts, candidates)
        )
        tail = _kl_tail_summary(
            kl,
            formal_gate_applied=design.stage == "confirmatory",
        )
        length = _length_summary(rollout.trajectories)
        observed = (
            ("mean_policy_to_reference_kl", float(tail["mean"])),
            ("prompt_mean_p95_kl", float(tail["p95"])),
            ("prompt_mean_p99_kl", float(tail["p99"])),
            ("prompt_mean_maximum_kl", float(tail["maximum"])),
            ("per_sequence_maximum_kl", float(tail["per_sequence_maximum"])),
            (
                "reached_max_length_rate",
                float(length["reached_max_length_rate"]),
            ),
        )
        measured_means[arm_name] = float(tail["mean"])
        observed_by_arm.append((arm_name, observed))
        for metric, value in observed:
            cap = threshold_by_metric[f"{metric}_cap"]
            if value > cap:
                violations.append(f"{arm_name}:{metric}")

    mean_safety = assess_measured_kl_safety(
        measured_means,
        cap=design.measured_kl_safety_cap,
    )
    expected_mean_violations = tuple(
        sorted(f"{arm_name}:mean_policy_to_reference_kl" for arm_name in mean_safety.violations)
    )
    observed_mean_violations = tuple(
        sorted(
            violation
            for violation in violations
            if violation.endswith(":mean_policy_to_reference_kl")
        )
    )
    if observed_mean_violations != expected_mean_violations:
        raise RuntimeError("mean-KL and unified pre-oracle safety arithmetic disagree")
    return Phase2PreOracleSafetyGate(
        design_stage=design.stage,
        pilot_phase=design.pilot_phase,
        mean_kl_safety=mean_safety,
        thresholds=thresholds,
        observed_by_arm=tuple(observed_by_arm),
        violations=tuple(violations),
    )


def _pilot_deployment_hashes(
    deployments: Mapping[str, Phase2ArmDeployment],
) -> dict[str, object]:
    evidence: dict[str, object] = {}
    for arm_name in PHASE2_ARM_ORDER:
        deployment = deployments[arm_name]
        evidence[arm_name] = {
            "beta_common": deployment.beta_common,
            "displacement_sha256": _tensor_sha256(deployment.displacement),
            "direction_evidence_sha256": (
                None
                if deployment.direction_evidence is None
                else _canonical_sha256(dict(deployment.direction_evidence))
            ),
            "common_beta_evidence_sha256": (
                None
                if deployment.common_beta_evidence is None
                else _canonical_sha256(dict(deployment.common_beta_evidence))
            ),
        }
    return evidence


_PILOT_AUDIT_VECTOR_KEYS = frozenset(
    {
        "head_weight",
        "head_weights",
        "direction",
        "natural_direction",
        "displacement",
        "oracle_displacement",
        "moment",
        "operator_direction",
        "projection_matrix",
        "true_rewards",
    }
)


def _sanitize_pilot_training_audit(value: object) -> object:
    """Drop train-time vectors while retaining scalar/hash audit evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_pilot_training_audit(item)
            for key, item in value.items()
            if key not in _PILOT_AUDIT_VECTOR_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_pilot_training_audit(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_pilot_training_audit(item) for item in value]
    return value


def _pilot_common_beta_evidence(
    calibration: CommonBetaCalibration,
    design: Phase2Design,
) -> tuple[str, dict[str, object]]:
    if design.stage != "pilot" or design.pilot_phase is None:
        raise ValueError("pilot common-beta evidence requires a pilot design")
    if design.pilot_phase == "calibration":
        return (
            "train_only_global_beta_calibration_candidate",
            {
                "schema_version": "global-beta-calibration-candidate/v1",
                "rule": design.common_beta_rule,
                "candidate_beta": calibration.beta_common,
                "frozen_global_beta": None,
                "oracle_natural_curvature": calibration.oracle_natural_curvature,
                "target_oracle_quadratic_kl": calibration.target_oracle_quadratic_kl,
                "predicted_oracle_quadratic_kl": (calibration.predicted_oracle_quadratic_kl),
                "calibration_split": "train_only",
                "formal_beta_selected": False,
                "formal_selection_rule": (
                    "maximum_pilot_seed_candidate_then_smallest_passing_frozen_kl_only_grid"
                ),
                "learner_specific_rescaling": False,
            },
        )

    beta = _positive_float(calibration.beta_common, name="calibration.beta_common")
    frozen = _positive_float(design.frozen_global_beta, name="design.frozen_global_beta")
    if beta != frozen:
        raise RuntimeError("pilot freeze deployment beta differs from its frozen scalar")
    source_sha = _validate_digest(
        design.beta_source_aggregate_sha256,
        name="design.beta_source_aggregate_sha256",
    )
    return (
        "pilot_fixed_global_beta_rehearsal",
        {
            "schema_version": "pilot-frozen-global-beta-rehearsal/v1",
            "rule": design.common_beta_rule,
            "beta_common": beta,
            "frozen_global_beta": frozen,
            "beta_matches_frozen_global_beta": True,
            "beta_source_aggregate_sha256": source_sha,
            "current_seed_oracle_natural_curvature": (calibration.oracle_natural_curvature),
            "reference_target_oracle_quadratic_kl": (calibration.target_oracle_quadratic_kl),
            "predicted_current_seed_oracle_quadratic_kl": (
                calibration.predicted_oracle_quadratic_kl
            ),
            "current_seed_curvature_role": "predicted_kl_diagnostic_only",
            "beta_selected_from_current_seed_curvature": False,
            "frozen_in_phase2_design_identity": True,
            "learner_specific_rescaling": False,
            "post_evaluation_retuning": False,
        },
    )


def _assemble_pilot_outputs(
    inputs: Phase2PreparedInputs,
    *,
    output_json: Path,
    diagnostics_path: Path,
    design: Phase2Design,
    train_oracle_rewards: torch.Tensor,
    head_training: Phase2HeadTrainingResult,
    calibration: CommonBetaCalibration,
    deployments: Mapping[str, Phase2ArmDeployment],
    rollouts: Mapping[str, Phase2PolicyRollout],
    safety: MeasuredKLSafety,
    pre_oracle_safety: Phase2PreOracleSafetyGate,
    current_process_identity: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Assemble the only publishable pilot artifact.

    The pilot is allowed to select optimization, response-horizon, and KL
    locality contracts.  It therefore stops before any final oracle session
    and publishes neither held-out targets nor finite-policy outcomes.
    """

    if design.stage != "pilot" or design.formal_eligibility:
        raise ValueError("pilot diagnostics require a formally ineligible pilot design")
    if design.pilot_phase is None:
        raise RuntimeError("pilot diagnostics lost the frozen pilot_phase")

    deployment_hashes = _pilot_deployment_hashes(deployments)
    beta_evidence_key, beta_evidence = _pilot_common_beta_evidence(
        calibration,
        design,
    )
    arm_diagnostics: dict[str, object] = {}
    records: list[dict[str, object]] = []
    for arm_name in PHASE2_ARM_ORDER:
        rollout = rollouts[arm_name]
        arm_kl = (
            rollout.per_sequence_kl_updated_to_reference.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(
                len(inputs.test_prompts),
                design.rollout_candidates_per_prompt,
            )
        )
        arm_diagnostics[arm_name] = {
            "deployment_hashes": deployment_hashes[arm_name],
            "rollout_length": _length_summary(rollout.trajectories),
            "mean_on_policy_kl_pi_updated_to_pi0": float(arm_kl.mean().item()),
            "on_policy_kl_tail": _kl_tail_summary(arm_kl),
        }
        for trajectory, kl_value in zip(
            rollout.trajectories,
            arm_kl.reshape(-1).tolist(),
            strict=True,
        ):
            records.append(
                trajectory.to_pilot_diagnostic_dict(
                    beta_common=calibration.beta_common,
                    pilot_phase=design.pilot_phase,
                    on_policy_kl=float(kl_value),
                )
            )

    payload: dict[str, object] = {
        "schema_version": PHASE2_PILOT_RESULT_SCHEMA,
        "design_stage": "pilot",
        "pilot_phase": design.pilot_phase,
        "formal_eligibility": False,
        "evidence_role": "optimization_horizon_and_kl_design_selection_only",
        "per_seed_supports_formal_claim": False,
        "source_config_hash": inputs.source_config_hash,
        "phase2_design_sha256": inputs.phase2_config_hash,
        "phase2_runtime_contract": design.to_dict(),
        "phase2_runtime_contract_sha256": design.sha256,
        "seed": inputs.seed,
        "artifact_dir": relative_posix_reference(inputs.artifact_dir, base=output_json.parent),
        "diagnostics_jsonl": relative_posix_reference(
            diagnostics_path,
            base=output_json.parent,
        ),
        "artifact_metadata_sha256": inputs.artifact_metadata_sha256,
        "run_manifest": relative_posix_reference(inputs.run_manifest, base=output_json.parent),
        "run_manifest_sha256": inputs.run_manifest_sha256,
        "environment_identity": dict(inputs.environment_identity),
        "current_process_identity": dict(current_process_identity),
        "train_oracle_rescore": {
            "source": "saved_train_candidates_rescored_with_pinned_oracle",
            "num_prompts": inputs.train.num_prompts,
            "num_candidates": inputs.train.num_candidates,
            "transformed_rewards_sha256": _tensor_sha256(train_oracle_rewards),
            "oracle_chat_template_sha256": inputs.oracle_chat_template_sha256,
            "frozen_transform": {
                "b": inputs.oracle_transform.b,
                "tau": inputs.oracle_transform.tau,
            },
            "raw_oracle_logits_serialized": False,
            "role": (
                "training_and_global_beta_calibration_candidate_only"
                if design.pilot_phase == "calibration"
                else "training_and_fixed_beta_safety_rehearsal_only"
            ),
        },
        "head_training": {
            "training_arm": head_training.training_arm,
            "training_design_sha256": head_training.training_design_sha256,
            "heads_sha256": head_training.heads_sha256,
            "head_weights_serialized": False,
            "audit": _sanitize_pilot_training_audit(head_training.audit),
            "audit_vector_fields_redacted": sorted(_PILOT_AUDIT_VECTOR_KEYS),
            "source": "trained_after_train_oracle_rescore",
            "old_phase1_comparison_heads_reused": False,
            "test_data_accessed": False,
        },
        "deployment_hashes": deployment_hashes,
        "measured_kl_safety": safety.to_dict(),
        "pre_oracle_safety_gate": pre_oracle_safety.to_dict(),
        "pilot_kl_safety_gate": {
            "schema_version": "pilot-measured-kl-gate/v1",
            "gate_passed": safety.passed,
            "measure_only": True,
            "supports_formal_claim": False,
            "violations": list(safety.violations),
            "on_violation": "publish_target_free_diagnostics_without_final_oracle",
        },
        "arms": arm_diagnostics,
        "information_boundary": {
            "calibration_split": (
                "train_only"
                if design.pilot_phase == "calibration"
                else "excluded_pilot_calibration_outputs_only"
            ),
            "new_rollout_prompts_used_for_calibration": False,
            "final_oracle_session_opened": False,
            "rollout_responses_oracle_scored": False,
            "heldout_evaluator_called": False,
            "oracle_outcomes_serialized": False,
            "prompt_or_response_text_serialized": False,
            "token_ids_or_response_masks_serialized": False,
            "source_artifact_format": "phase1_bridge",
            "source_artifact_may_contain_prior_heldout_candidate_scores": True,
            "source_artifact_heldout_targets_exposed_by_phase2_prepared_inputs": False,
            "prompt_semantics": _prompt_semantics_result_summary(inputs, rollouts),
        },
        "common_random_numbers": {
            "named_stream": "rollout",
            "seed": SeedBundle.from_base_seed(inputs.seed).rollout,
            "same_per_prompt_seed_reset_across_arms": True,
            "candidate_index_alignment": True,
        },
        "memory_schedule": [
            "oracle_train_rescore",
            "no_model_train_only_head_training",
            (
                "no_model_direction_and_beta_calibration_candidate"
                if design.pilot_phase == "calibration"
                else "no_model_direction_and_frozen_beta_binding"
            ),
            "policy_rollouts_and_on_policy_kl",
            "stop_before_final_oracle_and_heldout_evaluation",
        ],
        "policy_and_oracle_co_resident": False,
        "learner_specific_line_search": False,
        "diagnostics_sha256": None,
    }
    payload[beta_evidence_key] = beta_evidence
    _canonical_sha256(payload)
    return records, payload


def _confirmatory_common_beta_evidence(
    calibration: CommonBetaCalibration,
    design: Phase2Design,
) -> dict[str, object]:
    """Serialize proof that a confirmatory seed used the config-frozen beta."""

    if design.stage != "confirmatory" or not design.formal_eligibility:
        raise ValueError("frozen-global-beta evidence requires a confirmatory design")
    beta = _positive_float(calibration.beta_common, name="calibration.beta_common")
    frozen = _positive_float(design.frozen_global_beta, name="design.frozen_global_beta")
    if beta != frozen:
        raise RuntimeError("confirmatory deployment beta differs from the frozen design scalar")
    return {
        "schema_version": "common-beta-frozen-global/v1",
        "rule": design.common_beta_rule,
        "beta_selection_split": design.common_beta_calibration_split,
        "beta_source": design.common_beta_source,
        "beta_common": beta,
        "frozen_global_beta": frozen,
        "beta_source_aggregate_sha256": design.beta_source_aggregate_sha256,
        "beta_matches_frozen_global_beta": True,
        "beta_selected_from_current_seed_curvature": False,
        "current_seed_oracle_natural_curvature": calibration.oracle_natural_curvature,
        "reference_target_oracle_quadratic_kl": calibration.target_oracle_quadratic_kl,
        "predicted_current_seed_oracle_quadratic_kl": (calibration.predicted_oracle_quadratic_kl),
        "current_seed_curvature_role": "predicted_kl_diagnostic_only",
        "frozen_in_phase2_design_identity": True,
        "learner_specific_rescaling": False,
        "post_evaluation_retuning": False,
    }


def _assemble_outputs(
    inputs: Phase2PreparedInputs,
    *,
    output_json: Path,
    rollouts_path: Path,
    design: Phase2Design,
    train_oracle_rewards: torch.Tensor,
    head_training: Phase2HeadTrainingResult,
    calibration: CommonBetaCalibration,
    directions: Mapping[str, PolicyDirectionResult],
    deployments: Mapping[str, Phase2ArmDeployment],
    rollouts: Mapping[str, Phase2PolicyRollout],
    rewards: Mapping[str, torch.Tensor],
    heldout: Mapping[str, object],
    safety: MeasuredKLSafety,
    pre_oracle_safety: Phase2PreOracleSafetyGate,
    current_process_identity: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if design.stage != "confirmatory" or not design.formal_eligibility:
        raise ValueError("finite-policy endpoint outputs require a confirmatory design")
    rows = len(inputs.test_prompts)
    columns = design.rollout_candidates_per_prompt
    shape = (rows, columns)
    reference_rewards = rewards["zero_b"].reshape(shape)
    oracle_rewards = rewards["oracle_step"].reshape(shape)
    oracle_kl = (
        rollouts["oracle_step"]
        .per_sequence_kl_updated_to_reference.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(shape)
    )
    arm_results: dict[str, object] = {}
    output_records: list[dict[str, object]] = []
    for arm_name in PHASE2_ARM_ORDER:
        arm_rewards = rewards[arm_name].reshape(shape)
        arm_kl = (
            rollouts[arm_name]
            .per_sequence_kl_updated_to_reference.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(shape)
        )
        utility = summarize_downstream_utility(
            arm_rewards,
            arm_kl,
            reference_rewards,
            beta_common=calibration.beta_common,
            oracle_step_transformed_target_rewards=oracle_rewards,
            oracle_step_on_policy_updated_to_reference_kl=oracle_kl,
        )
        deployment = deployments[arm_name]
        arm_results[arm_name] = {
            "deployment": (
                {
                    "schema_version": "zero-b-deployment/v1",
                    "beta_common": calibration.beta_common,
                    "displacement_is_exact_zero": True,
                    "learner_specific_rescaling": False,
                }
                if arm_name == "zero_b"
                else {
                    "direction": dict(deployment.direction_evidence or {}),
                    "common_beta_direction": dict(deployment.common_beta_evidence or {}),
                }
            ),
            "rollout": _length_summary(rollouts[arm_name].trajectories),
            "mean_on_policy_kl_pi_updated_to_pi0": float(arm_kl.mean().item()),
            "on_policy_kl_tail": _kl_tail_summary(
                arm_kl,
                formal_gate_applied=True,
            ),
            "utility": utility.to_dict(),
        }
        for trajectory, kl_value, reward_value in zip(
            rollouts[arm_name].trajectories,
            arm_kl.reshape(-1).tolist(),
            arm_rewards.reshape(-1).tolist(),
            strict=True,
        ):
            output_records.append(
                {
                    **trajectory.to_unscored_dict(beta_common=calibration.beta_common),
                    "kl_orientation": KL_ORIENTATION,
                    "kl_history_source": KL_HISTORY_SOURCE,
                    "on_policy_kl_pi_updated_to_pi0": float(kl_value),
                    "transformed_oracle_reward": float(reward_value),
                    "target_utility": float(reward_value - calibration.beta_common * kl_value),
                    "raw_oracle_logit_serialized": False,
                }
            )

    payload: dict[str, object] = {
        "schema_version": PHASE2_RESULT_SCHEMA,
        "design_stage": design.stage,
        "formal_eligibility": design.formal_eligibility,
        "per_seed_supports_formal_claim": False,
        "source_config_hash": inputs.source_config_hash,
        "phase2_design_sha256": inputs.phase2_config_hash,
        "phase2_runtime_contract": design.to_dict(),
        "phase2_runtime_contract_sha256": design.sha256,
        "seed": inputs.seed,
        "artifact_dir": relative_posix_reference(inputs.artifact_dir, base=output_json.parent),
        "rollouts_jsonl": relative_posix_reference(rollouts_path, base=output_json.parent),
        "artifact_metadata_sha256": inputs.artifact_metadata_sha256,
        "run_manifest": relative_posix_reference(inputs.run_manifest, base=output_json.parent),
        "run_manifest_sha256": inputs.run_manifest_sha256,
        "environment_identity": dict(inputs.environment_identity),
        "current_process_identity": dict(current_process_identity),
        "train_oracle_rescore": {
            "source": "saved_train_candidates_rescored_with_pinned_oracle",
            "num_prompts": inputs.train.num_prompts,
            "num_candidates": inputs.train.num_candidates,
            "transformed_rewards_sha256": _tensor_sha256(train_oracle_rewards),
            "oracle_chat_template_sha256": inputs.oracle_chat_template_sha256,
            "frozen_transform": {
                "b": inputs.oracle_transform.b,
                "tau": inputs.oracle_transform.tau,
            },
            "raw_oracle_logits_serialized": False,
        },
        "head_training": {
            "training_arm": head_training.training_arm,
            "training_design_sha256": head_training.training_design_sha256,
            "heads_sha256": head_training.heads_sha256,
            "head_weights": {name: list(head_training.heads[name]) for name in CANONICAL_LEARNERS},
            "audit": dict(head_training.audit),
            "source": "trained_after_train_oracle_rescore",
            "old_phase1_comparison_heads_reused": False,
            "test_data_accessed": False,
        },
        "common_beta_calibration": _confirmatory_common_beta_evidence(
            calibration,
            design,
        ),
        "train_oracle_direction": directions["oracle_step"].to_dict(),
        "measured_kl_safety": safety.to_dict(),
        "pre_oracle_safety_gate": pre_oracle_safety.to_dict(),
        "arms": arm_results,
        "heldout_fixed_beta": dict(heldout),
        "heldout_fixed_beta_sha256": heldout_evaluation_sha256(heldout),
        "information_boundary": {
            "beta_selection_split": design.common_beta_calibration_split,
            "current_seed_train_curvature_role": "predicted_kl_diagnostic_only",
            "new_rollout_prompts_used_for_calibration": False,
            "source_materialization_heldout_scores_used_for_calibration": False,
            "new_rollout_oracle_scoring_after_heads_beta_and_directions_frozen": True,
            "heldout_candidate_rescore_after_heads_beta_and_directions_frozen": True,
            "heldout_directions_used_for_policy": False,
            "source_artifact_may_contain_prior_heldout_candidate_scores": True,
            "prompt_semantics": _prompt_semantics_result_summary(inputs, rollouts),
        },
        "common_random_numbers": {
            "named_stream": "rollout",
            "seed": SeedBundle.from_base_seed(inputs.seed).rollout,
            "same_per_prompt_seed_reset_across_arms": True,
            "candidate_index_alignment": True,
        },
        "memory_schedule": [
            "oracle_train_rescore",
            "no_model_train_only_head_training",
            "no_model_direction_and_frozen_global_beta_binding",
            "policy_rollouts_and_on_policy_kl",
            "oracle_rollout_and_saved_heldout_candidate_scoring",
            "heldout_local_metrics_without_policy_feedback",
        ],
        "policy_and_oracle_co_resident": False,
        "learner_specific_line_search": False,
        "rollouts_sha256": None,
    }
    return output_records, payload


def run_common_beta_rollouts(
    inputs: Phase2PreparedInputs,
    head_trainer: Phase2HeadTrainer,
    backend: Phase2RuntimeBackend,
    *,
    output_json: str | os.PathLike[str],
    design: Phase2Design | None = None,
) -> dict[str, object]:
    """Execute the strict common-beta state machine and publish immutable outputs."""

    if not isinstance(inputs, Phase2PreparedInputs):
        raise TypeError("inputs must be Phase2PreparedInputs")
    if design is None:
        design = Phase2Design()
    if not isinstance(design, Phase2Design):
        raise TypeError("design must be Phase2Design")
    if design.rollout_candidates_per_prompt != inputs.train.num_candidates:
        raise ValueError("Phase-2 rollout candidate count must match the source candidate geometry")

    destination = Path(output_json)
    sidecar_path = destination.with_name(
        f"{destination.stem}.diagnostics.jsonl"
        if design.stage == "pilot"
        else f"{destination.stem}.rollouts.jsonl"
    )
    if destination.resolve() == sidecar_path.resolve():
        raise ValueError("result JSON and sidecar JSONL paths must be distinct")
    for target in (destination, sidecar_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {target}")
    if config_hash(inputs.source_config) != inputs.source_config_hash:
        raise RuntimeError("source configuration changed after Phase-2 input preparation")
    # These objects contain geometry and candidate text only.  Their stored
    # identities are checked before any model is allocated and checked again
    # immediately before the final oracle reveal.
    inputs.heldout.verify_integrity()
    if _sha256_file(inputs.run_manifest) != inputs.run_manifest_sha256:
        raise ValueError("run manifest bytes do not match run_manifest_sha256")
    current_process_identity = collect_execution_identity()
    if inputs.environment_identity.get("formal") is True and (
        current_process_identity != dict(inputs.environment_identity)
    ):
        raise RuntimeError(
            "current process identity does not match the formal run manifest identity"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    # The first oracle session receives only saved train candidates.  It is
    # closed before any direction solve or policy model allocation.
    train_oracle_rewards = _score_training_oracle(
        inputs,
        backend,
        design=design,
    )
    head_training = head_trainer.train_heads(
        inputs.train,
        train_oracle_rewards.detach().clone(),
        seed=inputs.seed,
    )
    if not isinstance(head_training, Phase2HeadTrainingResult):
        raise TypeError("head_trainer must return Phase2HeadTrainingResult")
    if head_training.training_design_sha256 != inputs.phase2_config_hash:
        raise ValueError("head trainer design identity does not match the Phase-2 configuration")
    calibration, directions, common_beta_directions = _compute_common_beta_deployments(
        inputs,
        train_oracle_rewards,
        head_training,
        design=design,
    )
    deployments = _arm_deployments(
        inputs,
        calibration,
        directions,
        common_beta_directions,
    )
    frozen_heldout_state = (
        _freeze_heldout_evaluation_state(
            inputs,
            head_training,
            calibration,
            deployments,
            design=design,
        )
        if design.stage == "confirmatory"
        else None
    )
    rollouts, safety = _rollout_policy_arms(
        inputs,
        backend,
        deployments,
        design=design,
    )
    pre_oracle_safety = assess_phase2_pre_oracle_safety(
        rollouts,
        design=design,
    )
    if safety.to_dict() != pre_oracle_safety.mean_kl_safety.to_dict():
        raise RuntimeError("mean-KL safety changed during unified pre-oracle assessment")
    if design.stage == "confirmatory" and not pre_oracle_safety.passed:
        raise Phase2PreOracleSafetyError(pre_oracle_safety)
    if design.stage == "pilot":
        records, payload = _assemble_pilot_outputs(
            inputs,
            output_json=destination,
            diagnostics_path=sidecar_path,
            design=design,
            train_oracle_rewards=train_oracle_rewards,
            head_training=head_training,
            calibration=calibration,
            deployments=deployments,
            rollouts=rollouts,
            safety=safety,
            pre_oracle_safety=pre_oracle_safety,
            current_process_identity=current_process_identity,
        )
        sidecar_hash_field = "diagnostics_sha256"
    else:
        if frozen_heldout_state is None:
            raise RuntimeError("confirmatory run did not freeze held-out evaluation state")
        # Fail-closed KL safety is evaluated above.  The held-out oracle is
        # loaded only for a frozen, safe confirmatory set of trajectories.
        inputs.heldout.verify_integrity()
        final_rewards, heldout = _score_final_rollouts(
            inputs,
            backend,
            rollouts,
            frozen_heldout_state,
            design=design,
        )
        records, payload = _assemble_outputs(
            inputs,
            output_json=destination,
            rollouts_path=sidecar_path,
            design=design,
            train_oracle_rewards=train_oracle_rewards,
            head_training=head_training,
            calibration=calibration,
            directions=directions,
            deployments=deployments,
            rollouts=rollouts,
            rewards=final_rewards,
            heldout=heldout,
            safety=safety,
            pre_oracle_safety=pre_oracle_safety,
            current_process_identity=current_process_identity,
        )
        sidecar_hash_field = "rollouts_sha256"
    if _sha256_file(inputs.run_manifest) != inputs.run_manifest_sha256:
        raise RuntimeError("run manifest changed during Phase-2 execution")
    if config_hash(inputs.source_config) != inputs.source_config_hash:
        raise RuntimeError("source configuration changed during Phase-2 execution")

    staged_sidecar: Path | None = None
    staged_result: Path | None = None
    try:
        staged_sidecar = _stage_jsonl(sidecar_path, records)
        payload[sidecar_hash_field] = _sha256_file(staged_sidecar)
        staged_result = _stage_json(destination, payload)
        _publish_staged_pair(
            sidecar_path,
            staged_sidecar,
            destination,
            staged_result,
        )
        staged_sidecar = None
        staged_result = None
        return payload
    finally:
        for temporary in (staged_sidecar, staged_result):
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()


__all__ = [
    "CONFIRMATORY_COMMON_BETA_RULE",
    "KL_HISTORY_SOURCE",
    "KL_ORIENTATION",
    "PHASE2_ARM_ORDER",
    "PHASE2_DESIGN_SCHEMA",
    "PHASE2_PILOT_DIAGNOSTIC_SCHEMA",
    "PHASE2_PILOT_RESULT_SCHEMA",
    "PHASE2_RESULT_SCHEMA",
    "PHASE2_ROLLOUT_SCHEMA",
    "PER_SEQUENCE_MAXIMUM_KL_CAP",
    "PILOT_COMMON_BETA_RULE",
    "PILOT_FREEZE_COMMON_BETA_RULE",
    "PROMPT_MEAN_MAXIMUM_KL_CAP",
    "PROMPT_MEAN_P95_KL_CAP",
    "PROMPT_MEAN_P99_KL_CAP",
    "REACHED_MAX_LENGTH_RATE_CAP",
    "MEAN_POLICY_TO_REFERENCE_KL_CAP",
    "Phase2ArmDeployment",
    "Phase2Design",
    "Phase2HeadTrainer",
    "Phase2HeadTrainingResult",
    "Phase2KLSafetyError",
    "Phase2PreOracleSafetyError",
    "Phase2PreOracleSafetyGate",
    "Phase2OracleSession",
    "Phase2PolicyRollout",
    "Phase2PolicySession",
    "Phase2PreparedInputs",
    "Phase2RuntimeBackend",
    "Phase2Trajectory",
    "assess_phase2_pre_oracle_safety",
    "run_common_beta_rollouts",
]

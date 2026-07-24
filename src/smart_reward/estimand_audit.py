"""CPU-only audit of the policy estimands induced by saved train directions.

The Phase-1 comparison reports reward-model quality after solving a held-out
natural direction, while the matched-KL rollout serializes the *actual*
train-derived direction used to update the policy.  Those are different
estimands.  This module evaluates the latter on the immutable artifact test
geometry without loading an LLM, recomputing a reward head, or solving another
linear system.

For a saved direction ``d``, test target moment ``g*``, raw-node empirical
Fisher ``F_test``, and configured inverse-temperature ``beta``, it reports

``U_beta(delta) = <g*, delta> - beta/2 <delta, F_test delta>``.

The audit evaluates ``delta=d`` (the native train direction), ``delta=alpha d``
(the direction actually applied by measured-KL line search), and the same ray
renormalized to an exact test-quadratic KL ``K``.  The last quantity isolates
direction quality from the independent rollout line searches.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    artifact_metadata_sha256,
    load_controlled_feature_artifact,
)
from .config import config_hash, validate_config
from .contracts import (
    BT_MLE,
    CANONICAL_LEARNERS,
    LEGACY_V1_LEARNERS,
    MATCHED_KL_ROLLOUT_SCHEMA_V1,
    MATCHED_KL_ROLLOUT_SCHEMA_V2,
    PRORM_PLUS,
)
from .experiment import EvaluationTensorData
from .paths import relative_posix_reference
from .phase1_rollout import parse_comparison_heads
from .repro import atomic_write_json

ESTIMAND_AUDIT_SCHEMA_VERSION = "estimand-audit/v1"

_DIRECTION_SCHEMA_VERSION = "policy-direction/v1"
_UPDATE_SCHEMA_VERSION = "measured-kl-update/v1"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_DIRECTION_FIELDS = {
    "schema_version",
    "direction",
    "beta",
    "relative_damping",
    "absolute_damping",
    "mean_fisher_diagonal",
    "moment_norm",
    "direction_norm",
    "fisher_curvature",
    "damped_curvature",
    "moment_alignment",
    "pcg",
}
_PCG_FIELDS = {
    "iterations",
    "residual_norm",
    "relative_residual",
    "converged",
    "reason",
}
_UPDATE_FIELDS = {
    "schema_version",
    "target_kl",
    "initialization",
    "initial_step_size",
    "fisher_curvature",
    "best_step_size",
    "best_measured_kl",
    "applied_step_size",
    "applied_measured_kl",
    "line_search_evaluations",
    "converged",
    "applied",
    "reference_forward_evaluations",
    "tangent_dimension",
    "a_state_sha256",
}


@dataclass(frozen=True, slots=True)
class _SavedUpdate:
    direction: torch.Tensor
    beta: float
    train_fisher_curvature: float
    applied_alpha: float
    applied_measured_forward_kl: float
    target_kl: float


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _validate_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return value


def _close(recorded: float, expected: float) -> bool:
    return math.isclose(recorded, expected, rel_tol=1.0e-9, abs_tol=1.0e-12)


def _read_bound_json(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    """Parse exactly the bytes whose digest is returned."""

    source = Path(path)
    raw = source.read_bytes()
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds {_MAX_JSON_BYTES} bytes: {source}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            value[key] = item
        return value

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value, hashlib.sha256(raw).hexdigest()


def _require_exact_fields(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields do not match its schema: "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )


def _parse_direction(
    value: object,
    *,
    learner: str,
    dimension: int,
    expected_beta: float,
    expected_relative_damping: float,
) -> tuple[torch.Tensor, float]:
    name = f"learners.{learner}.direction"
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    _require_exact_fields(value, _DIRECTION_FIELDS, name=name)
    if value["schema_version"] != _DIRECTION_SCHEMA_VERSION:
        raise ValueError(f"{name} has an unsupported schema")

    raw_direction = value["direction"]
    if isinstance(raw_direction, (str, bytes, bytearray)) or not isinstance(
        raw_direction, Sequence
    ):
        raise TypeError(f"{name}.direction must be a sequence")
    if len(raw_direction) != dimension:
        raise ValueError(f"{name}.direction has length {len(raw_direction)}, expected {dimension}")
    direction = torch.tensor(
        tuple(
            _finite_float(item, name=f"{name}.direction[{index}]")
            for index, item in enumerate(raw_direction)
        ),
        dtype=torch.float64,
        device="cpu",
    )
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    recorded_norm = _positive_float(value["direction_norm"], name=f"{name}.direction_norm")
    if not _close(recorded_norm, direction_norm):
        raise ValueError(f"{name}.direction_norm does not match the serialized direction")

    beta = _positive_float(value["beta"], name=f"{name}.beta")
    if not _close(beta, expected_beta):
        raise ValueError(f"{name}.beta does not match objective.beta")
    relative_damping = _positive_float(
        value["relative_damping"],
        name=f"{name}.relative_damping",
    )
    if not _close(relative_damping, expected_relative_damping):
        raise ValueError(f"{name}.relative_damping does not match the primary configured damping")
    for field in (
        "absolute_damping",
        "mean_fisher_diagonal",
        "moment_norm",
        "damped_curvature",
    ):
        _nonnegative_float(value[field], name=f"{name}.{field}")
    _finite_float(value["moment_alignment"], name=f"{name}.moment_alignment")
    train_curvature = _positive_float(
        value["fisher_curvature"],
        name=f"{name}.fisher_curvature",
    )

    pcg = value["pcg"]
    if not isinstance(pcg, Mapping):
        raise TypeError(f"{name}.pcg must be an object")
    _require_exact_fields(pcg, _PCG_FIELDS, name=f"{name}.pcg")
    iterations = pcg["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError(f"{name}.pcg.iterations must be a non-negative integer")
    _nonnegative_float(pcg["residual_norm"], name=f"{name}.pcg.residual_norm")
    _nonnegative_float(
        pcg["relative_residual"],
        name=f"{name}.pcg.relative_residual",
    )
    if pcg["converged"] is not True or pcg["reason"] not in {"converged", "zero_rhs"}:
        raise ValueError(f"{name}.pcg must record a converged solve")
    return direction, train_curvature


def _parse_update(
    value: object,
    *,
    learner: str,
    dimension: int,
    expected_target_kl: float,
    direction_train_curvature: float,
) -> tuple[float, float, float]:
    name = f"learners.{learner}.measured_kl_update"
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    _require_exact_fields(value, _UPDATE_FIELDS, name=name)
    if value["schema_version"] != _UPDATE_SCHEMA_VERSION:
        raise ValueError(f"{name} has an unsupported schema")
    if value["initialization"] != "train_fisher_quadratic":
        raise ValueError(f"{name}.initialization must use the saved train Fisher curvature")
    if value["converged"] is not True or value["applied"] is not True:
        raise ValueError(f"{name} must record a converged, applied update")
    if value["reference_forward_evaluations"] != 1:
        raise ValueError(f"{name} must record exactly one reference forward evaluation")
    if value["tangent_dimension"] != dimension:
        raise ValueError(f"{name}.tangent_dimension does not match the artifact")
    _positive_integer(
        value["line_search_evaluations"],
        name=f"{name}.line_search_evaluations",
    )
    _positive_float(value["initial_step_size"], name=f"{name}.initial_step_size")
    _positive_float(value["best_step_size"], name=f"{name}.best_step_size")
    _nonnegative_float(value["best_measured_kl"], name=f"{name}.best_measured_kl")

    target_kl = _positive_float(value["target_kl"], name=f"{name}.target_kl")
    if not _close(target_kl, expected_target_kl):
        raise ValueError(f"{name}.target_kl does not match evaluation.kl_budget")
    update_curvature = _positive_float(
        value["fisher_curvature"],
        name=f"{name}.fisher_curvature",
    )
    if not _close(update_curvature, direction_train_curvature):
        raise ValueError(f"{name}.fisher_curvature does not match direction evidence")
    alpha = _positive_float(value["applied_step_size"], name=f"{name}.applied_step_size")
    measured_kl = _positive_float(
        value["applied_measured_kl"],
        name=f"{name}.applied_measured_kl",
    )
    _validate_digest(value["a_state_sha256"], name=f"{name}.a_state_sha256")
    return alpha, measured_kl, target_kl


def _parse_saved_updates(
    learner_evidence: Mapping[str, object],
    *,
    dimension: int,
    beta: float,
    relative_damping: float,
    fixed_k: float,
) -> dict[str, _SavedUpdate]:
    if set(learner_evidence) != set(CANONICAL_LEARNERS):
        raise ValueError(f"learner evidence must contain exactly {CANONICAL_LEARNERS!r}")
    parsed: dict[str, _SavedUpdate] = {}
    for learner in CANONICAL_LEARNERS:
        record = learner_evidence[learner]
        if not isinstance(record, Mapping):
            raise TypeError(f"learners.{learner} must be an object")
        if "direction" not in record or "measured_kl_update" not in record:
            raise ValueError(
                f"learners.{learner} must contain direction and measured_kl_update evidence"
            )
        direction, curvature = _parse_direction(
            record["direction"],
            learner=learner,
            dimension=dimension,
            expected_beta=beta,
            expected_relative_damping=relative_damping,
        )
        alpha, measured_kl, target_kl = _parse_update(
            record["measured_kl_update"],
            learner=learner,
            dimension=dimension,
            expected_target_kl=fixed_k,
            direction_train_curvature=curvature,
        )
        parsed[learner] = _SavedUpdate(
            direction=direction,
            beta=beta,
            train_fisher_curvature=curvature,
            applied_alpha=alpha,
            applied_measured_forward_kl=measured_kl,
            target_kl=target_kl,
        )
    return parsed


@torch.no_grad()
def evaluate_saved_policy_directions(
    test: EvaluationTensorData,
    learner_evidence: Mapping[str, object],
    *,
    beta: float,
    relative_damping: float,
    fixed_k: float,
    prompt_chunk_size: int = 16,
) -> dict[str, object]:
    """Evaluate actual train-derived directions on held-out ``F_test``/``g*``.

    All arithmetic is performed on CPU in float64.  The Fisher is applied
    matrix-free, so the function never materializes a ``D x D`` matrix.
    """

    if not isinstance(test, EvaluationTensorData):
        raise TypeError("test must be EvaluationTensorData")
    beta_value = _positive_float(beta, name="beta")
    damping_value = _positive_float(relative_damping, name="relative_damping")
    fixed_k_value = _positive_float(fixed_k, name="fixed_k")
    chunk_size = _positive_integer(prompt_chunk_size, name="prompt_chunk_size")
    parsed = _parse_saved_updates(
        learner_evidence,
        dimension=test.policy_dimension,
        beta=beta_value,
        relative_damping=damping_value,
        fixed_k=fixed_k_value,
    )

    target_moment = torch.zeros(test.policy_dimension, dtype=torch.float64, device="cpu")
    fisher_square_sums = {
        learner: torch.zeros((), dtype=torch.float64, device="cpu")
        for learner in CANONICAL_LEARNERS
    }
    scores = test.policy_scores.detach()
    rewards = test.true_rewards.detach()
    for start in range(0, test.num_prompts, chunk_size):
        stop = min(start + chunk_size, test.num_prompts)
        score_chunk = scores[start:stop].to(device="cpu", dtype=torch.float64)
        reward_chunk = rewards[start:stop].to(device="cpu", dtype=torch.float64)
        centered_scores = score_chunk - score_chunk.mean(dim=1, keepdim=True)
        centered_rewards = reward_chunk - reward_chunk.mean(dim=1, keepdim=True)
        target_moment.add_(torch.einsum("pmd,pm->d", centered_scores, centered_rewards))
        for learner, update in parsed.items():
            projections = torch.einsum("pmd,d->pm", score_chunk, update.direction)
            fisher_square_sums[learner].add_(torch.sum(projections.square()))

    target_moment.div_(test.num_prompts * (test.num_candidates - 1))
    if not bool(torch.isfinite(target_moment).all()):
        raise RuntimeError("test target moment computation produced NaN or infinity")
    target_norm = float(torch.linalg.vector_norm(target_moment).item())
    fisher_denominator = test.num_prompts * test.num_candidates

    learners: dict[str, object] = {}
    for learner in CANONICAL_LEARNERS:
        update = parsed[learner]
        linear = float(torch.dot(target_moment, update.direction).item())
        curvature = float((fisher_square_sums[learner] / fisher_denominator).item())
        if not math.isfinite(curvature) or curvature <= 0.0:
            raise ValueError(f"{learner} has zero or invalid curvature on the test Fisher")
        fisher_norm = math.sqrt(curvature)
        native_quadratic_kl = 0.5 * curvature
        native_utility = linear - beta_value * native_quadratic_kl

        alpha = update.applied_alpha
        applied_linear = alpha * linear
        applied_curvature = alpha * alpha * curvature
        applied_quadratic_kl = 0.5 * applied_curvature
        applied_utility = applied_linear - beta_value * applied_quadratic_kl

        normalization_step = math.sqrt(2.0 * fixed_k_value / curvature)
        normalized_linear_gain = normalization_step * linear
        normalized_utility = normalized_linear_gain - beta_value * fixed_k_value
        train_native_quadratic_kl = 0.5 * update.train_fisher_curvature

        learners[learner] = {
            "saved_train_update": {
                "beta": beta_value,
                "applied_alpha": alpha,
                "beta_eff": beta_value / alpha,
                "train_native_fisher_norm": math.sqrt(update.train_fisher_curvature),
                "train_native_quadratic_kl": train_native_quadratic_kl,
                "train_applied_quadratic_kl": alpha * alpha * train_native_quadratic_kl,
                "measured_applied_forward_kl": update.applied_measured_forward_kl,
                "measured_kl_target": update.target_kl,
            },
            "test_estimands": {
                "target_linear_term": linear,
                "native_fisher_norm": fisher_norm,
                "native_quadratic_kl": native_quadratic_kl,
                "native_fixed_beta_utility": native_utility,
                "applied_alpha": alpha,
                "applied_target_linear_term": applied_linear,
                "applied_fisher_norm": alpha * fisher_norm,
                "applied_quadratic_kl": applied_quadratic_kl,
                "applied_fixed_beta_utility": applied_utility,
                "reward_gain_per_fisher_norm": linear / fisher_norm,
                "fixed_k_normalization_step": normalization_step,
                "fixed_k_linear_gain": normalized_linear_gain,
                "fixed_k_fixed_beta_utility": normalized_utility,
            },
        }

    bt_test = learners[BT_MLE]["test_estimands"]
    prorm_test = learners[PRORM_PLUS]["test_estimands"]
    if not isinstance(bt_test, Mapping) or not isinstance(prorm_test, Mapping):
        raise RuntimeError("internal learner estimand assembly failed")
    compared_metrics = (
        "native_fixed_beta_utility",
        "applied_fixed_beta_utility",
        "reward_gain_per_fisher_norm",
        "fixed_k_linear_gain",
        "fixed_k_fixed_beta_utility",
    )
    paired_differences = {
        metric: float(prorm_test[metric]) - float(bt_test[metric]) for metric in compared_metrics
    }

    return {
        "geometry": {
            "split": "test",
            "num_prompts": test.num_prompts,
            "num_candidates_per_prompt": test.num_candidates,
            "tangent_dimension": test.policy_dimension,
            "fisher_estimator": "raw_node_mean",
            "target_moment_estimator": "per_prompt_unbiased_covariance",
            "target_moment_l2_norm": target_norm,
            "matrix_free": True,
            "device": "cpu",
            "dtype": "float64",
        },
        "estimand_definitions": {
            "fixed_beta": {
                "beta": beta_value,
                "formula": "g_star_dot_delta - beta_over_2_delta_T_F_test_delta",
            },
            "applied_alpha": {
                "formula": "delta_applied = applied_alpha_times_saved_train_direction",
            },
            "fixed_k_normalized": {
                "test_quadratic_kl": fixed_k_value,
                "formula": "delta_K = sqrt(2K_over_d_T_F_test_d)_times_d",
                "higher_is_better": True,
            },
        },
        "learners": learners,
        "paired_differences_prorm_plus_minus_bt": paired_differences,
    }


def _canonical_rollout_learners(
    rollout: Mapping[str, object],
) -> dict[str, object]:
    schema = rollout.get("schema_version")
    if schema == MATCHED_KL_ROLLOUT_SCHEMA_V2:
        serialized = CANONICAL_LEARNERS
    elif schema == MATCHED_KL_ROLLOUT_SCHEMA_V1:
        serialized = LEGACY_V1_LEARNERS
    else:
        raise ValueError(f"unsupported matched-KL rollout schema: {schema!r}")
    raw_learners = rollout.get("learners")
    if not isinstance(raw_learners, Mapping) or set(raw_learners) != set(serialized):
        raise ValueError(f"rollout learners must contain exactly {serialized!r}")
    return {
        canonical: raw_learners[serialized_name]
        for canonical, serialized_name in zip(
            CANONICAL_LEARNERS,
            serialized,
            strict=True,
        )
    }


def _declared_seeds(config: Mapping[str, object]) -> tuple[int, ...]:
    run = config["run"]
    if not isinstance(run, Mapping):
        raise TypeError("run must be an object")
    if "seed" in run:
        return (int(run["seed"]),)
    seeds = run["seeds"]
    if not isinstance(seeds, Sequence):
        raise TypeError("run.seeds must be a sequence")
    return tuple(int(seed) for seed in seeds)


def audit_phase1_estimands(
    config: Mapping[str, object],
    *,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    comparison_json: str | os.PathLike[str],
    rollout_json: str | os.PathLike[str],
    output_json: str | os.PathLike[str],
    prompt_chunk_size: int = 16,
) -> dict[str, object]:
    """Bind Phase-1 evidence and atomically write one CPU-only audit record."""

    normalized = validate_config(config)
    validated_seed = _validate_seed(seed)
    if validated_seed not in _declared_seeds(normalized):
        raise ValueError("seed is not declared by the validated configuration")
    digest = config_hash(normalized)
    objective = normalized["objective"]
    evaluation = normalized["evaluation"]
    if not isinstance(objective, Mapping) or not isinstance(evaluation, Mapping):
        raise TypeError("objective and evaluation must be objects")
    beta = float(objective["beta"])
    relative_damping = float(objective["damping_relative_to_mean_fisher_diagonal"])
    fixed_k = float(evaluation["kl_budget"])

    artifact_digest = artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=validated_seed,
    )
    experiment = load_controlled_feature_artifact(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=validated_seed,
    )
    comparison, comparison_digest = _read_bound_json(comparison_json)
    parse_comparison_heads(
        comparison,
        expected_config_hash=digest,
        expected_seed=validated_seed,
        expected_artifact_metadata_sha256=artifact_digest,
        expected_dimension=experiment.train.reward_dimension,
    )
    rollout, rollout_digest = _read_bound_json(rollout_json)

    if rollout.get("config_hash") != digest:
        raise ValueError("rollout config_hash does not match the validated configuration")
    if rollout.get("seed") != validated_seed:
        raise ValueError("rollout seed does not match the requested seed")
    if rollout.get("artifact_metadata_sha256") != artifact_digest:
        raise ValueError("rollout artifact metadata does not match the loaded artifact")
    if rollout.get("comparison_sha256") != comparison_digest:
        raise ValueError("rollout comparison_sha256 does not match the parsed comparison bytes")
    if rollout.get("run_manifest_sha256") != comparison.get("run_manifest_sha256"):
        raise ValueError("rollout and comparison bind different run manifests")
    if rollout.get("environment_identity") != comparison.get("environment_identity"):
        raise ValueError("rollout and comparison bind different execution environments")
    if rollout.get("train_oracle_values_accessed") is not False:
        raise ValueError("rollout must attest that train oracle values were not accessed")
    if rollout.get("raw_oracle_values_serialized") is not False:
        raise ValueError("rollout must attest that raw oracle values were not serialized")

    learner_evidence = _canonical_rollout_learners(rollout)
    evaluated = evaluate_saved_policy_directions(
        experiment.test,
        learner_evidence,
        beta=beta,
        relative_damping=relative_damping,
        fixed_k=fixed_k,
        prompt_chunk_size=prompt_chunk_size,
    )
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ESTIMAND_AUDIT_SCHEMA_VERSION,
        "config_hash": digest,
        "seed": validated_seed,
        "sources": {
            "artifact_dir": relative_posix_reference(
                artifact_dir,
                base=destination.parent,
            ),
            "artifact_metadata_sha256": artifact_digest,
            "comparison_json": relative_posix_reference(
                comparison_json,
                base=destination.parent,
            ),
            "comparison_sha256": comparison_digest,
            "matched_kl_rollout_json": relative_posix_reference(
                rollout_json,
                base=destination.parent,
            ),
            "matched_kl_rollout_sha256": rollout_digest,
        },
        "computation": {
            "policy_or_oracle_model_loaded": False,
            "reward_head_retrained": False,
            "natural_direction_resolved": False,
            "uses_serialized_actual_train_directions": True,
        },
        **evaluated,
    }
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "ESTIMAND_AUDIT_SCHEMA_VERSION",
    "audit_phase1_estimands",
    "evaluate_saved_policy_directions",
]

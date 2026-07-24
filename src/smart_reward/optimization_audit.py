"""CPU-only first-order optimization audit for saved Phase-1 reward heads.

This diagnostic reconstructs the two immutable heads from a hash-bound
controlled comparison and evaluates full-train objectives and *unclipped*
gradients on copies of those heads.  It never constructs an optimizer and
never changes the saved parameters.

BT-MLE uses the exact repeated-label, label-count-weighted likelihood.  ProRM+
solves a fresh FP64 dual system at the frozen head, requires that inner PCG
solve to converge, detaches the solution, and differentiates the corresponding
envelope surrogate.  The resulting gradient is therefore the outer
first-order diagnostic, not an optimizer step and not a convergence claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    artifact_metadata_sha256,
    load_controlled_feature_artifact,
)
from .baseline import repeated_btl_nll
from .config import config_hash, validate_config
from .contracts import (
    BT_MLE,
    CANONICAL_LEARNERS,
    CONTROLLED_COMPARISON_SCHEMA_V1,
    CONTROLLED_COMPARISON_SCHEMA_V2,
    LEGACY_V1_LEARNERS,
    PRORM_PLUS,
)
from .experiment import (
    TrainingTensorData,
    compile_feature_experiment_config,
)
from .linear import DampedEmpiricalFisher
from .objective import (
    dual_loss,
    dual_saddle_value,
    empirical_moment,
    envelope_surrogate,
    envelope_weights,
)
from .paths import relative_posix_reference
from .pcg import pcg
from .phase1_rollout import parse_comparison_heads
from .repro import atomic_write_json

OPTIMIZATION_AUDIT_SCHEMA_VERSION = "optimization-audit/v1"

_MAX_JSON_BYTES = 64 * 1024 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAIN_RESULT_COMMON_FIELDS = {
    "config",
    "train_absolute_damping",
    "bt_mle",
    "heldout_used_for_training",
}
_LEARNER_FIELDS = {
    "method",
    "initial_train_objective",
    "final_train_objective",
    "initial_head_sha256",
    "head_sha256",
    "head_weight",
    "validation",
    "test",
    "final_pcg",
}
_PCG_EVIDENCE_FIELDS = {
    "iterations",
    "residual_norm",
    "relative_residual",
    "converged",
}


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


def _validate_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return value


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_exact_fields(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields do not match its schema: "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )


def _read_bound_json(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds {_MAX_JSON_BYTES} bytes: {source}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value, hashlib.sha256(raw).hexdigest()


def _head_sha256(weight: torch.Tensor) -> str:
    value = weight.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(bytes(value.view(torch.uint8).tolist()))
    return digest.hexdigest()


@torch.no_grad()
def _mean_squared_score_fp64(scores: torch.Tensor, *, prompt_chunk_size: int = 16) -> float:
    """Compute ``mean(S**2)`` without retaining a second full score tensor."""

    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise TypeError("policy scores must be a three-dimensional tensor")
    total = torch.zeros((), dtype=torch.float64, device="cpu")
    for start in range(0, scores.shape[0], prompt_chunk_size):
        chunk = scores[start : start + prompt_chunk_size].to(
            device="cpu",
            dtype=torch.float64,
        )
        total.add_(torch.sum(chunk.square()))
    result = float((total / scores.numel()).item())
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("artifact train Fisher has invalid mean diagonal")
    return result


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


def _main_comparison_result(
    comparison: Mapping[str, object],
) -> tuple[Mapping[str, object], tuple[str, str], str]:
    schema = comparison.get("schema_version")
    if schema == CONTROLLED_COMPARISON_SCHEMA_V2:
        serialized_learners = CANONICAL_LEARNERS
        win_field = "prorm_plus_win_guaranteed"
    elif schema == CONTROLLED_COMPARISON_SCHEMA_V1:
        serialized_learners = LEGACY_V1_LEARNERS
        win_field = "srm_win_guaranteed"
    else:
        raise ValueError(f"unsupported comparison schema: {schema!r}")
    runs = comparison.get("damping_runs")
    if isinstance(runs, (str, bytes, bytearray)) or not isinstance(runs, Sequence):
        raise TypeError("comparison damping_runs must be a sequence")
    main_runs: list[Mapping[str, object]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping) or set(run) != {"damping_multiplier", "result"}:
            raise ValueError(f"comparison damping_runs[{index}] has an invalid schema")
        multiplier = _finite_float(
            run["damping_multiplier"],
            name=f"damping_runs[{index}].damping_multiplier",
        )
        if multiplier == 1.0:
            main_runs.append(run)
    if len(main_runs) != 1:
        raise ValueError("comparison must contain exactly one damping_multiplier=1 run")
    result = main_runs[0]["result"]
    if not isinstance(result, Mapping):
        raise TypeError("primary comparison result must be an object")
    expected_fields = {
        *_MAIN_RESULT_COMMON_FIELDS,
        serialized_learners[1],
        win_field,
        "train_absolute_damping",
    }
    _require_exact_fields(result, expected_fields, name="primary comparison result")
    if result["heldout_used_for_training"] is not False or result[win_field] is not False:
        raise ValueError("comparison contains an invalid leakage or guaranteed-win claim")
    return result, serialized_learners, win_field


def _comparison_training_records(
    comparison: Mapping[str, object],
    *,
    expected_runtime_config: Mapping[str, object],
    expected_initial_head_sha256: str,
    expected_absolute_damping: float,
) -> dict[str, dict[str, object]]:
    result, serialized_learners, _ = _main_comparison_result(comparison)
    result_config = result["config"]
    if not isinstance(result_config, Mapping) or dict(result_config) != dict(
        expected_runtime_config
    ):
        raise ValueError("primary comparison runtime config does not match the YAML config")
    recorded_damping = _positive_float(
        result["train_absolute_damping"],
        name="train_absolute_damping",
    )
    if not math.isclose(
        recorded_damping,
        expected_absolute_damping,
        rel_tol=1.0e-10,
        abs_tol=1.0e-14,
    ):
        raise ValueError("comparison train_absolute_damping does not match artifact geometry")

    records: dict[str, dict[str, object]] = {}
    for canonical, serialized in zip(
        CANONICAL_LEARNERS,
        serialized_learners,
        strict=True,
    ):
        learner = result[serialized]
        if not isinstance(learner, Mapping):
            raise TypeError(f"comparison learner {serialized!r} must be an object")
        _require_exact_fields(
            learner,
            _LEARNER_FIELDS,
            name=f"comparison learner {serialized!r}",
        )
        if learner["method"] != serialized:
            raise ValueError(f"comparison learner {serialized!r} has the wrong method")
        initial_sha = _validate_digest(
            learner["initial_head_sha256"],
            name=f"{serialized}.initial_head_sha256",
        )
        if initial_sha != expected_initial_head_sha256:
            raise ValueError(f"{serialized} does not bind the expected zero initialization")
        final_sha = _validate_digest(
            learner["head_sha256"],
            name=f"{serialized}.head_sha256",
        )
        initial_objective = _nonnegative_float(
            learner["initial_train_objective"],
            name=f"{serialized}.initial_train_objective",
        )
        final_objective = _nonnegative_float(
            learner["final_train_objective"],
            name=f"{serialized}.final_train_objective",
        )
        final_pcg = learner["final_pcg"]
        pcg_record: dict[str, object] | None
        if canonical == BT_MLE:
            if final_pcg is not None:
                raise ValueError("BT-MLE must not contain final PCG evidence")
            pcg_record = None
        else:
            if not isinstance(final_pcg, Mapping):
                raise TypeError("ProRM+ must contain final PCG evidence")
            _require_exact_fields(
                final_pcg,
                _PCG_EVIDENCE_FIELDS,
                name=f"{serialized}.final_pcg",
            )
            iterations = final_pcg["iterations"]
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
                raise ValueError("comparison ProRM+ final PCG iterations are invalid")
            residual = _nonnegative_float(
                final_pcg["residual_norm"],
                name=f"{serialized}.final_pcg.residual_norm",
            )
            relative_residual = _nonnegative_float(
                final_pcg["relative_residual"],
                name=f"{serialized}.final_pcg.relative_residual",
            )
            if final_pcg["converged"] is not True:
                raise ValueError("comparison ProRM+ final PCG solve did not converge")
            pcg_record = {
                "iterations": iterations,
                "residual_norm": residual,
                "relative_residual": relative_residual,
                "converged": True,
            }
        records[canonical] = {
            "initial_train_objective": initial_objective,
            "final_train_objective": final_objective,
            "initial_to_final_objective_change": final_objective - initial_objective,
            "initial_head_sha256": initial_sha,
            "head_sha256": final_sha,
            "recorded_final_inner_pcg": pcg_record,
        }
    return records


def _head_vector(
    value: object,
    *,
    learner: str,
    dimension: int,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        vector = value.detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            vector = torch.tensor(tuple(value), dtype=torch.float64, device="cpu")
        except (TypeError, ValueError) as error:
            raise TypeError(f"{learner} head must contain real scalars") from error
    else:
        raise TypeError(f"{learner} head must be a tensor or sequence")
    if vector.shape != (dimension,) or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{learner} head must be a finite vector of length {dimension}")
    return vector


def _gradient_diagnostics(head: torch.Tensor, gradient: torch.Tensor) -> dict[str, object]:
    if gradient.shape != head.shape or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("head gradient is missing, malformed, or non-finite")
    head_norm = float(torch.linalg.vector_norm(head.detach()).item())
    gradient_norm = float(torch.linalg.vector_norm(gradient.detach()).item())
    return {
        "gradient_l2_norm": gradient_norm,
        "head_l2_norm": head_norm,
        "gradient_to_head_norm_ratio": (None if head_norm == 0.0 else gradient_norm / head_norm),
        "head_norm_is_zero": head_norm == 0.0,
    }


def _bt_audit(
    feature_differences: torch.Tensor,
    left_wins: torch.Tensor,
    num_annotations: torch.Tensor,
    head_value: torch.Tensor,
) -> dict[str, object]:
    head = head_value.detach().clone().requires_grad_(True)
    margins = feature_differences @ head
    objective = repeated_btl_nll(margins, left_wins, num_annotations)
    (gradient,) = torch.autograd.grad(objective, head, create_graph=False)
    return {
        "objective": float(objective.detach().item()),
        **_gradient_diagnostics(head, gradient),
        "objective_definition": "exact_repeated_label_bt_nll",
        "label_weighting": "each_annotation",
        "gradient": "full_data_unclipped",
    }


def _prorm_audit(
    feature_differences: torch.Tensor,
    edge_scores: torch.Tensor,
    h: torch.Tensor,
    operator: DampedEmpiricalFisher,
    head_value: torch.Tensor,
    *,
    beta: float,
    pcg_max_iterations: int,
    pcg_tolerance: float,
    pcg_absolute_tolerance: float,
    pcg_residual_recompute_interval: int,
) -> dict[str, object]:
    head = head_value.detach().clone().requires_grad_(True)
    margins = feature_differences @ head
    moment = empirical_moment(edge_scores, margins.detach(), h)
    solved = pcg(
        operator.matvec,
        moment,
        inverse_diagonal=None,
        x0=None,
        max_iterations=pcg_max_iterations,
        tolerance=pcg_tolerance,
        absolute_tolerance=pcg_absolute_tolerance,
        residual_recompute_interval=pcg_residual_recompute_interval,
    )
    if not solved.converged:
        raise RuntimeError(
            "fresh FP64 ProRM+ dual solve did not converge: "
            f"iterations={solved.iterations}, "
            f"relative_residual={solved.relative_residual:.3e}"
        )
    direction = solved.solution.detach()
    objective = dual_loss(moment, direction, beta=beta)
    saddle = dual_saddle_value(
        moment,
        direction,
        operator.matvec(direction),
        beta=beta,
    )
    weights = envelope_weights(
        edge_scores,
        direction,
        beta=beta,
        detach_direction=True,
    )
    surrogate = envelope_surrogate(margins, h, weights)
    (gradient,) = torch.autograd.grad(surrogate, head, create_graph=False)
    return {
        "objective": float(objective.detach().item()),
        "dual_saddle_value": float(saddle.detach().item()),
        "dual_loss_minus_saddle_value": float((objective - saddle).detach().item()),
        **_gradient_diagnostics(head, gradient),
        "objective_definition": "damped_fisher_gmm_dual_loss",
        "gradient_definition": "fresh_dual_envelope_gradient",
        "gradient": "full_data_unclipped",
        "fresh_inner_pcg": {
            "dtype": "float64",
            "warm_start_used": False,
            "iterations": solved.iterations,
            "residual_norm": solved.residual_norm,
            "relative_residual": solved.relative_residual,
            "converged": True,
            "reason": solved.reason,
        },
    }


def evaluate_saved_head_optimization(
    train: TrainingTensorData,
    heads: Mapping[str, object],
    *,
    beta: float,
    absolute_damping: float,
    pcg_max_iterations: int,
    pcg_tolerance: float,
    pcg_absolute_tolerance: float = 0.0,
    pcg_residual_recompute_interval: int = 20,
) -> dict[str, object]:
    """Return full-data unclipped first-order diagnostics on frozen head copies."""

    if not isinstance(train, TrainingTensorData):
        raise TypeError("train must be TrainingTensorData")
    if set(heads) != set(CANONICAL_LEARNERS):
        raise ValueError(f"heads must contain exactly {CANONICAL_LEARNERS!r}")
    beta_value = _positive_float(beta, name="beta")
    damping_value = _positive_float(absolute_damping, name="absolute_damping")
    iterations = _positive_integer(pcg_max_iterations, name="pcg_max_iterations")
    tolerance = _positive_float(pcg_tolerance, name="pcg_tolerance")
    absolute_tolerance = _nonnegative_float(
        pcg_absolute_tolerance,
        name="pcg_absolute_tolerance",
    )
    recompute_interval = _positive_integer(
        pcg_residual_recompute_interval,
        name="pcg_residual_recompute_interval",
    )

    batch = train.to_training_batch()
    # These design matrices are first formed exactly as in training (in the
    # artifact dtype) and then promoted.  All audit arithmetic is FP64 CPU.
    feature_differences = (batch.left_features - batch.right_features).to(
        device="cpu",
        dtype=torch.float64,
    )
    edge_scores = batch.edge_scores.to(device="cpu", dtype=torch.float64)
    node_scores = batch.node_scores.to(device="cpu", dtype=torch.float64)
    h = batch.h.to(device="cpu", dtype=torch.float64)
    left_wins = batch.left_wins.to(device="cpu")
    num_annotations = batch.num_annotations.to(device="cpu")
    parsed_heads = {
        learner: _head_vector(
            heads[learner],
            learner=learner,
            dimension=train.reward_dimension,
        )
        for learner in CANONICAL_LEARNERS
    }
    operator = DampedEmpiricalFisher(node_scores, damping_value)

    bt = _bt_audit(
        feature_differences,
        left_wins,
        num_annotations,
        parsed_heads[BT_MLE],
    )
    prorm = _prorm_audit(
        feature_differences,
        edge_scores,
        h,
        operator,
        parsed_heads[PRORM_PLUS],
        beta=beta_value,
        pcg_max_iterations=iterations,
        pcg_tolerance=tolerance,
        pcg_absolute_tolerance=absolute_tolerance,
        pcg_residual_recompute_interval=recompute_interval,
    )
    return {
        "geometry": {
            "split": "train",
            "num_edges": train.num_prompts,
            "num_annotations": int(num_annotations.sum().item()),
            "reward_head_dimension": train.reward_dimension,
            "policy_tangent_dimension": train.policy_dimension,
            "absolute_damping": damping_value,
            "device": "cpu",
            "dtype": "float64",
        },
        "learners": {
            BT_MLE: bt,
            PRORM_PLUS: prorm,
        },
        "optimizer_constructed": False,
        "optimizer_step_called": False,
        "saved_heads_mutated": False,
    }


def audit_phase1_head_optimization(
    config: Mapping[str, object],
    *,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    comparison_json: str | os.PathLike[str],
    output_json: str | os.PathLike[str],
) -> dict[str, object]:
    """Verify Phase-1 identities and write one new optimization audit JSON."""

    destination = Path(output_json)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    normalized = validate_config(config)
    validated_seed = _validate_seed(seed)
    if validated_seed not in _declared_seeds(normalized):
        raise ValueError("seed is not declared by the validated configuration")
    digest = config_hash(normalized)
    runtime = compile_feature_experiment_config(normalized, damping_multiplier=1.0)
    runtime_dict = asdict(runtime)

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
    if experiment.train.reward_features.dtype != torch.float32:
        raise ValueError("formal Phase-1 comparison heads must use artifact float32 features")
    comparison, comparison_digest = _read_bound_json(comparison_json)
    heads = parse_comparison_heads(
        comparison,
        expected_config_hash=digest,
        expected_seed=validated_seed,
        expected_artifact_metadata_sha256=artifact_digest,
        expected_dimension=experiment.train.reward_dimension,
    )

    zero_head = torch.zeros(experiment.train.reward_dimension, dtype=torch.float32)
    zero_head_digest = _head_sha256(zero_head)
    mean_fisher_diagonal = _mean_squared_score_fp64(experiment.train.policy_scores)
    absolute_damping = runtime.relative_damping * mean_fisher_diagonal
    if not math.isfinite(absolute_damping) or absolute_damping <= 0.0:
        raise ValueError("artifact train Fisher has invalid mean diagonal")
    records = _comparison_training_records(
        comparison,
        expected_runtime_config=runtime_dict,
        expected_initial_head_sha256=zero_head_digest,
        expected_absolute_damping=absolute_damping,
    )
    main_result, _, _ = _main_comparison_result(comparison)
    recorded_absolute_damping = float(main_result["train_absolute_damping"])
    evaluated = evaluate_saved_head_optimization(
        experiment.train,
        heads,
        beta=runtime.beta,
        # Use the exact serialized scalar from training after verifying it
        # against artifact geometry above.
        absolute_damping=recorded_absolute_damping,
        pcg_max_iterations=runtime.pcg_max_iterations,
        pcg_tolerance=runtime.pcg_tolerance,
        pcg_absolute_tolerance=runtime.pcg_absolute_tolerance,
        pcg_residual_recompute_interval=runtime.pcg_residual_recompute_interval,
    )
    geometry = evaluated["geometry"]
    if not isinstance(geometry, dict):
        raise RuntimeError("internal optimization geometry assembly failed")
    geometry["artifact_recomputed_absolute_damping"] = absolute_damping
    geometry["comparison_recorded_absolute_damping"] = recorded_absolute_damping
    geometry["recorded_minus_recomputed_absolute_damping"] = (
        recorded_absolute_damping - absolute_damping
    )

    learners: dict[str, object] = {}
    for learner in CANONICAL_LEARNERS:
        audit_record = evaluated["learners"][learner]
        comparison_record = records[learner]
        if not isinstance(audit_record, Mapping):
            raise RuntimeError("internal optimization audit assembly failed")
        audit_objective = float(audit_record["objective"])
        recorded_final = float(comparison_record["final_train_objective"])
        learners[learner] = {
            "comparison_training_record": comparison_record,
            "frozen_head_full_data_audit": dict(audit_record),
            "objective_binding": {
                "audit_minus_comparison_final": audit_objective - recorded_final,
                "audit_objective": audit_objective,
                "comparison_final_objective": recorded_final,
            },
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": OPTIMIZATION_AUDIT_SCHEMA_VERSION,
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
            "comparison_schema_version": comparison["schema_version"],
        },
        "geometry": geometry,
        "learners": learners,
        "diagnostic_contract": {
            "full_data": True,
            "gradient_clipping_applied": False,
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "saved_heads_mutated": False,
            "optimization_convergence_threshold_declared": False,
            "optimization_convergence_claimed": False,
            "inner_pcg_convergence_is_not_outer_optimization_convergence": True,
        },
    }
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "OPTIMIZATION_AUDIT_SCHEMA_VERSION",
    "audit_phase1_head_optimization",
    "evaluate_saved_head_optimization",
]

"""Train-only prompt cross-fit for inverse-Fisher regularization selection."""

from __future__ import annotations

import hashlib
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
from .evaluation import GeometrySettings, solve_natural_direction, validate_natural_direction
from .exact import ExactSplitData, empirical_fisher_score_rows, policy_reward_moment
from .linear import DampedEmpiricalFisher
from .policy_update import scale_direction_to_quadratic_kl
from .runtime import producer_identity, sha256_file

CROSSFIT_SCHEMA = "prorm-fisher-crossfit/v1"
SELECTION_SCHEMA = "prorm-fisher-selection/v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def prompt_fold_assignment(prompt_ids: Sequence[str], folds: int) -> tuple[int, ...]:
    """Return a balanced fold assignment depending only on prompt IDs."""

    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if len(prompt_ids) < folds:
        raise ValueError("number of prompts must be at least the number of folds")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt_ids must be unique")
    order = sorted(
        range(len(prompt_ids)),
        key=lambda index: (
            hashlib.sha256(prompt_ids[index].encode("utf-8")).digest(),
            prompt_ids[index],
        ),
    )
    assignments = [-1] * len(prompt_ids)
    for rank, index in enumerate(order):
        assignments[index] = rank % folds
    return tuple(assignments)


def _subset(split: ExactSplitData, indices: Sequence[int]) -> ExactSplitData:
    index = torch.tensor(indices, device=split.policy_scores.device, dtype=torch.long)
    return ExactSplitData(
        prompt_ids=tuple(split.prompt_ids[item] for item in indices),
        policy_scores=split.policy_scores.index_select(0, index),
        reward_features=split.reward_features.index_select(0, index),
        true_rewards=split.true_rewards.index_select(0, index),
    )


def _finite_pool_metrics(split: ExactSplitData, update: torch.Tensor) -> dict[str, float]:
    scores = split.policy_scores.to(dtype=torch.float64)
    rewards = split.true_rewards.to(dtype=torch.float64)
    delta = update.to(device=scores.device, dtype=torch.float64)
    logits = torch.einsum("pmd,d->pm", scores, delta)
    probabilities = torch.softmax(logits, dim=1)
    reference_reward = rewards.mean(dim=1)
    updated_reward = (probabilities * rewards).sum(dim=1)
    log_reference = -math.log(split.num_candidates)
    forward_kl = (probabilities * (torch.log(probabilities) - log_reference)).sum(dim=1)
    moment = policy_reward_moment(scores, rewards)
    rows = empirical_fisher_score_rows(scores, "raw_second_moment")
    raw_fisher = DampedEmpiricalFisher(rows, damping=0.0)
    return {
        "heldout_oracle_reward_improvement": float(
            (updated_reward - reference_reward).mean().item()
        ),
        "heldout_finite_pool_forward_kl": float(forward_kl.mean().item()),
        "heldout_local_reward_improvement": float(torch.dot(moment, delta).item()),
        "heldout_quadratic_forward_kl": 0.5
        * float(torch.dot(delta, raw_fisher.matvec(delta)).item()),
    }


def crossfit_fisher_regularization(
    split: ExactSplitData,
    *,
    folds: int,
    relative_candidates: Sequence[float],
    kl_target: float,
    cg_tolerance: float,
    cg_max_iterations: int,
    residual_recompute_interval: int,
) -> dict[str, Any]:
    """Evaluate damping candidates using only held-out train prompts."""

    assignments = prompt_fold_assignment(split.prompt_ids, folds)
    fold_fingerprint = _canonical_sha256(
        sorted((prompt_id, assignments[index]) for index, prompt_id in enumerate(split.prompt_ids))
    )
    results: dict[str, Any] = {}
    for candidate_raw in relative_candidates:
        candidate = float(candidate_raw)
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise ValueError("relative damping candidates must be finite and positive")
        settings = GeometrySettings(
            fisher_estimator="raw_second_moment",
            relative_damping=candidate,
            cg_tolerance=float(cg_tolerance),
            cg_max_iterations=int(cg_max_iterations),
            residual_recompute_interval=int(residual_recompute_interval),
        )
        fold_records: list[dict[str, Any]] = []
        for fold in range(folds):
            fit_indices = [index for index, value in enumerate(assignments) if value != fold]
            heldout_indices = [index for index, value in enumerate(assignments) if value == fold]
            fit = _subset(split, fit_indices)
            heldout = _subset(split, heldout_indices)
            direction = solve_natural_direction(fit, fit.true_rewards, settings)
            residual = validate_natural_direction(
                fit,
                fit.true_rewards,
                direction,
                settings,
            )
            fit_rows = empirical_fisher_score_rows(
                fit.policy_scores.to(dtype=torch.float64),
                "raw_second_moment",
            )
            fit_fisher = DampedEmpiricalFisher(fit_rows, damping=0.0)
            update, scale, quadratic_kl = scale_direction_to_quadratic_kl(
                direction,
                fit_fisher.matvec,
                kl_target=kl_target,
            )
            metrics = _finite_pool_metrics(heldout, update)
            heldout_moment = policy_reward_moment(
                heldout.policy_scores.to(dtype=torch.float64),
                heldout.true_rewards.to(dtype=torch.float64),
            )
            cosine_denominator = torch.linalg.vector_norm(update) * torch.linalg.vector_norm(
                heldout_moment
            )
            gradient_cosine = (
                0.0
                if float(cosine_denominator.item()) == 0.0
                else float(
                    (torch.dot(update, heldout_moment) / cosine_denominator).clamp(-1.0, 1.0).item()
                )
            )
            fold_records.append(
                {
                    "fold": fold,
                    "fit_prompts": fit.num_prompts,
                    "heldout_prompts": heldout.num_prompts,
                    "relative_damping": candidate,
                    "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                    "step_scale": scale,
                    "fit_quadratic_forward_kl": quadratic_kl,
                    "pcg_relative_residual": residual,
                    "heldout_gradient_cosine": gradient_cosine,
                    **metrics,
                }
            )
        metric_names = [
            "heldout_oracle_reward_improvement",
            "heldout_finite_pool_forward_kl",
            "heldout_local_reward_improvement",
            "heldout_quadratic_forward_kl",
            "heldout_gradient_cosine",
            "direction_norm",
            "pcg_relative_residual",
        ]
        means = {
            name: float(statistics.fmean(float(record[name]) for record in fold_records))
            for name in metric_names
        }
        results[str(candidate)] = {"folds": fold_records, "mean": means}
    return {
        "folds": folds,
        "fold_assignment_sha256": fold_fingerprint,
        "kl_target": float(kl_target),
        "relative_candidates": [float(value) for value in relative_candidates],
        "results": results,
    }


def run_fisher_crossfit(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run and serialize one seed's train-only cross-fit receipt."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError(f"Fisher cross-fit requires protocol {TRPO_PROTOCOL}")
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
    train = _subset(experiment.train, list(range(experiment.train.num_prompts)))
    target_device = torch.device(device)
    train = ExactSplitData(
        prompt_ids=train.prompt_ids,
        policy_scores=train.policy_scores.to(target_device),
        reward_features=train.reward_features.to(target_device),
        true_rewards=train.true_rewards.to(target_device),
    )
    geometry = normalized["geometry"]
    selection = geometry["damping_selection"]
    result = crossfit_fisher_regularization(
        train,
        folds=int(selection["folds"]),
        relative_candidates=[float(value) for value in selection["relative_candidates"]],
        kl_target=float(normalized["policy_update"]["primary_kl_target"]),
        cg_tolerance=float(geometry["cg_tolerance"]),
        cg_max_iterations=int(geometry["cg_max_iterations"]),
        residual_recompute_interval=int(geometry["residual_recompute_interval"]),
    )
    payload = {
        "schema": CROSSFIT_SCHEMA,
        "protocol": TRPO_PROTOCOL,
        "config_sha256": digest,
        "artifact_metadata_sha256": artifact_identity,
        "seed": seed,
        "fit_split": "train",
        **result,
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def _load_crossfit(
    path: str | os.PathLike[str],
    *,
    expected_config_sha256: str,
) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema") != CROSSFIT_SCHEMA
        or value.get("protocol") != TRPO_PROTOCOL
        or value.get("config_sha256") != expected_config_sha256
    ):
        raise ValueError(f"invalid Fisher cross-fit result: {source}")
    return value


def load_fisher_crossfit(
    path: str | os.PathLike[str],
    *,
    expected_config_sha256: str,
    expected_seed: int | None = None,
    expected_artifact_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a cross-fit result and validate its experiment identities."""

    value = _load_crossfit(path, expected_config_sha256=expected_config_sha256)
    if expected_seed is not None and value.get("seed") != expected_seed:
        raise ValueError("Fisher cross-fit seed mismatch")
    if (
        expected_artifact_metadata_sha256 is not None
        and value.get("artifact_metadata_sha256") != expected_artifact_metadata_sha256
    ):
        raise ValueError("Fisher cross-fit artifact mismatch")
    if value.get("fit_split") != "train":
        raise ValueError("Fisher cross-fit did not use the train split")
    return value


def load_fisher_selection(
    path: str | os.PathLike[str],
    *,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Load the global shared damping selection."""

    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema") != SELECTION_SCHEMA
        or value.get("protocol") != TRPO_PROTOCOL
        or value.get("config_sha256") != expected_config_sha256
    ):
        raise ValueError("invalid Fisher selection result")
    selected = value.get("selected_relative_damping")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, (int, float))
        or not math.isfinite(float(selected))
        or float(selected) <= 0.0
    ):
        raise ValueError("selected Fisher damping is invalid")
    return value


def select_fisher_regularization(
    config: Mapping[str, object],
    crossfit_results: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Select one damping rule shared by every seed and reward source."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError(f"Fisher selection requires protocol {TRPO_PROTOCOL}")
    digest = config_hash(normalized)
    records = [
        (load_fisher_crossfit(path, expected_config_sha256=digest), Path(path))
        for path in crossfit_results
    ]
    expected_seeds = list(normalized["run"]["seeds"])
    observed_seeds = [record["seed"] for record, _ in records]
    if sorted(observed_seeds) != sorted(expected_seeds) or len(set(observed_seeds)) != len(
        observed_seeds
    ):
        raise ValueError("cross-fit inputs must contain exactly one result for every seed")
    fold_fingerprints = {record.get("fold_assignment_sha256") for record, _ in records}
    if len(fold_fingerprints) != 1:
        raise ValueError("prompt fold assignments differ across seeds")

    candidates = [
        float(value) for value in normalized["geometry"]["damping_selection"]["relative_candidates"]
    ]
    summaries: dict[str, Any] = {}
    eligible: list[float] = []
    for candidate in candidates:
        key = str(candidate)
        per_seed = {
            str(record["seed"]): float(
                record["results"][key]["mean"]["heldout_oracle_reward_improvement"]
            )
            for record, _ in records
        }
        values = list(per_seed.values())
        mean = float(statistics.fmean(values))
        sample_sd = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        standard_error = sample_sd / math.sqrt(len(values))
        is_eligible = all(value > 0.0 for value in values)
        if is_eligible:
            eligible.append(candidate)
        summaries[key] = {
            "per_seed_mean_reward_improvement": per_seed,
            "mean": mean,
            "sample_standard_deviation": sample_sd,
            "standard_error": standard_error,
            "eligible": is_eligible,
        }
    if not eligible:
        raise RuntimeError("no Fisher damping candidate has positive held-out mean in every seed")
    best = max(eligible, key=lambda value: summaries[str(value)]["mean"])
    threshold = summaries[str(best)]["mean"] - summaries[str(best)]["standard_error"]
    one_standard_error_set = [
        value for value in eligible if summaries[str(value)]["mean"] >= threshold
    ]
    selected = max(one_standard_error_set)
    payload = {
        "schema": SELECTION_SCHEMA,
        "protocol": TRPO_PROTOCOL,
        "config_sha256": digest,
        "selection_metric": "heldout_oracle_reward_improvement",
        "eligibility": "positive_mean_each_seed",
        "tie_break": "largest_in_best_one_standard_error_set",
        "best_mean_candidate": best,
        "best_one_standard_error_threshold": threshold,
        "one_standard_error_set": one_standard_error_set,
        "selected_relative_damping": selected,
        "candidate_summaries": summaries,
        "fold_assignment_sha256": next(iter(fold_fingerprints)),
        "inputs": {str(record["seed"]): sha256_file(path) for record, path in records},
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


__all__ = [
    "CROSSFIT_SCHEMA",
    "SELECTION_SCHEMA",
    "crossfit_fisher_regularization",
    "load_fisher_crossfit",
    "load_fisher_selection",
    "prompt_fold_assignment",
    "run_fisher_crossfit",
    "select_fisher_regularization",
]

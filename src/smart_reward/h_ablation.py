"""Finite-repeat annotation and MSE ablations for the exact-oracle experiment.

The module deliberately consumes the immutable exact-delta artifact instead of
materializing prompts or candidates again.  It adds the four missing cells of
the frozen 2 x 3 design: Oracle-MSE, H-MLE, H-MSE, and H-Pro.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from .artifacts import exact_delta_artifact_metadata_sha256, load_exact_delta_artifact
from .config import config_hash, load_config
from .evaluation import GeometrySettings, solve_natural_direction
from .exact import (
    ExactDeltaExperiment,
    ExactSplitData,
    MLETrainingConfig,
    ProTrainingConfig,
    _edge_design,
    _head_sha256,
    _mle_parameterization,
    _pro_inverse_fisher,
    _pro_reward_problem,
    empirical_fisher_score_rows,
    pair_indices,
    pairwise_differences,
    policy_reward_moment,
)
from .linear import DampedEmpiricalFisher
from .pcg import pcg
from .runtime import producer_identity, require_module, sha256_file
from .trpo_run import load_trpo_reward_comparison

CONFIG_SCHEMA = "prorm-h-mse-ablation-config/v1"
ANNOTATION_SCHEMA = "prorm-h-annotation-sidecar/v1"
RESULT_SCHEMA = "prorm-h-mse-reward-result/v1"
PROTOCOL = "prorm-h-mse-ablation-beta0p2/v1"
ANNOTATION_RNG_NAMESPACE = "prorm-h-annotation-v1"
FOLD_NAMESPACE = "approx-regret-cross-v1:"

NEW_METHODS = ("oracle_mse", "h_mle", "h_mse", "h_pro")
LEARNED_METHODS = (
    "oracle_mle",
    "oracle_mse",
    "oracle_pro",
    "h_mle",
    "h_mse",
    "h_pro",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise ValueError(f"refusing to replace non-identical output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def load_h_ablation_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported H/MSE ablation config")
    expected_keys = {
        "schema",
        "source_config",
        "source_config_sha256",
        "experiment",
        "annotation",
        "reward_training",
        "reward_evaluation",
        "policy_update",
        "rollout",
    }
    if set(value) != expected_keys:
        raise ValueError("H/MSE ablation config keys changed")
    experiment = value.get("experiment")
    annotation = value.get("annotation")
    reward_evaluation = value.get("reward_evaluation")
    policy_update = value.get("policy_update")
    rollout = value.get("rollout")
    if not isinstance(experiment, dict) or experiment.get("methods") != list(NEW_METHODS):
        raise ValueError("the four-method ablation identity changed")
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("experiment seeds must be a non-empty unique list")
    if not isinstance(annotation, dict) or annotation != {
        "gamma": 0.9,
        "geometric_support": "positive_integers",
        "edge_outer_weight": "uniform_one",
        "rng_namespace": ANNOTATION_RNG_NAMESPACE,
        "store": ["N", "S", "H"],
        "clip_H": False,
    }:
        raise ValueError("formal H annotation estimand changed")
    if reward_evaluation != {
        "metrics": ["NLL", "MSE", "approximate_regret"],
        "approximate_regret_estimator": "two_fold_cross_product",
        "fisher_source": "frozen_train_selected_damped_fisher",
        "folds": 2,
        "fold_namespace": FOLD_NAMESPACE,
        "test_usage": "evaluation_only_no_selection",
    }:
        raise ValueError("reward evaluation estimand changed")
    if policy_update != {
        "method": "one_step_ngd",
        "beta": 0.2,
        "fisher_source": "frozen_train_selected_damped_fisher",
        "step": "direction_divided_by_beta",
    }:
        raise ValueError("policy update estimand changed")
    if not isinstance(rollout, dict):
        raise ValueError("rollout configuration must be an object")
    for name in ("prompts", "base_responses_per_prompt", "responses_per_prompt"):
        raw = rollout.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"rollout.{name} must be positive")
    additional = rollout.get("additional_responses_per_prompt")
    if isinstance(additional, bool) or not isinstance(additional, int) or additional < 0:
        raise ValueError("rollout.additional_responses_per_prompt must be non-negative")
    if rollout["base_responses_per_prompt"] + additional != rollout["responses_per_prompt"]:
        raise ValueError("base and additional rollout counts do not sum to the total")
    if (
        rollout.get("base_seed_namespace") != "real-rollout-batch"
        or rollout.get("additional_seed_namespace") != "real-rollout-extension-4-to-6-batch"
    ):
        raise ValueError("rollout seed namespaces changed")
    if experiment.get("name") == "h-mse-ablation-beta0p2-v1" and (
        rollout["prompts"],
        rollout["base_responses_per_prompt"],
        rollout["additional_responses_per_prompt"],
        rollout["responses_per_prompt"],
    ) != (512, 4, 2, 6):
        raise ValueError("formal m=6 rollout dimensions changed")
    if (
        rollout.get("generation") != "fresh_test_prompt_rollout"
        or rollout.get("kl_estimator") != "rao_blackwellized_updated_policy_forward_kl"
    ):
        raise ValueError("rollout estimand changed")
    return value


def h_config_hash(value: Mapping[str, Any]) -> str:
    return _canonical_sha256(value)


def resolve_source_config(path: str | os.PathLike[str], value: Mapping[str, Any]) -> dict[str, Any]:
    source = load_config(Path(path).resolve().parent / str(value["source_config"]))
    if config_hash(source) != value["source_config_sha256"]:
        raise ValueError("source config digest mismatch")
    if list(value["experiment"]["seeds"]) != list(source["run"]["seeds"]):
        raise ValueError("H/MSE seeds differ from the frozen source")
    if int(source["data"]["num_candidates"]) != 6:
        raise ValueError("H/MSE formal experiment requires six candidates")
    return source


def _edge_seed(base_seed: int, prompt_id: str, left: int, right: int) -> int:
    payload = (f"{ANNOTATION_RNG_NAMESPACE}|{base_seed}|{prompt_id}|{left}|{right}").encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_positive_geometric(rng: random.Random, gamma: float) -> int:
    """Sample P(N=n)=(1-gamma) gamma**(n-1), n >= 1."""

    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    uniform = rng.random()
    return int(math.floor(math.log1p(-uniform) / math.log(gamma))) + 1


def randomized_logit_estimate(successes: int, trials: int, gamma: float) -> float:
    """Stable Russian-roulette estimate of logit(p) from Binomial counts.

    The kth success and failure powers are estimated by falling-factorial
    ratios.  Division by P(N>=k)=gamma**(k-1) removes random-truncation bias.
    """

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    success_ratio = 1.0
    failure_ratio = 1.0
    survival = 1.0
    total = 0.0
    failures = trials - successes
    for order in range(1, trials + 1):
        denominator = trials - order + 1
        success_ratio *= max(successes - order + 1, 0) / denominator
        failure_ratio *= max(failures - order + 1, 0) / denominator
        total += (success_ratio - failure_ratio) / (order * survival)
        survival *= gamma
    if not math.isfinite(total):
        raise FloatingPointError("randomized logit estimate is non-finite")
    return total


def sample_h_annotation(
    delta_r_star: float,
    *,
    gamma: float,
    rng: random.Random,
) -> tuple[int, int, float]:
    if not math.isfinite(delta_r_star):
        raise ValueError("delta_r_star must be finite")
    probability = (
        1.0 / (1.0 + math.exp(-delta_r_star))
        if delta_r_star >= 0
        else (math.exp(delta_r_star) / (1.0 + math.exp(delta_r_star)))
    )
    trials = sample_positive_geometric(rng, gamma)
    successes = sum(rng.random() < probability for _ in range(trials))
    estimate = randomized_logit_estimate(successes, trials, gamma)
    return trials, successes, estimate


def _annotation_summary(
    trials: torch.Tensor,
    successes: torch.Tensor,
    h: torch.Tensor,
) -> dict[str, Any]:
    sorted_n = torch.sort(trials.to(torch.float64)).values
    sorted_abs_h = torch.sort(h.abs()).values

    def quantile(values: torch.Tensor, probability: float) -> float:
        return float(torch.quantile(values, probability).item())

    return {
        "edge_count": int(trials.numel()),
        "total_labels": int(trials.sum().item()),
        "mean_N": float(trials.to(torch.float64).mean().item()),
        "N_quantiles": {
            "p50": quantile(sorted_n, 0.50),
            "p90": quantile(sorted_n, 0.90),
            "p95": quantile(sorted_n, 0.95),
            "p99": quantile(sorted_n, 0.99),
            "max": float(sorted_n[-1].item()),
        },
        "success_rate": float(successes.sum().item() / trials.sum().item()),
        "H": {
            "mean": float(h.mean().item()),
            "sample_sd": float(h.std(unbiased=True).item()),
            "min": float(h.min().item()),
            "max": float(h.max().item()),
            "abs_p90": quantile(sorted_abs_h, 0.90),
            "abs_p95": quantile(sorted_abs_h, 0.95),
            "abs_p99": quantile(sorted_abs_h, 0.99),
        },
    }


def validate_h_annotations(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    sidecar_dir: str | os.PathLike[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    extension = load_h_ablation_config(config_path)
    source = resolve_source_config(config_path, extension)
    if seed not in extension["experiment"]["seeds"]:
        raise ValueError("seed is not configured")
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    root = Path(sidecar_dir)
    metadata = _read_json(root / "metadata.json")
    tensors_path = root / "annotations.safetensors"
    expected = {
        "schema": ANNOTATION_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "artifact_metadata_sha256": artifact_identity,
        "seed": seed,
        "split": "train",
        "gamma": 0.9,
        "geometric_support": "positive_integers",
        "rng_namespace": ANNOTATION_RNG_NAMESPACE,
        "edge_orientation": "lower_candidate_index_first",
        "edge_outer_weight": "uniform_one",
        "edge_count": 46080,
        "tensors_sha256": sha256_file(tensors_path),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("H annotation metadata identity mismatch")
    if not isinstance(metadata.get("summary"), dict) or not isinstance(
        metadata.get("producer"), dict
    ):
        raise ValueError("H annotation evidence is incomplete")
    tensors = require_module("safetensors.torch").load_file(str(tensors_path), device="cpu")
    if set(tensors) != {"N", "S", "H"}:
        raise ValueError("H annotation tensor keys changed")
    n, s, h = tensors["N"], tensors["S"], tensors["H"]
    if n.dtype != torch.int64 or s.dtype != torch.int64 or h.dtype != torch.float64:
        raise ValueError("H annotation tensor dtypes changed")
    if n.shape != s.shape or n.shape != h.shape or n.ndim != 1 or n.numel() != 46080:
        raise ValueError("H annotation tensor shape changed")
    if not bool(((n >= 1) & (s >= 0) & (s <= n)).all()) or not bool(torch.isfinite(h).all()):
        raise ValueError("H annotation values are invalid")
    if metadata["summary"] != _annotation_summary(n, s, h):
        raise ValueError("H annotation summary differs from tensors")
    return metadata, tensors


def materialize_h_annotations(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    extension = load_h_ablation_config(config_path)
    source = resolve_source_config(config_path, extension)
    if seed not in extension["experiment"]["seeds"]:
        raise ValueError("seed is not configured")
    target = Path(output_dir)
    if target.exists():
        metadata, _ = validate_h_annotations(config_path, artifact_dir, target, seed=seed)
        return metadata
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    experiment = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    train = experiment.train
    if train.num_prompts != 3072 or train.num_candidates != 6 or train.num_edges != 46080:
        raise ValueError("frozen train edge population changed")
    margins = pairwise_differences(train.true_rewards.to(torch.float64)).reshape(-1)
    pairs = pair_indices(train.num_candidates).tolist()
    n_values: list[int] = []
    s_values: list[int] = []
    h_values: list[float] = []
    gamma = float(extension["annotation"]["gamma"])
    edge = 0
    for prompt_id in train.prompt_ids:
        for left, right in pairs:
            rng = random.Random(_edge_seed(seed, prompt_id, left, right))
            n, s, h = sample_h_annotation(float(margins[edge].item()), gamma=gamma, rng=rng)
            n_values.append(n)
            s_values.append(s)
            h_values.append(h)
            edge += 1
    if edge != train.num_edges:
        raise RuntimeError("H annotation edge accounting failed")
    tensors = {
        "N": torch.tensor(n_values, dtype=torch.int64),
        "S": torch.tensor(s_values, dtype=torch.int64),
        "H": torch.tensor(h_values, dtype=torch.float64),
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        tensor_path = staging / "annotations.safetensors"
        require_module("safetensors.torch").save_file(tensors, str(tensor_path))
        metadata = {
            "schema": ANNOTATION_SCHEMA,
            "protocol": PROTOCOL,
            "config_sha256": h_config_hash(extension),
            "source_config_sha256": config_hash(source),
            "artifact_metadata_sha256": artifact_identity,
            "seed": seed,
            "split": "train",
            "gamma": gamma,
            "geometric_support": "positive_integers",
            "rng_namespace": ANNOTATION_RNG_NAMESPACE,
            "edge_orientation": "lower_candidate_index_first",
            "edge_outer_weight": "uniform_one",
            "edge_count": train.num_edges,
            "tensors_sha256": sha256_file(tensor_path),
            "summary": _annotation_summary(tensors["N"], tensors["S"], tensors["H"]),
            "producer": producer_identity(),
        }
        _atomic_json(staging / "metadata.json", metadata)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    metadata, _ = validate_h_annotations(config_path, artifact_dir, target, seed=seed)
    return metadata


@dataclass(frozen=True, slots=True)
class AblationFit:
    method: str
    weight: torch.Tensor
    objective: float
    gradient_norm: float
    relative_residual: float
    converged: bool
    iterations: int
    inner_pcg_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "objective": self.objective,
            "gradient_norm": self.gradient_norm,
            "relative_residual": self.relative_residual,
            "converged": self.converged,
            "iterations": self.iterations,
            "inner_pcg_calls": self.inner_pcg_calls,
            "head_sha256": _head_sha256(self.weight),
            "head_weight": self.weight.detach().cpu().tolist(),
        }


def _edge_coordinates(
    train: ExactSplitData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    design, margins = _edge_design(train)
    design = design.to(torch.float64)
    coordinates, head_map = _mle_parameterization(
        design,
        maximum_identifiable_rank=train.num_prompts * (train.num_candidates - 1),
        source_epsilon=torch.finfo(torch.float64).eps,
    )
    if head_map is None:
        head_map = torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
    return design, margins.to(torch.float64), (coordinates, head_map)


def _fit_mse(
    method: str,
    design: torch.Tensor,
    coordinates_and_map: tuple[torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    *,
    tolerance: float,
) -> AblationFit:
    coordinates, head_map = coordinates_and_map
    target = target.to(device=design.device, dtype=torch.float64)
    gram_diagonal = coordinates.square().sum(dim=0)
    if not bool((gram_diagonal > 0.0).all()):
        raise RuntimeError("MSE coordinate design has a zero column")
    off_diagonal_error = torch.linalg.matrix_norm(
        coordinates.mT @ coordinates - torch.diag(gram_diagonal), ord=float("inf")
    )
    gram_scale = float(gram_diagonal.max().item())
    if float(off_diagonal_error.item()) > 1.0e-8 * max(1.0, gram_scale):
        raise RuntimeError("MSE coordinate design is not numerically orthogonal")
    coordinate_weight = (coordinates.mT @ target) / gram_diagonal
    weight = head_map @ coordinate_weight
    error = design @ weight - target
    gradient = 2.0 * (design.mT @ error) / design.shape[0]
    rhs = 2.0 * (design.mT @ target) / design.shape[0]
    rhs_norm = float(torch.linalg.vector_norm(rhs).item())
    relative = float(torch.linalg.vector_norm(gradient).item()) / max(
        rhs_norm, torch.finfo(torch.float64).tiny
    )
    return AblationFit(
        method=method,
        weight=weight.detach().clone(),
        objective=float(error.square().mean().item()),
        gradient_norm=float(torch.linalg.vector_norm(gradient).item()),
        relative_residual=relative,
        converged=relative <= tolerance,
        iterations=1,
    )


def _fit_h_mle(
    design: torch.Tensor,
    coordinates_and_map: tuple[torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    config: MLETrainingConfig,
) -> AblationFit:
    coordinates, head_map = coordinates_and_map
    targets = targets.to(device=design.device, dtype=torch.float64)
    if not bool(((targets >= 0.0) & (targets <= 1.0)).all()):
        raise ValueError("H-MLE frequencies must lie in [0, 1]")
    parameter = torch.zeros(
        coordinates.shape[1], dtype=torch.float64, device=design.device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=1.0,
        max_iter=config.max_iterations,
        history_size=config.history_size,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        total = coordinates.shape[0]
        objective = torch.zeros((), dtype=torch.float64, device=design.device)
        for start in range(0, total, config.microbatch_size):
            stop = min(start + config.microbatch_size, total)
            loss = F.binary_cross_entropy_with_logits(
                coordinates[start:stop] @ parameter, targets[start:stop]
            )
            scale = (stop - start) / total
            (loss * scale).backward()
            objective = objective + loss.detach() * scale
        return objective

    optimizer.step(closure)
    weight = head_map @ parameter.detach()
    logits = design @ weight
    objective = F.binary_cross_entropy_with_logits(logits, targets)
    gradient = design.mT @ (torch.sigmoid(logits) - targets) / design.shape[0]
    gradient_norm = float(torch.linalg.vector_norm(gradient).item())
    iterations = int(optimizer.state[parameter].get("n_iter", 0))
    initial_objective = math.log(2.0)
    improved = float(objective.item()) < initial_objective - 1.0e-10
    return AblationFit(
        method="h_mle",
        weight=weight.detach().clone(),
        objective=float(objective.item()),
        gradient_norm=gradient_norm,
        relative_residual=gradient_norm,
        converged=gradient_norm <= config.gradient_tolerance and improved,
        iterations=max(1, iterations),
    )


def h_policy_moment(train: ExactSplitData, h: torch.Tensor) -> torch.Tensor:
    """Return 1/2 E_edge[(s_i-s_j) H_ij] without materializing edge scores."""

    prompts, candidates, dimension = train.policy_scores.shape
    pairs = pair_indices(candidates, device=train.policy_scores.device)
    values = h.to(device=train.policy_scores.device, dtype=torch.float64).reshape(
        prompts, pairs.shape[0]
    )
    coefficients = torch.zeros(
        prompts, candidates, dtype=torch.float64, device=train.policy_scores.device
    )
    for edge_index, (left, right) in enumerate(pairs.tolist()):
        coefficients[:, left] += values[:, edge_index]
        coefficients[:, right] -= values[:, edge_index]
    numerator = torch.einsum("pmd,pm->d", train.policy_scores.to(torch.float64), coefficients)
    return 0.5 * numerator / (prompts * pairs.shape[0])


def _fit_h_pro(
    train: ExactSplitData,
    target_moment: torch.Tensor,
    config: ProTrainingConfig,
) -> AblationFit:
    feature_moment, _, fisher, inner_tolerance = _pro_reward_problem(train, config)
    target_moment = target_moment.to(device=feature_moment.device, dtype=torch.float64)
    inner_calls = 0

    def inverse_fisher(vector: torch.Tensor) -> torch.Tensor:
        nonlocal inner_calls
        inner_calls += 1
        return _pro_inverse_fisher(fisher, vector, config, tolerance=inner_tolerance)

    target_natural = inverse_fisher(target_moment)
    rhs = feature_moment.mT @ target_natural

    def normal_matvec(vector: torch.Tensor) -> torch.Tensor:
        return feature_moment.mT @ inverse_fisher(feature_moment @ vector)

    diagonal = feature_moment.square().sum(dim=0)
    scale = float(diagonal.mean().item())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("H-Pro feature moment has non-positive scale")
    inverse_diagonal = diagonal.clamp_min(torch.finfo(diagonal.dtype).eps * scale).reciprocal()
    solved = pcg(
        normal_matvec,
        rhs,
        inverse_diagonal=inverse_diagonal,
        max_iterations=config.outer_max_iterations,
        tolerance=config.outer_tolerance,
        residual_recompute_interval=config.residual_recompute_interval,
    )
    if not solved.converged:
        raise RuntimeError(
            f"H-Pro normal equation did not converge: {solved.relative_residual:.3e}"
        )
    weight = solved.solution.detach().clone()
    error_moment = feature_moment @ weight - target_moment
    weighted_error = inverse_fisher(error_moment)
    gradient = feature_moment.mT @ weighted_error
    return AblationFit(
        method="h_pro",
        weight=weight,
        objective=float((torch.dot(error_moment, weighted_error) / 2.0).item()),
        gradient_norm=float(torch.linalg.vector_norm(gradient).item()),
        relative_residual=solved.relative_residual,
        converged=True,
        iterations=max(1, solved.iterations),
        inner_pcg_calls=inner_calls,
    )


def _reward_metrics(
    train: ExactSplitData,
    test: ExactSplitData,
    weight: torch.Tensor,
    *,
    beta: float,
    relative_damping: float,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    predicted = test.reward_features.to(torch.float64) @ weight.to(
        device=test.reward_features.device, dtype=torch.float64
    )
    predicted = predicted - predicted.mean(dim=1, keepdim=True)
    target = test.true_rewards.to(torch.float64)
    target = target - target.mean(dim=1, keepdim=True)
    predicted_edges = pairwise_differences(predicted)
    target_edges = pairwise_differences(target)
    nll = F.binary_cross_entropy_with_logits(predicted_edges, torch.sigmoid(target_edges))
    mse = (predicted - target).square().mean()
    order = sorted(
        range(test.num_prompts),
        key=lambda index: (
            hashlib.sha256((FOLD_NAMESPACE + test.prompt_ids[index]).encode()).digest(),
            test.prompt_ids[index],
        ),
    )
    folds = [order[::2], order[1::2]]
    if [len(fold) for fold in folds] != [test.num_prompts // 2] * 2:
        raise ValueError("two-fold test split must be exactly balanced")
    reward_error = predicted - target
    moments = [
        policy_reward_moment(test.policy_scores[indices].to(torch.float64), reward_error[indices])
        for indices in folds
    ]
    rows = empirical_fisher_score_rows(
        train.policy_scores.to(torch.float64), geometry["fisher_estimator"]
    )
    raw = DampedEmpiricalFisher(rows, damping=0.0)
    damping = relative_damping * float(raw.diagonal().mean().item())
    fisher = DampedEmpiricalFisher(rows, damping=damping)
    solves = [
        pcg(
            fisher.matvec,
            moment,
            inverse_diagonal=fisher.pcg_inverse_diagonal(),
            max_iterations=int(geometry["cg_max_iterations"]),
            tolerance=float(geometry["cg_tolerance"]),
            residual_recompute_interval=int(geometry["residual_recompute_interval"]),
        )
        for moment in moments
    ]
    if not all(result.converged for result in solves):
        raise RuntimeError("two-fold cross-product Fisher solve did not converge")
    cross = 0.5 * (
        torch.dot(moments[0], solves[1].solution) + torch.dot(moments[1], solves[0].solution)
    )
    return {
        "NLL": float(nll.item()),
        "MSE": float(mse.item()),
        "approximate_regret": float(cross.item()) / (2.0 * beta),
        "cross_moment_inverse_fisher_quadratic": float(cross.item()),
        "folds": 2,
        "fold_sizes": [len(fold) for fold in folds],
        "damping": damping,
        "max_pcg_relative_residual": max(result.relative_residual for result in solves),
        "test_usage": "evaluation_only_no_selection",
    }


def _experiment_to_device(
    experiment: ExactDeltaExperiment, device: torch.device
) -> ExactDeltaExperiment:
    def move(split: ExactSplitData) -> ExactSplitData:
        return ExactSplitData(
            prompt_ids=split.prompt_ids,
            policy_scores=split.policy_scores.to(device),
            reward_features=split.reward_features.to(device),
            true_rewards=split.true_rewards.to(device),
        )

    return ExactDeltaExperiment(
        move(experiment.train),
        move(experiment.validation),
        move(experiment.test),
    )


def run_h_ablation_rewards(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    source_reward_result: str | os.PathLike[str],
    fisher_selection: str | os.PathLike[str],
    sidecar_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    extension = load_h_ablation_config(config_path)
    source = resolve_source_config(config_path, extension)
    if seed not in extension["experiment"]["seeds"]:
        raise ValueError("seed is not configured")
    target = Path(output)
    if target.exists():
        result = _read_json(target)
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("config_sha256") != h_config_hash(extension)
            or result.get("seed") != seed
        ):
            raise ValueError("existing H/MSE reward result identity mismatch")
        return result
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    source_reward = load_trpo_reward_comparison(
        source_reward_result,
        expected_config_sha256=extension["source_config_sha256"],
        expected_seed=seed,
    )
    if source_reward["artifact_metadata_sha256"] != artifact_identity:
        raise ValueError("source reward result and artifact differ")
    selection = _read_json(Path(fisher_selection))
    if (
        selection.get("config_sha256") != extension["source_config_sha256"]
        or selection.get("selected_relative_damping") != source_reward["selected_relative_damping"]
        or sha256_file(Path(fisher_selection)) != source_reward["fisher_selection_sha256"]
    ):
        raise ValueError("frozen Fisher selection differs from source reward result")
    annotation_metadata, annotation = validate_h_annotations(
        config_path, artifact_dir, sidecar_dir, seed=seed
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    experiment = _experiment_to_device(
        load_exact_delta_artifact(
            artifact_dir,
            expected_config_hash=extension["source_config_sha256"],
            expected_seed=seed,
        ),
        target_device,
    )
    design, oracle_margins, coordinates = _edge_coordinates(experiment.train)
    h = annotation["H"].to(target_device)
    frequencies = (annotation["S"].to(torch.float64) / annotation["N"]).to(target_device)
    training = extension["reward_training"]
    oracle_mse = _fit_mse(
        "oracle_mse",
        design,
        coordinates,
        oracle_margins,
        tolerance=float(training["mse_relative_residual_tolerance"]),
    )
    h_mle = _fit_h_mle(
        design,
        coordinates,
        frequencies,
        MLETrainingConfig(
            max_iterations=int(training["h_mle_max_iterations"]),
            history_size=int(training["h_mle_history_size"]),
            gradient_tolerance=float(training["h_mle_gradient_tolerance"]),
            change_tolerance=1.0e-12,
            microbatch_size=int(training["microbatch_size"]),
        ),
    )
    h_mse = _fit_mse(
        "h_mse",
        design,
        coordinates,
        h,
        tolerance=float(training["mse_relative_residual_tolerance"]),
    )
    geometry = source["geometry"]
    pro_raw = source["reward_model"]["pro"]
    pro_config = ProTrainingConfig(
        relative_damping=float(source_reward["selected_relative_damping"]),
        fisher_estimator=geometry["fisher_estimator"],
        inner_max_iterations=int(geometry["cg_max_iterations"]),
        inner_tolerance=float(geometry["cg_tolerance"]),
        outer_max_iterations=int(pro_raw["max_iterations"]),
        outer_tolerance=float(pro_raw["tolerance"]),
        residual_recompute_interval=int(pro_raw["residual_recompute_interval"]),
    )
    h_target_moment = h_policy_moment(experiment.train, h)
    h_pro = _fit_h_pro(experiment.train, h_target_moment, pro_config)
    fits = {
        "oracle_mse": oracle_mse,
        "h_mle": h_mle,
        "h_mse": h_mse,
        "h_pro": h_pro,
    }
    failed = [name for name, fit in fits.items() if not fit.converged]
    if failed:
        raise RuntimeError(f"new reward fits failed convergence: {failed}")
    old_weights = {
        "oracle_mle": torch.tensor(
            source_reward["methods"]["MLE-RM"]["head_weight"],
            dtype=torch.float64,
            device=target_device,
        ),
        "oracle_pro": torch.tensor(
            source_reward["methods"]["Pro-RM"]["head_weight"],
            dtype=torch.float64,
            device=target_device,
        ),
    }
    weights = {**old_weights, **{name: fit.weight for name, fit in fits.items()}}
    beta = float(extension["policy_update"]["beta"])
    metrics = {
        name: _reward_metrics(
            experiment.train,
            experiment.test,
            weight,
            beta=beta,
            relative_damping=float(source_reward["selected_relative_damping"]),
            geometry=geometry,
        )
        for name, weight in weights.items()
    }
    settings = GeometrySettings(
        fisher_estimator=geometry["fisher_estimator"],
        relative_damping=float(source_reward["selected_relative_damping"]),
        cg_tolerance=float(geometry["cg_tolerance"]),
        cg_max_iterations=int(geometry["cg_max_iterations"]),
        residual_recompute_interval=int(geometry["residual_recompute_interval"]),
    )
    directions = {
        name: solve_natural_direction(
            experiment.train,
            experiment.train.reward_features.to(torch.float64) @ fit.weight,
            settings,
        )
        for name, fit in fits.items()
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "artifact_metadata_sha256": artifact_identity,
        "source_reward_result_sha256": sha256_file(Path(source_reward_result)),
        "fisher_selection_sha256": sha256_file(Path(fisher_selection)),
        "annotation_metadata_sha256": sha256_file(Path(sidecar_dir) / "metadata.json"),
        "annotation_tensors_sha256": annotation_metadata["tensors_sha256"],
        "seed": seed,
        "beta": beta,
        "gamma": 0.9,
        "selected_relative_damping": float(source_reward["selected_relative_damping"]),
        "methods": {name: fit.to_dict() for name, fit in fits.items()},
        "reused_methods": {
            "oracle_mle": {
                "source_name": "MLE-RM",
                "head_sha256": source_reward["methods"]["MLE-RM"]["head_sha256"],
            },
            "oracle_pro": {
                "source_name": "Pro-RM",
                "head_sha256": source_reward["methods"]["Pro-RM"]["head_sha256"],
            },
        },
        "reward_evaluation": metrics,
        "policy_directions": {
            name: direction.detach().cpu().tolist() for name, direction in directions.items()
        },
        "direction_solve": {
            "split": "train",
            "fisher": "source_selected_damped_train_fisher",
            "beta_free": True,
        },
        "annotation_summary": annotation_metadata["summary"],
        "test_usage": "evaluation_only_no_selection",
        "producer": producer_identity(),
    }
    _atomic_json(target, payload)
    return payload


def aggregate_h_reward_results(
    config_path: str | os.PathLike[str],
    result_paths: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_h_ablation_config(config_path)
    expected_seeds = list(extension["experiment"]["seeds"])
    records_with_paths = [(_read_json(Path(path)), Path(path)) for path in result_paths]
    records_with_paths.sort(key=lambda item: expected_seeds.index(item[0].get("seed")))
    records = [record for record, _ in records_with_paths]
    if [record.get("seed") for record in records] != expected_seeds:
        raise ValueError("reward aggregate requires every declared seed exactly once")
    for record in records:
        if (
            record.get("schema") != RESULT_SCHEMA
            or record.get("protocol") != PROTOCOL
            or record.get("config_sha256") != h_config_hash(extension)
            or set(record.get("reward_evaluation", {})) != set(LEARNED_METHODS)
        ):
            raise ValueError("H/MSE reward result identity mismatch")
    summary: dict[str, Any] = {}
    for method in LEARNED_METHODS:
        summary[method] = {}
        for metric in ("NLL", "MSE", "approximate_regret"):
            values = [float(record["reward_evaluation"][method][metric]) for record in records]
            summary[method][metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
                "seed_values": values,
            }
    payload = {
        "schema": "prorm-h-mse-reward-aggregate/v1",
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "seeds": expected_seeds,
        "methods": summary,
        "input_sha256": {
            str(record["seed"]): sha256_file(path) for record, path in records_with_paths
        },
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


__all__ = [
    "ANNOTATION_RNG_NAMESPACE",
    "LEARNED_METHODS",
    "NEW_METHODS",
    "PROTOCOL",
    "aggregate_h_reward_results",
    "h_config_hash",
    "h_policy_moment",
    "load_h_ablation_config",
    "materialize_h_annotations",
    "randomized_logit_estimate",
    "resolve_source_config",
    "run_h_ablation_rewards",
    "sample_h_annotation",
    "sample_positive_geometric",
    "validate_h_annotations",
]

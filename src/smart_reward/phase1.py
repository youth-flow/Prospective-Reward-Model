"""Phase-1 on-policy feature materialization and leakage-safe assembly.

The tensor-only :func:`assemble_controlled_experiment` is the authoritative
join boundary.  Prompt order defines the first tensor axis, all four candidate
nodes are retained for the Fisher geometry, and only candidate ``0 - 1`` is an
annotated edge.  Phase 1 fits oracle calibration exclusively on training
nodes; Phase 2 may instead supply a design-bound global calibration learned
from permanently excluded Phase-1 training artifacts.

The Hugging Face entry point is intentionally lazy and fail closed.  It loads
only immutable revisions, defaults to the local cache, generates and scores
with one in-memory zero-B policy, then releases that policy before loading the
oracle.  The resulting artifact contains no training true-reward tensor.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .annotations import sample_geometric_repeated_labels
from .config import config_hash, validate_config
from .contracts import compatibility_value
from .data import (
    DEFAULT_CONTINUATION_PROBABILITY,
    CandidateNode,
    EvaluationEdgeRecord,
    TrainingEdgeRecord,
    repeated_labels_to_h,
    save_jsonl,
)
from .experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    TrainingTensorData,
)
from .hf import (
    assert_noop_logits,
    configure_fixed_a_lora,
    generate_exact_candidates,
    pool_final_response_hidden_state,
    score_exact_candidates,
    score_oracle_chats,
)
from .oracle import (
    RobustOracleTransform,
    btl_probabilities,
    fit_robust_oracle_transform,
)
from .prompts import (
    PromptRecord,
    build_policy_token_eligible_prompt_pool,
    load_multipref_parquet_snapshot,
    save_prompt_jsonl,
)
from .scores import per_sample_scores
from .seeding import SeedBundle, derive_seed

_SPLITS = ("train", "validation", "test")
_NUM_CANDIDATES = 4
_ASSEMBLY_SCHEMA = "phase1-assembly/v1"
_MATERIALIZATION_SCHEMA = "phase1-materialization/v1"
_POLICY_PROMPT_SEMANTICS_SCHEMA = "full-policy-prompt-semantics/v1"
_FROZEN_ORACLE_CALIBRATION_KEYS = frozenset(
    {
        "mode",
        "b",
        "tau",
        "aggregation_rule",
        "source_split",
        "source_config",
        "source_config_hash",
        "source_seeds_excluded_from_phase2",
        "source_artifacts",
    }
)
_FIXED_LORA_A_KEYS = frozenset(
    {
        "mode",
        "initialization_seed",
        "expected_sha256",
        "source_seed",
        "source_named_stream",
        "source_config",
        "source_config_hash",
        "source_artifact_metadata_sha256",
        "source_seed_excluded_from_phase2",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Phase1Assembly:
    """Complete tensor/edge result of the pure Phase-1 join."""

    experiment: ControlledFeatureExperiment
    training_edges: tuple[TrainingEdgeRecord, ...]
    validation_edges: tuple[EvaluationEdgeRecord, ...]
    test_edges: tuple[EvaluationEdgeRecord, ...]
    oracle_transform: RobustOracleTransform
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ControlledFeatureExperiment):
            raise TypeError("experiment must be ControlledFeatureExperiment")
        if not all(isinstance(edge, TrainingEdgeRecord) for edge in self.training_edges):
            raise TypeError("training_edges must contain TrainingEdgeRecord objects")
        for name, edges in (
            ("validation_edges", self.validation_edges),
            ("test_edges", self.test_edges),
        ):
            if not all(isinstance(edge, EvaluationEdgeRecord) for edge in edges):
                raise TypeError(f"{name} must contain EvaluationEdgeRecord objects")
        if not isinstance(self.oracle_transform, RobustOracleTransform):
            raise TypeError("oracle_transform must be RobustOracleTransform")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("evidence must be a mapping")


@dataclass(frozen=True)
class Phase1Materialization:
    """Materialized candidate graph plus its integrity-audited tensor assembly."""

    assembly: Phase1Assembly
    candidates: tuple[CandidateNode, ...]
    artifact_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, Phase1Assembly):
            raise TypeError("assembly must be Phase1Assembly")
        if not self.candidates or not all(
            isinstance(candidate, CandidateNode) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a non-empty CandidateNode tuple")
        if (
            len(self.candidates)
            != (
                self.assembly.experiment.train.num_prompts
                + self.assembly.experiment.validation.num_prompts
                + self.assembly.experiment.test.num_prompts
            )
            * _NUM_CANDIDATES
        ):
            raise ValueError("candidate count does not match the assembled prompt graph")
        if not isinstance(self.artifact_directory, Path):
            raise TypeError("artifact_directory must be a pathlib.Path")


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return seed


def _candidate_id(prompt_id: str, index: int) -> str:
    return f"{prompt_id}::candidate::{index}"


def _edge_id(prompt_id: str) -> str:
    return f"{prompt_id}::edge::0-1"


def _validate_assembler_inputs(
    prompt_records: Sequence[PromptRecord],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    raw_oracle_scores: torch.Tensor,
) -> tuple[tuple[PromptRecord, ...], int, int, int]:
    if isinstance(prompt_records, (str, bytes)) or not isinstance(prompt_records, Sequence):
        raise TypeError("prompt_records must be a sequence of PromptRecord objects")
    records = tuple(prompt_records)
    if not records or not all(isinstance(record, PromptRecord) for record in records):
        raise TypeError("prompt_records must contain at least one PromptRecord")
    prompt_ids = [record.prompt_id for record in records]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt_records must have unique prompt_id values")
    split_counts = {split: 0 for split in _SPLITS}
    for record in records:
        split_counts[record.split] += 1
    empty = [split for split, count in split_counts.items() if count == 0]
    if empty:
        raise ValueError(f"every split must be non-empty; empty={empty!r}")

    for name, tensor in (
        ("policy_scores", policy_scores),
        ("reward_features", reward_features),
        ("raw_oracle_scores", raw_oracle_scores),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if tensor.requires_grad:
            raise ValueError(f"{name} must be detached")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")

    if policy_scores.ndim != 3:
        raise ValueError("policy_scores must have shape (P, 4, D)")
    num_prompts, num_candidates, policy_dimension = policy_scores.shape
    if num_prompts != len(records):
        raise ValueError(
            "prompt_records order is the tensor join key: len(prompt_records) must equal P"
        )
    if num_candidates != _NUM_CANDIDATES:
        raise ValueError("the controlled Phase-1 graph requires exactly four candidates")
    if policy_dimension < 1:
        raise ValueError("policy score dimension must be positive")
    if reward_features.ndim != 3 or reward_features.shape[:2] != (
        num_prompts,
        _NUM_CANDIDATES,
    ):
        raise ValueError("reward_features must have shape (P, 4, H)")
    if reward_features.shape[2] < 1:
        raise ValueError("reward feature dimension must be positive")
    if raw_oracle_scores.shape != (num_prompts, _NUM_CANDIDATES):
        raise ValueError("raw_oracle_scores must have shape (P, 4)")
    for tensor in (reward_features, raw_oracle_scores):
        if tensor.dtype != policy_scores.dtype or tensor.device != policy_scores.device:
            raise ValueError("all assembler tensors must share dtype and device")
    return records, num_prompts, policy_dimension, reward_features.shape[2]


def _local_annotation_batch(
    probabilities: torch.Tensor,
    *,
    seed: int,
    gamma: float,
) -> tuple[list[tuple[int, ...]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample on CPU with a private generator, independent of global RNG state."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch = sample_geometric_repeated_labels(
        probabilities.detach().to(device="cpu", dtype=torch.float64),
        gamma=gamma,
        generator=generator,
    )
    labels_by_edge: list[tuple[int, ...]] = []
    offset = 0
    for count in batch.counts.reshape(-1).tolist():
        labels = tuple(int(value) for value in batch.labels[offset : offset + count].tolist())
        labels_by_edge.append(labels)
        offset += count
    if offset != batch.labels.numel():
        raise RuntimeError("internal repeated-label offset mismatch")
    wins = batch.wins.reshape(-1)
    totals = batch.counts.reshape(-1)
    h = torch.tensor(
        [repeated_labels_to_h(labels, gamma) for labels in labels_by_edge],
        dtype=torch.float64,
    )
    return labels_by_edge, wins, totals, h


def _reward_class_projection_diagnostic(
    reward_features: torch.Tensor,
    true_rewards: torch.Tensor,
) -> dict[str, float | str]:
    """Measure train-only misspecification modulo prompt reward gauges.

    A bias-free frozen-feature linear head can only represent rewards through
    ``features @ weight``.  Policy updates and BTL comparisons are unchanged by
    adding a prompt-specific constant to every candidate reward, so both the
    features and targets are centered over candidates before projection.  The
    solve is deliberately performed on a detached float64 CPU copy: this keeps
    the diagnostic independent of accelerator kernels and prevents oracle
    gradients or fitted weights from crossing the artifact boundary.
    """

    if reward_features.ndim != 3:
        raise ValueError("reward_features must have shape (P, M, H)")
    if true_rewards.shape != reward_features.shape[:2]:
        raise ValueError("true_rewards must have shape (P, M)")
    if not reward_features.is_floating_point() or not true_rewards.is_floating_point():
        raise TypeError("projection inputs must have floating-point dtypes")
    if not bool(torch.isfinite(reward_features).all()) or not bool(
        torch.isfinite(true_rewards).all()
    ):
        raise ValueError("projection inputs must be finite")

    features = reward_features.detach().to(device="cpu", dtype=torch.float64)
    targets = true_rewards.detach().to(device="cpu", dtype=torch.float64)
    centered_features = features - features.mean(dim=1, keepdim=True)
    centered_targets = targets - targets.mean(dim=1, keepdim=True)
    design = centered_features.reshape(-1, centered_features.shape[-1])
    target = centered_targets.reshape(-1)

    fitted_weight = torch.linalg.lstsq(design, target).solution
    residual = design @ fitted_weight - target
    normalizer = math.sqrt(target.numel())
    target_centered_rms = torch.linalg.vector_norm(target).item() / normalizer
    residual_rmse = torch.linalg.vector_norm(residual).item() / normalizer
    relative_residual = 0.0 if target_centered_rms == 0.0 else residual_rmse / target_centered_rms
    metrics = (target_centered_rms, residual_rmse, relative_residual)
    if not all(math.isfinite(value) and value >= 0.0 for value in metrics):
        raise FloatingPointError("reward-class projection diagnostic is non-finite")

    return {
        "fit_split": "train",
        "centering": "per_prompt_candidate_mean",
        "solver": "float64_cpu_lstsq",
        "target_centered_rms": target_centered_rms,
        "residual_rmse": residual_rmse,
        "relative_residual": relative_residual,
    }


def _normalize_frozen_oracle_calibration(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and detach the global transform provenance used by Phase 2."""

    if not isinstance(value, Mapping) or set(value) != _FROZEN_ORACLE_CALIBRATION_KEYS:
        raise ValueError("oracle transform calibration must contain the exact frozen-global schema")
    if value["mode"] != "frozen_global":
        raise ValueError("oracle transform calibration mode must equal 'frozen_global'")
    if value["aggregation_rule"] != "componentwise_median":
        raise ValueError(
            "oracle transform calibration aggregation_rule must equal 'componentwise_median'"
        )
    if value["source_split"] != "train":
        raise ValueError("oracle transform calibration source_split must equal 'train'")
    if value["source_seeds_excluded_from_phase2"] is not True:
        raise ValueError("oracle transform calibration source seeds must be excluded from Phase 2")
    for name in ("source_config", "source_config_hash"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"oracle transform calibration {name} must be non-empty")
    source_config_hash = str(value["source_config_hash"])
    if len(source_config_hash) != 64 or any(
        character not in _LOWER_HEX for character in source_config_hash
    ):
        raise ValueError("oracle transform calibration source_config_hash must be SHA-256")

    # RobustOracleTransform owns the finite-value and positive-scale checks.
    transform = RobustOracleTransform(b=value["b"], tau=value["tau"])
    raw_artifacts = value["source_artifacts"]
    if isinstance(raw_artifacts, (str, bytes, bytearray)) or not isinstance(
        raw_artifacts,
        Sequence,
    ):
        raise TypeError("oracle transform calibration source_artifacts must be a sequence")
    if not raw_artifacts:
        raise ValueError("oracle transform calibration source_artifacts must not be empty")
    artifacts: list[dict[str, object]] = []
    seen_seeds: set[int] = set()
    seen_hashes: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, Mapping) or set(raw_artifact) != {
            "seed",
            "metadata_sha256",
        }:
            raise ValueError(
                "oracle transform calibration source artifact must contain exactly "
                "seed and metadata_sha256"
            )
        seed = raw_artifact["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**63 - 1:
            raise ValueError(
                f"oracle transform calibration source_artifacts[{index}].seed is invalid"
            )
        digest = raw_artifact["metadata_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _LOWER_HEX for character in digest)
        ):
            raise ValueError(
                "oracle transform calibration source artifact metadata_sha256 is invalid"
            )
        if seed in seen_seeds or digest in seen_hashes:
            raise ValueError("oracle transform calibration source artifacts must be unique")
        seen_seeds.add(seed)
        seen_hashes.add(digest)
        artifacts.append({"seed": seed, "metadata_sha256": digest})

    return {
        "mode": "frozen_global",
        "b": transform.b,
        "tau": transform.tau,
        "aggregation_rule": "componentwise_median",
        "source_split": "train",
        "source_config": str(value["source_config"]),
        "source_config_hash": source_config_hash,
        "source_seeds_excluded_from_phase2": True,
        "source_artifacts": artifacts,
    }


def _policy_lora_a_initialization(
    policy_config: Mapping[str, object],
    *,
    run_seed: int,
) -> dict[str, object]:
    """Resolve the effective LoRA-A seed without changing Phase-1 semantics."""

    validated_seed = _validate_seed(run_seed)
    current_seed_derived = SeedBundle.from_base_seed(validated_seed).policy_lora_a
    fixed = policy_config.get("fixed_lora_a")
    if fixed is None:
        return {
            "mode": "current_seed_derived",
            "initialization_seed": current_seed_derived,
            "expected_sha256": None,
            "current_run_seed": validated_seed,
            "current_seed_derived_initialization_seed": current_seed_derived,
            "current_seed_derivation_used": True,
        }
    if not isinstance(fixed, Mapping) or set(fixed) != _FIXED_LORA_A_KEYS:
        raise ValueError("policy.fixed_lora_a must contain the exact frozen-global schema")
    if fixed["mode"] != "frozen_global":
        raise ValueError("policy.fixed_lora_a.mode must equal 'frozen_global'")
    initialization_seed = fixed["initialization_seed"]
    source_seed = fixed["source_seed"]
    if (
        isinstance(initialization_seed, bool)
        or not isinstance(initialization_seed, int)
        or initialization_seed < 0
        or initialization_seed > 2**63 - 1
    ):
        raise ValueError("policy.fixed_lora_a.initialization_seed is invalid")
    if (
        isinstance(source_seed, bool)
        or not isinstance(source_seed, int)
        or source_seed < 0
        or source_seed > 2**63 - 1
    ):
        raise ValueError("policy.fixed_lora_a.source_seed is invalid")
    if SeedBundle.from_base_seed(source_seed).policy_lora_a != initialization_seed:
        raise ValueError(
            "policy.fixed_lora_a.initialization_seed does not match its source named seed"
        )
    for name in (
        "expected_sha256",
        "source_config_hash",
        "source_artifact_metadata_sha256",
    ):
        digest = fixed[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _LOWER_HEX for character in digest)
        ):
            raise ValueError(f"policy.fixed_lora_a.{name} must be a lowercase SHA-256 digest")
    if fixed["source_named_stream"] != "policy_lora_a":
        raise ValueError("policy.fixed_lora_a.source_named_stream must equal 'policy_lora_a'")
    if not isinstance(fixed["source_config"], str) or not fixed["source_config"]:
        raise ValueError("policy.fixed_lora_a.source_config must be non-empty")
    if fixed["source_seed_excluded_from_phase2"] is not True:
        raise ValueError("policy.fixed_lora_a source seed must be excluded from Phase 2")

    return {
        **dict(fixed),
        "current_run_seed": validated_seed,
        "current_seed_derived_initialization_seed": current_seed_derived,
        "current_seed_derivation_used": False,
    }


def _policy_lora_a_initialization_evidence(
    policy_config: Mapping[str, object],
    *,
    run_seed: int,
    observed_sha256: str,
) -> dict[str, object]:
    """Bind the resolved seed/provenance to the observed LoRA-A tensor bytes."""

    if (
        not isinstance(observed_sha256, str)
        or len(observed_sha256) != 64
        or any(character not in _LOWER_HEX for character in observed_sha256)
    ):
        raise ValueError("observed LoRA-A fingerprint must be a lowercase SHA-256 digest")
    initialization = _policy_lora_a_initialization(
        policy_config,
        run_seed=run_seed,
    )
    expected = initialization["expected_sha256"]
    if expected is not None and observed_sha256 != expected:
        raise RuntimeError(
            "configured global LoRA-A fingerprint does not match the initialized policy"
        )
    return {
        **initialization,
        "observed_sha256": observed_sha256,
    }


def assemble_controlled_experiment(
    prompt_records: Sequence[PromptRecord],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    raw_oracle_scores: torch.Tensor,
    *,
    seed: int,
    gamma: float = DEFAULT_CONTINUATION_PROBABILITY,
    oracle_transform_calibration: Mapping[str, object] | None = None,
) -> Phase1Assembly:
    """Join ordered node tensors into the locked controlled experiment.

    ``prompt_records[i]`` is the explicit join key for tensor row ``i``.  The
    robust oracle transform is fit on all and only the ``train`` rows unless a
    validated ``frozen_global`` calibration is supplied.  In that Phase-2
    mode, the current seed's oracle scores never select ``b`` or ``tau``.  Raw
    or transformed rewards are never supplied to :class:`TrainingTensorData`.
    A private named CPU generator samples iid BTL labels and does not mutate
    Python, CPU-Torch, or CUDA global RNG state.
    """

    validated_seed = _validate_seed(seed)
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value != DEFAULT_CONTINUATION_PROBABILITY:
        raise ValueError(
            "gamma must equal the locked TrainingEdgeRecord value "
            f"{DEFAULT_CONTINUATION_PROBABILITY}"
        )
    records, _, _, _ = _validate_assembler_inputs(
        prompt_records,
        policy_scores,
        reward_features,
        raw_oracle_scores,
    )

    split_indices = {
        split: [index for index, record in enumerate(records) if record.split == split]
        for split in _SPLITS
    }
    train_index = torch.tensor(
        split_indices["train"], dtype=torch.int64, device=raw_oracle_scores.device
    )
    if oracle_transform_calibration is None:
        transform = fit_robust_oracle_transform(
            raw_oracle_scores.index_select(0, train_index).reshape(-1)
        )
        transform_calibration: dict[str, object] = {
            "mode": "current_seed_train_fit",
            "b": transform.b,
            "tau": transform.tau,
            "aggregation_rule": "sample_median_and_scaled_mad",
            "source_split": "train",
            "source_seed": validated_seed,
        }
        oracle_fit_split = "train"
        fitted_on_current_seed = True
    else:
        transform_calibration = _normalize_frozen_oracle_calibration(oracle_transform_calibration)
        transform = RobustOracleTransform(
            b=transform_calibration["b"],
            tau=transform_calibration["tau"],
        )
        oracle_fit_split = "excluded_phase1_train_artifacts"
        fitted_on_current_seed = False
    true_rewards = transform(raw_oracle_scores)
    train_reward_class_projection = _reward_class_projection_diagnostic(
        reward_features.index_select(0, train_index),
        true_rewards.index_select(0, train_index),
    )
    margins = true_rewards[:, 0] - true_rewards[:, 1]
    probabilities = btl_probabilities(margins)

    seeds = SeedBundle.from_base_seed(validated_seed)
    labels_by_edge: list[tuple[int, ...]] = [tuple() for _ in records]
    all_wins_cpu = torch.empty(len(records), dtype=torch.int64)
    all_totals_cpu = torch.empty(len(records), dtype=torch.int64)
    all_h_cpu = torch.empty(len(records), dtype=torch.float64)
    annotation_split_seeds: dict[str, int] = {}
    for split in _SPLITS:
        indices = split_indices[split]
        index = torch.tensor(indices, dtype=torch.int64, device=probabilities.device)
        split_seed = derive_seed(seeds.annotations, f"annotations:{split}")
        annotation_split_seeds[split] = split_seed
        split_labels, split_wins, split_totals, split_h = _local_annotation_batch(
            probabilities.index_select(0, index),
            seed=split_seed,
            gamma=gamma_value,
        )
        for local_index, global_index in enumerate(indices):
            labels_by_edge[global_index] = split_labels[local_index]
            all_wins_cpu[global_index] = split_wins[local_index]
            all_totals_cpu[global_index] = split_totals[local_index]
            all_h_cpu[global_index] = split_h[local_index]

    training_edges: list[TrainingEdgeRecord] = []
    validation_edges: list[EvaluationEdgeRecord] = []
    test_edges: list[EvaluationEdgeRecord] = []
    for index, record in enumerate(records):
        common: dict[str, Any] = {
            "edge_id": _edge_id(record.prompt_id),
            "prompt_id": record.prompt_id,
            "left_id": _candidate_id(record.prompt_id, 0),
            "right_id": _candidate_id(record.prompt_id, 1),
            "raw_labels": labels_by_edge[index],
            "num_annotations": int(all_totals_cpu[index].item()),
            "left_wins": int(all_wins_cpu[index].item()),
            "h": float(all_h_cpu[index].item()),
        }
        if record.split == "train":
            training_edges.append(TrainingEdgeRecord(**common))
        else:
            edge = EvaluationEdgeRecord(
                **common,
                true_margin=float(margins[index].item()),
            )
            if record.split == "validation":
                validation_edges.append(edge)
            else:
                test_edges.append(edge)

    def frozen_index(split: str) -> torch.Tensor:
        return torch.tensor(split_indices[split], dtype=torch.int64, device=policy_scores.device)

    train_rows = frozen_index("train")
    validation_rows = frozen_index("validation")
    test_rows = frozen_index("test")
    train_edge_positions = [
        index for index, record in enumerate(records) if record.split == "train"
    ]
    train_edge_index_cpu = torch.tensor(train_edge_positions, dtype=torch.int64)

    # Clone every slice: the immutable data object cannot be changed by later
    # mutation of a caller-owned materialization buffer.
    train = TrainingTensorData(
        prompt_ids=tuple(record.prompt_id for record in records if record.split == "train"),
        policy_scores=policy_scores.index_select(0, train_rows).detach().clone(),
        reward_features=reward_features.index_select(0, train_rows).detach().clone(),
        h=all_h_cpu.index_select(0, train_edge_index_cpu)
        .to(device=policy_scores.device, dtype=policy_scores.dtype)
        .clone(),
        left_wins=all_wins_cpu.index_select(0, train_edge_index_cpu)
        .to(device=policy_scores.device)
        .clone(),
        num_annotations=all_totals_cpu.index_select(0, train_edge_index_cpu)
        .to(device=policy_scores.device)
        .clone(),
    )

    def evaluation(split: str, rows: torch.Tensor) -> EvaluationTensorData:
        return EvaluationTensorData(
            prompt_ids=tuple(record.prompt_id for record in records if record.split == split),
            policy_scores=policy_scores.index_select(0, rows).detach().clone(),
            reward_features=reward_features.index_select(0, rows).detach().clone(),
            true_rewards=true_rewards.index_select(0, rows).detach().clone(),
        )

    experiment = ControlledFeatureExperiment(
        train=train,
        validation=evaluation("validation", validation_rows),
        test=evaluation("test", test_rows),
    )
    evidence: dict[str, Any] = {
        "schema": _ASSEMBLY_SCHEMA,
        "seed": validated_seed,
        "named_seeds": seeds.to_dict(),
        "annotation_split_seeds": annotation_split_seeds,
        "gamma": gamma_value,
        "num_candidates": _NUM_CANDIDATES,
        "split_sizes": {split: len(split_indices[split]) for split in _SPLITS},
        "oracle_transform": {"b": transform.b, "tau": transform.tau},
        "oracle_transform_calibration": transform_calibration,
        "oracle_fit_split": oracle_fit_split,
        "oracle_transform_fitted_on_current_seed": fitted_on_current_seed,
        "train_reward_class_projection": train_reward_class_projection,
        "edge_orientation": {"left_candidate": 0, "right_candidate": 1},
    }
    return Phase1Assembly(
        experiment=experiment,
        training_edges=tuple(training_edges),
        validation_edges=tuple(validation_edges),
        test_edges=tuple(test_edges),
        oracle_transform=transform,
        evidence=evidence,
    )


def _require_module(name: str, *, extra: str = "llm") -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            f"Phase-1 materialization requires optional dependency {name!r}; "
            f"install smart-reward-model[{extra}]"
        ) from error


def _local_snapshot_error(
    kind: str,
    identifier: str,
    revision: str,
) -> RuntimeError:
    return RuntimeError(
        f"pinned {kind} snapshot is unavailable from the local cache: "
        f"{identifier}@{revision}. Pre-stage this exact revision with the HPC4 "
        "Slurm staging job; "
        "formal compute-node runs must not download from the network."
    )


def _model_inputs(encoded: object, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(encoded, torch.Tensor):
        inputs = {"input_ids": encoded}
    elif isinstance(encoded, Mapping):
        inputs = {name: value for name, value in encoded.items() if isinstance(value, torch.Tensor)}
    else:
        raise TypeError("apply_chat_template must return a tensor or tensor mapping")
    if "input_ids" not in inputs:
        raise ValueError("chat template output must contain input_ids")
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return {name: value.to(device) for name, value in inputs.items()}


def _prompt_text(record: PromptRecord) -> str:
    if len(record.messages) != 1 or record.messages[0].role != "user":
        raise ValueError(
            "controlled MultiPref prompts must contain exactly one user message and no system role"
        )
    return record.messages[0].content


def _prompt_text_sha256(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _prompt_token_ids_sha256(token_ids: Sequence[int]) -> str:
    values = tuple(token_ids)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
    ):
        raise ValueError("prompt token IDs must be a non-empty sequence of non-negative integers")
    encoded = json.dumps(
        list(values),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_semantics_records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        list(records),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encode_full_policy_prompt(
    tokenizer: object,
    prompt: PromptRecord,
    *,
    max_prompt_tokens: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Render one complete policy prompt and fail before any silent truncation.

    The raw :class:`PromptRecord` remains the semantic source of truth.  The
    Qwen policy tokenizer may render it with its own chat template, but it is
    never allowed to shorten the rendered token sequence.  The returned
    evidence binds the raw user text to the exact token prefix consumed by the
    policy.
    """

    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        raise ValueError("max_prompt_tokens must be a positive integer")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    prompt_text = _prompt_text(prompt)
    encoded = tokenizer.apply_chat_template(
        [message.to_dict() for message in prompt.messages],
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = _model_inputs(encoded, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("one policy prompt must encode to a single token row")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("policy prompt attention mask must match input_ids")
    if not bool((attention_mask == 1).all()):
        raise ValueError("single-prompt policy encoding must not contain padding")
    token_count = int(input_ids.shape[1])
    if token_count > max_prompt_tokens:
        raise ValueError(
            "policy prompt exceeds the frozen max_prompt_tokens and truncation is forbidden: "
            f"prompt_id={prompt.prompt_id!r}, token_count={token_count}, "
            f"max_prompt_tokens={max_prompt_tokens}"
        )
    token_ids = tuple(int(value) for value in input_ids[0].detach().cpu().tolist())
    evidence = {
        "prompt_id": prompt.prompt_id,
        "raw_prompt_sha256": _prompt_text_sha256(prompt_text),
        "policy_chat_token_count": token_count,
        "policy_prompt_token_ids_sha256": _prompt_token_ids_sha256(token_ids),
        "max_prompt_tokens": max_prompt_tokens,
        "truncated": False,
        "raw_prompt_preserved": True,
    }
    return inputs, evidence


@contextmanager
def _fork_torch_seed(seed: int, device: torch.device) -> Iterator[None]:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        cuda_devices = [index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed(seed)
        yield


def _jsonl_sha256(records: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = record.to_dict()
        line = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _producer_identity_from_environment() -> dict[str, str]:
    """Read only formal-run producer digests; never snapshot the environment."""

    result: dict[str, str] = {}
    formal_environment_names = (
        "SLURM_JOB_ID",
        "PRORM_GIT_COMMIT",
        "SRM_GIT_COMMIT",
        "PRORM_IMAGE_SHA256",
        "SRM_IMAGE_SHA256",
        "PRORM_HF_INVENTORY_SHA256",
        "SRM_HF_INVENTORY_SHA256",
    )
    for canonical_name, legacy_name, evidence_name, lengths in (
        ("PRORM_GIT_COMMIT", "SRM_GIT_COMMIT", "git_commit", {40, 64}),
        ("PRORM_IMAGE_SHA256", "SRM_IMAGE_SHA256", "image_sha256", {64}),
        (
            "PRORM_HF_INVENTORY_SHA256",
            "SRM_HF_INVENTORY_SHA256",
            "hf_inventory_sha256",
            {64},
        ),
    ):
        value = compatibility_value(os.environ, canonical_name, legacy_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"{canonical_name} must be a string")
        if len(value) not in lengths or any(character not in _LOWER_HEX for character in value):
            raise ValueError(f"{canonical_name} must be a lowercase hexadecimal producer digest")
        result[evidence_name] = value
    formal_requested = any(bool(os.environ.get(name)) for name in formal_environment_names)
    expected_fields = {"git_commit", "image_sha256", "hf_inventory_sha256"}
    if formal_requested and set(result) != expected_fields:
        raise ValueError("formal artifact production requires Git, image, and HF inventory digests")
    return result


def _decode_response(tokenizer: object, token_ids: torch.Tensor) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("policy tokenizer must expose callable decode")
    return str(decode(token_ids.tolist(), skip_special_tokens=True))


def _load_prompts(
    datasets: Any,
    config: Mapping[str, Any],
    *,
    policy_tokenizer: object,
    split_seed: int,
    local_files_only: bool,
) -> tuple[list[PromptRecord], dict[str, object]]:
    data_config = config["data"]
    run_config = config["run"]
    try:
        if local_files_only:
            huggingface_hub = _require_module("huggingface_hub")
            snapshot = Path(
                huggingface_hub.snapshot_download(
                    repo_id=data_config["prompt_dataset"],
                    repo_type="dataset",
                    revision=data_config["prompt_revision"],
                    cache_dir=os.environ.get("HF_HUB_CACHE"),
                    local_files_only=True,
                    token=False,
                )
            )
            if snapshot.name != data_config["prompt_revision"]:
                raise RuntimeError("local dataset snapshot resolved to an unexpected revision")
            rows = load_multipref_parquet_snapshot(
                datasets,
                snapshot,
                datasets_cache=os.environ.get("HF_DATASETS_CACHE"),
            )
        else:
            rows = datasets.load_dataset(
                data_config["prompt_dataset"],
                revision=data_config["prompt_revision"],
                split="train",
            )
    except (OSError, FileNotFoundError, ConnectionError, RuntimeError) as error:
        if local_files_only:
            raise _local_snapshot_error(
                "dataset",
                data_config["prompt_dataset"],
                data_config["prompt_revision"],
            ) from error
        raise
    prompt_pool = build_policy_token_eligible_prompt_pool(
        rows,
        policy_tokenizer=policy_tokenizer,
        max_prompt_tokens=int(config["policy"]["max_prompt_tokens"]),
        eligibility_rule=data_config.get("prompt_eligibility"),
    )
    return prompt_pool.select(
        split_sizes=run_config["split_sizes"],
        seed=split_seed,
    )


def _load_pretrained(
    factory: Any,
    identifier: str,
    revision: str,
    *,
    local_files_only: bool,
    kind: str,
    **kwargs: Any,
) -> Any:
    try:
        return factory.from_pretrained(
            identifier,
            revision=revision,
            local_files_only=local_files_only,
            **kwargs,
        )
    except (OSError, FileNotFoundError, ConnectionError) as error:
        if local_files_only:
            raise _local_snapshot_error(kind, identifier, revision) from error
        raise


def _preflight_artifact_target(destination: Path) -> None:
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"artifact target is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {destination}")


def materialize_phase1(
    config: Mapping[str, object],
    *,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> Phase1Materialization:
    """Materialize the pinned four-candidate Phase-1 experiment.

    This operation never overwrites an existing file.  ``local_files_only`` is
    true by default so a formal HPC run cannot silently change its inputs or
    spend allocation time downloading a new snapshot.
    """

    validated_seed = _validate_seed(seed)
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be bool")
    destination = Path(artifact_dir)
    _preflight_artifact_target(destination)
    normalized = validate_config(config)
    configured_seeds = (
        [normalized["run"]["seed"]] if "seed" in normalized["run"] else normalized["run"]["seeds"]
    )
    if validated_seed not in configured_seeds:
        raise ValueError("seed must be one of the explicitly configured run seeds")
    if normalized["data"]["num_candidates"] != _NUM_CANDIDATES:
        raise ValueError("Phase-1 materialization requires data.num_candidates == 4")
    if any(
        normalized[section]["dtype"] != "float32"
        for section in ("policy", "reward_model", "oracle")
    ):
        raise ValueError("Phase-1 policy, reward features, and oracle must all use float32")
    if (
        normalized["reward_model"]["model"] != normalized["policy"]["model"]
        or normalized["reward_model"]["revision"] != normalized["policy"]["revision"]
    ):
        raise ValueError(
            "reward_model model/revision must match policy exactly because frozen reward "
            "features are extracted from the zero-B policy forward"
        )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    # Dependency checks are lazy and occur only after cheap configuration and
    # destination guards have succeeded.
    datasets = _require_module("datasets")
    transformers = _require_module("transformers")
    peft = _require_module("peft")
    _require_module("safetensors")
    from .artifacts import save_controlled_feature_artifact

    policy_config = normalized["policy"]
    lora_a_initialization = _policy_lora_a_initialization(
        policy_config,
        run_seed=validated_seed,
    )
    lora_a_seed = int(lora_a_initialization["initialization_seed"])
    tokenizer = _load_pretrained(
        transformers.AutoTokenizer,
        policy_config["model"],
        policy_config["revision"],
        local_files_only=local_files_only,
        kind="policy tokenizer",
        use_fast=True,
    )
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("policy tokenizer must provide a non-empty chat_template")
    tokenizer.truncation_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    seeds = SeedBundle.from_base_seed(validated_seed)
    prompts, prompt_pool_selection = _load_prompts(
        datasets,
        normalized,
        policy_tokenizer=tokenizer,
        split_seed=seeds.prompt_split,
        local_files_only=local_files_only,
    )

    with _fork_torch_seed(lora_a_seed, target_device):
        policy_model = _load_pretrained(
            transformers.AutoModelForCausalLM,
            policy_config["model"],
            policy_config["revision"],
            local_files_only=local_files_only,
            kind="policy model",
            torch_dtype=torch.float32,
        )
    policy_model.to(target_device)
    policy_model.eval()

    probe_inputs, _ = _encode_full_policy_prompt(
        tokenizer,
        prompts[0],
        max_prompt_tokens=int(policy_config["max_prompt_tokens"]),
        device=target_device,
    )
    with torch.inference_mode():
        reference_logits = policy_model(**probe_inputs, use_cache=False).logits.detach().clone()

    lora_config = peft.LoraConfig(
        r=policy_config["lora_rank"],
        lora_alpha=policy_config["lora_alpha"],
        lora_dropout=policy_config["lora_dropout"],
        target_modules=list(policy_config["lora_modules"]),
        layers_to_transform=list(policy_config["lora_layers"]),
        bias="none",
        init_lora_weights=True,
        task_type="CAUSAL_LM",
    )
    with _fork_torch_seed(lora_a_seed, target_device):
        setup = configure_fixed_a_lora(policy_model, lora_config)
    a_state_sha256 = setup.a_state_sha256
    lora_a_evidence = _policy_lora_a_initialization_evidence(
        policy_config,
        run_seed=validated_seed,
        observed_sha256=a_state_sha256,
    )
    policy_model = setup.model
    policy_model.eval()
    with torch.inference_mode():
        adapted_logits = policy_model(**probe_inputs, use_cache=False).logits
    zero_b_error = assert_noop_logits(reference_logits, adapted_logits)
    del reference_logits, adapted_logits

    candidate_nodes: list[CandidateNode] = []
    prompt_semantics_records: list[dict[str, object]] = []
    policy_score_rows: list[torch.Tensor] = []
    reward_feature_rows: list[torch.Tensor] = []
    generation_kwargs = {
        **dict(policy_config["sampling"]),
        "num_return_sequences": _NUM_CANDIDATES,
        "max_new_tokens": policy_config["max_response_tokens"],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    with _fork_torch_seed(seeds.candidate_generation, target_device):
        for prompt in prompts:
            prompt_inputs, prompt_semantics = _encode_full_policy_prompt(
                tokenizer,
                prompt,
                max_prompt_tokens=int(policy_config["max_prompt_tokens"]),
                device=target_device,
            )
            prompt_semantics_records.append(prompt_semantics)
            candidates = generate_exact_candidates(
                policy_model,
                prompt_inputs["input_ids"],
                prompt_attention_mask=prompt_inputs["attention_mask"],
                generation_kwargs=generation_kwargs,
            )
            log_probabilities = score_exact_candidates(policy_model, candidates)
            scores = per_sample_scores(
                log_probabilities,
                setup.named_tangent_parameters(),
                layout=setup.layout,
            )
            if scores.shape[0] != _NUM_CANDIDATES:
                raise RuntimeError("policy did not return exactly four candidate scores")
            policy_score_rows.append(scores.to(device="cpu", dtype=torch.float32))

            with torch.inference_mode():
                hidden_output = policy_model(
                    input_ids=candidates.input_ids,
                    attention_mask=candidates.attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                features = pool_final_response_hidden_state(
                    hidden_output.hidden_states,
                    candidates.response_mask,
                )
            reward_feature_rows.append(features.detach().to(device="cpu", dtype=torch.float32))

            prompt_text = _prompt_text(prompt)
            for candidate_index in range(_NUM_CANDIDATES):
                active_response_ids = candidates.input_ids[candidate_index][
                    candidates.response_mask[candidate_index].bool()
                ]
                candidate_nodes.append(
                    CandidateNode(
                        prompt_id=prompt.prompt_id,
                        candidate_id=_candidate_id(prompt.prompt_id, candidate_index),
                        prompt=prompt_text,
                        response=_decode_response(tokenizer, active_response_ids),
                        token_ids=tuple(
                            int(value) for value in candidates.input_ids[candidate_index].tolist()
                        ),
                        response_mask=tuple(
                            int(value)
                            for value in candidates.response_mask[candidate_index].tolist()
                        ),
                        terminated_by_eos=bool(
                            candidates.terminated_by_eos[candidate_index].item()
                        ),
                        reached_max_length=bool(
                            candidates.reached_max_length[candidate_index].item()
                        ),
                    )
                )
            del log_probabilities, scores, hidden_output, features, candidates

    policy_scores = torch.stack(policy_score_rows, dim=0)
    reward_features = torch.stack(reward_feature_rows, dim=0)
    layout_metadata = setup.layout.to_metadata()
    del setup, policy_model, policy_score_rows, reward_feature_rows, probe_inputs
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    oracle_config = normalized["oracle"]
    oracle_tokenizer = _load_pretrained(
        transformers.AutoTokenizer,
        oracle_config["model"],
        oracle_config["revision"],
        local_files_only=local_files_only,
        kind="oracle tokenizer",
        use_fast=True,
    )
    oracle_chat_template = getattr(oracle_tokenizer, "chat_template", None)
    if oracle_chat_template in (None, ""):
        raise ValueError("oracle tokenizer must provide a non-empty chat_template")
    oracle_chat_template_sha256 = hashlib.sha256(
        str(oracle_chat_template).encode("utf-8")
    ).hexdigest()
    oracle_model = _load_pretrained(
        transformers.AutoModelForSequenceClassification,
        oracle_config["model"],
        oracle_config["revision"],
        local_files_only=local_files_only,
        kind="oracle model",
        torch_dtype=torch.float32,
        num_labels=1,
    )
    oracle_model.to(target_device)
    oracle_model.eval()
    flat_prompts = [candidate.prompt for candidate in candidate_nodes]
    flat_responses = [candidate.response for candidate in candidate_nodes]
    oracle_batch_size = min(16, int(normalized["reward_model"]["microbatch_size"]))
    raw_batches: list[torch.Tensor] = []
    for start in range(0, len(candidate_nodes), oracle_batch_size):
        stop = min(start + oracle_batch_size, len(candidate_nodes))
        raw_batches.append(
            score_oracle_chats(
                oracle_model,
                oracle_tokenizer,
                flat_prompts[start:stop],
                flat_responses[start:stop],
                device=target_device,
            ).to(device="cpu", dtype=torch.float32)
        )
    raw_oracle_scores = torch.cat(raw_batches).reshape(len(prompts), _NUM_CANDIDATES)
    del oracle_model, oracle_tokenizer, raw_batches
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    assembly = assemble_controlled_experiment(
        prompts,
        policy_scores,
        reward_features,
        raw_oracle_scores,
        seed=validated_seed,
        gamma=float(normalized["annotations"]["gamma"]),
        oracle_transform_calibration=normalized["oracle"].get("transform_calibration"),
    )

    evaluation_edges = (*assembly.validation_edges, *assembly.test_edges)
    json_hashes = {
        "prompts.jsonl": _jsonl_sha256(prompts),
        "candidates.jsonl": _jsonl_sha256(candidate_nodes),
        "training_edges.jsonl": _jsonl_sha256(assembly.training_edges),
        "evaluation_edges.jsonl": _jsonl_sha256(evaluation_edges),
    }
    full_evidence = {
        **dict(assembly.evidence),
        "schema": _MATERIALIZATION_SCHEMA,
        "config_sha256": config_hash(normalized),
        "policy_a_sha256": a_state_sha256,
        "policy_lora_a_initialization": lora_a_evidence,
        "policy_layout": layout_metadata,
        "policy_zero_b_max_absolute_error": zero_b_error,
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "oracle_chat_template_sha256": oracle_chat_template_sha256,
        "prompt_pool_selection": prompt_pool_selection,
        "policy_prompt_semantics": {
            "schema_version": _POLICY_PROMPT_SEMANTICS_SCHEMA,
            "encoding": "policy_tokenizer_apply_chat_template",
            "add_generation_prompt": True,
            "truncation": False,
            "fail_closed_above_max_prompt_tokens": True,
            "max_prompt_tokens": int(policy_config["max_prompt_tokens"]),
            "num_prompts": len(prompt_semantics_records),
            "records_sha256": _prompt_semantics_records_sha256(prompt_semantics_records),
            "records": prompt_semantics_records,
        },
        "jsonl_sha256": json_hashes,
        "revisions": {
            "prompt_dataset": normalized["data"]["prompt_revision"],
            "policy_model": policy_config["revision"],
            "reward_feature_model": normalized["reward_model"]["revision"],
            "oracle_model": oracle_config["revision"],
        },
        "local_files_only": local_files_only,
        "producer": _producer_identity_from_environment(),
    }
    assembly = replace(assembly, evidence=full_evidence)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.phase1-", dir=destination.parent))
    try:
        save_controlled_feature_artifact(
            assembly.experiment,
            staging,
            config_hash=config_hash(normalized),
            seed=validated_seed,
            evidence=full_evidence,
            overwrite=False,
        )
        save_prompt_jsonl(staging / "prompts.jsonl", prompts)
        save_jsonl(staging / "candidates.jsonl", candidate_nodes)
        save_jsonl(staging / "training_edges.jsonl", assembly.training_edges)
        save_jsonl(staging / "evaluation_edges.jsonl", evaluation_edges)
        for filename, expected_digest in json_hashes.items():
            if _sha256_file(staging / filename) != expected_digest:
                raise RuntimeError(f"serialized digest mismatch for {filename}")
        if destination.exists():
            # Preflight proved this exact target directory was empty.
            destination.rmdir()
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return Phase1Materialization(
        assembly=assembly,
        candidates=tuple(candidate_nodes),
        artifact_directory=destination,
    )


__all__ = [
    "Phase1Assembly",
    "Phase1Materialization",
    "assemble_controlled_experiment",
    "materialize_phase1",
]

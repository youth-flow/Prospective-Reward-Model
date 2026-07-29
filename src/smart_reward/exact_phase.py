"""Real-model materialization for the frozen exact-delta main experiment."""

from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import save_exact_delta_artifact
from .config import PROTOCOL, config_hash, validate_config
from .data import NODE_SCHEMA, CandidateNode, save_jsonl
from .exact import ExactDeltaExperiment, ExactSplitData, pair_indices
from .hf import (
    assert_noop_logits,
    configure_fixed_a_lora,
    generate_exact_candidates,
    pool_final_response_hidden_state,
    score_exact_candidates,
    score_oracle_chats,
)
from .oracle import AffineOracleTransform, fit_affine_oracle_transform
from .prompts import PromptRecord, save_prompt_jsonl
from .runtime import (
    candidate_id,
    decode_response,
    fork_torch_seed,
    jsonl_sha256,
    load_pretrained,
    load_prompts,
    model_inputs,
    preflight_empty_directory,
    producer_identity,
    prompt_text,
    require_module,
    reward_class_projection,
    sha256_file,
    validate_seed,
)
from .scores import per_sample_scores
from .seeding import SeedBundle

_SPLITS = ("train", "validation", "test")
_ASSEMBLY_SCHEMA = "exact-delta-assembly/v1"
_MATERIALIZATION_SCHEMA = "exact-delta-materialization/v1"


@dataclass(frozen=True, slots=True)
class ExactEdgeRecord:
    """One deterministic ``j < k`` edge carrying the exact oracle margin."""

    edge_id: str
    prompt_id: str
    left_id: str
    right_id: str
    left_candidate_index: int
    right_candidate_index: int
    delta_r_star: float
    split: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.edge_id, self.prompt_id, self.left_id, self.right_id)
        ):
            raise ValueError("edge identifiers must be non-empty strings")
        if self.split not in _SPLITS:
            raise ValueError("edge split is invalid")
        if not (
            isinstance(self.left_candidate_index, int)
            and isinstance(self.right_candidate_index, int)
            and 0 <= self.left_candidate_index < self.right_candidate_index
        ):
            raise ValueError("edge orientation must satisfy 0 <= left < right")
        if not math.isfinite(float(self.delta_r_star)):
            raise ValueError("delta_r_star must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "prompt_id": self.prompt_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_candidate_index": self.left_candidate_index,
            "right_candidate_index": self.right_candidate_index,
            "delta_r_star": self.delta_r_star,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class ExactDeltaAssembly:
    experiment: ExactDeltaExperiment
    edges: tuple[ExactEdgeRecord, ...]
    oracle_transform: AffineOracleTransform
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExactDeltaMaterialization:
    assembly: ExactDeltaAssembly
    candidates: tuple[CandidateNode, ...]
    artifact_directory: Path


def _validate_inputs(
    prompt_records: Sequence[PromptRecord],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    raw_oracle_scores: torch.Tensor,
) -> tuple[tuple[PromptRecord, ...], int, int]:
    records = tuple(prompt_records)
    if not records or not all(isinstance(record, PromptRecord) for record in records):
        raise TypeError("prompt_records must contain PromptRecord objects")
    if len({record.prompt_id for record in records}) != len(records):
        raise ValueError("prompt IDs must be unique")
    if {record.split for record in records} != set(_SPLITS):
        raise ValueError("prompt records must contain train, validation, and test")
    for name, tensor in (
        ("policy_scores", policy_scores),
        ("reward_features", reward_features),
        ("raw_oracle_scores", raw_oracle_scores),
    ):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be detached and finite")
    if policy_scores.ndim != 3 or policy_scores.shape[0] != len(records):
        raise ValueError("policy_scores must have shape (P, M, D)")
    prompts, candidates, policy_dimension = policy_scores.shape
    if candidates < 2 or policy_dimension < 1:
        raise ValueError("policy_scores has an invalid candidate or policy dimension")
    if reward_features.ndim != 3 or reward_features.shape[:2] != (prompts, candidates):
        raise ValueError("reward_features must have shape (P, M, H)")
    if reward_features.shape[2] < 1:
        raise ValueError("reward feature dimension must be positive")
    if raw_oracle_scores.shape != (prompts, candidates):
        raise ValueError("raw_oracle_scores must have shape (P, M)")
    for tensor in (reward_features, raw_oracle_scores):
        if tensor.dtype != policy_scores.dtype or tensor.device != policy_scores.device:
            raise ValueError("all assembly tensors must share dtype and device")
    return records, candidates, policy_dimension


def assemble_exact_delta_experiment(
    prompt_records: Sequence[PromptRecord],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    raw_oracle_scores: torch.Tensor,
    *,
    oracle_scale_floor: float = 1.0e-6,
) -> ExactDeltaAssembly:
    """Apply train-only affine calibration and build every exact pair edge."""

    records, num_candidates, policy_dimension = _validate_inputs(
        prompt_records,
        policy_scores,
        reward_features,
        raw_oracle_scores,
    )
    split_indices = {
        split: [index for index, record in enumerate(records) if record.split == split]
        for split in _SPLITS
    }
    train_rows = torch.tensor(
        split_indices["train"],
        dtype=torch.int64,
        device=raw_oracle_scores.device,
    )
    transform = fit_affine_oracle_transform(
        raw_oracle_scores.index_select(0, train_rows).reshape(-1),
        scale_floor=oracle_scale_floor,
    )
    true_rewards = transform(raw_oracle_scores)

    def build_split(split: str) -> ExactSplitData:
        rows = torch.tensor(
            split_indices[split],
            dtype=torch.int64,
            device=policy_scores.device,
        )
        return ExactSplitData(
            prompt_ids=tuple(record.prompt_id for record in records if record.split == split),
            policy_scores=policy_scores.index_select(0, rows).detach().clone(),
            reward_features=reward_features.index_select(0, rows).detach().clone(),
            true_rewards=true_rewards.index_select(0, rows).detach().clone(),
        )

    experiment = ExactDeltaExperiment(
        train=build_split("train"),
        validation=build_split("validation"),
        test=build_split("test"),
    )
    pairs = pair_indices(num_candidates, device=true_rewards.device)
    edges: list[ExactEdgeRecord] = []
    for prompt_index, record in enumerate(records):
        for left, right in pairs.tolist():
            edges.append(
                ExactEdgeRecord(
                    edge_id=f"{record.prompt_id}::edge::{left}-{right}",
                    prompt_id=record.prompt_id,
                    left_id=candidate_id(record.prompt_id, left),
                    right_id=candidate_id(record.prompt_id, right),
                    left_candidate_index=left,
                    right_candidate_index=right,
                    delta_r_star=float(
                        (
                            true_rewards[prompt_index, left] - true_rewards[prompt_index, right]
                        ).item()
                    ),
                    split=record.split,
                )
            )
    projection = reward_class_projection(
        experiment.train.reward_features,
        experiment.train.true_rewards,
    )
    evidence = {
        "schema": _ASSEMBLY_SCHEMA,
        "protocol": PROTOCOL,
        "num_candidates": num_candidates,
        "edges_per_prompt": num_candidates * (num_candidates - 1) // 2,
        "edge_orientation": "lower_candidate_index_first",
        "edge_sampling_unit": "prompt_clustered_u_statistic",
        "policy_dimension": policy_dimension,
        "reward_dimension": experiment.train.reward_dimension,
        "oracle_transform": {
            "kind": "train_median_scaled_mad_affine",
            "b": transform.b,
            "tau": transform.tau,
            "fit_split": "train",
        },
        "train_reward_class_projection": projection,
        "split_sizes": {split: len(split_indices[split]) for split in _SPLITS},
    }
    return ExactDeltaAssembly(
        experiment=experiment,
        edges=tuple(edges),
        oracle_transform=transform,
        evidence=evidence,
    )


def _prompt_fits(tokenizer: Any, text: str, *, maximum: int) -> bool:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
    )
    if not isinstance(encoded, Mapping) or not isinstance(encoded.get("input_ids"), torch.Tensor):
        raise TypeError("policy chat template must return an input_ids tensor")
    return int(encoded["input_ids"].shape[-1]) <= maximum


def materialize_exact_delta(
    config: Mapping[str, object],
    *,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> ExactDeltaMaterialization:
    """Generate the configured candidates and persist the exact-delta node artifact."""

    validated_seed = validate_seed(seed)
    normalized = validate_config(config)
    if normalized.get("protocol") != PROTOCOL:
        raise ValueError(f"materialization requires protocol {PROTOCOL}")
    if validated_seed not in normalized["run"]["seeds"]:
        raise ValueError("seed must be one of the configured experiment seeds")
    destination = Path(artifact_dir)
    preflight_empty_directory(destination)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    datasets = require_module("datasets")
    transformers = require_module("transformers")
    peft = require_module("peft")
    require_module("safetensors")
    policy_config = normalized["policy"]
    tokenizer = load_pretrained(
        transformers.AutoTokenizer,
        policy_config["model"],
        policy_config["revision"],
        local_files_only=local_files_only,
        kind="policy tokenizer",
        use_fast=True,
    )
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("policy tokenizer must provide a non-empty chat_template")
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_limit = int(policy_config["max_prompt_tokens"])
    prompts = load_prompts(
        datasets,
        normalized,
        local_files_only=local_files_only,
        text_filter=lambda text: _prompt_fits(tokenizer, text, maximum=prompt_limit),
    )

    seeds = SeedBundle.from_base_seed(validated_seed)
    with fork_torch_seed(seeds.policy_lora_a, target_device):
        policy_model = load_pretrained(
            transformers.AutoModelForCausalLM,
            policy_config["model"],
            policy_config["revision"],
            local_files_only=local_files_only,
            kind="policy model",
            torch_dtype=torch.float32,
        )
    policy_model.to(target_device).eval()
    first_encoded = tokenizer.apply_chat_template(
        [message.to_dict() for message in prompts[0].messages],
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
    )
    probe_inputs = model_inputs(first_encoded, target_device)
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
    with fork_torch_seed(seeds.policy_lora_a, target_device):
        setup = configure_fixed_a_lora(policy_model, lora_config)
    policy_model = setup.model
    policy_model.eval()
    with torch.inference_mode():
        adapted_logits = policy_model(**probe_inputs, use_cache=False).logits
    zero_b_error = assert_noop_logits(reference_logits, adapted_logits)
    del reference_logits, adapted_logits

    num_candidates = int(normalized["data"]["num_candidates"])
    generation_kwargs = {
        **dict(policy_config["sampling"]),
        "num_return_sequences": num_candidates,
        "max_new_tokens": policy_config["max_response_tokens"],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    candidate_payloads: list[dict[str, Any]] = []
    policy_score_rows: list[torch.Tensor] = []
    reward_feature_rows: list[torch.Tensor] = []
    with fork_torch_seed(seeds.candidate_generation, target_device):
        for prompt in prompts:
            encoded = tokenizer.apply_chat_template(
                [message.to_dict() for message in prompt.messages],
                tokenize=True,
                add_generation_prompt=True,
                truncation=False,
                return_tensors="pt",
                return_dict=True,
            )
            prompt_inputs = model_inputs(encoded, target_device)
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
            if scores.shape[0] != num_candidates:
                raise RuntimeError("policy returned an unexpected candidate count")
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
            text = prompt_text(prompt)
            for candidate_index in range(num_candidates):
                active_response_ids = candidates.input_ids[candidate_index][
                    candidates.response_mask[candidate_index].bool()
                ]
                candidate_payloads.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "candidate_id": candidate_id(prompt.prompt_id, candidate_index),
                        "candidate_index": candidate_index,
                        "split": prompt.split,
                        "prompt": text,
                        "response": decode_response(tokenizer, active_response_ids),
                        "token_ids": tuple(
                            int(value) for value in candidates.input_ids[candidate_index].tolist()
                        ),
                        "response_mask": tuple(
                            int(value)
                            for value in candidates.response_mask[candidate_index].tolist()
                        ),
                        "terminated_by_eos": bool(
                            candidates.terminated_by_eos[candidate_index].item()
                        ),
                        "reached_max_length": bool(
                            candidates.reached_max_length[candidate_index].item()
                        ),
                    }
                )
            del log_probabilities, scores, hidden_output, features, candidates

    policy_scores = torch.stack(policy_score_rows, dim=0)
    reward_features = torch.stack(reward_feature_rows, dim=0)
    layout_metadata = setup.layout.to_metadata()
    a_state_sha256 = setup.a_state_sha256
    del setup, policy_model, policy_score_rows, reward_feature_rows, probe_inputs
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    oracle_config = normalized["oracle"]
    oracle_tokenizer = load_pretrained(
        transformers.AutoTokenizer,
        oracle_config["model"],
        oracle_config["revision"],
        local_files_only=local_files_only,
        kind="oracle tokenizer",
        use_fast=True,
    )
    oracle_template = getattr(oracle_tokenizer, "chat_template", None)
    if oracle_template in (None, ""):
        raise ValueError("oracle tokenizer must provide a non-empty chat_template")
    oracle_model = load_pretrained(
        transformers.AutoModelForSequenceClassification,
        oracle_config["model"],
        oracle_config["revision"],
        local_files_only=local_files_only,
        kind="oracle model",
        torch_dtype=torch.float32,
        num_labels=1,
    )
    oracle_model.to(target_device).eval()
    flat_prompts = [str(candidate["prompt"]) for candidate in candidate_payloads]
    flat_responses = [str(candidate["response"]) for candidate in candidate_payloads]
    raw_batches: list[torch.Tensor] = []
    oracle_batch_size = int(oracle_config["batch_size"])
    for start in range(0, len(candidate_payloads), oracle_batch_size):
        stop = min(start + oracle_batch_size, len(candidate_payloads))
        raw_batches.append(
            score_oracle_chats(
                oracle_model,
                oracle_tokenizer,
                flat_prompts[start:stop],
                flat_responses[start:stop],
                device=target_device,
            ).to(device="cpu", dtype=torch.float32)
        )
    raw_oracle_scores = torch.cat(raw_batches).reshape(len(prompts), num_candidates)
    del oracle_model, oracle_tokenizer, raw_batches
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    assembly = assemble_exact_delta_experiment(
        prompts,
        policy_scores,
        reward_features,
        raw_oracle_scores,
        oracle_scale_floor=float(oracle_config["robust_scale_floor"]),
    )
    standardized_rewards = assembly.oracle_transform(raw_oracle_scores)
    candidate_nodes = [
        CandidateNode(
            **payload,
            raw_oracle_score=float(raw_score.item()),
            oracle_reward=float(reward.item()),
        )
        for payload, raw_score, reward in zip(
            candidate_payloads,
            raw_oracle_scores.reshape(-1),
            standardized_rewards.reshape(-1),
            strict=True,
        )
    ]
    json_hashes = {
        "prompts.jsonl": jsonl_sha256(prompts),
        "candidates.jsonl": jsonl_sha256(candidate_nodes),
        "edges.jsonl": jsonl_sha256(assembly.edges),
    }
    full_evidence = {
        **dict(assembly.evidence),
        "schema": _MATERIALIZATION_SCHEMA,
        "config_sha256": config_hash(normalized),
        "seed": validated_seed,
        "prompt_split_seed": int(normalized["run"]["prompt_split_seed"]),
        "prompts_shared_across_seeds": True,
        "prompt_limit": prompt_limit,
        "prompt_overlength": "exclude_before_sampling",
        "candidate_node_schema": NODE_SCHEMA,
        "policy_a_sha256": a_state_sha256,
        "policy_layout": layout_metadata,
        "policy_zero_b_max_absolute_error": zero_b_error,
        "policy_chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "oracle_chat_template_sha256": hashlib.sha256(
            str(oracle_template).encode("utf-8")
        ).hexdigest(),
        "jsonl_sha256": json_hashes,
        "revisions": {
            "prompt_dataset": normalized["data"]["prompt_revision"],
            "policy_model": policy_config["revision"],
            "reward_feature_model": normalized["reward_model"]["revision"],
            "oracle_model": oracle_config["revision"],
        },
        "local_files_only": local_files_only,
        "producer": producer_identity(),
    }
    assembly = ExactDeltaAssembly(
        experiment=assembly.experiment,
        edges=assembly.edges,
        oracle_transform=assembly.oracle_transform,
        evidence=full_evidence,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.exact-", dir=destination.parent))
    try:
        save_exact_delta_artifact(
            assembly.experiment,
            staging,
            config_hash=config_hash(normalized),
            seed=validated_seed,
            evidence=full_evidence,
            overwrite=False,
        )
        save_prompt_jsonl(staging / "prompts.jsonl", prompts)
        save_jsonl(staging / "candidates.jsonl", candidate_nodes)
        save_jsonl(staging / "edges.jsonl", assembly.edges)
        for filename, expected_digest in json_hashes.items():
            if sha256_file(staging / filename) != expected_digest:
                raise RuntimeError(f"serialized digest mismatch for {filename}")
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return ExactDeltaMaterialization(
        assembly=assembly,
        candidates=tuple(candidate_nodes),
        artifact_directory=destination,
    )


__all__ = [
    "ExactDeltaAssembly",
    "ExactDeltaMaterialization",
    "ExactEdgeRecord",
    "assemble_exact_delta_experiment",
    "materialize_exact_delta",
]

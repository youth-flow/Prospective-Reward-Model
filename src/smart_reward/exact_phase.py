"""Real-model materialization for the frozen exact-delta main experiment."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import load_exact_delta_artifact, save_exact_delta_artifact
from .config import PROTOCOL, TRPO_PROTOCOL, config_hash, validate_config
from .data import NODE_SCHEMA, CandidateNode, load_jsonl, save_jsonl
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
from .prompts import PromptRecord, load_prompt_jsonl, save_prompt_jsonl
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
from .seeding import SeedBundle, derive_seed

_SPLITS = ("train", "validation", "test")
_ASSEMBLY_SCHEMA = "exact-delta-assembly/v1"
_MATERIALIZATION_SCHEMA = "exact-delta-materialization/v1"
_WORK_SCHEMA = "exact-delta-materialization-work/v1"
_SHARD_SCHEMA = "exact-delta-candidate-shard/v1"


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _save_candidate_shard(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    start: int,
    stop: int,
    prompt_ids: Sequence[str],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    payloads: Sequence[Mapping[str, Any]],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite materialization shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        tensors_path = staging / "tensors.safetensors"
        require_module("safetensors.torch").save_file(
            {
                "policy_scores": policy_scores.contiguous(),
                "reward_features": reward_features.contiguous(),
            },
            str(tensors_path),
        )
        payload_path = staging / "payloads.json"
        _atomic_json(payload_path, {"rows": list(payloads)})
        _atomic_json(
            staging / "metadata.json",
            {
                "schema": _SHARD_SCHEMA,
                "manifest": dict(manifest),
                "start": start,
                "stop": stop,
                "prompt_ids": list(prompt_ids),
                "tensors_sha256": sha256_file(tensors_path),
                "payloads_sha256": sha256_file(payload_path),
            },
        )
        os.replace(staging, path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_candidate_shard(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    start: int,
    stop: int,
    prompt_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    metadata = _read_json(path / "metadata.json")
    expected_header = {
        "schema": _SHARD_SCHEMA,
        "manifest": dict(manifest),
        "start": start,
        "stop": stop,
        "prompt_ids": list(prompt_ids),
    }
    for key, expected in expected_header.items():
        if metadata.get(key) != expected:
            raise ValueError(f"materialization shard identity mismatch: {path}")
    tensors_path = path / "tensors.safetensors"
    payload_path = path / "payloads.json"
    if metadata.get("tensors_sha256") != sha256_file(tensors_path):
        raise ValueError(f"materialization shard tensor digest mismatch: {path}")
    if metadata.get("payloads_sha256") != sha256_file(payload_path):
        raise ValueError(f"materialization shard payload digest mismatch: {path}")
    tensors = require_module("safetensors.torch").load_file(str(tensors_path), device="cpu")
    if set(tensors) != {"policy_scores", "reward_features"}:
        raise ValueError(f"materialization shard tensor keys mismatch: {path}")
    payload_value = _read_json(payload_path).get("rows")
    if not isinstance(payload_value, list) or not all(
        isinstance(row, dict) for row in payload_value
    ):
        raise ValueError(f"materialization shard payload is malformed: {path}")
    return tensors["policy_scores"], tensors["reward_features"], payload_value


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
    protocol: str = PROTOCOL,
) -> ExactDeltaAssembly:
    """Apply train-only affine calibration and build every exact pair edge."""

    if protocol not in {PROTOCOL, TRPO_PROTOCOL}:
        raise ValueError("assembly protocol is unsupported")
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
        "protocol": protocol,
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
    reuse_splits_from: str | os.PathLike[str] | None = None,
) -> ExactDeltaMaterialization:
    """Generate the configured candidates and persist the exact-delta node artifact."""

    validated_seed = validate_seed(seed)
    normalized = validate_config(config)
    if normalized.get("protocol") not in {PROTOCOL, TRPO_PROTOCOL}:
        raise ValueError("materialization requires a supported exact-delta protocol")
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
    tokenizer.padding_side = "left"
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
    layout_metadata = setup.layout.to_metadata()
    a_state_sha256 = setup.a_state_sha256
    reused_prompt_count = 0
    reused_policy_scores: torch.Tensor | None = None
    reused_reward_features: torch.Tensor | None = None
    reused_raw_oracle_scores: torch.Tensor | None = None
    reused_payloads: list[dict[str, Any]] = []
    split_reuse_evidence: dict[str, Any] | None = None
    if reuse_splits_from is not None:
        if normalized["protocol"] != TRPO_PROTOCOL:
            raise ValueError("split-level materialization reuse is only valid for Fisher-TRPO")
        source_root = Path(reuse_splits_from)
        source_experiment = load_exact_delta_artifact(source_root)
        source_metadata_path = source_root / "metadata.json"
        source_metadata = _read_json(source_metadata_path)
        source_evidence = source_metadata.get("evidence")
        if not isinstance(source_evidence, dict):
            raise ValueError("source materialization evidence is missing")
        if (
            source_evidence.get("policy_a_sha256") != a_state_sha256
            or source_evidence.get("policy_layout") != layout_metadata
            or source_evidence.get("revisions", {}).get("policy_model") != policy_config["revision"]
            or source_evidence.get("revisions", {}).get("oracle_model")
            != normalized["oracle"]["revision"]
        ):
            raise ValueError("source materialization model geometry does not match")
        reusable_splits = ("train", "validation")
        source_prompts = [
            record
            for record in load_prompt_jsonl(source_root / "prompts.jsonl")
            if record.split in reusable_splits
        ]
        reused_prompt_count = sum(
            int(normalized["run"]["split_sizes"][name]) for name in reusable_splits
        )
        if len(source_prompts) != reused_prompt_count:
            raise ValueError("source train/validation prompt inventory has the wrong size")
        if [record.to_dict() for record in source_prompts] != [
            record.to_dict() for record in prompts[:reused_prompt_count]
        ]:
            raise ValueError("source train/validation prompts differ from the frozen selection")
        reused_policy_scores = torch.cat(
            (source_experiment.train.policy_scores, source_experiment.validation.policy_scores),
            dim=0,
        )
        reused_reward_features = torch.cat(
            (
                source_experiment.train.reward_features,
                source_experiment.validation.reward_features,
            ),
            dim=0,
        )
        source_candidates = [
            record
            for record in load_jsonl(source_root / "candidates.jsonl", CandidateNode)
            if record.split in reusable_splits
        ]
        if len(source_candidates) != reused_prompt_count * num_candidates:
            raise ValueError("source train/validation candidate inventory has the wrong size")
        expected_candidate_ids = [
            candidate_id(prompt.prompt_id, index)
            for prompt in prompts[:reused_prompt_count]
            for index in range(num_candidates)
        ]
        if [record.candidate_id for record in source_candidates] != expected_candidate_ids:
            raise ValueError("source train/validation candidate ordering differs")
        reused_raw_oracle_scores = torch.tensor(
            [record.raw_oracle_score for record in source_candidates],
            dtype=torch.float32,
        ).reshape(reused_prompt_count, num_candidates)
        reused_payloads = [
            {
                key: value
                for key, value in record.to_dict().items()
                if key not in {"raw_oracle_score", "oracle_reward", "schema_version"}
            }
            for record in source_candidates
        ]
        split_reuse_evidence = {
            "schema": "prorm-split-component-reuse/v1",
            "splits": list(reusable_splits),
            "prompt_count": reused_prompt_count,
            "source_artifact_metadata_sha256": sha256_file(source_metadata_path),
            "source_artifact_tensors_sha256": sha256_file(source_root / "tensors.safetensors"),
            "source_prompts_sha256": sha256_file(source_root / "prompts.jsonl"),
            "source_candidates_sha256": sha256_file(source_root / "candidates.jsonl"),
            "source_producer": source_evidence.get("producer"),
        }
        source_receipt = source_root.parent / "stage_receipts" / "materialize.json"
        if source_receipt.is_file():
            split_reuse_evidence["source_materialize_receipt_sha256"] = sha256_file(source_receipt)
    execution = normalized["execution"]
    prompt_batch_size = int(execution["materialization_prompt_batch_size"])
    checkpoint_prompts = int(execution["materialization_checkpoint_prompts"])
    work = destination.parent / f".{destination.name}.materialize-work"
    work_manifest = {
        "schema": _WORK_SCHEMA,
        "config_sha256": config_hash(normalized),
        "seed": validated_seed,
        "producer": producer_identity(),
        "policy_a_sha256": a_state_sha256,
        "policy_layout": layout_metadata,
        "num_prompts": len(prompts),
        "num_candidates": num_candidates,
        "prompt_batch_size": prompt_batch_size,
        "checkpoint_prompts": checkpoint_prompts,
        "split_reuse": split_reuse_evidence,
    }
    work_manifest_path = work / "manifest.json"
    if work_manifest_path.exists():
        if _read_json(work_manifest_path) != work_manifest:
            raise ValueError(f"materialization work identity mismatch: {work}")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(f"unidentified materialization work directory: {work}")
        _atomic_json(work_manifest_path, work_manifest)

    candidate_payloads: list[dict[str, Any]] = []
    policy_score_chunks: list[torch.Tensor] = []
    reward_feature_chunks: list[torch.Tensor] = []
    for checkpoint_start in range(0, len(prompts), checkpoint_prompts):
        checkpoint_stop = min(checkpoint_start + checkpoint_prompts, len(prompts))
        checkpoint_records = prompts[checkpoint_start:checkpoint_stop]
        shard = work / "shards" / f"{checkpoint_start:06d}-{checkpoint_stop:06d}"
        if shard.exists():
            shard_scores, shard_features, shard_payloads = _load_candidate_shard(
                shard,
                manifest=work_manifest,
                start=checkpoint_start,
                stop=checkpoint_stop,
                prompt_ids=[record.prompt_id for record in checkpoint_records],
            )
            print(
                f"materialize candidates={checkpoint_stop}/{len(prompts)} status=reused",
                flush=True,
            )
        else:
            if checkpoint_stop <= reused_prompt_count:
                assert reused_policy_scores is not None
                assert reused_reward_features is not None
                candidate_start = checkpoint_start * num_candidates
                candidate_stop = checkpoint_stop * num_candidates
                shard_scores = reused_policy_scores[checkpoint_start:checkpoint_stop]
                shard_features = reused_reward_features[checkpoint_start:checkpoint_stop]
                shard_payloads = reused_payloads[candidate_start:candidate_stop]
                _save_candidate_shard(
                    shard,
                    manifest=work_manifest,
                    start=checkpoint_start,
                    stop=checkpoint_stop,
                    prompt_ids=[record.prompt_id for record in checkpoint_records],
                    policy_scores=shard_scores,
                    reward_features=shard_features,
                    payloads=shard_payloads,
                )
                print(
                    f"materialize candidates={checkpoint_stop}/{len(prompts)} status=imported",
                    flush=True,
                )
                policy_score_chunks.append(shard_scores)
                reward_feature_chunks.append(shard_features)
                candidate_payloads.extend(shard_payloads)
                continue
            if checkpoint_start < reused_prompt_count:
                raise ValueError(
                    "materialization checkpoint boundary crosses the reusable split boundary"
                )
            score_batches: list[torch.Tensor] = []
            feature_batches: list[torch.Tensor] = []
            shard_payloads = []
            for batch_start in range(checkpoint_start, checkpoint_stop, prompt_batch_size):
                batch_stop = min(batch_start + prompt_batch_size, checkpoint_stop)
                batch_records = prompts[batch_start:batch_stop]
                chats = [
                    [message.to_dict() for message in prompt.messages] for prompt in batch_records
                ]
                encoded = tokenizer.apply_chat_template(
                    chats,
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                    return_dict=True,
                )
                prompt_inputs = model_inputs(encoded, target_device)
                batch_seed = derive_seed(
                    seeds.candidate_generation,
                    f"candidate-batch:{batch_start}",
                )
                with fork_torch_seed(batch_seed, target_device):
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
                batch_count = len(batch_records)
                expected_candidates = batch_count * num_candidates
                if scores.shape[0] != expected_candidates:
                    raise RuntimeError("policy returned an unexpected candidate count")
                score_batches.append(
                    scores.reshape(batch_count, num_candidates, -1).to(
                        device="cpu", dtype=torch.float32
                    )
                )
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
                feature_batches.append(
                    features.reshape(batch_count, num_candidates, -1)
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                for prompt_index, prompt in enumerate(batch_records):
                    text = prompt_text(prompt)
                    for candidate_index in range(num_candidates):
                        flat_index = prompt_index * num_candidates + candidate_index
                        active_response_ids = candidates.input_ids[flat_index][
                            candidates.response_mask[flat_index].bool()
                        ]
                        shard_payloads.append(
                            {
                                "prompt_id": prompt.prompt_id,
                                "candidate_id": candidate_id(prompt.prompt_id, candidate_index),
                                "candidate_index": candidate_index,
                                "split": prompt.split,
                                "prompt": text,
                                "response": decode_response(tokenizer, active_response_ids),
                                "token_ids": [
                                    int(value)
                                    for value in candidates.input_ids[flat_index].tolist()
                                ],
                                "response_mask": [
                                    int(value)
                                    for value in candidates.response_mask[flat_index].tolist()
                                ],
                                "terminated_by_eos": bool(
                                    candidates.terminated_by_eos[flat_index].item()
                                ),
                                "reached_max_length": bool(
                                    candidates.reached_max_length[flat_index].item()
                                ),
                            }
                        )
                del log_probabilities, scores, hidden_output, features, candidates, prompt_inputs
            shard_scores = torch.cat(score_batches, dim=0)
            shard_features = torch.cat(feature_batches, dim=0)
            _save_candidate_shard(
                shard,
                manifest=work_manifest,
                start=checkpoint_start,
                stop=checkpoint_stop,
                prompt_ids=[record.prompt_id for record in checkpoint_records],
                policy_scores=shard_scores,
                reward_features=shard_features,
                payloads=shard_payloads,
            )
            print(
                f"materialize candidates={checkpoint_stop}/{len(prompts)} status=checkpointed",
                flush=True,
            )
        policy_score_chunks.append(shard_scores)
        reward_feature_chunks.append(shard_features)
        candidate_payloads.extend(shard_payloads)

    policy_scores = torch.cat(policy_score_chunks, dim=0)
    reward_features = torch.cat(reward_feature_chunks, dim=0)
    del setup, policy_model, policy_score_chunks, reward_feature_chunks, probe_inputs
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
    reused_candidate_count = reused_prompt_count * num_candidates
    flat_prompts = [
        str(candidate["prompt"]) for candidate in candidate_payloads[reused_candidate_count:]
    ]
    flat_responses = [
        str(candidate["response"]) for candidate in candidate_payloads[reused_candidate_count:]
    ]
    raw_batches: list[torch.Tensor] = []
    oracle_batch_size = int(oracle_config["batch_size"])
    for start in range(0, len(flat_prompts), oracle_batch_size):
        stop = min(start + oracle_batch_size, len(flat_prompts))
        raw_batches.append(
            score_oracle_chats(
                oracle_model,
                oracle_tokenizer,
                flat_prompts[start:stop],
                flat_responses[start:stop],
                device=target_device,
            ).to(device="cpu", dtype=torch.float32)
        )
        if stop == len(flat_prompts) or stop % (oracle_batch_size * 64) == 0:
            print(
                f"materialize oracle_scores={stop}/{len(flat_prompts)}",
                flush=True,
            )
    new_raw_oracle_scores = torch.cat(raw_batches).reshape(
        len(prompts) - reused_prompt_count,
        num_candidates,
    )
    raw_oracle_scores = (
        new_raw_oracle_scores
        if reused_raw_oracle_scores is None
        else torch.cat((reused_raw_oracle_scores, new_raw_oracle_scores), dim=0)
    )
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
        protocol=normalized["protocol"],
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
        "split_component_reuse": split_reuse_evidence,
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
        shutil.rmtree(work)
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

"""Validation-only realized forward-KL calibration rules."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .artifacts import exact_delta_artifact_metadata_sha256
from .config import TRPO_PROTOCOL, config_hash, validate_config
from .hf import (
    exact_candidate_logits,
    generate_exact_candidates,
    sequence_forward_kl,
)
from .prompts import load_prompt_jsonl
from .runtime import (
    fork_torch_seed,
    load_pretrained,
    model_inputs,
    producer_identity,
    require_module,
    sha256_file,
)
from .seeding import SeedBundle, derive_seed
from .trpo_policy import adapter_name, validate_trpo_adapter_metadata

_NORMAL_95 = 1.959963984540054
COMPONENT_SCHEMA = "prorm-trpo-kl-calibration-component/v1"
SCHEMA = "prorm-trpo-calibrated-adapters/v1"


def summarize_prompt_kl(
    prompt_forward_kl: Sequence[float],
    *,
    kl_target: float,
    point_relative_interval: Sequence[float],
    confidence_level: float,
    upper_confidence_multiplier: float,
) -> dict[str, Any]:
    """Apply the preregistered prompt-clustered point and CI gates."""

    values = [float(value) for value in prompt_forward_kl]
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("prompt_forward_kl must contain at least two finite prompt means")
    target = float(kl_target)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("kl_target must be finite and positive")
    interval = [float(value) for value in point_relative_interval]
    if len(interval) != 2 or not 0.0 < interval[0] < 1.0 < interval[1]:
        raise ValueError("point_relative_interval must straddle one")
    if float(confidence_level) != 0.95:
        raise ValueError("only the preregistered 95 percent normal interval is supported")
    upper_multiplier = float(upper_confidence_multiplier)
    if not math.isfinite(upper_multiplier) or upper_multiplier <= interval[1]:
        raise ValueError("upper_confidence_multiplier is invalid")
    mean = float(statistics.fmean(values))
    sample_sd = float(statistics.stdev(values))
    standard_error = sample_sd / math.sqrt(len(values))
    lower_confidence = mean - _NORMAL_95 * standard_error
    upper_confidence = mean + _NORMAL_95 * standard_error
    point_pass = interval[0] * target <= mean <= interval[1] * target
    upper_pass = upper_confidence <= upper_multiplier * target
    return {
        "sampling_unit": "prompt",
        "num_prompts": len(values),
        "mean_forward_kl": mean,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "confidence_level": 0.95,
        "confidence_interval": [lower_confidence, upper_confidence],
        "point_interval": [interval[0] * target, interval[1] * target],
        "upper_confidence_limit": upper_multiplier * target,
        "point_gate_passed": point_pass,
        "upper_confidence_gate_passed": upper_pass,
        "accepted": point_pass and upper_pass,
    }


def next_quadratic_ratio_scale(
    current_scale: float,
    observed_forward_kl: float,
    *,
    kl_target: float,
    max_scale_change: float = 4.0,
) -> float:
    """Use local KL quadraticity to choose the next deterministic scale."""

    current = float(current_scale)
    observed = float(observed_forward_kl)
    target = float(kl_target)
    if not math.isfinite(current) or current <= 0.0:
        raise ValueError("current_scale must be finite and positive")
    if not math.isfinite(observed):
        raise ValueError("observed_forward_kl must be finite")
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("kl_target must be finite and positive")
    maximum_change = float(max_scale_change)
    if not math.isfinite(maximum_change) or maximum_change <= 1.0:
        raise ValueError("max_scale_change must be finite and exceed one")
    if observed <= 0.0:
        return current * 2.0
    proposed = current * math.sqrt(target / observed)
    # One attempt cannot move more than four-fold in either direction.  This
    # deterministic safety bound prevents a noisy near-zero estimate from
    # producing an extreme adapter while retaining rapid quadratic correction.
    return min(
        current * maximum_change,
        max(current / maximum_change, proposed),
    )


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


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _load_calibration_model(
    config: Mapping[str, Any],
    initial_adapters: Path,
    *,
    policy_name: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[Any, Any]:
    transformers = require_module("transformers")
    peft = require_module("peft")
    policy = config["policy"]
    tokenizer = load_pretrained(
        transformers.AutoTokenizer,
        policy["model"],
        policy["revision"],
        local_files_only=local_files_only,
        kind="policy tokenizer",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    base = load_pretrained(
        transformers.AutoModelForCausalLM,
        policy["model"],
        policy["revision"],
        local_files_only=local_files_only,
        kind="policy model",
        torch_dtype=_dtype(policy["dtype"]),
    )
    model = peft.PeftModel.from_pretrained(
        base,
        initial_adapters / policy_name,
        adapter_name=policy_name,
        is_trainable=False,
    )
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def _prompt_forward_kl(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Any],
    *,
    responses: int,
    prompt_batch_size: int,
    base_seed: int,
    policy_name: str,
    device: torch.device,
    policy_config: Mapping[str, Any],
) -> list[float]:
    generation_kwargs = {
        **dict(policy_config["sampling"]),
        "num_return_sequences": responses,
        "max_new_tokens": policy_config["max_response_tokens"],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    result: list[float] = []
    for start in range(0, len(prompts), prompt_batch_size):
        batch = prompts[start : start + prompt_batch_size]
        chats = [[message.to_dict() for message in prompt.messages] for prompt in batch]
        encoded = tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = model_inputs(encoded, device)
        generation_seed = derive_seed(
            base_seed,
            f"kl-calibration:{policy_name}:batch:{start}",
        )
        with fork_torch_seed(generation_seed, device):
            candidates = generate_exact_candidates(
                model,
                inputs["input_ids"],
                prompt_attention_mask=inputs["attention_mask"],
                generation_kwargs=generation_kwargs,
            )
        updated_logits = exact_candidate_logits(model, candidates)
        with model.disable_adapter():
            reference_logits = exact_candidate_logits(model, candidates)
        trajectory_kl = sequence_forward_kl(
            updated_logits,
            reference_logits,
            candidates.response_mask,
        ).to(dtype=torch.float64)
        del updated_logits, reference_logits
        if trajectory_kl.numel() != len(batch) * responses:
            raise RuntimeError("KL calibration generated an unexpected response count")
        result.extend(
            float(value)
            for value in trajectory_kl.reshape(len(batch), responses).mean(dim=1).tolist()
        )
    return result


def _lora_b_parameters(model: Any, policy_name: str) -> list[tuple[str, torch.Tensor]]:
    all_b = [
        (name, parameter) for name, parameter in model.named_parameters() if ".lora_B." in name
    ]
    named = [(name, parameter) for name, parameter in all_b if f".{policy_name}." in name]
    selected = named or all_b
    if not selected:
        raise RuntimeError("loaded adapter exposes no LoRA-B parameters")
    return selected


def _scaled_adapter_copy(source: Path, target: Path, *, scale: float) -> dict[str, str]:
    safetensors = require_module("safetensors.torch")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite calibrated adapter: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for path in source.iterdir():
            if path.name == "adapter_model.safetensors":
                continue
            if path.is_file():
                shutil.copy2(path, staging / path.name)
        source_weights = source / "adapter_model.safetensors"
        tensors = safetensors.load_file(str(source_weights), device="cpu")
        if not any("lora_B" in key for key in tensors):
            raise ValueError("adapter safetensors contains no LoRA-B weights")
        scaled = {
            key: (value * scale if "lora_B" in key else value) for key, value in tensors.items()
        }
        safetensors.save_file(scaled, str(staging / source_weights.name))
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        path.relative_to(target).as_posix(): sha256_file(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def _policy_component_path(root: Path, policy_name: str) -> Path:
    return root / ".checkpoints" / f"{policy_name}.json"


def _policy_failure_path(root: Path, policy_name: str) -> Path:
    return root / ".failures" / f"{policy_name}.json"


def _quarantine_component(root: Path, policy_name: str) -> None:
    target = root / policy_name
    receipt = _policy_component_path(root, policy_name)
    if not target.exists() and not receipt.exists():
        return
    rejected_root = root / ".rejected"
    rejected_root.mkdir(exist_ok=True)
    rejected = Path(tempfile.mkdtemp(prefix=f"{policy_name}.", dir=rejected_root))
    if target.exists():
        os.replace(target, rejected / "adapter")
    if receipt.exists():
        os.replace(receipt, rejected / "component.json")


def calibrate_trpo_adapter_policy(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    initial_adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Calibrate one policy on validation prompts and emit a final adapter."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError("KL calibration requires the Fisher-TRPO protocol")
    if seed not in normalized["run"]["seeds"]:
        raise ValueError("seed must be configured")
    initial_root = Path(initial_adapter_dir)
    initial_metadata = validate_trpo_adapter_metadata(
        initial_root,
        expected_producer=producer_identity(),
    )
    if policy_name not in initial_metadata["adapters"]:
        raise ValueError(f"unknown TRPO policy: {policy_name}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ".checkpoints").mkdir(exist_ok=True)
    receipt_path = _policy_component_path(output_root, policy_name)
    failure_path = _policy_failure_path(output_root, policy_name)
    target = output_root / policy_name
    if receipt_path.is_file() and target.is_dir():
        with receipt_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if (
            isinstance(existing, dict)
            and existing.get("schema") == COMPONENT_SCHEMA
            and existing.get("status") == "complete"
            and existing.get("accepted") is True
            and all(
                sha256_file(target / relative) == digest
                for relative, digest in existing.get("files", {}).items()
            )
        ):
            return existing
        _quarantine_component(output_root, policy_name)
    if receipt_path.exists() or target.exists():
        _quarantine_component(output_root, policy_name)

    artifact_root = Path(artifact_dir)
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_root,
        expected_config_hash=config_hash(normalized),
        expected_seed=seed,
    )
    if initial_metadata["artifact_metadata_sha256"] != artifact_identity:
        raise ValueError("initial adapter artifact mismatch")
    prompts = [
        prompt
        for prompt in load_prompt_jsonl(artifact_root / "prompts.jsonl")
        if prompt.split == "validation"
    ]
    expected_prompts = int(normalized["run"]["split_sizes"]["validation"])
    if len(prompts) != expected_prompts:
        raise ValueError("artifact validation prompt inventory is incomplete")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, tokenizer = _load_calibration_model(
        normalized,
        initial_root,
        policy_name=policy_name,
        device=target_device,
        local_files_only=local_files_only,
    )
    parameters = _lora_b_parameters(model, policy_name)
    originals = [parameter.detach().clone() for _, parameter in parameters]
    record = initial_metadata["adapters"][policy_name]
    kl_target = float(record["kl_target"])
    calibration = normalized["policy_update"]["calibration"]
    attempts: list[dict[str, Any]] = []
    scale = 1.0
    accepted = False
    try:
        for attempt in range(int(calibration["max_attempts"])):
            with torch.no_grad():
                for (_, parameter), original in zip(parameters, originals, strict=True):
                    parameter.copy_(original * scale)
            prompt_kl = _prompt_forward_kl(
                model,
                tokenizer,
                prompts,
                responses=int(calibration["responses_per_prompt"]),
                prompt_batch_size=int(normalized["execution"]["rollout_prompt_batch_size"]),
                base_seed=SeedBundle.from_base_seed(seed).rollout,
                policy_name=policy_name,
                device=target_device,
                policy_config=normalized["policy"],
            )
            summary = summarize_prompt_kl(
                prompt_kl,
                kl_target=kl_target,
                point_relative_interval=calibration["point_relative_interval"],
                confidence_level=float(calibration["confidence_level"]),
                upper_confidence_multiplier=float(calibration["upper_confidence_multiplier"]),
            )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "scale_multiplier": scale,
                    "prompt_forward_kl": prompt_kl,
                    "summary": summary,
                }
            )
            diagnostic = {
                "schema": COMPONENT_SCHEMA,
                "protocol": TRPO_PROTOCOL,
                "status": "running",
                "accepted": False,
                "config_sha256": config_hash(normalized),
                "artifact_metadata_sha256": artifact_identity,
                "initial_adapter_metadata_sha256": sha256_file(initial_root / "metadata.json"),
                "seed": seed,
                "policy_name": policy_name,
                "reward_source": record["reward_source"],
                "kl_target": kl_target,
                "attempts": attempts,
                "producer": producer_identity(),
            }
            _atomic_json(failure_path, diagnostic)
            print(
                f"kl-calibration policy={policy_name} attempt={attempt + 1} "
                f"scale={scale:.9g} mean={summary['mean_forward_kl']:.9g} "
                f"ci_upper={summary['confidence_interval'][1]:.9g} "
                f"point_pass={summary['point_gate_passed']} "
                f"upper_pass={summary['upper_confidence_gate_passed']}",
                flush=True,
            )
            if summary["accepted"]:
                accepted = True
                break
            scale = next_quadratic_ratio_scale(
                scale,
                summary["mean_forward_kl"],
                kl_target=kl_target,
                max_scale_change=float(calibration["max_scale_change_per_attempt"]),
            )
    finally:
        del model, tokenizer, parameters, originals
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    if not accepted:
        diagnostic["status"] = "failed"
        _atomic_json(failure_path, diagnostic)
        raise RuntimeError(
            f"KL calibration failed closed after {len(attempts)} attempts: {policy_name}"
        )
    files = _scaled_adapter_copy(
        initial_root / policy_name,
        target,
        scale=scale,
    )
    receipt = {
        "schema": COMPONENT_SCHEMA,
        "protocol": TRPO_PROTOCOL,
        "status": "complete",
        "accepted": True,
        "config_sha256": config_hash(normalized),
        "artifact_metadata_sha256": artifact_identity,
        "initial_adapter_metadata_sha256": sha256_file(initial_root / "metadata.json"),
        "seed": seed,
        "policy_name": policy_name,
        "reward_source": record["reward_source"],
        "kl_target": kl_target,
        "final_scale_multiplier": scale,
        "attempts": attempts,
        "files": files,
        "producer": producer_identity(),
    }
    _atomic_json(receipt_path, receipt)
    failure_path.unlink(missing_ok=True)
    return receipt


def assemble_calibrated_trpo_adapters(
    config: Mapping[str, object],
    initial_adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    """Validate all nine components and write calibrated adapter metadata."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError("calibration assembly requires the Fisher-TRPO protocol")
    initial_root = Path(initial_adapter_dir)
    initial = validate_trpo_adapter_metadata(
        initial_root,
        expected_producer=producer_identity(),
    )
    root = Path(output_dir)
    records: dict[str, Any] = {}
    for method in ("mle_rm", "pro_rm", "oracle"):
        for target_raw in normalized["policy_update"]["kl_targets"]:
            name = adapter_name(method, float(target_raw))
            path = _policy_component_path(root, name)
            with path.open("r", encoding="utf-8") as stream:
                receipt = json.load(stream)
            if (
                not isinstance(receipt, dict)
                or receipt.get("schema") != COMPONENT_SCHEMA
                or receipt.get("accepted") is not True
                or receipt.get("config_sha256") != config_hash(normalized)
                or receipt.get("seed") != seed
                or receipt.get("producer") != producer_identity()
                or receipt.get("initial_adapter_metadata_sha256")
                != sha256_file(initial_root / "metadata.json")
            ):
                raise ValueError(f"invalid KL calibration component: {name}")
            for relative, digest in receipt["files"].items():
                if sha256_file(root / name / relative) != digest:
                    raise ValueError(f"calibrated adapter digest mismatch: {name}/{relative}")
            records[name] = {
                "reward_source": receipt["reward_source"],
                "kl_target": receipt["kl_target"],
                "final_scale_multiplier": receipt["final_scale_multiplier"],
                "realized_forward_kl": receipt["attempts"][-1]["summary"],
                "files": receipt["files"],
                "component_receipt_sha256": sha256_file(path),
            }
    metadata = {
        "schema": SCHEMA,
        "protocol": TRPO_PROTOCOL,
        "config_sha256": config_hash(normalized),
        "artifact_metadata_sha256": initial["artifact_metadata_sha256"],
        "reward_result_sha256": initial["reward_result_sha256"],
        "initial_adapter_metadata_sha256": sha256_file(initial_root / "metadata.json"),
        "seed": seed,
        "calibration_status": "complete",
        "kl_targets": [float(value) for value in normalized["policy_update"]["kl_targets"]],
        "adapters": records,
        "producer": producer_identity(),
    }
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != metadata:
                raise ValueError("existing calibrated adapter metadata differs")
    else:
        _atomic_json(metadata_path, metadata)
    return metadata


def validate_calibrated_trpo_adapters(
    config: Mapping[str, object],
    adapters: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    """Validate the final calibrated inventory and all component hashes."""

    normalized = validate_config(config)
    root = Path(adapters)
    with (root / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != SCHEMA
        or metadata.get("config_sha256") != config_hash(normalized)
        or metadata.get("seed") != seed
        or metadata.get("producer") != producer_identity()
        or metadata.get("calibration_status") != "complete"
    ):
        raise ValueError("calibrated TRPO adapter metadata is invalid")
    if len(metadata.get("adapters", {})) != 9:
        raise ValueError("calibrated TRPO adapter inventory is incomplete")
    for name, record in metadata["adapters"].items():
        if sha256_file(_policy_component_path(root, name)) != record.get(
            "component_receipt_sha256"
        ):
            raise ValueError(f"calibrated component receipt mismatch: {name}")
        for relative, digest in record["files"].items():
            if sha256_file(root / name / relative) != digest:
                raise ValueError(f"calibrated adapter digest mismatch: {name}/{relative}")
    return metadata


__all__ = [
    "COMPONENT_SCHEMA",
    "SCHEMA",
    "assemble_calibrated_trpo_adapters",
    "calibrate_trpo_adapter_policy",
    "next_quadratic_ratio_scale",
    "summarize_prompt_kl",
    "validate_calibrated_trpo_adapters",
]

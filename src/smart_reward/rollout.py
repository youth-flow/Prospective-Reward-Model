"""Resumable fresh test-prompt rollouts for common-beta policy instances."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .artifacts import exact_delta_artifact_metadata_sha256
from .checkpoints import validate_stage_receipt, write_stage_receipt
from .config import PROTOCOL, TRPO_PROTOCOL, config_hash, validate_config
from .evaluation import summarize_rollouts, summarize_trpo_rollouts
from .exact_policy import SCHEMA as ADAPTER_SCHEMA
from .hf import generate_exact_candidates, score_exact_candidates, score_oracle_chats
from .kl_calibration import validate_calibrated_trpo_adapters
from .prompts import PromptRecord, load_prompt_jsonl
from .runtime import (
    decode_response,
    fork_torch_seed,
    load_pretrained,
    model_inputs,
    producer_identity,
    prompt_text,
    require_module,
    sha256_file,
)
from .seeding import SeedBundle, derive_seed
from .trpo_policy import adapter_name as _trpo_adapter_name

SCHEMA = "prorm-policy-utility/v2"
POLICY_SCHEMA = "prorm-single-policy-rollout/v1"
SHARD_SCHEMA = "prorm-policy-rollout-shard/v1"


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _adapter_name(method: str, beta: float) -> str:
    return f"{method}__beta_{format(beta, 'g').replace('.', 'p')}"


def policy_instance_names(config: Mapping[str, object]) -> list[str]:
    normalized = validate_config(config)
    if normalized["protocol"] == TRPO_PROTOCOL:
        return [
            "pi0",
            *[
                _trpo_adapter_name(method, float(target))
                for method in ("mle_rm", "pro_rm", "oracle")
                for target in normalized["policy_update"]["kl_targets"]
            ],
        ]
    return [
        "pi0",
        *[
            _adapter_name(method, float(beta))
            for method in ("mle_rm", "pro_rm", "oracle")
            for beta in normalized["policy_update"]["beta_grid"]
        ],
    ]


def _generation_kwargs(config: Mapping[str, Any], responses: int, tokenizer: Any) -> dict[str, Any]:
    policy = config["policy"]
    return {
        **dict(policy["sampling"]),
        "num_return_sequences": responses,
        "max_new_tokens": int(policy["max_response_tokens"]),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }


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


def _load_adapter_metadata(
    config: Mapping[str, Any], adapter_dir: str | os.PathLike[str], *, seed: int
) -> tuple[Path, dict[str, Any], list[str]]:
    adapters_root = Path(adapter_dir)
    if config["protocol"] == TRPO_PROTOCOL:
        metadata = validate_calibrated_trpo_adapters(
            config,
            adapters_root,
            seed=seed,
        )
    else:
        metadata = _read_json(adapters_root / "metadata.json")
        if metadata.get("schema") != ADAPTER_SCHEMA:
            raise ValueError("unsupported adapter metadata")
    if metadata.get("config_sha256") != config_hash(config):
        raise ValueError("adapter config mismatch")
    if metadata.get("seed") != seed:
        raise ValueError("adapter seed mismatch")
    names = policy_instance_names(config)[1:]
    if set(metadata.get("adapters", {})) != set(names):
        raise ValueError("adapter inventory does not match the configured policy grid")
    return adapters_root, metadata, names


def _test_prompts(config: Mapping[str, Any], artifact_dir: Path) -> list[PromptRecord]:
    count = int(config["evaluation"]["rollout"]["prompts"])
    prompts = [
        prompt
        for prompt in load_prompt_jsonl(artifact_dir / "prompts.jsonl")
        if prompt.split == "test"
    ][:count]
    if len(prompts) != count:
        raise ValueError("artifact does not contain the configured test rollout prompts")
    return prompts


def _load_models(
    config: Mapping[str, Any],
    adapter_root: Path,
    *,
    adapter_name: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[Any, Any, Any, Any]:
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
        adapter_root / adapter_name,
        adapter_name=adapter_name,
        is_trainable=False,
    )
    model.to(device).eval()
    oracle = config["oracle"]
    oracle_tokenizer = load_pretrained(
        transformers.AutoTokenizer,
        oracle["model"],
        oracle["revision"],
        local_files_only=local_files_only,
        kind="oracle tokenizer",
        use_fast=True,
    )
    oracle_model = load_pretrained(
        transformers.AutoModelForSequenceClassification,
        oracle["model"],
        oracle["revision"],
        local_files_only=local_files_only,
        kind="oracle model",
        torch_dtype=_dtype(oracle["dtype"]),
        num_labels=1,
    )
    oracle_model.to(device).eval()
    return model, tokenizer, oracle_model, oracle_tokenizer


@torch.no_grad()
def _generate_policy_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    oracle_model: torch.nn.Module,
    oracle_tokenizer: Any,
    prompts: Sequence[PromptRecord],
    *,
    responses: int,
    generation_seed: int,
    device: torch.device,
    reference: bool,
    oracle_center: float,
    oracle_scale: float,
    policy_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    chats = [[message.to_dict() for message in prompt.messages] for prompt in prompts]
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
    call_kwargs = _generation_kwargs({"policy": policy_config}, responses, tokenizer)
    with fork_torch_seed(generation_seed, device):
        if reference:
            with model.disable_adapter():
                candidates = generate_exact_candidates(
                    model,
                    inputs["input_ids"],
                    prompt_attention_mask=inputs["attention_mask"],
                    generation_kwargs=call_kwargs,
                )
                updated_log_prob = score_exact_candidates(model, candidates)
            reference_log_prob = updated_log_prob
        else:
            candidates = generate_exact_candidates(
                model,
                inputs["input_ids"],
                prompt_attention_mask=inputs["attention_mask"],
                generation_kwargs=call_kwargs,
            )
            updated_log_prob = score_exact_candidates(model, candidates)
            with model.disable_adapter():
                reference_log_prob = score_exact_candidates(model, candidates)
    expected = len(prompts) * responses
    if candidates.input_ids.shape[0] != expected:
        raise RuntimeError("batched generation returned an unexpected candidate count")
    response_texts = [
        decode_response(
            tokenizer,
            candidates.input_ids[index][candidates.response_mask[index].bool()],
        )
        for index in range(expected)
    ]
    prompt_values = [prompt_text(prompt) for prompt in prompts]
    raw_rewards = score_oracle_chats(
        oracle_model,
        oracle_tokenizer,
        [value for value in prompt_values for _ in range(responses)],
        response_texts,
        device=device,
    ).to(dtype=torch.float64)
    rewards = (raw_rewards - oracle_center) / oracle_scale
    ratios = (updated_log_prob - reference_log_prob).detach().to(dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for response_index in range(responses):
            flat_index = prompt_index * responses + response_index
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "response_index": response_index,
                    "prompt": prompt_values[prompt_index],
                    "response": response_texts[flat_index],
                    "oracle_reward": float(rewards[flat_index].item()),
                    "forward_log_ratio": float(ratios[flat_index].item()),
                }
            )
    return rows


def _policy_descriptor(
    policy_name: str, adapter_metadata: Mapping[str, Any]
) -> dict[str, str | float | None]:
    if policy_name == "pi0":
        if adapter_metadata.get("protocol") == TRPO_PROTOCOL:
            return {"policy_instance": "pi0", "reward_source": "pi0", "kl_target": None}
        return {"policy_instance": "pi0", "reward_source": "pi0", "beta": None}
    record = adapter_metadata["adapters"][policy_name]
    if adapter_metadata.get("protocol") == TRPO_PROTOCOL:
        return {
            "policy_instance": policy_name,
            "reward_source": str(record["reward_source"]),
            "kl_target": float(record["kl_target"]),
        }
    return {
        "policy_instance": policy_name,
        "reward_source": str(record["reward_source"]),
        "beta": float(record["beta"]),
    }


def _policy_inputs(artifact_identity: str, adapter_identity: str) -> dict[str, str]:
    return {"artifact_metadata": artifact_identity, "adapter_metadata": adapter_identity}


def validate_single_policy_rollout(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = validate_config(config)
    artifact_root = Path(artifact_dir)
    adapters_root, adapter_metadata, _ = _load_adapter_metadata(normalized, adapter_dir, seed=seed)
    if policy_name not in policy_instance_names(normalized):
        raise ValueError(f"unknown policy instance: {policy_name}")
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_root,
        expected_config_hash=config_hash(normalized),
        expected_seed=seed,
    )
    if adapter_metadata.get("artifact_metadata_sha256") != artifact_identity:
        raise ValueError("adapter artifact mismatch")
    adapter_identity = sha256_file(adapters_root / "metadata.json")
    target = Path(output_dir)
    metadata_path = target / "metadata.json"
    rollouts_path = target / "rollouts.jsonl"
    metadata = _read_json(metadata_path)
    expected_metadata = {
        "schema": POLICY_SCHEMA,
        "protocol": normalized["protocol"],
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": producer_identity(),
        **_policy_descriptor(policy_name, adapter_metadata),
        "prompt_count": int(normalized["evaluation"]["rollout"]["prompts"]),
        "responses_per_prompt": int(normalized["evaluation"]["rollout"]["responses_per_prompt"]),
    }
    if metadata != expected_metadata:
        raise ValueError(f"single-policy rollout metadata mismatch: {target}")
    validate_stage_receipt(
        target / "receipt.json",
        normalized,
        stage=f"rollout:{policy_name}",
        seed=seed,
        inputs=_policy_inputs(artifact_identity, adapter_identity),
        outputs={
            "metadata": sha256_file(metadata_path),
            "rollouts": sha256_file(rollouts_path),
        },
    )
    rows: list[dict[str, Any]] = []
    with rollouts_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid rollout row {line_number}: {rollouts_path}")
            rows.append(value)
    expected_rows = metadata["prompt_count"] * metadata["responses_per_prompt"]
    if len(rows) != expected_rows:
        raise ValueError(f"single-policy rollout row count mismatch: {target}")
    return metadata, rows


def evaluate_single_policy_rollout(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    normalized = validate_config(config)
    if (
        normalized["protocol"] not in {PROTOCOL, TRPO_PROTOCOL}
        or seed not in normalized["run"]["seeds"]
    ):
        raise ValueError("protocol or seed mismatch")
    if policy_name not in policy_instance_names(normalized):
        raise ValueError(f"unknown policy instance: {policy_name}")
    target = Path(output_dir)
    if target.exists():
        metadata, _ = validate_single_policy_rollout(
            normalized,
            artifact_dir,
            adapter_dir,
            target,
            policy_name=policy_name,
            seed=seed,
        )
        work = target.parent / f".{target.name}.work"
        if work.exists():
            shutil.rmtree(work)
        print(f"rollout policy={policy_name} status=reused", flush=True)
        return metadata

    artifact_root = Path(artifact_dir)
    adapters_root, adapter_metadata, adapter_names = _load_adapter_metadata(
        normalized, adapter_dir, seed=seed
    )
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_root,
        expected_config_hash=config_hash(normalized),
        expected_seed=seed,
    )
    if adapter_metadata.get("artifact_metadata_sha256") != artifact_identity:
        raise ValueError("adapter artifact mismatch")
    adapter_identity = sha256_file(adapters_root / "metadata.json")
    prompts = _test_prompts(normalized, artifact_root)
    artifact_metadata = _read_json(artifact_root / "metadata.json")
    transform = artifact_metadata["evidence"]["oracle_transform"]
    descriptor = _policy_descriptor(policy_name, adapter_metadata)
    work = target.parent / f".{target.name}.work"
    manifest = {
        "schema": "prorm-policy-rollout-work/v1",
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "producer": producer_identity(),
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        **descriptor,
    }
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError(f"rollout work identity mismatch: {work}")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(f"unidentified rollout work directory: {work}")
        _atomic_json(manifest_path, manifest)

    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    load_name = adapter_names[0] if policy_name == "pi0" else policy_name
    model, tokenizer, oracle_model, oracle_tokenizer = _load_models(
        normalized,
        adapters_root,
        adapter_name=load_name,
        device=device_value,
        local_files_only=local_files_only,
    )
    if policy_name != "pi0":
        model.set_adapter(policy_name)
    responses = int(normalized["evaluation"]["rollout"]["responses_per_prompt"])
    prompt_batch = int(normalized["execution"]["rollout_prompt_batch_size"])
    checkpoint_prompts = int(normalized["execution"]["rollout_checkpoint_prompts"])
    base_seed = SeedBundle.from_base_seed(seed).rollout
    try:
        for checkpoint_start in range(0, len(prompts), checkpoint_prompts):
            checkpoint_stop = min(checkpoint_start + checkpoint_prompts, len(prompts))
            shard_path = work / "shards" / f"{checkpoint_start:06d}-{checkpoint_stop:06d}.json"
            if shard_path.exists():
                shard = _read_json(shard_path)
                if (
                    shard.get("schema") != SHARD_SCHEMA
                    or shard.get("manifest") != manifest
                    or shard.get("start") != checkpoint_start
                    or shard.get("stop") != checkpoint_stop
                ):
                    raise ValueError(f"rollout shard identity mismatch: {shard_path}")
                print(
                    f"rollout policy={policy_name} prompts={checkpoint_stop}/{len(prompts)} "
                    "status=reused",
                    flush=True,
                )
                continue
            rows: list[dict[str, Any]] = []
            for batch_start in range(checkpoint_start, checkpoint_stop, prompt_batch):
                batch_stop = min(batch_start + prompt_batch, checkpoint_stop)
                generation_seed = derive_seed(base_seed, f"rollout-batch:{batch_start}")
                batch_rows = _generate_policy_batch(
                    model,
                    tokenizer,
                    oracle_model,
                    oracle_tokenizer,
                    prompts[batch_start:batch_stop],
                    responses=responses,
                    generation_seed=generation_seed,
                    device=device_value,
                    reference=policy_name == "pi0",
                    oracle_center=float(transform["b"]),
                    oracle_scale=float(transform["tau"]),
                    policy_config=normalized["policy"],
                )
                rows.extend(
                    {
                        **row,
                        "policy": descriptor["reward_source"],
                        "policy_instance": descriptor["policy_instance"],
                        **(
                            {"kl_target": descriptor["kl_target"]}
                            if "kl_target" in descriptor
                            else {"beta": descriptor["beta"]}
                        ),
                    }
                    for row in batch_rows
                )
            _atomic_json(
                shard_path,
                {
                    "schema": SHARD_SCHEMA,
                    "manifest": manifest,
                    "start": checkpoint_start,
                    "stop": checkpoint_stop,
                    "rows": rows,
                },
            )
            print(
                f"rollout policy={policy_name} prompts={checkpoint_stop}/{len(prompts)} "
                "status=checkpointed",
                flush=True,
            )
    finally:
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if device_value.type == "cuda":
            torch.cuda.empty_cache()

    all_rows: list[dict[str, Any]] = []
    for checkpoint_start in range(0, len(prompts), checkpoint_prompts):
        checkpoint_stop = min(checkpoint_start + checkpoint_prompts, len(prompts))
        shard = _read_json(work / "shards" / f"{checkpoint_start:06d}-{checkpoint_stop:06d}.json")
        all_rows.extend(shard["rows"])
    metadata = {
        "schema": POLICY_SCHEMA,
        "protocol": normalized["protocol"],
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": producer_identity(),
        **descriptor,
        "prompt_count": len(prompts),
        "responses_per_prompt": responses,
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.final-", dir=target.parent))
    try:
        metadata_path = staging / "metadata.json"
        rollouts_path = staging / "rollouts.jsonl"
        _atomic_json(metadata_path, metadata)
        with rollouts_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_stage_receipt(
            staging / "receipt.json",
            normalized,
            stage=f"rollout:{policy_name}",
            seed=seed,
            inputs=_policy_inputs(artifact_identity, adapter_identity),
            outputs={
                "metadata": sha256_file(metadata_path),
                "rollouts": sha256_file(rollouts_path),
            },
        )
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    shutil.rmtree(work)
    validate_single_policy_rollout(
        normalized,
        artifact_root,
        adapters_root,
        target,
        policy_name=policy_name,
        seed=seed,
    )
    return metadata


def _rows_to_samples(
    rows: Sequence[Mapping[str, Any]], prompts: Sequence[PromptRecord], responses: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = [(prompt.prompt_id, index) for prompt in prompts for index in range(responses)]
    observed = [(str(row.get("prompt_id")), int(row.get("response_index", -1))) for row in rows]
    if observed != expected:
        raise ValueError("policy rollout rows are not in canonical prompt/response order")
    rewards = torch.tensor([float(row["oracle_reward"]) for row in rows], dtype=torch.float64)
    ratios = torch.tensor([float(row["forward_log_ratio"]) for row in rows], dtype=torch.float64)
    return rewards.reshape(len(prompts), responses), ratios.reshape(len(prompts), responses)


def assemble_policy_rollouts(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    policy_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    normalized = validate_config(config)
    artifact_root = Path(artifact_dir)
    adapters_root, _, _ = _load_adapter_metadata(normalized, adapter_dir, seed=seed)
    prompts = _test_prompts(normalized, artifact_root)
    responses = int(normalized["evaluation"]["rollout"]["responses_per_prompt"])
    root = Path(policy_root)
    names = policy_instance_names(normalized)
    metadata_by_name: dict[str, dict[str, Any]] = {}
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        metadata_by_name[name], rows_by_name[name] = validate_single_policy_rollout(
            normalized,
            artifact_root,
            adapters_root,
            root / name,
            policy_name=name,
            seed=seed,
        )
    samples = {name: _rows_to_samples(rows_by_name[name], prompts, responses) for name in names}
    reference_rewards, _ = samples["pi0"]
    metrics: dict[str, Any] = {}
    if normalized["protocol"] == TRPO_PROTOCOL:
        for target_raw in normalized["policy_update"]["kl_targets"]:
            target_value = float(target_raw)
            target_metrics = {
                method: summarize_trpo_rollouts(
                    *samples[_trpo_adapter_name(method, target_value)],
                    kl_target=target_value,
                    reference_oracle_rewards=reference_rewards,
                )
                for method in ("mle_rm", "pro_rm", "oracle")
            }
            target_metrics["pi0"] = summarize_trpo_rollouts(
                reference_rewards,
                torch.zeros_like(reference_rewards),
                kl_target=target_value,
                reference_oracle_rewards=reference_rewards,
            )
            metrics[str(target_value)] = target_metrics
    else:
        beta_grid = [float(value) for value in normalized["policy_update"]["beta_grid"]]
        for beta in beta_grid:
            oracle_reward, oracle_ratio = samples[_adapter_name("oracle", beta)]
            beta_metrics = {
                method: summarize_rollouts(
                    *samples[_adapter_name(method, beta)],
                    beta=beta,
                    reference_oracle_rewards=reference_rewards,
                    oracle_ngd_oracle_rewards=oracle_reward,
                    oracle_ngd_forward_log_ratios=oracle_ratio,
                )
                for method in ("mle_rm", "pro_rm", "oracle")
            }
            beta_metrics["pi0"] = summarize_rollouts(
                reference_rewards,
                torch.zeros_like(reference_rewards),
                beta=beta,
                reference_oracle_rewards=reference_rewards,
                oracle_ngd_oracle_rewards=oracle_reward,
                oracle_ngd_forward_log_ratios=oracle_ratio,
            )
            metrics[str(beta)] = beta_metrics
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_root,
        expected_config_hash=config_hash(normalized),
        expected_seed=seed,
    )
    adapter_identity = sha256_file(adapters_root / "metadata.json")
    payload = {
        "schema": SCHEMA,
        "protocol": normalized["protocol"],
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": producer_identity(),
        "kl_orientation": "updated_to_reference",
        "policy_instances": names,
        "metrics": metrics,
    }
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite rollout output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        rollouts_path = staging / "rollouts.jsonl"
        metrics_path = staging / "metrics.json"
        with rollouts_path.open("w", encoding="utf-8", newline="\n") as stream:
            for name in names:
                for row in rows_by_name[name]:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_json(metrics_path, payload)
        inputs = {
            "artifact_metadata": artifact_identity,
            "adapter_metadata": adapter_identity,
            **{f"policy_{name}": sha256_file(root / name / "receipt.json") for name in names},
        }
        write_stage_receipt(
            staging / "receipt.json",
            normalized,
            stage="rollout-aggregate",
            seed=seed,
            inputs=inputs,
            outputs={
                "metrics": sha256_file(metrics_path),
                "rollouts": sha256_file(rollouts_path),
            },
        )
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return payload


def evaluate_policy_rollouts(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper that resumes each policy before aggregating all ten."""

    normalized = validate_config(config)
    target = Path(output_dir)
    policy_root = target.parent / "policy_rollout_parts"
    for name in policy_instance_names(normalized):
        evaluate_single_policy_rollout(
            normalized,
            artifact_dir,
            adapter_dir,
            policy_root / name,
            policy_name=name,
            seed=seed,
            device=device,
            local_files_only=local_files_only,
        )
    return assemble_policy_rollouts(
        normalized,
        artifact_dir,
        adapter_dir,
        policy_root,
        target,
        seed=seed,
    )


__all__ = [
    "SCHEMA",
    "POLICY_SCHEMA",
    "assemble_policy_rollouts",
    "evaluate_policy_rollouts",
    "evaluate_single_policy_rollout",
    "policy_instance_names",
    "validate_single_policy_rollout",
]

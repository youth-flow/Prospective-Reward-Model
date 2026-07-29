"""Fresh test-prompt rollouts for all common-beta policy instances."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .artifacts import exact_delta_artifact_metadata_sha256
from .config import PROTOCOL, config_hash, validate_config
from .evaluation import summarize_rollouts
from .exact_policy import SCHEMA as ADAPTER_SCHEMA
from .hf import generate_exact_candidates, score_exact_candidates, score_oracle_chats
from .prompts import PromptRecord, load_prompt_jsonl
from .runtime import (
    decode_response,
    fork_torch_seed,
    load_pretrained,
    model_inputs,
    prompt_text,
    require_module,
    sha256_file,
)
from .seeding import SeedBundle

SCHEMA = "prorm-policy-utility/v1"


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _adapter_name(method: str, beta: float) -> str:
    return f"{method}__beta_{format(beta, 'g').replace('.', 'p')}"


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


@torch.no_grad()
def _generate_policy_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    oracle_model: torch.nn.Module,
    oracle_tokenizer: Any,
    prompts: list[PromptRecord],
    *,
    responses: int,
    generation_seed: int,
    device: torch.device,
    reference: bool,
    oracle_center: float,
    oracle_scale: float,
    policy_config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rewards: list[torch.Tensor] = []
    log_ratios: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    with fork_torch_seed(generation_seed, device):
        for prompt in prompts:
            encoded = tokenizer.apply_chat_template(
                [message.to_dict() for message in prompt.messages],
                tokenize=True,
                add_generation_prompt=True,
                truncation=False,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = model_inputs(encoded, device)
            call_kwargs = _generation_kwargs({"policy": policy_config}, responses, tokenizer)
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
            responses_text = [
                decode_response(
                    tokenizer,
                    candidates.input_ids[index][candidates.response_mask[index].bool()],
                )
                for index in range(responses)
            ]
            prompt_value = prompt_text(prompt)
            raw_oracle_values = score_oracle_chats(
                oracle_model,
                oracle_tokenizer,
                [prompt_value] * responses,
                responses_text,
                device=device,
            ).to(dtype=torch.float64)
            oracle_values = (raw_oracle_values - oracle_center) / oracle_scale
            ratios = (updated_log_prob - reference_log_prob).detach().to(dtype=torch.float64)
            rewards.append(oracle_values.detach().cpu())
            log_ratios.append(ratios.detach().cpu())
            for index, response in enumerate(responses_text):
                records.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "response_index": index,
                        "prompt": prompt_value,
                        "response": response,
                        "oracle_reward": float(oracle_values[index].item()),
                        "forward_log_ratio": float(ratios[index].item()),
                    }
                )
            del (
                candidates,
                updated_log_prob,
                reference_log_prob,
                raw_oracle_values,
                oracle_values,
                ratios,
            )
    return torch.stack(rewards), torch.stack(log_ratios), records


def _load_models(
    config: Mapping[str, Any],
    adapter_root: Path,
    adapter_names: list[str],
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
        adapter_root / adapter_names[0],
        adapter_name=adapter_names[0],
        is_trainable=False,
    )
    for name in adapter_names[1:]:
        model.load_adapter(adapter_root / name, adapter_name=name, is_trainable=False)
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
    normalized = validate_config(config)
    if normalized["protocol"] != PROTOCOL or seed not in normalized["run"]["seeds"]:
        raise ValueError("protocol or seed mismatch")
    adapters_root = Path(adapter_dir)
    adapter_metadata = json.loads((adapters_root / "metadata.json").read_text(encoding="utf-8"))
    if adapter_metadata.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("unsupported adapter metadata")
    if adapter_metadata.get("config_sha256") != config_hash(normalized):
        raise ValueError("adapter config mismatch")
    if adapter_metadata.get("seed") != seed:
        raise ValueError("adapter seed mismatch")
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=config_hash(normalized),
        expected_seed=seed,
    )
    if adapter_metadata.get("artifact_metadata_sha256") != artifact_identity:
        raise ValueError("adapter artifact mismatch")
    beta_grid = [float(value) for value in normalized["policy_update"]["beta_grid"]]
    adapter_names = [
        _adapter_name(method, beta)
        for method in ("mle_rm", "pro_rm", "oracle")
        for beta in beta_grid
    ]
    if set(adapter_metadata["adapters"]) != set(adapter_names):
        raise ValueError("adapter inventory does not match the configured beta grid")
    prompts = [
        prompt
        for prompt in load_prompt_jsonl(Path(artifact_dir) / "prompts.jsonl")
        if prompt.split == "test"
    ][: int(normalized["evaluation"]["rollout"]["prompts"])]
    artifact_metadata = json.loads(
        (Path(artifact_dir) / "metadata.json").read_text(encoding="utf-8")
    )
    transform = artifact_metadata["evidence"]["oracle_transform"]
    oracle_center = float(transform["b"])
    oracle_scale = float(transform["tau"])
    device_value = torch.device(device)
    model, tokenizer, oracle_model, oracle_tokenizer = _load_models(
        normalized, adapters_root, adapter_names, device_value, local_files_only
    )
    responses = int(normalized["evaluation"]["rollout"]["responses_per_prompt"])
    generation_seed = SeedBundle.from_base_seed(seed).rollout
    model.set_adapter(adapter_names[0])
    reference_rewards, _, reference_records = _generate_policy_samples(
        model,
        tokenizer,
        oracle_model,
        oracle_tokenizer,
        prompts,
        responses=responses,
        generation_seed=generation_seed,
        device=device_value,
        reference=True,
        oracle_center=oracle_center,
        oracle_scale=oracle_scale,
        policy_config=normalized["policy"],
    )
    records: list[dict[str, Any]] = [
        {**record, "policy": "pi0", "beta": None} for record in reference_records
    ]
    samples: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in adapter_names:
        model.set_adapter(name)
        reward, ratio, rows = _generate_policy_samples(
            model,
            tokenizer,
            oracle_model,
            oracle_tokenizer,
            prompts,
            responses=responses,
            generation_seed=generation_seed,
            device=device_value,
            reference=False,
            oracle_center=oracle_center,
            oracle_scale=oracle_scale,
            policy_config=normalized["policy"],
        )
        samples[name] = (reward, ratio)
        adapter = adapter_metadata["adapters"][name]
        records.extend(
            {
                **row,
                "policy": adapter["reward_source"],
                "beta": adapter["beta"],
            }
            for row in rows
        )
    metrics: dict[str, Any] = {}
    for beta in beta_grid:
        oracle_reward, oracle_ratio = samples[_adapter_name("oracle", beta)]
        zeros = torch.zeros_like(reference_rewards)
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
            zeros,
            beta=beta,
            reference_oracle_rewards=reference_rewards,
            oracle_ngd_oracle_rewards=oracle_reward,
            oracle_ngd_forward_log_ratios=oracle_ratio,
        )
        metrics[str(beta)] = beta_metrics
    payload = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": sha256_file(adapters_root / "metadata.json"),
        "kl_orientation": "updated_to_reference",
        "metrics": metrics,
    }
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite rollout output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        with (staging / "rollouts.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (staging / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if device_value.type == "cuda":
            torch.cuda.empty_cache()
    return payload


__all__ = ["evaluate_policy_rollouts"]

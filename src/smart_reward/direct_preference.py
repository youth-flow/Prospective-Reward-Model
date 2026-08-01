"""Paper-faithful log-policy DPO/AuxDPO on the frozen candidate graph.

The policy loss uses response-token sequence log-probability ratios.  Cached
LoRA tangent scores are used only for AuxDPO's reference-policy null-space
penalty and for the already-frozen approximate-regret evaluation geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as functional
import yaml

from .artifacts import exact_delta_artifact_metadata_sha256, load_exact_delta_artifact
from .config import config_hash, load_config
from .data import CandidateNode, load_jsonl
from .exact import ExactSplitData, empirical_fisher_score_rows, policy_reward_moment
from .exact_policy import _load_policy
from .linear import DampedEmpiricalFisher
from .pcg import pcg
from .runtime import producer_identity, sha256_file
from .scores import sequence_log_probs

CONFIG_SCHEMA = "prorm-dpo-auxdpo-extension-config/v1"
REFERENCE_SCHEMA = "prorm-direct-preference-reference-logps/v1"
FIT_SCHEMA = "prorm-direct-preference-fit/v1"
EVALUATION_SCHEMA = "prorm-dpo-auxdpo-seed-evaluation/v1"
METHODS = ("dpo", "auxdpo")


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_direct_preference_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported DPO/AuxDPO extension config")
    expected_top = {
        "schema",
        "source_config",
        "source_config_sha256",
        "experiment",
        "training",
        "auxdpo",
        "evaluation",
    }
    if set(value) != expected_top:
        raise ValueError("DPO/AuxDPO extension config keys changed")
    experiment = value["experiment"]
    if tuple(experiment.get("methods", ())) != METHODS:
        raise ValueError(f"methods must be exactly {METHODS!r}")
    betas = tuple(float(item) for item in experiment.get("betas", ()))
    seeds = tuple(experiment.get("seeds", ()))
    if experiment.get("name") == "dpo-auxdpo-main-v1":
        if betas != (0.1, 0.2, 0.3):
            raise ValueError("formal extension betas must be exactly (0.1, 0.2, 0.3)")
        if seeds != (20261001, 20261002, 20261003):
            raise ValueError("formal extension seeds changed")
        if value["training"].get("limit_prompts_per_split") is not None:
            raise ValueError("formal extension may not limit prompts")
    elif experiment.get("name") == "dpo-auxdpo-smoke-v1":
        if betas != (0.2,) or seeds != (20261001,):
            raise ValueError("smoke identity changed")
        limit = value["training"].get("limit_prompts_per_split")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
            raise ValueError("smoke prompt limit must be in [1, 8]")
    else:
        raise ValueError("undeclared direct-preference experiment name")
    training = value["training"]
    if training.get("objective") != "response_token_log_policy_ratio":
        raise ValueError("direct preference training must use real sequence log probabilities")
    if int(training.get("epochs", 0)) != 2:
        raise ValueError("direct-preference extension is frozen to two epochs")
    if str(training.get("compute_dtype")) != "bfloat16":
        raise ValueError("formal extension compute dtype must be bfloat16")
    aux = value["auxdpo"]
    if aux.get("reported_test_reward") != "policy_implied_implicit_reward":
        raise ValueError("AuxDPO test reward scope changed")
    digest = value.get("source_config_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("source config digest is invalid")
    return value


def resolve_source_config(
    extension_path: str | os.PathLike[str], extension: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    source = (Path(extension_path).resolve().parent / str(extension["source_config"])).resolve()
    config = load_config(source)
    observed = config_hash(config)
    if observed != extension["source_config_sha256"]:
        raise ValueError(
            "source config digest mismatch: "
            f"expected {extension['source_config_sha256']}, got {observed}"
        )
    if not set(extension["experiment"]["seeds"]).issubset(config["run"]["seeds"]):
        raise ValueError("extension seeds are not a subset of source seeds")
    if int(config["data"]["num_candidates"]) != 6:
        raise ValueError("formal extension requires the frozen six-candidate graph")
    return source, config


def extension_hash(extension: Mapping[str, Any]) -> str:
    return _canonical_sha256(extension)


def pair_indices(candidates: int, device: torch.device | None = None) -> torch.Tensor:
    if candidates < 2:
        raise ValueError("at least two candidates are required")
    return torch.triu_indices(candidates, candidates, offset=1, device=device)


def soft_preference_loss(
    node_rewards: torch.Tensor,
    true_rewards: torch.Tensor,
) -> torch.Tensor:
    if node_rewards.ndim != 2 or true_rewards.shape != node_rewards.shape:
        raise ValueError("node and oracle rewards must share shape (prompts, candidates)")
    pairs = pair_indices(node_rewards.shape[1], node_rewards.device)
    margins = node_rewards[:, pairs[0]] - node_rewards[:, pairs[1]]
    targets = torch.sigmoid(true_rewards[:, pairs[0]] - true_rewards[:, pairs[1]])
    return functional.binary_cross_entropy_with_logits(margins, targets)


def centered(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("candidate values must have shape (prompts, candidates)")
    return values - values.mean(dim=1, keepdim=True)


def auxdpo_loss(
    implicit_rewards: torch.Tensor,
    true_rewards: torch.Tensor,
    delta_raw: torch.Tensor,
    reference_scores: torch.Tensor,
    *,
    nullspace_weight: float,
    amplitude_weight: float,
    delta_cap: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if implicit_rewards.shape != true_rewards.shape or delta_raw.shape != true_rewards.shape:
        raise ValueError("implicit rewards, oracle rewards, and delta must share shape")
    if reference_scores.shape[:2] != true_rewards.shape or reference_scores.ndim != 3:
        raise ValueError("reference scores must have shape (prompts, candidates, dimension)")
    delta = centered(delta_cap * torch.tanh(delta_raw))
    preference = soft_preference_loss(implicit_rewards + delta, true_rewards)
    scores = reference_scores - reference_scores.mean(dim=1, keepdim=True)
    moment = torch.einsum("bmd,bm->d", scores, delta) / float(delta.numel())
    null_penalty = moment.square().sum()
    amplitude = delta.square().mean()
    loss = preference + nullspace_weight * null_penalty - amplitude_weight * amplitude
    return loss, {
        "preference_nll": preference,
        "nullspace_penalty": null_penalty,
        "delta_amplitude": amplitude,
        "delta": delta,
        "nullspace_moment": moment,
    }


def candidate_policy_metrics(
    log_ratios: torch.Tensor,
    true_rewards: torch.Tensor,
    *,
    beta: float,
) -> dict[str, Any]:
    if log_ratios.shape != true_rewards.shape or log_ratios.ndim != 2:
        raise ValueError("candidate log ratios and rewards must share shape")
    candidates = log_ratios.shape[1]
    log_probabilities = torch.log_softmax(log_ratios.to(torch.float64), dim=1)
    probabilities = log_probabilities.exp()
    rewards = true_rewards.to(torch.float64)
    log_reference = -math.log(candidates)
    reward = (probabilities * rewards).sum(dim=1).mean()
    kl = (probabilities * (log_probabilities - log_reference)).sum(dim=1).mean()
    objective = reward - beta * kl
    tabular_log = torch.log_softmax(rewards / beta, dim=1)
    tabular = tabular_log.exp()
    j_close = beta * (torch.logsumexp(rewards / beta, dim=1) - math.log(candidates)).mean()
    tabular_reward = (tabular * rewards).sum(dim=1).mean()
    tabular_kl = (tabular * (tabular_log - log_reference)).sum(dim=1).mean()
    j_tabular = tabular_reward - beta * tabular_kl
    kl_to_tabular = (probabilities * (log_probabilities - tabular_log)).sum(dim=1).mean()
    result = {
        "R": float(reward.item()),
        "K": float(kl.item()),
        "J": float(objective.item()),
        "delta_J": float((j_close - objective).item()),
        "beta_KL": float((beta * kl_to_tabular).item()),
        "J_close": float(j_close.item()),
        "identity_residuals": {
            "abs_J_tabular_minus_J_close": float((j_tabular - j_close).abs().item()),
            "abs_delta_J_minus_beta_KL": float(
                ((j_close - objective) - beta * kl_to_tabular).abs().item()
            ),
        },
    }
    if max(result["identity_residuals"].values()) > 1.0e-10 * (1.0 + abs(result["J_close"])):
        raise RuntimeError("candidate-pool Gibbs identity failed")
    return result


def collate_candidate_nodes(
    prompt_nodes: Sequence[Sequence[CandidateNode]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = tuple(tuple(group) for group in prompt_nodes)
    if not groups or not groups[0]:
        raise ValueError("candidate batch must not be empty")
    candidates = len(groups[0])
    if any(len(group) != candidates for group in groups):
        raise ValueError("every prompt must have the same candidate count")
    flat = [node for group in groups for node in group]
    maximum = max(len(node.token_ids) for node in flat)
    input_ids = torch.full((len(flat), maximum), int(pad_token_id), dtype=torch.long, device=device)
    response_mask = torch.zeros_like(input_ids)
    attention_mask = torch.zeros_like(input_ids)
    for row, node in enumerate(flat):
        length = len(node.token_ids)
        input_ids[row, :length] = torch.tensor(node.token_ids, dtype=torch.long, device=device)
        response_mask[row, :length] = torch.tensor(
            node.response_mask, dtype=torch.long, device=device
        )
        active = [index for index, value in enumerate(node.response_mask) if value]
        if not active:
            raise ValueError("candidate response mask is empty")
        attention_mask[row, : active[-1] + 1] = 1
    return input_ids, attention_mask, response_mask


def group_candidate_nodes(
    artifact_dir: str | os.PathLike[str],
    *,
    split: str,
    candidates: int,
) -> tuple[tuple[CandidateNode, ...], ...]:
    records = [
        record
        for record in load_jsonl(Path(artifact_dir) / "candidates.jsonl", CandidateNode)
        if record.split == split
    ]
    if len(records) % candidates:
        raise ValueError("candidate inventory is not prompt-complete")
    groups = tuple(
        tuple(records[start : start + candidates]) for start in range(0, len(records), candidates)
    )
    for group in groups:
        if len({node.prompt_id for node in group}) != 1:
            raise ValueError("candidate prompt grouping changed")
        if tuple(node.candidate_index for node in group) != tuple(range(candidates)):
            raise ValueError("candidate ordering changed")
    return groups


def response_log_probabilities(
    model: torch.nn.Module,
    prompt_nodes: Sequence[Sequence[CandidateNode]],
    *,
    pad_token_id: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    input_ids, attention_mask, response_mask = collate_candidate_nodes(
        prompt_nodes, pad_token_id=pad_token_id, device=device
    )
    enabled = device.type == "cuda" and compute_dtype == torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=enabled):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        values = sequence_log_probs(output.logits, input_ids, response_mask)
    batch = len(prompt_nodes)
    candidates = len(prompt_nodes[0])
    return values.reshape(batch, candidates)


def _model_and_setup(
    source_config: Mapping[str, Any], seed: int, device: torch.device
) -> tuple[Any, int]:
    setup = _load_policy(source_config, seed, device, True)
    policy = source_config["policy"]
    # The materialized records are unpadded prompt batches; EOS is also the pad token.
    from .runtime import load_pretrained, require_module

    transformers = require_module("transformers")
    tokenizer = load_pretrained(
        transformers.AutoTokenizer,
        policy["model"],
        policy["revision"],
        local_files_only=True,
        kind="policy tokenizer",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return setup, int(tokenizer.pad_token_id)


@torch.no_grad()
def compute_reference_logps(
    extension_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    extension = load_direct_preference_config(extension_path)
    _, source_config = resolve_source_config(extension_path, extension)
    if seed not in extension["experiment"]["seeds"]:
        raise ValueError("seed is not declared")
    target = Path(output_dir)
    metadata_path = target / "metadata.json"
    tensors_path = target / "reference_logps.safetensors"
    if metadata_path.exists() and tensors_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if (
            existing.get("schema") != REFERENCE_SCHEMA
            or existing.get("seed") != seed
            or existing.get("extension_config_sha256") != extension_hash(extension)
        ):
            raise ValueError("existing reference cache identity mismatch")
        if existing.get("tensors_sha256") != sha256_file(tensors_path):
            raise ValueError("existing reference cache digest mismatch")
        return existing
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    with (Path(artifact_dir) / "metadata.json").open("r", encoding="utf-8") as stream:
        artifact_metadata = json.load(stream)
    expected_a = artifact_metadata["evidence"]["policy_a_sha256"]
    target_device = torch.device(device)
    setup, pad_token_id = _model_and_setup(source_config, seed, target_device)
    if setup.a_state_sha256 != expected_a:
        raise ValueError("LoRA-A basis differs from the materialized policy geometry")
    setup.model.eval()
    batch_size = int(extension["training"]["prompt_batch_size"])
    prompt_limit = extension["training"].get("limit_prompts_per_split")
    tensors: dict[str, torch.Tensor] = {}
    for split in ("train", "test"):
        groups = group_candidate_nodes(
            artifact_dir, split=split, candidates=int(source_config["data"]["num_candidates"])
        )
        if prompt_limit is not None:
            groups = groups[: int(prompt_limit)]
        chunks = []
        for start in range(0, len(groups), batch_size):
            chunks.append(
                response_log_probabilities(
                    setup.model,
                    groups[start : start + batch_size],
                    pad_token_id=pad_token_id,
                    device=target_device,
                    compute_dtype=torch.bfloat16,
                )
                .detach()
                .cpu()
            )
            if (start // batch_size + 1) % 128 == 0 or start + batch_size >= len(groups):
                progress = min(start + batch_size, len(groups))
                print(
                    f"reference seed={seed} split={split} prompts={progress}/{len(groups)}",
                    flush=True,
                )
        tensors[split] = torch.cat(chunks).to(torch.float32).contiguous()
    target.mkdir(parents=True, exist_ok=True)
    safetensors = __import__("safetensors.torch", fromlist=["save_file"])
    temporary = target / ".reference_logps.tmp.safetensors"
    safetensors.save_file(tensors, str(temporary))
    os.replace(temporary, tensors_path)
    payload = {
        "schema": REFERENCE_SCHEMA,
        "status": "complete",
        "seed": seed,
        "source_config_sha256": extension["source_config_sha256"],
        "extension_config_sha256": extension_hash(extension),
        "artifact_metadata_sha256": artifact_identity,
        "artifact_candidates_sha256": sha256_file(Path(artifact_dir) / "candidates.jsonl"),
        "lora_a_sha256": setup.a_state_sha256,
        "response_log_probability": "sum over materialized response tokens including generated EOS",
        "compute_dtype": "bfloat16",
        "limit_prompts_per_split": prompt_limit,
        "tensors_sha256": sha256_file(tensors_path),
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
        "producer": producer_identity(),
    }
    _atomic_json(metadata_path, payload)
    return payload


def _load_reference(
    path: str | os.PathLike[str], *, seed: int, extension_digest: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    root = Path(path)
    with (root / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    tensors_path = root / "reference_logps.safetensors"
    if (
        metadata.get("schema") != REFERENCE_SCHEMA
        or metadata.get("seed") != seed
        or metadata.get("extension_config_sha256") != extension_digest
        or metadata.get("tensors_sha256") != sha256_file(tensors_path)
    ):
        raise ValueError("reference log-probability cache identity mismatch")
    safetensors = __import__("safetensors.torch", fromlist=["load_file"])
    return metadata, safetensors.load_file(str(tensors_path), device="cpu")


def _save_trainable_tensors(setup: Any, path: Path) -> None:
    tensors = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in setup.named_tangent_parameters()
    }
    safetensors = __import__("safetensors.torch", fromlist=["save_file"])
    temporary = path.with_name(f".{path.name}.tmp")
    safetensors.save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _load_trainable_tensors(setup: Any, path: Path) -> None:
    safetensors = __import__("safetensors.torch", fromlist=["load_file"])
    tensors = safetensors.load_file(str(path), device="cpu")
    expected = {name for name, _ in setup.named_tangent_parameters()}
    if set(tensors) != expected:
        raise ValueError("adapter tensor names differ from the fixed LoRA-B layout")
    with torch.no_grad():
        for name, parameter in setup.named_tangent_parameters():
            if tensors[name].shape != parameter.shape:
                raise ValueError(f"adapter tensor shape mismatch: {name}")
            parameter.copy_(tensors[name].to(device=parameter.device, dtype=parameter.dtype))


def _fit_directory(method: str, beta: float) -> str:
    return f"{method}__beta_{format(beta, 'g').replace('.', 'p')}"


def train_direct_preference(
    extension_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reference_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    beta: float,
    method: Literal["dpo", "auxdpo"],
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    extension = load_direct_preference_config(extension_path)
    _, source_config = resolve_source_config(extension_path, extension)
    if method not in METHODS or beta not in tuple(extension["experiment"]["betas"]):
        raise ValueError("undeclared direct-preference condition")
    if seed not in extension["experiment"]["seeds"]:
        raise ValueError("undeclared seed")
    digest = extension_hash(extension)
    reference_metadata, reference = _load_reference(
        reference_dir, seed=seed, extension_digest=digest
    )
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    if reference_metadata["artifact_metadata_sha256"] != artifact_identity:
        raise ValueError("reference cache and artifact identities differ")
    target = Path(output_dir) / _fit_directory(method, beta)
    result_path = target / "result.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if (
            existing.get("schema") == FIT_SCHEMA
            and existing.get("status") == "complete"
            and existing.get("seed") == seed
            and existing.get("method") == method
            and float(existing.get("beta")) == beta
        ):
            return existing
        raise ValueError("existing direct-preference result identity mismatch")
    target.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(device)
    setup, pad_token_id = _model_and_setup(source_config, seed, target_device)
    if setup.a_state_sha256 != reference_metadata["lora_a_sha256"]:
        raise ValueError("training LoRA-A basis differs from the reference cache")
    # Qwen has no active dropout here, but eval mode makes the frozen-candidate
    # log-ratio contract explicit while still allowing gradients through LoRA-B.
    setup.model.eval()
    train = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    ).train
    groups = group_candidate_nodes(artifact_dir, split="train", candidates=train.num_candidates)
    prompt_limit = extension["training"].get("limit_prompts_per_split")
    if prompt_limit is not None:
        groups = groups[: int(prompt_limit)]
    prompt_count = len(groups)
    policy_parameters = [parameter for _, parameter in setup.named_tangent_parameters()]
    delta_raw: torch.nn.Parameter | None = None
    groups_config: list[dict[str, Any]] = [
        {
            "params": policy_parameters,
            "lr": float(extension["training"]["policy_learning_rate"]),
        }
    ]
    if method == "auxdpo":
        delta_raw = torch.nn.Parameter(
            torch.zeros(
                (prompt_count, train.num_candidates),
                dtype=torch.float32,
                device=target_device,
            )
        )
        groups_config.append(
            {
                "params": [delta_raw],
                "lr": float(extension["auxdpo"]["auxiliary_learning_rate"]),
            }
        )
    optimizer = torch.optim.AdamW(
        groups_config,
        betas=tuple(float(item) for item in extension["training"]["adam_betas"]),
        eps=float(extension["training"]["adam_epsilon"]),
        weight_decay=float(extension["training"]["weight_decay"]),
    )
    batch_size = int(extension["training"]["prompt_batch_size"])
    epochs = int(extension["training"]["epochs"])
    condition_seed = int(seed * 1000 + round(beta * 100))
    schedule: list[list[int]] = []
    for epoch in range(epochs):
        epoch_order = list(range(prompt_count))
        random.Random(condition_seed + epoch).shuffle(epoch_order)
        schedule.extend(
            epoch_order[start : start + batch_size] for start in range(0, prompt_count, batch_size)
        )
    total_batches = len(schedule)
    if extension["training"].get("learning_rate_schedule") != "linear_to_zero":
        raise ValueError("direct-preference learning-rate schedule changed")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1.0 - step / max(total_batches, 1)),
    )
    checkpoint_path = target / "checkpoint.pt"
    start_batch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != [digest, seed, beta, method]:
            raise ValueError("training checkpoint identity mismatch")
        by_name = dict(setup.named_tangent_parameters())
        for name, value in checkpoint["adapter"].items():
            by_name[name].data.copy_(value.to(target_device))
        if delta_raw is not None:
            delta_raw.data.copy_(checkpoint["delta_raw"].to(target_device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_batch = int(checkpoint["next_batch"])
        if checkpoint["schedule"] != schedule:
            raise ValueError("training checkpoint schedule mismatch")
    checkpoint_every = int(extension["training"]["checkpoint_prompt_batches"])
    reference_train = reference["train"].to(torch.float32)
    true_train = train.true_rewards[:prompt_count].to(torch.float32)
    score_train = train.policy_scores[:prompt_count].to(torch.float32)
    running = {"loss": 0.0, "preference_nll": 0.0, "nullspace_penalty": 0.0, "delta_amplitude": 0.0}
    processed = 0
    if checkpoint_path.exists():
        running.update({key: float(value) for key, value in checkpoint["running"].items()})
        processed = int(checkpoint["processed"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(target_device)
    for batch_number, indices in enumerate(schedule):
        if batch_number < start_batch:
            continue
        nodes = tuple(groups[index] for index in indices)
        updated = response_log_probabilities(
            setup.model,
            nodes,
            pad_token_id=pad_token_id,
            device=target_device,
            compute_dtype=torch.bfloat16,
        )
        index_cpu = torch.tensor(indices, dtype=torch.long)
        reference_batch = reference_train.index_select(0, index_cpu).to(target_device)
        true_batch = true_train.index_select(0, index_cpu).to(target_device)
        implicit = beta * (updated - reference_batch)
        if delta_raw is None:
            loss = soft_preference_loss(implicit, true_batch)
            diagnostics = {
                "preference_nll": loss,
                "nullspace_penalty": loss.new_zeros(()),
                "delta_amplitude": loss.new_zeros(()),
            }
        else:
            score_batch = score_train.index_select(0, index_cpu).to(target_device)
            loss, diagnostics = auxdpo_loss(
                implicit,
                true_batch,
                delta_raw.index_select(0, index_cpu.to(target_device)),
                score_batch,
                nullspace_weight=float(extension["auxdpo"]["nullspace_weight"]),
                amplitude_weight=float(extension["auxdpo"]["amplitude_weight"]),
                delta_cap=float(extension["auxdpo"]["delta_cap"]),
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("direct-preference loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_parameters,
            max_norm=float(extension["training"]["max_gradient_norm"]),
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("policy gradient norm became non-finite")
        optimizer.step()
        scheduler.step()
        processed += 1
        running["loss"] += float(loss.detach().item())
        for key in ("preference_nll", "nullspace_penalty", "delta_amplitude"):
            running[key] += float(diagnostics[key].detach().item())
        next_batch = batch_number + 1
        if next_batch % checkpoint_every == 0 or next_batch >= total_batches:
            checkpoint = {
                "identity": [digest, seed, beta, method],
                "next_batch": next_batch,
                "schedule": schedule,
                "adapter": {
                    name: parameter.detach().cpu()
                    for name, parameter in setup.named_tangent_parameters()
                },
                "delta_raw": None if delta_raw is None else delta_raw.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "running": running,
                "processed": processed,
            }
            _atomic_torch(checkpoint_path, checkpoint)
            print(
                f"train seed={seed} beta={beta:g} method={method} "
                f"batches={next_batch}/{total_batches} "
                f"loss={float(loss.item()):.8f}",
                flush=True,
            )
    adapter_path = target / "adapter.safetensors"
    _save_trainable_tensors(setup, adapter_path)
    delta_path: Path | None = None
    if delta_raw is not None:
        delta_path = target / "delta.safetensors"
        delta = (
            centered(float(extension["auxdpo"]["delta_cap"]) * torch.tanh(delta_raw.detach()))
            .cpu()
            .contiguous()
        )
        safetensors = __import__("safetensors.torch", fromlist=["save_file"])
        temporary = target / ".delta.tmp.safetensors"
        safetensors.save_file({"train.delta": delta}, str(temporary))
        os.replace(temporary, delta_path)
    setup.model.eval()
    fitted_logps: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for split in ("train", "test"):
            split_groups = group_candidate_nodes(
                artifact_dir, split=split, candidates=train.num_candidates
            )
            if prompt_limit is not None:
                split_groups = split_groups[: int(prompt_limit)]
            chunks = []
            for start in range(0, len(split_groups), batch_size):
                chunks.append(
                    response_log_probabilities(
                        setup.model,
                        split_groups[start : start + batch_size],
                        pad_token_id=pad_token_id,
                        device=target_device,
                        compute_dtype=torch.bfloat16,
                    )
                    .detach()
                    .cpu()
                )
            fitted_logps[split] = torch.cat(chunks).to(torch.float32).contiguous()
    logps_path = target / "updated_logps.safetensors"
    safetensors = __import__("safetensors.torch", fromlist=["save_file"])
    temporary = target / ".updated_logps.tmp.safetensors"
    safetensors.save_file(fitted_logps, str(temporary))
    os.replace(temporary, logps_path)
    payload = {
        "schema": FIT_SCHEMA,
        "status": "complete",
        "method": method,
        "seed": seed,
        "beta": beta,
        "source_config_sha256": extension["source_config_sha256"],
        "extension_config_sha256": digest,
        "artifact_metadata_sha256": artifact_identity,
        "reference_metadata_sha256": sha256_file(Path(reference_dir) / "metadata.json"),
        "training": {
            "objective": "exact_soft_BTL_over_response_token_log_policy_ratio",
            "epochs": epochs,
            "limit_prompts_per_split": prompt_limit,
            "prompt_batches": total_batches,
            "mean_online_metrics": {
                key: value / max(processed, 1) for key, value in running.items()
            },
            "policy_learning_rate": float(extension["training"]["policy_learning_rate"]),
            "learning_rate_schedule": extension["training"]["learning_rate_schedule"],
            "auxiliary_learning_rate": (
                None if method == "dpo" else float(extension["auxdpo"]["auxiliary_learning_rate"])
            ),
        },
        "files": {
            "adapter.safetensors": sha256_file(adapter_path),
            "updated_logps.safetensors": sha256_file(logps_path),
            **({} if delta_path is None else {"delta.safetensors": sha256_file(delta_path)}),
        },
        "lora_a_sha256": setup.a_state_sha256,
        "producer": producer_identity(),
    }
    _atomic_json(result_path, payload)
    checkpoint_path.unlink(missing_ok=True)
    return payload


def _reward_metrics(
    split: ExactSplitData,
    implicit_reward: torch.Tensor,
    train: ExactSplitData,
    *,
    beta: float,
    relative_damping: float,
    cg_tolerance: float,
    cg_max_iterations: int,
    residual_recompute_interval: int,
) -> dict[str, float]:
    predicted = centered(implicit_reward.to(torch.float64))
    target = centered(split.true_rewards.to(torch.float64))
    nll = float(soft_preference_loss(predicted, target).item())
    mse = float((predicted - target).square().mean().item())
    oracle_moment = policy_reward_moment(
        split.policy_scores.to(torch.float64), split.true_rewards.to(torch.float64)
    )
    predicted_moment = policy_reward_moment(split.policy_scores.to(torch.float64), predicted)
    error = predicted_moment - oracle_moment
    rows = empirical_fisher_score_rows(train.policy_scores.to(torch.float64), "raw_second_moment")
    raw = DampedEmpiricalFisher(rows, damping=0.0)
    damping = relative_damping * float(raw.diagonal().mean().item())
    fisher = DampedEmpiricalFisher(rows, damping=damping)
    solve = pcg(
        fisher.matvec,
        error,
        inverse_diagonal=fisher.pcg_inverse_diagonal(),
        max_iterations=cg_max_iterations,
        tolerance=cg_tolerance,
        residual_recompute_interval=residual_recompute_interval,
    )
    if not solve.converged:
        raise RuntimeError("direct-preference approximate-regret Fisher solve did not converge")
    quadratic = float(torch.dot(error, solve.solution).item())
    if quadratic < -1.0e-10:
        raise RuntimeError("direct-preference approximate regret became negative")
    return {
        "NLL": nll,
        "MSE": mse,
        "approximate_regret": max(0.0, quadratic) / (2.0 * beta),
        "moment_error_inverse_fisher_quadratic": max(0.0, quadratic),
        "moment_error_pcg_relative_residual": float(solve.relative_residual),
    }


def evaluate_direct_preference_seed(
    extension_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    source_reward_result: str | os.PathLike[str],
    reference_dir: str | os.PathLike[str],
    fits_dir: str | os.PathLike[str],
    baseline_evaluation: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    source_reward_result = Path(source_reward_result)
    reference_dir = Path(reference_dir)
    fits_dir = Path(fits_dir)
    baseline_evaluation = Path(baseline_evaluation)
    output = Path(output)
    extension = load_direct_preference_config(extension_path)
    _, source_config = resolve_source_config(extension_path, extension)
    digest = extension_hash(extension)
    reference_metadata, reference = _load_reference(
        reference_dir, seed=seed, extension_digest=digest
    )
    experiment = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    with Path(source_reward_result).open("r", encoding="utf-8") as stream:
        reward_source = json.load(stream)
    relative_damping = float(reward_source["selected_relative_damping"])
    with Path(baseline_evaluation).open("r", encoding="utf-8") as stream:
        baseline = json.load(stream)
    if (
        baseline.get("seed") != seed
        or baseline.get("provenance_bridge", {}).get("source_artifact_metadata_sha256")
        != reference_metadata["artifact_metadata_sha256"]
    ):
        raise ValueError("baseline evaluation and reused source artifact differ")
    geometry = source_config["geometry"]
    conditions: dict[str, Any] = {}
    unified: dict[str, Any] = {}
    for beta_raw in extension["experiment"]["betas"]:
        beta = float(beta_raw)
        beta_record: dict[str, Any] = {}
        for method in METHODS:
            root = Path(fits_dir) / _fit_directory(method, beta)
            with (root / "result.json").open("r", encoding="utf-8") as stream:
                fit = json.load(stream)
            if (
                fit.get("schema") != FIT_SCHEMA
                or fit.get("status") != "complete"
                or fit.get("extension_config_sha256") != digest
                or fit.get("seed") != seed
                or fit.get("method") != method
                or float(fit.get("beta")) != beta
            ):
                raise ValueError("direct-preference fit identity mismatch")
            logps_path = root / "updated_logps.safetensors"
            if fit["files"]["updated_logps.safetensors"] != sha256_file(logps_path):
                raise ValueError("updated log-probability digest mismatch")
            safetensors = __import__("safetensors.torch", fromlist=["load_file"])
            updated = safetensors.load_file(str(logps_path), device="cpu")
            log_ratio_test = updated["test"].to(torch.float64) - reference["test"].to(torch.float64)
            implicit_test = beta * log_ratio_test
            reward_metrics = _reward_metrics(
                experiment.test,
                implicit_test,
                experiment.train,
                beta=beta,
                relative_damping=relative_damping,
                cg_tolerance=float(geometry["cg_tolerance"]),
                cg_max_iterations=int(geometry["cg_max_iterations"]),
                residual_recompute_interval=int(geometry["residual_recompute_interval"]),
            )
            policy_metrics = candidate_policy_metrics(
                log_ratio_test, experiment.test.true_rewards, beta=beta
            )
            diagnostics: dict[str, Any] = {}
            if method == "auxdpo":
                delta_path = root / "delta.safetensors"
                if fit["files"]["delta.safetensors"] != sha256_file(delta_path):
                    raise ValueError("AuxDPO delta digest mismatch")
                delta = safetensors.load_file(str(delta_path), device="cpu")["train.delta"].to(
                    torch.float64
                )
                implicit_train = beta * (
                    updated["train"].to(torch.float64) - reference["train"].to(torch.float64)
                )
                scores = experiment.train.policy_scores.to(torch.float64)
                scores = scores - scores.mean(dim=1, keepdim=True)
                moment = torch.einsum("pmd,pm->d", scores, delta) / float(delta.numel())
                augmented_nll = soft_preference_loss(
                    implicit_train + delta, experiment.train.true_rewards.to(torch.float64)
                )
                diagnostics = {
                    "train_augmented_NLL": float(augmented_nll.item()),
                    "global_nullspace_moment_norm": float(torch.linalg.vector_norm(moment).item()),
                    "delta_rms": float(delta.square().mean().sqrt().item()),
                    "delta_max_abs": float(delta.abs().max().item()),
                }
            beta_record[method] = {
                "reward_scope": "policy_implied_implicit_reward",
                "reward": reward_metrics,
                "policy": policy_metrics,
                "auxiliary_train_diagnostics": diagnostics,
                "fit_result_sha256": sha256_file(root / "result.json"),
            }
        conditions[str(beta)] = beta_record
        baseline_policy = baseline["policy"][str(beta)]
        direct_policies = {
            method: {
                key: value
                for key, value in beta_record[method]["policy"].items()
                if key not in {"J_close", "identity_residuals"}
            }
            for method in METHODS
        }
        reward_rows = {}
        for method in ("mle", "pro"):
            record = baseline["reward"][method]
            reward_rows[method] = {
                "NLL": record["NLL"],
                "MSE": record["MSE"],
                "approximate_regret": record["approximate_regret"][str(beta)],
            }
        for method in METHODS:
            reward_rows[method] = dict(beta_record[method]["reward"])
        unified[str(beta)] = {
            "J_close": baseline_policy["J_close"],
            "policies": {**baseline_policy["policies"], **direct_policies},
            "reward": reward_rows,
        }
    payload = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete",
        "seed": seed,
        "betas": list(extension["experiment"]["betas"]),
        "methods": list(METHODS),
        "source_config_sha256": extension["source_config_sha256"],
        "extension_config_sha256": digest,
        "artifact_metadata_sha256": reference_metadata["artifact_metadata_sha256"],
        "source_reward_result_sha256": sha256_file(source_reward_result),
        "baseline_evaluation_sha256": sha256_file(baseline_evaluation),
        "reference_metadata_sha256": sha256_file(Path(reference_dir) / "metadata.json"),
        "test_usage": "evaluation_only_no_selection",
        "auxdpo_test_reward_scope": "policy_implied_implicit_reward_excludes_train_only_delta",
        "conditions": conditions,
        "unified": unified,
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def aggregate_direct_preference(
    extension_path: str | os.PathLike[str],
    results: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_direct_preference_config(extension_path)
    digest = extension_hash(extension)
    records_with_paths: list[tuple[dict[str, Any], Path]] = []
    for path in results:
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        if (
            record.get("schema") != EVALUATION_SCHEMA
            or record.get("status") != "complete"
            or record.get("extension_config_sha256") != digest
        ):
            raise ValueError("unsupported direct-preference seed evaluation")
        records_with_paths.append((record, source))
    expected_seeds = list(extension["experiment"]["seeds"])
    if sorted(record["seed"] for record, _ in records_with_paths) != expected_seeds:
        raise ValueError("direct-preference aggregate does not contain exactly three seeds")
    records_with_paths.sort(key=lambda item: item[0]["seed"])
    records = [record for record, _ in records_with_paths]

    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        }

    aggregate: dict[str, Any] = {}
    for beta_raw in extension["experiment"]["betas"]:
        beta = str(float(beta_raw))
        first = records[0]["unified"][beta]
        policy = {}
        for method in first["policies"]:
            policy[method] = {
                metric: summary(
                    [
                        float(record["unified"][beta]["policies"][method][metric])
                        for record in records
                    ]
                )
                for metric in ("R", "K", "J", "delta_J", "beta_KL")
            }
        reward = {}
        for method in first["reward"]:
            reward[method] = {
                metric: summary(
                    [float(record["unified"][beta]["reward"][method][metric]) for record in records]
                )
                for metric in ("NLL", "MSE", "approximate_regret")
            }
        aggregate[beta] = {
            "J_close": summary([float(record["unified"][beta]["J_close"]) for record in records]),
            "policy": policy,
            "reward": reward,
        }
    payload = {
        "schema": "prorm-dpo-auxdpo-aggregate/v1",
        "status": "complete",
        "extension_config_sha256": digest,
        "source_config_sha256": extension["source_config_sha256"],
        "seeds": expected_seeds,
        "betas": list(extension["experiment"]["betas"]),
        "evaluation": aggregate,
        "inputs": {str(record["seed"]): sha256_file(path) for record, path in records_with_paths},
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def audit_direct_preference(
    extension_path: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_direct_preference_config(extension_path)
    root = Path(run_root)
    result_paths = [
        root / f"seed-{seed}" / "evaluation.json" for seed in extension["experiment"]["seeds"]
    ]
    aggregate_path = root / "aggregate.json"
    with aggregate_path.open("r", encoding="utf-8") as stream:
        stored = json.load(stream)
    with tempfile.TemporaryDirectory(prefix="prorm-direct-audit-") as temporary:
        recomputed_path = Path(temporary) / "aggregate.json"
        recomputed = aggregate_direct_preference(extension_path, result_paths, recomputed_path)
    if stored != recomputed:
        raise ValueError("stored direct-preference aggregate differs from recomputation")
    seed_checks = {}
    for seed, result_path in zip(extension["experiment"]["seeds"], result_paths, strict=True):
        with result_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        maximum = 0.0
        for beta in result["unified"].values():
            for policy in beta["policies"].values():
                maximum = max(maximum, abs(float(policy["delta_J"]) - float(policy["beta_KL"])))
        if maximum > 1.0e-10:
            raise ValueError(f"seed {seed} violates the Gibbs gap identity")
        seed_checks[str(seed)] = {
            "evaluation_sha256": sha256_file(result_path),
            "baseline_evaluation_sha256": result["baseline_evaluation_sha256"],
            "artifact_metadata_sha256": result["artifact_metadata_sha256"],
            "max_abs_delta_J_minus_beta_KL": maximum,
            "status": "passed",
        }
    payload = {
        "schema": "prorm-dpo-auxdpo-integrity-audit/v1",
        "status": "passed",
        "extension_config_sha256": extension_hash(extension),
        "source_config_sha256": extension["source_config_sha256"],
        "seeds": seed_checks,
        "aggregate_sha256": sha256_file(aggregate_path),
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


__all__ = [
    "CONFIG_SCHEMA",
    "EVALUATION_SCHEMA",
    "FIT_SCHEMA",
    "METHODS",
    "REFERENCE_SCHEMA",
    "auxdpo_loss",
    "aggregate_direct_preference",
    "audit_direct_preference",
    "candidate_policy_metrics",
    "centered",
    "collate_candidate_nodes",
    "compute_reference_logps",
    "evaluate_direct_preference_seed",
    "extension_hash",
    "group_candidate_nodes",
    "load_direct_preference_config",
    "pair_indices",
    "resolve_source_config",
    "response_log_probabilities",
    "soft_preference_loss",
    "train_direct_preference",
]

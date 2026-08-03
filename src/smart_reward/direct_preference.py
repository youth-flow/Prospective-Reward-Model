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
ADAPTIVE_EXPERIMENTS = (
    "dpo-auxdpo-converged-v1",
    "dpo-auxdpo-converged-smoke-v1",
)


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
    experiment_name = experiment.get("name")
    if experiment_name == "dpo-auxdpo-main-v1":
        if betas != (0.1, 0.2, 0.3):
            raise ValueError("formal extension betas must be exactly (0.1, 0.2, 0.3)")
        if seeds != (20261001, 20261002, 20261003):
            raise ValueError("formal extension seeds changed")
        if value["training"].get("limit_prompts_per_split") is not None:
            raise ValueError("formal extension may not limit prompts")
    elif experiment_name == "dpo-auxdpo-smoke-v1":
        if betas != (0.2,) or seeds != (20261001,):
            raise ValueError("smoke identity changed")
        limit = value["training"].get("limit_prompts_per_split")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
            raise ValueError("smoke prompt limit must be in [1, 8]")
    elif experiment_name == "dpo-auxdpo-converged-v1":
        if betas != (0.2,) or seeds != (20261001, 20261002, 20261003):
            raise ValueError("converged experiment identity changed")
        if value["training"].get("limit_prompts_per_split") is not None:
            raise ValueError("formal converged experiment may not limit prompts")
    elif experiment_name == "dpo-auxdpo-converged-smoke-v1":
        if betas != (0.2,) or seeds != (20261001,):
            raise ValueError("converged smoke identity changed")
        limit = value["training"].get("limit_prompts_per_split")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
            raise ValueError("converged smoke prompt limit must be in [1, 8]")
    else:
        raise ValueError("undeclared direct-preference experiment name")
    training = value["training"]
    if training.get("objective") != "response_token_log_policy_ratio":
        raise ValueError("direct preference training must use real sequence log probabilities")
    if experiment_name in ADAPTIVE_EXPERIMENTS:
        expected_training = {
            "objective",
            "max_epochs",
            "min_epochs",
            "prompt_batch_size",
            "gradient_accumulation_steps",
            "policy_optimizer",
            "learning_rate_schedule",
            "warmup_epochs",
            "policy_learning_rate",
            "min_policy_learning_rate",
            "validation_min_delta",
            "minimum_validation_improvement",
            "lr_reduction_factor",
            "lr_reduction_patience",
            "minimum_lr_reductions",
            "early_stopping_patience",
            "weight_decay",
            "adam_betas",
            "adam_epsilon",
            "max_gradient_norm",
            "checkpoint_prompt_batches",
            "compute_dtype",
            "model_dtype",
            "shuffle",
            "validation_selection_metric",
            "restore_best_validation_checkpoint",
            "test_usage",
            "limit_prompts_per_split",
        }
        if set(training) != expected_training:
            raise ValueError("adaptive direct-preference training keys changed")
        if training["learning_rate_schedule"] != "warmup_then_validation_plateau":
            raise ValueError("adaptive learning-rate schedule changed")
        if training["validation_selection_metric"] != "policy_implied_soft_btl_nll":
            raise ValueError("validation selection metric changed")
        if training["test_usage"] != "final_evaluation_only":
            raise ValueError("test may only be used for final evaluation")
        if training["restore_best_validation_checkpoint"] is not True:
            raise ValueError("best validation checkpoint must be restored")
        for key in (
            "max_epochs",
            "min_epochs",
            "prompt_batch_size",
            "warmup_epochs",
            "lr_reduction_patience",
            "early_stopping_patience",
            "checkpoint_prompt_batches",
        ):
            if isinstance(training[key], bool) or not isinstance(training[key], int):
                raise ValueError(f"training.{key} must be an integer")
        if not 1 <= training["min_epochs"] <= training["max_epochs"]:
            raise ValueError("adaptive epoch bounds are invalid")
        if not 0 <= training["warmup_epochs"] <= training["min_epochs"]:
            raise ValueError("warmup_epochs must be within the minimum training budget")
        if training["prompt_batch_size"] < 1 or training["checkpoint_prompt_batches"] < 1:
            raise ValueError("adaptive batch sizes must be positive")
        if training["gradient_accumulation_steps"] != 1:
            raise ValueError("AuxDPO moment estimation forbids microbatch accumulation")
        if training["policy_optimizer"] != "adamw":
            raise ValueError("adaptive direct preference optimizer changed")
        if training["shuffle"] != "deterministic_prompt_level":
            raise ValueError("adaptive prompt shuffle changed")
        if training["minimum_lr_reductions"] < 0:
            raise ValueError("minimum_lr_reductions must be nonnegative")
        for key in (
            "policy_learning_rate",
            "min_policy_learning_rate",
            "validation_min_delta",
            "lr_reduction_factor",
            "max_gradient_norm",
        ):
            number = float(training[key])
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"training.{key} must be finite and positive")
        minimum_improvement = float(training["minimum_validation_improvement"])
        if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
            raise ValueError("minimum_validation_improvement must be finite and nonnegative")
        if not 0.0 < float(training["lr_reduction_factor"]) < 1.0:
            raise ValueError("lr_reduction_factor must lie in (0, 1)")
        if float(training["min_policy_learning_rate"]) >= float(
            training["policy_learning_rate"]
        ):
            raise ValueError("minimum policy learning rate must be below its initial value")
    elif int(training.get("epochs", 0)) != 2:
        raise ValueError("direct-preference extension is frozen to two epochs")
    if str(training.get("compute_dtype")) != "bfloat16":
        raise ValueError("formal extension compute dtype must be bfloat16")
    aux = value["auxdpo"]
    if aux.get("reported_test_reward") != "policy_implied_implicit_reward":
        raise ValueError("AuxDPO test reward scope changed")
    if experiment_name in ADAPTIVE_EXPERIMENTS:
        minimum_aux = float(aux.get("min_auxiliary_learning_rate", math.nan))
        initial_aux = float(aux.get("auxiliary_learning_rate", math.nan))
        if not 0.0 < minimum_aux < initial_aux:
            raise ValueError("adaptive auxiliary learning-rate bounds are invalid")
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
    split_names = (
        ("train", "validation", "test")
        if extension["experiment"]["name"] in ADAPTIVE_EXPERIMENTS
        else ("train", "test")
    )
    for split in split_names:
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
    if extension["experiment"]["name"] in ADAPTIVE_EXPERIMENTS:
        return _train_adaptive_direct_preference(
            extension_path,
            artifact_dir,
            reference_dir,
            output_dir,
            seed=seed,
            beta=beta,
            method=method,
            device=device,
        )
    return _train_fixed_direct_preference(
        extension_path,
        artifact_dir,
        reference_dir,
        output_dir,
        seed=seed,
        beta=beta,
        method=method,
        device=device,
    )


def _train_fixed_direct_preference(
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


def _train_adaptive_direct_preference(
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
    """Fit one policy with validation-only scheduling and fail-closed convergence."""

    extension = load_direct_preference_config(extension_path)
    _, source_config = resolve_source_config(extension_path, extension)
    if method not in METHODS or beta not in tuple(extension["experiment"]["betas"]):
        raise ValueError("undeclared adaptive direct-preference condition")
    if seed not in extension["experiment"]["seeds"]:
        raise ValueError("undeclared adaptive seed")
    digest = extension_hash(extension)
    reference_metadata, reference = _load_reference(
        reference_dir, seed=seed, extension_digest=digest
    )
    if set(reference) != {"train", "validation", "test"}:
        raise ValueError("adaptive reference cache must contain train, validation, and test")
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
            and existing.get("extension_config_sha256") == digest
            and existing.get("training", {}).get("converged") is True
        ):
            return existing
        raise ValueError("existing adaptive direct-preference result identity mismatch")

    target.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(device)
    setup, pad_token_id = _model_and_setup(source_config, seed, target_device)
    if setup.a_state_sha256 != reference_metadata["lora_a_sha256"]:
        raise ValueError("training LoRA-A basis differs from the reference cache")
    setup.model.eval()

    experiment = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    prompt_limit = extension["training"].get("limit_prompts_per_split")
    train_groups = group_candidate_nodes(
        artifact_dir, split="train", candidates=experiment.train.num_candidates
    )
    validation_groups = group_candidate_nodes(
        artifact_dir, split="validation", candidates=experiment.train.num_candidates
    )
    if prompt_limit is not None:
        train_groups = train_groups[: int(prompt_limit)]
        validation_groups = validation_groups[: int(prompt_limit)]
    prompt_count = len(train_groups)
    validation_count = len(validation_groups)
    reference_train = reference["train"][:prompt_count].to(torch.float32)
    reference_validation = reference["validation"][:validation_count].to(torch.float32)
    true_train = experiment.train.true_rewards[:prompt_count].to(torch.float32)
    true_validation = experiment.validation.true_rewards[:validation_count].to(torch.float32)
    score_train = experiment.train.policy_scores[:prompt_count].to(torch.float32)

    training = extension["training"]
    batch_size = int(training["prompt_batch_size"])
    max_epochs = int(training["max_epochs"])
    min_epochs = int(training["min_epochs"])
    condition_seed = int(seed * 1000 + round(beta * 100))
    schedule: list[tuple[int, list[int]]] = []
    for epoch in range(max_epochs):
        epoch_order = list(range(prompt_count))
        random.Random(condition_seed + epoch).shuffle(epoch_order)
        schedule.extend(
            (epoch, epoch_order[start : start + batch_size])
            for start in range(0, prompt_count, batch_size)
        )
    schedule_digest = _canonical_sha256(schedule)
    batches_per_epoch = math.ceil(prompt_count / batch_size)
    warmup_steps = int(training["warmup_epochs"]) * batches_per_epoch

    policy_parameters = [parameter for _, parameter in setup.named_tangent_parameters()]
    policy_lr = float(training["policy_learning_rate"])
    parameter_groups: list[dict[str, Any]] = [{"params": policy_parameters, "lr": policy_lr}]
    initial_group_lrs = [policy_lr]
    minimum_group_lrs = [float(training["min_policy_learning_rate"])]
    delta_raw: torch.nn.Parameter | None = None
    if method == "auxdpo":
        delta_raw = torch.nn.Parameter(
            torch.zeros(
                (prompt_count, experiment.train.num_candidates),
                dtype=torch.float32,
                device=target_device,
            )
        )
        aux_lr = float(extension["auxdpo"]["auxiliary_learning_rate"])
        parameter_groups.append({"params": [delta_raw], "lr": aux_lr})
        initial_group_lrs.append(aux_lr)
        minimum_group_lrs.append(float(extension["auxdpo"]["min_auxiliary_learning_rate"]))

    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=tuple(float(item) for item in training["adam_betas"]),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["lr_reduction_factor"]),
        patience=int(training["lr_reduction_patience"]),
        threshold=float(training["validation_min_delta"]),
        threshold_mode="abs",
        min_lr=minimum_group_lrs,
    )

    @torch.no_grad()
    def heldout_metrics() -> tuple[float, float]:
        chunks: list[torch.Tensor] = []
        for start in range(0, validation_count, batch_size):
            chunks.append(
                response_log_probabilities(
                    setup.model,
                    validation_groups[start : start + batch_size],
                    pad_token_id=pad_token_id,
                    device=target_device,
                    compute_dtype=torch.bfloat16,
                )
                .detach()
                .cpu()
            )
        updated = torch.cat(chunks).to(torch.float32)
        implicit = beta * (updated - reference_validation)
        nll = float(soft_preference_loss(implicit, true_validation).item())
        mse = float(
            (centered(implicit.to(torch.float64)) - centered(true_validation.to(torch.float64)))
            .square()
            .mean()
            .item()
        )
        return nll, mse

    initial_validation_nll = float(
        soft_preference_loss(torch.zeros_like(true_validation), true_validation).item()
    )
    best_validation_nll = initial_validation_nll
    best_validation_mse = float(centered(true_validation.to(torch.float64)).square().mean().item())
    best_epoch = 0
    best_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in setup.named_tangent_parameters()
    }
    best_delta_raw = None if delta_raw is None else delta_raw.detach().cpu().clone()
    history: list[dict[str, Any]] = []
    bad_epochs = 0
    lr_reductions = 0
    start_batch = 0
    processed_batches = 0
    epoch_running = {
        "prompts": 0,
        "batches": 0,
        "loss_sum": 0.0,
        "preference_nll_sum": 0.0,
        "nullspace_penalty_sum": 0.0,
        "delta_amplitude_sum": 0.0,
        "policy_grad_norm_sum": 0.0,
        "policy_grad_norm_max": 0.0,
    }
    checkpoint_path = target / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != [digest, seed, beta, method]:
            raise ValueError("adaptive training checkpoint identity mismatch")
        if checkpoint.get("schedule_sha256") != schedule_digest:
            raise ValueError("adaptive training checkpoint schedule mismatch")
        by_name = dict(setup.named_tangent_parameters())
        for name, value in checkpoint["adapter"].items():
            by_name[name].data.copy_(value.to(target_device))
        if delta_raw is not None:
            delta_raw.data.copy_(checkpoint["delta_raw"].to(target_device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_batch = int(checkpoint["next_batch"])
        processed_batches = int(checkpoint["processed_batches"])
        history = list(checkpoint["history"])
        bad_epochs = int(checkpoint["bad_epochs"])
        lr_reductions = int(checkpoint["lr_reductions"])
        best_validation_nll = float(checkpoint["best_validation_nll"])
        best_validation_mse = float(checkpoint["best_validation_mse"])
        best_epoch = int(checkpoint["best_epoch"])
        best_adapter = checkpoint["best_adapter"]
        best_delta_raw = checkpoint["best_delta_raw"]
        epoch_running = dict(checkpoint["epoch_running"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(target_device)

    def save_checkpoint(next_batch: int) -> None:
        _atomic_torch(
            checkpoint_path,
            {
                "identity": [digest, seed, beta, method],
                "schedule_sha256": schedule_digest,
                "next_batch": next_batch,
                "processed_batches": processed_batches,
                "adapter": {
                    name: parameter.detach().cpu()
                    for name, parameter in setup.named_tangent_parameters()
                },
                "delta_raw": None if delta_raw is None else delta_raw.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "bad_epochs": bad_epochs,
                "lr_reductions": lr_reductions,
                "best_validation_nll": best_validation_nll,
                "best_validation_mse": best_validation_mse,
                "best_epoch": best_epoch,
                "best_adapter": best_adapter,
                "best_delta_raw": best_delta_raw,
                "epoch_running": epoch_running,
            },
        )

    checkpoint_every = int(training["checkpoint_prompt_batches"])
    stopped = False
    stop_reason: str | None = None
    last_next_batch = start_batch
    for batch_number, (epoch, indices) in enumerate(schedule):
        if batch_number < start_batch:
            continue
        if batch_number < warmup_steps:
            scale = float(batch_number + 1) / float(max(warmup_steps, 1))
            for group, initial_lr in zip(optimizer.param_groups, initial_group_lrs, strict=True):
                group["lr"] = initial_lr * scale

        nodes = tuple(train_groups[index] for index in indices)
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
            raise RuntimeError("adaptive direct-preference loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_parameters, max_norm=float(training["max_gradient_norm"])
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("adaptive policy gradient norm became non-finite")
        optimizer.step()

        batch_prompts = len(indices)
        processed_batches += 1
        epoch_running["prompts"] += batch_prompts
        epoch_running["batches"] += 1
        epoch_running["loss_sum"] += float(loss.detach().item()) * batch_prompts
        for key in ("preference_nll", "nullspace_penalty", "delta_amplitude"):
            epoch_running[f"{key}_sum"] += (
                float(diagnostics[key].detach().item()) * batch_prompts
            )
        grad_value = float(grad_norm.detach().item())
        epoch_running["policy_grad_norm_sum"] += grad_value
        epoch_running["policy_grad_norm_max"] = max(
            float(epoch_running["policy_grad_norm_max"]), grad_value
        )
        next_batch = batch_number + 1
        last_next_batch = next_batch
        epoch_complete = next_batch == len(schedule) or schedule[next_batch][0] != epoch
        if epoch_complete:
            validation_nll, validation_mse = heldout_metrics()
            min_delta = float(training["validation_min_delta"])
            improved = validation_nll < best_validation_nll - min_delta
            if improved:
                best_validation_nll = validation_nll
                best_validation_mse = validation_mse
                best_epoch = epoch + 1
                best_adapter = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in setup.named_tangent_parameters()
                }
                best_delta_raw = (
                    None if delta_raw is None else delta_raw.detach().cpu().clone()
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
            policy_lr_before = float(optimizer.param_groups[0]["lr"])
            scheduler.step(validation_nll)
            policy_lr_after = float(optimizer.param_groups[0]["lr"])
            if policy_lr_after < policy_lr_before * (1.0 - 1.0e-12):
                lr_reductions += 1
            prompts_seen = max(int(epoch_running["prompts"]), 1)
            batches_seen = max(int(epoch_running["batches"]), 1)
            record = {
                "epoch": epoch + 1,
                "train_online_loss": float(epoch_running["loss_sum"]) / prompts_seen,
                "train_online_preference_nll": (
                    float(epoch_running["preference_nll_sum"]) / prompts_seen
                ),
                "train_online_nullspace_penalty": (
                    float(epoch_running["nullspace_penalty_sum"]) / prompts_seen
                ),
                "train_online_delta_amplitude": (
                    float(epoch_running["delta_amplitude_sum"]) / prompts_seen
                ),
                "validation_policy_nll": validation_nll,
                "validation_policy_centered_reward_mse": validation_mse,
                "policy_grad_norm_mean": (
                    float(epoch_running["policy_grad_norm_sum"]) / batches_seen
                ),
                "policy_grad_norm_max": float(epoch_running["policy_grad_norm_max"]),
                "policy_lr_before_plateau_step": policy_lr_before,
                "policy_lr_after_plateau_step": policy_lr_after,
                "auxiliary_lr_after_plateau_step": (
                    None if delta_raw is None else float(optimizer.param_groups[1]["lr"])
                ),
                "improved_by_min_delta": improved,
                "bad_epochs": bad_epochs,
                "lr_reductions": lr_reductions,
            }
            history.append(record)
            print(
                f"adaptive-train seed={seed} beta={beta:g} method={method} "
                f"epoch={epoch + 1}/{max_epochs} train={record['train_online_loss']:.8f} "
                f"valid={validation_nll:.8f} lr={policy_lr_after:.3e} "
                f"best_epoch={best_epoch} bad={bad_epochs} reductions={lr_reductions}",
                flush=True,
            )
            epoch_running = {key: 0.0 for key in epoch_running}
            epoch_running["prompts"] = 0
            epoch_running["batches"] = 0
            improvement = initial_validation_nll - best_validation_nll
            if (
                epoch + 1 >= min_epochs
                and bad_epochs >= int(training["early_stopping_patience"])
                and lr_reductions >= int(training["minimum_lr_reductions"])
                and improvement >= float(training["minimum_validation_improvement"])
            ):
                stopped = True
                stop_reason = "validation_plateau_after_required_lr_reductions"
        if next_batch % checkpoint_every == 0 or epoch_complete or stopped:
            save_checkpoint(next_batch)
        if stopped:
            break

    is_smoke = extension["experiment"]["name"] == "dpo-auxdpo-converged-smoke-v1"
    improvement = initial_validation_nll - best_validation_nll
    if is_smoke:
        converged = True
        stop_reason = stop_reason or "bounded_resource_and_control_flow_smoke_complete"
    else:
        converged = bool(
            stopped
            and best_epoch > 0
            and improvement >= float(training["minimum_validation_improvement"])
        )
    if not converged:
        save_checkpoint(last_next_batch)
        raise RuntimeError(
            "adaptive direct-preference fit exhausted its budget without the preregistered "
            f"convergence gate: best_epoch={best_epoch}, improvement={improvement:.8g}, "
            f"bad_epochs={bad_epochs}, lr_reductions={lr_reductions}"
        )

    by_name = dict(setup.named_tangent_parameters())
    with torch.no_grad():
        for name, value in best_adapter.items():
            by_name[name].copy_(value.to(target_device))
        if delta_raw is not None and best_delta_raw is not None:
            delta_raw.copy_(best_delta_raw.to(target_device))

    adapter_path = target / "adapter.safetensors"
    _save_trainable_tensors(setup, adapter_path)
    delta_path: Path | None = None
    if delta_raw is not None:
        delta_path = target / "delta.safetensors"
        delta = centered(
            float(extension["auxdpo"]["delta_cap"]) * torch.tanh(delta_raw.detach())
        ).cpu().contiguous()
        safetensors = __import__("safetensors.torch", fromlist=["save_file"])
        temporary = target / ".delta.tmp.safetensors"
        safetensors.save_file({"train.delta": delta}, str(temporary))
        os.replace(temporary, delta_path)

    fitted_logps: dict[str, torch.Tensor] = {}
    split_data = {
        "train": experiment.train,
        "validation": experiment.validation,
        "test": experiment.test,
    }
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            split_groups = group_candidate_nodes(
                artifact_dir, split=split, candidates=experiment.train.num_candidates
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
            expected_shape = split_data[split].true_rewards[: len(split_groups)].shape
            if fitted_logps[split].shape != expected_shape:
                raise RuntimeError(f"final {split} log-probability shape changed")
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
            "initialization": "reference_policy_with_zero_lora_B",
            "optimizer_resume": "not_applicable_fresh_adaptive_trajectory",
            "converged": True,
            "stop_reason": stop_reason,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "initial_validation_policy_nll": initial_validation_nll,
            "best_validation_policy_nll": best_validation_nll,
            "best_validation_policy_centered_reward_mse": best_validation_mse,
            "validation_nll_improvement": improvement,
            "validation_selection_metric": training["validation_selection_metric"],
            "test_usage": training["test_usage"],
            "prompt_batch_size": batch_size,
            "prompt_batches_processed": processed_batches,
            "policy_learning_rate": policy_lr,
            "min_policy_learning_rate": float(training["min_policy_learning_rate"]),
            "learning_rate_schedule": training["learning_rate_schedule"],
            "lr_reductions": lr_reductions,
            "auxiliary_learning_rate": (
                None if method == "dpo" else float(extension["auxdpo"]["auxiliary_learning_rate"])
            ),
            "history": history,
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

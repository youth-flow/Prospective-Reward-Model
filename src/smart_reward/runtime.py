"""Shared model, dataset, serialization, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from .prompts import PromptRecord, load_multipref_parquet_snapshot, prepare_multipref_prompts


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return seed


def candidate_id(prompt_id: str, index: int) -> str:
    return f"{prompt_id}::candidate::{index}"


def require_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            f"optional dependency {name!r} is required; install prospective-reward-model[llm]"
        ) from error


def model_inputs(encoded: object, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(encoded, torch.Tensor):
        result = {"input_ids": encoded}
    elif isinstance(encoded, Mapping):
        result = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    else:
        raise TypeError("chat template output must be a tensor or tensor mapping")
    if "input_ids" not in result:
        raise ValueError("chat template output is missing input_ids")
    result.setdefault("attention_mask", torch.ones_like(result["input_ids"]))
    return {key: value.to(device) for key, value in result.items()}


def prompt_text(record: PromptRecord) -> str:
    if len(record.messages) != 1 or record.messages[0].role != "user":
        raise ValueError("MultiPref records must contain exactly one user message")
    return record.messages[0].content


@contextmanager
def fork_torch_seed(seed: int, device: torch.device) -> Iterator[None]:
    devices: list[int] = []
    if device.type == "cuda":
        devices = [torch.cuda.current_device() if device.index is None else device.index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        yield


def jsonl_sha256(records: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for record in records:
        line = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def producer_identity() -> dict[str, str]:
    result: dict[str, str] = {}
    for output, variable in (
        ("git_commit", "PRORM_GIT_COMMIT"),
        ("image_sha256", "PRORM_IMAGE_SHA256"),
        ("hf_inventory_sha256", "PRORM_HF_INVENTORY_SHA256"),
    ):
        value = os.environ.get(variable)
        if value is not None:
            if len(value) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{variable} must be a lowercase immutable digest")
            result[output] = value
    formal = bool(os.environ.get("SLURM_JOB_ID"))
    if formal and set(result) != {"git_commit", "image_sha256", "hf_inventory_sha256"}:
        raise ValueError("Slurm production requires Git, image, and HF inventory identities")
    return result


def decode_response(tokenizer: object, token_ids: torch.Tensor) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must expose decode")
    return str(decode(token_ids.tolist(), skip_special_tokens=True))


def load_prompts(
    datasets: Any,
    config: Mapping[str, Any],
    *,
    local_files_only: bool,
    text_filter: Any | None = None,
) -> list[PromptRecord]:
    data = config["data"]
    run = config["run"]
    if local_files_only:
        hub = require_module("huggingface_hub")
        try:
            snapshot = Path(
                hub.snapshot_download(
                    repo_id=data["prompt_dataset"],
                    repo_type="dataset",
                    revision=data["prompt_revision"],
                    cache_dir=os.environ.get("HF_HUB_CACHE"),
                    local_files_only=True,
                    token=False,
                )
            )
            rows = load_multipref_parquet_snapshot(
                datasets,
                snapshot,
                datasets_cache=os.environ.get("HF_DATASETS_CACHE"),
            )
        except (OSError, FileNotFoundError, RuntimeError) as error:
            raise RuntimeError(
                "pinned MultiPref snapshot is unavailable locally; run the HPC staging job"
            ) from error
    else:
        rows = datasets.load_dataset(
            data["prompt_dataset"],
            revision=data["prompt_revision"],
            split="train",
        )
    return prepare_multipref_prompts(
        rows,
        split_sizes=run["split_sizes"],
        seed=int(run["prompt_split_seed"]),
        text_filter=text_filter,
    )


def load_pretrained(
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
            raise RuntimeError(
                f"pinned {kind} snapshot is unavailable locally: {identifier}@{revision}"
            ) from error
        raise


def preflight_empty_directory(destination: Path) -> None:
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {destination}")


def reward_class_projection(
    reward_features: torch.Tensor,
    true_rewards: torch.Tensor,
) -> dict[str, float | str]:
    features = reward_features.detach().to(device="cpu", dtype=torch.float64)
    rewards = true_rewards.detach().to(device="cpu", dtype=torch.float64)
    centered_features = features - features.mean(dim=1, keepdim=True)
    centered_rewards = rewards - rewards.mean(dim=1, keepdim=True)
    design = centered_features.reshape(-1, centered_features.shape[-1])
    target = centered_rewards.reshape(-1)
    residual = design @ torch.linalg.lstsq(design, target).solution - target
    target_rms = float(torch.sqrt(target.square().mean()).item())
    residual_rmse = float(torch.sqrt(residual.square().mean()).item())
    relative = 0.0 if target_rms == 0.0 else residual_rmse / target_rms
    if not all(math.isfinite(value) for value in (target_rms, residual_rmse, relative)):
        raise FloatingPointError("reward-class projection is non-finite")
    return {
        "fit_split": "train",
        "centering": "per_prompt_candidate_mean",
        "solver": "float64_cpu_lstsq",
        "target_centered_rms": target_rms,
        "residual_rmse": residual_rmse,
        "relative_residual": relative,
    }


__all__ = [
    "candidate_id",
    "decode_response",
    "fork_torch_seed",
    "jsonl_sha256",
    "load_pretrained",
    "load_prompts",
    "model_inputs",
    "preflight_empty_directory",
    "producer_identity",
    "prompt_text",
    "require_module",
    "reward_class_projection",
    "sha256_file",
    "validate_seed",
]

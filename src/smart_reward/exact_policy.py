"""One-step common-beta NGD updates in the fixed LoRA-B tangent."""

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

from .artifacts import (
    exact_delta_artifact_metadata_sha256,
    load_exact_delta_artifact,
)
from .config import PROTOCOL, config_hash, validate_config
from .exact_run import load_exact_reward_comparison
from .hf import configure_fixed_a_lora
from .policy_update import set_tangent_update_
from .runtime import (
    fork_torch_seed,
    load_pretrained,
    producer_identity,
    require_module,
    sha256_file,
)
from .seeding import SeedBundle

SCHEMA = "prorm-ngd-adapters/v1"


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


@torch.no_grad()
def _zero_b(setup: Any) -> None:
    for _, parameter in setup.named_tangent_parameters():
        parameter.zero_()


def _load_policy(config: Mapping[str, Any], seed: int, device: torch.device, local: bool) -> Any:
    transformers = require_module("transformers")
    peft = require_module("peft")
    policy = config["policy"]
    seeds = SeedBundle.from_base_seed(seed)
    with fork_torch_seed(seeds.policy_lora_a, device):
        model = load_pretrained(
            transformers.AutoModelForCausalLM,
            policy["model"],
            policy["revision"],
            local_files_only=local,
            kind="policy model",
            torch_dtype=_dtype(policy["dtype"]),
        )
        lora = peft.LoraConfig(
            r=policy["lora_rank"],
            lora_alpha=policy["lora_alpha"],
            lora_dropout=policy["lora_dropout"],
            target_modules=list(policy["lora_modules"]),
            layers_to_transform=list(policy["lora_layers"]),
            bias="none",
            init_lora_weights=True,
            task_type="CAUSAL_LM",
        )
        setup = configure_fixed_a_lora(model, lora)
    setup.model.to(device).eval()
    return setup


def _method_directory(method: str, beta: float) -> str:
    beta_text = format(beta, "g").replace(".", "p")
    return f"{method}__beta_{beta_text}"


def export_exact_ngd_adapters(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    comparison_json: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the three train-fitted directions and export all beta-scaled adapters."""

    normalized = validate_config(config)
    if normalized["protocol"] != PROTOCOL or seed not in normalized["run"]["seeds"]:
        raise ValueError("protocol or seed mismatch")
    digest = config_hash(normalized)
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    _ = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    comparison = load_exact_reward_comparison(
        comparison_json,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    if comparison["artifact_metadata_sha256"] != artifact_identity:
        raise ValueError("reward comparison belongs to another artifact")
    comparison_identity = sha256_file(Path(comparison_json))
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    directions = {
        method: torch.tensor(
            values,
            device=target_device,
            dtype=torch.float64,
        )
        for method, values in comparison["policy_directions"].items()
    }
    setup = _load_policy(normalized, seed, target_device, local_files_only)
    evidence = json.loads((Path(artifact_dir) / "metadata.json").read_text(encoding="utf-8"))[
        "evidence"
    ]
    if setup.a_state_sha256 != evidence["policy_a_sha256"]:
        raise RuntimeError("reloaded fixed LoRA-A does not match materialization")
    if setup.layout.to_metadata() != evidence["policy_layout"]:
        raise RuntimeError("reloaded LoRA-B layout does not match materialization")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite adapter directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    adapters: dict[str, Any] = {}
    try:
        for method in ("mle_rm", "pro_rm", "oracle"):
            direction = directions[method]
            for beta_raw in normalized["policy_update"]["beta_grid"]:
                beta = float(beta_raw)
                _zero_b(setup)
                set_tangent_update_(
                    setup.named_tangent_parameters(),
                    setup.layout,
                    direction,
                    step_size=1.0 / beta,
                )
                directory = _method_directory(method, beta)
                setup.model.save_pretrained(staging / directory, safe_serialization=True)
                saved_files = {
                    path.relative_to(staging / directory).as_posix(): sha256_file(path)
                    for path in sorted((staging / directory).rglob("*"))
                    if path.is_file()
                }
                if not saved_files:
                    raise RuntimeError(f"adapter serialization produced no files: {directory}")
                adapters[directory] = {
                    "reward_source": method,
                    "beta": beta,
                    "step_scale": 1.0 / beta,
                    "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                    "files": saved_files,
                }
                print(f"adapter name={directory} status=checkpointed", flush=True)
        _zero_b(setup)
        metadata = {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "config_sha256": digest,
            "artifact_metadata_sha256": artifact_identity,
            "reward_result_sha256": comparison_identity,
            "seed": seed,
            "beta_grid": [float(value) for value in normalized["policy_update"]["beta_grid"]],
            "policy_families": ["pi0", "mle_ngd", "pro_ngd", "oracle_ngd"],
            "updated_adapter_count": len(adapters),
            "lora_a_sha256": setup.a_state_sha256,
            "lora_layout": setup.layout.to_metadata(),
            "directions": {
                method: {
                    "fit_split": "train",
                    "norm": float(torch.linalg.vector_norm(value).item()),
                }
                for method, value in directions.items()
            },
            "reward_heads": {
                method: comparison["methods"][serialized]["head_sha256"]
                for method, serialized in (("mle_rm", "MLE-RM"), ("pro_rm", "Pro-RM"))
            },
            "adapters": adapters,
            "producer": producer_identity(),
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    finally:
        _zero_b(setup)
        if staging.exists():
            shutil.rmtree(staging)
        del setup
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return metadata


__all__ = ["export_exact_ngd_adapters"]

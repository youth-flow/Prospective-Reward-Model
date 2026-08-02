"""Auditable fresh-rollout evaluation for trained DPO and AuxDPO policies.

This protocol deliberately reuses only the frozen training artifacts.  It
loads the learned LoRA-B tensors into the policy, serializes a real PEFT
adapter, generates six fresh responses on every fixed test prompt, and reports
the same oracle-reward, forward-KL, and regularized-utility estimands as the
four-policy beta=0.2 experiment.
"""

from __future__ import annotations

import gc
import hashlib
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
import yaml

from .artifacts import exact_delta_artifact_metadata_sha256, load_exact_delta_artifact
from .config import config_hash, load_config
from .direct_preference import (
    FIT_SCHEMA,
    _fit_directory,
    _load_trainable_tensors,
    _model_and_setup,
    centered,
    load_direct_preference_config,
    soft_preference_loss,
)
from .direct_preference import (
    extension_hash as direct_config_hash,
)
from .exact import empirical_fisher_score_rows, policy_reward_moment
from .linear import DampedEmpiricalFisher
from .pcg import pcg
from .real_policy_evaluation import _policy_metrics
from .rollout import _generate_policy_batch, _load_models, _test_prompts
from .runtime import producer_identity, sha256_file
from .seeding import SeedBundle, derive_seed

CONFIG_SCHEMA = "prorm-real-policy-dpo-aux-m6-config/v1"
ADAPTER_SCHEMA = "prorm-direct-policy-adapter/v1"
ROLLOUT_SCHEMA = "prorm-direct-policy-rollout-m6/v1"
SEED_SCHEMA = "prorm-six-policy-evaluation-m6/v1"
AGGREGATE_SCHEMA = "prorm-six-policy-aggregate-m6/v1"
AUDIT_SCHEMA = "prorm-six-policy-audit-m6/v1"
PROTOCOL = "prorm-real-policy-dpo-aux-beta0p2-m6/v1"
METHODS = ("dpo", "auxdpo")
BETA = 0.2
PROMPTS = 512
BASE_RESPONSES = 4
ADDITIONAL_RESPONSES = 2
TOTAL_RESPONSES = 6


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise ValueError(f"refusing to replace non-identical output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def policy_name(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown direct policy method: {method}")
    return f"{method}__beta_0p2"


def load_direct_policy_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported direct-policy rollout config")
    if set(value) != {
        "schema",
        "source_config",
        "source_config_sha256",
        "direct_config",
        "direct_config_sha256",
        "experiment",
        "rollout",
        "reward_evaluation",
    }:
        raise ValueError("direct-policy rollout config keys changed")
    experiment = value["experiment"]
    if experiment != {
        "name": "real-policy-dpo-aux-beta0p2-m6-v1",
        "seeds": [20261001, 20261002, 20261003],
        "beta": BETA,
        "source_policies": ["pi0", "mle_rm__beta_0p2", "pro_rm__beta_0p2", "oracle__beta_0p2"],
        "new_policies": [policy_name(method) for method in METHODS],
    }:
        raise ValueError("formal six-policy experiment identity changed")
    rollout = value["rollout"]
    if rollout != {
        "prompts": PROMPTS,
        "base_responses_per_prompt": BASE_RESPONSES,
        "additional_responses_per_prompt": ADDITIONAL_RESPONSES,
        "responses_per_prompt": TOTAL_RESPONSES,
        "base_seed_namespace": "real-rollout-batch",
        "additional_seed_namespace": "real-rollout-extension-4-to-6-batch",
        "generation": "fresh_test_prompt_rollout",
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
        "metrics": ["R", "K", "J"],
    }:
        raise ValueError("fresh rollout estimand changed")
    reward = value["reward_evaluation"]
    if reward != {
        "metrics": ["NLL", "MSE", "approximate_regret"],
        "approximate_regret_estimator": "two_fold_cross_U",
        "fisher_source": "train_raw_second_moment_with_frozen_selected_damping",
        "folds": 2,
        "fold_seed_namespace": "approx-regret-cross-v1:",
        "test_usage": "evaluation_only_no_selection",
    }:
        raise ValueError("reward evaluation estimand changed")
    return value


def resolve_configs(
    path: str | os.PathLike[str], value: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(path).resolve().parent
    source = load_config(root / str(value["source_config"]))
    direct = load_direct_preference_config(root / str(value["direct_config"]))
    if config_hash(source) != value["source_config_sha256"]:
        raise ValueError("source config digest mismatch")
    if direct_config_hash(direct) != value["direct_config_sha256"]:
        raise ValueError("DPO/AuxDPO config digest mismatch")
    if direct["source_config_sha256"] != value["source_config_sha256"]:
        raise ValueError("DPO/AuxDPO and rollout source configurations differ")
    if BETA not in tuple(float(item) for item in direct["experiment"]["betas"]):
        raise ValueError("DPO/AuxDPO source did not declare beta=0.2")
    return source, direct


def _validate_fit(
    config_path: str | os.PathLike[str],
    artifact_dir: Path,
    fit_dir: Path,
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    _, direct = resolve_configs(config_path, config)
    fit = _read_json(fit_dir / "result.json")
    adapter = fit_dir / "adapter.safetensors"
    artifact_sha = exact_delta_artifact_metadata_sha256(
        artifact_dir, expected_config_hash=config["source_config_sha256"], expected_seed=seed
    )
    expected = {
        "schema": FIT_SCHEMA,
        "status": "complete",
        "seed": seed,
        "method": method,
        "beta": BETA,
        "source_config_sha256": config["source_config_sha256"],
        "extension_config_sha256": direct_config_hash(direct),
        "artifact_metadata_sha256": artifact_sha,
    }
    if any(fit.get(key) != item for key, item in expected.items()):
        raise ValueError(f"DPO/AuxDPO fit identity mismatch: {fit_dir}")
    if fit.get("files", {}).get("adapter.safetensors") != sha256_file(adapter):
        raise ValueError("trained LoRA-B digest mismatch")
    artifact = _read_json(artifact_dir / "metadata.json")
    if fit.get("lora_a_sha256") != artifact["evidence"]["policy_a_sha256"]:
        raise ValueError("trained adapter and materialized LoRA-A basis differ")
    return fit


def export_direct_policy_adapter(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    fit_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    source, _ = resolve_configs(config_path, config)
    artifact_path, fit_path, target = Path(artifact_dir), Path(fit_dir), Path(output_dir)
    fit = _validate_fit(config_path, artifact_path, fit_path, seed=seed, method=method)
    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        return validate_direct_policy_adapter(
            config_path, artifact_path, fit_path, target, seed=seed, method=method
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    setup, _ = _model_and_setup(source, seed, target_device)
    try:
        if setup.a_state_sha256 != fit["lora_a_sha256"]:
            raise RuntimeError("reloaded LoRA-A basis differs from trained adapter")
        raw_path = fit_path / "adapter.safetensors"
        _load_trainable_tensors(setup, raw_path)
        safetensors = __import__("safetensors.torch", fromlist=["load_file"])
        intended = safetensors.load_file(str(raw_path), device="cpu")
        update_l2 = math.sqrt(
            sum(
                float(tensor.to(torch.float64).square().sum().item())
                for tensor in intended.values()
            )
        )
        if not math.isfinite(update_l2) or update_l2 <= 0.0:
            raise RuntimeError("trained direct-policy adapter is identically zero")
        errors = [
            float((parameter.detach().cpu().float() - intended[name].float()).abs().max().item())
            for name, parameter in setup.named_tangent_parameters()
        ]
        max_error = max(errors, default=0.0)
        if max_error > 1.0e-7:
            raise RuntimeError(f"trained LoRA-B writeback mismatch: {max_error:.3e}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            setup.model.save_pretrained(staging, safe_serialization=True)
            files = {
                p.relative_to(staging).as_posix(): sha256_file(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            }
            if not files:
                raise RuntimeError("PEFT adapter serialization produced no files")
            metadata = {
                "schema": ADAPTER_SCHEMA,
                "protocol": PROTOCOL,
                "seed": seed,
                "method": method,
                "policy_name": policy_name(method),
                "beta": BETA,
                "source_config_sha256": config["source_config_sha256"],
                "direct_config_sha256": config["direct_config_sha256"],
                "artifact_metadata_sha256": fit["artifact_metadata_sha256"],
                "source_fit_result_sha256": sha256_file(fit_path / "result.json"),
                "source_raw_adapter_sha256": sha256_file(raw_path),
                "lora_a_sha256": fit["lora_a_sha256"],
                "update_l2": update_l2,
                "writeback_max_abs_error": max_error,
                "files": files,
                "producer": producer_identity(),
            }
            _atomic_json(staging / "metadata.json", metadata)
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    finally:
        del setup
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return validate_direct_policy_adapter(
        config_path, artifact_path, fit_path, target, seed=seed, method=method
    )


def validate_direct_policy_adapter(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    fit_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    fit_path, target = Path(fit_dir), Path(adapter_dir)
    fit = _validate_fit(config_path, Path(artifact_dir), fit_path, seed=seed, method=method)
    metadata = _read_json(target / "metadata.json")
    checks = {
        "schema": ADAPTER_SCHEMA,
        "protocol": PROTOCOL,
        "seed": seed,
        "method": method,
        "policy_name": policy_name(method),
        "beta": BETA,
        "source_fit_result_sha256": sha256_file(fit_path / "result.json"),
        "source_raw_adapter_sha256": sha256_file(fit_path / "adapter.safetensors"),
        "lora_a_sha256": fit["lora_a_sha256"],
    }
    if any(metadata.get(key) != value for key, value in checks.items()):
        raise ValueError("serialized direct-policy adapter metadata mismatch")
    if float(metadata.get("writeback_max_abs_error", math.inf)) > 1.0e-7:
        raise ValueError("serialized direct-policy adapter failed writeback gate")
    if (
        not math.isfinite(float(metadata.get("update_l2", math.nan)))
        or float(metadata["update_l2"]) <= 0.0
    ):
        raise ValueError("serialized direct-policy adapter is identically zero")
    for relative, digest in metadata.get("files", {}).items():
        path = target / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"serialized adapter file digest mismatch: {relative}")
    return metadata


def _validate_rows(rows: Sequence[Mapping[str, Any]], prompt_ids: Sequence[str], name: str) -> None:
    expected = [(prompt_id, index) for prompt_id in prompt_ids for index in range(TOTAL_RESPONSES)]
    observed = [(str(row.get("prompt_id")), int(row.get("response_index", -1))) for row in rows]
    if observed != expected:
        raise ValueError("direct-policy rollout rows are not in canonical m=6 order")
    for row in rows:
        if row.get("policy_instance") != name or not isinstance(row.get("response"), str):
            raise ValueError("direct-policy rollout row identity mismatch")
        for key in ("oracle_reward", "forward_kl"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"invalid rollout {key}")
        if float(row["forward_kl"]) < -1.0e-7:
            raise ValueError("Rao-Blackwellized forward KL is materially negative")


def run_direct_policy_rollout(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    fit_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    source, _ = resolve_configs(config_path, config)
    artifact_path, target = Path(artifact_dir), Path(output_dir)
    adapter = validate_direct_policy_adapter(
        config_path, artifact_path, fit_dir, adapter_dir, seed=seed, method=method
    )
    if target.exists():
        return validate_direct_policy_rollout(
            config_path, artifact_path, fit_dir, adapter_dir, target, seed=seed, method=method
        )[0]
    prompts = _test_prompts(source, artifact_path)
    if len(prompts) != PROMPTS:
        raise ValueError("fixed test prompt count changed")
    transform = _read_json(artifact_path / "metadata.json")["evidence"]["oracle_transform"]
    target_device = torch.device(device)
    name = policy_name(method)
    model, tokenizer, oracle_model, oracle_tokenizer = _load_models(
        source,
        Path(adapter_dir).parent,
        adapter_name=name,
        device=target_device,
        local_files_only=True,
    )
    model.set_adapter(name)
    prompt_batch = int(source["execution"]["rollout_prompt_batch_size"])
    checkpoint_prompts = int(source["execution"]["rollout_checkpoint_prompts"])
    base_seed = SeedBundle.from_base_seed(seed).rollout
    work = target.parent / f".{target.name}.work"
    manifest = {
        "schema": "prorm-direct-policy-rollout-work/v1",
        "protocol": PROTOCOL,
        "config_sha256": _canonical_sha256(config),
        "seed": seed,
        "method": method,
        "policy_name": name,
        "artifact_metadata_sha256": adapter["artifact_metadata_sha256"],
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
    }
    if (work / "manifest.json").exists():
        if _read_json(work / "manifest.json") != manifest:
            raise ValueError("direct-policy rollout work identity mismatch")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(f"unidentified rollout work directory: {work}")
        _atomic_json(work / "manifest.json", manifest)
    try:
        for start in range(0, len(prompts), checkpoint_prompts):
            stop = min(start + checkpoint_prompts, len(prompts))
            shard_path = work / "shards" / f"{start:06d}-{stop:06d}.json"
            if shard_path.exists():
                shard = _read_json(shard_path)
                if (
                    shard.get("manifest") != manifest
                    or shard.get("start") != start
                    or shard.get("stop") != stop
                ):
                    raise ValueError("direct-policy rollout shard identity mismatch")
                continue
            rows: list[dict[str, Any]] = []
            for batch_start in range(start, stop, prompt_batch):
                batch_stop = min(batch_start + prompt_batch, stop)
                batch_prompts = prompts[batch_start:batch_stop]
                common = dict(
                    device=target_device,
                    reference=False,
                    rao_blackwellized_kl=True,
                    oracle_center=float(transform["b"]),
                    oracle_scale=float(transform["tau"]),
                    policy_config=source["policy"],
                )
                base = _generate_policy_batch(
                    model,
                    tokenizer,
                    oracle_model,
                    oracle_tokenizer,
                    batch_prompts,
                    responses=BASE_RESPONSES,
                    generation_seed=derive_seed(base_seed, f"real-rollout-batch:{batch_start}"),
                    **common,
                )
                extra = _generate_policy_batch(
                    model,
                    tokenizer,
                    oracle_model,
                    oracle_tokenizer,
                    batch_prompts,
                    responses=ADDITIONAL_RESPONSES,
                    generation_seed=derive_seed(
                        base_seed, f"real-rollout-extension-4-to-6-batch:{batch_start}"
                    ),
                    **common,
                )
                for prompt_offset in range(len(batch_prompts)):
                    rows.extend(
                        {**row, "policy_instance": name, "policy_method": method, "beta": BETA}
                        for row in base[
                            prompt_offset * BASE_RESPONSES : (prompt_offset + 1) * BASE_RESPONSES
                        ]
                    )
                    rows.extend(
                        {
                            **row,
                            "response_index": int(row["response_index"]) + BASE_RESPONSES,
                            "policy_instance": name,
                            "policy_method": method,
                            "beta": BETA,
                        }
                        for row in extra[
                            prompt_offset * ADDITIONAL_RESPONSES : (prompt_offset + 1)
                            * ADDITIONAL_RESPONSES
                        ]
                    )
            _atomic_json(
                shard_path,
                {
                    "schema": "prorm-direct-policy-rollout-shard/v1",
                    "manifest": manifest,
                    "start": start,
                    "stop": stop,
                    "rows": rows,
                },
            )
            print(
                f"rollout policy={name} prompts={stop}/{len(prompts)} status=checkpointed",
                flush=True,
            )
    finally:
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for start in range(0, len(prompts), checkpoint_prompts):
        stop = min(start + checkpoint_prompts, len(prompts))
        rows.extend(_read_json(work / "shards" / f"{start:06d}-{stop:06d}.json")["rows"])
    _validate_rows(rows, [prompt.prompt_id for prompt in prompts], name)
    metadata = {
        "schema": ROLLOUT_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": _canonical_sha256(config),
        "seed": seed,
        "method": method,
        "policy_name": name,
        "beta": BETA,
        "prompt_count": PROMPTS,
        "responses_per_prompt": TOTAL_RESPONSES,
        "generation": "fresh_test_prompt_rollout",
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
        "artifact_metadata_sha256": adapter["artifact_metadata_sha256"],
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "source_fit_result_sha256": adapter["source_fit_result_sha256"],
        "producer": producer_identity(),
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.final-", dir=target.parent))
    try:
        _atomic_json(staging / "metadata.json", metadata)
        rows_path = staging / "rollouts.jsonl"
        with rows_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "schema": "prorm-direct-policy-rollout-receipt/v1",
            "metadata_sha256": sha256_file(staging / "metadata.json"),
            "rollouts_sha256": sha256_file(rows_path),
            "row_count": len(rows),
            "producer": producer_identity(),
        }
        _atomic_json(staging / "receipt.json", receipt)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    shutil.rmtree(work)
    return validate_direct_policy_rollout(
        config_path, artifact_path, fit_dir, adapter_dir, target, seed=seed, method=method
    )[0]


def smoke_direct_policy(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    fit_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    source, _ = resolve_configs(config_path, config)
    adapter = validate_direct_policy_adapter(
        config_path, artifact_dir, fit_dir, adapter_dir, seed=seed, method=method
    )
    target_device = torch.device(device)
    name = policy_name(method)
    prompts = _test_prompts(source, Path(artifact_dir))
    transform = _read_json(Path(artifact_dir) / "metadata.json")["evidence"]["oracle_transform"]
    model, tokenizer, oracle_model, oracle_tokenizer = _load_models(
        source,
        Path(adapter_dir).parent,
        adapter_name=name,
        device=target_device,
        local_files_only=True,
    )
    model.set_adapter(name)
    try:
        rows = _generate_policy_batch(
            model,
            tokenizer,
            oracle_model,
            oracle_tokenizer,
            prompts[:1],
            responses=1,
            generation_seed=derive_seed(
                SeedBundle.from_base_seed(seed).rollout, f"direct-policy-smoke:{method}"
            ),
            device=target_device,
            reference=False,
            rao_blackwellized_kl=True,
            oracle_center=float(transform["b"]),
            oracle_scale=float(transform["tau"]),
            policy_config=source["policy"],
        )
    finally:
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    if (
        len(rows) != 1
        or not rows[0]["response"]
        or not all(math.isfinite(float(rows[0][key])) for key in ("oracle_reward", "forward_kl"))
        or float(rows[0]["forward_kl"]) < -1.0e-7
    ):
        raise RuntimeError("direct-policy smoke failed generation/evaluation gate")
    payload = {
        "schema": "prorm-direct-policy-smoke/v1",
        "status": "passed",
        "seed": seed,
        "method": method,
        "policy_name": name,
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "source_raw_adapter_sha256": adapter["source_raw_adapter_sha256"],
        "response_nonempty": True,
        "oracle_reward": float(rows[0]["oracle_reward"]),
        "forward_kl": float(rows[0]["forward_kl"]),
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def validate_direct_policy_rollout(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    fit_dir: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    rollout_dir: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_direct_policy_config(config_path)
    source, _ = resolve_configs(config_path, config)
    adapter = validate_direct_policy_adapter(
        config_path, artifact_dir, fit_dir, adapter_dir, seed=seed, method=method
    )
    target = Path(rollout_dir)
    metadata = _read_json(target / "metadata.json")
    expected = {
        "schema": ROLLOUT_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": _canonical_sha256(config),
        "seed": seed,
        "method": method,
        "policy_name": policy_name(method),
        "beta": BETA,
        "prompt_count": PROMPTS,
        "responses_per_prompt": TOTAL_RESPONSES,
        "generation": "fresh_test_prompt_rollout",
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
        "artifact_metadata_sha256": adapter["artifact_metadata_sha256"],
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "source_fit_result_sha256": adapter["source_fit_result_sha256"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("direct-policy rollout metadata mismatch")
    receipt = _read_json(target / "receipt.json")
    if receipt.get("metadata_sha256") != sha256_file(target / "metadata.json") or receipt.get(
        "rollouts_sha256"
    ) != sha256_file(target / "rollouts.jsonl"):
        raise ValueError("direct-policy rollout receipt mismatch")
    with (target / "rollouts.jsonl").open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    prompts = _test_prompts(source, Path(artifact_dir))
    _validate_rows(rows, [prompt.prompt_id for prompt in prompts], policy_name(method))
    if receipt.get("row_count") != PROMPTS * TOTAL_RESPONSES:
        raise ValueError("direct-policy rollout row count mismatch")
    return metadata, rows


def _cross_u_regret(
    train_scores: torch.Tensor,
    test_scores: torch.Tensor,
    reward_error: torch.Tensor,
    prompt_ids: Sequence[str],
    *,
    relative_damping: float,
    geometry: Mapping[str, Any],
) -> dict[str, float]:
    rows = empirical_fisher_score_rows(train_scores.to(torch.float64), "raw_second_moment")
    raw = DampedEmpiricalFisher(rows, damping=0.0)
    damping = relative_damping * float(raw.diagonal().mean().item())
    fisher = DampedEmpiricalFisher(rows, damping=damping)
    order = sorted(
        range(len(prompt_ids)),
        key=lambda index: (
            hashlib.sha256(("approx-regret-cross-v1:" + prompt_ids[index]).encode()).digest(),
            prompt_ids[index],
        ),
    )
    fold_indices = [order[::2], order[1::2]]
    moments = [
        policy_reward_moment(
            test_scores[indices].to(torch.float64), reward_error[indices].to(torch.float64)
        )
        for indices in fold_indices
    ]
    solves = [
        pcg(
            fisher.matvec,
            moment,
            inverse_diagonal=fisher.pcg_inverse_diagonal(),
            max_iterations=int(geometry["cg_max_iterations"]),
            tolerance=float(geometry["cg_tolerance"]),
            residual_recompute_interval=int(geometry["residual_recompute_interval"]),
        )
        for moment in moments
    ]
    if not all(result.converged for result in solves):
        raise RuntimeError("two-fold cross-U Fisher solve did not converge")
    cross = 0.5 * (
        torch.dot(moments[0], solves[1].solution) + torch.dot(moments[1], solves[0].solution)
    )
    return {
        "approximate_regret": float(cross.item()) / (2.0 * BETA),
        "cross_moment_inverse_fisher_quadratic": float(cross.item()),
        "folds": 2,
        "damping": damping,
        "max_pcg_relative_residual": max(result.relative_residual for result in solves),
    }


def direct_reward_metrics(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    fit_dir: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    source, _ = resolve_configs(config_path, config)
    fit = _validate_fit(config_path, Path(artifact_dir), Path(fit_dir), seed=seed, method=method)
    safetensors = __import__("safetensors.torch", fromlist=["load_file"])
    updated = safetensors.load_file(str(Path(fit_dir) / "updated_logps.safetensors"), device="cpu")
    if fit.get("files", {}).get("updated_logps.safetensors") != sha256_file(
        Path(fit_dir) / "updated_logps.safetensors"
    ):
        raise ValueError("updated log-probability digest mismatch")
    reference_metadata_sha = fit["reference_metadata_sha256"]
    # The immutable fit stores the implicit reward through updated-reference logps;
    # the sibling reference cache is validated by its digest before use.
    reference_dir = Path(fit_dir).parents[1] / "reference"
    metadata = _read_json(reference_dir / "metadata.json")
    if sha256_file(reference_dir / "metadata.json") != reference_metadata_sha:
        raise ValueError("reference log-probability metadata digest mismatch")
    reference = safetensors.load_file(
        str(reference_dir / "reference_logps.safetensors"), device="cpu"
    )
    if metadata.get("tensors_sha256") != sha256_file(reference_dir / "reference_logps.safetensors"):
        raise ValueError("reference log-probability tensor digest mismatch")
    experiment = load_exact_delta_artifact(
        artifact_dir, expected_config_hash=config["source_config_sha256"], expected_seed=seed
    )
    predicted = centered(
        BETA * (updated["test"].to(torch.float64) - reference["test"].to(torch.float64))
    )
    target = centered(experiment.test.true_rewards.to(torch.float64))
    reward_source = _read_json(Path(reward_result))
    cross = _cross_u_regret(
        experiment.train.policy_scores,
        experiment.test.policy_scores,
        predicted - target,
        experiment.test.prompt_ids,
        relative_damping=float(reward_source["selected_relative_damping"]),
        geometry=source["geometry"],
    )
    return {
        "NLL": float(soft_preference_loss(predicted, target).item()),
        "MSE": float((predicted - target).square().mean().item()),
        **cross,
        "scope": "policy_implied_implicit_reward",
        "test_usage": "evaluation_only_no_selection",
    }


def assemble_six_policy_seed(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    direct_seed_root: str | os.PathLike[str],
    source_evaluation: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    source_eval_path = Path(source_evaluation)
    source_eval = _read_json(source_eval_path)
    source_names = config["experiment"]["source_policies"]
    if (
        source_eval.get("seed") != seed
        or source_eval.get("beta") != BETA
        or source_eval.get("responses_per_prompt") != TOTAL_RESPONSES
        or set(source_eval.get("policies", {})) != set(source_names)
    ):
        raise ValueError("immutable four-policy m=6 source evaluation mismatch")
    policies = dict(source_eval["policies"])
    rewards = {}
    receipts = {}
    root = Path(direct_seed_root)
    for method in METHODS:
        fit_dir = root / "fits" / _fit_directory(method, BETA)
        adapter_dir = root / "adapters" / policy_name(method)
        rollout_dir = root / "policy_rollouts" / policy_name(method)
        _, rows = validate_direct_policy_rollout(
            config_path, artifact_dir, fit_dir, adapter_dir, rollout_dir, seed=seed, method=method
        )
        policies[policy_name(method)] = _policy_metrics(rows)
        rewards[method] = direct_reward_metrics(
            config_path, artifact_dir, reward_result, fit_dir, seed=seed, method=method
        )
        receipts[policy_name(method)] = sha256_file(rollout_dir / "receipt.json")
    for name, metrics in policies.items():
        if (
            metrics["K"] < -1.0e-7
            or abs(metrics["J"] - (metrics["R"] - BETA * metrics["K"])) > 1.0e-12
        ):
            raise RuntimeError(f"policy metric identity failed: {name}")
    payload = {
        "schema": SEED_SCHEMA,
        "protocol": PROTOCOL,
        "seed": seed,
        "beta": BETA,
        "prompt_count": PROMPTS,
        "responses_per_prompt": TOTAL_RESPONSES,
        "policies": policies,
        "direct_reward": rewards,
        "source_four_policy_evaluation_sha256": sha256_file(source_eval_path),
        "direct_policy_receipt_sha256": receipts,
        "definitions": {
            "R": "mean oracle reward on fresh fixed-test responses",
            "K": "mean Rao-Blackwellized forward KL(pi||pi0) on pi trajectories",
            "J": "R - beta*K",
        },
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def aggregate_six_policy(
    config_path: str | os.PathLike[str],
    results: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    expected_seeds = config["experiment"]["seeds"]
    records = [(_read_json(Path(path)), Path(path)) for path in results]
    records.sort(key=lambda item: expected_seeds.index(item[0].get("seed")))
    if [item[0].get("seed") for item in records] != expected_seeds:
        raise ValueError("six-policy aggregate requires exactly the three frozen seeds")
    names = config["experiment"]["source_policies"] + config["experiment"]["new_policies"]
    for record, _ in records:
        if (
            record.get("schema") != SEED_SCHEMA
            or record.get("protocol") != PROTOCOL
            or record.get("beta") != BETA
            or record.get("responses_per_prompt") != TOTAL_RESPONSES
            or set(record.get("policies", {})) != set(names)
            or set(record.get("direct_reward", {})) != set(METHODS)
        ):
            raise ValueError("six-policy seed evaluation identity mismatch")
    summary = {
        name: {
            metric: {
                "mean": statistics.fmean(
                    float(record["policies"][name][metric]) for record, _ in records
                ),
                "sample_sd": statistics.stdev(
                    float(record["policies"][name][metric]) for record, _ in records
                ),
                "seed_values": [float(record["policies"][name][metric]) for record, _ in records],
            }
            for metric in ("R", "K", "J")
        }
        for name in names
    }
    reward = {
        method: {
            metric: {
                "mean": statistics.fmean(
                    float(record["direct_reward"][method][metric]) for record, _ in records
                ),
                "sample_sd": statistics.stdev(
                    float(record["direct_reward"][method][metric]) for record, _ in records
                ),
                "seed_values": [
                    float(record["direct_reward"][method][metric]) for record, _ in records
                ],
            }
            for metric in ("NLL", "MSE", "approximate_regret")
        }
        for method in METHODS
    }
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "seeds": expected_seeds,
        "beta": BETA,
        "prompt_count": PROMPTS,
        "responses_per_prompt": TOTAL_RESPONSES,
        "policies": summary,
        "direct_reward": reward,
        "inputs": {str(record["seed"]): sha256_file(path) for record, path in records},
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def audit_six_policy_run(
    config_path: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    config = load_direct_policy_config(config_path)
    root = Path(run_root)
    results = [root / f"seed-{seed}" / "evaluation.json" for seed in config["experiment"]["seeds"]]
    with tempfile.TemporaryDirectory(prefix="prorm-six-policy-audit-") as temporary:
        recomputed = aggregate_six_policy(config_path, results, Path(temporary) / "aggregate.json")
    if _read_json(root / "aggregate.json") != recomputed:
        raise ValueError("stored six-policy aggregate differs from recomputation")
    checks = []
    for seed, result_path in zip(config["experiment"]["seeds"], results, strict=True):
        checks.append(
            {
                "seed": seed,
                "status": "passed",
                "evaluation_sha256": sha256_file(result_path),
                "new_policy_count": 2,
                "new_rows": 2 * PROMPTS * TOTAL_RESPONSES,
                "source_policy_count": 4,
                "source_rows_referenced": 4 * PROMPTS * TOTAL_RESPONSES,
            }
        )
    payload = {
        "schema": AUDIT_SCHEMA,
        "protocol": PROTOCOL,
        "status": "passed",
        "aggregate_sha256": sha256_file(root / "aggregate.json"),
        "checks": checks,
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


__all__ = [
    "aggregate_six_policy",
    "assemble_six_policy_seed",
    "audit_six_policy_run",
    "direct_reward_metrics",
    "export_direct_policy_adapter",
    "load_direct_policy_config",
    "policy_name",
    "run_direct_policy_rollout",
    "smoke_direct_policy",
    "validate_direct_policy_adapter",
    "validate_direct_policy_rollout",
]

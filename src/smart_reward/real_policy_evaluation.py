"""Auditable fixed-beta NGD writeback and fresh test-policy rollouts.

This extension deliberately starts at the first affected stage.  It reuses the
validated Fisher-TRPO materialization and beta-free natural directions, writes
``direction / beta`` into the fixed LoRA-B tangent, and then samples new model
responses on the frozen test prompts.  It never evaluates a frozen candidate
pool and exposes no tabular quantities.
"""

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

from .artifacts import exact_delta_artifact_metadata_sha256, load_exact_delta_artifact
from .checkpoints import validate_stage_receipt, write_stage_receipt
from .config import TRPO_PROTOCOL, config_hash, validate_config
from .exact_policy import (
    _atomic_json,
    _canonical_sha256,
    _load_policy,
    _quarantine_adapter_component,
    _zero_b,
)
from .policy_update import set_tangent_update_
from .rollout import _generate_policy_batch, _load_models, _read_json, _test_prompts
from .runtime import producer_identity, sha256_file, validate_seed
from .seeding import SeedBundle, derive_seed
from .trpo_run import load_trpo_reward_comparison

PROTOCOL = "prorm-real-policy-common-beta-ngd-v1"
ADAPTER_SCHEMA = "prorm-real-policy-ngd-adapters/v1"
ADAPTER_COMPONENT_SCHEMA = "prorm-real-policy-ngd-adapter-component/v1"
POLICY_SCHEMA = "prorm-real-policy-rollout/v1"
SHARD_SCHEMA = "prorm-real-policy-rollout-shard/v1"
SEED_SCHEMA = "prorm-real-policy-evaluation/v1"
AGGREGATE_SCHEMA = "prorm-real-policy-aggregate/v1"
AUDIT_SCHEMA = "prorm-real-policy-audit/v1"

BETA = 0.2
METHODS = ("mle_rm", "pro_rm", "oracle")


def _producer() -> dict[str, str]:
    """Record the code commit, immutable SIF, and an explicit bind bridge if used."""

    identity = producer_identity()
    source_commit = os.environ.get("PRORM_IMAGE_SOURCE_COMMIT")
    if source_commit is not None:
        if len(source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in source_commit
        ):
            raise ValueError("PRORM_IMAGE_SOURCE_COMMIT must be a 40-character digest")
        identity["image_source_commit"] = source_commit
    return identity


def adapter_name(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown reward source: {method}")
    return f"{method}__beta_0p2"


def policy_names() -> list[str]:
    return ["pi0", *(adapter_name(method) for method in METHODS)]


def _validate_source(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    *,
    seed: int,
    verify_artifact_tensors: bool = False,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError("real-policy NGD requires the validated Fisher-TRPO source protocol")
    seed = validate_seed(seed)
    if seed not in normalized["run"]["seeds"]:
        raise ValueError("seed is not declared by the source configuration")
    digest = config_hash(normalized)
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    if verify_artifact_tensors:
        _ = load_exact_delta_artifact(
            artifact_dir,
            expected_config_hash=digest,
            expected_seed=seed,
        )
    reward = load_trpo_reward_comparison(
        reward_result,
        expected_config_sha256=digest,
        expected_seed=seed,
    )
    if reward.get("artifact_metadata_sha256") != artifact_identity:
        raise ValueError("reward result and materialization identities differ")
    directions = reward.get("policy_directions")
    dimension = reward.get("dimensions", {}).get("policy_tangent")
    if not isinstance(directions, dict) or set(directions) != set(METHODS):
        raise ValueError("source reward result must contain exactly three natural directions")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("source policy-tangent dimension is invalid")
    for method in METHODS:
        values = directions[method]
        if (
            not isinstance(values, list)
            or len(values) != dimension
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError(f"source natural direction is invalid: {method}")
    return normalized, artifact_identity, reward


def _component_record(receipt: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "reward_source": receipt["reward_source"],
        "beta": receipt["beta"],
        "step_scale": receipt["step_scale"],
        "direction_sha256": receipt["direction_sha256"],
        "update_sha256": receipt["update_sha256"],
        "writeback_max_abs_error": receipt["writeback_max_abs_error"],
        "files": dict(receipt["files"]),
        "component_receipt_sha256": digest,
    }


def _validate_component(root: Path, name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    directory = root / name
    receipt_path = root / ".checkpoints" / f"{name}.json"
    if not directory.is_dir() or not receipt_path.is_file():
        raise FileNotFoundError(f"incomplete real-policy adapter: {name}")
    receipt = _read_json(receipt_path)
    files = receipt.get("files")
    fixed = {
        key: value
        for key, value in receipt.items()
        if key not in {"files", "writeback_max_abs_error"}
    }
    expected_fixed = {
        key: value for key, value in expected.items() if key != "writeback_max_abs_error"
    }
    writeback_error = receipt.get("writeback_max_abs_error")
    if (
        fixed != expected_fixed
        or isinstance(writeback_error, bool)
        or not isinstance(writeback_error, (int, float))
        or not math.isfinite(float(writeback_error))
        or float(writeback_error) < 0.0
    ):
        raise ValueError(f"real-policy adapter identity mismatch: {name}")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"real-policy adapter files are missing: {name}")
    for relative, expected_sha in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"adapter file inventory escapes its directory: {name}")
        if sha256_file(directory / relative_path) != expected_sha:
            raise ValueError(f"adapter file digest mismatch: {name}/{relative}")
    return _component_record(receipt, sha256_file(receipt_path))


def validate_real_policy_adapters(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    normalized, artifact_identity, reward = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    root = Path(adapter_dir)
    metadata = _read_json(root / "metadata.json")
    expected_names = {adapter_name(method) for method in METHODS}
    if (
        metadata.get("schema") != ADAPTER_SCHEMA
        or metadata.get("protocol") != PROTOCOL
        or metadata.get("source_config_sha256") != config_hash(normalized)
        or metadata.get("artifact_metadata_sha256") != artifact_identity
        or metadata.get("source_reward_result_sha256") != sha256_file(Path(reward_result))
        or metadata.get("seed") != seed
        or metadata.get("beta") != BETA
        or set(metadata.get("adapters", {})) != expected_names
        or metadata.get("producer") != _producer()
    ):
        raise ValueError("real-policy adapter metadata mismatch")
    for name, record in metadata["adapters"].items():
        receipt_path = root / ".checkpoints" / f"{name}.json"
        receipt = _read_json(receipt_path)
        method = receipt.get("reward_source")
        direction = reward.get("policy_directions", {}).get(method)
        if not isinstance(direction, list):
            raise ValueError(f"real-policy adapter has unknown reward source: {name}")
        expected_update = [float(value) / BETA for value in direction]
        if (
            receipt.get("schema") != ADAPTER_COMPONENT_SCHEMA
            or receipt.get("status") != "complete"
            or receipt.get("protocol") != PROTOCOL
            or receipt.get("source_config_sha256") != config_hash(normalized)
            or receipt.get("artifact_metadata_sha256") != artifact_identity
            or receipt.get("source_reward_result_sha256") != sha256_file(Path(reward_result))
            or receipt.get("seed") != seed
            or receipt.get("adapter_name") != name
            or name != adapter_name(str(method))
            or receipt.get("beta") != BETA
            or receipt.get("step_scale") != 1.0 / BETA
            or receipt.get("direction_sha256") != _canonical_sha256(direction)
            or receipt.get("update_sha256") != _canonical_sha256(expected_update)
            or receipt.get("lora_a_sha256") != metadata.get("lora_a_sha256")
            or receipt.get("lora_layout_sha256") != _canonical_sha256(metadata.get("lora_layout"))
            or receipt.get("producer") != metadata["producer"]
            or _component_record(receipt, sha256_file(receipt_path)) != record
        ):
            raise ValueError(f"real-policy adapter receipt mismatch: {name}")
        for relative, expected_sha in record["files"].items():
            if sha256_file(root / name / relative) != expected_sha:
                raise ValueError(f"real-policy adapter digest mismatch: {name}/{relative}")
    return metadata


def export_real_policy_adapters(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Write each validated beta-free direction divided by the frozen beta."""

    normalized, artifact_identity, reward = _validate_source(
        config,
        artifact_dir,
        reward_result,
        seed=seed,
        verify_artifact_tensors=True,
    )
    digest = config_hash(normalized)
    reward_path = Path(reward_result)
    evidence = _read_json(Path(artifact_dir) / "metadata.json")["evidence"]
    lora_a_sha256 = evidence["policy_a_sha256"]
    lora_layout = evidence["policy_layout"]
    producer = _producer()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".checkpoints").mkdir(exist_ok=True)
    records: dict[str, Any] = {}
    missing: list[tuple[str, torch.Tensor, dict[str, Any]]] = []
    for method in METHODS:
        direction = torch.tensor(
            reward["policy_directions"][method], dtype=torch.float64, device=target_device
        )
        name = adapter_name(method)
        expected = {
            "schema": ADAPTER_COMPONENT_SCHEMA,
            "status": "complete",
            "protocol": PROTOCOL,
            "source_config_sha256": digest,
            "artifact_metadata_sha256": artifact_identity,
            "source_reward_result_sha256": sha256_file(reward_path),
            "seed": seed,
            "adapter_name": name,
            "reward_source": method,
            "beta": BETA,
            "step_scale": 1.0 / BETA,
            "direction_sha256": _canonical_sha256(direction.cpu().tolist()),
            "update_sha256": _canonical_sha256(
                [float(value) / BETA for value in reward["policy_directions"][method]]
            ),
            "lora_a_sha256": lora_a_sha256,
            "lora_layout_sha256": _canonical_sha256(lora_layout),
            "writeback_max_abs_error": 0.0,
            "producer": producer,
        }
        try:
            records[name] = _validate_component(root, name, expected)
        except (FileNotFoundError, ValueError):
            _quarantine_adapter_component(root, name)
            missing.append((name, direction, expected))

    setup: Any | None = None
    try:
        if missing:
            setup = _load_policy(normalized, seed, target_device, local_files_only)
            if setup.a_state_sha256 != lora_a_sha256:
                raise RuntimeError("reloaded fixed LoRA-A differs from materialization")
            if setup.layout.to_metadata() != lora_layout:
                raise RuntimeError("reloaded LoRA-B layout differs from materialization")
        for name, direction, expected in missing:
            assert setup is not None
            _zero_b(setup)
            set_tangent_update_(
                setup.named_tangent_parameters(), setup.layout, direction, step_size=1.0 / BETA
            )
            written = torch.cat(
                [
                    parameter.detach().to(dtype=torch.float64).reshape(-1)
                    for _, parameter in setup.named_tangent_parameters()
                ]
            )
            intended = direction / BETA
            max_error = float((written - intended).abs().max().item())
            tolerance = 5.0e-6 * max(1.0, float(intended.abs().max().item()))
            if max_error > tolerance:
                raise RuntimeError(f"LoRA-B writeback identity failed for {name}: {max_error:.3e}")
            fixed = {**expected, "writeback_max_abs_error": max_error}
            staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
            try:
                setup.model.save_pretrained(staging, safe_serialization=True)
                files = {
                    path.relative_to(staging).as_posix(): sha256_file(path)
                    for path in sorted(staging.rglob("*"))
                    if path.is_file()
                }
                if not files:
                    raise RuntimeError(f"adapter serialization produced no files: {name}")
                os.replace(staging, root / name)
                receipt = {**fixed, "files": files}
                receipt_path = root / ".checkpoints" / f"{name}.json"
                _atomic_json(receipt_path, receipt)
                records[name] = _component_record(receipt, sha256_file(receipt_path))
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        metadata = {
            "schema": ADAPTER_SCHEMA,
            "protocol": PROTOCOL,
            "source_config_sha256": digest,
            "artifact_metadata_sha256": artifact_identity,
            "source_reward_result_sha256": sha256_file(reward_path),
            "source_protocol": reward["protocol"],
            "source_producer": reward.get("producer", {}),
            "seed": seed,
            "beta": BETA,
            "update_rule": "lora_B = beta_free_natural_direction / beta",
            "lora_a_sha256": lora_a_sha256,
            "lora_layout": lora_layout,
            "adapters": records,
            "producer": producer,
        }
        metadata_path = root / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError("refusing to replace non-identical real-policy adapter metadata")
        if not metadata_path.exists():
            _atomic_json(metadata_path, metadata)
    finally:
        if setup is not None:
            _zero_b(setup)
            del setup
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return validate_real_policy_adapters(normalized, artifact_dir, reward_result, root, seed=seed)


def _rollout_inputs(artifact_identity: str, adapter_identity: str) -> dict[str, str]:
    return {"artifact_metadata": artifact_identity, "adapter_metadata": adapter_identity}


def _descriptor(policy_name: str) -> dict[str, Any]:
    if policy_name == "pi0":
        return {"policy_instance": policy_name, "reward_source": "pi0", "beta": BETA}
    for method in METHODS:
        if policy_name == adapter_name(method):
            return {"policy_instance": policy_name, "reward_source": method, "beta": BETA}
    raise ValueError(f"unknown real-policy instance: {policy_name}")


def validate_real_policy_rollout(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    _ = validate_real_policy_adapters(
        normalized, artifact_dir, reward_result, adapter_dir, seed=seed
    )
    descriptor = _descriptor(policy_name)
    adapter_identity = sha256_file(Path(adapter_dir) / "metadata.json")
    target = Path(output_dir)
    metadata_path = target / "metadata.json"
    rows_path = target / "rollouts.jsonl"
    expected = {
        "schema": POLICY_SCHEMA,
        "protocol": PROTOCOL,
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": _producer(),
        **descriptor,
        "prompt_count": int(normalized["evaluation"]["rollout"]["prompts"]),
        "responses_per_prompt": int(normalized["evaluation"]["rollout"]["responses_per_prompt"]),
        "generation": "fresh_test_prompt_rollout",
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
    }
    if _read_json(metadata_path) != expected:
        raise ValueError(f"real-policy rollout metadata mismatch: {target}")
    validate_stage_receipt(
        target / "receipt.json",
        normalized,
        stage=f"real-rollout:{policy_name}",
        seed=seed,
        inputs=_rollout_inputs(artifact_identity, adapter_identity),
        outputs={"metadata": sha256_file(metadata_path), "rollouts": sha256_file(rows_path)},
    )
    rows: list[dict[str, Any]] = []
    with rows_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("real-policy rollout row must be an object")
            rows.append(row)
    prompts = _test_prompts(normalized, Path(artifact_dir))
    responses = expected["responses_per_prompt"]
    canonical = [(prompt.prompt_id, index) for prompt in prompts for index in range(responses)]
    observed = [(str(row.get("prompt_id")), int(row.get("response_index", -1))) for row in rows]
    if observed != canonical:
        raise ValueError("real-policy rollout rows are not in canonical test order")
    for row in rows:
        if row.get("policy_instance") != policy_name:
            raise ValueError("rollout row policy identity mismatch")
        for field in ("oracle_reward", "forward_kl"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"rollout row has invalid {field}")
        if float(row["forward_kl"]) < -1.0e-7:
            raise ValueError("Rao-Blackwellized forward KL is materially negative")
    return expected, rows


def run_real_policy_rollout(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    adapters = validate_real_policy_adapters(
        normalized, artifact_dir, reward_result, adapter_dir, seed=seed
    )
    descriptor = _descriptor(policy_name)
    target = Path(output_dir)
    if target.exists():
        metadata, _ = validate_real_policy_rollout(
            normalized,
            artifact_dir,
            reward_result,
            adapter_dir,
            target,
            policy_name=policy_name,
            seed=seed,
        )
        return metadata
    adapter_identity = sha256_file(Path(adapter_dir) / "metadata.json")
    prompts = _test_prompts(normalized, Path(artifact_dir))
    transform = _read_json(Path(artifact_dir) / "metadata.json")["evidence"]["oracle_transform"]
    work = target.parent / f".{target.name}.work"
    manifest = {
        "schema": "prorm-real-policy-rollout-work/v1",
        "protocol": PROTOCOL,
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "producer": _producer(),
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
    first_adapter = next(iter(adapters["adapters"]))
    load_name = first_adapter if policy_name == "pi0" else policy_name
    model, tokenizer, oracle_model, oracle_tokenizer = _load_models(
        normalized,
        Path(adapter_dir),
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
        for start in range(0, len(prompts), checkpoint_prompts):
            stop = min(start + checkpoint_prompts, len(prompts))
            shard_path = work / "shards" / f"{start:06d}-{stop:06d}.json"
            if shard_path.exists():
                shard = _read_json(shard_path)
                if (
                    shard.get("schema") != SHARD_SCHEMA
                    or shard.get("manifest") != manifest
                    or shard.get("start") != start
                    or shard.get("stop") != stop
                ):
                    raise ValueError(f"rollout shard identity mismatch: {shard_path}")
                print(f"rollout policy={policy_name} prompts={stop}/{len(prompts)} status=reused")
                continue
            rows: list[dict[str, Any]] = []
            for batch_start in range(start, stop, prompt_batch):
                batch_stop = min(batch_start + prompt_batch, stop)
                generated = _generate_policy_batch(
                    model,
                    tokenizer,
                    oracle_model,
                    oracle_tokenizer,
                    prompts[batch_start:batch_stop],
                    responses=responses,
                    generation_seed=derive_seed(base_seed, f"real-rollout-batch:{batch_start}"),
                    device=device_value,
                    reference=policy_name == "pi0",
                    rao_blackwellized_kl=True,
                    oracle_center=float(transform["b"]),
                    oracle_scale=float(transform["tau"]),
                    policy_config=normalized["policy"],
                )
                rows.extend({**row, **descriptor} for row in generated)
            _atomic_json(
                shard_path,
                {
                    "schema": SHARD_SCHEMA,
                    "manifest": manifest,
                    "start": start,
                    "stop": stop,
                    "rows": rows,
                },
            )
            print(f"rollout policy={policy_name} prompts={stop}/{len(prompts)} status=checkpointed")
    finally:
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if device_value.type == "cuda":
            torch.cuda.empty_cache()

    all_rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), checkpoint_prompts):
        stop = min(start + checkpoint_prompts, len(prompts))
        all_rows.extend(_read_json(work / "shards" / f"{start:06d}-{stop:06d}.json")["rows"])
    metadata = {
        "schema": POLICY_SCHEMA,
        "protocol": PROTOCOL,
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": _producer(),
        **descriptor,
        "prompt_count": len(prompts),
        "responses_per_prompt": responses,
        "generation": "fresh_test_prompt_rollout",
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.final-", dir=target.parent))
    try:
        metadata_path = staging / "metadata.json"
        rows_path = staging / "rollouts.jsonl"
        _atomic_json(metadata_path, metadata)
        with rows_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_stage_receipt(
            staging / "receipt.json",
            normalized,
            stage=f"real-rollout:{policy_name}",
            seed=seed,
            inputs=_rollout_inputs(artifact_identity, adapter_identity),
            outputs={"metadata": sha256_file(metadata_path), "rollouts": sha256_file(rows_path)},
        )
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    shutil.rmtree(work)
    validate_real_policy_rollout(
        normalized,
        artifact_dir,
        reward_result,
        adapter_dir,
        target,
        policy_name=policy_name,
        seed=seed,
    )
    return metadata


def smoke_real_policy_writeback(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Exercise adapter load, updated-policy generation, oracle scoring, and KL."""

    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    adapters = validate_real_policy_adapters(
        normalized, artifact_dir, reward_result, adapter_dir, seed=seed
    )
    policy_name = adapter_name("mle_rm")
    prompts = _test_prompts(normalized, Path(artifact_dir))[:1]
    transform = _read_json(Path(artifact_dir) / "metadata.json")["evidence"]["oracle_transform"]
    device_value = torch.device(device)
    model, tokenizer, oracle_model, oracle_tokenizer = _load_models(
        normalized,
        Path(adapter_dir),
        adapter_name=policy_name,
        device=device_value,
        local_files_only=local_files_only,
    )
    model.set_adapter(policy_name)
    smoke_policy = dict(normalized["policy"])
    smoke_policy["max_response_tokens"] = min(32, int(smoke_policy["max_response_tokens"]))
    try:
        rows = _generate_policy_batch(
            model,
            tokenizer,
            oracle_model,
            oracle_tokenizer,
            prompts,
            responses=1,
            generation_seed=derive_seed(
                SeedBundle.from_base_seed(seed).rollout, "real-policy-smoke"
            ),
            device=device_value,
            reference=False,
            rao_blackwellized_kl=True,
            oracle_center=float(transform["b"]),
            oracle_scale=float(transform["tau"]),
            policy_config=smoke_policy,
        )
    finally:
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if device_value.type == "cuda":
            torch.cuda.empty_cache()
    if len(rows) != 1 or not rows[0].get("response"):
        raise RuntimeError("real-policy smoke did not generate one non-empty response")
    reward = float(rows[0]["oracle_reward"])
    kl = float(rows[0]["forward_kl"])
    if not math.isfinite(reward) or not math.isfinite(kl) or kl < -1.0e-7:
        raise RuntimeError("real-policy smoke produced invalid reward or forward KL")
    payload = {
        "schema": "prorm-real-policy-smoke/v1",
        "protocol": PROTOCOL,
        "smoke_only": True,
        "source_config_sha256": config_hash(normalized),
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "seed": seed,
        "beta": BETA,
        "policy_instance": policy_name,
        "checks": {
            "updated_adapter_loaded": policy_name in adapters["adapters"],
            "fresh_generation_nonempty": True,
            "oracle_reward_finite": True,
            "rao_blackwellized_forward_kl_finite": True,
        },
        "sample": {"oracle_reward": reward, "forward_kl": kl},
        "producer": _producer(),
    }
    _atomic_json(Path(output), payload)
    return payload


def _policy_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["oracle_reward"]) for row in rows]
    kls = [float(row["forward_kl"]) for row in rows]
    reward = statistics.fmean(rewards)
    kl = statistics.fmean(kls)
    return {"R": reward, "K": kl, "J": reward - BETA * kl}


def assemble_real_policy_seed(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    policy_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    root = Path(policy_root)
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in policy_names():
        _, rows[name] = validate_real_policy_rollout(
            normalized,
            artifact_dir,
            reward_result,
            adapter_dir,
            root / name,
            policy_name=name,
            seed=seed,
        )
    metrics = {name: _policy_metrics(rows[name]) for name in policy_names()}
    if abs(metrics["pi0"]["K"]) > 1.0e-12:
        raise RuntimeError("reference-policy KL must be exactly zero")
    for name, values in metrics.items():
        if values["K"] < -1.0e-7:
            raise RuntimeError(f"negative forward KL for {name}")
        if abs(values["J"] - (values["R"] - BETA * values["K"])) > 1.0e-12:
            raise RuntimeError(f"J identity failed for {name}")
    payload = {
        "schema": SEED_SCHEMA,
        "protocol": PROTOCOL,
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "beta": BETA,
        "policies": metrics,
        "definitions": {
            "R": "mean oracle reward on newly generated fixed-test responses",
            "K": "mean Rao-Blackwellized sequence KL(pi_updated || pi0) on pi_updated trajectories",
            "J": "R - beta*K",
        },
        "test_usage": "formal_evaluation_only",
        "artifact_metadata_sha256": artifact_identity,
        "source_reward_result_sha256": sha256_file(Path(reward_result)),
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "policy_receipt_sha256": {
            name: sha256_file(root / name / "receipt.json") for name in policy_names()
        },
        "producer": _producer(),
    }
    target = Path(output)
    if target.exists():
        if _read_json(target) != payload:
            raise ValueError("existing seed evaluation differs from validated rollouts")
    else:
        _atomic_json(target, payload)
    return payload


def aggregate_real_policy(
    config: Mapping[str, object],
    result_paths: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    normalized = validate_config(config)
    expected_seeds = list(normalized["run"]["seeds"])
    records_with_paths = [(_read_json(Path(path)), Path(path)) for path in result_paths]
    records_with_paths.sort(key=lambda item: expected_seeds.index(item[0].get("seed")))
    records = [item[0] for item in records_with_paths]
    if [record.get("seed") for record in records] != expected_seeds:
        raise ValueError("real-policy aggregate requires every declared seed exactly once")
    for record in records:
        if (
            record.get("schema") != SEED_SCHEMA
            or record.get("protocol") != PROTOCOL
            or record.get("beta") != BETA
            or set(record.get("policies", {})) != set(policy_names())
        ):
            raise ValueError("real-policy seed result identity mismatch")
    summary: dict[str, Any] = {}
    for name in policy_names():
        summary[name] = {}
        for metric in ("R", "K", "J"):
            values = [float(record["policies"][name][metric]) for record in records]
            summary[name][metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
                "seed_values": values,
            }
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "source_config_sha256": config_hash(normalized),
        "seeds": expected_seeds,
        "beta": BETA,
        "policies": summary,
        "input_sha256": {
            str(record["seed"]): sha256_file(path) for record, path in records_with_paths
        },
        "producer": _producer(),
    }
    target = Path(output)
    if target.exists():
        if _read_json(target) != payload:
            raise ValueError("existing real-policy aggregate differs from inputs")
    else:
        _atomic_json(target, payload)
    return payload


def audit_real_policy_run(
    config: Mapping[str, object],
    source_run_root: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    normalized = validate_config(config)
    source_root = Path(source_run_root)
    root = Path(run_root)
    aggregate = _read_json(root / "aggregate.json")
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("source_config_sha256") != config_hash(normalized)
        or aggregate.get("seeds") != normalized["run"]["seeds"]
        or aggregate.get("beta") != BETA
    ):
        raise ValueError("real-policy aggregate failed identity audit")
    checks: list[dict[str, Any]] = []
    for seed in normalized["run"]["seeds"]:
        source_seed = source_root / f"seed-{seed}"
        seed_root = root / f"seed-{seed}"
        adapters = validate_real_policy_adapters(
            normalized,
            source_seed / "artifact",
            source_seed / "reward_result.json",
            seed_root / "adapters",
            seed=seed,
        )
        result = assemble_real_policy_seed(
            normalized,
            source_seed / "artifact",
            source_seed / "reward_result.json",
            seed_root / "adapters",
            seed_root / "policy_rollouts",
            seed_root / "evaluation.json",
            seed=seed,
        )
        if aggregate["input_sha256"].get(str(seed)) != sha256_file(seed_root / "evaluation.json"):
            raise ValueError(f"aggregate input digest mismatch for seed {seed}")
        checks.append(
            {
                "seed": seed,
                "status": "passed",
                "source_artifact_metadata_sha256": result["artifact_metadata_sha256"],
                "source_reward_result_sha256": result["source_reward_result_sha256"],
                "adapter_metadata_sha256": sha256_file(seed_root / "adapters" / "metadata.json"),
                "updated_adapter_count": len(adapters["adapters"]),
                "fresh_rollout_policy_count": len(policy_names()),
            }
        )
    payload = {
        "schema": AUDIT_SCHEMA,
        "protocol": PROTOCOL,
        "status": "passed",
        "source_config_sha256": config_hash(normalized),
        "aggregate_sha256": sha256_file(root / "aggregate.json"),
        "beta": BETA,
        "checks": checks,
        "producer": _producer(),
    }
    _atomic_json(Path(output), payload)
    return payload


__all__ = [
    "BETA",
    "PROTOCOL",
    "adapter_name",
    "aggregate_real_policy",
    "assemble_real_policy_seed",
    "audit_real_policy_run",
    "export_real_policy_adapters",
    "policy_names",
    "run_real_policy_rollout",
    "smoke_real_policy_writeback",
    "validate_real_policy_adapters",
    "validate_real_policy_rollout",
]

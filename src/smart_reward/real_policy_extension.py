"""Append two fresh trajectories to the audited four-response policy rollouts.

The source run is immutable.  Every extended output records and validates the
source metadata, receipt, and rollout digests before combining the original
four responses with two newly generated responses under a disjoint seed
namespace.
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
import yaml

from .checkpoints import validate_source_stage_receipt, validate_stage_receipt, write_stage_receipt
from .config import config_hash, load_config
from .real_policy_evaluation import (
    BETA,
    _canonical_sha256,
    _descriptor,
    _policy_metrics,
    _producer,
    _rollout_inputs,
    _validate_source,
    policy_names,
    validate_real_policy_adapters,
)
from .real_policy_evaluation import (
    POLICY_SCHEMA as BASE_POLICY_SCHEMA,
)
from .real_policy_evaluation import (
    PROTOCOL as BASE_PROTOCOL,
)
from .rollout import _generate_policy_batch, _load_models, _read_json, _test_prompts
from .runtime import sha256_file
from .seeding import SeedBundle, derive_seed

CONFIG_SCHEMA = "prorm-real-policy-rollout-extension-config/v1"
PROTOCOL = "prorm-real-policy-rollout-extension-4-to-6/v1"
POLICY_SCHEMA = "prorm-real-policy-rollout-m6/v1"
SHARD_SCHEMA = "prorm-real-policy-rollout-m6-shard/v1"
SEED_SCHEMA = "prorm-real-policy-evaluation-m6/v1"
AGGREGATE_SCHEMA = "prorm-real-policy-aggregate-m6/v1"
AUDIT_SCHEMA = "prorm-real-policy-audit-m6/v1"

BASE_RESPONSES = 4
ADDITIONAL_RESPONSES = 2
TOTAL_RESPONSES = 6
PROMPTS = 512
SEED_NAMESPACE = "real-rollout-extension-4-to-6-batch"


def load_real_policy_extension_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    expected_top = {
        "schema",
        "source_config",
        "source_config_sha256",
        "experiment",
        "base_rollout",
        "increment",
        "evaluation",
    }
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported real-policy rollout extension config")
    if set(value) != expected_top:
        raise ValueError("real-policy extension config keys changed")
    experiment = value["experiment"]
    if (
        experiment.get("name") != "real-policy-beta0p2-m6-extension-v1"
        or tuple(experiment.get("seeds", ())) != (20261001, 20261002, 20261003)
        or list(experiment.get("policies", ())) != policy_names()
        or float(experiment.get("beta", -1.0)) != BETA
    ):
        raise ValueError("formal real-policy extension identity changed")
    base = value["base_rollout"]
    increment = value["increment"]
    evaluation = value["evaluation"]
    if base != {"protocol": BASE_PROTOCOL, "responses_per_prompt": BASE_RESPONSES}:
        raise ValueError("base rollout contract must remain m=4")
    if increment != {
        "responses_per_prompt": ADDITIONAL_RESPONSES,
        "seed_namespace": SEED_NAMESPACE,
    }:
        raise ValueError("increment contract must remain two fresh responses")
    if (
        int(evaluation.get("prompts", 0)) != PROMPTS
        or int(evaluation.get("responses_per_prompt", 0)) != TOTAL_RESPONSES
        or list(evaluation.get("metrics", ())) != ["R", "K", "J"]
        or evaluation.get("kl_estimator") != "rao_blackwellized_updated_policy_forward_kl"
    ):
        raise ValueError("m=6 evaluation contract changed")
    digest = value.get("source_config_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("source config digest is invalid")
    return value


def extension_hash(extension: Mapping[str, Any]) -> str:
    return _canonical_sha256(extension)


def resolve_source_config(
    extension_path: str | os.PathLike[str], extension: Mapping[str, Any]
) -> dict[str, Any]:
    path = (Path(extension_path).resolve().parent / str(extension["source_config"])).resolve()
    config = load_config(path)
    if config_hash(config) != extension["source_config_sha256"]:
        raise ValueError("real-policy extension source config digest mismatch")
    if (
        config["run"]["seeds"] != extension["experiment"]["seeds"]
        or int(config["data"]["num_candidates"]) != TOTAL_RESPONSES
        or int(config["evaluation"]["rollout"]["prompts"]) != PROMPTS
        or int(config["evaluation"]["rollout"]["responses_per_prompt"]) != BASE_RESPONSES
    ):
        raise ValueError("source configuration no longer matches the frozen m=4 run")
    return config


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"rollout row must be an object: {path}")
            rows.append(value)
    return rows


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[Any],
    *,
    responses: int,
    policy_name: str,
) -> None:
    canonical = [(prompt.prompt_id, index) for prompt in prompts for index in range(responses)]
    observed = [(str(row.get("prompt_id")), int(row.get("response_index", -1))) for row in rows]
    if observed != canonical:
        raise ValueError("rollout rows are not in canonical prompt/response order")
    for row in rows:
        if row.get("policy_instance") != policy_name:
            raise ValueError("rollout row policy identity mismatch")
        if not isinstance(row.get("response"), str):
            raise ValueError("rollout response must be text")
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


def _source_adapter_metadata(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    metadata = _read_json(Path(adapter_dir) / "metadata.json")
    producer = metadata.get("producer")
    if not isinstance(producer, dict) or not producer:
        raise ValueError("source adapter producer is missing")
    return validate_real_policy_adapters(
        config,
        artifact_dir,
        reward_result,
        adapter_dir,
        seed=seed,
        expected_producer=producer,
    )


def _validate_source_rollout(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    rollout_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    adapters = _source_adapter_metadata(
        normalized, artifact_dir, reward_result, adapter_dir, seed=seed
    )
    adapter_identity = sha256_file(Path(adapter_dir) / "metadata.json")
    target = Path(rollout_dir)
    metadata_path = target / "metadata.json"
    receipt_path = target / "receipt.json"
    rows_path = target / "rollouts.jsonl"
    producer = adapters["producer"]
    expected = {
        "schema": BASE_POLICY_SCHEMA,
        "protocol": BASE_PROTOCOL,
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": producer,
        **_descriptor(policy_name),
        "prompt_count": PROMPTS,
        "responses_per_prompt": BASE_RESPONSES,
        "generation": "fresh_test_prompt_rollout",
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
    }
    if _read_json(metadata_path) != expected:
        raise ValueError(f"source m=4 rollout metadata mismatch: {target}")
    validate_source_stage_receipt(
        receipt_path,
        normalized,
        stage=f"real-rollout:{policy_name}",
        seed=seed,
        inputs=_rollout_inputs(artifact_identity, adapter_identity),
        outputs={"metadata": sha256_file(metadata_path), "rollouts": sha256_file(rows_path)},
    )
    rows = _read_rows(rows_path)
    prompts = _test_prompts(normalized, Path(artifact_dir))
    _validate_rows(rows, prompts, responses=BASE_RESPONSES, policy_name=policy_name)
    identities = {
        "source_rollout_metadata": sha256_file(metadata_path),
        "source_rollout_receipt": sha256_file(receipt_path),
        "source_rollouts": sha256_file(rows_path),
    }
    return expected, rows, identities


def _extension_inputs(
    artifact_identity: str, adapter_identity: str, source_identities: Mapping[str, str]
) -> dict[str, str]:
    return {
        "artifact_metadata": artifact_identity,
        "adapter_metadata": adapter_identity,
        **dict(source_identities),
    }


def validate_extended_real_policy_rollout(
    extension_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    source_rollout_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    extension = load_real_policy_extension_config(extension_path)
    config = resolve_source_config(extension_path, extension)
    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    source_metadata, source_rows, source_identities = _validate_source_rollout(
        normalized,
        artifact_dir,
        reward_result,
        adapter_dir,
        source_rollout_dir,
        policy_name=policy_name,
        seed=seed,
    )
    adapter_identity = sha256_file(Path(adapter_dir) / "metadata.json")
    target = Path(output_dir)
    metadata_path = target / "metadata.json"
    receipt_path = target / "receipt.json"
    rows_path = target / "rollouts.jsonl"
    expected = {
        "schema": POLICY_SCHEMA,
        "protocol": PROTOCOL,
        "extension_config_sha256": extension_hash(extension),
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": _producer(),
        **_descriptor(policy_name),
        "prompt_count": PROMPTS,
        "base_responses_per_prompt": BASE_RESPONSES,
        "additional_responses_per_prompt": ADDITIONAL_RESPONSES,
        "responses_per_prompt": TOTAL_RESPONSES,
        "generation": "fresh_test_prompt_rollout_incremental_4_to_6",
        "additional_seed_namespace": SEED_NAMESPACE,
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
        "source_rollout": {
            **source_identities,
            "producer": source_metadata["producer"],
        },
    }
    if _read_json(metadata_path) != expected:
        raise ValueError(f"extended m=6 rollout metadata mismatch: {target}")
    validate_stage_receipt(
        receipt_path,
        normalized,
        stage=f"real-rollout-extension-4-to-6:{policy_name}",
        seed=seed,
        inputs=_extension_inputs(artifact_identity, adapter_identity, source_identities),
        outputs={"metadata": sha256_file(metadata_path), "rollouts": sha256_file(rows_path)},
    )
    rows = _read_rows(rows_path)
    prompts = _test_prompts(normalized, Path(artifact_dir))
    _validate_rows(rows, prompts, responses=TOTAL_RESPONSES, policy_name=policy_name)
    retained = [row for row in rows if int(row["response_index"]) < BASE_RESPONSES]
    if retained != source_rows:
        raise ValueError("m=6 rollout did not retain the source m=4 rows byte-for-byte")
    return expected, rows


def extend_real_policy_rollout(
    extension_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    source_rollout_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    extension = load_real_policy_extension_config(extension_path)
    config = resolve_source_config(extension_path, extension)
    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    source_metadata, source_rows, source_identities = _validate_source_rollout(
        normalized,
        artifact_dir,
        reward_result,
        adapter_dir,
        source_rollout_dir,
        policy_name=policy_name,
        seed=seed,
    )
    target = Path(output_dir)
    if target.exists():
        metadata, _ = validate_extended_real_policy_rollout(
            extension_path,
            artifact_dir,
            reward_result,
            adapter_dir,
            source_rollout_dir,
            target,
            policy_name=policy_name,
            seed=seed,
        )
        return metadata
    adapters = _source_adapter_metadata(
        normalized, artifact_dir, reward_result, adapter_dir, seed=seed
    )
    adapter_identity = sha256_file(Path(adapter_dir) / "metadata.json")
    prompts = _test_prompts(normalized, Path(artifact_dir))
    transform = _read_json(Path(artifact_dir) / "metadata.json")["evidence"]["oracle_transform"]
    descriptor = _descriptor(policy_name)
    work = target.parent / f".{target.name}.work"
    manifest = {
        "schema": "prorm-real-policy-rollout-m6-work/v1",
        "protocol": PROTOCOL,
        "extension_config_sha256": extension_hash(extension),
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "producer": _producer(),
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "source_rollout": source_identities,
        "additional_seed_namespace": SEED_NAMESPACE,
        **descriptor,
    }
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError(f"m=6 extension work identity mismatch: {work}")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(f"unidentified m=6 extension work directory: {work}")
        _atomic_write_json(manifest_path, manifest)

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
                    raise ValueError(f"m=6 extension shard identity mismatch: {shard_path}")
                print(f"extend policy={policy_name} prompts={stop}/{len(prompts)} status=reused")
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
                    responses=ADDITIONAL_RESPONSES,
                    generation_seed=derive_seed(base_seed, f"{SEED_NAMESPACE}:{batch_start}"),
                    device=device_value,
                    reference=policy_name == "pi0",
                    rao_blackwellized_kl=True,
                    oracle_center=float(transform["b"]),
                    oracle_scale=float(transform["tau"]),
                    policy_config=normalized["policy"],
                )
                rows.extend(
                    {
                        **row,
                        "response_index": int(row["response_index"]) + BASE_RESPONSES,
                        **descriptor,
                    }
                    for row in generated
                )
            _atomic_write_json(
                shard_path,
                {
                    "schema": SHARD_SCHEMA,
                    "manifest": manifest,
                    "start": start,
                    "stop": stop,
                    "rows": rows,
                },
            )
            print(f"extend policy={policy_name} prompts={stop}/{len(prompts)} status=checkpointed")
    finally:
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if device_value.type == "cuda":
            torch.cuda.empty_cache()

    additional_rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), checkpoint_prompts):
        stop = min(start + checkpoint_prompts, len(prompts))
        additional_rows.extend(_read_json(work / "shards" / f"{start:06d}-{stop:06d}.json")["rows"])
    source_by_prompt = {
        prompt.prompt_id: source_rows[index * BASE_RESPONSES : (index + 1) * BASE_RESPONSES]
        for index, prompt in enumerate(prompts)
    }
    additional_by_prompt = {
        prompt.prompt_id: additional_rows[
            index * ADDITIONAL_RESPONSES : (index + 1) * ADDITIONAL_RESPONSES
        ]
        for index, prompt in enumerate(prompts)
    }
    all_rows = [
        row
        for prompt in prompts
        for row in (*source_by_prompt[prompt.prompt_id], *additional_by_prompt[prompt.prompt_id])
    ]
    metadata = {
        "schema": POLICY_SCHEMA,
        "protocol": PROTOCOL,
        "extension_config_sha256": extension_hash(extension),
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "adapter_metadata_sha256": adapter_identity,
        "producer": _producer(),
        **descriptor,
        "prompt_count": len(prompts),
        "base_responses_per_prompt": BASE_RESPONSES,
        "additional_responses_per_prompt": ADDITIONAL_RESPONSES,
        "responses_per_prompt": TOTAL_RESPONSES,
        "generation": "fresh_test_prompt_rollout_incremental_4_to_6",
        "additional_seed_namespace": SEED_NAMESPACE,
        "kl_estimator": "rao_blackwellized_updated_policy_forward_kl",
        "source_rollout": {
            **source_identities,
            "producer": source_metadata["producer"],
        },
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.final-", dir=target.parent))
    try:
        metadata_path = staging / "metadata.json"
        rows_path = staging / "rollouts.jsonl"
        _atomic_write_json(metadata_path, metadata)
        with rows_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_stage_receipt(
            staging / "receipt.json",
            normalized,
            stage=f"real-rollout-extension-4-to-6:{policy_name}",
            seed=seed,
            inputs=_extension_inputs(artifact_identity, adapter_identity, source_identities),
            outputs={"metadata": sha256_file(metadata_path), "rollouts": sha256_file(rows_path)},
        )
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    shutil.rmtree(work)
    validate_extended_real_policy_rollout(
        extension_path,
        artifact_dir,
        reward_result,
        adapter_dir,
        source_rollout_dir,
        target,
        policy_name=policy_name,
        seed=seed,
    )
    return metadata


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def assemble_extended_real_policy_seed(
    extension_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    source_policy_root: str | os.PathLike[str],
    policy_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    extension = load_real_policy_extension_config(extension_path)
    config = resolve_source_config(extension_path, extension)
    normalized, artifact_identity, _ = _validate_source(
        config, artifact_dir, reward_result, seed=seed
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    source_root = Path(source_policy_root)
    root = Path(policy_root)
    for name in policy_names():
        _, rows[name] = validate_extended_real_policy_rollout(
            extension_path,
            artifact_dir,
            reward_result,
            adapter_dir,
            source_root / name,
            root / name,
            policy_name=name,
            seed=seed,
        )
    metrics = {name: _policy_metrics(rows[name]) for name in policy_names()}
    if abs(metrics["pi0"]["K"]) > 1.0e-12:
        raise RuntimeError("reference-policy KL must be exactly zero")
    payload = {
        "schema": SEED_SCHEMA,
        "protocol": PROTOCOL,
        "extension_config_sha256": extension_hash(extension),
        "source_config_sha256": config_hash(normalized),
        "seed": seed,
        "beta": BETA,
        "prompt_count": PROMPTS,
        "base_responses_per_prompt": BASE_RESPONSES,
        "additional_responses_per_prompt": ADDITIONAL_RESPONSES,
        "responses_per_prompt": TOTAL_RESPONSES,
        "policies": metrics,
        "definitions": {
            "R": "mean oracle reward on six fresh fixed-test responses per prompt",
            "K": "mean Rao-Blackwellized sequence KL(pi_updated || pi0) on trajectories",
            "J": "R - beta*K",
        },
        "test_usage": "formal_evaluation_only",
        "artifact_metadata_sha256": artifact_identity,
        "source_reward_result_sha256": sha256_file(Path(reward_result)),
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "policy_receipt_sha256": {
            name: sha256_file(root / name / "receipt.json") for name in policy_names()
        },
        "source_policy_receipt_sha256": {
            name: sha256_file(source_root / name / "receipt.json") for name in policy_names()
        },
        "producer": _producer(),
    }
    target = Path(output)
    if target.exists():
        if _read_json(target) != payload:
            raise ValueError("existing m=6 seed evaluation differs from validated rollouts")
    else:
        _atomic_write_json(target, payload)
    return payload


def aggregate_extended_real_policy(
    extension_path: str | os.PathLike[str],
    result_paths: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_real_policy_extension_config(extension_path)
    config = resolve_source_config(extension_path, extension)
    expected_seeds = list(extension["experiment"]["seeds"])
    records_with_paths = [(_read_json(Path(path)), Path(path)) for path in result_paths]
    records_with_paths.sort(key=lambda item: expected_seeds.index(item[0].get("seed")))
    records = [item[0] for item in records_with_paths]
    if [record.get("seed") for record in records] != expected_seeds:
        raise ValueError("m=6 aggregate requires every declared seed exactly once")
    for record in records:
        if (
            record.get("schema") != SEED_SCHEMA
            or record.get("protocol") != PROTOCOL
            or record.get("extension_config_sha256") != extension_hash(extension)
            or record.get("beta") != BETA
            or record.get("responses_per_prompt") != TOTAL_RESPONSES
            or set(record.get("policies", {})) != set(policy_names())
        ):
            raise ValueError("m=6 seed result identity mismatch")
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
        "extension_config_sha256": extension_hash(extension),
        "source_config_sha256": config_hash(config),
        "seeds": expected_seeds,
        "beta": BETA,
        "prompt_count": PROMPTS,
        "base_responses_per_prompt": BASE_RESPONSES,
        "additional_responses_per_prompt": ADDITIONAL_RESPONSES,
        "responses_per_prompt": TOTAL_RESPONSES,
        "policies": summary,
        "input_sha256": {
            str(record["seed"]): sha256_file(path) for record, path in records_with_paths
        },
        "producer": _producer(),
    }
    target = Path(output)
    if target.exists():
        if _read_json(target) != payload:
            raise ValueError("existing m=6 aggregate differs from inputs")
    else:
        _atomic_write_json(target, payload)
    return payload


def audit_extended_real_policy_run(
    extension_path: str | os.PathLike[str],
    source_run_root: str | os.PathLike[str],
    base_real_run_root: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_real_policy_extension_config(extension_path)
    config = resolve_source_config(extension_path, extension)
    source_root = Path(source_run_root)
    base_root = Path(base_real_run_root)
    root = Path(run_root)
    aggregate = _read_json(root / "aggregate.json")
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("protocol") != PROTOCOL
        or aggregate.get("extension_config_sha256") != extension_hash(extension)
        or aggregate.get("responses_per_prompt") != TOTAL_RESPONSES
        or aggregate.get("seeds") != extension["experiment"]["seeds"]
        or aggregate.get("beta") != BETA
    ):
        raise ValueError("m=6 aggregate failed identity audit")
    checks: list[dict[str, Any]] = []
    for seed in extension["experiment"]["seeds"]:
        source_seed = source_root / f"seed-{seed}"
        base_seed = base_root / f"seed-{seed}"
        seed_root = root / f"seed-{seed}"
        adapters = _source_adapter_metadata(
            config,
            source_seed / "artifact",
            source_seed / "reward_result.json",
            base_seed / "adapters",
            seed=seed,
        )
        result = assemble_extended_real_policy_seed(
            extension_path,
            source_seed / "artifact",
            source_seed / "reward_result.json",
            base_seed / "adapters",
            base_seed / "policy_rollouts",
            seed_root / "policy_rollouts",
            seed_root / "evaluation.json",
            seed=seed,
        )
        if aggregate["input_sha256"].get(str(seed)) != sha256_file(seed_root / "evaluation.json"):
            raise ValueError(f"m=6 aggregate input digest mismatch for seed {seed}")
        checks.append(
            {
                "seed": seed,
                "status": "passed",
                "source_artifact_metadata_sha256": result["artifact_metadata_sha256"],
                "source_reward_result_sha256": result["source_reward_result_sha256"],
                "source_adapter_metadata_sha256": result["adapter_metadata_sha256"],
                "updated_adapter_count": len(adapters["adapters"]),
                "fresh_rollout_policy_count": len(policy_names()),
                "base_responses_per_prompt": BASE_RESPONSES,
                "additional_responses_per_prompt": ADDITIONAL_RESPONSES,
                "responses_per_prompt": TOTAL_RESPONSES,
                "rows_per_policy": PROMPTS * TOTAL_RESPONSES,
            }
        )
    payload = {
        "schema": AUDIT_SCHEMA,
        "protocol": PROTOCOL,
        "status": "passed",
        "extension_config_sha256": extension_hash(extension),
        "source_config_sha256": config_hash(config),
        "aggregate_sha256": sha256_file(root / "aggregate.json"),
        "beta": BETA,
        "checks": checks,
        "producer": _producer(),
    }
    _atomic_write_json(Path(output), payload)
    return payload


__all__ = [
    "ADDITIONAL_RESPONSES",
    "BASE_RESPONSES",
    "TOTAL_RESPONSES",
    "aggregate_extended_real_policy",
    "assemble_extended_real_policy_seed",
    "audit_extended_real_policy_run",
    "extend_real_policy_rollout",
    "load_real_policy_extension_config",
    "validate_extended_real_policy_rollout",
]

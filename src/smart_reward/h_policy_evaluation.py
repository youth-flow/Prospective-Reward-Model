"""Real LoRA writeback and fresh rollout evaluation for H/MSE ablations."""

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

from .artifacts import exact_delta_artifact_metadata_sha256
from .config import config_hash
from .h_ablation import (
    NEW_METHODS,
    RESULT_SCHEMA,
    h_config_hash,
    load_h_ablation_config,
    resolve_source_config,
)
from .h_ablation import (
    PROTOCOL as REWARD_PROTOCOL,
)
from .policy_update import set_tangent_update_
from .real_policy_evaluation import _load_policy, _zero_b
from .rollout import _generate_policy_batch, _load_models, _test_prompts
from .runtime import producer_identity, sha256_file
from .seeding import SeedBundle, derive_seed

PROTOCOL = "prorm-h-mse-real-policy-beta0p2-m6/v1"
ADAPTER_SCHEMA = "prorm-h-mse-ngd-adapters/v1"
ADAPTER_COMPONENT_SCHEMA = "prorm-h-mse-ngd-adapter-component/v1"
ROLLOUT_SCHEMA = "prorm-h-mse-policy-rollout/v1"
ROLLOUT_SHARD_SCHEMA = "prorm-h-mse-policy-rollout-shard/v1"
SEED_SCHEMA = "prorm-eight-policy-evaluation-m6/v1"
AGGREGATE_SCHEMA = "prorm-eight-policy-aggregate-m6/v1"
AUDIT_SCHEMA = "prorm-h-mse-integrity-audit/v1"
PROVENANCE_SCHEMA = "prorm-h-mse-provenance-bridge/v1"

SOURCE_POLICIES = (
    "pi0",
    "mle_rm__beta_0p2",
    "pro_rm__beta_0p2",
    "oracle__beta_0p2",
)
BETA = 0.2


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
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


def policy_name(method: str) -> str:
    if method not in NEW_METHODS:
        raise ValueError(f"unknown H/MSE method: {method}")
    return f"{method}__beta_0p2"


def new_policy_names() -> list[str]:
    return [policy_name(method) for method in NEW_METHODS]


def all_policy_names() -> list[str]:
    return [
        "pi0",
        "mle_rm__beta_0p2",
        "oracle_mse__beta_0p2",
        "pro_rm__beta_0p2",
        "h_mle__beta_0p2",
        "h_mse__beta_0p2",
        "h_pro__beta_0p2",
        "oracle__beta_0p2",
    ]


def _provenance_payload(
    config_path: str | os.PathLike[str],
    source_run_root: str | os.PathLike[str],
    source_m6_run_root: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_h_ablation_config(config_path)
    source = resolve_source_config(config_path, extension)
    source_root = Path(source_run_root)
    source_m6 = Path(source_m6_run_root)
    source_audit = _read_json(source_root / "integrity-audit.json")
    source_m6_audit = _read_json(source_m6 / "integrity-audit.json")
    if source_audit.get("status") != "passed" or source_audit.get("config_sha256") != config_hash(
        source
    ):
        raise ValueError("frozen source run did not pass its identity audit")
    if source_m6_audit.get("status") != "passed" or source_m6_audit.get(
        "source_config_sha256"
    ) != config_hash(source):
        raise ValueError("frozen m=6 run did not pass its identity audit")
    per_seed: dict[str, Any] = {}
    for seed in extension["experiment"]["seeds"]:
        source_seed = source_root / f"seed-{seed}"
        m6_seed = source_m6 / f"seed-{seed}"
        per_seed[str(seed)] = {
            "artifact_metadata_sha256": sha256_file(source_seed / "artifact" / "metadata.json"),
            "reward_result_sha256": sha256_file(source_seed / "reward_result.json"),
            "m6_evaluation_sha256": sha256_file(m6_seed / "evaluation.json"),
            "m6_rollout_receipt_sha256": {
                name: sha256_file(m6_seed / "policy_rollouts" / name / "receipt.json")
                for name in SOURCE_POLICIES
            },
        }
    return {
        "schema": PROVENANCE_SCHEMA,
        "protocol": PROTOCOL,
        "status": "accepted",
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "sources": {
            "frozen_main_run": {
                "path": str(source_root),
                "integrity_audit_sha256": sha256_file(source_root / "integrity-audit.json"),
                "fisher_selection_sha256": sha256_file(source_root / "fisher_selection.json"),
            },
            "frozen_m6_run": {
                "path": str(source_m6),
                "integrity_audit_sha256": sha256_file(source_m6 / "integrity-audit.json"),
                "aggregate_sha256": sha256_file(source_m6 / "aggregate.json"),
            },
        },
        "per_seed": per_seed,
        "dependency_closure": {
            "reused_read_only": [
                "prompt_splits",
                "six_candidate_nodes",
                "complete_edge_graph",
                "policy_scores",
                "reward_features",
                "true_rewards",
                "selected_train_fisher",
                "oracle_mle_reward",
                "oracle_pro_reward",
                "four_source_policy_rollouts_m6",
            ],
            "recomputed": [
                "h_annotation_sidecar",
                "oracle_mse_reward",
                "h_mle_reward",
                "h_mse_reward",
                "h_pro_reward",
                "four_new_ngd_adapters",
                "four_new_fresh_m6_rollouts",
                "six_reward_evaluation",
                "eight_policy_aggregation",
            ],
        },
        "producer": producer_identity(),
    }


def write_h_provenance_bridge(
    config_path: str | os.PathLike[str],
    source_run_root: str | os.PathLike[str],
    source_m6_run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    payload = _provenance_payload(config_path, source_run_root, source_m6_run_root)
    output_path = Path(output)
    if output_path.is_file():
        observed = _read_json(output_path)
        if observed != payload:
            raise ValueError("existing provenance bridge differs from immutable sources")
        return observed
    _atomic_json(output_path, payload)
    return payload


def _validate_reward_result(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    extension = load_h_ablation_config(config_path)
    source = resolve_source_config(config_path, extension)
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=extension["source_config_sha256"],
        expected_seed=seed,
    )
    result_path = Path(reward_result)
    result = _read_json(result_path)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("protocol") != REWARD_PROTOCOL
        or result.get("config_sha256") != h_config_hash(extension)
        or result.get("source_config_sha256") != config_hash(source)
        or result.get("artifact_metadata_sha256") != artifact_identity
        or result.get("seed") != seed
        or result.get("beta") != BETA
        or set(result.get("methods", {})) != set(NEW_METHODS)
        or set(result.get("policy_directions", {})) != set(NEW_METHODS)
        or result.get("test_usage") != "evaluation_only_no_selection"
    ):
        raise ValueError("H/MSE reward result identity mismatch")
    for method in NEW_METHODS:
        fit = result["methods"][method]
        direction = result["policy_directions"][method]
        if (
            fit.get("converged") is not True
            or not isinstance(direction, list)
            or len(direction) != 7168
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in direction
            )
        ):
            raise ValueError(f"invalid converged fit or direction: {method}")
    return extension, source, result, artifact_identity


def _component_record(receipt: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "reward_source": receipt["reward_source"],
        "beta": receipt["beta"],
        "step_scale": receipt["step_scale"],
        "direction_norm": receipt["direction_norm"],
        "writeback_max_abs_error": receipt["writeback_max_abs_error"],
        "files": dict(receipt["files"]),
        "component_receipt_sha256": digest,
    }


def _validate_component(root: Path, name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    directory = root / name
    receipt_path = root / ".checkpoints" / f"{name}.json"
    if not directory.is_dir() or not receipt_path.is_file():
        raise FileNotFoundError(f"adapter component is incomplete: {name}")
    receipt = _read_json(receipt_path)
    files = receipt.get("files")
    fixed = {key: value for key, value in receipt.items() if key != "files"}
    if fixed != dict(expected) or not isinstance(files, dict) or not files:
        raise ValueError(f"adapter component identity mismatch: {name}")
    for relative, digest in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("adapter component file escapes its directory")
        if sha256_file(directory / relative_path) != digest:
            raise ValueError(f"adapter component digest mismatch: {name}/{relative}")
    return _component_record(receipt, sha256_file(receipt_path))


def _quarantine_component(root: Path, name: str) -> None:
    directory = root / name
    receipt = root / ".checkpoints" / f"{name}.json"
    if not directory.exists() and not receipt.exists():
        return
    rejected = Path(tempfile.mkdtemp(prefix=f"{name}.", dir=root / ".rejected"))
    if directory.exists():
        os.replace(directory, rejected / "adapter")
    if receipt.exists():
        os.replace(receipt, rejected / "component.json")


def validate_h_adapters(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    extension, source, result, artifact_identity = _validate_reward_result(
        config_path, artifact_dir, reward_result, seed=seed
    )
    root = Path(adapter_dir)
    metadata = _read_json(root / "metadata.json")
    if (
        metadata.get("schema") != ADAPTER_SCHEMA
        or metadata.get("protocol") != PROTOCOL
        or metadata.get("config_sha256") != h_config_hash(extension)
        or metadata.get("source_config_sha256") != config_hash(source)
        or metadata.get("artifact_metadata_sha256") != artifact_identity
        or metadata.get("reward_result_sha256") != sha256_file(Path(reward_result))
        or metadata.get("seed") != seed
        or metadata.get("beta") != BETA
        or set(metadata.get("adapters", {})) != set(new_policy_names())
        or not isinstance(metadata.get("producer"), dict)
    ):
        raise ValueError("H/MSE adapter metadata identity mismatch")
    for method in NEW_METHODS:
        name = policy_name(method)
        direction = result["policy_directions"][method]
        expected = {
            "schema": ADAPTER_COMPONENT_SCHEMA,
            "status": "complete",
            "protocol": PROTOCOL,
            "config_sha256": h_config_hash(extension),
            "source_config_sha256": config_hash(source),
            "artifact_metadata_sha256": artifact_identity,
            "reward_result_sha256": sha256_file(Path(reward_result)),
            "seed": seed,
            "adapter_name": name,
            "reward_source": method,
            "beta": BETA,
            "step_scale": 1.0 / BETA,
            "direction_sha256": _canonical_sha256(direction),
            "update_sha256": _canonical_sha256([float(value) / BETA for value in direction]),
            "direction_norm": float(
                torch.linalg.vector_norm(torch.tensor(direction, dtype=torch.float64)).item()
            ),
            "lora_a_sha256": metadata["lora_a_sha256"],
            "lora_layout_sha256": _canonical_sha256(metadata["lora_layout"]),
            "writeback_max_abs_error": metadata["adapters"][name]["writeback_max_abs_error"],
            "producer": metadata["producer"],
        }
        record = _validate_component(root, name, expected)
        if record != metadata["adapters"][name]:
            raise ValueError(f"adapter metadata/component mismatch: {name}")
    return metadata


def export_h_adapters(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    extension, source, result, artifact_identity = _validate_reward_result(
        config_path, artifact_dir, reward_result, seed=seed
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    evidence = _read_json(Path(artifact_dir) / "metadata.json")["evidence"]
    lora_a_sha256 = evidence["policy_a_sha256"]
    lora_layout = evidence["policy_layout"]
    producer = producer_identity()
    root = Path(output_dir)
    if (root / "metadata.json").is_file():
        return validate_h_adapters(config_path, artifact_dir, reward_result, root, seed=seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".checkpoints").mkdir(exist_ok=True)
    (root / ".rejected").mkdir(exist_ok=True)
    records: dict[str, Any] = {}
    missing: list[tuple[str, str, torch.Tensor, dict[str, Any]]] = []
    for method in NEW_METHODS:
        name = policy_name(method)
        values = result["policy_directions"][method]
        direction = torch.tensor(values, dtype=torch.float64, device=target_device)
        expected = {
            "schema": ADAPTER_COMPONENT_SCHEMA,
            "status": "complete",
            "protocol": PROTOCOL,
            "config_sha256": h_config_hash(extension),
            "source_config_sha256": config_hash(source),
            "artifact_metadata_sha256": artifact_identity,
            "reward_result_sha256": sha256_file(Path(reward_result)),
            "seed": seed,
            "adapter_name": name,
            "reward_source": method,
            "beta": BETA,
            "step_scale": 1.0 / BETA,
            "direction_sha256": _canonical_sha256(values),
            "update_sha256": _canonical_sha256([float(value) / BETA for value in values]),
            "direction_norm": float(torch.linalg.vector_norm(direction).item()),
            "lora_a_sha256": lora_a_sha256,
            "lora_layout_sha256": _canonical_sha256(lora_layout),
            "writeback_max_abs_error": 0.0,
            "producer": producer,
        }
        receipt_path = root / ".checkpoints" / f"{name}.json"
        if receipt_path.is_file():
            observed_error = _read_json(receipt_path).get("writeback_max_abs_error")
            if (
                not isinstance(observed_error, bool)
                and isinstance(observed_error, (int, float))
                and math.isfinite(float(observed_error))
                and float(observed_error) >= 0.0
            ):
                expected = {
                    **expected,
                    "writeback_max_abs_error": float(observed_error),
                }
        try:
            records[name] = _validate_component(root, name, expected)
        except (FileNotFoundError, ValueError):
            _quarantine_component(root, name)
            missing.append((name, method, direction, expected))
    setup: Any | None = None
    try:
        if missing:
            setup = _load_policy(source, seed, target_device, local_files_only)
            if setup.a_state_sha256 != lora_a_sha256:
                raise RuntimeError("reloaded LoRA-A differs from the artifact")
            if setup.layout.to_metadata() != lora_layout:
                raise RuntimeError("reloaded LoRA-B layout differs from the artifact")
        for name, _method, direction, expected in missing:
            assert setup is not None
            _zero_b(setup)
            set_tangent_update_(
                setup.named_tangent_parameters(), setup.layout, direction, step_size=1.0 / BETA
            )
            written = torch.cat(
                [
                    parameter.detach().to(torch.float64).reshape(-1)
                    for _, parameter in setup.named_tangent_parameters()
                ]
            )
            intended = direction / BETA
            max_error = float((written - intended).abs().max().item())
            tolerance = 5.0e-6 * max(1.0, float(intended.abs().max().item()))
            if max_error > tolerance:
                raise RuntimeError(f"LoRA-B writeback failed for {name}: {max_error:.3e}")
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
            "config_sha256": h_config_hash(extension),
            "source_config_sha256": config_hash(source),
            "artifact_metadata_sha256": artifact_identity,
            "reward_result_sha256": sha256_file(Path(reward_result)),
            "seed": seed,
            "beta": BETA,
            "update_rule": "lora_B = beta_free_natural_direction / beta",
            "lora_a_sha256": lora_a_sha256,
            "lora_layout": lora_layout,
            "adapters": records,
            "producer": producer,
        }
        _atomic_json(root / "metadata.json", metadata)
    finally:
        if setup is not None:
            _zero_b(setup)
            del setup
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return validate_h_adapters(config_path, artifact_dir, reward_result, root, seed=seed)


def _descriptor(method: str) -> dict[str, Any]:
    return {
        "policy_instance": policy_name(method),
        "reward_source": method,
        "beta": BETA,
    }


def _canonical_rows(prompt_ids: Sequence[str], responses: int) -> list[tuple[str, int]]:
    return [(prompt_id, index) for prompt_id in prompt_ids for index in range(responses)]


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[str],
    responses: int,
    policy: str,
) -> None:
    observed = [(str(row.get("prompt_id")), int(row.get("response_index", -1))) for row in rows]
    if observed != _canonical_rows(prompt_ids, responses):
        raise ValueError("rollout rows are not in canonical prompt/response order")
    for row in rows:
        if row.get("policy_instance") != policy:
            raise ValueError("rollout policy identity mismatch")
        for field in ("oracle_reward", "forward_kl"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"invalid rollout {field}")
        if float(row["forward_kl"]) < -1.0e-7:
            raise ValueError("forward KL is materially negative")


def validate_h_rollout(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    extension, source, _result, artifact_identity = _validate_reward_result(
        config_path, artifact_dir, reward_result, seed=seed
    )
    adapters = validate_h_adapters(
        config_path,
        artifact_dir,
        reward_result,
        adapter_dir,
        seed=seed,
    )
    rollout = extension["rollout"]
    target = Path(output_dir)
    metadata = _read_json(target / "metadata.json")
    expected = {
        "schema": ROLLOUT_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "artifact_metadata_sha256": artifact_identity,
        "reward_result_sha256": sha256_file(Path(reward_result)),
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "seed": seed,
        **_descriptor(method),
        "prompt_count": rollout["prompts"],
        "base_responses_per_prompt": rollout["base_responses_per_prompt"],
        "additional_responses_per_prompt": rollout["additional_responses_per_prompt"],
        "responses_per_prompt": rollout["responses_per_prompt"],
        "base_seed_namespace": rollout["base_seed_namespace"],
        "additional_seed_namespace": rollout["additional_seed_namespace"],
        "generation": rollout["generation"],
        "kl_estimator": rollout["kl_estimator"],
        "producer": producer_identity(),
    }
    if metadata != expected:
        raise ValueError("H/MSE rollout metadata mismatch")
    receipt = _read_json(target / "receipt.json")
    expected_receipt = {
        "schema": "prorm-h-mse-rollout-receipt/v1",
        "status": "complete",
        "metadata_sha256": sha256_file(target / "metadata.json"),
        "rollouts_sha256": sha256_file(target / "rollouts.jsonl"),
        "row_count": rollout["prompts"] * rollout["responses_per_prompt"],
        "producer": producer_identity(),
    }
    if receipt != expected_receipt:
        raise ValueError("H/MSE rollout receipt mismatch")
    with (target / "rollouts.jsonl").open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    prompts = _test_prompts(source, Path(artifact_dir))[: rollout["prompts"]]
    _validate_rows(
        rows,
        [prompt.prompt_id for prompt in prompts],
        rollout["responses_per_prompt"],
        policy_name(method),
    )
    if adapters["adapters"][policy_name(method)]["reward_source"] != method:
        raise ValueError("rollout adapter reward source mismatch")
    return metadata, rows


def run_h_rollout(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    method: str,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    extension, source, _result, artifact_identity = _validate_reward_result(
        config_path, artifact_dir, reward_result, seed=seed
    )
    _ = validate_h_adapters(config_path, artifact_dir, reward_result, adapter_dir, seed=seed)
    if method not in NEW_METHODS:
        raise ValueError("unknown H/MSE rollout method")
    target = Path(output_dir)
    if target.exists():
        metadata, _ = validate_h_rollout(
            config_path,
            artifact_dir,
            reward_result,
            adapter_dir,
            target,
            seed=seed,
            method=method,
        )
        return metadata
    target.parent.mkdir(parents=True, exist_ok=True)
    rollout = extension["rollout"]
    prompts = _test_prompts(source, Path(artifact_dir))[: rollout["prompts"]]
    transform = _read_json(Path(artifact_dir) / "metadata.json")["evidence"]["oracle_transform"]
    work = target.parent / f".{target.name}.work"
    manifest = {
        "schema": "prorm-h-mse-policy-rollout-work/v1",
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "artifact_metadata_sha256": artifact_identity,
        "reward_result_sha256": sha256_file(Path(reward_result)),
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "seed": seed,
        **_descriptor(method),
        "producer": producer_identity(),
    }
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError("rollout work identity mismatch")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("unidentified rollout work directory")
        _atomic_json(manifest_path, manifest)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    name = policy_name(method)
    model, tokenizer, oracle_model, oracle_tokenizer = _load_models(
        source,
        Path(adapter_dir),
        adapter_name=name,
        device=target_device,
        local_files_only=local_files_only,
    )
    model.set_adapter(name)
    checkpoint_prompts = int(source["execution"]["rollout_checkpoint_prompts"])
    prompt_batch = int(source["execution"]["rollout_prompt_batch_size"])
    base_seed = SeedBundle.from_base_seed(seed).rollout
    base_responses = int(rollout["base_responses_per_prompt"])
    additional_responses = int(rollout["additional_responses_per_prompt"])
    try:
        for start in range(0, len(prompts), checkpoint_prompts):
            stop = min(start + checkpoint_prompts, len(prompts))
            shard_path = work / "shards" / f"{start:06d}-{stop:06d}.json"
            if shard_path.exists():
                shard = _read_json(shard_path)
                if (
                    shard.get("schema") != ROLLOUT_SHARD_SCHEMA
                    or shard.get("manifest") != manifest
                    or shard.get("start") != start
                    or shard.get("stop") != stop
                ):
                    raise ValueError("rollout shard identity mismatch")
                continue
            rows: list[dict[str, Any]] = []
            for batch_start in range(start, stop, prompt_batch):
                batch_stop = min(batch_start + prompt_batch, stop)
                batch_prompts = prompts[batch_start:batch_stop]
                base_rows = _generate_policy_batch(
                    model,
                    tokenizer,
                    oracle_model,
                    oracle_tokenizer,
                    batch_prompts,
                    responses=base_responses,
                    generation_seed=derive_seed(
                        base_seed, f"{rollout['base_seed_namespace']}:{batch_start}"
                    ),
                    device=target_device,
                    reference=False,
                    rao_blackwellized_kl=True,
                    oracle_center=float(transform["b"]),
                    oracle_scale=float(transform["tau"]),
                    policy_config=source["policy"],
                )
                rows.extend({**row, **_descriptor(method)} for row in base_rows)
                if additional_responses:
                    extra_rows = _generate_policy_batch(
                        model,
                        tokenizer,
                        oracle_model,
                        oracle_tokenizer,
                        batch_prompts,
                        responses=additional_responses,
                        generation_seed=derive_seed(
                            base_seed,
                            f"{rollout['additional_seed_namespace']}:{batch_start}",
                        ),
                        device=target_device,
                        reference=False,
                        rao_blackwellized_kl=True,
                        oracle_center=float(transform["b"]),
                        oracle_scale=float(transform["tau"]),
                        policy_config=source["policy"],
                    )
                    rows.extend(
                        {
                            **row,
                            "response_index": int(row["response_index"]) + base_responses,
                            **_descriptor(method),
                        }
                        for row in extra_rows
                    )
            rows.sort(key=lambda row: (row["prompt_id"], row["response_index"]))
            prompt_order = {prompt.prompt_id: index for index, prompt in enumerate(prompts)}
            rows.sort(key=lambda row: (prompt_order[row["prompt_id"]], row["response_index"]))
            _atomic_json(
                shard_path,
                {
                    "schema": ROLLOUT_SHARD_SCHEMA,
                    "manifest": manifest,
                    "start": start,
                    "stop": stop,
                    "rows": rows,
                },
            )
            print(f"rollout policy={name} prompts={stop}/{len(prompts)} checkpointed", flush=True)
    finally:
        del model, tokenizer, oracle_model, oracle_tokenizer
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    all_rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), checkpoint_prompts):
        stop = min(start + checkpoint_prompts, len(prompts))
        all_rows.extend(_read_json(work / "shards" / f"{start:06d}-{stop:06d}.json")["rows"])
    metadata = {
        "schema": ROLLOUT_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "artifact_metadata_sha256": artifact_identity,
        "reward_result_sha256": sha256_file(Path(reward_result)),
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "seed": seed,
        **_descriptor(method),
        "prompt_count": rollout["prompts"],
        "base_responses_per_prompt": base_responses,
        "additional_responses_per_prompt": additional_responses,
        "responses_per_prompt": rollout["responses_per_prompt"],
        "base_seed_namespace": rollout["base_seed_namespace"],
        "additional_seed_namespace": rollout["additional_seed_namespace"],
        "generation": rollout["generation"],
        "kl_estimator": rollout["kl_estimator"],
        "producer": producer_identity(),
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.final-", dir=target.parent))
    try:
        _atomic_json(staging / "metadata.json", metadata)
        with (staging / "rollouts.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "schema": "prorm-h-mse-rollout-receipt/v1",
            "status": "complete",
            "metadata_sha256": sha256_file(staging / "metadata.json"),
            "rollouts_sha256": sha256_file(staging / "rollouts.jsonl"),
            "row_count": len(all_rows),
            "producer": producer_identity(),
        }
        _atomic_json(staging / "receipt.json", receipt)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    shutil.rmtree(work)
    validate_h_rollout(
        config_path,
        artifact_dir,
        reward_result,
        adapter_dir,
        target,
        seed=seed,
        method=method,
    )
    return metadata


def _policy_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    reward = statistics.fmean(float(row["oracle_reward"]) for row in rows)
    kl = statistics.fmean(float(row["forward_kl"]) for row in rows)
    return {"R": reward, "K": kl, "J": reward - BETA * kl}


def assemble_h_smoke(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    policy_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    extension, _source, result, artifact_identity = _validate_reward_result(
        config_path, artifact_dir, reward_result, seed=seed
    )
    _ = validate_h_adapters(config_path, artifact_dir, reward_result, adapter_dir, seed=seed)
    metrics: dict[str, Any] = {}
    receipts: dict[str, str] = {}
    for method in NEW_METHODS:
        name = policy_name(method)
        rollout_dir = Path(policy_root) / name
        _, rows = validate_h_rollout(
            config_path,
            artifact_dir,
            reward_result,
            adapter_dir,
            rollout_dir,
            seed=seed,
            method=method,
        )
        metrics[name] = _policy_metrics(rows)
        receipts[name] = sha256_file(rollout_dir / "receipt.json")
    payload = {
        "schema": "prorm-h-mse-smoke/v1",
        "protocol": PROTOCOL,
        "status": "passed",
        "config_sha256": h_config_hash(extension),
        "seed": seed,
        "artifact_metadata_sha256": artifact_identity,
        "reward_result_sha256": sha256_file(Path(reward_result)),
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "annotation_metadata_sha256": result["annotation_metadata_sha256"],
        "prompt_count": extension["rollout"]["prompts"],
        "responses_per_prompt": extension["rollout"]["responses_per_prompt"],
        "new_reward_count": len(result["methods"]),
        "new_policy_count": len(metrics),
        "policies": metrics,
        "policy_receipt_sha256": receipts,
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def _load_source_m6(
    source_seed_root: Path,
    prompt_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    evaluation_path = source_seed_root / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    if (
        evaluation.get("seed") != seed
        or evaluation.get("beta") != BETA
        or evaluation.get("prompt_count") != 512
        or evaluation.get("responses_per_prompt") != 6
        or set(evaluation.get("policies", {})) != set(SOURCE_POLICIES)
    ):
        raise ValueError("immutable m=6 source evaluation identity mismatch")
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for name in SOURCE_POLICIES:
        root = source_seed_root / "policy_rollouts" / name
        if sha256_file(root / "receipt.json") != evaluation["policy_receipt_sha256"][name]:
            raise ValueError(f"source m=6 receipt digest mismatch: {name}")
        with (root / "rollouts.jsonl").open("r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        _validate_rows(rows, prompt_ids, 6, name)
        if _policy_metrics(rows) != evaluation["policies"][name]:
            raise ValueError(f"source m=6 metrics differ from rows: {name}")
        rows_by_policy[name] = rows
    return evaluation, rows_by_policy


def assemble_h_policy_seed(
    config_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    reward_result: str | os.PathLike[str],
    adapter_dir: str | os.PathLike[str],
    source_m6_seed_root: str | os.PathLike[str],
    policy_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    seed: int,
) -> dict[str, Any]:
    extension, source, result, artifact_identity = _validate_reward_result(
        config_path, artifact_dir, reward_result, seed=seed
    )
    rollout = extension["rollout"]
    if rollout["prompts"] != 512 or rollout["responses_per_prompt"] != 6:
        raise ValueError("formal seed assembly requires 512 x 6 rollouts")
    _ = validate_h_adapters(
        config_path,
        artifact_dir,
        reward_result,
        adapter_dir,
        seed=seed,
    )
    prompts = _test_prompts(source, Path(artifact_dir))
    prompt_ids = [prompt.prompt_id for prompt in prompts]
    source_evaluation, source_rows = _load_source_m6(
        Path(source_m6_seed_root), prompt_ids, seed=seed
    )
    policies = dict(source_evaluation["policies"])
    new_receipts: dict[str, str] = {}
    new_rows: dict[str, list[dict[str, Any]]] = {}
    for method in NEW_METHODS:
        name = policy_name(method)
        rollout_dir = Path(policy_root) / name
        _, rows = validate_h_rollout(
            config_path,
            artifact_dir,
            reward_result,
            adapter_dir,
            rollout_dir,
            seed=seed,
            method=method,
        )
        policies[name] = _policy_metrics(rows)
        new_rows[name] = rows
        new_receipts[name] = sha256_file(rollout_dir / "receipt.json")
    if set(policies) != set(all_policy_names()):
        raise RuntimeError("eight-policy identity mismatch")
    for metrics in policies.values():
        if abs(metrics["J"] - (metrics["R"] - BETA * metrics["K"])) > 1.0e-12:
            raise RuntimeError("policy utility identity failed")
    payload = {
        "schema": SEED_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "seed": seed,
        "beta": BETA,
        "prompt_count": 512,
        "responses_per_prompt": 6,
        "policies": policies,
        "reward_evaluation": result["reward_evaluation"],
        "source_m6_evaluation_sha256": sha256_file(Path(source_m6_seed_root) / "evaluation.json"),
        "source_policy_receipt_sha256": source_evaluation["policy_receipt_sha256"],
        "new_policy_receipt_sha256": new_receipts,
        "artifact_metadata_sha256": artifact_identity,
        "reward_result_sha256": sha256_file(Path(reward_result)),
        "adapter_metadata_sha256": sha256_file(Path(adapter_dir) / "metadata.json"),
        "row_count": sum(len(rows) for rows in source_rows.values())
        + sum(len(rows) for rows in new_rows.values()),
        "definitions": {
            "R": "mean synthetic-oracle reward on six fresh test responses per prompt",
            "K": "mean Rao-Blackwellized forward KL(pi || pi0) on pi trajectories",
            "J": "R - 0.2*K",
        },
        "producer": producer_identity(),
    }
    if payload["row_count"] != 8 * 512 * 6:
        raise RuntimeError("eight-policy row count mismatch")
    output_path = Path(output)
    if output_path.is_file():
        observed = _read_json(output_path)
        if observed != payload:
            raise ValueError("existing eight-policy evaluation differs from recomputation")
        return observed
    _atomic_json(output_path, payload)
    return payload


def aggregate_h_policy(
    config_path: str | os.PathLike[str],
    result_paths: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_h_ablation_config(config_path)
    source = resolve_source_config(config_path, extension)
    expected_seeds = list(extension["experiment"]["seeds"])
    records_with_paths = [(_read_json(Path(path)), Path(path)) for path in result_paths]
    records_with_paths.sort(key=lambda item: expected_seeds.index(item[0].get("seed")))
    records = [record for record, _ in records_with_paths]
    if [record.get("seed") for record in records] != expected_seeds:
        raise ValueError("policy aggregate requires every seed exactly once")
    for record in records:
        if (
            record.get("schema") != SEED_SCHEMA
            or record.get("protocol") != PROTOCOL
            or record.get("config_sha256") != h_config_hash(extension)
            or record.get("row_count") != 8 * 512 * 6
            or set(record.get("policies", {})) != set(all_policy_names())
        ):
            raise ValueError("eight-policy seed result identity mismatch")
    policy_summary: dict[str, Any] = {}
    reward_summary: dict[str, Any] = {}
    for name in all_policy_names():
        policy_summary[name] = {}
        for metric in ("R", "K", "J"):
            values = [float(record["policies"][name][metric]) for record in records]
            policy_summary[name][metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
                "seed_values": values,
            }
    for method in (
        "oracle_mle",
        "oracle_mse",
        "oracle_pro",
        "h_mle",
        "h_mse",
        "h_pro",
    ):
        reward_summary[method] = {}
        for metric in ("NLL", "MSE", "approximate_regret"):
            values = [float(record["reward_evaluation"][method][metric]) for record in records]
            reward_summary[method][metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
                "seed_values": values,
            }
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "seeds": expected_seeds,
        "beta": BETA,
        "prompt_count": 512,
        "responses_per_prompt": 6,
        "policy_count": 8,
        "total_rollout_rows": 8 * len(expected_seeds) * 512 * 6,
        "policies": policy_summary,
        "rewards": reward_summary,
        "input_sha256": {
            str(record["seed"]): sha256_file(path) for record, path in records_with_paths
        },
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


def audit_h_run(
    config_path: str | os.PathLike[str],
    source_run_root: str | os.PathLike[str],
    source_m6_run_root: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    extension = load_h_ablation_config(config_path)
    source = resolve_source_config(config_path, extension)
    source_root = Path(source_run_root)
    source_m6 = Path(source_m6_run_root)
    root = Path(run_root)
    provenance = _read_json(root / "provenance.json")
    if provenance != _provenance_payload(config_path, source_run_root, source_m6_run_root):
        raise ValueError("H/MSE provenance bridge differs from immutable sources")
    source_audit = _read_json(source_m6 / "integrity-audit.json")
    if source_audit.get("status") != "passed" or source_audit.get("beta") != BETA:
        raise ValueError("source m=6 run did not pass its immutable audit")
    if source_audit.get("aggregate_sha256") != sha256_file(source_m6 / "aggregate.json"):
        raise ValueError("source m=6 aggregate digest mismatch")
    aggregate = _read_json(root / "aggregate.json")
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("config_sha256") != h_config_hash(extension)
        or aggregate.get("total_rollout_rows") != 8 * 3 * 512 * 6
    ):
        raise ValueError("H/MSE aggregate identity mismatch")
    checks: list[dict[str, Any]] = []
    for seed in extension["experiment"]["seeds"]:
        source_seed = source_root / f"seed-{seed}"
        source_m6_seed = source_m6 / f"seed-{seed}"
        seed_root = root / f"seed-{seed}"
        result = assemble_h_policy_seed(
            config_path,
            source_seed / "artifact",
            seed_root / "reward_result.json",
            seed_root / "adapters",
            source_m6_seed,
            seed_root / "policy_rollouts",
            seed_root / "evaluation.json",
            seed=seed,
        )
        if aggregate["input_sha256"][str(seed)] != sha256_file(seed_root / "evaluation.json"):
            raise ValueError(f"aggregate input digest mismatch for seed {seed}")
        annotation = _read_json(seed_root / "h_annotations" / "metadata.json")
        checks.append(
            {
                "seed": seed,
                "status": "passed",
                "artifact_metadata_sha256": result["artifact_metadata_sha256"],
                "source_m6_evaluation_sha256": result["source_m6_evaluation_sha256"],
                "reward_result_sha256": result["reward_result_sha256"],
                "adapter_metadata_sha256": result["adapter_metadata_sha256"],
                "annotation_metadata_sha256": sha256_file(
                    seed_root / "h_annotations" / "metadata.json"
                ),
                "annotation_tensors_sha256": annotation["tensors_sha256"],
                "new_reward_count": 4,
                "learned_reward_count": 6,
                "new_policy_count": 4,
                "policy_count": 8,
                "new_rollout_rows": 4 * 512 * 6,
                "total_rollout_rows": result["row_count"],
            }
        )
    payload = {
        "schema": AUDIT_SCHEMA,
        "protocol": PROTOCOL,
        "status": "passed",
        "config_sha256": h_config_hash(extension),
        "source_config_sha256": config_hash(source),
        "source_m6_integrity_audit_sha256": sha256_file(source_m6 / "integrity-audit.json"),
        "source_m6_aggregate_sha256": sha256_file(source_m6 / "aggregate.json"),
        "provenance_bridge_sha256": sha256_file(root / "provenance.json"),
        "aggregate_sha256": sha256_file(root / "aggregate.json"),
        "checks": checks,
        "total_new_rollout_rows": 4 * 3 * 512 * 6,
        "total_combined_rollout_rows": 8 * 3 * 512 * 6,
        "producer": producer_identity(),
    }
    _atomic_json(Path(output), payload)
    return payload


__all__ = [
    "PROTOCOL",
    "aggregate_h_policy",
    "all_policy_names",
    "assemble_h_smoke",
    "assemble_h_policy_seed",
    "audit_h_run",
    "export_h_adapters",
    "new_policy_names",
    "policy_name",
    "run_h_rollout",
    "validate_h_adapters",
    "validate_h_rollout",
    "write_h_provenance_bridge",
]

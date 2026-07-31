"""Completion audit for the formal Fisher-corrected TRPO experiment."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .checkpoints import validate_stage_receipt
from .config import TRPO_PROTOCOL, config_hash, validate_config
from .fisher_crossfit import load_fisher_crossfit, load_fisher_selection
from .pipeline import (
    _materialization_inputs,
    _receipt,
    _rollout_aggregate_inputs,
    _validated_adapters,
    _validated_calibrated_adapters,
    _validated_materialization,
    _validated_reward,
    run_rollout_aggregate_stage,
)
from .prompts import load_prompt_jsonl
from .runtime import producer_identity, sha256_file
from .statistics import aggregate_results

SCHEMA = "prorm-fisher-trpo-integrity-audit/v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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


def audit_fisher_trpo_run(
    config: Mapping[str, object],
    run_root: str | os.PathLike[str],
    source_run_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Recompute every completion gate before archival."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL:
        raise ValueError("the Fisher-TRPO audit requires its v2 protocol")
    root = Path(run_root)
    source_root = Path(source_run_root)
    digest = config_hash(normalized)
    selection_path = root / "fisher_selection.json"
    selection = load_fisher_selection(
        selection_path,
        expected_config_sha256=digest,
    )
    seeds = list(normalized["run"]["seeds"])
    if set(selection["inputs"]) != {str(seed) for seed in seeds}:
        raise ValueError("Fisher selection does not cover every configured seed")

    seed_evidence: dict[str, Any] = {}
    reward_paths: list[Path] = []
    rollout_paths: list[Path] = []
    for seed in seeds:
        seed_root = root / f"seed-{seed}"
        source_seed_root = source_root / f"seed-{seed}"
        materialization = _validated_materialization(normalized, seed_root, seed=seed)
        crossfit_path = seed_root / "fisher_crossfit.json"
        crossfit = load_fisher_crossfit(
            crossfit_path,
            expected_config_sha256=digest,
            expected_seed=seed,
            expected_artifact_metadata_sha256=materialization["artifact_metadata"],
        )
        crossfit_outputs = {"fisher_crossfit": sha256_file(crossfit_path)}
        validate_stage_receipt(
            _receipt(seed_root, "fisher-crossfit"),
            normalized,
            stage="fisher-crossfit",
            seed=seed,
            inputs=_materialization_inputs(materialization),
            outputs=crossfit_outputs,
        )
        if selection["inputs"][str(seed)] != crossfit_outputs["fisher_crossfit"]:
            raise ValueError(f"Fisher selection input mismatch for seed {seed}")
        reward, reward_sha256 = _validated_reward(normalized, seed_root, seed=seed)
        initial_adapters = _validated_adapters(normalized, seed_root, seed=seed)
        calibrated = _validated_calibrated_adapters(normalized, seed_root, seed=seed)
        rollout_inputs = _rollout_aggregate_inputs(normalized, seed_root, seed=seed)
        rollout = run_rollout_aggregate_stage(normalized, seed_root, seed=seed)

        prompts = load_prompt_jsonl(seed_root / "artifact" / "prompts.jsonl")
        source_prompts = load_prompt_jsonl(source_seed_root / "artifact" / "prompts.jsonl")
        new_by_split = {
            split: [record.to_dict() for record in prompts if record.split == split]
            for split in ("train", "validation", "test")
        }
        source_by_split = {
            split: [record.to_dict() for record in source_prompts if record.split == split]
            for split in ("train", "validation", "test")
        }
        if new_by_split["train"] != source_by_split["train"]:
            raise ValueError(f"train prompts changed for seed {seed}")
        if new_by_split["validation"] != source_by_split["validation"]:
            raise ValueError(f"validation prompts changed for seed {seed}")
        legacy_ids = {
            record["prompt_id"]
            for split in ("train", "validation", "test")
            for record in source_by_split[split]
        }
        fresh_ids = {record["prompt_id"] for record in new_by_split["test"]}
        if legacy_ids.intersection(fresh_ids):
            raise ValueError(f"fresh test overlaps a legacy split for seed {seed}")
        evidence = _read_json(seed_root / "artifact" / "metadata.json")["evidence"]
        reuse = evidence.get("split_component_reuse")
        if (
            not isinstance(reuse, dict)
            or reuse.get("source_artifact_metadata_sha256")
            != sha256_file(source_seed_root / "artifact" / "metadata.json")
            or reuse.get("source_artifact_tensors_sha256")
            != sha256_file(source_seed_root / "artifact" / "tensors.safetensors")
        ):
            raise ValueError(f"split reuse provenance mismatch for seed {seed}")

        reward_path = seed_root / "reward_result.json"
        rollout_path = seed_root / "policy_utility" / "metrics.json"
        reward_paths.append(reward_path)
        rollout_paths.append(rollout_path)
        seed_evidence[str(seed)] = {
            "artifact_metadata_sha256": materialization["artifact_metadata"],
            "crossfit_sha256": crossfit_outputs["fisher_crossfit"],
            "crossfit_fold_assignment_sha256": crossfit["fold_assignment_sha256"],
            "reward_result_sha256": reward_sha256,
            "selected_relative_damping": reward["selected_relative_damping"],
            "initial_adapter_metadata_sha256": initial_adapters["adapter_metadata"],
            "calibrated_adapter_metadata_sha256": calibrated["adapter_metadata"],
            "rollout_input_receipts": rollout_inputs,
            "rollout_metrics_sha256": sha256_file(rollout_path),
            "fresh_test_prompt_count": len(fresh_ids),
            "rollout_protocol": rollout["protocol"],
        }

    aggregate_path = root / "aggregate.json"
    aggregate = _read_json(aggregate_path)
    recomputed = aggregate_results(
        normalized,
        reward_paths,
        rollout_paths,
    )
    if aggregate != recomputed:
        raise ValueError("stored three-seed aggregate differs from independent recomputation")
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "protocol": TRPO_PROTOCOL,
        "config_sha256": digest,
        "producer": producer_identity(),
        "fisher_selection_sha256": sha256_file(selection_path),
        "selected_relative_damping": selection["selected_relative_damping"],
        "seeds": seed_evidence,
        "aggregate_sha256": sha256_file(aggregate_path),
        "primary_estimand": aggregate["primary_estimand"],
    }
    target = Path(output)
    if target.exists():
        if _read_json(target) != payload:
            raise ValueError("existing integrity audit differs from recomputation")
    else:
        _atomic_json(target, payload)
    return payload


__all__ = ["SCHEMA", "audit_fisher_trpo_run"]

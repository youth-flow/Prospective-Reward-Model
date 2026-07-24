"""Integrity-checked Phase-2 input assembly.

This module is the sole bridge from the reusable Phase-1 materialization
artifact to the common-beta state machine.  It verifies every artifact and
manifest identity, restores the exact train-candidate/tensor order, and then
constructs :class:`~smart_reward.phase2_rollout.Phase2PreparedInputs`, whose
type has no validation/test reward field.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import phase1 as _phase1
from .artifacts import (
    artifact_metadata_sha256,
    load_controlled_feature_artifact,
)
from .cli import _formal_execution_requested, _run_environment_identity
from .config import config_hash
from .data import CandidateNode, load_jsonl
from .phase1_rollout import (
    _artifact_contract,
    _read_json_object,
    _validate_prompt_candidate_join,
)
from .phase2_config import load_phase2_config_bundle
from .phase2_heldout import DeferredHeldoutInputs, DeferredHeldoutSplit
from .phase2_rollout import Phase2PreparedInputs
from .prompts import PromptRecord, load_prompt_jsonl


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return seed


def _declared_seeds(config: Mapping[str, object]) -> tuple[int, ...]:
    run = config.get("run")
    if not isinstance(run, Mapping):
        raise TypeError("Phase-2 run config must be a mapping")
    seeds = run.get("seeds")
    if isinstance(seeds, (str, bytes, bytearray)) or not isinstance(
        seeds,
        Sequence,
    ):
        raise TypeError("Phase-2 run.seeds must be a sequence")
    return tuple(_validate_seed(seed) for seed in seeds)


def _expected_split_sizes(config: Mapping[str, object]) -> dict[str, int]:
    run = config.get("run")
    if not isinstance(run, Mapping):
        raise TypeError("source run config must be a mapping")
    split_sizes = run.get("split_sizes")
    if not isinstance(split_sizes, Mapping) or set(split_sizes) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("source run.split_sizes must contain the three frozen splits")
    result: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        value = split_sizes[split]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"source split size {split!r} must be positive")
        result[split] = value
    return result


def _ordered_candidates(
    candidates: Sequence[CandidateNode],
    prompt_ids: Sequence[str | int],
    *,
    num_candidates: int,
) -> tuple[CandidateNode, ...]:
    by_identity: dict[tuple[str | int, str | int], CandidateNode] = {}
    for candidate in candidates:
        identity = (candidate.prompt_id, candidate.candidate_id)
        if identity in by_identity:
            raise ValueError("candidate JSONL contains a duplicate identity")
        by_identity[identity] = candidate

    ordered: list[CandidateNode] = []
    expected_identities: set[tuple[str | int, str | int]] = set()
    for prompt_id in prompt_ids:
        if not isinstance(prompt_id, str):
            raise TypeError("controlled MultiPref prompt IDs must be strings")
        for candidate_index in range(num_candidates):
            candidate_id = _phase1._candidate_id(prompt_id, candidate_index)
            identity = (prompt_id, candidate_id)
            expected_identities.add(identity)
            try:
                ordered.append(by_identity[identity])
            except KeyError as error:
                raise ValueError(
                    "candidate JSONL is missing the canonical prompt/candidate identity "
                    f"{identity!r}"
                ) from error
    if set(by_identity) != expected_identities:
        unexpected = sorted(repr(identity) for identity in set(by_identity) - expected_identities)
        raise ValueError(f"candidate JSONL contains unexpected identities: {unexpected!r}")
    return tuple(ordered)


def _load_and_validate_prompt_semantics(
    artifact_path: Path,
    *,
    prompts: Sequence[PromptRecord],
    candidates: Sequence[CandidateNode],
    max_prompt_tokens: int,
    policy_chat_template_sha256: str,
) -> dict[str, object]:
    """Verify materialization's full-prompt evidence against saved graph bytes."""

    metadata = _read_json_object(artifact_path / "metadata.json")
    evidence = metadata.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("artifact metadata must contain an evidence object")
    semantics = evidence.get("policy_prompt_semantics")
    expected_semantics_keys = {
        "schema_version",
        "encoding",
        "add_generation_prompt",
        "truncation",
        "fail_closed_above_max_prompt_tokens",
        "max_prompt_tokens",
        "num_prompts",
        "records_sha256",
        "records",
    }
    if not isinstance(semantics, Mapping) or set(semantics) != expected_semantics_keys:
        raise ValueError(
            "Phase-2 requires exact full-policy-prompt semantics evidence in artifact metadata"
        )
    if (
        semantics["schema_version"] != _phase1._POLICY_PROMPT_SEMANTICS_SCHEMA
        or semantics["encoding"] != "policy_tokenizer_apply_chat_template"
        or semantics["add_generation_prompt"] is not True
        or semantics["truncation"] is not False
        or semantics["fail_closed_above_max_prompt_tokens"] is not True
        or semantics["max_prompt_tokens"] != max_prompt_tokens
    ):
        raise ValueError("artifact policy prompt semantics do not match the fail-closed contract")

    records = semantics["records"]
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("artifact policy prompt semantics records must be a sequence")
    if semantics["num_prompts"] != len(prompts) or len(records) != len(prompts):
        raise ValueError("artifact policy prompt semantics count does not match prompts.jsonl")
    expected_records_sha = _phase1._prompt_semantics_records_sha256(records)
    if semantics["records_sha256"] != expected_records_sha:
        raise ValueError("artifact policy prompt semantics records SHA256 mismatch")

    grouped: dict[str | int, list[CandidateNode]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.prompt_id, []).append(candidate)

    token_counts: list[int] = []
    raw_preserved = 0
    for prompt, raw_record in zip(prompts, records, strict=True):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "prompt_id",
            "raw_prompt_sha256",
            "policy_chat_token_count",
            "policy_prompt_token_ids_sha256",
            "max_prompt_tokens",
            "truncated",
            "raw_prompt_preserved",
        }:
            raise ValueError("artifact policy prompt semantics record has an invalid schema")
        prompt_id = getattr(prompt, "prompt_id", None)
        if raw_record["prompt_id"] != prompt_id:
            raise ValueError("artifact policy prompt semantics record order/identity mismatch")
        prompt_text = _phase1._prompt_text(prompt)
        if raw_record["raw_prompt_sha256"] != _phase1._prompt_text_sha256(prompt_text):
            raise ValueError("artifact policy prompt raw-text SHA256 mismatch")
        count = raw_record["policy_chat_token_count"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 < count <= max_prompt_tokens
        ):
            raise ValueError("artifact policy prompt token count is invalid or above the cap")
        if (
            raw_record["max_prompt_tokens"] != max_prompt_tokens
            or raw_record["truncated"] is not False
            or raw_record["raw_prompt_preserved"] is not True
        ):
            raise ValueError("artifact prompt record does not prove full raw-prompt preservation")
        token_digest = raw_record["policy_prompt_token_ids_sha256"]
        if (
            not isinstance(token_digest, str)
            or len(token_digest) != 64
            or any(character not in "0123456789abcdef" for character in token_digest)
        ):
            raise ValueError("artifact policy prompt token-prefix SHA256 is invalid")
        nodes = grouped.get(prompt_id)
        if not nodes:
            raise ValueError("artifact prompt semantics have no matching candidate nodes")
        for node in nodes:
            active = tuple(index for index, value in enumerate(node.response_mask) if value)
            if not active or active != tuple(range(active[0], active[-1] + 1)):
                raise ValueError("candidate response mask must select one contiguous response span")
            if active[0] != count:
                raise ValueError(
                    "candidate prompt-token prefix length differs from materialization evidence"
                )
            observed_digest = _phase1._prompt_token_ids_sha256(node.token_ids[:count])
            if observed_digest != token_digest:
                raise ValueError(
                    "candidate prompt-token prefix SHA256 differs from materialization evidence"
                )
        token_counts.append(count)
        raw_preserved += 1

    if set(grouped) != {getattr(prompt, "prompt_id", None) for prompt in prompts}:
        raise ValueError("candidate prompt IDs differ from prompt semantics evidence")
    return {
        "schema_version": _phase1._POLICY_PROMPT_SEMANTICS_SCHEMA,
        "policy_chat_template_sha256": policy_chat_template_sha256,
        "encoding": "policy_tokenizer_apply_chat_template",
        "add_generation_prompt": True,
        "truncation": False,
        "fail_closed_above_max_prompt_tokens": True,
        "max_prompt_tokens": max_prompt_tokens,
        "num_prompts": len(token_counts),
        "minimum_policy_chat_token_count": min(token_counts),
        "maximum_policy_chat_token_count": max(token_counts),
        "mean_policy_chat_token_count": sum(token_counts) / len(token_counts),
        "over_limit_prompt_count": 0,
        "truncated_prompt_count": 0,
        "raw_prompt_preserved_count": raw_preserved,
        "records_sha256": expected_records_sha,
        "candidate_prefixes_verified": True,
        "records": [dict(record) for record in records],
    }


def prepare_phase2_inputs(
    phase2_config: str | os.PathLike[str],
    *,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    run_manifest: str | os.PathLike[str],
    training_device: str | None = None,
    require_formal: bool | None = None,
    match_current_environment: bool = True,
) -> Phase2PreparedInputs:
    """Load and bind one source artifact without exposing held-out rewards.

    ``require_formal=None`` follows the current Slurm/image environment.  The
    formal HPC4 CLI leaves both identity checks enabled; tests and explicitly
    local diagnostics may pass ``False``.  ``training_device`` moves only the
    reward-free train tensors; sealed validation/test geometry stays on CPU
    until the final oracle session.
    """

    validated_seed = _validate_seed(seed)
    if require_formal is not None and not isinstance(require_formal, bool):
        raise TypeError("require_formal must be bool or None")
    if not isinstance(match_current_environment, bool):
        raise TypeError("match_current_environment must be bool")

    bundle = load_phase2_config_bundle(phase2_config)
    if validated_seed not in _declared_seeds(bundle.config):
        raise ValueError("seed is not declared by the Phase-2 design")
    source_config = bundle.base_config
    source_digest = config_hash(source_config)
    design_digest = bundle.design_identity
    artifact_path = Path(artifact_dir).resolve()
    manifest_path = Path(run_manifest).resolve()

    manifest_sha256, environment_identity = _run_environment_identity(
        manifest_path,
        expected_config_hash=source_digest,
        expected_seed=validated_seed,
        require_formal=(
            _formal_execution_requested() if require_formal is None else require_formal
        ),
        match_current_environment=match_current_environment,
    )
    metadata_digest = artifact_metadata_sha256(
        artifact_path,
        expected_config_hash=source_digest,
        expected_seed=validated_seed,
    )
    experiment = load_controlled_feature_artifact(
        artifact_path,
        expected_config_hash=source_digest,
        expected_seed=validated_seed,
    )
    expected_sizes = _expected_split_sizes(source_config)
    observed_sizes = {
        "train": experiment.train.num_prompts,
        "validation": experiment.validation.num_prompts,
        "test": experiment.test.num_prompts,
    }
    if observed_sizes != expected_sizes:
        raise ValueError(
            "artifact split geometry differs from the source config: "
            f"expected={expected_sizes!r}, observed={observed_sizes!r}"
        )

    contract = _artifact_contract(
        artifact_path,
        normalized_config=source_config,
        expected_config_hash=source_digest,
        expected_seed=validated_seed,
    )
    if contract.layout.dimension != experiment.train.policy_dimension:
        raise ValueError("artifact policy layout does not match train policy scores")
    if (
        artifact_metadata_sha256(
            artifact_path,
            expected_config_hash=source_digest,
            expected_seed=validated_seed,
        )
        != metadata_digest
    ):
        raise RuntimeError("artifact metadata changed while Phase-2 inputs were loaded")

    prompts = load_prompt_jsonl(artifact_path / "prompts.jsonl")
    candidates = load_jsonl(artifact_path / "candidates.jsonl", CandidateNode)
    ordered_prompts = _validate_prompt_candidate_join(
        experiment,
        prompts,
        candidates,
        num_candidates=experiment.train.num_candidates,
    )
    expected_prompt_ids = (
        *experiment.train.prompt_ids,
        *experiment.validation.prompt_ids,
        *experiment.test.prompt_ids,
    )
    if tuple(prompt.prompt_id for prompt in ordered_prompts) != expected_prompt_ids:
        raise RuntimeError("prompt join did not preserve tensor split order")
    policy = source_config.get("policy")
    if not isinstance(policy, Mapping):
        raise TypeError("source policy config must be a mapping")
    max_prompt_tokens = policy.get("max_prompt_tokens")
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        raise ValueError("source policy.max_prompt_tokens must be a positive integer")
    materialization_prompt_semantics = _load_and_validate_prompt_semantics(
        artifact_path,
        prompts=ordered_prompts,
        candidates=candidates,
        max_prompt_tokens=max_prompt_tokens,
        policy_chat_template_sha256=contract.policy_chat_template_sha256,
    )
    ordered_candidates = _ordered_candidates(
        candidates,
        expected_prompt_ids,
        num_candidates=experiment.train.num_candidates,
    )
    train_count = experiment.train.num_prompts * experiment.train.num_candidates
    validation_count = experiment.validation.num_prompts * experiment.validation.num_candidates
    train_candidates = ordered_candidates[:train_count]
    validation_candidates = ordered_candidates[train_count : train_count + validation_count]
    test_candidates = ordered_candidates[train_count + validation_count :]
    by_prompt_id = {prompt.prompt_id: prompt for prompt in ordered_prompts}
    test_prompts = tuple(by_prompt_id[prompt_id] for prompt_id in experiment.test.prompt_ids)
    heldout = DeferredHeldoutInputs(
        validation=DeferredHeldoutSplit.from_evaluation_tensor(
            "validation",
            experiment.validation,
            validation_candidates,
        ),
        test=DeferredHeldoutSplit.from_evaluation_tensor(
            "test",
            experiment.test,
            test_candidates,
        ),
    )

    # The returned object contains train tensors plus a target-free,
    # integrity-sealed validation/test geometry payload.  Validation/test
    # target tensors loaded for artifact schema verification become
    # unreachable before the first oracle or trainer session is opened.
    return Phase2PreparedInputs(
        source_config=source_config,
        source_config_hash=source_digest,
        phase2_config_hash=design_digest,
        seed=validated_seed,
        train=(
            experiment.train if training_device is None else experiment.train.to(training_device)
        ),
        train_candidates=train_candidates,
        test_prompts=test_prompts,
        heldout=heldout,
        oracle_transform=contract.oracle_transform,
        policy_layout=contract.layout,
        policy_a_sha256=contract.a_state_sha256,
        policy_chat_template_sha256=contract.policy_chat_template_sha256,
        oracle_chat_template_sha256=contract.oracle_chat_template_sha256,
        artifact_dir=artifact_path,
        artifact_metadata_sha256=metadata_digest,
        run_manifest=manifest_path,
        run_manifest_sha256=manifest_sha256,
        environment_identity=environment_identity,
        materialization_prompt_semantics=materialization_prompt_semantics,
    )


__all__ = ["prepare_phase2_inputs"]

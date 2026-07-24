from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import smart_reward.phase1 as phase1
import smart_reward.phase2_inputs as phase2_inputs
from smart_reward.config import config_hash
from smart_reward.data import CandidateNode
from smart_reward.experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    TrainingTensorData,
)
from smart_reward.oracle import RobustOracleTransform
from smart_reward.phase2_inputs import prepare_phase2_inputs
from smart_reward.prompts import ChatMessage, PromptRecord
from smart_reward.scores import ParameterLayout


def _train() -> TrainingTensorData:
    scores = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 10.0
    features = torch.flip(scores, dims=(2,))
    return TrainingTensorData(
        prompt_ids=("train-0", "train-1"),
        policy_scores=scores,
        reward_features=features,
        h=torch.tensor([0.1, -0.2], dtype=torch.float32),
        left_wins=torch.tensor([2, 1], dtype=torch.int64),
        num_annotations=torch.tensor([3, 2], dtype=torch.int64),
    )


def _evaluation(split: str) -> EvaluationTensorData:
    scores = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 8.0
    features = scores / 2.0
    return EvaluationTensorData(
        prompt_ids=(f"{split}-0", f"{split}-1"),
        policy_scores=scores,
        reward_features=features,
        true_rewards=features[..., 0],
    )


def _experiment() -> ControlledFeatureExperiment:
    return ControlledFeatureExperiment(
        train=_train(),
        validation=_evaluation("validation"),
        test=_evaluation("test"),
    )


def _prompt(prompt_id: str, split: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        messages=(ChatMessage(role="user", content=f"prompt {prompt_id}"),),
        split=split,
    )


def _candidate(prompt_id: str, index: int) -> CandidateNode:
    return CandidateNode(
        prompt_id=prompt_id,
        candidate_id=f"{prompt_id}::candidate::{index}",
        prompt=f"prompt {prompt_id}",
        response=f"response {index}",
        token_ids=(1, index + 2),
        response_mask=(0, 1),
        terminated_by_eos=True,
        reached_max_length=False,
    )


def _patch_valid_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    metadata_digests: tuple[str, ...] = ("a" * 64, "a" * 64),
):
    experiment = _experiment()
    base_config = {
        "run": {
            "seeds": [11],
            "split_sizes": {"train": 2, "validation": 2, "test": 2},
        },
        "policy": {"max_prompt_tokens": 1024},
    }
    overlay = {"run": {"seeds": [11]}}
    bundle = SimpleNamespace(
        config=overlay,
        base_config=base_config,
        design_identity="d" * 64,
    )
    prompts = [
        _prompt(prompt_id, split)
        for split, prompt_ids in (
            ("train", experiment.train.prompt_ids),
            ("validation", experiment.validation.prompt_ids),
            ("test", experiment.test.prompt_ids),
        )
        for prompt_id in prompt_ids
    ]
    candidates = [_candidate(prompt.prompt_id, index) for prompt in prompts for index in range(4)]
    candidates.reverse()
    prompt_semantics_records = [
        {
            "prompt_id": prompt.prompt_id,
            "raw_prompt_sha256": phase1._prompt_text_sha256(prompt.messages[0].content),
            "policy_chat_token_count": 1,
            "policy_prompt_token_ids_sha256": phase1._prompt_token_ids_sha256((1,)),
            "max_prompt_tokens": 1024,
            "truncated": False,
            "raw_prompt_preserved": True,
        }
        for prompt in prompts
    ]
    prompt_semantics = {
        "schema_version": phase1._POLICY_PROMPT_SEMANTICS_SCHEMA,
        "encoding": "policy_tokenizer_apply_chat_template",
        "add_generation_prompt": True,
        "truncation": False,
        "fail_closed_above_max_prompt_tokens": True,
        "max_prompt_tokens": 1024,
        "num_prompts": len(prompts),
        "records_sha256": phase1._prompt_semantics_records_sha256(prompt_semantics_records),
        "records": prompt_semantics_records,
    }
    tangent = torch.zeros(2, requires_grad=True)
    layout = ParameterLayout.from_named_parameters((("adapter.lora_B", tangent),))
    contract = SimpleNamespace(
        layout=layout,
        oracle_transform=RobustOracleTransform(b=0.5, tau=1.25),
        a_state_sha256="b" * 64,
        policy_chat_template_sha256="c" * 64,
        oracle_chat_template_sha256="e" * 64,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        phase2_inputs,
        "load_phase2_config_bundle",
        lambda path: bundle,
    )
    metadata_values = iter(metadata_digests)

    def fake_metadata(*args, **kwargs):
        del args, kwargs
        calls["metadata_calls"] = int(calls.get("metadata_calls", 0)) + 1
        return next(metadata_values)

    monkeypatch.setattr(
        phase2_inputs,
        "artifact_metadata_sha256",
        fake_metadata,
    )
    monkeypatch.setattr(
        phase2_inputs,
        "load_controlled_feature_artifact",
        lambda *args, **kwargs: experiment,
    )
    monkeypatch.setattr(
        phase2_inputs,
        "_artifact_contract",
        lambda *args, **kwargs: contract,
    )
    monkeypatch.setattr(
        phase2_inputs,
        "load_prompt_jsonl",
        lambda path: prompts,
    )
    monkeypatch.setattr(
        phase2_inputs,
        "load_jsonl",
        lambda path, record_type: candidates,
    )
    monkeypatch.setattr(
        phase2_inputs,
        "_read_json_object",
        lambda path: {"evidence": {"policy_prompt_semantics": prompt_semantics}},
    )

    def fake_manifest(path, **kwargs):
        calls["manifest"] = (Path(path), kwargs)
        return "f" * 64, {
            "formal": False,
            "git_commit": None,
            "image_sha256": None,
            "hf_inventory_sha256": None,
            "account": None,
            "partition": None,
            "gpu_models": [],
        }

    monkeypatch.setattr(
        phase2_inputs,
        "_run_environment_identity",
        fake_manifest,
    )
    calls["prompt_semantics"] = prompt_semantics
    return experiment, base_config, calls


def test_prepare_phase2_inputs_restores_tensor_order_and_drops_heldout_rewards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment, base_config, calls = _patch_valid_sources(monkeypatch, tmp_path)

    prepared = prepare_phase2_inputs(
        tmp_path / "phase2.yaml",
        seed=11,
        artifact_dir=tmp_path / "artifact",
        run_manifest=tmp_path / "run-manifest.json",
        require_formal=False,
        match_current_environment=False,
    )

    assert prepared.source_config_hash == config_hash(base_config)
    assert prepared.phase2_config_hash == "d" * 64
    assert prepared.train is experiment.train
    assert [candidate.candidate_id for candidate in prepared.train_candidates] == [
        f"{prompt_id}::candidate::{index}"
        for prompt_id in experiment.train.prompt_ids
        for index in range(4)
    ]
    assert tuple(prompt.prompt_id for prompt in prepared.test_prompts) == (
        experiment.test.prompt_ids
    )
    assert not hasattr(prepared, "validation")
    assert not hasattr(prepared, "test_rewards")
    assert not hasattr(prepared.heldout.validation, "true_rewards")
    assert not hasattr(prepared.heldout.test, "true_rewards")
    assert prepared.heldout.validation.prompt_ids == experiment.validation.prompt_ids
    assert prepared.heldout.test.prompt_ids == experiment.test.prompt_ids
    assert prepared.heldout.validation.identity_payload()["contains_oracle_targets"] is False
    assert prepared.heldout.test.identity_payload()["contains_oracle_targets"] is False
    assert prepared.run_manifest_sha256 == "f" * 64
    manifest_kwargs = calls["manifest"][1]
    assert manifest_kwargs["require_formal"] is False
    assert manifest_kwargs["match_current_environment"] is False


def test_prepare_phase2_inputs_moves_only_training_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment, _, _ = _patch_valid_sources(monkeypatch, tmp_path)

    prepared = prepare_phase2_inputs(
        tmp_path / "phase2.yaml",
        seed=11,
        artifact_dir=tmp_path / "artifact",
        run_manifest=tmp_path / "run-manifest.json",
        training_device="cpu",
        require_formal=False,
        match_current_environment=False,
    )

    assert prepared.train is not experiment.train
    assert prepared.train.policy_scores.device.type == "cpu"
    assert prepared.heldout.validation.policy_scores.device.type == "cpu"
    assert prepared.heldout.test.policy_scores.device.type == "cpu"
    assert prepared.heldout.validation.policy_scores.data_ptr() != (
        experiment.validation.policy_scores.data_ptr()
    )


def test_undeclared_seed_fails_before_artifact_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_valid_sources(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="not declared by the Phase-2 design"):
        prepare_phase2_inputs(
            tmp_path / "phase2.yaml",
            seed=12,
            artifact_dir=tmp_path / "artifact",
            run_manifest=tmp_path / "manifest.json",
        )


def test_artifact_metadata_change_is_detected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_valid_sources(
        monkeypatch,
        tmp_path,
        metadata_digests=("a" * 64, "9" * 64),
    )
    with pytest.raises(RuntimeError, match="metadata changed"):
        prepare_phase2_inputs(
            tmp_path / "phase2.yaml",
            seed=11,
            artifact_dir=tmp_path / "artifact",
            run_manifest=tmp_path / "manifest.json",
            require_formal=False,
        )


def test_source_split_geometry_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment, _, _ = _patch_valid_sources(monkeypatch, tmp_path)
    malformed = {
        "run": {
            "seeds": [11],
            "split_sizes": {"train": 3, "validation": 2, "test": 2},
        },
        "policy": {"max_prompt_tokens": 1024},
    }
    monkeypatch.setattr(
        phase2_inputs,
        "load_phase2_config_bundle",
        lambda path: SimpleNamespace(
            config={"run": {"seeds": [11]}},
            base_config=malformed,
            design_identity="d" * 64,
        ),
    )
    monkeypatch.setattr(
        phase2_inputs,
        "load_controlled_feature_artifact",
        lambda *args, **kwargs: experiment,
    )
    with pytest.raises(ValueError, match="split geometry"):
        prepare_phase2_inputs(
            tmp_path / "phase2.yaml",
            seed=11,
            artifact_dir=tmp_path / "artifact",
            run_manifest=tmp_path / "manifest.json",
            require_formal=False,
        )


def test_prompt_semantics_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, calls = _patch_valid_sources(monkeypatch, tmp_path)
    semantics = calls["prompt_semantics"]
    semantics["records"][0]["raw_prompt_sha256"] = "0" * 64
    semantics["records_sha256"] = phase1._prompt_semantics_records_sha256(semantics["records"])
    with pytest.raises(ValueError, match="raw-text SHA256 mismatch"):
        prepare_phase2_inputs(
            tmp_path / "phase2.yaml",
            seed=11,
            artifact_dir=tmp_path / "artifact",
            run_manifest=tmp_path / "manifest.json",
            require_formal=False,
        )

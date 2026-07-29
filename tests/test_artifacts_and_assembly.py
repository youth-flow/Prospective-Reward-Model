from __future__ import annotations

import json

import pytest
import torch

from smart_reward.artifacts import (
    ArtifactIntegrityError,
    exact_delta_artifact_metadata_sha256,
    load_exact_delta_artifact,
    save_exact_delta_artifact,
)
from smart_reward.exact_phase import assemble_exact_delta_experiment
from smart_reward.prompts import ChatMessage, PromptRecord


def _records() -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id=f"{split}-{index}",
            messages=(ChatMessage(role="user", content=f"question {split} {index}"),),
            split=split,
        )
        for split, count in (("train", 3), ("validation", 1), ("test", 1))
        for index in range(count)
    ]


def _assembly():
    records = _records()
    generator = torch.Generator().manual_seed(3)
    return assemble_exact_delta_experiment(
        records,
        torch.randn(5, 6, 4, generator=generator),
        torch.randn(5, 6, 3, generator=generator),
        torch.randn(5, 6, generator=generator),
    )


def test_assembly_builds_all_edges_and_disjoint_splits() -> None:
    assembly = _assembly()
    assert len(assembly.edges) == 5 * 15
    assert assembly.evidence["edges_per_prompt"] == 15
    assert assembly.experiment.train.num_candidates == 6
    assert all(edge.left_candidate_index < edge.right_candidate_index for edge in assembly.edges)


def test_artifact_roundtrip_and_identity(tmp_path) -> None:
    experiment = _assembly().experiment
    digest = "a" * 64
    target = tmp_path / "artifact"
    save_exact_delta_artifact(
        experiment,
        target,
        config_hash=digest,
        seed=7,
        evidence={"oracle_transform": {"b": 0.0, "tau": 1.0}},
    )
    loaded = load_exact_delta_artifact(
        target,
        expected_config_hash=digest,
        expected_seed=7,
    )
    assert loaded.train.prompt_ids == experiment.train.prompt_ids
    assert torch.equal(loaded.test.true_rewards, experiment.test.true_rewards)
    assert len(exact_delta_artifact_metadata_sha256(target)) == 64


def test_artifact_detects_tensor_tampering(tmp_path) -> None:
    target = tmp_path / "artifact"
    save_exact_delta_artifact(
        _assembly().experiment,
        target,
        config_hash="b" * 64,
        seed=9,
    )
    path = target / "tensors.safetensors"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_exact_delta_artifact(target)


def test_metadata_is_human_inspectable(tmp_path) -> None:
    target = tmp_path / "artifact"
    save_exact_delta_artifact(
        _assembly().experiment,
        target,
        config_hash="c" * 64,
        seed=11,
    )
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 11
    assert metadata["prompt_ids"]["test"] == ["test-0"]

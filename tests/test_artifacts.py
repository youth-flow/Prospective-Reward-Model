import hashlib
import json
from pathlib import Path

import pytest
import torch

import smart_reward.artifacts as artifact_module
from smart_reward.artifacts import (
    ArtifactDependencyError,
    ArtifactError,
    ArtifactIntegrityError,
    artifact_metadata_sha256,
    load_controlled_feature_artifact,
    save_controlled_feature_artifact,
)
from smart_reward.experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    TrainingTensorData,
)

CONFIG_HASH = "a" * 64
EXPECTED_KEYS = {
    "train.policy_scores",
    "train.reward_features",
    "train.h",
    "train.left_wins",
    "train.num_annotations",
    "validation.policy_scores",
    "validation.reward_features",
    "validation.true_rewards",
    "test.policy_scores",
    "test.reward_features",
    "test.true_rewards",
}


def _node_tensors(offset: float, prompts: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(prompts * 3, dtype=torch.float64).reshape(prompts, 3)
    policy_scores = torch.stack(
        (
            torch.sin(values + offset),
            torch.cos(0.5 * values - offset),
        ),
        dim=-1,
    )
    reward_features = torch.stack(
        (
            0.1 * values + offset,
            torch.sin(values * 0.3 + offset),
            torch.cos(values * 0.2 - offset),
            torch.ones_like(values),
        ),
        dim=-1,
    )
    return policy_scores, reward_features


def _evaluation(prefix: str, offset: float) -> EvaluationTensorData:
    scores, features = _node_tensors(offset, 3)
    rewards = 0.4 * features[..., 0] - 0.7 * features[..., 1] + scores[..., 0]
    return EvaluationTensorData(
        prompt_ids=tuple(f"{prefix}-{index}" for index in range(3)),
        policy_scores=scores,
        reward_features=features,
        true_rewards=rewards,
    )


def _experiment() -> ControlledFeatureExperiment:
    scores, features = _node_tensors(0.1, 4)
    train = TrainingTensorData(
        prompt_ids=tuple(f"train-{index}" for index in range(4)),
        policy_scores=scores,
        reward_features=features,
        h=torch.tensor([0.3, -0.2, 0.8, -0.7], dtype=torch.float64),
        left_wins=torch.tensor([5, 2, 7, 1], dtype=torch.int64),
        num_annotations=torch.tensor([8, 7, 9, 6], dtype=torch.int64),
    )
    return ControlledFeatureExperiment(
        train=train,
        validation=_evaluation("validation", 1.1),
        test=_evaluation("test", 2.2),
    )


def _assert_same_experiment(
    left: ControlledFeatureExperiment,
    right: ControlledFeatureExperiment,
) -> None:
    for split_name in ("train", "validation", "test"):
        left_split = getattr(left, split_name)
        right_split = getattr(right, split_name)
        assert left_split.prompt_ids == right_split.prompt_ids
        names = ["policy_scores", "reward_features"]
        if split_name == "train":
            names.extend(["h", "left_wins", "num_annotations"])
        else:
            names.append("true_rewards")
        for name in names:
            left_tensor = getattr(left_split, name)
            right_tensor = getattr(right_split, name)
            assert torch.equal(left_tensor.cpu(), right_tensor)
            assert right_tensor.device.type == "cpu"
            assert not right_tensor.requires_grad


def _metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _write_metadata(path: Path, value: dict[str, object]) -> None:
    (path / "metadata.json").write_text(
        json.dumps(value, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


def _rehash_tensors(path: Path) -> None:
    tensor_bytes = (path / "tensors.safetensors").read_bytes()
    metadata = _metadata(path)
    metadata["tensor_sha256"] = hashlib.sha256(tensor_bytes).hexdigest()
    _write_metadata(path, metadata)


def test_roundtrip_metadata_contract_and_default_no_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    source = _experiment()
    artifact = tmp_path / "phase-1"
    evidence = {
        "git_commit": "f" * 40,
        "upstream_hashes": {
            "multipref_revision": "1" * 40,
            "oracle_revision": "2" * 40,
        },
        "note": "controlled frozen-feature run",
    }
    result_path = save_controlled_feature_artifact(
        source,
        artifact,
        config_hash=CONFIG_HASH,
        seed=17,
        evidence=evidence,
    )
    assert result_path == artifact
    assert {item.name for item in artifact.iterdir()} == {
        "metadata.json",
        "tensors.safetensors",
    }

    metadata = _metadata(artifact)
    assert metadata["schema"] == "controlled-feature-artifact/v1"
    assert metadata["config_hash"] == CONFIG_HASH
    assert metadata["seed"] == 17
    assert metadata["evidence"] == evidence
    assert set(metadata["tensors"]) == EXPECTED_KEYS
    assert len(metadata["tensor_sha256"]) == 64
    for key, spec in metadata["tensors"].items():
        assert spec["shape"] == list(
            getattr(source, key.split(".")[0]).__getattribute__(key.split(".")[1]).shape
        )
        assert spec["dtype"] in {"float64", "int64"}

    loaded = load_controlled_feature_artifact(
        artifact,
        expected_config_hash=CONFIG_HASH,
        expected_seed=17,
    )
    _assert_same_experiment(source, loaded)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_controlled_feature_artifact(
            source,
            artifact,
            config_hash=CONFIG_HASH,
            seed=17,
        )


def test_hash_tampering_is_rejected_before_optional_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("safetensors.torch")
    artifact = tmp_path / "phase-1"
    save_controlled_feature_artifact(
        _experiment(),
        artifact,
        config_hash=CONFIG_HASH,
        seed=3,
    )
    tensor_path = artifact / "tensors.safetensors"
    content = bytearray(tensor_path.read_bytes())
    content[-1] ^= 1
    tensor_path.write_bytes(content)

    def dependency_must_not_be_reached() -> object:
        raise AssertionError("safetensors import happened before digest verification")

    monkeypatch.setattr(
        artifact_module,
        "_require_safetensors_torch",
        dependency_must_not_be_reached,
    )
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        load_controlled_feature_artifact(artifact)


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_missing_and_extra_tensor_keys_are_rejected(tmp_path: Path, mode: str) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    artifact = tmp_path / mode
    source = _experiment()
    save_controlled_feature_artifact(
        source,
        artifact,
        config_hash=CONFIG_HASH,
        seed=9,
    )
    tensor_path = artifact / "tensors.safetensors"
    tensors = safetensors_torch.load(tensor_path.read_bytes())
    if mode == "missing":
        tensors.pop("train.h")
    else:
        tensors["train.true_rewards"] = torch.zeros(
            source.train.num_prompts,
            source.train.num_candidates,
            dtype=torch.float64,
        )
    safetensors_torch.save_file(tensors, str(tensor_path))
    _rehash_tensors(artifact)

    with pytest.raises(ArtifactError, match="safetensors keys do not match schema"):
        load_controlled_feature_artifact(artifact)


def test_nonfinite_payload_is_rejected_after_valid_hash(tmp_path: Path) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    artifact = tmp_path / "nonfinite"
    save_controlled_feature_artifact(
        _experiment(),
        artifact,
        config_hash=CONFIG_HASH,
        seed=9,
    )
    tensor_path = artifact / "tensors.safetensors"
    tensors = safetensors_torch.load(tensor_path.read_bytes())
    rewards = tensors["validation.true_rewards"].clone()
    rewards[0, 0] = float("nan")
    tensors["validation.true_rewards"] = rewards
    safetensors_torch.save_file(tensors, str(tensor_path))
    _rehash_tensors(artifact)

    with pytest.raises(ArtifactError, match="contains NaN or infinity"):
        load_controlled_feature_artifact(artifact)


def test_prompt_id_leakage_is_rejected_from_metadata(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    artifact = tmp_path / "leaked"
    save_controlled_feature_artifact(
        _experiment(),
        artifact,
        config_hash=CONFIG_HASH,
        seed=9,
    )
    metadata = _metadata(artifact)
    metadata["splits"]["validation"]["prompt_ids"][0] = metadata["splits"]["train"]["prompt_ids"][0]
    _write_metadata(artifact, metadata)

    with pytest.raises(ArtifactError, match="prompt ID leakage"):
        load_controlled_feature_artifact(artifact)


def test_training_payload_has_no_oracle_channel(tmp_path: Path) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    artifact = tmp_path / "no-leakage"
    save_controlled_feature_artifact(
        _experiment(),
        artifact,
        config_hash=CONFIG_HASH,
        seed=12,
    )
    keys = set(safetensors_torch.load((artifact / "tensors.safetensors").read_bytes()))
    assert keys == EXPECTED_KEYS
    train_keys = {key for key in keys if key.startswith("train.")}
    assert not any("true" in key or "oracle" in key for key in train_keys)
    assert {key.removeprefix("train.") for key in train_keys} == {
        "policy_scores",
        "reward_features",
        "h",
        "left_wins",
        "num_annotations",
    }


@pytest.mark.parametrize(
    "evidence",
    [
        {"environment": {"HOME": "private"}},
        {"huggingface_token": "not-even-a-real-secret"},
        {"nested": {"password": "value"}},
        {"note": float("nan")},
    ],
)
def test_environment_credentials_and_nonfinite_evidence_are_refused(
    tmp_path: Path,
    evidence: dict[str, object],
) -> None:
    pytest.importorskip("safetensors.torch")
    with pytest.raises(ArtifactError):
        save_controlled_feature_artifact(
            _experiment(),
            tmp_path / "unsafe",
            config_hash=CONFIG_HASH,
            seed=1,
            evidence=evidence,
        )


def test_tokenizer_metadata_is_not_misclassified_as_a_credential(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    evidence = {
        "prompt_pool_selection": {
            "selected_minimum_policy_chat_token_count": 8,
            "selected_maximum_policy_chat_token_count": 19,
        },
        "policy_prompt_semantics": {
            "max_prompt_tokens": 1024,
            "fail_closed_above_max_prompt_tokens": True,
            "records": [
                {
                    "policy_chat_token_count": 8,
                    "policy_prompt_token_ids_sha256": "a" * 64,
                }
            ],
        },
        "reward_model": {"feature_pooling": "last_response_token"},
    }
    artifact = tmp_path / "tokenizer-metadata"
    save_controlled_feature_artifact(
        _experiment(),
        artifact,
        config_hash=CONFIG_HASH,
        seed=1,
        evidence=evidence,
    )

    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["evidence"] == evidence


@pytest.mark.parametrize(
    "credential_key",
    [
        "token",
        "access_token",
        "access_token_count",
        "auth_token",
        "auth_token_sha256",
        "huggingface_token",
        "huggingface_token_prefix",
        "session_token",
        "session_token_ids",
        "api_token_digest",
        "id_token",
        "access_tokens",
        "auth_tokens",
        "session_tokens",
        "huggingface_tokens",
        "api_tokens",
        "jwt_token_count",
        "csrf_token_count",
        "xsrf_token_ids",
        "slack_token_ids",
        "deploy_token_count",
        "registry_token_ids",
    ],
)
def test_credential_token_keys_remain_forbidden(
    tmp_path: Path,
    credential_key: str,
) -> None:
    pytest.importorskip("safetensors.torch")
    with pytest.raises(ArtifactError, match="environment/credential field"):
        save_controlled_feature_artifact(
            _experiment(),
            tmp_path / credential_key,
            config_hash=CONFIG_HASH,
            seed=1,
            evidence={credential_key: "not-even-a-real-secret"},
        )


def test_expected_identity_is_strict(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    artifact = tmp_path / "identity"
    save_controlled_feature_artifact(
        _experiment(),
        artifact,
        config_hash=CONFIG_HASH,
        seed=41,
    )
    with pytest.raises(ArtifactError, match="config hash mismatch"):
        load_controlled_feature_artifact(artifact, expected_config_hash="b" * 64)
    with pytest.raises(ArtifactError, match="seed mismatch"):
        load_controlled_feature_artifact(artifact, expected_seed=42)

    expected = hashlib.sha256((artifact / "metadata.json").read_bytes()).hexdigest()
    assert (
        artifact_metadata_sha256(
            artifact,
            expected_config_hash=CONFIG_HASH,
            expected_seed=41,
        )
        == expected
    )


def test_missing_optional_dependency_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = artifact_module.importlib.import_module

    def missing_safetensors(name: str) -> object:
        if name == "safetensors.torch":
            raise ModuleNotFoundError(name)
        return real_import_module(name)

    monkeypatch.setattr(artifact_module.importlib, "import_module", missing_safetensors)
    with pytest.raises(ArtifactDependencyError, match=r"smart-reward-model\[llm\]"):
        save_controlled_feature_artifact(
            _experiment(),
            tmp_path / "missing-dependency",
            config_hash=CONFIG_HASH,
            seed=1,
        )
    assert not (tmp_path / "missing-dependency").exists()

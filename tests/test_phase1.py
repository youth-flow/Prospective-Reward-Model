from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smart_reward.config import load_config
from smart_reward.data import repeated_labels_to_h
from smart_reward.oracle import fit_robust_oracle_transform
from smart_reward.phase1 import (
    _encode_full_policy_prompt,
    _load_prompts,
    _policy_lora_a_initialization,
    _policy_lora_a_initialization_evidence,
    _reward_class_projection_diagnostic,
    assemble_controlled_experiment,
    materialize_phase1,
)
from smart_reward.prompts import ChatMessage, PromptRecord


def _records() -> list[PromptRecord]:
    definitions = (
        ("train-b", "train"),
        ("validation-a", "validation"),
        ("train-a", "train"),
        ("test-b", "test"),
        ("validation-b", "validation"),
        ("test-a", "test"),
    )
    return [
        PromptRecord(
            prompt_id=prompt_id,
            messages=(ChatMessage(role="user", content=f"question for {prompt_id}"),),
            split=split,  # type: ignore[arg-type]
        )
        for prompt_id, split in definitions
    ]


def _tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1047)
    policy_scores = torch.randn(6, 4, 3, generator=generator, dtype=torch.float64)
    reward_features = torch.randn(6, 4, 2, generator=generator, dtype=torch.float64)
    raw_oracle_scores = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 1.0],
            [100.0, -100.0, 75.0, -75.0],
            [2.0, 3.0, 4.0, 5.0],
            [18.0, 17.0, 16.0, 15.0],
            [-31.0, 29.0, -27.0, 25.0],
            [9.0, 8.0, 7.0, 6.0],
        ],
        dtype=torch.float64,
    )
    return policy_scores, reward_features, raw_oracle_scores


def test_assembler_fits_oracle_on_training_nodes_only_and_never_leaks() -> None:
    records = _records()
    policy_scores, reward_features, raw_scores = _tensors()
    result = assemble_controlled_experiment(
        records,
        policy_scores,
        reward_features,
        raw_scores,
        seed=20260722,
    )

    expected_transform = fit_robust_oracle_transform(raw_scores[[0, 2]].reshape(-1))
    assert result.oracle_transform == expected_transform
    assert result.evidence["oracle_fit_split"] == "train"
    assert result.experiment.train.prompt_ids == ("train-b", "train-a")
    assert result.experiment.validation.prompt_ids == (
        "validation-a",
        "validation-b",
    )
    assert result.experiment.test.prompt_ids == ("test-b", "test-a")
    assert not hasattr(result.experiment.train, "true_rewards")
    assert all("true_margin" not in edge.to_dict() for edge in result.training_edges)
    assert all(not any("oracle" in key for key in edge.to_dict()) for edge in result.training_edges)

    expected_rewards = expected_transform(raw_scores)
    expected_projection = _reward_class_projection_diagnostic(
        reward_features[[0, 2]], expected_rewards[[0, 2]]
    )
    assert result.evidence["train_reward_class_projection"] == expected_projection
    assert torch.equal(
        result.experiment.validation.true_rewards,
        expected_rewards[[1, 4]],
    )
    assert torch.equal(result.experiment.test.true_rewards, expected_rewards[[3, 5]])

    changed_heldout = raw_scores.clone()
    changed_heldout[[1, 3, 4, 5]] *= 1_000_000.0
    changed_heldout_features = reward_features.clone()
    changed_heldout_features[[1, 3, 4, 5]] *= 1_000_000.0
    changed = assemble_controlled_experiment(
        records,
        policy_scores,
        changed_heldout_features,
        changed_heldout,
        seed=20260722,
    )
    assert changed.oracle_transform == result.oracle_transform
    assert torch.equal(changed.experiment.train.h, result.experiment.train.h)
    assert (
        changed.evidence["train_reward_class_projection"]
        == result.evidence["train_reward_class_projection"]
    )


def test_phase2_frozen_oracle_transform_is_reused_without_current_seed_refit() -> None:
    calibration = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "common_beta_pilot_base.yaml"
    )["oracle"]["transform_calibration"]
    policy_scores, reward_features, raw_scores = _tensors()
    first = assemble_controlled_experiment(
        _records(),
        policy_scores,
        reward_features,
        raw_scores,
        seed=20260801,
        oracle_transform_calibration=calibration,
    )
    shifted_scores = raw_scores * 1000.0 + 777.0
    second = assemble_controlled_experiment(
        _records(),
        policy_scores,
        reward_features,
        shifted_scores,
        seed=20260802,
        oracle_transform_calibration=calibration,
    )

    assert first.oracle_transform == second.oracle_transform
    assert first.oracle_transform.b == -4.500244140625
    assert first.oracle_transform.tau == 2.7715682983398438
    assert second.oracle_transform != fit_robust_oracle_transform(
        shifted_scores[[0, 2]].reshape(-1)
    )
    for result in (first, second):
        assert result.evidence["oracle_transform_calibration"] == calibration
        assert result.evidence["oracle_fit_split"] == "excluded_phase1_train_artifacts"
        assert result.evidence["oracle_transform_fitted_on_current_seed"] is False
        assert [entry["seed"] for entry in calibration["source_artifacts"]] == [
            20260722,
            20260723,
            20260724,
            20260725,
            20260726,
        ]


def test_phase2_reuses_one_global_lora_a_seed_and_phase1_keeps_per_seed_semantics() -> None:
    root = Path(__file__).resolve().parents[1]
    phase2_policy = load_config(root / "configs" / "common_beta_pilot_base.yaml")["policy"]
    first = _policy_lora_a_initialization(phase2_policy, run_seed=20260801)
    second = _policy_lora_a_initialization(phase2_policy, run_seed=20260802)

    assert first["initialization_seed"] == second["initialization_seed"]
    assert first["initialization_seed"] == 946081152281754541
    assert first["expected_sha256"] == second["expected_sha256"]
    assert first["expected_sha256"] == (
        "a2b5804109396f76b96cde98d1e2060f175a47724b1ca9fef317c7a10cb9a838"
    )
    assert (
        first["current_seed_derived_initialization_seed"]
        != second["current_seed_derived_initialization_seed"]
    )
    assert first["current_seed_derivation_used"] is False
    assert second["current_seed_derivation_used"] is False
    assert first["source_seed"] == second["source_seed"] == 20260722
    assert first["source_artifact_metadata_sha256"] == (
        "8ec7ca893a7c93d4bbfac0b71b829b37ec63af3ca0c7a9496465ed60815b155f"
    )
    first_evidence = _policy_lora_a_initialization_evidence(
        phase2_policy,
        run_seed=20260801,
        observed_sha256=str(first["expected_sha256"]),
    )
    second_evidence = _policy_lora_a_initialization_evidence(
        phase2_policy,
        run_seed=20260802,
        observed_sha256=str(second["expected_sha256"]),
    )
    assert first_evidence["observed_sha256"] == second_evidence["observed_sha256"]
    assert first_evidence["initialization_seed"] == second_evidence["initialization_seed"]
    with pytest.raises(RuntimeError, match="fingerprint"):
        _policy_lora_a_initialization_evidence(
            phase2_policy,
            run_seed=20260801,
            observed_sha256="0" * 64,
        )

    phase1_policy = load_config(root / "configs" / "main.yaml")["policy"]
    legacy_first = _policy_lora_a_initialization(phase1_policy, run_seed=20260722)
    legacy_second = _policy_lora_a_initialization(phase1_policy, run_seed=20260723)
    assert legacy_first["mode"] == legacy_second["mode"] == "current_seed_derived"
    assert legacy_first["current_seed_derivation_used"] is True
    assert legacy_second["current_seed_derivation_used"] is True
    assert legacy_first["initialization_seed"] != legacy_second["initialization_seed"]


def test_reward_class_projection_is_prompt_gauge_invariant() -> None:
    features = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]],
            [[-2.0, 0.0], [-1.0, 2.0], [0.0, 1.0], [2.0, -1.0]],
        ],
        dtype=torch.float64,
    )
    rewards = torch.tensor(
        [[-1.0, 0.5, 1.25, 3.0], [4.0, -2.0, 0.0, 1.5]],
        dtype=torch.float64,
    )
    feature_gauge = torch.tensor([[[91.0, -17.0]], [[-8.0, 23.0]]], dtype=torch.float64)
    reward_gauge = torch.tensor([[1000.0], [-3000.0]], dtype=torch.float64)

    reference = _reward_class_projection_diagnostic(features, rewards)
    shifted = _reward_class_projection_diagnostic(
        features + feature_gauge,
        rewards + reward_gauge,
    )

    assert set(reference) == {
        "fit_split",
        "centering",
        "solver",
        "target_centered_rms",
        "residual_rmse",
        "relative_residual",
    }
    assert reference["fit_split"] == "train"
    assert reference["centering"] == "per_prompt_candidate_mean"
    assert reference["solver"] == "float64_cpu_lstsq"
    for metric in ("target_centered_rms", "residual_rmse", "relative_residual"):
        assert shifted[metric] == pytest.approx(reference[metric], rel=0.0, abs=1.0e-12)
    assert 0.0 < float(reference["relative_residual"]) <= 1.0


def test_assembler_edges_are_oriented_consistent_and_use_only_zero_vs_one() -> None:
    records = _records()
    policy_scores, reward_features, raw_scores = _tensors()
    result = assemble_controlled_experiment(
        records,
        policy_scores,
        reward_features,
        raw_scores,
        seed=77,
    )
    transformed = result.oracle_transform(raw_scores)
    by_prompt = {record.prompt_id: index for index, record in enumerate(records)}

    assert len(result.training_edges) == 2
    assert len(result.validation_edges) == 2
    assert len(result.test_edges) == 2
    all_edges = (
        *result.training_edges,
        *result.validation_edges,
        *result.test_edges,
    )
    for edge in all_edges:
        assert edge.left_id == f"{edge.prompt_id}::candidate::0"
        assert edge.right_id == f"{edge.prompt_id}::candidate::1"
        assert edge.left_wins == sum(edge.raw_labels)
        assert edge.num_annotations == len(edge.raw_labels)
        assert edge.h == pytest.approx(repeated_labels_to_h(edge.raw_labels), rel=0.0, abs=1e-14)
        if hasattr(edge, "true_margin"):
            row = by_prompt[str(edge.prompt_id)]
            assert edge.true_margin == pytest.approx(
                float((transformed[row, 0] - transformed[row, 1]).item())
            )

    for tensor_row, edge in enumerate(result.training_edges):
        assert result.experiment.train.left_wins[tensor_row].item() == edge.left_wins
        assert result.experiment.train.num_annotations[tensor_row].item() == edge.num_annotations
        assert result.experiment.train.h[tensor_row].item() == pytest.approx(edge.h)
    expected_edge_scores = policy_scores[[0, 2], 0] - policy_scores[[0, 2], 1]
    assert torch.equal(
        result.experiment.train.to_training_batch().edge_scores,
        expected_edge_scores,
    )


def test_assembler_is_deterministic_and_does_not_mutate_global_rng() -> None:
    records = _records()
    tensors = _tensors()
    torch.manual_seed(987654)
    random.seed(12345)
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()

    first = assemble_controlled_experiment(*((records,) + tensors), seed=9123)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    second = assemble_controlled_experiment(*((records,) + tensors), seed=9123)

    assert first.training_edges == second.training_edges
    assert first.validation_edges == second.validation_edges
    assert first.test_edges == second.test_edges
    assert torch.equal(first.experiment.train.h, second.experiment.train.h)
    assert first.evidence == second.evidence


def test_training_annotations_do_not_depend_on_heldout_split_sizes() -> None:
    records = _records()
    policy_scores, reward_features, raw_scores = _tensors()
    reference = assemble_controlled_experiment(
        records,
        policy_scores,
        reward_features,
        raw_scores,
        seed=9123,
    )

    extra = PromptRecord(
        prompt_id="test-extra",
        messages=(ChatMessage(role="user", content="extra heldout question"),),
        split="test",
    )
    expanded = assemble_controlled_experiment(
        [*records, extra],
        torch.cat((policy_scores, policy_scores[-1:].clone()), dim=0),
        torch.cat((reward_features, reward_features[-1:].clone()), dim=0),
        torch.cat((raw_scores, raw_scores[-1:].clone()), dim=0),
        seed=9123,
    )

    assert expanded.training_edges == reference.training_edges
    assert torch.equal(expanded.experiment.train.h, reference.experiment.train.h)
    assert (
        expanded.evidence["annotation_split_seeds"]["train"]
        == reference.evidence["annotation_split_seeds"]["train"]
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("three_candidates", "exactly four"),
        ("wrong_prompt_count", "join key"),
        ("nonfinite", "finite"),
        ("duplicate_prompt", "unique"),
        ("missing_split", "non-empty"),
        ("wrong_dtype", "share dtype"),
    ],
)
def test_assembler_fails_closed_on_graph_join_errors(mutation: str, match: str) -> None:
    records = _records()
    policy_scores, reward_features, raw_scores = _tensors()
    if mutation == "three_candidates":
        policy_scores = policy_scores[:, :3]
        reward_features = reward_features[:, :3]
        raw_scores = raw_scores[:, :3]
    elif mutation == "wrong_prompt_count":
        records = records[:-1]
    elif mutation == "nonfinite":
        raw_scores[0, 0] = torch.nan
    elif mutation == "duplicate_prompt":
        records[-1] = PromptRecord(
            prompt_id=records[0].prompt_id,
            messages=records[-1].messages,
            split=records[-1].split,
        )
    elif mutation == "missing_split":
        records = [record for record in records if record.split != "test"]
        policy_scores = policy_scores[:4]
        reward_features = reward_features[:4]
        raw_scores = raw_scores[:4]
    elif mutation == "wrong_dtype":
        raw_scores = raw_scores.float()
    with pytest.raises(ValueError, match=match):
        assemble_controlled_experiment(
            records,
            policy_scores,
            reward_features,
            raw_scores,
            seed=1,
        )


def test_assembler_rejects_unlocked_gamma() -> None:
    with pytest.raises(ValueError, match="locked"):
        assemble_controlled_experiment(_records(), *_tensors(), seed=1, gamma=0.8)


def test_materializer_refuses_existing_files_before_optional_imports(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_phase1({}, seed=1, artifact_dir=target, device="cpu")
    assert (target / "keep.txt").read_text(encoding="utf-8") == "user data"


class _PromptTokenizer:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count
        self.kwargs: dict[str, object] | None = None

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "complete original prompt"}]
        self.kwargs = kwargs
        return {
            "input_ids": torch.arange(self.token_count, dtype=torch.int64).reshape(1, -1),
            "attention_mask": torch.ones((1, self.token_count), dtype=torch.int64),
        }


def test_materialization_prompt_encoding_preserves_complete_raw_prompt() -> None:
    tokenizer = _PromptTokenizer(7)
    prompt = PromptRecord(
        prompt_id="p-full",
        messages=(ChatMessage(role="user", content="complete original prompt"),),
        split="train",
    )
    inputs, evidence = _encode_full_policy_prompt(
        tokenizer,
        prompt,
        max_prompt_tokens=8,
        device=torch.device("cpu"),
    )
    assert tokenizer.kwargs is not None
    assert tokenizer.kwargs["truncation"] is False
    assert "max_length" not in tokenizer.kwargs
    assert inputs["input_ids"].shape == (1, 7)
    assert evidence["policy_chat_token_count"] == 7
    assert evidence["truncated"] is False
    assert evidence["raw_prompt_preserved"] is True
    assert prompt.messages[0].content == "complete original prompt"


def test_materialization_prompt_encoding_fails_closed_above_frozen_cap() -> None:
    tokenizer = _PromptTokenizer(9)
    prompt = PromptRecord(
        prompt_id="p-too-long",
        messages=(ChatMessage(role="user", content="complete original prompt"),),
        split="train",
    )
    with pytest.raises(ValueError, match="truncation is forbidden"):
        _encode_full_policy_prompt(
            tokenizer,
            prompt,
            max_prompt_tokens=8,
            device=torch.device("cpu"),
        )


def test_materializer_reports_missing_optional_dependency_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/smoke.yaml"))

    def missing(name: str, *, extra: str = "llm") -> object:
        raise ImportError(f"missing {name} from {extra}")

    monkeypatch.setattr("smart_reward.phase1._require_module", missing)
    with pytest.raises(ImportError, match="missing datasets"):
        materialize_phase1(
            config,
            seed=20260722,
            artifact_dir=tmp_path / "new-artifact",
            device="cpu",
        )
    assert not (tmp_path / "new-artifact").exists()


def test_formal_prompt_loader_resolves_snapshot_then_reads_local_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    revision = config["data"]["prompt_revision"]
    snapshot = tmp_path / "hub" / "datasets--allenai--multipref" / "snapshots" / revision
    parquet = snapshot / "data" / "train-00000-of-00001.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"test parquet placeholder")
    rows = [
        {
            "prompt_id": f"p-{index}",
            "text": f"{'long ' if index < 6 else ''}prompt {index}",
        }
        for index in range(70)
    ]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DownloadConfig:
        def __init__(self, *, local_files_only: bool) -> None:
            self.local_files_only = local_files_only

    def load_dataset(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return rows

    datasets = SimpleNamespace(DownloadConfig=DownloadConfig, load_dataset=load_dataset)
    hub = SimpleNamespace(
        snapshot_download=lambda **kwargs: (
            snapshot
            if kwargs
            == {
                "repo_id": "allenai/multipref",
                "repo_type": "dataset",
                "revision": revision,
                "cache_dir": str(tmp_path / "hub"),
                "local_files_only": True,
                "token": False,
            }
            else (_ for _ in ()).throw(AssertionError(kwargs))
        )
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    monkeypatch.setattr(
        "smart_reward.phase1._require_module",
        lambda name: (
            hub if name == "huggingface_hub" else (_ for _ in ()).throw(AssertionError(name))
        ),
    )
    config["data"]["prompt_eligibility"] = (
        "policy_chat_template_tokens_lte_max_prompt_tokens_before_seeded_shuffle"
    )

    class Tokenizer:
        chat_template = "test-template"

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "truncation": False,
            }
            return [1] * (400 if messages[0]["content"].startswith("long ") else 8)

    prompts, selection = _load_prompts(
        datasets,
        config,
        policy_tokenizer=Tokenizer(),
        split_seed=123,
        local_files_only=True,
    )

    assert len(prompts) == 64
    assert selection["dataset_unique_prompt_count"] == 70
    assert selection["eligible_prompt_count"] == 64
    assert selection["excluded_over_limit_prompt_count"] == 6
    assert selection["filter_applied_before_seeded_shuffle"] is True
    assert selection["selected_maximum_policy_chat_token_count"] == 8
    assert all(not prompt.messages[0].content.startswith("long ") for prompt in prompts)
    assert calls[0][0] == ("parquet",)
    assert calls[0][1]["data_files"] == {"train": [str(parquet)]}
    assert calls[0][1]["download_config"].local_files_only is True


def test_producer_identity_reads_only_validated_digest_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smart_reward.phase1 import _producer_identity_from_environment

    monkeypatch.setenv("UNRELATED_SECRET", "must-not-appear")
    monkeypatch.setenv("PRORM_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PRORM_IMAGE_SHA256", "b" * 64)
    monkeypatch.setenv("PRORM_HF_INVENTORY_SHA256", "c" * 64)
    assert _producer_identity_from_environment() == {
        "git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
    }

    monkeypatch.setenv("PRORM_IMAGE_SHA256", "not-a-digest")
    with pytest.raises(ValueError, match="producer digest"):
        _producer_identity_from_environment()


def test_formal_producer_requires_hf_inventory_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smart_reward.phase1 import _producer_identity_from_environment

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("PRORM_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PRORM_IMAGE_SHA256", "b" * 64)
    monkeypatch.delenv("PRORM_HF_INVENTORY_SHA256", raising=False)
    monkeypatch.delenv("SRM_HF_INVENTORY_SHA256", raising=False)
    with pytest.raises(ValueError, match="requires Git, image, and HF inventory"):
        _producer_identity_from_environment()

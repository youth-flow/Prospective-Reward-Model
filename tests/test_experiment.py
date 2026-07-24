import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import smart_reward.experiment as experiment_module
from smart_reward.config import load_config
from smart_reward.experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    FeatureExperimentConfig,
    TrainingTensorData,
    compile_feature_experiment_config,
    run_feature_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def _nodes(offset: float, num_prompts: int) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    base = torch.arange(num_prompts * 3, dtype=dtype).reshape(num_prompts, 3)
    policy_scores = torch.stack(
        (
            torch.sin(base + offset) + 0.2,
            torch.cos(0.7 * base + offset) - 0.1,
        ),
        dim=-1,
    )
    reward_features = torch.stack(
        (
            0.3 * base + offset,
            torch.sin(0.4 * base - offset),
        ),
        dim=-1,
    )
    return policy_scores, reward_features


def _train() -> TrainingTensorData:
    scores, features = _nodes(0.2, 6)
    return TrainingTensorData(
        prompt_ids=tuple(f"train-{index}" for index in range(6)),
        policy_scores=scores,
        reward_features=features,
        h=torch.tensor([0.9, -0.4, 0.7, -0.6, 0.2, 1.1], dtype=torch.float64),
        left_wins=torch.tensor([12, 5, 11, 4, 9, 14], dtype=torch.int64),
        num_annotations=torch.tensor([16, 15, 17, 14, 18, 19], dtype=torch.int64),
    )


def _evaluation(prefix: str, offset: float) -> EvaluationTensorData:
    scores, features = _nodes(offset, 4)
    true_rewards = (
        0.8 * features[..., 0] - 0.5 * features[..., 1] + 0.35 * scores[..., 0] * features[..., 1]
    )
    return EvaluationTensorData(
        prompt_ids=tuple(f"{prefix}-{index}" for index in range(4)),
        policy_scores=scores,
        reward_features=features,
        true_rewards=true_rewards,
    )


def _experiment() -> ControlledFeatureExperiment:
    return ControlledFeatureExperiment(
        train=_train(),
        validation=_evaluation("validation", 1.1),
        test=_evaluation("test", 2.3),
    )


def _config() -> FeatureExperimentConfig:
    return FeatureExperimentConfig(
        outer_steps=3,
        learning_rate=0.025,
        beta=1.2,
        relative_damping=0.07,
        pcg_max_iterations=20,
        pcg_tolerance=1.0e-11,
        pcg_residual_recompute_interval=5,
        microbatch_size=2,
        max_grad_norm=5.0,
    )


def test_training_schema_structurally_forbids_oracle_leakage() -> None:
    assert {item.name for item in fields(TrainingTensorData)} == {
        "prompt_ids",
        "policy_scores",
        "reward_features",
        "h",
        "left_wins",
        "num_annotations",
    }
    values = {item.name: getattr(_train(), item.name) for item in fields(TrainingTensorData)}
    with pytest.raises(TypeError):
        TrainingTensorData(**values, true_rewards=torch.zeros(6))
    with pytest.raises(TypeError):
        TrainingTensorData(**values, oracle_rewards=torch.zeros(6))


def test_canonical_edge_conversion_and_shape_validation() -> None:
    train = _train()
    batch = train.to_training_batch()
    assert torch.equal(batch.left_features, train.reward_features[:, 0])
    assert torch.equal(batch.right_features, train.reward_features[:, 1])
    assert torch.equal(
        batch.edge_scores,
        train.policy_scores[:, 0] - train.policy_scores[:, 1],
    )
    assert torch.equal(
        batch.node_scores,
        train.policy_scores.reshape(-1, train.policy_dimension),
    )

    one_candidate_scores = train.policy_scores[:, :1]
    one_candidate_features = train.reward_features[:, :1]
    with pytest.raises(ValueError, match="at least two candidates"):
        replace(
            train,
            policy_scores=one_candidate_scores,
            reward_features=one_candidate_features,
        )
    with pytest.raises(ValueError, match="finite"):
        replace(train, h=train.h.clone().index_fill(0, torch.tensor([0]), float("nan")))


def test_training_tensor_device_transfer_preserves_frozen_schema() -> None:
    train = _train()
    moved = train.to("cpu")

    assert moved is not train
    assert moved.prompt_ids == train.prompt_ids
    for name in (
        "policy_scores",
        "reward_features",
        "h",
        "left_wins",
        "num_annotations",
    ):
        original = getattr(train, name)
        transferred = getattr(moved, name)
        assert transferred.device.type == "cpu"
        assert transferred.dtype == original.dtype
        assert torch.equal(transferred, original)
        assert transferred.requires_grad is False


def test_split_ids_are_pairwise_disjoint_and_layouts_match() -> None:
    experiment = _experiment()
    leaked_validation = replace(
        experiment.validation,
        prompt_ids=(experiment.train.prompt_ids[0],) + experiment.validation.prompt_ids[1:],
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        ControlledFeatureExperiment(
            train=experiment.train,
            validation=leaked_validation,
            test=experiment.test,
        )

    mismatched_test = EvaluationTensorData(
        prompt_ids=experiment.test.prompt_ids,
        policy_scores=experiment.test.policy_scores[:, :2],
        reward_features=experiment.test.reward_features[:, :2],
        true_rewards=experiment.test.true_rewards[:, :2],
    )
    with pytest.raises(ValueError, match="candidate count"):
        ControlledFeatureExperiment(
            train=experiment.train,
            validation=experiment.validation,
            test=mismatched_test,
        )


def test_relative_damping_fair_initialization_and_json_contract() -> None:
    experiment = _experiment()
    config = _config()
    result = run_feature_experiment(experiment, config)

    expected_train_damping = config.relative_damping * float(
        experiment.train.policy_scores.square().mean().item()
    )
    expected_validation_damping = config.relative_damping * float(
        experiment.validation.policy_scores.square().mean().item()
    )
    assert result.train_absolute_damping == pytest.approx(expected_train_damping)
    assert result.bt_mle.validation.absolute_damping == pytest.approx(expected_validation_damping)
    assert result.prorm_plus.validation.absolute_damping == pytest.approx(
        expected_validation_damping
    )
    assert result.bt_mle.initial_head_sha256 == result.prorm_plus.initial_head_sha256
    assert result.bt_mle.head_sha256 != result.bt_mle.initial_head_sha256
    assert result.prorm_plus.head_sha256 != result.prorm_plus.initial_head_sha256
    assert len(result.bt_mle.head_weight) == experiment.train.reward_dimension
    assert len(result.prorm_plus.head_weight) == experiment.train.reward_dimension
    assert result.prorm_plus.final_pcg is not None
    assert result.prorm_plus.final_pcg.converged
    assert 0.0 <= result.bt_mle.test.pairwise_accuracy <= 1.0
    assert 0.0 <= result.prorm_plus.test.pairwise_accuracy <= 1.0
    assert result.prorm_plus.test.oracle_pairwise_nll >= 0.0
    assert 0.0 <= result.prorm_plus.test.oracle_probability_mae <= 1.0
    assert not result.heldout_used_for_training
    assert not result.prorm_plus_win_guaranteed
    payload = result.to_dict()
    assert "prorm_plus" in payload and "srm_plus" not in payload
    assert "prorm_plus_win_guaranteed" in payload and "srm_win_guaranteed" not in payload
    json.dumps(payload, allow_nan=False, sort_keys=True)

    pairs = torch.combinations(torch.arange(experiment.test.num_candidates), r=2)
    head = torch.tensor(result.prorm_plus.head_weight, dtype=torch.float64)
    predicted = experiment.test.reward_features @ head
    predicted_margins = predicted[:, pairs[:, 0]] - predicted[:, pairs[:, 1]]
    target_margins = (
        experiment.test.true_rewards[:, pairs[:, 0]] - experiment.test.true_rewards[:, pairs[:, 1]]
    )
    oracle_probabilities = torch.sigmoid(target_margins)
    assert result.prorm_plus.test.oracle_pairwise_nll == pytest.approx(
        F.binary_cross_entropy_with_logits(predicted_margins, oracle_probabilities).item()
    )
    assert result.prorm_plus.test.oracle_probability_mae == pytest.approx(
        torch.mean(torch.abs(torch.sigmoid(predicted_margins) - oracle_probabilities)).item()
    )

    assert config.weight_decay == 0.0
    with pytest.raises(TypeError):
        FeatureExperimentConfig(weight_decay=0.1)


def test_runner_is_deterministic_finite_and_calls_both_real_trainers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"bt": 0, "prorm_plus": 0}
    original_bt_fit = experiment_module.BTMLETrainer.fit
    original_prorm_fit = experiment_module.ProRMPlusTrainer.fit

    def counted_bt_fit(self: object, steps: int) -> object:
        calls["bt"] += 1
        return original_bt_fit(self, steps)

    def counted_prorm_fit(self: object, steps: int) -> object:
        calls["prorm_plus"] += 1
        return original_prorm_fit(self, steps)

    monkeypatch.setattr(experiment_module.BTMLETrainer, "fit", counted_bt_fit)
    monkeypatch.setattr(experiment_module.ProRMPlusTrainer, "fit", counted_prorm_fit)
    first = run_feature_experiment(_experiment(), _config())
    second = run_feature_experiment(_experiment(), _config())
    assert first == second
    assert calls == {"bt": 2, "prorm_plus": 2}
    assert math_is_finite_tree(first.to_dict())


def math_is_finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(math_is_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(math_is_finite_tree(item) for item in value)
    if isinstance(value, float):
        return torch.isfinite(torch.tensor(value)).item()
    return True


def test_heldout_targets_cannot_change_training() -> None:
    experiment = _experiment()
    altered = ControlledFeatureExperiment(
        train=experiment.train,
        validation=replace(
            experiment.validation,
            true_rewards=-3.0 * experiment.validation.true_rewards,
        ),
        test=replace(experiment.test, true_rewards=5.0 + experiment.test.true_rewards),
    )
    original_result = run_feature_experiment(experiment, _config())
    altered_result = run_feature_experiment(altered, _config())

    assert original_result.bt_mle.head_sha256 == altered_result.bt_mle.head_sha256
    assert original_result.prorm_plus.head_sha256 == altered_result.prorm_plus.head_sha256
    assert (
        original_result.bt_mle.final_train_objective == altered_result.bt_mle.final_train_objective
    )
    assert (
        original_result.prorm_plus.final_train_objective
        == altered_result.prorm_plus.final_train_objective
    )
    assert original_result.bt_mle.test.local_regret != altered_result.bt_mle.test.local_regret


def test_no_policy_signal_is_rejected_before_relative_damping() -> None:
    experiment = _experiment()
    zero_scores = torch.zeros_like(experiment.train.policy_scores)
    degenerate = ControlledFeatureExperiment(
        train=replace(experiment.train, policy_scores=zero_scores),
        validation=experiment.validation,
        test=experiment.test,
    )
    with pytest.raises(ValueError, match="policy tangent is degenerate"):
        run_feature_experiment(degenerate, _config())


def test_yaml_compiles_to_the_single_runtime_config() -> None:
    source = load_config(ROOT / "configs" / "smoke.yaml")
    runtime = compile_feature_experiment_config(source, damping_multiplier=10.0)

    assert runtime.outer_steps == 10
    assert runtime.learning_rate == pytest.approx(1.0e-3)
    assert runtime.relative_damping == pytest.approx(1.0e-2)
    assert runtime.microbatch_size == 16
    assert runtime.max_grad_norm == pytest.approx(1.0)
    assert runtime.weight_decay == 0.0
